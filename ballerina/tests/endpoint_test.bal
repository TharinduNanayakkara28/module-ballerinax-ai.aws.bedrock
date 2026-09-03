// Copyright (c) 2026 WSO2 LLC. (http://www.wso2.com).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import ballerina/ai;
import ballerina/test;

// Endpoint + SigV4 canonical-URI encoding. Regression tests
// for the double-encoding fix from the sub-agent review.

const ARN = "arn:aws:bedrock:us-west-2:123456789012:provisioned-model/abc123";

@test:Config {}
function testWirePathSingleEncodesArnModelIdSegment() returns error? {
    Route route = check resolveRoute(ARN, REGION, {apiFamily: INVOKE});
    Endpoint ep = check buildEndpoint(route);
    test:assertTrue(ep.path.startsWith("/model/") && ep.path.endsWith("/invoke"));
    test:assertTrue(ep.path.includes("%3A"), "ARN colons must be %3A on the wire");
    test:assertTrue(ep.path.includes("%2F"), "ARN internal slash must be %2F on the wire");
    test:assertFalse(ep.path.includes("arn:aws"), "raw colons must not appear on the wire path");
}

@test:Config {}
function testCanonicalUriDoubleEncodesWirePath() returns error? {
    // SigV4 non-S3 rule: the canonical URI is the wire path encoded again.
    Route route = check resolveRoute(ARN, REGION, {apiFamily: INVOKE});
    Endpoint ep = check buildEndpoint(route);
    string canonical = getCanonicalUri(ep.path);
    test:assertTrue(canonical.includes("%253A"), "canonical URI must double-encode the colon");
    test:assertTrue(canonical.includes("%252F"), "canonical URI must double-encode the ARN slash");
    test:assertTrue(canonical.startsWith("/model/") && canonical.endsWith("/invoke"),
        "structural separators must stay literal '/'");
}

@test:Config {}
function testBareIdWirePathHasNoEncodingArtifacts() returns error? {
    Route route = check resolveRoute("us.anthropic.claude-opus-4-8", REGION);
    Endpoint ep = check buildEndpoint(route);
    test:assertEquals(ep.path, "/model/us.anthropic.claude-opus-4-8/converse");
    // Unreserved chars: single == double, so the signature matches without ARNs.
    test:assertEquals(getCanonicalUri(ep.path), "/model/us.anthropic.claude-opus-4-8/converse");
}

@test:Config {}
function testMantleEndpointHostAndSigningService() returns error? {
    Route route = check resolveRoute("anthropic.claude-mythos-preview", "us-east-1");
    Endpoint ep = check buildEndpoint(route);
    test:assertEquals(ep.host, "bedrock-mantle.us-east-1.api.aws");
    test:assertEquals(ep.path, "/anthropic/v1/messages");
    test:assertEquals(ep.signingService, "bedrock-mantle");
}

// GovCloud is a SUPPORTED Mantle partition — `us-gov-west-1` carries bedrock-mantle
// per AWS's endpoint availability table — so the partition guard must not reject it.
// The `api.aws` suffix is partition-neutral, hence no GovCloud-specific host shape.
// https://docs.aws.amazon.com/bedrock/latest/userguide/endpoints-region-availability.html
@test:Config {}
function testMantleEndpointOnGovCloudPartition() returns error? {
    Route route = check resolveRoute("anthropic.claude-mythos-preview", "us-gov-west-1");
    test:assertEquals(route.partition, "aws-us-gov");

    Endpoint ep = check buildEndpoint(route);
    test:assertEquals(ep.host, "bedrock-mantle.us-gov-west-1.api.aws");
    test:assertEquals(ep.signingService, "bedrock-mantle");
}

@test:Config {}
function testConverseSigningServiceIsBedrock() returns error? {
    // nova-pro is Converse-default (not Mantle-capable); opus-4-8 now prefers Mantle
    // under AUTO, so it no longer exercises the runtime signing path.
    Route route = check resolveRoute("amazon.nova-pro-v1:0", "us-east-1");
    Endpoint ep = check buildEndpoint(route);
    test:assertEquals(ep.host, "bedrock-runtime.us-east-1.amazonaws.com");
    test:assertEquals(ep.signingService, "bedrock");
}

// The Invoke guardrail-fired signal is a response BODY field; only the request id
// arrives in a header.
//
// REGRESSION: these tests previously asserted that a `GUARDRAIL_ACTION_HEADER` map
// entry produced INTERVENED, under the comment "the guardrail signal arrives in
// response headers". They passed by hand-injecting an entry the transport could
// never populate — InvokeModel documents no such response header. Green tests,
// dropped safety signal.
// https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html

@test:Config {}
function testAugmentFromHeadersSurfacesRequestId() {
    DecodedResponse decoded = {
        message: {role: ai:ASSISTANT, content: "hi"},
        usage: {inputTokens: 1, outputTokens: 1},
        stopReason: "end_turn",
        responseId: (),
        guardrailAction: ()
    };
    augmentFromHeaders(decoded, {[REQUEST_ID_HEADER]: "req-123"});
    test:assertEquals(decoded.responseId, "req-123");
}

@test:Config {}
function testAugmentFromHeadersDoesNotOverrideAResponseIdFromTheBody() {
    DecodedResponse decoded = {
        message: {role: ai:ASSISTANT, content: ""},
        usage: {inputTokens: 1, outputTokens: 0},
        stopReason: "end_turn",
        responseId: "existing",
        guardrailAction: ()
    };
    augmentFromHeaders(decoded, {[REQUEST_ID_HEADER]: "other"});
    test:assertEquals(decoded.responseId, "existing");
}

@test:Config {}
function testInvokeGuardrailActionReadsTheBodyField() {
    test:assertEquals(invokeGuardrailAction({"amazon-bedrock-guardrailAction": "INTERVENED"}), INTERVENED);
    test:assertEquals(invokeGuardrailAction({"amazon-bedrock-guardrailAction": "NONE"}), NONE);
    test:assertEquals(invokeGuardrailAction({"other": 1}), (), "absent field means no signal, not NONE");
}

@test:Config {}
function testEveryInvokeConverterSurfacesAFiredGuardrail() returns error? {
    // the fired signal is never dropped on ANY Invoke dialect. Each of these
    // decoders previously hardcoded `guardrailAction: ()`.
    json mistralChat = {
        "choices": [{"message": {"role": "assistant", "content": "blocked"}, "stop_reason": "stop"}],
        "amazon-bedrock-guardrailAction": "INTERVENED"
    };
    test:assertEquals((check decodeMistralChat(mistralChat)).guardrailAction, INTERVENED);

    json mistralText = {
        "outputs": [{"text": "blocked", "stop_reason": "stop"}],
        "amazon-bedrock-guardrailAction": "INTERVENED"
    };
    test:assertEquals((check decodeMistralText(mistralText)).guardrailAction, INTERVENED);

    json openAIChat = {
        "choices": [{"message": {"role": "assistant", "content": "blocked"}, "finish_reason": "stop"}],
        "amazon-bedrock-guardrailAction": "INTERVENED"
    };
    test:assertEquals((check decodeOpenAIChat(openAIChat)).guardrailAction, INTERVENED);

    json deepSeek = {
        "choices": [{"text": "blocked", "stop_reason": "stop"}],
        "amazon-bedrock-guardrailAction": "INTERVENED"
    };
    test:assertEquals((check decodeDeepSeekInvoke(deepSeek)).guardrailAction, INTERVENED);

    // Nova-on-Invoke shares the Converse decoder but reports via the body field.
    json novaInvoke = {
        "output": {"message": {"role": "assistant", "content": [{"text": "blocked"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1},
        "amazon-bedrock-guardrailAction": "INTERVENED"
    };
    test:assertEquals((check decodeConverse(novaInvoke)).guardrailAction, INTERVENED);

    // Converse proper still reports it via stopReason.
    json converse = {
        "output": {"message": {"role": "assistant", "content": [{"text": "blocked"}]}},
        "stopReason": "guardrail_intervened",
        "usage": {"inputTokens": 1, "outputTokens": 1}
    };
    test:assertEquals((check decodeConverse(converse)).guardrailAction, INTERVENED);
}

// ---- RFC 3986 path-segment encoding (SigV4), not form encoding ----

@test:Config {}
function testPathSegmentEncoderFollowsRfc3986NotFormEncoding() {
    // `url:encode` would produce `+` for a space and escape `~`. SigV4 requires
    // `%20` and leaves every unreserved character literal; getting this wrong is a
    // SignatureDoesNotMatch with nothing in the message pointing at the encoder.
    test:assertEquals(encodePathSegment("a b"), "a%20b", "a space must be %20, never '+'");
    test:assertEquals(encodePathSegment("a~b"), "a~b", "'~' is unreserved and stays literal");
    test:assertEquals(encodePathSegment("a*b"), "a%2Ab", "'*' is reserved and must be escaped");
    test:assertEquals(encodePathSegment("a-b._c"), "a-b._c", "unreserved set passes through");
    test:assertEquals(encodePathSegment("v1:0"), "v1%3A0");
    test:assertEquals(encodePathSegment("a/b"), "a%2Fb");
    test:assertEquals(encodePathSegment("%"), "%25", "a literal % must double-encode");
}

@test:Config {}
function testCanonicalUriAgreesWithTheWirePathEncoder() returns error? {
    // The two must use the SAME rule or the server cannot reconstruct what we
    // signed. A model id with a space is the case that used to diverge.
    Route r = check resolveRoute("converse/some vendor.model~x", "us-east-1");
    Endpoint ep = check buildEndpoint(r);
    test:assertEquals(ep.path, "/model/some%20vendor.model~x/converse");
    test:assertEquals(getCanonicalUri(ep.path), "/model/some%2520vendor.model~x/converse");
}

// ---- serviceUrl: the default is a TEMPLATE, not a constant host ----
//
// A constant default is impossible here: the origin is decided by the resolved
// route (`bedrock-runtime` vs `bedrock-mantle`, and three DNS suffixes). The
// template resolves per route, and — crucially — substitution is a no-op on a
// string with no placeholders, so a concrete URL needs no sentinel.

@test:Config {}
function testDefaultServiceUrlTemplateResolvesPerRouteFamily() returns error? {
    Route converse = check resolveRoute("anthropic.claude-sonnet-4-6", "eu-west-1");
    test:assertEquals((check buildEndpoint(converse)).baseUrl,
            "https://bedrock-runtime.eu-west-1.amazonaws.com");

    // Same template, different family → the `api.aws` suffix, not `amazonaws.com`.
    Route mantle = check resolveRoute("openai.gpt-5.4", "eu-west-1");
    test:assertEquals((check buildEndpoint(mantle)).baseUrl,
            "https://bedrock-mantle.eu-west-1.api.aws");
}

@test:Config {}
function testDefaultServiceUrlTemplateFollowsThePartitionSuffix() returns error? {
    Route cn = check resolveRoute("anthropic.claude-sonnet-4-6", "cn-north-1");
    test:assertEquals((check buildEndpoint(cn)).baseUrl,
            "https://bedrock-runtime.cn-north-1.amazonaws.com.cn");
}

@test:Config {}
function testConcreteServiceUrlPassesThroughAndKeepsTheRouteDerivedPath() returns error? {
    // The no-sentinel property: a URL with no placeholders is returned untouched, so
    // nothing has to ask "did the caller accept the default?".
    Route r = check resolveRoute("anthropic.claude-sonnet-4-6", "us-east-1");
    Endpoint ep = check buildEndpoint(r,
            "https://vpce-0abc.bedrock-runtime.us-east-1.vpce.amazonaws.com");
    test:assertEquals(ep.baseUrl, "https://vpce-0abc.bedrock-runtime.us-east-1.vpce.amazonaws.com");
    test:assertEquals(ep.host, "vpce-0abc.bedrock-runtime.us-east-1.vpce.amazonaws.com",
            "Host header / SigV4 canonical host must follow the override");
    test:assertEquals(ep.path, "/model/anthropic.claude-sonnet-4-6/converse",
            "serviceUrl replaces the ORIGIN only — the path stays route-derived");
    test:assertEquals(ep.signingService, SIGNING_BEDROCK,
            "a custom host must NOT change the SigV4 signing scope");
}

@test:Config {}
function testPartialServiceUrlOverrideKeepsRegionAndDomainAutomatic() returns error? {
    // The FIPS case: override the service segment, let region/domain resolve. This is
    // what a nilable-with-derived-default could not express without hardcoding both.
    Route r = check resolveRoute("anthropic.claude-sonnet-4-6", "us-gov-west-1");
    Endpoint ep = check buildEndpoint(r, "https://bedrock-{endpoint}-fips.{region}.{domain}");
    test:assertEquals(ep.baseUrl, "https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com");
}

@test:Config {}
function testUnresolvedPlaceholderFailsAtConstruction() returns error? {
    // A surviving brace is ALWAYS a typo — braces are not legal in DNS names — so
    // catching it here turns a silent DNS failure into a named construction error.
    Route r = check resolveRoute("anthropic.claude-sonnet-4-6", "us-east-1");
    Endpoint|error ep = buildEndpoint(r, "https://bedrock-{endpoint}.{regoin}.{domain}");
    test:assertTrue(ep is error);
    if ep is error {
        test:assertTrue(ep.message().includes("unresolved placeholder"), ep.message());
        test:assertTrue(ep.message().includes("{regoin}"), ep.message());
    }
}

@test:Config {}
function testServiceUrlTrailingSlashIsTrimmed() returns error? {
    // Otherwise it doubles up against the leading slash of the route-derived path.
    Route r = check resolveRoute("anthropic.claude-sonnet-4-6", "us-east-1");
    test:assertEquals((check buildEndpoint(r, "https://gw.corp/")).baseUrl, "https://gw.corp");
}

// ---------------------------------------------------------------------------
// SDK endpoint metadata (ballerinax/aws). The default template now defers wholly
// to `aws:resolveEndpoint`; the tests above already pin the three host shapes it
// must keep producing. These pin the parts that are NEW.
// ---------------------------------------------------------------------------

@test:Config {}
function testFipsResolvesToTheFipsHostFromSdkMetadata() returns error? {
    // The host spelling is AWS's, not ours — that is the whole point of routing
    // this through SDK metadata rather than string-building `-fips` ourselves.
    Route r = check resolveRoute("anthropic.claude-sonnet-4-6", "us-east-1");
    Endpoint ep = check buildEndpoint(r, DEFAULT_SERVICE_URL, true);
    test:assertEquals(ep.baseUrl, "https://bedrock-runtime-fips.us-east-1.amazonaws.com");
    test:assertEquals(ep.host, "bedrock-runtime-fips.us-east-1.amazonaws.com");
    test:assertEquals(ep.signingService, SIGNING_BEDROCK,
            "FIPS changes the HOST only — never the SigV4 signing scope");
}

@test:Config {}
function testFipsIsRejectedOnAMantleRouteBeforeAnyIo() returns error? {
    // There is no `bedrock-mantle-fips` host. Without this guard the SDK fallback
    // would synthesise one and the failure would surface as an opaque DNS error.
    Route mantle = check resolveRoute("openai.gpt-5.4", "us-east-1");
    Endpoint|error ep = buildEndpoint(mantle, DEFAULT_SERVICE_URL, true);
    test:assertTrue(ep is error);
    if ep is error {
        test:assertTrue(ep.message().includes("fips"), ep.message());
        test:assertTrue(ep.message().includes("Mantle"), ep.message());
    }
}

@test:Config {}
function testMantleRequiresTheDualstackVariantToReachApiAws() returns error? {
    // REGRESSION GUARD. `aws:resolveEndpoint("bedrock-mantle", region)` WITHOUT
    // dualstack returns `bedrock-mantle.{region}.amazonaws.com`, which does not
    // resolve — `api.aws` is modelled as the dualstack suffix. If someone drops
    // the `dualstack: mantle` flag in resolveServiceUrl, every Mantle call breaks
    // at DNS, and this is the only thing that would catch it.
    Route mantle = check resolveRoute("openai.gpt-5.4", "us-east-1");
    test:assertEquals((check buildEndpoint(mantle)).baseUrl,
            "https://bedrock-mantle.us-east-1.api.aws");
}

@test:Config {}
function testCustomTemplateStillTakesItsDomainFromSdkMetadata() returns error? {
    // The `{domain}` placeholder is no longer a hardcoded suffix table; it is
    // derived from the resolved host, so China still lands on `.com.cn`.
    Route cn = check resolveRoute("anthropic.claude-sonnet-4-6", "cn-north-1");
    Endpoint ep = check buildEndpoint(cn, "https://bedrock-{endpoint}.{region}.{domain}");
    test:assertEquals(ep.baseUrl, "https://bedrock-runtime.cn-north-1.amazonaws.com.cn");
}
