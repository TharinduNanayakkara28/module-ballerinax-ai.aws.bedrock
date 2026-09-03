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

import ballerina/test;

// Table tests on the pure resolver. No AWS credentials needed.

const REGION = "us-east-1";

// ---- bare / CRIS-prefixed ids ----

@test:Config {}
function testBareIdResolvesToConverse() returns error? {
    // nova-pro is Converse-default (not Mantle-capable). opus-4-8 now prefers Mantle
    // under AUTO, so a Converse-clean-fields assertion needs a model
    // that genuinely defaults to Converse.
    Route r = check resolveRoute("amazon.nova-pro-v1:0", REGION);
    test:assertEquals(r.family, CONVERSE);
    test:assertEquals(r.bareModelId, "amazon.nova-pro-v1:0");
    test:assertEquals(r.effectiveModelId, "amazon.nova-pro-v1:0");
    test:assertEquals(r.geoPrefix, ());
    test:assertEquals(r.region, REGION);
    test:assertEquals(r.mantleEntry, ());
}

@test:Config {}
function testCrisPrefixStrippedForLookupAndReappliedOnWire() returns error? {
    // the correct runtime id must resolve, and the prefix must survive to the wire.
    Route r = check resolveRoute("us.anthropic.claude-opus-4-8", REGION);
    test:assertEquals(r.family, CONVERSE);
    test:assertEquals(r.bareModelId, "anthropic.claude-opus-4-8", "prefix must be stripped for lookup");
    test:assertEquals(r.geoPrefix, "us");
    test:assertEquals(r.effectiveModelId, "us.anthropic.claude-opus-4-8", "prefix must be re-applied on the wire");
}

@test:Config {}
function testGlobalPrefixNormalization() returns error? {
    Route r = check resolveRoute("global.anthropic.claude-sonnet-4-6", REGION);
    test:assertEquals(r.geoPrefix, "global");
    test:assertEquals(r.bareModelId, "anthropic.claude-sonnet-4-6");
    test:assertEquals(r.effectiveModelId, "global.anthropic.claude-sonnet-4-6");
}

@test:Config {}
function testUnknownBareModelSinksToConverseNeverMantle() returns error? {
    // The fallback trap: absence from a map is not evidence of Mantle.
    Route r = check resolveRoute("acme.brand-new-model-v9", REGION);
    test:assertEquals(r.family, CONVERSE);
    test:assertNotEquals(r.family, MANTLE);
    test:assertEquals(r.mantleEntry, ());
}

// ---- Mantle defaults ----

@test:Config {}
function testMantleOnlyModelDefaultsToMantle() returns error? {
    Route r = check resolveRoute("openai.gpt-5.4", REGION);
    test:assertEquals(r.family, MANTLE);
    test:assertEquals(r.effectiveModelId, "openai.gpt-5.4", "Mantle takes the bare id on the wire");
    MantleEntry entry = check r.mantleEntry.ensureType();
    test:assertEquals(entry.path, "/openai/v1/responses");
    test:assertFalse(usesApiKeyHeader(entry.path));
    test:assertEquals((check mantleConverterForPath(entry.path)).toolChoice, RESPONSES_TOOL_CHOICE);
}

@test:Config {}
function testMythosDefaultsToMantleWithMessagesPath() returns error? {
    Route r = check resolveRoute("anthropic.claude-mythos-preview", REGION);
    test:assertEquals(r.family, MANTLE);
    MantleEntry entry = check r.mantleEntry.ensureType();
    test:assertEquals(entry.path, "/anthropic/v1/messages");
    test:assertTrue(usesApiKeyHeader(entry.path));
    test:assertEquals((check mantleConverterForPath(entry.path)).toolChoice, ANTHROPIC_TOOL_CHOICE);
}

// ---- explicit overrides ----

@test:Config {}
function testForceMantleOnDualEndpointModelResolvesViaCapable() returns error? {
    // capability, not membership — a dual-endpoint model forced to Mantle must resolve.
    Route r = check resolveRoute("anthropic.claude-haiku-4-5", REGION, {apiFamily: MANTLE});
    test:assertEquals(r.family, MANTLE);
    MantleEntry entry = check r.mantleEntry.ensureType();
    test:assertEquals(entry.path, "/anthropic/v1/messages");
}

@test:Config {}
function testForceMantleOnConverseOnlyModelErrors() {
    // not Mantle-capable → clean construction error, not a hard 400 later.
    //
    // This previously used `anthropic.claude-opus-4-8` as the example — but that
    // model's card says `bedrock-mantle: YES`, so the assertion was false and only
    // passed because MANTLE_CAPABLE was missing every dual-endpoint model. Sonnet
    // 4.6 is genuinely runtime-only per AWS's endpoint-availability table.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
    Route|error r = resolveRoute("anthropic.claude-sonnet-4-6", REGION, {apiFamily: MANTLE});
    test:assertTrue(r is error);
    if r is error {
        test:assertTrue(r.message().includes("not available on Mantle"), r.message());
    }
}

@test:Config {}
function testMantlePrefixOverride() returns error? {
    Route r = check resolveRoute("mantle/anthropic.claude-haiku-4-5", REGION);
    test:assertEquals(r.family, MANTLE);
}

@test:Config {}
function testConversePrefixOverridesMantleDefault() returns error? {
    // Explicit override outranks the Mantle default.
    Route r = check resolveRoute("converse/openai.gpt-5.4", REGION);
    test:assertEquals(r.family, CONVERSE);
    test:assertEquals(r.effectiveModelId, "openai.gpt-5.4");
}

// ---- ARN dispatch ----

@test:Config {}
function testFoundationModelArnStripsToBareId() returns error? {
    Route r = check resolveRoute(
        "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-sonnet-4-6", REGION);
    test:assertEquals(r.family, CONVERSE);
    test:assertEquals(r.bareModelId, "anthropic.claude-sonnet-4-6");
    test:assertEquals(r.region, "us-west-2", "ARN region overrides config.region");
}

@test:Config {}
function testImportedModelArnIsRefusedByName() {
    // Custom Model Import is out of scope: AWS applies no default chat template to
    // imported weights, so no request body can be built without the caller naming the
    // dialect. Refuse at construction rather than fail opaquely on the wire.
    Route|error r = resolveRoute(
        "arn:aws:bedrock:us-west-2:123456789012:imported-model/abc123def456", REGION);
    test:assertTrue(r is error);
    if r is error {
        test:assertTrue(r.message().includes("imported-model"), r.message());
    }
}

@test:Config {}
function testProvisionedModelArnResolvesToConverse() returns error? {
    Route r = check resolveRoute(
        "arn:aws:bedrock:eu-west-1:123456789012:provisioned-model/xyz", REGION);
    test:assertEquals(r.family, CONVERSE);
    test:assertEquals(r.region, "eu-west-1");
}

@test:Config {}
function testCustomModelDeploymentArnResolvesToConverse() returns error? {
    Route r = check resolveRoute(
        "arn:aws:bedrock:us-east-1:123456789012:custom-model-deployment/xyz", REGION);
    test:assertEquals(r.family, CONVERSE);
}

@test:Config {}
function testInferenceProfileArnResolvesToConverse() returns error? {
    Route r = check resolveRoute(
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-4-8", REGION);
    test:assertEquals(r.family, CONVERSE);
}

@test:Config {}
function testApplicationInferenceProfileArnResolvesToConverse() returns error? {
    Route r = check resolveRoute(
        "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/opaque123", REGION);
    test:assertEquals(r.family, CONVERSE);
}

@test:Config {}
function testCustomModelArnErrors() {
    // Policy choice: artifact, not a deployment.
    Route|error r = resolveRoute(
        "arn:aws:bedrock:us-east-1:123456789012:custom-model/mymodel", REGION);
    test:assertTrue(r is error);
    if r is error {
        test:assertTrue(r.message().includes("artifact"), r.message());
    }
}

@test:Config {}
function testChinaPartitionArn() returns error? {
    Route r = check resolveRoute(
        "arn:aws-cn:bedrock:cn-north-1:123456789012:provisioned-model/xyz", REGION);
    test:assertEquals(r.partition, "aws-cn");
    test:assertEquals(r.region, "cn-north-1");
}

@test:Config {}
function testChinaPartitionArnBuildsTheCnHostAndSignsAsBedrock() returns error? {
    // Partition inference is only half the job — the whole point of tracking the
    // partition is the DNS suffix, and a hardcoded `.amazonaws.com` would still
    // pass the resolver assertions above.
    Route r = check resolveRoute(
        "arn:aws-cn:bedrock:cn-north-1:123456789012:provisioned-model/xyz", REGION);
    Endpoint ep = check buildEndpoint(r);
    test:assertEquals(ep.host, "bedrock-runtime.cn-north-1.amazonaws.com.cn",
            "aws-cn must use the .com.cn suffix");
    test:assertEquals(ep.path,
            "/model/arn%3Aaws-cn%3Abedrock%3Acn-north-1%3A123456789012%3Aprovisioned-model%2Fxyz/converse");
    test:assertEquals(ep.signingService, SIGNING_BEDROCK,
            "signing name follows the route family, not the partition");
}

@test:Config {}
function testMantleIsRejectedOnTheChinaPartitionBeforeAnyIo() {
    // The `api.aws` Mantle host is not partition-templated, so this must be
    // a construction error rather than a request to a host that cannot exist.
    // Rejection may land in either stage, so BOTH branches assert — an `if r is
    // Route` wrapper alone would let the test pass without running one assertion
    // the day resolution starts erroring instead.
    Route|error r = resolveRoute("mantle/anthropic.claude-opus-4-8", "cn-north-1");
    if r is error {
        test:assertTrue(r.message().includes("partition") || r.message().includes("Mantle"),
                "rejected at resolution, but the message must say why: " + r.message());
        return;
    }
    Endpoint|error ep = buildEndpoint(r);
    test:assertTrue(ep is error, "Mantle must not build an endpoint on aws-cn");
    if ep is error {
        test:assertTrue(ep.message().includes("partition"), ep.message());
    }
}

// ---- partition inference for bare ids ----

@test:Config {}
function testGovCloudRegionInfersPartition() returns error? {
    Route r = check resolveRoute("anthropic.claude-opus-4-8", "us-gov-west-1");
    test:assertEquals(r.partition, "aws-us-gov");
}

@test:Config {}
function testGovCloudRouteBuildsACommercialSuffixHost() returns error? {
    // GovCloud keeps `.amazonaws.com` — only aws-cn differs. Asserted so the
    // awsDomain() branch cannot be "simplified" into applying to both.
    Route r = check resolveRoute("converse/anthropic.claude-opus-4-8", "us-gov-west-1");
    Endpoint ep = check buildEndpoint(r);
    test:assertEquals(ep.host, "bedrock-runtime.us-gov-west-1.amazonaws.com");
    test:assertEquals(ep.signingService, SIGNING_BEDROCK);
}

@test:Config {}
function testCommercialRegionInfersAwsPartition() returns error? {
    Route r = check resolveRoute("anthropic.claude-opus-4-8", "us-east-1");
    test:assertEquals(r.partition, "aws");
}

// ---- ARN parsing ----

@test:Config {}
function testParseArnSegments() returns error? {
    ParsedArn arn = check parseArn("arn:aws:bedrock:us-west-2:123456789012:imported-model/abc123");
    test:assertEquals(arn.partition, "aws");
    test:assertEquals(arn.'service, "bedrock");
    test:assertEquals(arn.region, "us-west-2");
    test:assertEquals(arn.accountId, "123456789012");
    test:assertEquals(arn.resourceType, "imported-model");
    test:assertEquals(arn.resourceId, "abc123");
}

@test:Config {}
function testParseArnRejectsNonArn() {
    ParsedArn|error r = parseArn("anthropic.claude-opus-4-8");
    test:assertTrue(r is error);
}

@test:Config {}
function testNormalizeModelIdKeepsNonCrisDotPrefix() {
    // "anthropic" is a vendor prefix, not a CRIS geo prefix — must not be stripped.
    [string, string?] [bareId, geoPrefix] = normalizeModelId("anthropic.claude-opus-4-8");
    test:assertEquals(bareId, "anthropic.claude-opus-4-8");
    test:assertEquals(geoPrefix, ());
}

// ---- Mantle escape hatch for dual-endpoint models ----

@test:Config {}
function testDualHomedModelCanBeForcedOntoMantle() returns error? {
    // REGRESSION: MANTLE_CAPABLE held only Mantle-only models, so forcing Mantle on
    // a dual-endpoint model errored "not available on Mantle" — which its own card
    // contradicts. The table must list every Mantle-capable model.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-8.html
    Route forced = check resolveRoute("anthropic.claude-opus-4-8", REGION, {apiFamily: MANTLE});
    test:assertEquals(forced.family, MANTLE);
    Endpoint ep = check buildEndpoint(forced);
    test:assertEquals(ep.path, "/anthropic/v1/messages");
    test:assertEquals(ep.signingService, "bedrock-mantle");

    // ...and the prefix form must agree with the config form.
    Route prefixed = check resolveRoute("mantle/anthropic.claude-opus-4-8", REGION);
    test:assertEquals(prefixed.family, MANTLE);

    // It also DEFAULTS to Mantle under AUTO (Mantle-capable →
    // Mantle); `converse/` (or apiFamily=CONVERSE) is needed for the runtime surface.
    Route auto = check resolveRoute("anthropic.claude-opus-4-8", REGION);
    test:assertEquals(auto.family, MANTLE);
    Route converse = check resolveRoute("converse/anthropic.claude-opus-4-8", REGION);
    test:assertEquals(converse.family, CONVERSE);
}

@test:Config {}
function testMantleUsesItsOwnModelIdWhenTheEndpointsDisagree() returns error? {
    // gpt-oss is `openai.gpt-oss-120b-1:0` on bedrock-runtime but plain
    // `openai.gpt-oss-120b` on bedrock-mantle. Sending the runtime id to Mantle
    // fails, so MantleEntry.modelId overrides the wire id.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-120b.html
    Route mantle = check resolveRoute("openai.gpt-oss-120b-1:0", REGION, {apiFamily: MANTLE});
    test:assertEquals(mantle.family, MANTLE);
    test:assertEquals(mantle.effectiveModelId, "openai.gpt-oss-120b", "Mantle has its own id for this model");
    test:assertEquals(mantle.bareModelId, "openai.gpt-oss-120b-1:0", "the lookup key stays the runtime id");

    // The runtime routes keep the `-1:0` id. gpt-oss is Mantle-capable, so it now
    // prefers Mantle under AUTO — force CONVERSE for the runtime form.
    Route converse = check resolveRoute("openai.gpt-oss-120b-1:0", REGION, {apiFamily: CONVERSE});
    test:assertEquals(converse.family, CONVERSE);
    test:assertEquals(converse.effectiveModelId, "openai.gpt-oss-120b-1:0");
}

@test:Config {}
function testForcingMantleOnAnUnknownModelStillErrors() {
    // The hatch must not become a fabricator: no entry and no override → error,
    // never a guessed path.
    Route|error route = resolveRoute("acme.totally-new", REGION, {apiFamily: MANTLE});
    test:assertTrue(route is error);
}

// ---- ARN structural segments ----

@test:Config {}
function testGlobalArnWithNoRegionFallsBackToTheCallerRegion() returns error? {
    // Foundation-model ARNs are commonly written without a region. Copying "" into
    // Route.region built the host `bedrock-runtime..amazonaws.com`, which surfaced
    // as an opaque DNS failure instead of anything actionable.
    Route r = check resolveRoute(
        "arn:aws:bedrock::123456789012:foundation-model/anthropic.claude-sonnet-4-6", "eu-west-1");
    test:assertEquals(r.region, "eu-west-1", "an empty ARN region must fall back to the caller's");
    Endpoint ep = check buildEndpoint(r);
    test:assertEquals(ep.host, "bedrock-runtime.eu-west-1.amazonaws.com");
}

@test:Config {}
function testArnRegionStillOverridesTheCallerRegionWhenPresent() returns error? {
    Route r = check resolveRoute(
        "arn:aws:bedrock:ap-northeast-1:123456789012:inference-profile/apac.anthropic.claude-sonnet-4-6",
        "us-east-1");
    test:assertEquals(r.region, "ap-northeast-1", "a present ARN region stays authoritative");
}

@test:Config {}
function testMalformedArnWithAnEmptyPartitionIsRejected() {
    Route|error r = resolveRoute("arn::bedrock:us-east-1:123456789012:provisioned-model/xyz", REGION);
    test:assertTrue(r is error);
    if r is error {
        test:assertTrue(r.message().includes("partition"), r.message());
    }
}

@test:Config {}
function testNonBedrockArnIsRejectedAtConstruction() {
    // Without this the S3 ARN resolved to CONVERSE and the user learned about it
    // from an opaque AWS error after a network call.
    Route|error r = resolveRoute("arn:aws:s3:us-east-1:123456789012:bucket/foo", REGION);
    test:assertTrue(r is error);
    if r is error {
        test:assertTrue(r.message().includes("Bedrock"), r.message());
    }
}
