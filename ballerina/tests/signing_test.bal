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
import ballerina/time;

// Golden tests for SigV4 signing.
//
// `signedHeaders` is the highest-risk function in the module and had NO coverage —
// which is exactly why an invalid-timestamp bug shipped (see the amzDate tests
// below). Signing is unobservable without live AWS: a wrong signature is a 403 that
// looks identical to a permissions problem.
//
// The expected signatures below were produced by an INDEPENDENT implementation of
// the AWS spec (Python hmac/hashlib), not by this module — so these assert
// cross-implementation agreement, not that the code reproduces itself. Any drift in
// the canonical request, the string-to-sign, the header set, the sort order, or the
// key derivation changes the signature and fails here.
// https://docs.aws.amazon.com/IAM/latest/UserGuide/create-signed-request.html

// A fixed clock — signing is a pure function of its inputs once the time is fixed.
final [string, string] FIXED_CLOCK = ["20260717T120000Z", "20260717"];

function transportFor(string host, string path, string signingService, string region)
        returns BedrockTransport|error =>
    new (TEST_CREDS_SIGNING, region,
        {baseUrl: string `https://${host}`, host, path, signingService});

// Kept separate from TEST_CREDS so the golden signatures never move if that changes.
final BedrockCredentials TEST_CREDS_SIGNING = {accessKeyId: "AKIATEST", secretAccessKey: "secret"};

@test:Config {}
function testSignedHeadersMatchAnIndependentImplementationOnConverse() returns error? {
    BedrockTransport transport = check transportFor("bedrock-runtime.us-east-1.amazonaws.com",
            "/model/anthropic.claude-opus-4-8/converse", SIGNING_BEDROCK, "us-east-1");
    map<string> headers = check transport.signedHeaders("{\"messages\":[]}", {}, FIXED_CLOCK);
    test:assertEquals(headers["Authorization"],
            "AWS4-HMAC-SHA256 Credential=AKIATEST/20260717/us-east-1/bedrock/aws4_request, " +
            "SignedHeaders=content-type;host;x-amz-date, " +
            "Signature=427e89ea28eba206f76052e8f1d6760391503708ce9fbf58d7ca9ab100206f53");
    test:assertEquals(headers["X-Amz-Date"], "20260717T120000Z");
}

@test:Config {}
function testSignedHeadersUseTheDoubleEncodedCanonicalUriForAnArn() returns error? {
    // The wire path is single-encoded (buildEndpoint); the SIGNATURE must be over
    // the double-encoded path. The reference below was computed over the
    // double-encoded URI, so this fails if the transport signs the wire path.
    BedrockTransport transport = check transportFor("bedrock-runtime.us-west-2.amazonaws.com",
            "/model/arn%3Aaws%3Abedrock%3Aus-west-2%3A123456789012%3Aimported-model%2Fabc123/invoke",
            SIGNING_BEDROCK, "us-west-2");
    map<string> headers = check transport.signedHeaders("{\"prompt\":\"hi\"}", {}, FIXED_CLOCK);
    test:assertEquals(headers["Authorization"],
            "AWS4-HMAC-SHA256 Credential=AKIATEST/20260717/us-west-2/bedrock/aws4_request, " +
            "SignedHeaders=content-type;host;x-amz-date, " +
            "Signature=a58a636721156c4e6157baaf6b9ba9b0e1c59ad95aebdef58809fe30f71cc31c");
}

@test:Config {}
function testSignedHeadersUseTheMantleScopeAndSignExtraHeaders() returns error? {
    // Mantle signs `bedrock-mantle`, not `bedrock` — and every extra header must be
    // in the canonical request AND the SignedHeaders list, sorted.
    BedrockTransport transport = check transportFor("bedrock-mantle.us-east-1.api.aws",
            "/anthropic/v1/messages", SIGNING_BEDROCK_MANTLE, "us-east-1");
    map<string> extra = {"anthropic-version": "2023-06-01", "x-api-key": "secret-key"};
    map<string> headers =
        check transport.signedHeaders("{\"model\":\"anthropic.claude-mythos-preview\"}", extra, FIXED_CLOCK);
    test:assertEquals(headers["Authorization"],
            "AWS4-HMAC-SHA256 Credential=AKIATEST/20260717/us-east-1/bedrock-mantle/aws4_request, " +
            "SignedHeaders=anthropic-version;content-type;host;x-amz-date;x-api-key, " +
            "Signature=f51554fbe37b913ad9d3af1dfda5cefebab3934ce04c14b47435ddd4ad9d55f0");
}

@test:Config {}
function testBearerCredentialsSkipSigV4Entirely() returns error? {
    // A Bedrock API key is first-class on both endpoints: no signature.
    BedrockTransport transport = check new ({apiKey: "bedrock-api-key"}, "us-east-1",
            {baseUrl: string `https://bedrock-runtime.us-east-1.amazonaws.com`, host: "bedrock-runtime.us-east-1.amazonaws.com", path: "/model/m/converse",
                signingService: SIGNING_BEDROCK});
    map<string> headers = check transport.signedHeaders("{}", {}, FIXED_CLOCK);
    test:assertEquals(headers["Authorization"], "Bearer bedrock-api-key");
    test:assertFalse(headers.hasKey("X-Amz-Date"), "no SigV4 means no signing headers");
}

@test:Config {}
function testBearerWithXApiKeyMantleRouteSendsOnlyXApiKey() returns error? {
    // REGRESSION (live 401, 2026-08-03): an X_API_KEY Mantle route on a bearer credential
    // attaches `x-api-key` (addMantleApiKeyHeader); the transport must then NOT also add
    // `Authorization: Bearer`, because Anthropic's Mantle surface rejects a request carrying
    // BOTH with 401 "must not include both 'authorization' and 'x-api-key' headers".
    // Exactly one auth header may reach the wire. The merged set had no coverage, which is
    // how the collision shipped.
    BedrockTransport transport = check new ({apiKey: "bedrock-api-key"}, "us-east-1",
            {baseUrl: string `https://bedrock-mantle.us-east-1.api.aws`, host: "bedrock-mantle.us-east-1.api.aws", path: "/anthropic/v1/messages",
                signingService: SIGNING_BEDROCK_MANTLE});
    map<string> headers = check transport.signedHeaders("{}",
            {"x-api-key": "bedrock-api-key", "anthropic-version": "2023-06-01"}, FIXED_CLOCK);
    test:assertEquals(headers["x-api-key"], "bedrock-api-key");
    test:assertFalse(headers.hasKey("Authorization"),
            "must not send Authorization alongside x-api-key — Anthropic Mantle 401s on both");
}

@test:Config {}
function testBearerWithMixedCaseXApiKeyStillSuppressesAuthorization() returns error? {
    // The suppression is case-insensitive: HTTP header names are case-insensitive, so a
    // `routeOverrides` entry using `X-Api-Key` must not slip past the guard and resurrect
    // both headers.
    BedrockTransport transport = check new ({apiKey: "bedrock-api-key"}, "us-east-1",
            {baseUrl: string `https://bedrock-mantle.us-east-1.api.aws`, host: "bedrock-mantle.us-east-1.api.aws", path: "/anthropic/v1/messages",
                signingService: SIGNING_BEDROCK_MANTLE});
    map<string> headers = check transport.signedHeaders("{}",
            {"X-Api-Key": "bedrock-api-key"}, FIXED_CLOCK);
    test:assertFalse(headers.hasKey("Authorization"),
            "case-insensitive guard: X-Api-Key must also suppress Authorization");
}

@test:Config {}
function testStsCredentialsSignAndSendTheSecurityToken() returns error? {
    // The session token must be BOTH sent and signed — Bedrock requires it in the
    // canonical request, so it has to appear in SignedHeaders too.
    BedrockTransport transport = check new (
            {accessKeyId: "AKIATEST", secretAccessKey: "secret", sessionToken: "session-token-value"},
            "us-east-1",
            {baseUrl: string `https://bedrock-runtime.us-east-1.amazonaws.com`, host: "bedrock-runtime.us-east-1.amazonaws.com", path: "/model/m/converse",
                signingService: SIGNING_BEDROCK});
    map<string> headers = check transport.signedHeaders("{}", {}, FIXED_CLOCK);
    test:assertEquals(headers["X-Amz-Security-Token"], "session-token-value");
    string auth = headers["Authorization"] ?: "";
    test:assertTrue(auth.includes("x-amz-security-token"), "the token must be signed, not just sent: " + auth);
}

// ---- amzDate format: the boundary that shipped a bug ----

@test:Config {}
function testAmzDateTruncatesSecondsRatherThanRounding() returns error? {
    // REGRESSION: `time:Civil.second` is a decimal, and Ballerina's `<int>` ROUNDS
    // (half-to-even) rather than truncating — `<int>59.7d` is 60. That emitted
    // "...T235960Z", an invalid ISO 8601 basic timestamp. AWS rejects it, and the
    // resulting 403 is NOT retryable, so ~1 request in 120 (P(sec==59) * P(frac>=.5))
    // failed with an unreproducible AccessDeniedException.
    // https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv-signing-elements.html
    time:Civil atBoundary = {year: 2026, month: 7, day: 17, hour: 23, minute: 59, second: 59.734};
    [string, string] [amzDate, dateStamp] = check formatAmzTimestamps(atBoundary);
    test:assertEquals(amzDate, "20260717T235959Z", "seconds must truncate; 60 is not a valid seconds field");
    test:assertEquals(dateStamp, "20260717");
}

@test:Config {}
function testAmzDateFormatIsIso8601BasicAndZeroPadded() returns error? {
    time:Civil c = {year: 2026, month: 1, day: 5, hour: 4, minute: 3, second: 2.0};
    [string, string] [amzDate, dateStamp] = check formatAmzTimestamps(c);
    test:assertEquals(amzDate, "20260105T040302Z", "every field is zero-padded, no milliseconds");
    test:assertEquals(dateStamp, "20260105");
}

@test:Config {}
function testAmzDateHandlesAWholeSecondAndAMissingSecond() returns error? {
    time:Civil whole = {year: 2026, month: 7, day: 17, hour: 12, minute: 0, second: 0.0};
    [string, string] [wholeDate, _] = check formatAmzTimestamps(whole);
    test:assertEquals(wholeDate, "20260717T120000Z");
    // `second` is optional on time:Civil.
    time:Civil noSecond = {year: 2026, month: 7, day: 17, hour: 12, minute: 0};
    [string, string] [noSecDate, _] = check formatAmzTimestamps(noSecond);
    test:assertEquals(noSecDate, "20260717T120000Z");
}
