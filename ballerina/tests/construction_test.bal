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

// Construction-error tests. `init` fails before any I/O.

final BedrockCredentials TEST_CREDS = {accessKeyId: "AKIATEST", secretAccessKey: "secret"};

@test:Config {}
function testGuardrailOnMantleRouteFailsAtConstruction() {
    // guardrail on a Mantle route → error naming ApplyGuardrail.
    AnthropicModelProvider|ai:Error provider = new (
        "anthropic.claude-mythos-preview", TEST_CREDS, "us-east-1",
        guardrail = {guardrailIdentifier: "gr-1", guardrailVersion: "1"});
    test:assertTrue(provider is ai:Error);
    if provider is ai:Error {
        test:assertTrue(provider.message().includes("ApplyGuardrail"), provider.message());
    }
}

@test:Config {}
function testMantleOnChinaPartitionFailsAtConstruction() {
    // Mantle 'api.aws' host is not partition-templated.
    AnthropicModelProvider|ai:Error provider = new (
        "anthropic.claude-mythos-preview", TEST_CREDS, "cn-north-1");
    test:assertTrue(provider is ai:Error);
    if provider is ai:Error {
        test:assertTrue(provider.message().toLowerAscii().includes("partition") ||
            provider.message().toLowerAscii().includes("mantle"), provider.message());
    }
}

@test:Config {}
function testImportedModelArnIsRefusedAtConstruction() {
    // Custom Model Import is out of scope — refused by name, before any I/O.
    AnthropicModelProvider|ai:Error provider = new (
        "arn:aws:bedrock:us-west-2:123456789012:imported-model/abc123", TEST_CREDS, "us-east-1");
    test:assertTrue(provider is ai:Error);
    if provider is ai:Error {
        test:assertTrue(provider.message().includes("imported-model"), provider.message());
    }
}

@test:Config {}
function testConverseHappyPathConstructsWithoutError() returns error? {
    // A valid Converse model constructs (no I/O until chat()). Sonnet 4.6 is
    // runtime-only — absent from MANTLE_CAPABLE — so AUTO resolves it to Converse.
    // A dual-homed id (e.g. `anthropic.claude-opus-4-8`) would resolve to Mantle
    // under the AUTO preference order and would not exercise this path.
    Route route = check resolveRoute("anthropic.claude-sonnet-4-6", "us-east-1");
    test:assertEquals(route.family, CONVERSE, "test must exercise the Converse route");

    AnthropicModelProvider provider = check new ("anthropic.claude-sonnet-4-6", TEST_CREDS, "us-east-1");
    test:assertTrue(provider is AnthropicModelProvider);
}

@test:Config {}
function testBearerTokenCredentialsConstruct() returns error? {
    BedrockCredentials bearer = {apiKey: "bedrock-api-key"};
    AnthropicModelProvider provider = check new ("anthropic.claude-sonnet-4-6", bearer, "us-east-1");
    test:assertTrue(provider is AnthropicModelProvider);
}

// ---------------------------------------------------------------------------
// Region resolution. `region` now defaults to AWS_REGION/AWS_DEFAULT_REGION, so
// the "no region" case has to be a named construction error rather than the host
// `bedrock-runtime..amazonaws.com` and an opaque DNS failure.
// ---------------------------------------------------------------------------

@test:Config {}
function testMissingRegionFailsAtConstructionWithANamedError() returns error? {
    AnthropicModelProvider|error provider =
        new ("anthropic.claude-sonnet-4-6", TEST_CREDS, "");
    test:assertTrue(provider is error);
    if provider is error {
        test:assertTrue(provider.message().includes("AWS_REGION"), provider.message());
    }
}

@test:Config {}
function testAnArnCarryingItsOwnRegionNeedsNoRegionArgument() returns error? {
    // The region guard must run on the RESOLVED route, not the argument: an ARN
    // supplies its own region, so this is well-formed with no region and no
    // AWS_REGION in the environment.
    AnthropicModelProvider|error provider = new (
            "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/eu.anthropic.claude-sonnet-4-6",
            TEST_CREDS, "");
    test:assertFalse(provider is error, provider is error ? provider.message() : "");
}
