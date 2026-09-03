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

// Routing for the flagship/workhorse ids added to the enums. Each pins a fact from
// that model's card that a wrong id or wrong routing assumption would violate.

@test:Config {}
function testDeepSeekV32BareIdDefaultsToMantle() returns error? {
    // The card marks In-Region YES, so the BARE id is callable (the opposite of R1,
    // which needs the `us.` CRIS prefix). It is dual-homed, so under AUTO the resolver
    // prefers Mantle. Mantle takes the bare id verbatim (no CRIS prefix on this id).
    Route route = check resolveRoute(DEEPSEEK_V3_2, "us-east-1");
    test:assertEquals(route.family, MANTLE);
    test:assertEquals(route.effectiveModelId, "deepseek.v3.2");
    // Forcing CONVERSE gives the runtime surface (needed for typed generate()).
    Route converse = check resolveRoute(DEEPSEEK_V3_2, "us-east-1", {apiFamily: CONVERSE});
    test:assertEquals(converse.family, CONVERSE);
    test:assertEquals(converse.effectiveModelId, "deepseek.v3.2");
}

@test:Config {}
function testMistralLarge3DefaultsToMantle() returns error? {
    Route route = check resolveRoute(MISTRAL_LARGE_3, "us-east-1");
    test:assertEquals(route.family, MANTLE, "dual-homed model prefers Mantle under AUTO");
    test:assertEquals(route.effectiveModelId, "mistral.mistral-large-3-675b-instruct");
}

@test:Config {}
function testQwen3Coder480BUsesItsMantleIdUnderAuto() returns error? {
    // The enum carries the bedrock-RUNTIME id; the mantle id differs. Under AUTO it
    // now routes to Mantle, so the MANTLE-side id must go on the wire.
    Route route = check resolveRoute(QWEN3_CODER_480B, "us-east-1");
    test:assertEquals(route.family, MANTLE);
    test:assertEquals(route.effectiveModelId, "qwen.qwen3-coder-480b-a35b-instruct",
            "Mantle has its own id for this model");
    test:assertEquals(route.bareModelId, "qwen.qwen3-coder-480b-a35b-v1:0", "lookup key stays the runtime id");
    // Forcing CONVERSE keeps the runtime id on the wire.
    Route converse = check resolveRoute(QWEN3_CODER_480B, "us-east-1", {apiFamily: CONVERSE});
    test:assertEquals(converse.effectiveModelId, "qwen.qwen3-coder-480b-a35b-v1:0");
}

@test:Config {}
function testClaudeSonnet5DefaultsToMantleAndUsesTheMessagesPath() returns error? {
    // Dual-homed: AUTO prefers Mantle. The Messages entry is what
    // makes that resolve to `/anthropic/v1/messages`.
    Route auto = check resolveRoute(CLAUDE_SONNET_5, "us-east-1");
    test:assertEquals(auto.family, MANTLE);
    MantleEntry? entry = auto.mantleEntry;
    if entry is () {
        test:assertFail("Sonnet 5 under AUTO must yield a Mantle entry");
    }
    test:assertEquals(entry.path, "/anthropic/v1/messages");
    test:assertEquals((check mantleConverterForPath(entry.path)).toolChoice, ANTHROPIC_TOOL_CHOICE);
    // And CONVERSE is still reachable explicitly (for typed generate()).
    Route converse = check resolveRoute(CLAUDE_SONNET_5, "us-east-1", {apiFamily: CONVERSE});
    test:assertEquals(converse.family, CONVERSE);
}

@test:Config {}
function testClaudeOpus5DefaultsToMantleAndUsesTheMessagesPath() returns error? {
    // Dual-homed (bedrock-runtime YES + bedrock-mantle YES), Messages API, same id on
    // both endpoints — so AUTO prefers Mantle on `/anthropic/v1/messages`.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-5.html
    Route auto = check resolveRoute(CLAUDE_OPUS_5, "us-east-1");
    test:assertEquals(auto.family, MANTLE);
    MantleEntry? entry = auto.mantleEntry;
    if entry is () {
        test:assertFail("Opus 5 under AUTO must yield a Mantle entry");
    }
    test:assertEquals(entry.path, "/anthropic/v1/messages");
    test:assertEquals((check mantleConverterForPath(entry.path)).toolChoice, ANTHROPIC_TOOL_CHOICE);
    test:assertTrue(usesApiKeyHeader(entry.path));
    // The card lists no separate Mantle id, so the wire id must not be rewritten.
    test:assertEquals(auto.effectiveModelId, "anthropic.claude-opus-5");
    // CONVERSE stays reachable explicitly (for typed generate()).
    Route converse = check resolveRoute(CLAUDE_OPUS_5, "us-east-1", {apiFamily: CONVERSE});
    test:assertEquals(converse.family, CONVERSE);
}

@test:Config {}
function testClaudeOpus5AcceptsItsGeoAndGlobalProfiles() returns error? {
    // The card lists `us.`/`eu.`/`au.` geo ids and `global.` — all must strip for
    // lookup and re-apply on the Converse wire.
    foreach string id in ["us.anthropic.claude-opus-5", "eu.anthropic.claude-opus-5",
        "au.anthropic.claude-opus-5", "global.anthropic.claude-opus-5"] {
        Route route = check resolveRoute(id, "us-east-1");
        test:assertEquals(route.family, CONVERSE, id + " must leave Mantle once prefixed");
        test:assertEquals(route.effectiveModelId, id, "the prefix must survive onto the wire");
    }
}

@test:Config {}
function testClaudeSonnet5AcceptsItsUsCrisProfile() returns error? {
    // The card lists `us.anthropic.claude-sonnet-5` as the US geo id: it must strip
    // for lookup and re-apply on the Converse wire.
    Route route = check resolveRoute("us.anthropic.claude-sonnet-5", "us-east-1");
    test:assertEquals(route.family, CONVERSE);
    test:assertEquals(route.effectiveModelId, "us.anthropic.claude-sonnet-5",
            "the geo prefix must survive onto the Converse wire");
}

@test:Config {}
function testNewModelProvidersConstruct() returns error? {
    _ = check new AnthropicModelProvider(CLAUDE_SONNET_5, TEST_CREDS, REGION);
    _ = check new MistralModelProvider(MISTRAL_LARGE_3, TEST_CREDS, REGION);
    _ = check new QwenModelProvider(QWEN3_CODER_480B, TEST_CREDS, REGION);
    _ = check new DeepSeekModelProvider(DEEPSEEK_V3_2, TEST_CREDS, REGION);
}

// ---- The sharp edge: generate() on an AUTO-routed Mantle model ----

type FruitShape record {|
    string name;
|};

@test:Config {}
function testTypedGenerateFallsBackToConverseOnAnAutoRoutedDualHomedModel() returns error? {
    // Under AUTO a dual-homed model routes CHAT to Mantle, which has no structured
    // output. Rather than fail a typed generate() on exactly the flagship models,
    // generate() resolves a SECOND spine on bedrock-runtime and uses that.
    // chat() is untouched.
    RouteConfig auto = {};
    Route chatRoute = check resolveRoute(CLAUDE_SONNET_5, REGION, auto);
    test:assertEquals(chatRoute.family, MANTLE, "chat() still goes to Mantle under AUTO");

    [ApiFamily, string, readonly & ModelConverter, BedrockTransport, map<string>] genSpine =
        check generateSpineFor(CLAUDE_SONNET_5, auto);
    test:assertEquals(genSpine[0], CONVERSE, "generate() must fall back to Converse");
    test:assertEquals(genSpine[1], CLAUDE_SONNET_5, "the runtime id goes on the wire");
}

@test:Config {}
function testTypedGenerateStillErrorsOnAMantleOnlyModel() returns error? {
    // GPT-5.4 is Mantle-ONLY (Responses; no Converse, no Invoke). There is no
    // bedrock-runtime route to fall back to, so the clean local error stays — far
    // better than sending the request to an endpoint that does not serve the model.
    OpenAIModelProvider provider = check new ("openai.gpt-5.4", TEST_CREDS, REGION);
    FruitShape|ai:Error typed = provider->generate(`Name a fruit.`);
    test:assertTrue(typed is ai:Error, "a Mantle-only model cannot do structured output");
    if typed is ai:Error {
        test:assertTrue(typed.message().includes("bedrock-mantle"), typed.message());
        test:assertTrue(typed.message().includes("openai.gpt-5.4"), typed.message());
    }
}

@test:Config {}
function testExplicitMantleDoesNotSilentlyFallBack() returns error? {
    // The fallback is an AUTO convenience only. An EXPLICIT apiFamily = MANTLE is the
    // caller naming a destination; quietly going somewhere else would break the one
    // guarantee an explicit override exists to provide.
    [ApiFamily, string, readonly & ModelConverter, BedrockTransport, map<string>] genSpine =
        check generateSpineFor(CLAUDE_SONNET_5, {apiFamily: MANTLE});
    test:assertEquals(genSpine[0], MANTLE, "explicit MANTLE must stay on Mantle");
}

// Resolves the chat spine, then asks which spine generate() would use.
function generateSpineFor(string model, RouteConfig routeConfig)
        returns [ApiFamily, string, readonly & ModelConverter, BedrockTransport, map<string>]|error {
    [Route, readonly & ModelConverter, BedrockTransport] [route, converter, transport] =
        check resolveSpine("Test", TEST_CREDS, model, REGION, DEFAULT_SERVICE_URL, routeConfig,
            (), (), ());
    return resolveGenerateSpine("Test", TEST_CREDS, model, REGION, DEFAULT_SERVICE_URL, routeConfig,
        (), (), (), route, converter, transport, {});
}

// Qwen3 32B is served under a DIFFERENT id on Mantle (`qwen.qwen3-32b`) than on
// bedrock-runtime (`qwen.qwen3-32b-v1:0`). The caller always passes the RUNTIME id
// — MANTLE_CAPABLE is keyed by it — and the resolver performs the swap. Getting
// this backwards sends an id Mantle does not know, so both halves are asserted.
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-qwen-qwen3-32b.html
@test:Config {}
function testQwen332bUsesItsOwnIdOnMantle() returns error? {
    Route mantle = check resolveRoute("qwen.qwen3-32b-v1:0", REGION, {apiFamily: MANTLE});
    test:assertEquals(mantle.family, MANTLE);
    test:assertEquals(mantle.effectiveModelId, "qwen.qwen3-32b",
            "Mantle serves this model under its own id");
    test:assertEquals(mantle.bareModelId, "qwen.qwen3-32b-v1:0",
            "the lookup key stays the runtime id");

    Route converse = check resolveRoute("qwen.qwen3-32b-v1:0", REGION, {apiFamily: CONVERSE});
    test:assertEquals(converse.effectiveModelId, "qwen.qwen3-32b-v1:0",
            "the runtime surface keeps the runtime id");
}
