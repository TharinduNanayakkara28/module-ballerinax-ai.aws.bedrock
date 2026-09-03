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

// Vendor facades: AUTO routing, no Tier-2,
// per-model supportsStructuredOutput).

// ---- ApiFamily.AUTO ----

@test:Config {}
function testAutoIsTheDefaultAndRunsTheResolver() returns error? {
    // AUTO must behave exactly like "no forced family". By default
    // the resolver prefers Mantle for a Mantle-capable model, so opus-4-8 → Mantle;
    // the point of this test is that AUTO and "no family" agree.
    Route auto = check resolveRoute("anthropic.claude-opus-4-8", REGION, {apiFamily: AUTO});
    Route implicit = check resolveRoute("anthropic.claude-opus-4-8", REGION);
    test:assertEquals(auto.family, MANTLE);
    test:assertEquals(auto.family, implicit.family);
}

@test:Config {}
function testAutoStillLetsMantleOnlyModelsDefaultToMantle() returns error? {
    Route route = check resolveRoute("openai.gpt-5.4", REGION, {apiFamily: AUTO});
    test:assertEquals(route.family, MANTLE, "AUTO runs the ladder, which defaults gpt-5.4 to Mantle");
}

@test:Config {}
function testForcingInvokeOverridesTheResolver() returns error? {
    Route route = check resolveRoute("anthropic.claude-opus-4-8", REGION, {apiFamily: INVOKE});
    test:assertEquals(route.family, INVOKE);
}

// ---- supportsStructuredOutput ----

@test:Config {}
function testMantleRejectsTypedStructuredOutput() returns error? {
    // Mantle has no structured-output path: a non-string target must fail clean.
    Route route = check resolveRoute("anthropic.claude-mythos-preview", "us-east-1");
    Endpoint ep = check buildEndpoint(route);
    readonly & ModelConverter converter = check selectConverter(route);
    BedrockTransport transport = check new (TEST_CREDS, route.region, ep);
    ai:Prompt prompt = `Give me a number`;

    anydata|ai:Error result = structuredGenerate(false, MANTLE, converter, transport,
        route.effectiveModelId, {}, {temperature: 0.5d, maxTokens: 16}, prompt, int);
    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        test:assertTrue(result.message().includes("bedrock-mantle"), result.message());
        test:assertTrue(result.message().includes("anthropic.claude-mythos-preview"), result.message());
    }
}

@test:Config {}
function testConverseRouteSupportsStructuredOutputFlag() returns error? {
    // The flag is derived from the resolved route: true off Mantle, false on it.
    // nova-pro is Converse-default; opus-4-8 now prefers Mantle under AUTO.
    Route converse = check resolveRoute("amazon.nova-pro-v1:0", REGION);
    Route mantle = check resolveRoute("anthropic.claude-mythos-preview", REGION);
    test:assertTrue(converse.family != MANTLE, "Converse route → structured output supported");
    test:assertTrue(mantle.family == MANTLE, "Mantle route → structured output unsupported");
}

// ---- Nova Invoke: the schemaVersion landmine ----

@test:Config {}
function testNovaInvokeEmitsSchemaVersion() returns error? {
    InferenceParams params = {temperature: 0.5d, maxTokens: 100};
    map<json> body = check encodeNovaInvoke(SAMPLE_SYSTEM, SAMPLE_MESSAGES, [], (), params).ensureType();
    test:assertEquals(body["schemaVersion"], "messages-v1", "omit it and Nova fails validation");
    test:assertTrue(body.hasKey("system"), "system is top-level, not a message");
    test:assertFalse(body.hasKey("anthropic_version"), "Nova must not carry the Anthropic body field");
}

@test:Config {}
function testNovaDecodeSharesConverseShape() returns error? {
    json canned = {
        "output": {"message": {"role": "assistant", "content": [{"text": "Nova here"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 4, "outputTokens": 2}
    };
    DecodedResponse decoded = check decodeConverse(canned);
    test:assertEquals(decoded.message.content, "Nova here");
    test:assertEquals(decoded.usage.inputTokens, 4);
    test:assertEquals(decoded.stopReason, "end_turn");
}

// ---- OpenAI chat converter ----

@test:Config {}
function testOpenAIChatEmitsSystemAsMessageRole() returns error? {
    // Unlike Converse/Anthropic, the OpenAI wire format DOES use role:system.
    InferenceParams params = {temperature: 0.5d, maxTokens: 100};
    map<json> body = check encodeOpenAIChat(SAMPLE_SYSTEM, SAMPLE_MESSAGES, [], (), params).ensureType();
    json[] messages = check body["messages"].ensureType();
    map<json> first = check messages[0].ensureType();
    test:assertEquals(first["role"], "system");
    test:assertEquals(first["content"], "Be brief");
}

@test:Config {}
function testOpenAIChatDecodePopulatesUsageAndStopReason() returns error? {
    json canned = {
        "id": "chatcmpl-1",
        "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4}
    };
    DecodedResponse decoded = check decodeOpenAIChat(canned);
    test:assertEquals(decoded.message.content, "Hi");
    test:assertEquals(decoded.usage.inputTokens, 9);
    test:assertEquals(decoded.usage.outputTokens, 4);
    test:assertEquals(decoded.stopReason, "stop");
    test:assertEquals(decoded.responseId, "chatcmpl-1");
}

// ---- Mantle Responses converter (GPT-5.x) ----

@test:Config {}
function testResponsesDecodePopulatesUsageAndStopReason() returns error? {
    json canned = {
        "id": "resp_1",
        "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Yo"}]}],
        "usage": {"input_tokens": 6, "output_tokens": 2}
    };
    DecodedResponse decoded = check decodeResponses(canned);
    test:assertEquals(decoded.message.content, "Yo");
    test:assertEquals(decoded.usage.inputTokens, 6);
    test:assertEquals(decoded.usage.outputTokens, 2);
    test:assertEquals(decoded.stopReason, "completed");
    test:assertEquals(decoded.responseId, "resp_1");
}

@test:Config {}
function testResponsesEncodesSystemAsInstructions() returns error? {
    InferenceParams params = {temperature: 0.5d, maxTokens: 100};
    map<json> body = check encodeResponses(SAMPLE_SYSTEM, SAMPLE_MESSAGES, [], (), params).ensureType();
    test:assertEquals(body["instructions"], "Be brief");
    test:assertEquals(body["max_output_tokens"], 100);
}

// ---- vendor construction smoke tests (no I/O) ----

@test:Config {}
function testAllVendorProvidersConstruct() returns error? {
    // Smoke test: every vendor facade constructs (no I/O). Note routing varies —
    // qwen3-32b and gpt-oss now resolve to Mantle under AUTO — but
    // construction succeeds on any route.
    AmazonModelProvider amazon = check new ("amazon.nova-pro-v1:0", TEST_CREDS, REGION);
    MistralModelProvider mistral = check new ("mistral.mistral-large-2407-v1:0", TEST_CREDS, REGION);
    QwenModelProvider qwen = check new ("qwen.qwen3-32b-v1:0", TEST_CREDS, REGION);
    GoogleModelProvider google = check new ("google.gemma-3-27b-it", TEST_CREDS, REGION);
    DeepSeekModelProvider deepseek = check new ("us.deepseek.r1-v1:0", TEST_CREDS, REGION);
    OpenAIModelProvider openai = check new ("openai.gpt-oss-120b-1:0", TEST_CREDS, REGION);
    test:assertTrue(amazon is AmazonModelProvider);
    test:assertTrue(mistral is MistralModelProvider);
    test:assertTrue(qwen is QwenModelProvider);
    test:assertTrue(google is GoogleModelProvider);
    test:assertTrue(deepseek is DeepSeekModelProvider);
    test:assertTrue(openai is OpenAIModelProvider);
}

@test:Config {}
function testOpenAIMantleOnlyModelResolvesToMantleResponses() returns error? {
    // GPT-5.4 exists only on Mantle — the reason this module exists.
    Route route = check resolveRoute("openai.gpt-5.4", REGION);
    test:assertEquals(route.family, MANTLE);
    Endpoint ep = check buildEndpoint(route);
    test:assertEquals(ep.path, "/openai/v1/responses");
    test:assertEquals(ep.signingService, "bedrock-mantle");
}

@test:Config {}
function testGemma3DefaultsToMantleOnItsOwnChatCompletionsPath() returns error? {
    // Gemma 3 is dual-homed, so under AUTO the resolver prefers Mantle. The point that
    // matters: Gemma 3's Mantle path is `/v1/chat/completions`, DIFFERENT from Gemma
    // 4's `/openai/v1/responses` — one vendor prefix, two Mantle path families.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-3-27b-pt.html
    Route route = check resolveRoute("google.gemma-3-27b-it", REGION);
    test:assertEquals(route.family, MANTLE);
    Endpoint ep = check buildEndpoint(route);
    test:assertEquals(ep.signingService, "bedrock-mantle");
    test:assertEquals(ep.path, "/v1/chat/completions", "Gemma 3 uses Chat Completions, not Responses");

    // Forcing CONVERSE still reaches the runtime surface.
    Route converse = check resolveRoute("google.gemma-3-27b-it", REGION, {apiFamily: CONVERSE});
    test:assertEquals(converse.family, CONVERSE);
    Endpoint cep = check buildEndpoint(converse);
    test:assertEquals(cep.signingService, "bedrock");
    test:assertTrue(cep.host.startsWith("bedrock-runtime."));
}

@test:Config {}
function testGemma4IsMantleOnlyNotConverse() returns error? {
    // REGRESSION: this previously asserted CONVERSE + signing `bedrock`, which is
    // what the code did and what the card contradicts — the Gemma 4 support matrix
    // marks bedrock-runtime / Converse / Invoke / Messages all NO. Signing scope
    // `bedrock` against a Mantle-only model is a 403 on every single call.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-4-31b.html
    foreach string id in ["google.gemma-4-31b", "google.gemma-4-e2b", "google.gemma-4-26b-a4b"] {
        Route route = check resolveRoute(id, REGION);
        test:assertEquals(route.family, MANTLE, id + " is served ONLY on bedrock-mantle");
        Endpoint ep = check buildEndpoint(route);
        test:assertEquals(ep.signingService, "bedrock-mantle", "wrong signing scope for " + id);
        test:assertTrue(ep.host.startsWith("bedrock-mantle."), "wrong host for " + id);
        // The card is explicit that this path differs from the `/v1/responses`
        // other Mantle models use.
        test:assertEquals(ep.path, "/openai/v1/responses", "wrong Mantle path for " + id);
    }
}

@test:Config {}
function testGemma4CannotDoStructuredOutput() returns error? {
    // Falls out of being Mantle-only: no Converse route means no forced tools.
    GoogleModelProvider provider = check new (GEMMA_4_31B, TEST_CREDS, REGION);
    LiveFruitShape|ai:Error result = provider->generate(`Name a fruit.`);
    test:assertTrue(result is ai:Error, "Gemma 4 must refuse a typed target (Mantle route)");
}

type LiveFruitShape record {|
    string name;
|};

@test:Config {}
function testGptOssModelIdWithColonIsEncodedOnTheWire() returns error? {
    // `openai.gpt-oss-120b-1:0` carries a colon — the SigV4 double-encoding case. It
    // is Mantle-capable so AUTO now prefers Mantle; force CONVERSE to
    // exercise the runtime `/model/{id}` path where the colon lands in the URI.
    Route route = check resolveRoute("openai.gpt-oss-120b-1:0", REGION, {apiFamily: CONVERSE});
    Endpoint ep = check buildEndpoint(route);
    test:assertTrue(ep.path.includes("%3A"), "the model id's colon must be encoded on the wire");
    string canonical = getCanonicalUri(ep.path);
    test:assertTrue(canonical.includes("%253A"), "and double-encoded in the canonical URI");
}

@test:Config {}
function testAllKnownMantleOnlyModelsResolveToMantle() returns error? {
    // Cross-checked against AWS's endpoint-availability table, which is the only
    // page listing runtime-vs-mantle for every model in one place. Each id below is
    // marked `bedrock-runtime: NO` there and verified against its own card.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
    map<string> mantleOnly = {
        "openai.gpt-5.5": "/openai/v1/responses",
        "openai.gpt-5.4": "/openai/v1/responses",
        "openai.gpt-5.6-sol": "/openai/v1/responses",
        "openai.gpt-5.6-terra": "/openai/v1/responses",
        "openai.gpt-5.6-luna": "/openai/v1/responses",
        "anthropic.claude-mythos-preview": "/anthropic/v1/messages",
        "anthropic.claude-mythos-5": "/anthropic/v1/messages",
        "google.gemma-4-31b": "/openai/v1/responses",
        "google.gemma-4-e2b": "/openai/v1/responses",
        "google.gemma-4-26b-a4b": "/openai/v1/responses"
    };
    foreach [string, string] [id, expectedPath] in mantleOnly.entries() {
        Route route = check resolveRoute(id, REGION);
        test:assertEquals(route.family, MANTLE, id + " is served ONLY on bedrock-mantle");
        Endpoint ep = check buildEndpoint(route);
        test:assertEquals(ep.signingService, "bedrock-mantle", "wrong signing scope for " + id);
        test:assertEquals(ep.path, expectedPath, "wrong Mantle path for " + id);
    }
}

@test:Config {}
function testMantleCapableModelsDefaultToMantleUnderAuto() returns error? {
    // AUTO prefers MANTLE → CONVERSE → INVOKE, so every model with a
    // verified MANTLE_CAPABLE entry defaults to Mantle. (These previously defaulted
    // to Converse; that assertion is now inverted.)
    foreach string id in ["anthropic.claude-haiku-4-5", "anthropic.claude-opus-4-8", "zai.glm-5",
            "deepseek.v3.2", "mistral.mistral-large-3-675b-instruct", "qwen.qwen3-coder-480b-a35b-v1:0",
            "qwen.qwen3-32b-v1:0", "google.gemma-3-27b-it", "google.gemma-3-12b-it",
            "google.gemma-3-4b-it"] {
        Route route = check resolveRoute(id, REGION);
        test:assertEquals(route.family, MANTLE, id + " is Mantle-capable and must default to Mantle");
    }
}

@test:Config {}
function testRuntimeOnlyModelsDefaultToConverse() returns error? {
    // Models with NO Mantle entry sink to Converse — the table-driven safety: we
    // prefer Mantle only where we hold a verified wire shape, never by guessing.
    // sonnet-4-6 is genuinely runtime-only (its card marks bedrock-mantle NO), so it
    // will never gain an entry; nova-pro simply has none.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html
    foreach string id in ["anthropic.claude-sonnet-4-6", "amazon.nova-pro-v1:0",
            "mistral.mistral-large-2407-v1:0"] {
        Route route = check resolveRoute(id, REGION);
        test:assertEquals(route.family, CONVERSE, id + " has no Mantle entry, so it defaults to Converse");
    }
}
