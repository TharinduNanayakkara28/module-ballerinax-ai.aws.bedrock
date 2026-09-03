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

// Golden-file converter tests. Fixed messages in → asserted body out;
// canned response in → asserted DecodedResponse out.

final ResolvedMessage[] & readonly SAMPLE_MESSAGES = [{parts: [{text: "Hello"}]}];
final string SAMPLE_SYSTEM = "Be brief";

// Encoders take ALREADY-RESOLVED messages (see content_parts.bal), so the golden
// tests build them directly — no network, no credentials, exactly as before.
isolated function userText(string text) returns ResolvedUserMessage => {parts: [{text}]};
isolated function userImage(string mimeType, byte[] data) returns ResolvedUserMessage =>
    {parts: [{mimeType, data}]};

// ---- Converse encode ----

@test:Config {}
function testConverseHoistsSystemToTopLevelField() returns error? {
    InferenceParams params = {temperature: 0.5, maxTokens: 100};
    map<json> body = check encodeConverse(SAMPLE_SYSTEM, SAMPLE_MESSAGES, [], (), params).ensureType();
    test:assertTrue(body.hasKey("system"), "system must be a top-level field, not a message");
    json system = body["system"];
    test:assertEquals(system, <json>[{"text": "Be brief"}]);
    // Ensure no message carries role:system.
    json[] messages = check body["messages"].ensureType();
    foreach json m in messages {
        map<json> msg = check m.ensureType();
        test:assertNotEquals(msg["role"], "system");
    }
}

@test:Config {}
function testConversePerCallStopOverridesConfiguredStopSequences() returns error? {
    InferenceParams params = {temperature: 0.5, maxTokens: 100, stopSequences: ["CONFIGURED"]};
    map<json> body = check encodeConverse((), SAMPLE_MESSAGES, [], "PERCALL", params).ensureType();
    map<json> inferenceConfig = check body["inferenceConfig"].ensureType();
    test:assertEquals(inferenceConfig["stopSequences"], <json>["PERCALL"], "per-call stop must win outright");
}

@test:Config {}
function testConverseForwardsAdditionalModelRequestFieldsVerbatim() returns error? {
    InferenceParams params = {temperature: 0.5, maxTokens: 100, additionalModelRequestFields: {"top_k": 200}};
    map<json> body = check encodeConverse((), SAMPLE_MESSAGES, [], (), params).ensureType();
    test:assertEquals(body["additionalModelRequestFields"], <json>{"top_k": 200});
}

@test:Config {}
function testConverseGuardrailIsBodyField() returns error? {
    InferenceParams params = {
        temperature: 0.5,
        maxTokens: 100,
        guardrail: {guardrailIdentifier: "gr-1", guardrailVersion: "DRAFT"}
    };
    map<json> body = check encodeConverse((), SAMPLE_MESSAGES, [], (), params).ensureType();
    map<json> guardrailConfig = check body["guardrailConfig"].ensureType();
    test:assertEquals(guardrailConfig["guardrailIdentifier"], "gr-1");
}

// ---- Invoke-Anthropic vs Mantle Messages encode ----

@test:Config {}
function testInvokeAnthropicEmitsBedrockVersionBodyField() returns error? {
    InferenceParams params = {temperature: 0.5, maxTokens: 100};
    map<json> body = check encodeInvokeAnthropic(SAMPLE_SYSTEM, SAMPLE_MESSAGES, [], (), params).ensureType();
    test:assertEquals(body["anthropic_version"], "bedrock-2023-05-31", "required body field");
    test:assertEquals(body["system"], "Be brief", "system is top-level");
}

@test:Config {}
function testMantleMessagesOmitsBedrockVersionBodyField() returns error? {
    // Mantle carries anthropic-version in the HEADER, not the body.
    InferenceParams params = {temperature: 0.5, maxTokens: 100};
    map<json> body = check encodeMantleMessages(SAMPLE_SYSTEM, SAMPLE_MESSAGES, [], (), params).ensureType();
    test:assertFalse(body.hasKey("anthropic_version"), "Mantle must NOT emit the Invoke body version field");
}

// ---- Converse decode: usage + stopReason always populated ----

@test:Config {}
function testConverseDecodePopulatesUsageAndStopReason() returns error? {
    json canned = {
        "output": {"message": {"role": "assistant", "content": [{"text": "Hi there"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
    };
    DecodedResponse decoded = check decodeConverse(canned);
    test:assertEquals(decoded.message.content, "Hi there");
    test:assertEquals(decoded.usage.inputTokens, 10);
    test:assertEquals(decoded.usage.outputTokens, 5);
    test:assertEquals(decoded.stopReason, "end_turn");
}

@test:Config {}
function testConverseDecodeSurfacesGuardrailIntervention() returns error? {
    // never silently drop the fact a guardrail fired.
    json canned = {
        "output": {"message": {"role": "assistant", "content": [{"text": ""}]}},
        "stopReason": "guardrail_intervened",
        "usage": {"inputTokens": 3, "outputTokens": 0}
    };
    DecodedResponse decoded = check decodeConverse(canned);
    test:assertEquals(decoded.guardrailAction, INTERVENED);
}

@test:Config {}
function testConverseDecodeExtractsToolUse() returns error? {
    json canned = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "tu_1", "name": "getWeather", "input": {"city": "NYC"}}}]
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 8, "outputTokens": 12}
    };
    DecodedResponse decoded = check decodeConverse(canned);
    ai:FunctionCall[] toolCalls = check decoded.message.toolCalls.ensureType();
    test:assertEquals(toolCalls[0].name, "getWeather");
    test:assertEquals(toolCalls[0].arguments, <map<json>>{"city": "NYC"});
    test:assertEquals(toolCalls[0].id, "tu_1");
}

// ---- Anthropic Messages decode ----

@test:Config {}
function testAnthropicMessagesDecodePopulatesUsageStopReasonAndId() returns error? {
    json canned = {
        "id": "msg_01",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 7, "output_tokens": 3}
    };
    DecodedResponse decoded = check decodeAnthropicMessages(canned);
    test:assertEquals(decoded.message.content, "Hello!");
    test:assertEquals(decoded.responseId, "msg_01");
    test:assertEquals(decoded.usage.inputTokens, 7);
    test:assertEquals(decoded.usage.outputTokens, 3);
    test:assertEquals(decoded.stopReason, "end_turn");
}

@test:Config {}
function testInvokeAnthropicDecodeSurfacesBodyGuardrailAction() returns error? {
    // InvokeModel API ref: the fired signal is a response-BODY field.
    json canned = {
        "id": "msg_gr",
        "role": "assistant",
        "content": [{"type": "text", "text": "blocked"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 2, "output_tokens": 1},
        "amazon-bedrock-guardrailAction": "INTERVENED"
    };
    DecodedResponse decoded = check decodeAnthropicMessages(canned);
    test:assertEquals(decoded.guardrailAction, INTERVENED);
}

@test:Config {}
function testAnthropicMessagesDecodeExtractsToolUse() returns error? {
    json canned = {
        "id": "msg_02",
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 4, "output_tokens": 9}
    };
    DecodedResponse decoded = check decodeAnthropicMessages(canned);
    ai:FunctionCall[] toolCalls = check decoded.message.toolCalls.ensureType();
    test:assertEquals(toolCalls[0].name, "lookup");
    test:assertEquals(toolCalls[0].id, "toolu_1");
}

// ---- Mistral: two dialects, one vendor prefix ----

@test:Config {}
function testMistralTextEncodesPromptNotMessages() returns error? {
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-text-completion.html
    map<json> body = check encodeMistralText((), SAMPLE_MESSAGES, [], (),
            {temperature: 0.5, maxTokens: 100}).ensureType();
    test:assertTrue(body.hasKey("prompt"), "the text dialect sends `prompt`, never `messages`");
    test:assertFalse(body.hasKey("messages"));
    test:assertEquals(body["prompt"], <json>"<s>[INST] Hello [/INST]",
            "user text must sit inside Mistral's [INST] template");
}

@test:Config {}
function testMistralTextFoldsSystemIntoTheFirstInstructionBlock() returns error? {
    // The template has no system slot; system must never become a role:system
    // message, so it is prepended to the first [INST] block.
    map<json> body = check encodeMistralText(SAMPLE_SYSTEM, SAMPLE_MESSAGES, [], (),
            {temperature: 0.5, maxTokens: 100}).ensureType();
    test:assertEquals(body["prompt"], <json>"<s>[INST] Be brief\n\nHello [/INST]");
}

@test:Config {}
function testMistralTextRendersMultiTurnTemplate() returns error? {
    ResolvedMessage[] messages = [
        userText("First?"),
        {role: ai:ASSISTANT, content: "Answer."},
        userText("Second?")
    ];
    map<json> body = check encodeMistralText((), messages, [], (),
            {temperature: 0.5, maxTokens: 100}).ensureType();
    // Assistant turns sit OUTSIDE the [INST] tokens and are closed with </s>.
    test:assertEquals(body["prompt"], <json>"<s>[INST] First? [/INST] Answer.</s>[INST] Second? [/INST]");
}

@test:Config {}
function testMistralTextDecodePopulatesUsageAndStopReason() returns error? {
    json canned = {"outputs": [{"text": "hi there", "stop_reason": "length"}]};
    DecodedResponse decoded = check decodeMistralText(canned);
    test:assertEquals(decoded.message.content, "hi there");
    // This dialect returns no token counts at all — usage must still be populated.
    test:assertEquals(decoded.usage, {inputTokens: 0, outputTokens: 0});
    test:assertEquals(decoded.stopReason, "length");
}

@test:Config {}
function testMistralChatDecodeReadsStopReasonNotFinishReason() returns error? {
    // The regression that motivated splitting Mistral off the OpenAI converter: this
    // dialect spells it `stop_reason`, so the OpenAI decoder left stopReason empty.
    json canned = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "stop_reason": "stop"}]
    };
    DecodedResponse decoded = check decodeMistralChat(canned);
    test:assertEquals(decoded.stopReason, "stop");
    test:assertEquals(decoded.message.content, "hello");
    test:assertEquals(decoded.usage, {inputTokens: 0, outputTokens: 0});
}

@test:Config {}
function testMistralChatKeepsSystemAsALeadingMessage() returns error? {
    // Unlike Converse/Anthropic, this dialect DOES model system as a role (per the
    // AWS page), so the hoisted system is re-added as the first message.
    map<json> body = check encodeMistralChat(SAMPLE_SYSTEM, SAMPLE_MESSAGES, [], (),
            {temperature: 0.5, maxTokens: 100}).ensureType();
    json[] messages = check body["messages"].ensureType();
    map<json> first = check messages[0].ensureType();
    test:assertEquals(first["role"], "system");
    test:assertEquals(first["content"], "Be brief");
}

@test:Config {}
function testMistralDialectIsSelectedByModelId() returns error? {
    // mistral-large-2402 is text-completion but mistral-large-2407 is chat: same
    // family, four months apart, opposite wire shapes.
    test:assertTrue(usesMistralTextDialect("mistral.mistral-large-2402-v1:0"));
    test:assertFalse(usesMistralTextDialect("mistral.mistral-large-2407-v1:0"));
    test:assertTrue(usesMistralTextDialect("mistral.mistral-7b-instruct-v0:2"));
    test:assertTrue(usesMistralTextDialect("mistral.mixtral-8x7b-instruct-v0:1"));
    // An id we have never seen defaults to the chat dialect, not the legacy one.
    test:assertFalse(usesMistralTextDialect("mistral.mistral-large-9999-v1:0"));
}

@test:Config {}
function testSelectInvokeConverterPicksTheRightMistralDialect() returns error? {
    readonly & ModelConverter text = check selectInvokeConverter("mistral.mistral-7b-instruct-v0:2");
    test:assertEquals(text.toolChoice, NO_TOOL_CHOICE);
    readonly & ModelConverter chat = check selectInvokeConverter("mistral.mistral-large-2407-v1:0");
    test:assertEquals(chat.toolChoice, MISTRAL_TOOL_CHOICE);
}

// ---- DeepSeek: text completion, not the OpenAI chat shape ----

@test:Config {}
function testDeepSeekInvokeEmitsPromptNotMessages() returns error? {
    // REGRESSION: DeepSeek was routed to the OpenAI chat converter, which emits
    // `messages` — a 400 on every DeepSeek Invoke request. AWS documents this
    // dialect as text completion.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html
    map<json> body = check encodeDeepSeekInvoke((), SAMPLE_MESSAGES, [], (),
            {temperature: 0.5, maxTokens: 100}).ensureType();
    test:assertTrue(body.hasKey("prompt"));
    test:assertFalse(body.hasKey("messages"), "DeepSeek's Invoke dialect has no messages array");
    string prompt = check body["prompt"].ensureType();
    test:assertTrue(prompt.includes("Hello"));
    test:assertTrue(prompt.startsWith("<｜begin▁of▁sentence｜>"), "must use DeepSeek's own delimiters");
    test:assertTrue(prompt.endsWith("<｜Assistant｜><think>\n"), "must hand the turn over and open <think>");
}

@test:Config {}
function testDeepSeekDecodeReadsChoicesTextNotMessage() returns error? {
    // The `choices` wrapper makes it look OpenAI-shaped, but the payload is
    // `choices[].text` + `stop_reason`, and there is no usage object at all.
    json canned = {"choices": [{"text": "the answer", "stop_reason": "length"}]};
    DecodedResponse decoded = check decodeDeepSeekInvoke(canned);
    test:assertEquals(decoded.message.content, "the answer");
    test:assertEquals(decoded.stopReason, "length");
    test:assertEquals(decoded.usage, {inputTokens: 0, outputTokens: 0});
}

@test:Config {}
function testDeepSeekRejectsToolsRatherThanDroppingThem() {
    json|ai:Error encoded = encodeDeepSeekInvoke((), SAMPLE_MESSAGES, [RESULT_TOOL_DEF], (),
            {temperature: 0.5, maxTokens: 100});
    test:assertTrue(encoded is ai:Error, "this dialect has no tools; dropping them silently would be worse");
}

@test:Config {}
function testDeepSeekSelectsItsOwnInvokeConverter() returns error? {
    readonly & ModelConverter byPrefix = check selectInvokeConverter("deepseek.r1-v1:0");
    test:assertEquals(byPrefix.toolChoice, NO_TOOL_CHOICE);
    // GPT-OSS/Qwen keep the OpenAI chat converter.
    readonly & ModelConverter openai = check selectInvokeConverter("openai.gpt-oss-120b-1:0");
    test:assertEquals(openai.toolChoice, OPENAI_CHAT_TOOL_CHOICE);
}

@test:Config {}
function testDeepSeekDialectIsSelectedByModelId() {
    // R1 is the ONLY DeepSeek id on the text-completion dialect. V3.1/V3.2 take
    // `messages` on InvokeModel per their model cards, so a new `deepseek.` id
    // defaults to chat — the opposite guess is a hard 400.
    test:assertTrue(usesDeepSeekTextDialect("deepseek.r1-v1:0"));
    test:assertFalse(usesDeepSeekTextDialect("deepseek.v3.2"));
    test:assertFalse(usesDeepSeekTextDialect("deepseek.v3-v1:0"));
    test:assertFalse(usesDeepSeekTextDialect("deepseek.v4-whatever"));
}

@test:Config {}
function testDeepSeekV32InvokeEmitsMessagesNotPrompt() returns error? {
    // REGRESSION: every `deepseek.` id went to the text-completion converter, so
    // INVOKE on deepseek.v3.2 sent `prompt` and Bedrock answered
    // `ValidationException ... missing field messages`.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html
    readonly & ModelConverter converter = check selectInvokeConverter("deepseek.v3.2");
    test:assertEquals(converter.toolChoice, OPENAI_CHAT_TOOL_CHOICE);
    RequestEncoder encode = converter.encode;
    map<json> body = check encode((), SAMPLE_MESSAGES, [], (),
            {temperature: 0.5, maxTokens: 100}).ensureType();
    test:assertTrue(body.hasKey("messages"));
    test:assertFalse(body.hasKey("prompt"), "V3.x is chat-shaped, not text completion");
}

// ---- Converse serviceTier is an object, not a string ----

@test:Config {}
function testConverseServiceTierIsAnObjectWithAwsValues() returns error? {
    // REGRESSION: emitted as a bare string `"standard"` — wrong shape AND wrong
    // value (AWS's baseline tier is `default`; there is no `standard`). Any user
    // setting serviceTier got a 400 whose message blamed the route.
    // https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
    map<json> body = check encodeConverse((), SAMPLE_MESSAGES, [], (),
            {temperature: 0.5, maxTokens: 100, serviceTier: TIER_FLEX}).ensureType();
    test:assertEquals(body["serviceTier"], <json>{"type": "flex"});

    map<json> dflt = check encodeConverse((), SAMPLE_MESSAGES, [], (),
            {temperature: 0.5, maxTokens: 100, serviceTier: TIER_DEFAULT}).ensureType();
    test:assertEquals(dflt["serviceTier"], <json>{"type": "default"});
}

@test:Config {}
function testConverseOmitsServiceTierWhenUnset() returns error? {
    map<json> body = check encodeConverse((), SAMPLE_MESSAGES, [], (),
            {temperature: 0.5, maxTokens: 100}).ensureType();
    test:assertFalse(body.hasKey("serviceTier"));
}

// ---- Converse performanceConfig.latency ----

@test:Config {}
function testConverseEmitsLatencyOptimizedAsAnObject() returns error? {
    // `latencyOptimized = true` → `"performanceConfig": {"latency": "optimized"}`,
    // an object like serviceTier, not a bare string.
    // https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
    map<json> body = check encodeConverse((), SAMPLE_MESSAGES, [], (),
            {temperature: 0.5, maxTokens: 100, latencyOptimized: true}).ensureType();
    test:assertEquals(body["performanceConfig"], <json>{"latency": "optimized"});
}

@test:Config {}
function testConverseOmitsPerformanceConfigWhenUnsetOrFalse() returns error? {
    // Unset sends nothing — `standard` is the default, so there is no value to emit.
    map<json> unset = check encodeConverse((), SAMPLE_MESSAGES, [], (),
            {temperature: 0.5, maxTokens: 100}).ensureType();
    test:assertFalse(unset.hasKey("performanceConfig"));
    // Explicit false is also `standard` — the same as unset, never a `standard` value.
    map<json> off = check encodeConverse((), SAMPLE_MESSAGES, [], (),
            {temperature: 0.5, maxTokens: 100, latencyOptimized: false}).ensureType();
    test:assertFalse(off.hasKey("performanceConfig"), "false must not emit a standard value");
}

// ---- Responses decode: reasoning/refusal must not leak into assistant content ----

@test:Config {}
function testResponsesDecodeDropsReasoningAndRefusalText() returns error? {
    // GPT-5.x on Mantle returns `reasoning` items alongside `message` items, and a
    // `message` item's content can hold a `refusal` block. Both carry a `text`
    // field. Appending every block's text leaks the model's chain-of-thought into
    // ai:ChatAssistantMessage.content and hands it back to the caller as output.
    json canned = {
        "id": "resp_01",
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "SECRET-THINKING"}]
            },
            {
                "type": "message",
                "content": [
                    {"type": "refusal", "text": "SECRET-REFUSAL"},
                    {"type": "output_text", "text": "Visible answer"}
                ]
            }
        ],
        "usage": {"input_tokens": 11, "output_tokens": 4},
        "status": "completed"
    };
    DecodedResponse decoded = check decodeResponses(canned);
    test:assertEquals(decoded.message.content, "Visible answer",
            "only output_text blocks of message items may reach the assistant content");
    string content = decoded.message.content ?: "";
    test:assertFalse(content.includes("SECRET-THINKING"), "reasoning text must never be returned as output");
    test:assertFalse(content.includes("SECRET-REFUSAL"), "refusal text must never be merged into output");
    test:assertEquals(decoded.usage.inputTokens, 11);
    test:assertEquals(decoded.usage.outputTokens, 4);
}

// ---- Nova passthrough must merge into inferenceConfig, not replace it ----

@test:Config {}
function testNovaPassthroughMergesIntoInferenceConfig() returns error? {
    // Nova nests its inference knobs under a body key the converter also builds. A
    // passthrough entry for that key used to REPLACE it, silently discarding the
    // caller's maxTokens/temperature/stopSequences.
    InferenceParams params = {
        temperature: 0.3,
        maxTokens: 256,
        stopSequences: ["STOP"],
        additionalModelRequestFields: {"inferenceConfig": {"topK": 20}}
    };
    map<json> body = check encodeNovaInvoke((), SAMPLE_MESSAGES, [], (), params).ensureType();
    map<json> inferenceConfig = check body["inferenceConfig"].ensureType();
    test:assertEquals(inferenceConfig["topK"], 20, "the passthrough key must be merged in");
    test:assertEquals(inferenceConfig["maxTokens"], 256, "maxTokens must survive the merge");
    test:assertEquals(inferenceConfig["temperature"], 0.3d, "temperature must survive the merge");
    test:assertEquals(inferenceConfig["stopSequences"], <json>["STOP"], "stopSequences must survive the merge");
    test:assertEquals(body["schemaVersion"], "messages-v1");
}

@test:Config {}
function testNovaPassthroughStillOverwritesNonNestedKeys() returns error? {
    // Only `inferenceConfig` merges; every other passthrough key keeps the
    // existing overwrite behaviour shared with the other converters.
    InferenceParams params = {
        temperature: 0.3,
        maxTokens: 256,
        additionalModelRequestFields: {"reasoningConfig": {"type": "enabled"}}
    };
    map<json> body = check encodeNovaInvoke((), SAMPLE_MESSAGES, [], (), params).ensureType();
    test:assertEquals(body["reasoningConfig"], <json>{"type": "enabled"});
}

// ---- intField must not invent a token count from a fractional decimal ----

@test:Config {}
function testIntFieldRejectsAFractionalDecimal() {
    test:assertEquals(intField({"n": 5d}, "n"), 5, "an integral decimal converts faithfully");
    test:assertEquals(intField({"n": 1.5d}, "n"), (),
            "a fractional token count is unusable and must be absent, not rounded to 2");
}

// ---- temperature is OMITTED when unset, on every converter ----

@test:Config {}
function testTemperatureIsOmittedFromEveryConverterWhenUnset() returns error? {
    // Not a cosmetic default. Claude 4.7+/Opus 5/Sonnet 5/Mythos 5 and OpenAI's
    // GPT-5.x reject `temperature` outright, so a module default made those model
    // ids return 400 on every single request. Unset MUST mean absent on the wire.
    InferenceParams bare = {maxTokens: 100};

    map<json> converse = check encodeConverse((), SAMPLE_MESSAGES, [], (), bare).ensureType();
    map<json> converseInference = check converse["inferenceConfig"].ensureType();
    test:assertFalse(converseInference.hasKey("temperature"), "Converse must omit temperature");

    map<json> nova = check encodeNovaInvoke((), SAMPLE_MESSAGES, [], (), bare).ensureType();
    map<json> novaInference = check nova["inferenceConfig"].ensureType();
    test:assertFalse(novaInference.hasKey("temperature"), "Nova must omit temperature");

    map<json> anthropic = check encodeAnthropicMessages((), SAMPLE_MESSAGES, [], (), bare, false).ensureType();
    test:assertFalse(anthropic.hasKey("temperature"), "Anthropic Messages must omit temperature");

    map<json> responses = check encodeResponses((), SAMPLE_MESSAGES, [], (), bare).ensureType();
    test:assertFalse(responses.hasKey("temperature"), "Responses must omit temperature");

    map<json> chat = check encodeOpenAIChat((), SAMPLE_MESSAGES, [], (), bare).ensureType();
    test:assertFalse(chat.hasKey("temperature"), "OpenAI chat must omit temperature");

    map<json> mistral = check encodeMistralChat((), SAMPLE_MESSAGES, [], (), bare).ensureType();
    test:assertFalse(mistral.hasKey("temperature"), "Mistral chat must omit temperature");
}

@test:Config {}
function testTemperatureIsStillEmittedWhenTheCallerSetsIt() returns error? {
    // The omission must be driven by "unset", not by dropping support.
    InferenceParams withTemp = {maxTokens: 100, temperature: 0.25};
    map<json> converse = check encodeConverse((), SAMPLE_MESSAGES, [], (), withTemp).ensureType();
    map<json> inference = check converse["inferenceConfig"].ensureType();
    test:assertEquals(inference["temperature"], 0.25d);

    map<json> anthropic = check encodeAnthropicMessages((), SAMPLE_MESSAGES, [], (), withTemp, false).ensureType();
    test:assertEquals(anthropic["temperature"], 0.25d);
}

@test:Config {}
function testBuildInferenceParamsDoesNotInventATemperature() {
    InferenceParams none = buildInferenceParams((), (), (), (), (), (), ());
    test:assertEquals(none?.temperature, (), "an unset temperature must stay unset");
    test:assertEquals(none.maxTokens, DEFAULT_MAX_TOKEN_COUNT, "maxTokens still defaults");

    InferenceParams set = buildInferenceParams((), 0.9d, (), (), (), (), ());
    test:assertEquals(set?.temperature, 0.9d);
}

// ---- the passthrough must stay OPEN ----

@test:Config {}
function testUnknownFutureFieldRidesThePassthroughVerbatim() returns error? {
    // The scenario the open record exists for: AWS ships a parameter this module has
    // never heard of. It must reach the wire with no module update — which is why
    // `AdditionalRequestFields` is an OPEN record and not a closed one. A closed
    // record would make this literal a compile error.
    //
    // Note the QUOTED keys: `{randomshit: 5}` does not compile, because identifiers
    // cannot be used as rest-field keys.
    AdditionalRequestFields newFields = {
        "randomshit": 5,
        "nested_thing": {"deep": ["a", "b"]},
        "top_p": 0.9
    };
    InferenceParams params = buildInferenceParams(100, (), (), newFields, (), (), ());
    map<json> body = check encodeConverse((), SAMPLE_MESSAGES, [], (), params).ensureType();

    // Forwarded verbatim — not filtered, reshaped, or dropped.
    test:assertEquals(body["additionalModelRequestFields"], <json>{
        "randomshit": 5,
        "nested_thing": {"deep": ["a", "b"]},
        "top_p": 0.9
    });
}

@test:Config {}
function testVendorExtrasMergeIntoThePassthroughWithoutClobbering() returns error? {
    // A vendor knob folded in by a facade must not wipe out what the caller set.
    AdditionalRequestFields caller = {"randomshit": 5};
    AdditionalRequestFields? merged = foldRequestFields(caller, {"enable_thinking": true});
    if merged is () {
        test:assertFail("fold must return the merged object");
    }
    test:assertEquals(merged["randomshit"], <json>5, "caller field survived the fold");
    test:assertEquals(merged["enable_thinking"], <json>true, "vendor extra was added");
}

@test:Config {}
function testAnydataRestValueIsConvertedRatherThanDropped() returns error? {
    // `AdditionalRequestFields` is `record {}`, so its rest type is `anydata` — wider
    // than json. A value with no direct JSON form must be CONVERTED at the wire
    // boundary, never silently dropped and never a runtime failure: `toJson` is total.
    AdditionalRequestFields odd = {"weird": xml `<p>hi</p>`, "normal": 1};
    InferenceParams params = buildInferenceParams(100, (), (), odd, (), (), ());
    map<json> body = check encodeConverse((), SAMPLE_MESSAGES, [], (), params).ensureType();
    map<json> fwd = check body["additionalModelRequestFields"].ensureType();
    test:assertEquals(fwd["normal"], <json>1, "json-shaped values pass through untouched");
    test:assertEquals(fwd["weird"], <json>"<p>hi</p>", "xml degrades to its string form");
}

@test:Config {}
function testEmptyPassthroughIsOmittedEntirely() returns error? {
    // An empty object must not appear on the wire — some dialects reject it.
    InferenceParams params = buildInferenceParams(100, (), (), {}, (), (), ());
    map<json> body = check encodeConverse((), SAMPLE_MESSAGES, [], (), params).ensureType();
    test:assertFalse(body.hasKey("additionalModelRequestFields"));
}

@test:Config {}
function testConverseEffortFoldsIntoThePassthroughNotANativeMember() returns error? {
    // Live-verified 2026-08-11: a native `outputConfig` member 400s on Converse —
    // "This model doesn't support the effort field" — on every model tried,
    // including opus-4-7 (on Anthropic's adaptive-only list, so not a "wrong
    // model" 400). The identical value folded into
    // additionalModelRequestFields.output_config.effort is accepted on the same
    // model/route. See testLiveConverseEffortIsAccepted and
    // converter_converse.bal's `effort` comment.
    InferenceParams params = buildInferenceParams(100, (), (), (), (), (), (), (), EFFORT_LOW);
    map<json> body = check encodeConverse((), SAMPLE_MESSAGES, [], (), params).ensureType();
    test:assertFalse(body.hasKey("outputConfig"),
            "effort must not ride a native outputConfig member — Bedrock rejects it");
    map<json> extra = check body["additionalModelRequestFields"].ensureType();
    test:assertEquals(extra["output_config"], <json>{"effort": "low"});
}

@test:Config {}
function testConverseThinkingAndEffortFoldTogetherWithoutClobbering() returns error? {
    // `thinking` and `effort` are folded into the same passthrough object in two
    // separate steps (converter_converse.bal) — this locks in that the second fold
    // doesn't wipe out the first.
    InferenceParams params = buildInferenceParams(100, (), (), (), (), (), (),
            {mode: ADAPTIVE}, EFFORT_HIGH);
    map<json> body = check encodeConverse((), SAMPLE_MESSAGES, [], (), params).ensureType();
    map<json> extra = check body["additionalModelRequestFields"].ensureType();
    test:assertEquals(extra["thinking"], <json>{"type": "adaptive"});
    test:assertEquals(extra["output_config"], <json>{"effort": "high"});
}
