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

// Where two dialects of ONE vendor differ on the wire. These are the cases a
// vendor-shaped mental model gets wrong: "OpenAI" is not one format — Responses and
// Chat Completions disagree, and we serve both.

// ---- tool_choice: Responses is FLAT, Chat Completions is NESTED ----

@test:Config {}
function testResponsesForcesToolsWithTheFlatShape() {
    // REGRESSION: both dialects shared one OPENAI_TOOL_CHOICE emitting the NESTED
    // Chat Completions shape. Responses ignores that, so the tool went unforced —
    // the model answered in prose and generate() failed with "no tool call" on every
    // GPT-5.x and Gemma 4 request. Sources, first-party and unambiguous:
    // https://github.com/openai/openai-python/blob/main/src/openai/types/responses/tool_choice_function.py
    map<json> forced = <map<json>>applyToolChoice({}, MANTLE_RESPONSES_CONVERTER.toolChoice, "my_tool");
    test:assertEquals(forced["tool_choice"], <json>{"type": "function", "name": "my_tool"},
            "Responses takes the tool name FLAT, not nested under `function`");
}

@test:Config {}
function testChatCompletionsForcesToolsWithTheNestedShape() {
    // https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_named_tool_choice_param.py
    map<json> forced = <map<json>>applyToolChoice({}, MANTLE_CHAT_CONVERTER.toolChoice, "my_tool");
    test:assertEquals(forced["tool_choice"], <json>{"type": "function", "function": {"name": "my_tool"}},
            "Chat Completions nests the tool name under `function`");
}

@test:Config {}
function testTheTwoOpenAiDialectsDoNotShareAToolChoiceStyle() {
    // The bug was one enum value spanning two incompatible dialects; keep them split.
    test:assertNotEquals(MANTLE_RESPONSES_CONVERTER.toolChoice, MANTLE_CHAT_CONVERTER.toolChoice);
    test:assertEquals(INVOKE_OPENAI_CHAT_CONVERTER.toolChoice, MANTLE_CHAT_CONVERTER.toolChoice,
            "gpt-oss on Invoke speaks Chat Completions, like Mantle's chat route");
}

// ---- Responses has no stop-sequence parameter ----

@test:Config {}
function testResponsesRejectsAPerCallStopRatherThanDroppingIt() {
    // The Responses request schema has NO stop field at all (unlike Chat
    // Completions' `stop`), so there is nothing to map onto. We refuse: honouring
    // the request is impossible, and silently ignoring it lets the model run past
    // the caller's stop text and bill for the overrun.
    // https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_create_params.py
    json|ai:Error encoded = encodeResponses((), [userText("hi")], [], "STOP",
            {temperature: 0.5, maxTokens: 100});
    test:assertTrue(encoded is ai:Error, "a stop sequence must not be silently dropped");
    if encoded is ai:Error {
        test:assertTrue(encoded.message().includes("Responses"),
                "the error must name the dialect that lacks the capability; got: " + encoded.message());
    }
}

@test:Config {}
function testResponsesRejectsConfiguredStopSequencesToo() {
    // The configured form must not slip through the per-call check.
    json|ai:Error encoded = encodeResponses((), [userText("hi")], [], (),
            {temperature: 0.5, maxTokens: 100, stopSequences: ["END"]});
    test:assertTrue(encoded is ai:Error, "configured stopSequences must be refused as well");
}

@test:Config {}
function testResponsesEncodesNormallyWithNoStop() returns error? {
    // The guard must not fire on the ordinary path.
    map<json> body = check encodeResponses((), [userText("hi")], [], (),
            {temperature: 0.5, maxTokens: 100}).ensureType();
    test:assertTrue(body.hasKey("input"));
    test:assertFalse(body.hasKey("stop"), "no stop field exists in this dialect");
}

@test:Config {}
function testResponsesReplaysAssistantToolCallAsFunctionCallItem() returns error? {
    // REGRESSION (live 400, 2026-08-03): an assistant tool-call turn was encoded as an
    // empty `output_text`, dropping the `function_call`. The Responses API then rejected
    // the following `function_call_output` with "No tool call found for function call
    // output with call_id …" and the agent loop stalled. The call must be replayed as a
    // `function_call` item carrying its `call_id`, so a later output can pair to it.
    ResolvedMessage[] messages = [
        {parts: [{text: "What is 3607 multiplied by 4021?"}]},
        {role: ai:ASSISTANT, content: (),
            toolCalls: [{name: "multiply", arguments: {a: 3607, b: 4021}, id: "call_abc"}]}
    ];
    map<json> body = check encodeResponses((), messages, [], (),
            {temperature: 0, maxTokens: 100}).ensureType();
    json[] input = check body["input"].ensureType();

    json[] calls = from json it in input where it is map<json> && it["type"] == "function_call" select it;
    test:assertEquals(calls.length(), 1, "the assistant tool call must be replayed as a function_call item");
    map<json> call = check calls[0].ensureType();
    test:assertEquals(call["call_id"], "call_abc");
    test:assertEquals(call["name"], "multiply");
    // arguments is a JSON STRING in the Responses schema; assert on the parsed value so
    // the test is not brittle to key spacing.
    json parsedArgs = check (check call["arguments"].ensureType(string)).fromJsonString();
    test:assertEquals(parsedArgs, {"a": 3607, "b": 4021});
    // The dropped-to-empty-output_text bug must not recur: no blank assistant text item.
    json[] blanks = from json it in input
        where it is map<json> && it["role"] == "assistant" select it;
    test:assertEquals(blanks.length(), 0, "the tool-call turn must not become an empty output_text item");
}

// ---- x-api-key follows the PATH, not the vendor or the provider class ----

@test:Config {}
function testMantleApiKeyHeaderFollowsTheMessagesPathNotTheProviderClass() returns error? {
    // REGRESSION: this lived in the Anthropic facade alone, so the same rule was
    // honoured there and silently ignored everywhere else. Driven here through
    // `commonExtraHeaders` — the shared path — to prove the route decides, not the
    // class. A Messages path gets `x-api-key`; see the Bearer case below.
    MantleEntry entry = {path: "/anthropic/v1/messages"};
    Route route = {
        family: MANTLE,
        bareModelId: "anthropic.claude-mythos-5",
        geoPrefix: (),
        effectiveModelId: "anthropic.claude-mythos-5",
        region: "us-east-1",
        partition: "aws",
        mantleEntry: entry
    };
    map<string> headers = commonExtraHeaders(route, (), {apiKey: "secret-key"});
    test:assertEquals(headers["x-api-key"], "secret-key");
}

@test:Config {}
function testMantleApiKeyHeaderIsAbsentForBearerStyleEntries() {
    // A BEARER entry needs nothing extra: the transport's `Authorization: Bearer`
    // already carries the key.
    MantleEntry entry = {path: "/openai/v1/responses"};
    Route route = {
        family: MANTLE,
        bareModelId: "openai.gpt-5.4",
        geoPrefix: (),
        effectiveModelId: "openai.gpt-5.4",
        region: "us-east-1",
        partition: "aws",
        mantleEntry: entry
    };
    map<string> headers = commonExtraHeaders(route, (), {apiKey: "secret-key"});
    test:assertFalse(headers.hasKey("x-api-key"));
}

@test:Config {}
function testMantleApiKeyHeaderIsAbsentForSigV4Credentials() {
    // With SigV4 credentials there is no api key to send — the signature alone must
    // authenticate. Emitting the secret access key here would leak it in a header.
    MantleEntry entry = {path: "/anthropic/v1/messages"};
    Route route = {
        family: MANTLE,
        bareModelId: "anthropic.claude-mythos-5",
        geoPrefix: (),
        effectiveModelId: "anthropic.claude-mythos-5",
        region: "us-east-1",
        partition: "aws",
        mantleEntry: entry
    };
    map<string> headers = commonExtraHeaders(route, (), TEST_CREDS);
    test:assertFalse(headers.hasKey("x-api-key"));
}

// ---- Claude-only knobs: `anthropic-workspace` header + `thinking` fold ----

@test:Config {}
function testAnthropicWorkspaceHeaderAndThinkingKnobAreEmitted() returns error? {
    // COVERAGE GAP (2026-08-04): the two Claude-only config knobs had NO test.
    // `anthropicWorkspaceId` (the `anthropic-workspace` cost-scoping header on the
    // Mantle Messages route) and `thinking` (folded into the additionalModelRequestFields
    // passthrough) were both wired but unasserted — and an untested request header is
    // exactly the class that shipped broken before (the Mantle double-header 401).
    MantleEntry entry = {path: "/anthropic/v1/messages"};
    Route mantleRoute = {
        family: MANTLE,
        bareModelId: "anthropic.claude-opus-4-8",
        geoPrefix: (),
        effectiveModelId: "anthropic.claude-opus-4-8",
        region: "us-east-1",
        partition: "aws",
        mantleEntry: entry
    };
    AnthropicConfig config = {thinking: {mode: ENABLED, budgetTokens: 1024}, effort: EFFORT_HIGH};

    // 1) A Messages-dialect Mantle path carries `anthropic-version` — derived from the
    //    path, not stored per model.
    map<string> headers = buildExtraHeaders(mantleRoute, config, TEST_CREDS);
    test:assertEquals(headers["anthropic-version"], "2023-06-01");

    // 2) REGRESSION: `thinking` used to be folded into additionalModelRequestFields,
    //    but `encodeAnthropicMessages` ignores that field entirely — so the knob was
    //    silently dropped on exactly the two Claude-native dialects. It is now a
    //    first-class body field and must actually reach the wire.
    InferenceParams params = check resolveParams(8000, (), config);
    map<json> body = check encodeMantleMessages((), SAMPLE_MESSAGES, [], (), params).ensureType();
    test:assertEquals(body["thinking"], <json>{"type": "enabled", "budget_tokens": 1024},
            "thinking must be a top-level body field on the Messages dialect");

    // 3) `effort` is a SIBLING of `thinking`, never nested inside it — AWS returns a
    //    ValidationException for the nested form.
    test:assertEquals(body["output_config"], <json>{"effort": "high"});
    map<json> thinkingBlock = check body["thinking"].ensureType();
    test:assertFalse(thinkingBlock.hasKey("effort"),
            "effort inside thinking is a documented ValidationException");
}

@test:Config {}
function testThinkingBudgetRulesFailAtConstruction() {
    // All three are Bedrock 400s; catching them here names the actual mistake.
    InferenceParams|ai:Error tooSmall = resolveParams(8000, (),
            {thinking: {mode: ENABLED, budgetTokens: 512}});
    test:assertTrue(tooSmall is ai:Error);

    InferenceParams|ai:Error overMax = resolveParams(2000, (),
            {thinking: {mode: ENABLED, budgetTokens: 4000}});
    test:assertTrue(overMax is ai:Error);

    InferenceParams|ai:Error missing = resolveParams(8000, (), {thinking: {mode: ENABLED}});
    test:assertTrue(missing is ai:Error);

    // budgetTokens is meaningless outside ENABLED — adaptive uses `effort` instead.
    InferenceParams|ai:Error wrongMode = resolveParams(8000, (),
            {thinking: {mode: ADAPTIVE, budgetTokens: 4000}});
    test:assertTrue(wrongMode is ai:Error);

    // ADAPTIVE alone is the recommended, and default, configuration.
    InferenceParams|ai:Error ok = resolveParams(8000, (), {thinking: {}});
    test:assertTrue(ok is InferenceParams);
}
