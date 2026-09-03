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

// generate() tests: the tool-forcing path emits the schema as a
// forced tool and parses the tool-call arguments back into the record; the routes
// with no structured-output path refuse cleanly.
//
// The two halves are tested either side of the transport, which is covered
// separately (`transport_path_test.bal`) and cannot be intercepted here because it
// hardcodes https. The guard paths ARE driven through `structuredGenerate` itself,
// since they must return before any I/O — constructing a `BedrockTransport` opens
// no connection.

type Review record {|
    string sentiment;
    int score;
|};

final map<json> REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"sentiment": {"type": "string"}, "score": {"type": "integer"}},
    "required": ["sentiment", "score"]
};

final ai:ChatCompletionFunctions RESULT_TOOL_DEF = {
    name: RESULT_TOOL,
    description: "Return the result strictly as structured arguments in the required schema.",
    parameters: REVIEW_SCHEMA
};

final readonly & InferenceParams GEN_PARAMS = {temperature: 0.5, maxTokens: 256};

// ---- The schema goes out as a forced tool ----

@test:Config {}
function testToolForcingEmitsSchemaAsForcedToolOnConverse() returns error? {
    json encoded = check encodeConverse((), [userText("Rate this")], [RESULT_TOOL_DEF], (),
            GEN_PARAMS);
    map<json> body = check applyToolChoice(encoded, CONVERSE_CONVERTER.toolChoice, RESULT_TOOL).ensureType();

    map<json> toolConfig = check body["toolConfig"].ensureType();
    // The expected type's schema must reach the wire as the tool's input schema.
    json[] tools = check toolConfig["tools"].ensureType();
    test:assertEquals(tools.length(), 1, "generate() must send exactly one tool");
    map<json> toolSpec = check tools[0].toolSpec.ensureType();
    test:assertEquals(toolSpec["name"], RESULT_TOOL);
    test:assertEquals(toolSpec["inputSchema"], <json>{"json": REVIEW_SCHEMA},
            "the typedesc's JSON schema must be the tool input schema");
    // ...and the tool must be FORCED, not merely offered.
    test:assertEquals(toolConfig["toolChoice"], <json>{"tool": {"name": RESULT_TOOL}});
}

@test:Config {}
function testToolForcingEmitsSchemaAsForcedToolOnInvokeAnthropic() returns error? {
    json encoded = check encodeInvokeAnthropic((), [userText("Rate this")], [RESULT_TOOL_DEF], (),
            GEN_PARAMS);
    map<json> body = check applyToolChoice(encoded, INVOKE_ANTHROPIC_CONVERTER.toolChoice, RESULT_TOOL).ensureType();

    json[] tools = check body["tools"].ensureType();
    test:assertEquals(tools.length(), 1);
    map<json> tool = check tools[0].ensureType();
    test:assertEquals(tool["name"], RESULT_TOOL);
    test:assertEquals(tool["input_schema"], <json>REVIEW_SCHEMA);
    test:assertEquals(body["tool_choice"], <json>{"type": "tool", "name": RESULT_TOOL});
}

// ---- Tool forcing is per-DIALECT, not per-route-family ----
//
// Regression: keying tool_choice off `ApiFamily` sent Anthropic's `tool_choice` to
// every non-Converse dialect. Those dialects ignore the unknown field, so the model
// answered in prose and generate() failed with "no tool call" — a silent 1-line bug
// that only shows up against live AWS.

@test:Config {}
function testNovaOnInvokeForcesToolTheConverseWay() returns error? {
    // Nova's InvokeModel body is Converse-shaped even though the family is INVOKE.
    json encoded = check encodeNovaInvoke((), [userText("Rate this")], [RESULT_TOOL_DEF], (),
            GEN_PARAMS);
    map<json> body = check applyToolChoice(encoded, INVOKE_NOVA_CONVERTER.toolChoice, RESULT_TOOL).ensureType();
    map<json> toolConfig = check body["toolConfig"].ensureType();
    test:assertEquals(toolConfig["toolChoice"], <json>{"tool": {"name": RESULT_TOOL}},
            "Nova is Converse-shaped: it must NOT get Anthropic's tool_choice");
    test:assertFalse(body.hasKey("tool_choice"), "Anthropic's tool_choice must not leak onto Nova");
}

@test:Config {}
function testMistralChatForcesToolWithBareAnyString() returns error? {
    // Mistral cannot name the forced tool: tool_choice is the bare string "any".
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-chat-completion.html
    json encoded = check encodeMistralChat((), [userText("Rate this")], [RESULT_TOOL_DEF], (),
            GEN_PARAMS);
    map<json> body = check applyToolChoice(encoded, INVOKE_MISTRAL_CHAT_CONVERTER.toolChoice, RESULT_TOOL).ensureType();
    test:assertEquals(body["tool_choice"], <json>"any", "Mistral's tool_choice is a bare string, not an object");
    json[] tools = check body["tools"].ensureType();
    map<json> fn = check tools[0].'function.ensureType();
    test:assertEquals(fn["parameters"], <json>REVIEW_SCHEMA);
}

@test:Config {}
function testOpenAIChatForcesToolWithFunctionObject() returns error? {
    json encoded = check encodeOpenAIChat((), [userText("Rate this")], [RESULT_TOOL_DEF], (),
            GEN_PARAMS);
    map<json> body = check applyToolChoice(encoded, INVOKE_OPENAI_CHAT_CONVERTER.toolChoice, RESULT_TOOL).ensureType();
    test:assertEquals(body["tool_choice"], <json>{"type": "function", "function": {"name": RESULT_TOOL}});
}

// ---- The tool-call arguments come back as the record ----

@test:Config {}
function testToolForcingParsesToolCallArgsBackIntoRecordOnConverse() returns error? {
    json canned = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{
                    "toolUse": {
                        "toolUseId": "tu_1",
                        "name": RESULT_TOOL,
                        "input": {"sentiment": "positive", "score": 9}
                    }
                }]
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 12, "outputTokens": 7}
    };
    DecodedResponse decoded = check decodeConverse(canned);
    ai:FunctionCall[] toolCalls = check decoded.message.toolCalls.ensureType();
    Review review = check bindJson(toolCalls[0].arguments ?: {}, Review).ensureType();
    test:assertEquals(review, {sentiment: "positive", score: 9});
}

@test:Config {}
function testToolForcingParsesToolCallArgsBackIntoRecordOnMistralChat() returns error? {
    // Mistral returns the arguments as a JSON *string*, not an object.
    json canned = {
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": RESULT_TOOL, "arguments": "{\"sentiment\": \"negative\", \"score\": 2}"}
                }]
            },
            "stop_reason": "tool_calls"
        }]
    };
    DecodedResponse decoded = check decodeMistralChat(canned);
    ai:FunctionCall[] toolCalls = check decoded.message.toolCalls.ensureType();
    Review review = check bindJson(toolCalls[0].arguments ?: {}, Review).ensureType();
    test:assertEquals(review, {sentiment: "negative", score: 2});
}

@test:Config {}
function testBindJsonRejectsAMismatchedShape() {
    anydata|ai:Error bound = bindJson({"sentiment": "positive"}, Review); // `score` missing
    test:assertTrue(bound is ai:LlmInvalidGenerationError,
            "a response that does not fit the expected type must be a clean ai:Error");
}

@test:Config {}
function testExtractJsonRecoversFencedJsonFromText() returns error? {
    // Fallback for models that answer with the JSON in prose instead of a tool call.
    json parsed = check extractJson("Sure!\n```json\n{\"sentiment\": \"ok\", \"score\": 5}\n```");
    Review review = check bindJson(parsed, Review).ensureType();
    test:assertEquals(review, {sentiment: "ok", score: 5});
}

@test:Config {}
function testExtractJsonSignalsAbsenceWithAnError() {
    test:assertTrue(extractJson("no json here at all") is error);
}

// ---- Routes with no structured-output path refuse cleanly, before any I/O ----

function mantleTransport() returns BedrockTransport|error =>
    new (TEST_CREDS, "us-east-1",
        {baseUrl: string `https://bedrock-mantle.us-east-1.api.aws`,
            host: "bedrock-mantle.us-east-1.api.aws", path: "/openai/v1/responses",
            signingService: SIGNING_BEDROCK_MANTLE});

@test:Config {}
function testMantleRefusesStructuredOutputNamingTheModel() returns error? {
    BedrockTransport transport = check mantleTransport();
    anydata|ai:Error result = structuredGenerate(false, MANTLE, MANTLE_RESPONSES_CONVERTER, transport,
            "openai.gpt-5.4", {}, GEN_PARAMS, `Rate this`, Review);
    test:assertTrue(result is ai:Error, "a typed target on Mantle must be a clean error");
    if result is ai:Error {
        string message = result.message();
        test:assertTrue(message.includes("openai.gpt-5.4"), "the error must name the model; got: " + message);
        test:assertTrue(message.includes("bedrock-mantle"),
                "the error must name the route that lacks the capability; got: " + message);
    }
}

@test:Config {}
function testMistralTextDialectRefusesStructuredOutput() returns error? {
    // supportsStructuredOutput is true here (INVOKE, not Mantle) — the refusal must
    // come from the CONVERTER having no tool-calling at all.
    BedrockTransport transport = check mantleTransport();
    anydata|ai:Error result = structuredGenerate(true, INVOKE, INVOKE_MISTRAL_TEXT_CONVERTER, transport,
            "mistral.mistral-7b-instruct-v0:2", {}, GEN_PARAMS, `Rate this`, Review);
    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        string message = result.message();
        test:assertTrue(message.includes("mistral.mistral-7b-instruct-v0:2"),
                "the error must name the model; got: " + message);
        test:assertTrue(message.includes("Converse"), "the error must point at the way out; got: " + message);
    }
}

@test:Config {}
function testMistralTextDialectRejectsToolsRatherThanDroppingThem() {
    json|ai:Error encoded = encodeMistralText((), [userText("Hi")], [RESULT_TOOL_DEF], (),
            GEN_PARAMS);
    test:assertTrue(encoded is ai:Error, "tools on a dialect with no tool support must fail loudly");
}

// ---------------------------------------------------------------------------
// Object envelope for non-object target types
// ---------------------------------------------------------------------------

@test:Config {}
function testObjectSchemaIsSentUnwrapped() {
    // A record target already has an object root; wrapping it would change the
    // arguments the model returns for no reason.
    [map<json>, boolean] [schema, wasObject] = objectEnvelope(REVIEW_SCHEMA);
    test:assertTrue(wasObject);
    test:assertEquals(schema, REVIEW_SCHEMA);
    test:assertEquals(unwrapResult({"sentiment": "positive", "score": 4}, wasObject),
            {"sentiment": "positive", "score": 4});
}

@test:Config {}
function testScalarSchemaIsWrappedInAnObjectRoot() {
    // REGRESSION, found live 2026-08-25: `generate()` with a `string` target sent
    // `{"type": "string"}` as the tool input schema and Bedrock rejected it —
    // "inputSchema.json.type must be one of the following: object" on Converse,
    // "input_schema.type: Input should be 'object'" on Anthropic-on-Invoke.
    [map<json>, boolean] [schema, wasObject] = objectEnvelope({"type": "string"});
    test:assertFalse(wasObject);
    test:assertEquals(schema, {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"]
    });
    test:assertEquals(unwrapResult({"result": "a joke"}, wasObject), "a joke");
}

@test:Config {}
function testArraySchemaIsWrappedInAnObjectRoot() {
    [map<json>, boolean] [schema, wasObject] = objectEnvelope({"type": "array", "items": {"type": "integer"}});
    test:assertFalse(wasObject);
    test:assertEquals(schema, {
        "type": "object",
        "properties": {"result": {"type": "array", "items": {"type": "integer"}}},
        "required": ["result"]
    });
    test:assertEquals(unwrapResult({"result": [1, 2, 3]}, wasObject), [1, 2, 3]);
}

@test:Config {}
function testSchemaMetadataStaysAtTheEnvelopeRoot() {
    // `title`/`description` describe the whole tool input, so they belong at the
    // root; the value keywords move down onto the wrapped property.
    [map<json>, boolean] [schema, _] = objectEnvelope({
        "title": "Score",
        "description": "the score",
        "type": "integer",
        "minimum": 0
    });
    test:assertEquals(schema, {
        "title": "Score",
        "description": "the score",
        "type": "object",
        "properties": {"result": {"type": "integer", "minimum": 0}},
        "required": ["result"]
    });
}

@test:Config {}
function testUnwrapLeavesABareValueAloneOnTheTextFallback() {
    // A model that answered in prose rather than with a tool call may have written
    // the bare value. Lifting a missing `result` key would turn that into a null.
    test:assertEquals(unwrapResult("a joke", false), "a joke");
    test:assertEquals(unwrapResult({"sentiment": "positive"}, false), {"sentiment": "positive"});
}
