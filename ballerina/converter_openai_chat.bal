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

// OpenAI Chat-Completions wire format. Shared by the Mantle
// Chat Completions route (GLM) and the Invoke route for the OpenAI-shaped vendors
// (GPT-OSS, Qwen, DeepSeek, Mistral chat). Unlike Converse/Anthropic, THIS format
// carries `system` as a `role: system` MESSAGE — that is the wire contract here,
// so the hoisted system is re-added as the leading message.

// Encodes an OpenAI Chat-Completions request body.
isolated function encodeOpenAIChat(string? system, ResolvedMessage[] messages,
        ai:ChatCompletionFunctions[] tools, string? stop, InferenceParams params) returns json|ai:Error {
    // UNVERIFIED: no first-party source states whether Bedrock's OpenAI-compatible
    // dialects accept image content parts. Refuse rather than guess — see README.
    check rejectImagesIn(messages, "the OpenAI chat-completions dialect", true);
    json[] wire = [];
    if system is string {
        // OpenAI's wire format DOES use a role:system message (not a top-level field).
        wire.push({"role": "system", "content": system});
    }
    foreach ResolvedMessage m in messages {
        wire.push(openAIMessage(m));
    }

    map<json> body = {
        "messages": wire,
        "max_tokens": params.maxTokens
    };
    setTemperature(body, params);
    string[]? stops = params.stopSequences;
    if stop is string {
        stops = [stop]; // per-call stop overrides configured stopSequences
    }
    if stops is string[] && stops.length() > 0 {
        body["stop"] = stops;
    }
    if tools.length() > 0 {
        json[] toolDefs = [];
        foreach ai:ChatCompletionFunctions t in tools {
            toolDefs.push({
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": toolParameters(t)}
            });
        }
        body["tools"] = toolDefs;
    }
    // Vendor passthrough (e.g. Qwen `enable_thinking`) rides additionalModelRequestFields.
    map<json>? extra = additionalFieldsToJson(params?.additionalModelRequestFields);
    if extra is map<json> {
        foreach [string, json] [k, v] in extra.entries() {
            body[k] = v;
        }
    }
    return body;
}

// Maps one resolved message to an OpenAI Chat-Completions message.
isolated function openAIMessage(ResolvedMessage m) returns json {
    if m is ResolvedUserMessage {
        return {"role": "user", "content": openAIContentParts(m.parts)};
    }
    if m is ai:ChatAssistantMessage {
        map<json> msg = {"role": "assistant", "content": m.content};
        ai:FunctionCall[]? toolCalls = m.toolCalls;
        if toolCalls is ai:FunctionCall[] && toolCalls.length() > 0 {
            json[] calls = [];
            foreach ai:FunctionCall fc in toolCalls {
                calls.push({
                    "id": fc.id ?: fc.name,
                    "type": "function",
                    "function": {"name": fc.name, "arguments": (fc.arguments ?: {}).toJsonString()}
                });
            }
            msg["tool_calls"] = calls;
        }
        return msg;
    }
    // ai:FUNCTION result → OpenAI role:tool message.
    return {"role": "tool", "tool_call_id": m.id ?: m.name, "content": m.content ?: ""};
}

// Decodes an OpenAI Chat-Completions response. Always populates
// `usage` and `stopReason`.
isolated function decodeOpenAIChat(json response) returns DecodedResponse|ai:Error {
    map<json>|error rr = response.ensureType();
    if rr is error {
        return error ai:LlmInvalidResponseError("Chat-Completions response was not a JSON object", rr);
    }
    map<json> r = rr;
    json[]? choices = arrField(r, "choices");
    if choices is () || choices.length() == 0 {
        return error ai:LlmInvalidResponseError("Chat-Completions response had no choices");
    }
    map<json>|error choiceResult = choices[0].ensureType();
    if choiceResult is error {
        return error ai:LlmInvalidResponseError("Chat-Completions choice was not an object", choiceResult);
    }
    map<json> choice = choiceResult;
    string stopReason = strField(choice, "finish_reason") ?: "stop";

    string text = "";
    ai:FunctionCall[] toolCalls = [];
    map<json>? message = mapField(choice, "message");
    if message is map<json> {
        text = strField(message, "content") ?: "";
        json[]? calls = arrField(message, "tool_calls");
        if calls is json[] {
            foreach json call in calls {
                if call is map<json> {
                    map<json>? fn = mapField(call, "function");
                    if fn is map<json> {
                        map<json> args = {};
                        string? argStr = strField(fn, "arguments");
                        if argStr is string {
                            json|error parsed = argStr.fromJsonString();
                            if parsed is map<json> {
                                args = parsed;
                            }
                        }
                        toolCalls.push({name: strField(fn, "name") ?: "", arguments: args, id: strField(call, "id")});
                    }
                }
            }
        }
    }

    int inputTokens = 0;
    int outputTokens = 0;
    map<json>? usage = mapField(r, "usage");
    if usage is map<json> {
        inputTokens = intField(usage, "prompt_tokens") ?: 0;
        outputTokens = intField(usage, "completion_tokens") ?: 0;
    }
    ai:ChatAssistantMessage assistant = {role: ai:ASSISTANT, content: text == "" ? () : text};
    if toolCalls.length() > 0 {
        assistant.toolCalls = toolCalls;
    }
    return {
        message: assistant,
        usage: {inputTokens, outputTokens},
        stopReason,
        responseId: strField(r, "id"),
        guardrailAction: invokeGuardrailAction(r) // body field
    };
}
