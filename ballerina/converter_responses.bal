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

// OpenAI Responses wire format on Mantle. GPT-5.5/5.4 are
// Mantle-only and use `/openai/v1/responses`. System text is the `instructions`
// field; turns are `input` items; the model reply is in `output` items.

// Encodes an OpenAI Responses request body.
isolated function encodeResponses(string? system, ResolvedMessage[] messages,
        ai:ChatCompletionFunctions[] tools, string? stop, InferenceParams params) returns json|ai:Error {
    // UNVERIFIED: image support on the Mantle /openai/v1/responses path is not stated
    // by any first-party source. Refuse rather than guess — see README.
    check rejectImagesIn(messages, "the OpenAI Responses dialect", true);
    json[] input = [];
    foreach ResolvedMessage m in messages {
        input.push(...responsesInputItems(m));
    }
    // The Responses dialect has NO stop-sequence parameter — it is absent from the
    // request schema entirely (unlike Chat Completions' `stop`), so there is nothing
    // to map onto. Accepting one silently would let the model run past the caller's
    // stop text: wrong output, and billed tokens they asked us not to spend.
    // https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_create_params.py
    string[]? configuredStops = params.stopSequences;
    if stop is string || (configuredStops is string[] && configuredStops.length() > 0) {
        return error ai:LlmInvalidGenerationError(
            "Stop sequences are not supported on the bedrock-mantle Responses route: the OpenAI " +
            "Responses API has no stop-sequence parameter. Remove 'stop'/'stopSequences', or use a " +
            "Converse/Invoke model.");
    }

    map<json> body = {"input": input, "max_output_tokens": params.maxTokens};
    setTemperature(body, params);
    if system is string {
        body["instructions"] = system; // system → instructions
    }
    if tools.length() > 0 {
        json[] toolDefs = [];
        foreach ai:ChatCompletionFunctions t in tools {
            toolDefs.push({"type": "function", "name": t.name, "description": t.description, "parameters": toolParameters(t)});
        }
        body["tools"] = toolDefs;
    }
    map<json>? extra = additionalFieldsToJson(params?.additionalModelRequestFields);
    if extra is map<json> {
        foreach [string, json] [k, v] in extra.entries() {
            body[k] = v;
        }
    }
    return body;
}

// Maps one resolved message to one or more Responses `input` items.
//
// An assistant turn that made tool calls MUST expand to a `function_call` item per
// call, carrying its `call_id`: the Responses API pairs every `function_call_output`
// to a preceding `function_call` by `call_id`. Dropping the call — as this once did,
// encoding it as an empty `output_text` — makes AWS reject the following tool result
// with 400 "No tool call found for function call output with call_id …", so the agent
// loop never completes (verified live 2026-08-03). One assistant message can carry
// several tool calls, which is why this returns json[].
isolated function responsesInputItems(ResolvedMessage m) returns json[] {
    if m is ResolvedUserMessage {
        return [{"role": "user", "content": responsesContentParts(m.parts)}];
    }
    if m is ai:ChatAssistantMessage {
        json[] items = [];
        string? content = m.content;
        if content is string && content != "" {
            items.push({"role": "assistant", "content": [{"type": "output_text", "text": content}]});
        }
        ai:FunctionCall[]? toolCalls = m.toolCalls;
        if toolCalls is ai:FunctionCall[] {
            foreach ai:FunctionCall fc in toolCalls {
                items.push({
                    "type": "function_call",
                    "call_id": fc.id ?: fc.name,
                    "name": fc.name,
                    "arguments": (fc.arguments ?: {}).toJsonString()
                });
            }
        }
        // A bare assistant turn (no text, no calls) still needs an item so it is not
        // silently dropped from the transcript.
        if items.length() == 0 {
            items.push({"role": "assistant", "content": [{"type": "output_text", "text": ""}]});
        }
        return items;
    }
    return [{"type": "function_call_output", "call_id": m.id ?: m.name, "output": m.content ?: ""}];
}

// Decodes an OpenAI Responses response. Always populates `usage` and
// `stopReason`.
isolated function decodeResponses(json response) returns DecodedResponse|ai:Error {
    map<json>|error rr = response.ensureType();
    if rr is error {
        return error ai:LlmInvalidResponseError("Responses payload was not a JSON object", rr);
    }
    map<json> r = rr;

    string text = "";
    ai:FunctionCall[] toolCalls = [];
    json[]? output = arrField(r, "output");
    if output is json[] {
        foreach json item in output {
            if item is map<json> {
                string? itemType = strField(item, "type");
                if itemType == "function_call" {
                    map<json> args = {};
                    string? argStr = strField(item, "arguments");
                    if argStr is string {
                        json|error parsed = argStr.fromJsonString();
                        if parsed is map<json> {
                            args = parsed;
                        }
                    }
                    toolCalls.push({name: strField(item, "name") ?: "", arguments: args, id: strField(item, "call_id")});
                } else if itemType == "message" {
                    // Two gates, both required. `output` also carries `reasoning`
                    // items, and a `message` item's `content` can hold `refusal`
                    // blocks — both have a `text` field, so an unfiltered append
                    // leaks the model's chain-of-thought (and refusal prose) into
                    // the assistant content returned to the caller.
                    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-55.html
                    json[]? content = arrField(item, "content");
                    if content is json[] {
                        foreach json block in content {
                            if block is map<json> && strField(block, "type") == "output_text" {
                                text += strField(block, "text") ?: "";
                            }
                        }
                    }
                }
            }
        }
    }

    int inputTokens = 0;
    int outputTokens = 0;
    map<json>? usage = mapField(r, "usage");
    if usage is map<json> {
        inputTokens = intField(usage, "input_tokens") ?: 0;
        outputTokens = intField(usage, "output_tokens") ?: 0;
    }
    ai:ChatAssistantMessage message = {role: ai:ASSISTANT, content: text == "" ? () : text};
    if toolCalls.length() > 0 {
        message.toolCalls = toolCalls;
    }
    return {
        message,
        usage: {inputTokens, outputTokens},
        stopReason: strField(r, "status") ?: "completed",
        responseId: strField(r, "id"),
        guardrailAction: ()
    };
}
