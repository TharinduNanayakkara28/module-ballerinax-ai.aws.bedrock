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

// Mistral on the InvokeModel route. Mistral ships TWO mutually
// incompatible Invoke dialects, and the model id is the only discriminator:
//
//   text completion  — `prompt` (a `<s>[INST]…[/INST]` template) → `outputs[].text`
//                      Mistral 7B Instruct, Mixtral 8X7B, Mistral Large 24.02. No tools.
//                      https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-text-completion.html
//   chat completion  — `messages`/`tools` → `choices[].message`
//                      Mistral Large 24.07.
//                      https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-chat-completion.html
//                      https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-large-2407.html
//
// The chat dialect resembles OpenAI's but is NOT interchangeable with it: the stop
// reason is `stop_reason` (not `finish_reason`), tools are forced with the bare
// string `"any"` (not an object naming the tool), and AWS documents no `usage`
// block at all. Routing Mistral through the OpenAI converter leaves `stopReason` empty
// and breaks tool forcing, so both dialects get their own converter here.

// ============================================================================
// Chat completion — Mistral Large 24.07.
// ============================================================================

// Encodes a Mistral chat-completion request body. This dialect DOES carry
// `system` as a `role: system` message — the AWS page lists `"system"` among the
// valid roles — so the hoisted system is re-added as the leading message.
isolated function encodeMistralChat(string? system, ResolvedMessage[] messages,
        ai:ChatCompletionFunctions[] tools, string? stop, InferenceParams params) returns json|ai:Error {
    // UNVERIFIED and CONTESTED: AWS documents this dialect's `content` as a string,
    // while Mistral's own API documents image_url chunks. Two first-party sources
    // disagree, so per the module's ground rules this refuses rather than picking one.
    check rejectImagesIn(messages, "the Mistral chat-completion dialect", true);
    json[] wire = [];
    if system is string {
        wire.push({"role": "system", "content": system});
    }
    foreach ResolvedMessage m in messages {
        wire.push(mistralChatMessage(m));
    }

    map<json> body = {
        "messages": wire,
        "max_tokens": params.maxTokens
    };
    setTemperature(body, params);
    // AWS's own page is internally inconsistent here: `stop` is absent from the
    // chat-completion parameter list, yet the `stop_reason` description refers to
    // "the stop sequences that you define in the stop request parameter". We emit
    // it only when the caller asked for one, so the documented parameter set is
    // sent verbatim by default.
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
    map<json>? extra = additionalFieldsToJson(params?.additionalModelRequestFields);
    if extra is map<json> {
        foreach [string, json] [k, v] in extra.entries() {
            body[k] = v;
        }
    }
    return body;
}

// Maps one resolved message to a Mistral chat-completion message.
isolated function mistralChatMessage(ResolvedMessage m) returns json {
    if m is ResolvedUserMessage {
        return {"role": "user", "content": openAIContentParts(m.parts)};
    }
    if m is ai:ChatAssistantMessage {
        map<json> msg = {"role": "assistant", "content": m.content ?: ""};
        ai:FunctionCall[]? toolCalls = m.toolCalls;
        if toolCalls is ai:FunctionCall[] && toolCalls.length() > 0 {
            json[] calls = [];
            foreach ai:FunctionCall fc in toolCalls {
                // Mistral's assistant tool_call carries only `id` + `function`;
                // there is no `type` field on this dialect.
                calls.push({
                    "id": fc.id ?: fc.name,
                    "function": {"name": fc.name, "arguments": (fc.arguments ?: {}).toJsonString()}
                });
            }
            msg["tool_calls"] = calls;
        }
        return msg;
    }
    return {"role": "tool", "tool_call_id": m.id ?: m.name, "content": m.content ?: ""};
}

// Decodes a Mistral chat-completion response. The stop reason is
// `stop_reason` — NOT OpenAI's `finish_reason`.
isolated function decodeMistralChat(json response) returns DecodedResponse|ai:Error {
    map<json>|error rr = response.ensureType();
    if rr is error {
        return error ai:LlmInvalidResponseError("Mistral chat response was not a JSON object", rr);
    }
    map<json> r = rr;
    json[]? choices = arrField(r, "choices");
    if choices is () || choices.length() == 0 {
        return error ai:LlmInvalidResponseError("Mistral chat response had no choices");
    }
    map<json>|error choiceResult = choices[0].ensureType();
    if choiceResult is error {
        return error ai:LlmInvalidResponseError("Mistral chat choice was not an object", choiceResult);
    }
    map<json> choice = choiceResult;
    // `stop_reason` per the AWS page; fall back to `finish_reason` because the
    // live API also emits the OpenAI spelling on some ids. stopReason is a module
    // invariant — it is never left empty.
    string stopReason = strField(choice, "stop_reason") ?: strField(choice, "finish_reason") ?: "stop";

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

    // AWS documents NO usage block for this dialect. Read the Mistral-native
    // spelling when present and fall back to zeroes so `usage` is always populated.
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

// ============================================================================
// Text completion — Mistral 7B Instruct, Mixtral 8X7B, Mistral Large 24.02.
// ============================================================================

// Encodes a Mistral text-completion request body. There is no `messages`
// array on this dialect: the conversation must be flattened into one `prompt`
// string using Mistral's instruction template.
isolated function encodeMistralText(string? system, ResolvedMessage[] messages,
        ai:ChatCompletionFunctions[] tools, string? stop, InferenceParams params) returns json|ai:Error {
    // Text-only by construction: a single prompt string, no content-part array.
    check rejectImagesIn(messages, "the Mistral text-completion dialect");
    if tools.length() > 0 {
        // Fail loudly rather than drop the tools: this dialect has no tool support
        // at all, so a silent no-op would look like the model ignoring the tool.
        return error ai:Error(
            "The Mistral text-completion dialect does not support tools. Use the Converse route " +
            "(the module default) or a chat-completion model such as 'mistral.mistral-large-2407-v1:0'.");
    }

    map<json> body = {
        "prompt": mistralInstructPrompt(system, messages),
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
    // `top_k` is a text-completion-only parameter; it rides the passthrough.
    map<json>? extra = additionalFieldsToJson(params?.additionalModelRequestFields);
    if extra is map<json> {
        foreach [string, json] [k, v] in extra.entries() {
            body[k] = v;
        }
    }
    return body;
}

// Renders messages into Mistral's instruction template, per the AWS page:
//
//   <s>[INST] What is your favourite condiment? [/INST]
//   Well, I'm quite partial to a good squeeze of fresh lemon juice.</s>
//   [INST] Do you have mayonnaise recipes? [/INST]
//
// User text sits inside `[INST]…[/INST]`; assistant text sits outside and is
// closed with `</s>`.
//
// NOTE: AWS does not document how `system` maps onto this dialect — the template
// has no system slot. We prepend it to the first instruction block, which is what
// Mistral's own chat template does, and it preserves the module invariant that
// system is never emitted as a `role: system` message.
isolated function mistralInstructPrompt(string? system, ResolvedMessage[] messages) returns string {
    // Images are rejected by the caller (`encodeMistralText`): this dialect is a single
    // prompt string with no content-part array to put one in.
    string prompt = "<s>";
    boolean systemPending = system is string;
    string systemText = system ?: "";

    foreach ResolvedMessage m in messages {
        if m is ai:ChatAssistantMessage {
            string? content = m.content;
            prompt += string ` ${content ?: ""}</s>`;
            continue;
        }
        // User (and any tool result, which has nowhere else to go on this dialect)
        // becomes an instruction block.
        string text = m is ai:ChatFunctionMessage ? (m.content ?: "") : partsText(m.parts);
        if systemPending {
            text = systemText + "\n\n" + text;
            systemPending = false;
        }
        prompt += string `[INST] ${text} [/INST]`;
    }
    if systemPending {
        // System with no user turn to attach to.
        prompt += string `[INST] ${systemText} [/INST]`;
    }
    return prompt;
}

// Decodes a Mistral text-completion response: `outputs[].text` +
// `outputs[].stop_reason`. This dialect returns no token counts and no id.
isolated function decodeMistralText(json response) returns DecodedResponse|ai:Error {
    map<json>|error rr = response.ensureType();
    if rr is error {
        return error ai:LlmInvalidResponseError("Mistral text response was not a JSON object", rr);
    }
    map<json> r = rr;
    json[]? outputs = arrField(r, "outputs");
    if outputs is () || outputs.length() == 0 {
        return error ai:LlmInvalidResponseError("Mistral text response had no outputs");
    }
    map<json>|error outputResult = outputs[0].ensureType();
    if outputResult is error {
        return error ai:LlmInvalidResponseError("Mistral text output was not an object", outputResult);
    }
    map<json> output = outputResult;
    string text = strField(output, "text") ?: "";
    return {
        message: {role: ai:ASSISTANT, content: text == "" ? () : text},
        // No usage on this dialect; `usage` stays populated with zeroes.
        usage: {inputTokens: 0, outputTokens: 0},
        stopReason: strField(output, "stop_reason") ?: "stop",
        // No response id in the body; the transport fills it from the request-id
        // header.
        responseId: (),
        guardrailAction: invokeGuardrailAction(r) // body field
    };
}
