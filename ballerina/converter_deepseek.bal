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

// DeepSeek-R1 on the InvokeModel route.
//
// R1's Invoke dialect is TEXT COMPLETION, not chat — despite the `choices`
// wrapper making it look OpenAI-shaped at a glance:
//
//   request:  {"prompt": string, "temperature": float, "top_p": float,
//              "max_tokens": int, "stop": [string]}
//   response: {"choices": [{"text": string, "stop_reason": "stop"|"length"}]}
//
// Note `choices[].text`, NOT `choices[].message.content`; `stop_reason`, NOT
// `finish_reason`; and NO `usage` object at all. Routing DeepSeek through the
// OpenAI chat converter sends `messages` (a 400 on encode) and, if it somehow got a
// response, would read every field from the wrong place.
//
// This converter serves R1 ONLY (`usesDeepSeekTextDialect` in converters.bal). DeepSeek
// V3.1/V3.2 take `{"messages": [...]}` on InvokeModel and go through the OpenAI
// chat converter instead — sending `prompt` to V3.2 returns `ValidationException ...
// missing field messages`.
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html

// Encodes a DeepSeek text-completion request body.
isolated function encodeDeepSeekInvoke(string? system, ResolvedMessage[] messages,
        ai:ChatCompletionFunctions[] tools, string? stop, InferenceParams params) returns json|ai:Error {
    // Text-only by construction: the whole conversation is one prompt string.
    check rejectImagesIn(messages, "the DeepSeek-R1 prompt dialect");
    if tools.length() > 0 {
        // Fail loudly rather than drop them: this dialect models no tools, so a
        // silent no-op would look like the model ignoring the tool.
        return error ai:Error(
            "DeepSeek's InvokeModel dialect is text-completion and does not support tools. " +
            "Use the Converse route (the module default), which does.");
    }

    map<json> body = {
        "prompt": deepSeekPrompt(system, messages),
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
    map<json>? extra = additionalFieldsToJson(params?.additionalModelRequestFields);
    if extra is map<json> {
        foreach [string, json] [k, v] in extra.entries() {
            body[k] = v;
        }
    }
    return body;
}

// Renders messages into DeepSeek-R1's documented instruction template:
//
//   <｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>\n
//
// The tokens are DeepSeek's own full-width delimiters — they are NOT ASCII pipes,
// and substituting ASCII silently degrades the model's output rather than erroring.
//
// NOTE: AWS documents only the single-turn form above. Multi-turn and `system`
// placement are NOT specified for this dialect, so the mapping below (system folded
// ahead of the first user turn; turns alternated with the same delimiters) follows
// DeepSeek's own chat template. It preserves the module invariant that system is
// never emitted as a `role: system` message.
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html
isolated function deepSeekPrompt(string? system, ResolvedMessage[] messages) returns string {
    string prompt = "<｜begin▁of▁sentence｜>";
    if system is string {
        prompt += system;
    }
    foreach ResolvedMessage m in messages {
        if m is ai:ChatAssistantMessage {
            prompt += string `<｜Assistant｜>${m.content ?: ""}`;
            continue;
        }
        string text = m is ai:ChatFunctionMessage ? (m.content ?: "") : partsText(m.parts);
        prompt += string `<｜User｜>${text}`;
    }
    // Hand the turn to the model, opening its reasoning channel as AWS's example does.
    prompt += "<｜Assistant｜><think>\n";
    return prompt;
}

// Decodes a DeepSeek text-completion response: `choices[].text` +
// `choices[].stop_reason`. This dialect returns no token counts and no id.
isolated function decodeDeepSeekInvoke(json response) returns DecodedResponse|ai:Error {
    map<json>|error rr = response.ensureType();
    if rr is error {
        return error ai:LlmInvalidResponseError("DeepSeek response was not a JSON object", rr);
    }
    map<json> r = rr;
    json[]? choices = arrField(r, "choices");
    if choices is () || choices.length() == 0 {
        return error ai:LlmInvalidResponseError("DeepSeek response had no choices");
    }
    map<json>|error choiceResult = choices[0].ensureType();
    if choiceResult is error {
        return error ai:LlmInvalidResponseError("DeepSeek choice was not an object", choiceResult);
    }
    map<json> choice = choiceResult;
    // `text`, not `message.content` — this is a completion, not a chat turn.
    string text = strField(choice, "text") ?: "";
    return {
        message: {role: ai:ASSISTANT, content: text == "" ? () : text},
        // AWS documents no usage for this dialect; `usage` stays populated.
        usage: {inputTokens: 0, outputTokens: 0},
        // `stop_reason`, not `finish_reason`.
        stopReason: strField(choice, "stop_reason") ?: "stop",
        responseId: (),
        guardrailAction: invokeGuardrailAction(r) // body field
    };
}
