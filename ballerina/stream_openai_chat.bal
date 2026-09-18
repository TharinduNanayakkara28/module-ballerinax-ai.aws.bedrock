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

// OpenAI `chat.completion.chunk` -> `ai:ChatMessageChunk`.
//
// Serves FOUR converters across BOTH wires, because the chunk object is the same
// wherever this dialect is spoken:
//
//   MANTLE_CHAT_CONVERTER            SSE on `/v1/chat/completions`   (GLM, Gemma 3, …)
//   INVOKE_OPENAI_CHAT_CONVERTER     event-stream frames             (GPT-OSS, Qwen, DeepSeek V3.x)
//   INVOKE_MISTRAL_CHAT_CONVERTER    event-stream frames             (Mistral Large 24.07)
//
// MISTRAL SHARES IT DESPITE HAVING ITS OWN CODEC. The two differ on the buffered
// path in ways that matter — Mistral spells the stop reason `stop_reason`, forces
// tools with the bare string `"any"`, and documents no `usage` block (see
// converter_mistral.bal) — but a STREAMED chunk differs only in that spelling, which
// `finish_reason ?: stop_reason` absorbs. A second decoder would be the same code
// with one field name changed.
//
// The event has NO name on this dialect: SSE sends bare `data:` lines with no
// `event:`, and every Invoke frame is the constant `chunk`. Everything is read out
// of the payload.
//
// https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html

# Decodes one `chat.completion.chunk` stream.
class OpenAIChatStreamDecoder {
    *StreamChunkDecoder;

    // Tool-call indices whose first fragment has already gone out. Some vendors
    // repeat `id`/`name` on later fragments; the contract sends them once.
    private final map<boolean> announced = {};

    isolated function decode(string eventType, json payload) returns StreamUpdate|ai:Error? {
        map<json> p = payload is map<json> ? payload : {};

        // An in-band failure. The OpenAI dialects can put an `error` object on an
        // otherwise ordinary data line instead of using an `event: error` frame, so
        // it is checked here as well as in the SSE source.
        map<json>? err = mapField(p, "error");
        if err is map<json> {
            string detail = strField(err, "message") ?: "unknown";
            string kind = strField(err, "type") ?: strField(err, "code") ?: "error";
            return error ai:LlmError(string `Bedrock stream error (${kind}): ${detail}`);
        }

        ai:ChatMessageChunk chunk = {role: ai:ASSISTANT};
        json[]? choices = arrField(p, "choices");
        if choices is json[] && choices.length() > 0 {
            json first = choices[0];
            if first is map<json> {
                // `delta` is the streaming member. `message` is accepted as a
                // fallback because the Invoke vendors reuse their BUFFERED chunk
                // shape on the stream — the field the non-streaming body uses —
                // and reading only `delta` there would drop every token.
                map<json>? d = mapField(first, "delta") ?: mapField(first, "message");
                if d is map<json> {
                    self.mapDelta(d, chunk);
                }
                // `stop_reason` is Mistral's spelling of the same field.
                chunk.finishReason = mapOpenAIFinishReason(
                        strField(first, "finish_reason") ?: strField(first, "stop_reason"));
            }
        }

        StreamUpdate update = {};
        // A role-only opener, a keep-alive or an empty delta carries nothing for the
        // caller, so no chunk is emitted for it.
        if chunk.content is string || chunk.reasoning is string || chunk.toolCalls is ai:ToolCallChunk[]
                || chunk.finishReason is ai:FinishReason {
            update.chunk = chunk;
        }
        // Usage rides the FINAL chunk, which carries an EMPTY `choices` array. On
        // Mantle it appears only when the request asked for it
        // (`stream_options: {include_usage: true}`, set by the converter); on Invoke
        // it arrives as Bedrock's own invocation metrics.
        StreamUsage? usage = openAIChatUsage(p) ?: invocationMetricsUsage(p);
        if usage is StreamUsage {
            update.usage = usage;
        }
        string? id = strField(p, "id");
        if id is string {
            update.responseId = id;
        }
        return update.length() == 0 ? () : update;
    }

    // One `delta` (or buffered `message`) object -> the chunk's content members.
    private isolated function mapDelta(map<json> d, ai:ChatMessageChunk chunk) {
        string? content = strField(d, "content");
        if content is string && content != "" {
            chunk.content = content;
        }
        // Reasoning models on this dialect (DeepSeek V3.x, Qwen with thinking on)
        // stream their chain-of-thought in a SEPARATE member, so it reaches the
        // caller as `reasoning` and never merges into the answer text. `reasoning`
        // is the newer spelling of the same field.
        string? reasoning = strField(d, "reasoning_content") ?: strField(d, "reasoning");
        if reasoning is string && reasoning != "" {
            chunk.reasoning = reasoning;
        }

        json[]? toolCalls = arrField(d, "tool_calls");
        if toolCalls is json[] && toolCalls.length() > 0 {
            ai:ToolCallChunk[] calls = [];
            int position = 0;
            foreach json call in toolCalls {
                if call is map<json> {
                    ai:ToolCallChunk? mapped = self.mapToolCall(call, position);
                    if mapped is ai:ToolCallChunk {
                        calls.push(mapped);
                    }
                }
                position += 1;
            }
            if calls.length() > 0 {
                chunk.toolCalls = calls;
            }
        }
    }

    // One streamed tool-call fragment, or `()` if it carries nothing.
    //
    // NO `ToolIndexMap` here, unlike every other dialect: `tool_calls[].index`
    // already numbers the TOOL CALLS from 0 — it is the field `ai:ToolCallChunk.index`
    // was modelled on. The array position is the fallback for a vendor that omits
    // it, which is correct for the single-chunk tool calls those vendors emit.
    private isolated function mapToolCall(map<json> call, int position) returns ai:ToolCallChunk? {
        int index = intField(call, "index") ?: position;
        map<json> fn = mapField(call, "function") ?: {};
        string key = index.toString();
        boolean first = !self.announced.hasKey(key);
        string? id = first ? strField(call, "id") : ();
        string? name = first ? strField(fn, "name") : ();
        if id is string || name is string {
            self.announced[key] = true;
        }
        ai:ToolCallChunk mapped = toolCallChunk(index, id, name, strField(fn, "arguments"));
        return mapped.length() == 1 ? () : mapped;
    }
}

// The OpenAI `usage` object. Absent on all but the final chunk, and absent
// entirely unless `stream_options: {include_usage: true}` was sent. `"usage": null`
// on every non-final chunk is the documented shape.
isolated function openAIChatUsage(map<json> payload) returns StreamUsage? {
    map<json>? usage = mapField(payload, "usage");
    if usage is () {
        return ();
    }
    return streamUsage(intField(usage, "prompt_tokens"), intField(usage, "completion_tokens"));
}

// OpenAI `finish_reason` (and Mistral's `stop_reason`) -> `ai:FinishReason`.
//
// `ai:FinishReason` is modelled on this very set, so the mapping is near-identity;
// `function_call` is the pre-`tool_calls` spelling, still emitted by some vendors.
// An unrecognised reason returns `()` rather than a guess — the field is explicitly
// nullable, and `()` reads as "no reason reported", which is true.
isolated function mapOpenAIFinishReason(string? finishReason) returns ai:FinishReason? {
    match finishReason {
        "stop"|"end_turn"|"stop_sequence" => {
            return ai:STOP;
        }
        "length"|"model_length"|"max_tokens" => {
            return ai:LENGTH;
        }
        "tool_calls"|"function_call"|"tool_use" => {
            return ai:TOOL_CALLS;
        }
        "content_filter"|"guardrail_intervened" => {
            return ai:CONTENT_FILTER;
        }
    }
    return ();
}
