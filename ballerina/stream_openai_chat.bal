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

// OpenAI `chat.completion.chunk` -> `ai:ChatCompletionChunk`.
//
// Serves FOUR codecs across BOTH wires, because the chunk object is the same
// wherever this dialect is spoken:
//
//   MANTLE_CHAT_CODEC            SSE on `/v1/chat/completions`   (GLM, Gemma 3, …)
//   INVOKE_OPENAI_CHAT_CODEC     event-stream frames             (GPT-OSS, Qwen, DeepSeek V3.x)
//   INVOKE_MISTRAL_CHAT_CODEC    event-stream frames             (Mistral Large 24.07)
//
// MISTRAL SHARES IT DESPITE HAVING ITS OWN CODEC. The two differ on the buffered
// path in ways that matter — Mistral spells the stop reason `stop_reason`, forces
// tools with the bare string `"any"`, and documents no `usage` block (see
// codec_mistral.bal) — but a STREAMED chunk differs only in that spelling, which
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

    isolated function decode(string eventType, json payload) returns ai:ChatCompletionChunk|ai:Error? {
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

        ai:ChatCompletionChunkDelta delta = {};
        ai:FinishReason? finishReason = ();

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
                    delta = self.mapDelta(d);
                }
                // `stop_reason` is Mistral's spelling of the same field (§7.2).
                finishReason = mapOpenAIFinishReason(
                        strField(first, "finish_reason") ?: strField(first, "stop_reason"));
            }
        }

        // Usage rides the FINAL chunk, which carries an EMPTY `choices` array — so
        // what decides whether an event is worth emitting is the CONTENT it produced,
        // never the choice count. On Mantle usage appears only when the request asked
        // for it (`stream_options: {include_usage: true}`, set by the codec); on
        // Invoke it arrives as Bedrock's own invocation metrics, which is the only
        // usage those vendors report at all.
        ai:CompletionTokenUsage? usage = openAIChatUsage(p) ?: invocationMetricsUsage(p);
        if usage is () && finishReason is () && isEmptyDelta(delta) {
            // A keep-alive, a lifecycle-only object, or a choice whose delta carried
            // nothing. Skipping keeps an empty chunk — one a caller would have to
            // filter out of its own output — off the stream, the same way
            // `contentBlockStop` is skipped on Converse.
            return ();
        }
        ai:ChatCompletionChunk chunk = singleChoiceChunk(delta, finishReason);
        if usage is ai:CompletionTokenUsage {
            chunk.usage = usage;
        }
        // This dialect names itself, unlike Converse — keep what it sent rather than
        // letting the iterator backfill the request id and the configured model id.
        string? id = strField(p, "id");
        if id is string {
            chunk.id = id;
        }
        string? model = strField(p, "model");
        if model is string {
            chunk.model = model;
        }
        return chunk;
    }

    // One `delta` (or buffered `message`) object -> the normalized delta.
    private isolated function mapDelta(map<json> d) returns ai:ChatCompletionChunkDelta {
        ai:ChatCompletionChunkDelta delta = {};
        ai:ROLE? role = mapOpenAIRole(strField(d, "role"));
        if role is ai:ROLE {
            delta.role = role;
        }
        string? content = strField(d, "content");
        if content is string {
            delta.content = content;
        }
        // Reasoning models on this dialect (DeepSeek V3.x, Qwen with thinking on)
        // stream their chain-of-thought in a SEPARATE member, so it reaches the
        // caller as `reasoning` and never merges into the answer text. `reasoning`
        // is the newer spelling of the same field.
        string? reasoning = strField(d, "reasoning_content") ?: strField(d, "reasoning");
        if reasoning is string {
            delta.reasoning = reasoning;
        }

        json[]? toolCalls = arrField(d, "tool_calls");
        if toolCalls is json[] && toolCalls.length() > 0 {
            ai:ToolCallChunk[] chunks = [];
            int position = 0;
            foreach json call in toolCalls {
                if call is map<json> {
                    chunks.push(self.mapToolCall(call, position));
                }
                position += 1;
            }
            if chunks.length() > 0 {
                delta.toolCalls = chunks;
            }
        }
        return delta;
    }

    // One streamed tool-call fragment.
    //
    // NO `ToolIndexMap` here, unlike every other dialect: `tool_calls[].index`
    // already numbers the TOOL CALLS from 0 — it is the field `ai:ToolCallChunk.index`
    // was modelled on — so remapping it would be a no-op at best and a renumbering
    // at worst. The array position is the fallback for a vendor that omits it, which
    // is correct for the single-chunk tool calls those vendors emit.
    private isolated function mapToolCall(map<json> call, int position) returns ai:ToolCallChunk {
        ai:ToolCallChunk chunk = {index: intField(call, "index") ?: position};
        string? id = strField(call, "id");
        if id is string {
            chunk.id = id;
        }
        map<json>? fn = mapField(call, "function");
        if fn is map<json> {
            ai:FunctionCallChunk fragment = {};
            string? name = strField(fn, "name");
            if name is string {
                fragment.name = name;
            }
            // Partial JSON, forwarded verbatim on every fragment — exactly what
            // `FunctionCallChunk.arguments` is defined to carry.
            string? arguments = strField(fn, "arguments");
            if arguments is string {
                fragment.arguments = arguments;
            }
            chunk.'function = fragment;
        }
        return chunk;
    }
}

// The OpenAI `usage` object. Absent on all but the final chunk, and absent
// entirely unless `stream_options: {include_usage: true}` was sent.
isolated function openAIChatUsage(map<json> payload) returns ai:CompletionTokenUsage? {
    map<json>? usage = mapField(payload, "usage");
    if usage is () {
        return ();
    }
    int? promptTokens = intField(usage, "prompt_tokens");
    int? completionTokens = intField(usage, "completion_tokens");
    int? totalTokens = intField(usage, "total_tokens");
    if promptTokens is () && completionTokens is () && totalTokens is () {
        // `"usage": null` on every non-final chunk is the documented shape; treat an
        // empty object the same way rather than emitting a usage chunk of zeroes.
        return ();
    }
    ai:CompletionTokenUsage mapped = {};
    if promptTokens is int {
        mapped.promptTokens = promptTokens;
    }
    if completionTokens is int {
        mapped.completionTokens = completionTokens;
    }
    if totalTokens is int {
        mapped.totalTokens = totalTokens;
    }
    return mapped;
}

// OpenAI `role` -> `ai:ROLE`. A lookup, never a `<ai:ROLE>` cast, which would panic
// mid-stream on a value the enum does not carry.
isolated function mapOpenAIRole(string? role) returns ai:ROLE? {
    match role {
        "assistant" => {
            return ai:ASSISTANT;
        }
        "user" => {
            return ai:USER;
        }
        "system" => {
            return ai:SYSTEM;
        }
    }
    return ();
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

// Whether a delta carries nothing at all. `ai:ChatCompletionChunkDelta` defaults
// `content`, `reasoning` and `toolCalls` to `()`, so an all-defaults record is the
// "no news" chunk some vendors emit between real ones.
isolated function isEmptyDelta(ai:ChatCompletionChunkDelta delta) returns boolean =>
    delta.content is () && delta.reasoning is () && delta.toolCalls is () && delta?.role is ();
