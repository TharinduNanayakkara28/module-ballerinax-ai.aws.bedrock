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

// Anthropic streaming events on `InvokeModelWithResponseStream` ->
// `ai:ChatCompletionChunk`.
//
// The one dialect that is NOT Converse-shaped. Every frame arrives with
// `:event-type: chunk`, so the event is named by the payload's own `type` field
// rather than by the frame header — the reason `StreamChunkDecoder.decode` is
// handed both.
//
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

// Payload `type` values.
const string ANTHROPIC_EVT_MESSAGE_START = "message_start";
const string ANTHROPIC_EVT_CONTENT_BLOCK_START = "content_block_start";
const string ANTHROPIC_EVT_CONTENT_BLOCK_DELTA = "content_block_delta";
const string ANTHROPIC_EVT_CONTENT_BLOCK_STOP = "content_block_stop";
const string ANTHROPIC_EVT_MESSAGE_DELTA = "message_delta";
const string ANTHROPIC_EVT_MESSAGE_STOP = "message_stop";
const string ANTHROPIC_EVT_PING = "ping";
const string ANTHROPIC_EVT_ERROR = "error";

# Decodes one Anthropic `InvokeModelWithResponseStream` response.
class AnthropicStreamDecoder {
    *StreamChunkDecoder;

    private final ToolIndexMap toolIndex = new;
    // The prompt-token count, carried from `message_start` to `message_delta` so the
    // two halves of `usage` reach the caller together. See `decodeMessageStart`.
    private int promptTokens = 0;

    isolated function decode(string eventType, json payload) returns ai:ChatCompletionChunk|ai:Error? {
        map<json> p = payload is map<json> ? payload : {};
        // The frame header is the constant `chunk` here; the dialect names its own
        // events in the body.
        string 'type = strField(p, "type") ?: "";

        match 'type {
            ANTHROPIC_EVT_MESSAGE_START => {
                return self.decodeMessageStart(p);
            }
            ANTHROPIC_EVT_CONTENT_BLOCK_START => {
                return self.decodeBlockStart(p);
            }
            ANTHROPIC_EVT_CONTENT_BLOCK_DELTA => {
                return self.decodeBlockDelta(p);
            }
            ANTHROPIC_EVT_MESSAGE_DELTA => {
                return self.decodeMessageDelta(p);
            }
            ANTHROPIC_EVT_CONTENT_BLOCK_STOP|ANTHROPIC_EVT_MESSAGE_STOP|ANTHROPIC_EVT_PING => {
                // `ping` is a keep-alive and the two stops carry nothing the
                // contract can express; the finish reason came on `message_delta`.
                return ();
            }
            ANTHROPIC_EVT_ERROR => {
                // A mid-stream error event. Surfaced as an error rather than
                // skipped: the generation has failed, and silently ending the
                // stream would look to a caller like a clean, complete answer.
                map<json>? err = mapField(p, "error");
                string detail = err is map<json> ? (strField(err, "message") ?: "unknown") : "unknown";
                string kind = err is map<json> ? (strField(err, "type") ?: "error") : "error";
                return error ai:LlmError(string `Bedrock stream error (${kind}): ${detail}`);
            }
        }
        return ();
    }

    // `message_start` carries the response id, the resolved model, and the input
    // token count. Anthropic reports prompt tokens HERE and completion tokens on
    // `message_delta` — the two halves of `usage` arrive in different events, unlike
    // Converse which sends both together in `metadata`.
    //
    // The prompt count is STASHED rather than emitted on this chunk. `ai` documents
    // `usage` as "present only on the final chunk", so a caller reading usage off the
    // last chunk — the natural reading, and what the non-streaming path returns —
    // would otherwise see `completionTokens` alone and silently lose the prompt half.
    // `message_delta` re-joins them.
    private isolated function decodeMessageStart(map<json> p) returns ai:ChatCompletionChunk? {
        map<json>? message = mapField(p, "message");
        ai:ChatCompletionChunk chunk = singleChoiceChunk({role: ai:ASSISTANT});
        if message is () {
            return chunk;
        }
        string? id = strField(message, "id");
        if id is string {
            chunk.id = id;
        }
        string? model = strField(message, "model");
        if model is string {
            chunk.model = model;
        }
        map<json>? usage = mapField(message, "usage");
        if usage is map<json> {
            int? inputTokens = intField(usage, "input_tokens");
            if inputTokens is int {
                self.promptTokens = inputTokens;
            }
        }
        return chunk;
    }

    // `content_block_start` opens a block. Only a `tool_use` block carries anything
    // the contract wants — the tool's id and name, which arrive nowhere else.
    private isolated function decodeBlockStart(map<json> p) returns ai:ChatCompletionChunk? {
        map<json>? block = mapField(p, "content_block");
        if block is () || strField(block, "type") != "tool_use" {
            return ();
        }
        int blockIndex = intField(p, "index") ?: 0;
        ai:ToolCallChunk chunk = {
            index: self.toolIndex.indexFor(blockIndex),
            id: strField(block, "id"),
            'function: {name: strField(block, "name")}
        };
        return singleChoiceChunk({toolCalls: [chunk]});
    }

    // `content_block_delta` carries every incremental fragment, tagged by
    // `delta.type`.
    private isolated function decodeBlockDelta(map<json> p) returns ai:ChatCompletionChunk? {
        map<json>? delta = mapField(p, "delta");
        if delta is () {
            return ();
        }
        int blockIndex = intField(p, "index") ?: 0;

        match strField(delta, "type") {
            "text_delta" => {
                string? text = strField(delta, "text");
                return text is string ? singleChoiceChunk({content: text}) : ();
            }
            "input_json_delta" => {
                // Tool arguments as partial JSON. Forwarded on EVERY fragment,
                // keyed by the index the opening `content_block_start` used — only
                // the first fragment carries id and name, only the rest carry
                // arguments, and a caller needs both halves to reconstruct the call.
                string? partial = strField(delta, "partial_json");
                if partial is () {
                    return ();
                }
                ai:ToolCallChunk chunk = {
                    index: self.toolIndex.indexFor(blockIndex),
                    'function: {arguments: partial}
                };
                return singleChoiceChunk({toolCalls: [chunk]});
            }
            "thinking_delta" => {
                string? thinking = strField(delta, "thinking");
                return thinking is string ? singleChoiceChunk({reasoning: thinking}) : ();
            }
            "signature_delta" => {
                // The encrypted replay token for a thinking block — not readable
                // text, and the contract has nowhere to carry it. Skipped rather
                // than mixed into `reasoning`.
                return ();
            }
        }
        return ();
    }

    // `message_delta` closes the response: the stop reason plus the OUTPUT token
    // count. It is the LAST chunk this decoder emits — `message_stop` and the final
    // `ping` carry nothing the contract can express — so it is where the complete
    // `usage` belongs, rejoined with the prompt count stashed on `message_start`.
    //
    // `totalTokens` is derived rather than read: Anthropic reports the two halves and
    // no total, while Converse sends all three. Computing it here means a caller sees
    // the same fully-populated `usage` on either dialect.
    private isolated function decodeMessageDelta(map<json> p) returns ai:ChatCompletionChunk? {
        map<json>? delta = mapField(p, "delta");
        ai:FinishReason? finishReason = delta is map<json>
            ? mapAnthropicFinishReason(strField(delta, "stop_reason"))
            : ();
        ai:ChatCompletionChunk chunk = singleChoiceChunk({}, finishReason);
        int? outputTokens = ();
        map<json>? usage = mapField(p, "usage");
        if usage is map<json> {
            outputTokens = intField(usage, "output_tokens");
        }
        if outputTokens is int {
            chunk.usage = {
                promptTokens: self.promptTokens,
                completionTokens: outputTokens,
                totalTokens: self.promptTokens + outputTokens
            };
        } else if self.promptTokens > 0 {
            // A `message_delta` with no output count still has to carry the prompt
            // half; dropping it would lose the only usage the response reported.
            chunk.usage = {promptTokens: self.promptTokens};
        }
        return chunk;
    }
}

// Anthropic `stop_reason` -> `ai:FinishReason`. A lookup, never a cast; an
// unrecognised reason yields `()`.
//
// Anthropic spells the natural stop `end_turn` where OpenAI says `stop`, and adds
// `refusal`, which has no dedicated member — CONTENT_FILTER is the honest fit, since
// a refusal is the model declining to produce the content.
isolated function mapAnthropicFinishReason(string? stopReason) returns ai:FinishReason? {
    match stopReason {
        "end_turn"|"stop_sequence" => {
            return ai:STOP;
        }
        "max_tokens" => {
            return ai:LENGTH;
        }
        "tool_use" => {
            return ai:TOOL_CALLS;
        }
        "refusal" => {
            return ai:CONTENT_FILTER;
        }
    }
    return ();
}
