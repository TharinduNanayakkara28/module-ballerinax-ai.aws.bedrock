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

// Anthropic streaming events on `InvokeModelWithResponseStream` and Mantle Messages
// -> `ai:ChatMessageChunk`.
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

    isolated function decode(string eventType, json payload) returns StreamUpdate|ai:Error? {
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

    // `message_start` carries the message id and the INPUT token count — Anthropic
    // reports the output count separately, on `message_delta`. Nothing here is for
    // the caller, so it yields no chunk; the id is stamped on every later chunk.
    private isolated function decodeMessageStart(map<json> p) returns StreamUpdate? {
        map<json>? message = mapField(p, "message");
        if message is () {
            return ();
        }
        StreamUpdate update = {};
        string? id = strField(message, "id");
        if id is string {
            update.responseId = id;
        }
        map<json>? usage = mapField(message, "usage");
        if usage is map<json> {
            StreamUsage? mapped = streamUsage(intField(usage, "input_tokens"), ());
            if mapped is StreamUsage {
                update.usage = mapped;
            }
        }
        return update;
    }

    // `content_block_start` opens a block. Only a `tool_use` block carries anything
    // the contract wants — the tool's id and name, which arrive nowhere else.
    private isolated function decodeBlockStart(map<json> p) returns StreamUpdate? {
        map<json>? block = mapField(p, "content_block");
        if block is () || strField(block, "type") != "tool_use" {
            return ();
        }
        int blockIndex = intField(p, "index") ?: 0;
        return toolCallUpdate(toolCallChunk(self.toolIndex.indexFor(blockIndex),
                id = strField(block, "id"), name = strField(block, "name")));
    }

    // `content_block_delta` carries every incremental fragment, tagged by
    // `delta.type`.
    private isolated function decodeBlockDelta(map<json> p) returns StreamUpdate? {
        map<json>? delta = mapField(p, "delta");
        if delta is () {
            return ();
        }
        int blockIndex = intField(p, "index") ?: 0;

        match strField(delta, "type") {
            "text_delta" => {
                return contentUpdate(strField(delta, "text"));
            }
            "input_json_delta" => {
                // Tool arguments as partial JSON, forwarded raw under the index the
                // opening `content_block_start` used. The first delta is often "".
                string? partial = strField(delta, "partial_json");
                if partial is () || partial == "" {
                    return ();
                }
                return toolCallUpdate(toolCallChunk(self.toolIndex.indexFor(blockIndex), arguments = partial));
            }
            "thinking_delta" => {
                return reasoningUpdate(strField(delta, "thinking"));
            }
        }
        // `signature_delta` is the encrypted replay token for a thinking block — not
        // readable text, so it is skipped rather than mixed into `reasoning`.
        return ();
    }

    // `message_delta` closes the response: the stop reason plus the OUTPUT token
    // count (and, on newer API versions, a restated input count).
    private isolated function decodeMessageDelta(map<json> p) returns StreamUpdate? {
        StreamUpdate update = {};
        map<json>? delta = mapField(p, "delta");
        ai:FinishReason? finishReason = delta is map<json>
            ? mapAnthropicFinishReason(strField(delta, "stop_reason"))
            : ();
        if finishReason is ai:FinishReason {
            update.chunk = {role: ai:ASSISTANT, finishReason};
        }
        map<json>? usage = mapField(p, "usage");
        if usage is map<json> {
            StreamUsage? mapped = streamUsage(intField(usage, "input_tokens"), intField(usage, "output_tokens"));
            if mapped is StreamUsage {
                update.usage = mapped;
            }
        }
        return update;
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
