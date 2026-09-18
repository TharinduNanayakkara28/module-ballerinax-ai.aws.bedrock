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

// ConverseStream event -> `ai:ChatMessageChunk`.
//
// The highest-leverage decoder in the module: `ConverseStream` is model-agnostic,
// so this one mapping serves all seven vendor facades on the default route. Nova on
// `InvokeModelWithResponseStream` reuses it too — its framed payloads are
// Converse-shaped, the same pairing that lets `INVOKE_NOVA_CONVERTER` reuse
// `decodeConverse`.
//
// https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html

// `:event-type` header values.
const string CONVERSE_EVT_MESSAGE_START = "messageStart";
const string CONVERSE_EVT_CONTENT_BLOCK_START = "contentBlockStart";
const string CONVERSE_EVT_CONTENT_BLOCK_DELTA = "contentBlockDelta";
const string CONVERSE_EVT_CONTENT_BLOCK_STOP = "contentBlockStop";
const string CONVERSE_EVT_MESSAGE_STOP = "messageStop";
const string CONVERSE_EVT_METADATA = "metadata";

# Decodes one `ConverseStream` response.
class ConverseStreamDecoder {
    *StreamChunkDecoder;

    private final ToolIndexMap toolIndex = new;

    isolated function decode(string eventType, json payload) returns StreamUpdate|ai:Error? {
        map<json> p = payload is map<json> ? payload : {};

        match eventType {
            CONVERSE_EVT_CONTENT_BLOCK_START => {
                return self.decodeBlockStart(p);
            }
            CONVERSE_EVT_CONTENT_BLOCK_DELTA => {
                return self.decodeBlockDelta(p);
            }
            CONVERSE_EVT_MESSAGE_STOP => {
                ai:FinishReason? finishReason = mapConverseFinishReason(strField(p, "stopReason"));
                return finishReason is ai:FinishReason ? {chunk: {role: ai:ASSISTANT, finishReason}} : ();
            }
            CONVERSE_EVT_METADATA => {
                map<json>? usage = mapField(p, "usage");
                if usage is () {
                    return ();
                }
                StreamUsage? mapped = streamUsage(intField(usage, "inputTokens"), intField(usage, "outputTokens"));
                return mapped is StreamUsage ? {usage: mapped} : ();
            }
        }
        // `messageStart` only restates the role, which every chunk carries anyway;
        // `contentBlockStop` has no counterpart in the contract; and an event type
        // AWS adds later must not break a stream that is otherwise readable.
        return ();
    }

    // `contentBlockStart` opens a tool call and carries its id and name — the only
    // event that does. Text blocks have no meaningful start payload.
    private isolated function decodeBlockStart(map<json> p) returns StreamUpdate? {
        map<json>? blockStart = mapField(p, "start");
        if blockStart is () {
            return ();
        }
        map<json>? toolUse = mapField(blockStart, "toolUse");
        if toolUse is () {
            return ();
        }
        int blockIndex = intField(p, "contentBlockIndex") ?: 0;
        return toolCallUpdate(toolCallChunk(self.toolIndex.indexFor(blockIndex),
                id = strField(toolUse, "toolUseId"), name = strField(toolUse, "name")));
    }

    // `contentBlockDelta` is where all the incremental content lives. The `delta`
    // object is a tagged union — exactly one member is present.
    private isolated function decodeBlockDelta(map<json> p) returns StreamUpdate? {
        map<json>? delta = mapField(p, "delta");
        if delta is () {
            return ();
        }
        int blockIndex = intField(p, "contentBlockIndex") ?: 0;

        if delta.hasKey("text") {
            return contentUpdate(strField(delta, "text"));
        }

        // Tool arguments stream as a partial JSON STRING, fragment by fragment,
        // forwarded raw under the index the opening `contentBlockStart` used.
        map<json>? toolUse = mapField(delta, "toolUse");
        if toolUse is map<json> {
            string? input = strField(toolUse, "input");
            if input is () || input == "" {
                return ();
            }
            return toolCallUpdate(toolCallChunk(self.toolIndex.indexFor(blockIndex), arguments = input));
        }

        // Extended thinking. Bedrock streams reasoning as readable text for Claude
        // and Nova. `signature` and `redactedContent` are encrypted replay tokens,
        // not text, so they are skipped rather than leaked into `reasoning`.
        map<json>? reasoning = mapField(delta, "reasoningContent");
        if reasoning is map<json> {
            return reasoningUpdate(strField(reasoning, "text"));
        }
        return ();
    }
}

// Converse `stopReason` -> `ai:FinishReason`.
//
// `ai:FinishReason` has four members; Converse has more, so several collapse.
// `guardrail_intervened` maps to CONTENT_FILTER, which is the closest honest fit —
// the non-streaming path has the same problem and routes the signal to the observe
// span instead (see `runChat`), a channel a stream has no equivalent of.
//
// An unrecognised reason returns `()` rather than a guess: the field is explicitly
// nullable, and `()` reads as "no reason reported", which is true.
isolated function mapConverseFinishReason(string? stopReason) returns ai:FinishReason? {
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
        "content_filtered"|"guardrail_intervened" => {
            return ai:CONTENT_FILTER;
        }
    }
    return ();
}
