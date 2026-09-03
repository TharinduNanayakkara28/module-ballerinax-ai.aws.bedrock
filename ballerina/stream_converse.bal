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

// ConverseStream event -> `ai:ChatCompletionChunk`.
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

    isolated function decode(string eventType, json payload) returns ai:ChatCompletionChunk|ai:Error? {
        map<json> p = payload is map<json> ? payload : {};

        match eventType {
            CONVERSE_EVT_MESSAGE_START => {
                // Role arrives once, on the opening event, exactly as the contract's
                // "only sent on the first delta" expects. Converse spells the
                // assistant role `assistant`, so the lookup is a formality — but it
                // stays a lookup, never a `<ai:ROLE>` cast, which would panic on an
                // unexpected value rather than degrade.
                return singleChoiceChunk({role: mapConverseRole(strField(p, "role"))});
            }
            CONVERSE_EVT_CONTENT_BLOCK_START => {
                return self.decodeBlockStart(p);
            }
            CONVERSE_EVT_CONTENT_BLOCK_DELTA => {
                return self.decodeBlockDelta(p);
            }
            CONVERSE_EVT_CONTENT_BLOCK_STOP => {
                // Nothing to surface: the block's content already streamed, and the
                // contract has no "block ended" signal.
                return ();
            }
            CONVERSE_EVT_MESSAGE_STOP => {
                return singleChoiceChunk({}, mapConverseFinishReason(strField(p, "stopReason")));
            }
            CONVERSE_EVT_METADATA => {
                return self.decodeMetadata(p);
            }
        }
        // An event type AWS added after this was written. Skipping is right: the
        // contract has no way to express an unknown event, and erroring would break
        // a stream that is otherwise perfectly readable.
        return ();
    }

    // `contentBlockStart` opens a tool call and carries its id and name — the only
    // event that does. Text blocks have no meaningful start payload.
    private isolated function decodeBlockStart(map<json> p) returns ai:ChatCompletionChunk? {
        map<json>? blockStart = mapField(p, "start");
        if blockStart is () {
            return ();
        }
        map<json>? toolUse = mapField(blockStart, "toolUse");
        if toolUse is () {
            return ();
        }
        int blockIndex = intField(p, "contentBlockIndex") ?: 0;
        ai:ToolCallChunk chunk = {
            index: self.toolIndex.indexFor(blockIndex),
            id: strField(toolUse, "toolUseId"),
            'function: {name: strField(toolUse, "name")}
        };
        return singleChoiceChunk({toolCalls: [chunk]});
    }

    // `contentBlockDelta` is where all the incremental content lives. The `delta`
    // object is a tagged union — exactly one member is present.
    private isolated function decodeBlockDelta(map<json> p) returns ai:ChatCompletionChunk? {
        map<json>? delta = mapField(p, "delta");
        if delta is () {
            return ();
        }
        int blockIndex = intField(p, "contentBlockIndex") ?: 0;

        string? text = strField(delta, "text");
        if text is string {
            return singleChoiceChunk({content: text});
        }

        // Tool arguments stream as a partial JSON STRING, fragment by fragment,
        // which is exactly what `FunctionCallChunk.arguments` is defined to carry.
        // Every fragment is forwarded, keyed by the same index the opening
        // `contentBlockStart` used — forwarding only the first would deliver a tool
        // call with a name and no arguments.
        map<json>? toolUse = mapField(delta, "toolUse");
        if toolUse is map<json> {
            string? input = strField(toolUse, "input");
            if input is string {
                ai:ToolCallChunk chunk = {
                    index: self.toolIndex.indexFor(blockIndex),
                    'function: {arguments: input}
                };
                return singleChoiceChunk({toolCalls: [chunk]});
            }
            return ();
        }

        // Extended thinking. Unlike Gemini — whose thought signatures are opaque —
        // Bedrock streams reasoning as readable text for Claude and Nova, so the
        // contract's `reasoning` field is genuinely populated here.
        //
        // `signature` and `redactedContent` are the encrypted replay tokens that
        // ride alongside; they are not human-readable text and there is nowhere in
        // `ChatCompletionChunkDelta` to put them, so they are skipped rather than
        // leaked into `reasoning` as noise.
        map<json>? reasoning = mapField(delta, "reasoningContent");
        if reasoning is map<json> {
            string? reasoningText = strField(reasoning, "text");
            if reasoningText is string {
                return singleChoiceChunk({reasoning: reasoningText});
            }
        }
        return ();
    }

    // `metadata` closes the response with token usage. `ai:CompletionTokenUsage`
    // has all three members optional, but Converse always sends all three.
    private isolated function decodeMetadata(map<json> p) returns ai:ChatCompletionChunk? {
        map<json>? usage = mapField(p, "usage");
        if usage is () {
            return ();
        }
        int? inputTokens = intField(usage, "inputTokens");
        int? outputTokens = intField(usage, "outputTokens");
        int? totalTokens = intField(usage, "totalTokens");
        ai:CompletionTokenUsage mapped = {};
        if inputTokens is int {
            mapped.promptTokens = inputTokens;
        }
        if outputTokens is int {
            mapped.completionTokens = outputTokens;
        }
        if totalTokens is int {
            mapped.totalTokens = totalTokens;
        }
        return usageChunk(mapped);
    }
}

// Converse role -> `ai:ROLE`. A lookup, never a cast: an unrecognised role yields
// `()` (the field is optional) instead of panicking mid-stream.
isolated function mapConverseRole(string? role) returns ai:ROLE? {
    if role == "assistant" {
        return ai:ASSISTANT;
    }
    if role == "user" {
        return ai:USER;
    }
    if role == "system" {
        return ai:SYSTEM;
    }
    return ();
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
