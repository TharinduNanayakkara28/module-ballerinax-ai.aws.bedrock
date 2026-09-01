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

// Streaming contract plumbing: the per-response decoder shape, dialect selection,
// and the tool-index remapping every dialect needs.

# Converts ONE native stream event into the normalized `ai:ChatCompletionChunk`.
#
# An OBJECT, not the plain function pointer the `encode`/`decode` codec members use,
# because a stream decoder is inherently stateful: tool-call fragments have to be
# correlated across events, and the ordinal a caller sees must be derived from that
# running state. One instance per `chatStream` call.
type StreamChunkDecoder object {

    # + eventType - The frame's `:event-type` header. Names the event on Converse;
    #               always `chunk` on Invoke, where the dialect names it in the body
    # + payload - The frame's decoded JSON body
    # + return - The chunk to emit, `()` for an event with nothing to surface
    #            (`contentBlockStop`, `ping`, …), or an `ai:Error`
    isolated function decode(string eventType, json payload) returns ai:ChatCompletionChunk|ai:Error?;
};

# Which native stream dialect a route speaks.
#
# Carried on `ModelCodec.streamDialect`, so a route's dialect is settled by the same
# selection that settles its codec — see the field's own note for why it is not
# derived from the model id.
enum StreamDialect {
    # `ConverseStream` events — and Nova on `InvokeModelWithResponseStream`, whose
    # framed payloads are Converse-shaped (the same pairing that lets
    # `INVOKE_NOVA_CODEC` reuse `decodeConverse`).
    CONVERSE_STREAM,
    # Anthropic's own event model on `InvokeModelWithResponseStream`.
    ANTHROPIC_STREAM
}

// Builds a fresh decoder for one response.
isolated function newStreamDecoder(StreamDialect dialect) returns StreamChunkDecoder {
    if dialect == ANTHROPIC_STREAM {
        return new AnthropicStreamDecoder();
    }
    return new ConverseStreamDecoder();
}

# Maps a dialect's native content-block ordinal onto the tool-call `index` the `ai`
# contract expects.
#
# WHY A REMAP RATHER THAN A PASSTHROUGH — both dialects number CONTENT BLOCKS, and a
# tool call is only one KIND of block. A reply with text in block 0 and tool calls in
# blocks 1 and 2 would hand a caller tool indices 1 and 2. `ai:ToolCallChunk.index`
# is the OpenAI-shaped "index used to accumulate fragments of the same tool call",
# which numbers the TOOL CALLS, from 0. Forwarding the block ordinal is not merely
# cosmetic: an accumulator that sizes its array from the indices it sees would leave
# a hole at 0 and mis-order parallel calls.
class ToolIndexMap {
    private map<int> byBlock = {};
    private int next = 0;

    # The tool-call ordinal for a content-block index, assigning the next one on
    # first sight. Stable for the life of one response.
    isolated function indexFor(int blockIndex) returns int {
        string key = blockIndex.toString();
        int? existing = self.byBlock[key];
        if existing is int {
            return existing;
        }
        int assigned = self.next;
        self.byBlock[key] = assigned;
        self.next += 1;
        return assigned;
    }
}

// A chunk carrying a single choice — the shape every event maps onto. Streamed
// Bedrock responses have exactly one candidate, so `index` is always 0.
isolated function singleChoiceChunk(ai:ChatCompletionChunkDelta delta, ai:FinishReason? finishReason = ())
        returns ai:ChatCompletionChunk
    => {choices: [{index: 0, delta, finishReason}]};

// A chunk that carries only token usage. `choices` is required by the contract, so
// a usage-only event still emits an empty delta rather than omitting the choice.
isolated function usageChunk(ai:CompletionTokenUsage usage) returns ai:ChatCompletionChunk
    => {choices: [{index: 0, delta: {}, finishReason: ()}], usage};
