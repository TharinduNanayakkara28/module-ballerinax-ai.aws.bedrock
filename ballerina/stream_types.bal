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

// Streaming contract plumbing: the wire-format-agnostic event source, the
// per-response decoder shape, dialect selection, and the tool-index remapping every
// dialect needs.

# ONE native stream event, lifted out of whatever framing carried it.
#
# The two Bedrock endpoints do not share a streaming protocol — `bedrock-runtime`
# answers in AWS's binary event-stream, `bedrock-mantle` in SSE — but every dialect
# above that line is decoded from the same pair: a name and a JSON body. Normalizing
# to this record is what lets one `BedrockChunkIterator` (span accounting, id
# backfill, failure latching, connection release) serve both wires.
type StreamEvent record {|
    # The event's name. The `:event-type` header on event-stream, the `event:` line
    # on SSE, and `""` (or the constant `chunk`) where the dialect names its own
    # event inside the payload instead.
    string eventType;
    # The event's decoded JSON body.
    json payload;
|};

# Pulls whole events off a live response, hiding the wire framing.
#
# An OBJECT rather than a function pointer for the same reason `StreamChunkDecoder`
# is: framing is inherently stateful — a partially-filled byte buffer on one wire, a
# live SSE parser on the other — and it owns the response body, so it has to be
# closable.
type StreamEventSource object {

    # The next event, `()` at a clean end of the response, or an `ai:Error` if the
    # response failed or was truncated.
    isolated function next() returns StreamEvent|ai:Error?;

    # Releases the underlying response. Idempotent, and a no-op once the body has
    # been drained.
    isolated function close() returns ai:Error?;
};

# Converts ONE native stream event into the normalized `ai:ChatCompletionChunk`.
#
# An OBJECT, not the plain function pointer the `encode`/`decode` codec members use,
# because a stream decoder is inherently stateful: tool-call fragments have to be
# correlated across events, and the ordinal a caller sees must be derived from that
# running state. One instance per `chatStream` call.
type StreamChunkDecoder object {

    # + eventType - The event's name. Names the event on Converse and Responses;
    #               always `chunk` on Invoke and often empty on SSE, where the
    #               dialect names it in the body
    # + payload - The event's decoded JSON body
    # + return - The chunk to emit, `()` for an event with nothing to surface
    #            (`contentBlockStop`, `ping`, …), or an `ai:Error`
    isolated function decode(string eventType, json payload) returns ai:ChatCompletionChunk|ai:Error?;
};

# Which native stream dialect a route speaks.
#
# Carried on `ModelCodec.streamDialect`, so a route's dialect is settled by the same
# selection that settles its codec — see the field's own note for why it is not
# derived from the model id.
#
# The dialect is the EVENT MODEL, never the wire framing: `ANTHROPIC_STREAM` serves
# Claude on both `InvokeModelWithResponseStream` (event-stream frames) and Mantle
# Messages (SSE), because Anthropic emits the same events down either. The framing
# is chosen separately, from the route family — see `streamWireFor`.
enum StreamDialect {
    # `ConverseStream` events — and Nova on `InvokeModelWithResponseStream`, whose
    # framed payloads are Converse-shaped (the same pairing that lets
    # `INVOKE_NOVA_CODEC` reuse `decodeConverse`).
    CONVERSE_STREAM,
    # Anthropic's own event model: `InvokeModelWithResponseStream` and Mantle Messages.
    ANTHROPIC_STREAM,
    # OpenAI `chat.completion.chunk` objects: Mantle Chat Completions, and the
    # OpenAI-shaped Invoke vendors (GPT-OSS, Qwen, DeepSeek V3.x, Mistral chat).
    OPENAI_CHAT_STREAM,
    # OpenAI Responses lifecycle events (`response.output_text.delta`, …) on Mantle.
    RESPONSES_STREAM,
    # The two Invoke TEXT-completion dialects, whose streamed chunks repeat the
    # non-streaming body shape with a partial `text` — Mistral text (`outputs`) and
    # DeepSeek R1 (`choices`).
    TEXT_COMPLETION_STREAM
}

# Which wire format frames a route's stream.
#
# Derived from the route FAMILY, not the codec, because it is a property of the
# ENDPOINT: everything on `bedrock-runtime` is event-stream framed and everything on
# `bedrock-mantle` is SSE, whatever dialect rides inside. Deriving it here keeps the
# one fact in one place — `selectCodec` already keys on the family, so the two
# cannot disagree.
enum StreamWire {
    # `application/vnd.amazon.eventstream` — ConverseStream and
    # InvokeModelWithResponseStream. Framed by `EventStreamFramer`.
    EVENT_STREAM,
    # `text/event-stream` — Mantle serves the vendor APIs verbatim, so it streams
    # the way those APIs do. Parsed by `http:Response.getSseEventStream()`.
    SSE
}

// The wire format a route's family streams in.
isolated function streamWireFor(ApiFamily family) returns StreamWire => family == MANTLE ? SSE : EVENT_STREAM;

// Builds a fresh decoder for one response.
isolated function newStreamDecoder(StreamDialect dialect) returns StreamChunkDecoder {
    match dialect {
        ANTHROPIC_STREAM => {
            return new AnthropicStreamDecoder();
        }
        OPENAI_CHAT_STREAM => {
            return new OpenAIChatStreamDecoder();
        }
        RESPONSES_STREAM => {
            return new ResponsesStreamDecoder();
        }
        TEXT_COMPLETION_STREAM => {
            return new TextCompletionStreamDecoder();
        }
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
#
# The OpenAI chat dialect needs no remap — its `tool_calls[].index` already numbers
# the tool calls — which is why `OpenAIChatStreamDecoder` forwards it untouched.
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

// Reads the token counts Bedrock staples onto the LAST frame of an
// `InvokeModelWithResponseStream` response.
//
// For the Invoke text and chat dialects this is the ONLY usage that ever arrives:
// their non-streaming bodies carry no `usage` object at all (see `decodeMistralText`
// / `decodeDeepSeekInvoke`), so without this a streamed response would report zero
// tokens where the buffered one reports zero too — but the metrics frame has the
// real numbers, and dropping them wastes the one chance to bill accurately.
//
//     {"choices":[…], "amazon-bedrock-invocationMetrics":
//         {"inputTokenCount":12,"outputTokenCount":34,
//          "invocationLatency":880,"firstByteLatency":320}}
//
// `totalTokens` is derived: the metrics block reports the two halves and no total.
isolated function invocationMetricsUsage(map<json> payload) returns ai:CompletionTokenUsage? {
    map<json>? metrics = mapField(payload, "amazon-bedrock-invocationMetrics");
    if metrics is () {
        return ();
    }
    int? inputTokens = intField(metrics, "inputTokenCount");
    int? outputTokens = intField(metrics, "outputTokenCount");
    if inputTokens is () && outputTokens is () {
        return ();
    }
    int prompt = inputTokens ?: 0;
    int completion = outputTokens ?: 0;
    return {promptTokens: prompt, completionTokens: completion, totalTokens: prompt + completion};
}
