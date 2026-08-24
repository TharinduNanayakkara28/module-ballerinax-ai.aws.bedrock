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
import ballerina/test;

// Native stream event -> normalized `ai:ChatCompletionChunk`. Pure mapping, driven
// with the exact event shapes the two dialects put on the wire.

// Decodes one event, asserting it produced a chunk.
function decodeOne(StreamChunkDecoder decoder, string eventType, json payload) returns ai:ChatCompletionChunk {
    ai:ChatCompletionChunk|ai:Error? chunk = decoder.decode(eventType, payload);
    if chunk !is ai:ChatCompletionChunk {
        panic error("expected a chunk for event " + eventType);
    }
    return chunk;
}

// The single delta every Bedrock chunk carries.
function deltaOf(ai:ChatCompletionChunk chunk) returns ai:ChatCompletionChunkDelta => chunk.choices[0].delta;

// ---------------------------------------------------------------------------
// Converse
// ---------------------------------------------------------------------------

@test:Config {}
function testConverseStreamMapsTextDeltaToContent() {
    ConverseStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, CONVERSE_EVT_CONTENT_BLOCK_DELTA,
            {"contentBlockIndex": 0, "delta": {"text": "Hello"}});
    test:assertEquals(deltaOf(chunk).content, "Hello");
    test:assertEquals(chunk.choices[0].index, 0);
    test:assertEquals(chunk.choices[0].finishReason, ());
}

@test:Config {}
function testConverseStreamMapsRoleOnlyOnMessageStart() {
    ConverseStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, CONVERSE_EVT_MESSAGE_START, {"role": "assistant"});
    test:assertEquals(deltaOf(chunk).role, ai:ASSISTANT);
    test:assertEquals(deltaOf(chunk).content, ());
}

@test:Config {}
function testConverseStreamMapsReasoningContentToReasoning() {
    // Bedrock streams Claude/Nova thinking as READABLE text, so `reasoning` is
    // genuinely populated here — unlike providers whose thought traces are opaque.
    ConverseStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, CONVERSE_EVT_CONTENT_BLOCK_DELTA,
            {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "let me think"}}});
    test:assertEquals(deltaOf(chunk).reasoning, "let me think");
    test:assertEquals(deltaOf(chunk).content, (), "reasoning must not leak into the answer text");
}

@test:Config {}
function testConverseStreamSkipsOpaqueReasoningSignature() {
    // The encrypted replay token is not readable text and has nowhere to go.
    ConverseStreamDecoder decoder = new;
    ai:ChatCompletionChunk|ai:Error? chunk = decoder.decode(CONVERSE_EVT_CONTENT_BLOCK_DELTA,
            {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "AbC123=="}}});
    test:assertTrue(chunk is (), "an opaque signature delta must be skipped, not emitted as reasoning");
}

@test:Config {}
function testConverseStreamStreamsToolCallArgumentFragments() {
    // The whole point of forwarding EVERY fragment: id and name arrive only on the
    // opening event, arguments only on the deltas. Keeping just the first would
    // yield a named tool call with no arguments.
    ConverseStreamDecoder decoder = new;

    ai:ChatCompletionChunk opened = decodeOne(decoder, CONVERSE_EVT_CONTENT_BLOCK_START,
            {"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "tu_1", "name": "get_weather"}}});
    ai:ToolCallChunk[] openCalls = <ai:ToolCallChunk[]>deltaOf(opened).toolCalls;
    test:assertEquals(openCalls[0].id, "tu_1");
    test:assertEquals(openCalls[0]?.'function?.name, "get_weather");

    string[] fragments = ["{\"cit", "y\":\"Par", "is\"}"];
    string reassembled = "";
    foreach string fragment in fragments {
        ai:ChatCompletionChunk chunk = decodeOne(decoder, CONVERSE_EVT_CONTENT_BLOCK_DELTA,
                {"contentBlockIndex": 0, "delta": {"toolUse": {"input": fragment}}});
        ai:ToolCallChunk[] calls = <ai:ToolCallChunk[]>deltaOf(chunk).toolCalls;
        test:assertEquals(calls[0].index, 0, "every fragment must carry the same accumulation index");
        reassembled += calls[0]?.'function?.arguments ?: "";
    }
    test:assertEquals(reassembled, "{\"city\":\"Paris\"}",
            "concatenating the streamed fragments must rebuild the full argument JSON");
}

@test:Config {}
function testConverseStreamRenumbersToolCallsFromZero() {
    // Converse numbers CONTENT BLOCKS. With text in block 0 and tool calls in
    // blocks 1 and 2, the tool-call indices the contract wants are 0 and 1 — not
    // the raw block ordinals, which would leave a hole at 0.
    ConverseStreamDecoder decoder = new;

    ai:ChatCompletionChunk _ = decodeOne(decoder, CONVERSE_EVT_CONTENT_BLOCK_DELTA,
            {"contentBlockIndex": 0, "delta": {"text": "working on it"}});

    ai:ChatCompletionChunk first = decodeOne(decoder, CONVERSE_EVT_CONTENT_BLOCK_START,
            {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "a", "name": "one"}}});
    ai:ChatCompletionChunk second = decodeOne(decoder, CONVERSE_EVT_CONTENT_BLOCK_START,
            {"contentBlockIndex": 2, "start": {"toolUse": {"toolUseId": "b", "name": "two"}}});

    test:assertEquals((<ai:ToolCallChunk[]>deltaOf(first).toolCalls)[0].index, 0);
    test:assertEquals((<ai:ToolCallChunk[]>deltaOf(second).toolCalls)[0].index, 1);

    // A later fragment for block 1 must map back to the SAME tool index, not a new one.
    ai:ChatCompletionChunk more = decodeOne(decoder, CONVERSE_EVT_CONTENT_BLOCK_DELTA,
            {"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{}"}}});
    test:assertEquals((<ai:ToolCallChunk[]>deltaOf(more).toolCalls)[0].index, 0);
}

@test:Config {}
function testConverseStreamMapsStopReasonAndUsage() {
    ConverseStreamDecoder decoder = new;

    ai:ChatCompletionChunk stopped = decodeOne(decoder, CONVERSE_EVT_MESSAGE_STOP, {"stopReason": "tool_use"});
    test:assertEquals(stopped.choices[0].finishReason, ai:TOOL_CALLS);

    ai:ChatCompletionChunk metadata = decodeOne(decoder, CONVERSE_EVT_METADATA,
            {"usage": {"inputTokens": 12, "outputTokens": 34, "totalTokens": 46}});
    ai:CompletionTokenUsage usage = <ai:CompletionTokenUsage>metadata?.usage;
    test:assertEquals(usage?.promptTokens, 12);
    test:assertEquals(usage?.completionTokens, 34);
    test:assertEquals(usage?.totalTokens, 46);
}

@test:Config {}
function testConverseStreamSkipsEventsWithNothingToSurface() {
    ConverseStreamDecoder decoder = new;
    test:assertTrue(decoder.decode(CONVERSE_EVT_CONTENT_BLOCK_STOP, {"contentBlockIndex": 0}) is ());
    // An event type AWS adds later must not break a stream that is otherwise fine.
    test:assertTrue(decoder.decode("someFutureEvent", {"whatever": true}) is ());
}

@test:Config {}
function testConverseFinishReasonMapping() {
    test:assertEquals(mapConverseFinishReason("end_turn"), ai:STOP);
    test:assertEquals(mapConverseFinishReason("stop_sequence"), ai:STOP);
    test:assertEquals(mapConverseFinishReason("max_tokens"), ai:LENGTH);
    test:assertEquals(mapConverseFinishReason("tool_use"), ai:TOOL_CALLS);
    test:assertEquals(mapConverseFinishReason("content_filtered"), ai:CONTENT_FILTER);
    test:assertEquals(mapConverseFinishReason("guardrail_intervened"), ai:CONTENT_FILTER);
    // Unknown and absent both yield `()` rather than a guess — and crucially never
    // panic, which a `<ai:FinishReason>` cast would.
    test:assertEquals(mapConverseFinishReason("something_new"), ());
    test:assertEquals(mapConverseFinishReason(()), ());
}

// ---------------------------------------------------------------------------
// Anthropic on Invoke
// ---------------------------------------------------------------------------

@test:Config {}
function testAnthropicStreamMapsMessageStartIdentityAndInputTokens() {
    AnthropicStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, "chunk", {
        "type": "message_start",
        "message": {
            "id": "msg_01ABC",
            "model": "claude-sonnet-5",
            "role": "assistant",
            "usage": {"input_tokens": 25}
        }
    });
    test:assertEquals(chunk.id, "msg_01ABC");
    test:assertEquals(chunk.model, "claude-sonnet-5");
    test:assertEquals(deltaOf(chunk).role, ai:ASSISTANT);
    // Anthropic splits usage across two events; the input half lands here.
    test:assertEquals((<ai:CompletionTokenUsage>chunk?.usage)?.promptTokens, 25);
}

@test:Config {}
function testAnthropicStreamStreamsToolCallArgumentFragments() {
    AnthropicStreamDecoder decoder = new;

    ai:ChatCompletionChunk opened = decodeOne(decoder, "chunk", {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "tool_use", "id": "toolu_9", "name": "lookup"}
    });
    ai:ToolCallChunk[] openCalls = <ai:ToolCallChunk[]>deltaOf(opened).toolCalls;
    test:assertEquals(openCalls[0].id, "toolu_9");
    test:assertEquals(openCalls[0]?.'function?.name, "lookup");
    test:assertEquals(openCalls[0].index, 0, "the first tool call is index 0 even though it is content block 1");

    string reassembled = "";
    foreach string fragment in ["{\"q\":", "\"bal\"}"] {
        ai:ChatCompletionChunk chunk = decodeOne(decoder, "chunk", {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": fragment}
        });
        ai:ToolCallChunk[] calls = <ai:ToolCallChunk[]>deltaOf(chunk).toolCalls;
        test:assertEquals(calls[0].index, 0);
        reassembled += calls[0]?.'function?.arguments ?: "";
    }
    test:assertEquals(reassembled, "{\"q\":\"bal\"}");
}

@test:Config {}
function testAnthropicStreamMapsTextAndThinkingDeltas() {
    AnthropicStreamDecoder decoder = new;

    ai:ChatCompletionChunk text = decodeOne(decoder, "chunk",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Sure"}});
    test:assertEquals(deltaOf(text).content, "Sure");

    ai:ChatCompletionChunk thinking = decodeOne(decoder, "chunk",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}});
    test:assertEquals(deltaOf(thinking).reasoning, "hmm");
    test:assertEquals(deltaOf(thinking).content, ());

    // The signature delta is the opaque replay token — skipped entirely.
    test:assertTrue(decoder.decode("chunk",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "x"}}) is ());
}

@test:Config {}
function testAnthropicStreamMapsMessageDeltaStopAndOutputTokens() {
    AnthropicStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, "chunk", {
        "type": "message_delta",
        "delta": {"stop_reason": "max_tokens"},
        "usage": {"output_tokens": 99}
    });
    test:assertEquals(chunk.choices[0].finishReason, ai:LENGTH);
    test:assertEquals((<ai:CompletionTokenUsage>chunk?.usage)?.completionTokens, 99);
}

@test:Config {}
function testAnthropicStreamSurfacesAnErrorEvent() {
    // A mid-stream failure must not look like a clean end of generation.
    AnthropicStreamDecoder decoder = new;
    ai:ChatCompletionChunk|ai:Error? result = decoder.decode("chunk",
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}});
    if result !is ai:Error {
        test:assertFail("an error event must surface as an ai:Error");
    }
    test:assertTrue(result.message().includes("overloaded_error"));
    test:assertTrue(result.message().includes("Overloaded"));
}

@test:Config {}
function testAnthropicStreamSkipsPingAndStops() {
    AnthropicStreamDecoder decoder = new;
    test:assertTrue(decoder.decode("chunk", {"type": "ping"}) is ());
    test:assertTrue(decoder.decode("chunk", {"type": "content_block_stop", "index": 0}) is ());
    test:assertTrue(decoder.decode("chunk", {"type": "message_stop"}) is ());
}

@test:Config {}
function testAnthropicFinishReasonMapping() {
    test:assertEquals(mapAnthropicFinishReason("end_turn"), ai:STOP);
    test:assertEquals(mapAnthropicFinishReason("stop_sequence"), ai:STOP);
    test:assertEquals(mapAnthropicFinishReason("max_tokens"), ai:LENGTH);
    test:assertEquals(mapAnthropicFinishReason("tool_use"), ai:TOOL_CALLS);
    test:assertEquals(mapAnthropicFinishReason("refusal"), ai:CONTENT_FILTER);
    test:assertEquals(mapAnthropicFinishReason("brand_new_reason"), ());
    test:assertEquals(mapAnthropicFinishReason(()), ());
}

// ---------------------------------------------------------------------------
// Dialect selection
// ---------------------------------------------------------------------------

@test:Config {}
function testStreamDialectSelection() {
    // Converse is model-agnostic — every vendor, one dialect.
    test:assertEquals(checkpanic selectStreamDialect(CONVERSE, "anthropic.claude-sonnet-5"), CONVERSE_STREAM);
    test:assertEquals(checkpanic selectStreamDialect(CONVERSE, "mistral.mistral-large-2407"), CONVERSE_STREAM);

    // On Invoke the dialect is the vendor's own.
    test:assertEquals(checkpanic selectStreamDialect(INVOKE, "anthropic.claude-opus-5"), ANTHROPIC_STREAM);
    // Nova's Invoke frames are Converse-shaped — the same pairing that lets
    // INVOKE_NOVA_CODEC reuse decodeConverse.
    test:assertEquals(checkpanic selectStreamDialect(INVOKE, "amazon.nova-pro-v1:0"), CONVERSE_STREAM);
}

@test:Config {}
function testStreamDialectRejectsAnUnsupportedInvokeVendor() {
    StreamDialect|ai:Error dialect = selectStreamDialect(INVOKE, "mistral.mistral-large-2407");
    test:assertTrue(dialect is ai:Error, "Mistral has no Invoke streaming dialect implemented");
}

@test:Config {}
function testEveryCodecClaimingStreamingHasADialect() {
    // The guard in `runChatStream` reads `supportsStreaming`, then
    // `selectStreamDialect` must succeed. If a codec ever claims streaming without
    // a matching dialect the pair has drifted, and a caller would get an
    // internal-sounding error instead of a clean refusal.
    test:assertTrue(CONVERSE_CODEC.supportsStreaming);
    test:assertTrue(INVOKE_ANTHROPIC_CODEC.supportsStreaming);
    test:assertTrue(INVOKE_NOVA_CODEC.supportsStreaming);

    test:assertTrue(selectStreamDialect(CONVERSE, "anything.at.all") is StreamDialect);
    test:assertTrue(selectStreamDialect(INVOKE, "anthropic.claude-opus-5") is StreamDialect);
    test:assertTrue(selectStreamDialect(INVOKE, "amazon.nova-lite-v1:0") is StreamDialect);

    // And the converse: the codecs that do NOT claim streaming stay refused.
    test:assertFalse(MANTLE_MESSAGES_CODEC.supportsStreaming);
    test:assertFalse(MANTLE_RESPONSES_CODEC.supportsStreaming);
    test:assertFalse(MANTLE_CHAT_CODEC.supportsStreaming);
    test:assertFalse(INVOKE_OPENAI_CHAT_CODEC.supportsStreaming);
    test:assertFalse(INVOKE_DEEPSEEK_CODEC.supportsStreaming);
    test:assertFalse(INVOKE_MISTRAL_CHAT_CODEC.supportsStreaming);
    test:assertFalse(INVOKE_MISTRAL_TEXT_CODEC.supportsStreaming);
}
