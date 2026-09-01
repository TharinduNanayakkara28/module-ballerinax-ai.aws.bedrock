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
    // Anthropic splits usage across two events. The input half is STASHED, not
    // emitted here: `ai` documents usage as present only on the final chunk, so it
    // is rejoined with the output half on `message_delta` — see the test below.
    test:assertEquals(chunk?.usage, (), "the opening chunk must carry no partial usage");
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
function testAnthropicStreamRejoinsBothHalvesOfUsageOnTheFinalChunk() {
    // The two halves arrive on different events; a caller reading `usage` off the
    // last chunk — the natural reading, and what the non-streaming path returns —
    // must see both, plus the total Anthropic never sends.
    AnthropicStreamDecoder decoder = new;
    _ = decodeOne(decoder, "chunk", {
        "type": "message_start",
        "message": {"id": "msg_1", "role": "assistant", "usage": {"input_tokens": 25}}
    });
    ai:ChatCompletionChunk last = decodeOne(decoder, "chunk", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 99}
    });
    ai:CompletionTokenUsage usage = <ai:CompletionTokenUsage>last?.usage;
    test:assertEquals(usage?.promptTokens, 25);
    test:assertEquals(usage?.completionTokens, 99);
    test:assertEquals(usage?.totalTokens, 124, "totalTokens is derived — Anthropic reports no total");
}

@test:Config {}
function testAnthropicStreamKeepsThePromptHalfWhenNoOutputCountArrives() {
    // A `message_delta` with no `usage` must not drop the only count the response
    // reported.
    AnthropicStreamDecoder decoder = new;
    _ = decodeOne(decoder, "chunk", {
        "type": "message_start",
        "message": {"role": "assistant", "usage": {"input_tokens": 25}}
    });
    ai:ChatCompletionChunk last = decodeOne(decoder, "chunk", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"}
    });
    ai:CompletionTokenUsage usage = <ai:CompletionTokenUsage>last?.usage;
    test:assertEquals(usage?.promptTokens, 25);
    test:assertEquals(usage?.completionTokens, ());
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
function testStreamDialectIsCarriedByTheCodec() returns error? {
    // Converse is model-agnostic — every vendor, one dialect, one codec.
    // A runtime-only Claude: `claude-sonnet-5` would auto-route to Mantle, which has
    // no streaming dialect yet.
    test:assertEquals((check selectCodec(check resolveRoute("anthropic.claude-sonnet-4-6", REGION))).streamDialect,
            CONVERSE_STREAM);
    test:assertEquals((check selectCodec(check resolveRoute("mistral.mistral-large-2407", REGION))).streamDialect,
            CONVERSE_STREAM);

    // On Invoke the dialect is the vendor's own.
    test:assertEquals(INVOKE_ANTHROPIC_CODEC.streamDialect, ANTHROPIC_STREAM);
    // Nova's Invoke frames are Converse-shaped — the same pairing that lets
    // INVOKE_NOVA_CODEC reuse decodeConverse.
    test:assertEquals(INVOKE_NOVA_CODEC.streamDialect, CONVERSE_STREAM);
}

@test:Config {}
function testCodecsWithoutAStreamingDecoderCarryNoDialect() {
    // The refusal in `runChatStream` is exactly `streamDialect is ()`, so these are
    // the routes that get the clean "use apiFamily = CONVERSE" error.
    test:assertEquals(INVOKE_MISTRAL_CHAT_CODEC.streamDialect, (), "Mistral has no Invoke streaming dialect");
    test:assertEquals(INVOKE_MISTRAL_TEXT_CODEC.streamDialect, ());
    test:assertEquals(INVOKE_OPENAI_CHAT_CODEC.streamDialect, ());
    test:assertEquals(INVOKE_DEEPSEEK_CODEC.streamDialect, ());
    // Mantle streams as SSE on the same path; that dialect is not implemented.
    test:assertEquals(MANTLE_MESSAGES_CODEC.streamDialect, ());
    test:assertEquals(MANTLE_RESPONSES_CODEC.streamDialect, ());
    test:assertEquals(MANTLE_CHAT_CODEC.streamDialect, ());
}

@test:Config {}
function testEveryCodecsDialectMatchesItsDecoder() {
    // A dialect names the decoder that reads that codec's frames. If a codec is ever
    // given a dialect whose decoder cannot read its wire shape, the stream decodes to
    // silence rather than failing — so the pairing is pinned here.
    test:assertTrue(newStreamDecoder(<StreamDialect>CONVERSE_CODEC.streamDialect) is ConverseStreamDecoder);
    test:assertTrue(newStreamDecoder(<StreamDialect>INVOKE_NOVA_CODEC.streamDialect) is ConverseStreamDecoder);
    test:assertTrue(newStreamDecoder(<StreamDialect>INVOKE_ANTHROPIC_CODEC.streamDialect) is AnthropicStreamDecoder);
}

@test:Config {}
function testArnRoutedInvokeModelStillResolvesAStreamDialect() returns error? {
    // REGRESSION. The dialect used to be looked up from `bareModelId`, which for an
    // opaque ARN IS the ARN string — matching no vendor prefix. So an imported-model
    // ARN with `modelSchema` got a codec claiming streaming and then failed dialect
    // selection every time: `chat()` worked and `chatStream()` never did, with an
    // internal-sounding "No streaming dialect for 'arn:aws:...'" message.
    string arn = "arn:aws:bedrock:us-west-2:123456789012:imported-model/abc123def456";

    Route claude = check resolveRoute(arn, REGION, {modelSchema: ANTHROPIC});
    test:assertEquals(claude.bareModelId, arn, "an opaque ARN is its own bare id — the old lookup key");
    test:assertEquals((check selectCodec(claude, ANTHROPIC)).streamDialect, ANTHROPIC_STREAM);

    Route nova = check resolveRoute(arn, REGION, {modelSchema: NOVA});
    test:assertEquals((check selectCodec(nova, NOVA)).streamDialect, CONVERSE_STREAM);

    // And an ARN that resolves to Converse streams like any other Converse route.
    Route provisioned = check resolveRoute(
            "arn:aws:bedrock:eu-west-1:123456789012:provisioned-model/xyz", REGION);
    test:assertEquals((check selectCodec(provisioned)).streamDialect, CONVERSE_STREAM);
}

// ---------------------------------------------------------------------------
// Invoke envelope: where the event NAME lives
// ---------------------------------------------------------------------------

@test:Config {}
function testNovaInvokeNamesItsEventInThePayloadKey() {
    // REGRESSION, found live 2026-08-24: Nova on InvokeModelWithResponseStream
    // produced a clean, EMPTY stream — 0 chunks, no error. On Invoke the
    // `:event-type` header is the constant `chunk`, and Nova names the event with
    // the single top-level key of the payload instead. The decoder matched `chunk`,
    // found no case, and skipped every frame.
    json novaShaped = {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hi"}}};
    [string, json] [eventType, payload] = unwrapNamedEvent("chunk", novaShaped);
    test:assertEquals(eventType, CONVERSE_EVT_CONTENT_BLOCK_DELTA);

    ConverseStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, eventType, payload);
    test:assertEquals(deltaOf(chunk).content, "hi");
}

@test:Config {}
function testNovaInvokeMetadataFrameCarriesInvocationMetricsAlongside() {
    // REGRESSION, found live 2026-08-25: every Nova Invoke frame is single-key
    // EXCEPT the terminal one, which Bedrock decorates with its own
    // `amazon-bedrock-invocationMetrics`. An arity guard on the unwrap dropped it,
    // so the stream reported no usage at all — the metadata event is the only place
    // a Converse-shaped stream ever sends it.
    json novaMetadata = {
        "metadata": {"usage": {"inputTokens": 7, "outputTokens": 6}, "metrics": {}, "trace": {}},
        "amazon-bedrock-invocationMetrics": {"inputTokenCount": 7, "outputTokenCount": 6}
    };
    [string, json] [eventType, payload] = unwrapNamedEvent("chunk", novaMetadata);
    test:assertEquals(eventType, CONVERSE_EVT_METADATA);

    ConverseStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, eventType, payload);
    ai:CompletionTokenUsage usage = <ai:CompletionTokenUsage>chunk?.usage;
    test:assertEquals(usage?.promptTokens, 7);
    test:assertEquals(usage?.completionTokens, 6);
    // Nova omits `totalTokens` on Invoke, unlike ConverseStream, which sends all three.
    test:assertEquals(usage?.totalTokens, ());
}

@test:Config {}
function testHeaderNamedEventsAreLeftAlone() {
    // ConverseStream on bedrock-runtime names the event in the HEADER, and its
    // payload is not single-key-wrapped. The fallback must return it untouched.
    json converseShaped = {"contentBlockIndex": 0, "delta": {"text": "hi"}};
    [string, json] [eventType, payload] = unwrapNamedEvent(CONVERSE_EVT_CONTENT_BLOCK_DELTA, converseShaped);
    test:assertEquals(eventType, CONVERSE_EVT_CONTENT_BLOCK_DELTA);
    test:assertEquals(payload, converseShaped);
}

@test:Config {}
function testAnthropicInvokePayloadsAreNotMistakenForNamedEvents() {
    // Anthropic-on-Invoke carries `type` ALONGSIDE other members, so it is never
    // single-key and must fall through to the header. A single-key payload whose
    // key is not a Converse event name must also fall through.
    json anthropicShaped = {"type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta", "text": "hi"}};
    [string, json] [eventType, payload] = unwrapNamedEvent("chunk", anthropicShaped);
    test:assertEquals(eventType, "chunk");
    test:assertEquals(payload, anthropicShaped);

    [string, json] [pingType, _] = unwrapNamedEvent("chunk", {"type": "ping"});
    test:assertEquals(pingType, "chunk", "a single-key payload keyed on 'type' is not a Converse event");
}
