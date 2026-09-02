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
function testEveryShippedCodecCarriesAStreamingDialect() {
    // The capability matrix, pinned. `runChatStream` refuses exactly
    // `streamDialect is ()`, so a codec that loses its dialect stops streaming — a
    // silent capability regression for every model on that route.
    (readonly & ModelCodec)[] codecs = [
        CONVERSE_CODEC,
        INVOKE_ANTHROPIC_CODEC,
        INVOKE_NOVA_CODEC,
        INVOKE_OPENAI_CHAT_CODEC,
        INVOKE_DEEPSEEK_CODEC,
        INVOKE_MISTRAL_CHAT_CODEC,
        INVOKE_MISTRAL_TEXT_CODEC,
        MANTLE_MESSAGES_CODEC,
        MANTLE_RESPONSES_CODEC,
        MANTLE_CHAT_CODEC
    ];
    foreach readonly & ModelCodec codec in codecs {
        test:assertTrue(codec.streamDialect is StreamDialect, "every shipped codec must stream");
    }

    // Mantle reuses the vendors' own event models; only the framing differs.
    test:assertEquals(MANTLE_MESSAGES_CODEC.streamDialect, ANTHROPIC_STREAM);
    test:assertEquals(MANTLE_RESPONSES_CODEC.streamDialect, RESPONSES_STREAM);
    test:assertEquals(MANTLE_CHAT_CODEC.streamDialect, OPENAI_CHAT_STREAM);
    // Mistral chat streams through the OpenAI decoder despite its own codec: the
    // buffered dialects differ, the streamed chunk differs only in the stop-reason
    // spelling.
    test:assertEquals(INVOKE_MISTRAL_CHAT_CODEC.streamDialect, OPENAI_CHAT_STREAM);
    test:assertEquals(INVOKE_OPENAI_CHAT_CODEC.streamDialect, OPENAI_CHAT_STREAM);
    test:assertEquals(INVOKE_MISTRAL_TEXT_CODEC.streamDialect, TEXT_COMPLETION_STREAM);
    test:assertEquals(INVOKE_DEEPSEEK_CODEC.streamDialect, TEXT_COMPLETION_STREAM);
}

@test:Config {}
function testOnlyMantleCodecsAskForTheStreamInTheBody() {
    // Bedrock selects streaming by OPERATION on `bedrock-runtime` and by a BODY FLAG
    // on Mantle. A stray `"stream": true` sent to InvokeModelWithResponseStream is an
    // unknown inference parameter, and a missing one on Mantle silently returns the
    // whole answer in a single buffered chunk at the end.
    test:assertEquals(CONVERSE_CODEC.streamFields, ());
    test:assertEquals(INVOKE_ANTHROPIC_CODEC.streamFields, ());
    test:assertEquals(INVOKE_NOVA_CODEC.streamFields, ());
    test:assertEquals(INVOKE_OPENAI_CHAT_CODEC.streamFields, ());
    test:assertEquals(INVOKE_MISTRAL_TEXT_CODEC.streamFields, ());

    test:assertEquals(MANTLE_MESSAGES_CODEC.streamFields, {"stream": true});
    test:assertEquals(MANTLE_RESPONSES_CODEC.streamFields, {"stream": true});
    // Chat Completions reports NO usage on a stream unless asked.
    test:assertEquals(MANTLE_CHAT_CODEC.streamFields,
            {"stream": true, "stream_options": {"include_usage": true}});
}

@test:Config {}
function testStreamFieldsAreMergedIntoTheEncodedBody() returns error? {
    // The encoder is shared with the buffered path, so the flag has to be added
    // after encoding — without disturbing what the codec produced.
    json encoded = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100};
    json body = check withStreamFields(encoded, MANTLE_CHAT_CODEC.streamFields);
    map<json> merged = check body.ensureType();
    test:assertEquals(merged["stream"], true);
    test:assertEquals(merged["stream_options"], <json>{"include_usage": true});
    test:assertEquals(merged["max_tokens"], 100, "the encoded body must survive intact");

    // A runtime route's body is handed back untouched.
    test:assertEquals(check withStreamFields(encoded, ()), encoded);
}

@test:Config {}
function testEveryCodecsDialectMatchesItsDecoder() {
    // A dialect names the decoder that reads that codec's frames. If a codec is ever
    // given a dialect whose decoder cannot read its wire shape, the stream decodes to
    // silence rather than failing — so the pairing is pinned here.
    test:assertTrue(newStreamDecoder(<StreamDialect>CONVERSE_CODEC.streamDialect) is ConverseStreamDecoder);
    test:assertTrue(newStreamDecoder(<StreamDialect>INVOKE_NOVA_CODEC.streamDialect) is ConverseStreamDecoder);
    test:assertTrue(newStreamDecoder(<StreamDialect>INVOKE_ANTHROPIC_CODEC.streamDialect) is AnthropicStreamDecoder);
    // Mantle Messages rides the SAME decoder as Invoke-Anthropic — one event model,
    // two wire formats.
    test:assertTrue(newStreamDecoder(<StreamDialect>MANTLE_MESSAGES_CODEC.streamDialect) is AnthropicStreamDecoder);
    test:assertTrue(newStreamDecoder(<StreamDialect>MANTLE_CHAT_CODEC.streamDialect) is OpenAIChatStreamDecoder);
    test:assertTrue(newStreamDecoder(<StreamDialect>MANTLE_RESPONSES_CODEC.streamDialect) is ResponsesStreamDecoder);
    test:assertTrue(
            newStreamDecoder(<StreamDialect>INVOKE_MISTRAL_CHAT_CODEC.streamDialect) is OpenAIChatStreamDecoder);
    test:assertTrue(
            newStreamDecoder(<StreamDialect>INVOKE_MISTRAL_TEXT_CODEC.streamDialect) is TextCompletionStreamDecoder);
    test:assertTrue(
            newStreamDecoder(<StreamDialect>INVOKE_DEEPSEEK_CODEC.streamDialect) is TextCompletionStreamDecoder);
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

// ---------------------------------------------------------------------------
// OpenAI Chat Completions — Mantle Chat, and the OpenAI-shaped Invoke vendors
// ---------------------------------------------------------------------------

// This dialect names nothing: SSE sends bare `data:` lines and every Invoke frame is
// the constant `chunk`, so the decoder is driven with an empty event name.
const string NO_EVENT_NAME = "";

@test:Config {}
function testOpenAIChatStreamMapsDeltaContentAndIdentity() {
    OpenAIChatStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, NO_EVENT_NAME, {
        "id": "chatcmpl-1",
        "model": "zai.glm-5",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": null}]
    });
    test:assertEquals(deltaOf(chunk).content, "Hel");
    test:assertEquals(deltaOf(chunk).role, ai:ASSISTANT);
    // Unlike Converse, this dialect names itself — the iterator must not overwrite it.
    test:assertEquals(chunk.id, "chatcmpl-1");
    test:assertEquals(chunk.model, "zai.glm-5");
}

@test:Config {}
function testOpenAIChatStreamMapsReasoningToItsOwnField() {
    // DeepSeek V3.x and Qwen-with-thinking stream chain-of-thought in a separate
    // member; it must never merge into the answer text.
    OpenAIChatStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, NO_EVENT_NAME,
            {"choices": [{"delta": {"reasoning_content": "let me think"}}]});
    test:assertEquals(deltaOf(chunk).reasoning, "let me think");
    test:assertEquals(deltaOf(chunk).content, (), "reasoning must not leak into the answer text");
}

@test:Config {}
function testOpenAIChatStreamForwardsTheVendorsToolCallIndex() {
    // NO remap on this dialect: `tool_calls[].index` already numbers the TOOL CALLS
    // from 0 — it is the field `ai:ToolCallChunk.index` was modelled on. Renumbering
    // it would mis-order parallel calls.
    OpenAIChatStreamDecoder decoder = new;
    ai:ChatCompletionChunk first = decodeOne(decoder, NO_EVENT_NAME, {
        "choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_a", "type": "function", "function": {"name": "getWeather", "arguments": ""}},
            {"index": 1, "id": "call_b", "type": "function", "function": {"name": "getTime", "arguments": ""}}
        ]}}]
    });
    ai:ToolCallChunk[] calls = <ai:ToolCallChunk[]>deltaOf(first).toolCalls;
    test:assertEquals(calls.length(), 2);
    test:assertEquals(calls[0].index, 0);
    test:assertEquals(calls[0].id, "call_a");
    test:assertEquals(calls[0].'function?.name, "getWeather");
    test:assertEquals(calls[1].index, 1);
    test:assertEquals(calls[1].id, "call_b");

    // Argument fragments arrive later with the index and nothing else.
    ai:ChatCompletionChunk args = decodeOne(decoder, NO_EVENT_NAME,
            {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "{\"tz\":"}}]}}]});
    ai:ToolCallChunk[] fragment = <ai:ToolCallChunk[]>deltaOf(args).toolCalls;
    test:assertEquals(fragment[0].index, 1);
    test:assertEquals(fragment[0].'function?.arguments, "{\"tz\":");
}

@test:Config {}
function testOpenAIChatStreamReadsUsageOffTheFinalEmptyChoicesChunk() {
    // With `stream_options: {include_usage: true}` the last chunk carries usage and
    // an EMPTY choices array — so a length check alone would drop it.
    OpenAIChatStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, NO_EVENT_NAME, {
        "choices": [],
        "usage": {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33}
    });
    ai:CompletionTokenUsage usage = <ai:CompletionTokenUsage>chunk?.usage;
    test:assertEquals(usage.promptTokens, 11);
    test:assertEquals(usage.completionTokens, 22);
    test:assertEquals(usage.totalTokens, 33);
}

@test:Config {}
function testOpenAIChatStreamIgnoresTheNullUsageOnEveryOtherChunk() {
    // `"usage": null` rides every non-final chunk. Emitting a usage record of zeroes
    // for each one would overwrite the real totals on the span.
    OpenAIChatStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, NO_EVENT_NAME,
            {"choices": [{"delta": {"content": "x"}}], "usage": null});
    test:assertTrue(chunk?.usage is (), "a null usage must not become a usage record");
}

@test:Config {}
function testOpenAIChatStreamAcceptsMistralsStopReasonSpelling() {
    // Mistral spells the field `stop_reason` where OpenAI says `finish_reason`. That
    // one difference is why the two share this decoder instead of getting two.
    OpenAIChatStreamDecoder decoder = new;
    ai:ChatCompletionChunk mistral = decodeOne(decoder, NO_EVENT_NAME,
            {"choices": [{"message": {"content": "done"}, "stop_reason": "stop"}]});
    test:assertEquals(mistral.choices[0].finishReason, ai:STOP);
    // ... and Mistral reuses the BUFFERED `message` member on the stream, which is
    // why the decoder falls back to it.
    test:assertEquals(deltaOf(mistral).content, "done");
}

@test:Config {}
function testOpenAIChatStreamReadsBedrocksInvocationMetrics() {
    // On Invoke these vendors report no `usage` at all; Bedrock staples the counts
    // onto the last frame instead. Dropping them loses the only token accounting the
    // route ever produces.
    OpenAIChatStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, NO_EVENT_NAME, {
        "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
        "amazon-bedrock-invocationMetrics": {"inputTokenCount": 7, "outputTokenCount": 5}
    });
    ai:CompletionTokenUsage usage = <ai:CompletionTokenUsage>chunk?.usage;
    test:assertEquals(usage.promptTokens, 7);
    test:assertEquals(usage.completionTokens, 5);
    test:assertEquals(usage.totalTokens, 12, "the metrics block reports no total; it is derived");
}

@test:Config {}
function testOpenAIChatStreamSurfacesAnInBandError() {
    OpenAIChatStreamDecoder decoder = new;
    ai:ChatCompletionChunk|ai:Error? result = decoder.decode(NO_EVENT_NAME,
            {"error": {"type": "server_error", "message": "upstream failed"}});
    if result !is ai:Error {
        test:assertFail("an in-band error object must end the stream");
    }
    test:assertTrue(result.message().includes("upstream failed"), result.message());
}

@test:Config {}
function testOpenAIChatFinishReasonMapping() {
    test:assertEquals(mapOpenAIFinishReason("stop"), ai:STOP);
    test:assertEquals(mapOpenAIFinishReason("length"), ai:LENGTH);
    test:assertEquals(mapOpenAIFinishReason("tool_calls"), ai:TOOL_CALLS);
    // The pre-`tool_calls` spelling some vendors still emit.
    test:assertEquals(mapOpenAIFinishReason("function_call"), ai:TOOL_CALLS);
    test:assertEquals(mapOpenAIFinishReason("content_filter"), ai:CONTENT_FILTER);
    test:assertEquals(mapOpenAIFinishReason("something_new"), (), "an unknown reason is not guessed");
    test:assertEquals(mapOpenAIFinishReason(()), ());
}

// ---------------------------------------------------------------------------
// OpenAI Responses — Mantle only
// ---------------------------------------------------------------------------

@test:Config {}
function testResponsesStreamMapsCreatedAndTextDeltas() {
    ResponsesStreamDecoder decoder = new;
    ai:ChatCompletionChunk created = decodeOne(decoder, RESPONSES_EVT_CREATED,
            {"type": RESPONSES_EVT_CREATED, "response": {"id": "resp_1", "model": "openai.gpt-5.5"}});
    test:assertEquals(created.id, "resp_1");
    test:assertEquals(created.model, "openai.gpt-5.5");
    test:assertEquals(deltaOf(created).role, ai:ASSISTANT, "the role arrives once, on the opening event");

    ai:ChatCompletionChunk text = decodeOne(decoder, RESPONSES_EVT_OUTPUT_TEXT_DELTA,
            {"type": RESPONSES_EVT_OUTPUT_TEXT_DELTA, "output_index": 0, "delta": "Once upon"});
    test:assertEquals(deltaOf(text).content, "Once upon");
}

@test:Config {}
function testResponsesStreamMapsReasoningSummaryToReasoning() {
    // GPT-5.x returns a SUMMARY of its reasoning rather than the raw trace — readable
    // text either way, so it lands in `reasoning`, never in the answer.
    ResponsesStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, RESPONSES_EVT_REASONING_SUMMARY_DELTA,
            {"type": RESPONSES_EVT_REASONING_SUMMARY_DELTA, "delta": "weighing options"});
    test:assertEquals(deltaOf(chunk).reasoning, "weighing options");
    test:assertEquals(deltaOf(chunk).content, ());
}

@test:Config {}
function testResponsesStreamRenumbersToolCallsFromZero() {
    // `output_index` counts OUTPUT ITEMS — messages and reasoning items included — so
    // a tool call announced at item 1 must still be tool-call 0 for the caller.
    ResponsesStreamDecoder decoder = new;
    ai:ChatCompletionChunk added = decodeOne(decoder, RESPONSES_EVT_OUTPUT_ITEM_ADDED, {
        "type": RESPONSES_EVT_OUTPUT_ITEM_ADDED,
        "output_index": 1,
        "item": {"type": "function_call", "id": "fc_1", "call_id": "call_abc", "name": "getWeather"}
    });
    ai:ToolCallChunk[] calls = <ai:ToolCallChunk[]>deltaOf(added).toolCalls;
    test:assertEquals(calls[0].index, 0, "the first tool call is tool-call 0, whatever its item ordinal");
    test:assertEquals(calls[0].id, "call_abc", "a tool result is addressed to call_id, not to the item id");
    test:assertEquals(calls[0].'function?.name, "getWeather");

    // Arguments arrive on their own events, correlated by the same output_index.
    ai:ChatCompletionChunk args = decodeOne(decoder, RESPONSES_EVT_FUNCTION_ARGS_DELTA,
            {"type": RESPONSES_EVT_FUNCTION_ARGS_DELTA, "output_index": 1, "delta": "{\"city\":"});
    ai:ToolCallChunk[] fragment = <ai:ToolCallChunk[]>deltaOf(args).toolCalls;
    test:assertEquals(fragment[0].index, 0, "the fragment must join the call it belongs to");
    test:assertEquals(fragment[0].'function?.arguments, "{\"city\":");
}

@test:Config {}
function testResponsesStreamMapsTerminalStatusAndUsage() {
    ResponsesStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, RESPONSES_EVT_COMPLETED, {
        "type": RESPONSES_EVT_COMPLETED,
        "response": {
            "status": "completed",
            "usage": {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42}
        }
    });
    test:assertEquals(chunk.choices[0].finishReason, ai:STOP);
    ai:CompletionTokenUsage usage = <ai:CompletionTokenUsage>chunk?.usage;
    test:assertEquals(usage.promptTokens, 30);
    test:assertEquals(usage.completionTokens, 12);
    test:assertEquals(usage.totalTokens, 42);
}

@test:Config {}
function testResponsesStreamReportsToolCallsAsTheFinishReason() {
    // The Responses API has no `finish_reason`, only a lifecycle `status` — but an
    // agent loop reads TOOL_CALLS to decide whether to run a tool and continue, so a
    // completed response that produced one must not report a plain STOP.
    ResponsesStreamDecoder decoder = new;
    _ = decodeOne(decoder, RESPONSES_EVT_OUTPUT_ITEM_ADDED, {
        "type": RESPONSES_EVT_OUTPUT_ITEM_ADDED,
        "output_index": 0,
        "item": {"type": "function_call", "call_id": "call_1", "name": "lookup"}
    });
    ai:ChatCompletionChunk done = decodeOne(decoder, RESPONSES_EVT_COMPLETED,
            {"type": RESPONSES_EVT_COMPLETED, "response": {"status": "completed"}});
    test:assertEquals(done.choices[0].finishReason, ai:TOOL_CALLS);
}

@test:Config {}
function testResponsesStreamMapsAnIncompleteResponseToLength() {
    ResponsesStreamDecoder decoder = new;
    ai:ChatCompletionChunk truncated = decodeOne(decoder, RESPONSES_EVT_INCOMPLETE, {
        "type": RESPONSES_EVT_INCOMPLETE,
        "response": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    });
    test:assertEquals(truncated.choices[0].finishReason, ai:LENGTH);

    ResponsesStreamDecoder filtered = new;
    ai:ChatCompletionChunk blocked = decodeOne(filtered, RESPONSES_EVT_INCOMPLETE, {
        "type": RESPONSES_EVT_INCOMPLETE,
        "response": {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}}
    });
    test:assertEquals(blocked.choices[0].finishReason, ai:CONTENT_FILTER);
}

@test:Config {}
function testResponsesStreamSkipsTheRestatingLifecycleEvents() {
    // `.output_item.done`, `.output_text.done`, `.content_part.added` and friends
    // restate what a delta already delivered — emitting them would double the text.
    ResponsesStreamDecoder decoder = new;
    test:assertTrue(decoder.decode("response.in_progress", {"type": "response.in_progress"}) is ());
    test:assertTrue(decoder.decode("response.output_text.done",
            {"type": "response.output_text.done", "text": "the whole answer again"}) is ());
    test:assertTrue(decoder.decode("response.output_item.done", {"type": "response.output_item.done"}) is ());
}

@test:Config {}
function testResponsesStreamSurfacesAFailure() {
    ResponsesStreamDecoder decoder = new;
    ai:ChatCompletionChunk|ai:Error? result = decoder.decode(RESPONSES_EVT_FAILED, {
        "type": RESPONSES_EVT_FAILED,
        "response": {"status": "failed", "error": {"code": "server_error", "message": "model unavailable"}}
    });
    if result !is ai:Error {
        test:assertFail("a failed response must end the stream with an error");
    }
    test:assertTrue(result.message().includes("model unavailable"), result.message());
}

@test:Config {}
function testResponsesStreamFallsBackToTheEventNameWhenThePayloadIsUnnamed() {
    // The name arrives twice — the SSE `event:` line and the payload's `type`. The
    // payload wins, but a payload without one must still decode.
    ResponsesStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, RESPONSES_EVT_OUTPUT_TEXT_DELTA, {"delta": "hi"});
    test:assertEquals(deltaOf(chunk).content, "hi");
}

// ---------------------------------------------------------------------------
// Invoke text completion — Mistral text and DeepSeek R1
// ---------------------------------------------------------------------------

@test:Config {}
function testTextCompletionStreamReadsEitherVendorsArrayKey() {
    // One decoder, two dialects: Mistral wraps in `outputs`, DeepSeek in `choices`,
    // and on the stream nothing else tells them apart.
    TextCompletionStreamDecoder mistral = new;
    ai:ChatCompletionChunk fromOutputs = decodeOne(mistral, "chunk",
            {"outputs": [{"text": "Bon", "stop_reason": null}]});
    test:assertEquals(deltaOf(fromOutputs).content, "Bon");

    TextCompletionStreamDecoder deepseek = new;
    ai:ChatCompletionChunk fromChoices = decodeOne(deepseek, "chunk",
            {"choices": [{"text": "jour", "stop_reason": null}]});
    test:assertEquals(deltaOf(fromChoices).content, "jour");
}

@test:Config {}
function testTextCompletionStreamMapsStopReasonAndMetrics() {
    TextCompletionStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, "chunk", {
        "outputs": [{"text": "!", "stop_reason": "length"}],
        "amazon-bedrock-invocationMetrics": {"inputTokenCount": 4, "outputTokenCount": 96}
    });
    test:assertEquals(deltaOf(chunk).content, "!");
    test:assertEquals(chunk.choices[0].finishReason, ai:LENGTH);
    ai:CompletionTokenUsage usage = <ai:CompletionTokenUsage>chunk?.usage;
    test:assertEquals(usage.promptTokens, 4);
    test:assertEquals(usage.completionTokens, 96);
}

@test:Config {}
function testTextCompletionStreamEmitsAMetricsOnlyFrameAsUsage() {
    // The metrics can arrive on a frame of their own, after the last text.
    TextCompletionStreamDecoder decoder = new;
    ai:ChatCompletionChunk chunk = decodeOne(decoder, "chunk",
            {"amazon-bedrock-invocationMetrics": {"inputTokenCount": 3, "outputTokenCount": 9}});
    test:assertEquals(deltaOf(chunk).content, (), "a usage-only chunk carries no text");
    ai:CompletionTokenUsage usage = <ai:CompletionTokenUsage>chunk?.usage;
    test:assertEquals(usage.totalTokens, 12);
}

@test:Config {}
function testTextCompletionStreamSkipsAnEventItCannotRead() {
    // A frame with neither array nor metrics has nothing to surface. Skipping beats
    // erroring: the shape is derived from the buffered body, not quoted from a
    // streaming example, so an unexpected frame must not kill a good stream.
    TextCompletionStreamDecoder decoder = new;
    test:assertTrue(decoder.decode("chunk", {"something": "else"}) is ());
}

@test:Config {}
function testOpenAIChatStreamSkipsAnEventThatCarriesNothing() {
    // Some vendors interleave no-news objects between real chunks. Emitting them
    // hands the caller chunks with an empty delta to filter out of its own output —
    // the same reason `contentBlockStop` is skipped on Converse.
    OpenAIChatStreamDecoder decoder = new;
    test:assertTrue(decoder.decode(NO_EVENT_NAME, {"choices": [{"delta": {}}]}) is ());
    test:assertTrue(decoder.decode(NO_EVENT_NAME, {"id": "chatcmpl-1", "object": "chat.completion.chunk"}) is ());
    // ... but a chunk whose ONLY content is the finish reason still counts.
    ai:ChatCompletionChunk done = decodeOne(decoder, NO_EVENT_NAME,
            {"choices": [{"delta": {}, "finish_reason": "stop"}]});
    test:assertEquals(done.choices[0].finishReason, ai:STOP);
    // ... as does a role-only opener.
    ai:ChatCompletionChunk opener = decodeOne(decoder, NO_EVENT_NAME,
            {"choices": [{"delta": {"role": "assistant"}}]});
    test:assertEquals(deltaOf(opener).role, ai:ASSISTANT);
}
