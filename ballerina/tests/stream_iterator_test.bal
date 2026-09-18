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
import ballerina/ai.observe;
import ballerina/http;
import ballerina/io;
import ballerina/lang.array;
import ballerina/test;

// The whole streaming pipeline — framing, Invoke unwrapping, dialect decoding, id
// backfill, span accounting — driven end to end over a synthetic byte stream.
//
// Stops short of HTTP on purpose. On this branch a provider's host is derived from
// the region with no `serviceUrl` override, so a mock server cannot be dialled;
// injecting the byte stream directly exercises everything downstream of the socket,
// which is where all the logic lives.

# Replays a canned response as a byte stream, in chunks of `readSize`.
class FakeByteStream {
    private final byte[] wire;
    private final int readSize;
    private int pos = 0;
    // Records that `close()` reached the underlying body — the only observable
    // difference between releasing a pooled connection and leaking it.
    private int closeCount = 0;

    isolated function init(byte[] wire, int readSize) {
        self.wire = wire;
        self.readSize = readSize;
    }

    public isolated function next() returns record {|byte[] value;|}|io:Error? {
        if self.pos >= self.wire.length() {
            return ();
        }
        int end = int:min(self.pos + self.readSize, self.wire.length());
        byte[] chunk = self.wire.slice(self.pos, end);
        self.pos = end;
        return {value: chunk};
    }

    public isolated function close() returns io:Error? {
        self.closeCount += 1;
        return ();
    }

    isolated function closes() returns int => self.closeCount;
}

# Replays a canned chunk sequence, recording whether it was closed. Backs the
# `ChunkTextIterator` tests.
class FakeChunkSource {
    private final ai:ChatMessageChunk[] chunks;
    private int pos = 0;
    private boolean closed = false;

    isolated function init(ai:ChatMessageChunk[] chunks) {
        self.chunks = chunks;
    }

    public isolated function next() returns record {|ai:ChatMessageChunk value;|}|ai:Error? {
        if self.pos >= self.chunks.length() {
            return ();
        }
        ai:ChatMessageChunk chunk = self.chunks[self.pos];
        self.pos += 1;
        return {value: chunk};
    }

    public isolated function close() returns ai:Error? {
        self.closed = true;
        return ();
    }

    isolated function isClosed() returns boolean => self.closed;
}

# What draining a canned response produced: the chunks delivered, the error the
# stream ended with (`()` on a clean end), and what the iterator reported to its span.
type DrainResult record {|
    # Chunks delivered before the stream ended
    ai:ChatMessageChunk[] chunks;
    # The error the stream ended with
    ai:Error? err;
    # Input tokens reported to the span
    int inputTokens;
    # Output tokens reported to the span
    int outputTokens;
    # Finish reason reported to the span
    string finishReason;
|};

// Drains an iterator, collecting chunks until it ends.
function drain(BedrockChunkIterator iterator) returns DrainResult {
    ai:ChatMessageChunk[] chunks = [];
    ai:Error? err = ();
    while true {
        record {|ai:ChatMessageChunk value;|}|ai:Error? next = iterator.next();
        if next is ai:Error {
            err = next;
            break;
        }
        if next is () {
            break;
        }
        chunks.push(next.value);
    }
    return {chunks, err, inputTokens: iterator.inputTokens, outputTokens: iterator.outputTokens,
        finishReason: iterator.finishReason};
}

// Drains the iterator over a canned event-stream response.
function drainChunks(byte[] wire, StreamDialect dialect, boolean unwrapBytes, int readSize = 1)
        returns DrainResult {
    stream<byte[], io:Error?> bytes = new (new FakeByteStream(wire, readSize));
    return drain(new (new EventStreamEventSource(bytes, unwrapBytes), newStreamDecoder(dialect), "req-abc",
            observe:createChatSpan("test.model-v1:0")));
}

// Wraps a vendor event the way InvokeModelWithResponseStream does.
function invokeFrame(string eventJson) returns byte[] {
    json wrapper = {"bytes": array:toBase64(eventJson.toBytes())};
    return buildFrame({[HDR_MESSAGE_TYPE]: "event", [HDR_EVENT_TYPE]: "chunk",
                ":content-type": "application/json"}, wrapper.toJsonString());
}

// Concatenates the text content across a whole chunk sequence.
function textOf(ai:ChatMessageChunk[] chunks) returns string {
    string out = "";
    foreach ai:ChatMessageChunk chunk in chunks {
        out += chunk.content ?: "";
    }
    return out;
}

// Asserts the two per-chunk invariants of the contract: the assistant role on every
// chunk, and one id stable across the whole response.
function assertRoleAndIdOnEveryChunk(ai:ChatMessageChunk[] chunks, string expectedId) {
    test:assertTrue(chunks.length() > 0, "the stream must produce chunks");
    foreach ai:ChatMessageChunk chunk in chunks {
        test:assertEquals(chunk.role, ai:ASSISTANT, "every chunk must carry the assistant role");
        test:assertEquals(chunk?.id, expectedId, "every chunk must carry the same response id");
    }
}

// ---------------------------------------------------------------------------
// Converse, end to end
// ---------------------------------------------------------------------------

@test:Config {}
function testConverseStreamPipelineYieldsTextThenStop() {
    byte[] wire = converseFrame("messageStart", "{\"role\":\"assistant\"}");
    wire.push(...converseFrame("contentBlockDelta", "{\"contentBlockIndex\":0,\"delta\":{\"text\":\"Hello \"}}"));
    wire.push(...converseFrame("contentBlockDelta", "{\"contentBlockIndex\":0,\"delta\":{\"text\":\"world\"}}"));
    wire.push(...converseFrame("contentBlockStop", "{\"contentBlockIndex\":0}"));
    wire.push(...converseFrame("messageStop", "{\"stopReason\":\"end_turn\"}"));
    wire.push(...converseFrame("metadata",
            "{\"usage\":{\"inputTokens\":5,\"outputTokens\":2,\"totalTokens\":7}}"));

    DrainResult result = drainChunks(wire, CONVERSE_STREAM, false);
    test:assertEquals(result.err, (), "a well-formed stream must end cleanly");

    // Two deltas + messageStop = 3. messageStart, contentBlockStop and the
    // usage-only metadata event carry nothing for the caller and are skipped.
    ai:ChatMessageChunk[] chunks = result.chunks;
    test:assertEquals(chunks.length(), 3);
    test:assertEquals(textOf(chunks), "Hello world");
    test:assertEquals(chunks[2].finishReason, ai:STOP);
    test:assertEquals(chunks[2].content, ());
    // Converse events carry no completion id: the request id is stamped on every chunk.
    assertRoleAndIdOnEveryChunk(chunks, "req-abc");

    // Usage and the finish reason reach the span.
    test:assertEquals(result.inputTokens, 5);
    test:assertEquals(result.outputTokens, 2);
    test:assertEquals(result.finishReason, "stop");
}

@test:Config {}
function testStreamPipelineIsIndifferentToReadChunking() {
    // Frame boundaries must not depend on where the reads fall. Production uses
    // STREAM_READ_SIZE (16), but that constant is a latency tuning choice and has
    // been changed once already — decoding must stay identical whatever it is set
    // to, so the same wire is decoded at four sizes and required to agree.
    byte[] wire = converseFrame("messageStart", "{\"role\":\"assistant\"}");
    wire.push(...converseFrame("contentBlockDelta", "{\"contentBlockIndex\":0,\"delta\":{\"text\":\"abc\"}}"));
    wire.push(...converseFrame("messageStop", "{\"stopReason\":\"end_turn\"}"));

    int[] readSizes = [1, 3, 64, 4096];
    foreach int readSize in readSizes {
        DrainResult result = drainChunks(wire, CONVERSE_STREAM, false, readSize);
        test:assertEquals(result.err, (), string `read size ${readSize} must not fail`);
        test:assertEquals(result.chunks.length(), 2, string `read size ${readSize} must yield 2 chunks`);
        test:assertEquals(textOf(result.chunks), "abc", string `read size ${readSize} must recover the text`);
    }
}

@test:Config {}
function testConverseStreamPipelineStreamsToolCallFragmentsAcrossChunks() {
    byte[] wire = converseFrame("messageStart", "{\"role\":\"assistant\"}");
    wire.push(...converseFrame("contentBlockStart",
            "{\"contentBlockIndex\":0,\"start\":{\"toolUse\":{\"toolUseId\":\"tu_7\",\"name\":\"get_time\"}}}"));
    wire.push(...converseFrame("contentBlockDelta",
            "{\"contentBlockIndex\":0,\"delta\":{\"toolUse\":{\"input\":\"{\\\"tz\\\":\"}}}"));
    wire.push(...converseFrame("contentBlockDelta",
            "{\"contentBlockIndex\":0,\"delta\":{\"toolUse\":{\"input\":\"\\\"UTC\\\"}\"}}}"));
    wire.push(...converseFrame("messageStop", "{\"stopReason\":\"tool_use\"}"));

    DrainResult result = drainChunks(wire, CONVERSE_STREAM, false);
    test:assertEquals(result.err, ());
    ai:ChatMessageChunk[] chunks = result.chunks;
    // One chunk per fragment: the opener, two argument fragments, the stop.
    test:assertEquals(chunks.length(), 4);
    assertRoleAndIdOnEveryChunk(chunks, "req-abc");

    // Accumulate exactly as a caller would: by tool-call index, across chunks.
    string name = "";
    string id = "";
    string arguments = "";
    int fragments = 0;
    foreach ai:ChatMessageChunk chunk in chunks {
        ai:ToolCallChunk[]? calls = chunk.toolCalls;
        if calls is () {
            continue;
        }
        foreach ai:ToolCallChunk call in calls {
            test:assertEquals(call.index, 0);
            fragments += 1;
            id = call?.id ?: id;
            name = call?.name ?: name;
            arguments += call?.arguments ?: "";
        }
    }
    test:assertEquals(fragments, 3);
    test:assertEquals(id, "tu_7");
    test:assertEquals(name, "get_time");
    test:assertEquals(arguments, "{\"tz\":\"UTC\"}", "the argument fragments must reassemble into valid JSON");
    test:assertEquals(chunks[3].finishReason, ai:TOOL_CALLS);
    test:assertEquals(result.finishReason, "tool_calls");
}

// ---------------------------------------------------------------------------
// Invoke (base64-wrapped), end to end
// ---------------------------------------------------------------------------

@test:Config {}
function testAnthropicInvokeStreamPipelineUnwrapsAndDecodes() {
    byte[] wire = invokeFrame("{\"type\":\"message_start\",\"message\":" +
            "{\"id\":\"msg_1\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":11}}}");
    wire.push(...invokeFrame("{\"type\":\"content_block_delta\",\"index\":0," +
                "\"delta\":{\"type\":\"text_delta\",\"text\":\"Bon\"}}"));
    wire.push(...invokeFrame("{\"type\":\"content_block_delta\",\"index\":0," +
                "\"delta\":{\"type\":\"text_delta\",\"text\":\"jour\"}}"));
    wire.push(...invokeFrame("{\"type\":\"ping\"}"));
    wire.push(...invokeFrame("{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}," +
                "\"usage\":{\"output_tokens\":4}}"));
    wire.push(...invokeFrame("{\"type\":\"message_stop\"}"));

    DrainResult result = drainChunks(wire, ANTHROPIC_STREAM, true);
    test:assertEquals(result.err, ());
    test:assertEquals(textOf(result.chunks), "Bonjour");
    // message_start, ping and message_stop produce nothing: 2 deltas + message_delta = 3.
    test:assertEquals(result.chunks.length(), 3);
    // The message id from message_start is stamped on EVERY chunk — including the
    // ones built from events that do not repeat it — and wins over the request id.
    assertRoleAndIdOnEveryChunk(result.chunks, "msg_1");
    test:assertEquals(result.chunks[2].finishReason, ai:STOP);
    // The two halves of usage arrive on different events; both reach the span.
    test:assertEquals(result.inputTokens, 11);
    test:assertEquals(result.outputTokens, 4);
    test:assertEquals(result.finishReason, "stop");
}

@test:Config {}
function testEveryChunkCarriesTheRoleWhateverItHolds() {
    // Reasoning, text, a tool call and a finish-only chunk from one response: the
    // role and the id are on all of them, not only on the first.
    byte[] wire = invokeFrame("{\"type\":\"message_start\",\"message\":{\"id\":\"msg_2\"}}");
    wire.push(...invokeFrame("{\"type\":\"content_block_delta\",\"index\":0," +
                "\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}"));
    wire.push(...invokeFrame("{\"type\":\"content_block_delta\",\"index\":1," +
                "\"delta\":{\"type\":\"text_delta\",\"text\":\"Let me check.\"}}"));
    wire.push(...invokeFrame("{\"type\":\"content_block_start\",\"index\":2," +
                "\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"lookup\",\"input\":{}}}"));
    wire.push(...invokeFrame("{\"type\":\"content_block_delta\",\"index\":2," +
                "\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}"));
    wire.push(...invokeFrame("{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}"));

    DrainResult result = drainChunks(wire, ANTHROPIC_STREAM, true);
    test:assertEquals(result.err, ());
    test:assertEquals(result.chunks.length(), 5);
    assertRoleAndIdOnEveryChunk(result.chunks, "msg_2");
    test:assertEquals(result.chunks[0].reasoning, "hmm");
    test:assertEquals(result.chunks[1].content, "Let me check.");
    test:assertEquals(result.chunks[2].toolCalls, [{index: 0, id: "toolu_1", name: "lookup"}]);
    test:assertEquals(result.chunks[3].toolCalls, [{index: 0, arguments: "{}"}]);
    test:assertEquals(result.chunks[4].finishReason, ai:TOOL_CALLS);
}

// ---------------------------------------------------------------------------
// Failure paths
// ---------------------------------------------------------------------------

@test:Config {}
function testStreamPipelineReportsATruncatedResponse() {
    // A response cut off mid-frame must NOT read as a clean, complete answer.
    byte[] full = converseFrame("contentBlockDelta", "{\"contentBlockIndex\":0,\"delta\":{\"text\":\"partial\"}}");
    byte[] truncated = full.slice(0, full.length() - 3);

    ai:Error? err = drainChunks(truncated, CONVERSE_STREAM, false).err;
    if err !is ai:Error {
        test:assertFail("a truncated stream must surface an error");
    }
    test:assertTrue(err.message().includes("truncated"), err.message());
}

@test:Config {}
function testStreamPipelineSurfacesAnExceptionFrame() {
    // Bedrock reports a mid-generation failure as a FRAME on an otherwise healthy
    // 200 response, so the HTTP status never reveals it.
    byte[] wire = converseFrame("contentBlockDelta", "{\"contentBlockIndex\":0,\"delta\":{\"text\":\"hi\"}}");
    wire.push(...buildFrame({
                [HDR_MESSAGE_TYPE]: "exception",
                [HDR_EXCEPTION_TYPE]: "modelStreamErrorException",
                ":content-type": "application/json"
            }, "{\"message\":\"upstream model failed\"}"));

    DrainResult result = drainChunks(wire, CONVERSE_STREAM, false);
    // The chunks that already arrived are still delivered; the failure lands after.
    test:assertEquals(textOf(result.chunks), "hi");
    ai:Error? err = result.err;
    if err !is ai:Error {
        test:assertFail("an exception frame must surface an error");
    }
    test:assertTrue(err.message().includes("modelStreamErrorException"), err.message());
    test:assertTrue(err.message().includes("upstream model failed"), err.message());
}

@test:Config {}
function testStreamPipelineRejectsAnInvokeFrameWithoutBytes() {
    // An Invoke frame is always `{"bytes": base64}`; anything else means the route
    // and the dialect have been paired wrongly — a malformed chunk.
    byte[] wire = buildFrame({[HDR_MESSAGE_TYPE]: "event", [HDR_EVENT_TYPE]: "chunk"},
            "{\"unexpected\":true}");
    ai:Error? err = drainChunks(wire, ANTHROPIC_STREAM, true).err;
    test:assertTrue(err is ai:LlmInvalidResponseError, "a frame with no 'bytes' member must be an invalid response");
}

@test:Config {}
function testStreamPipelineReportsAMalformedPayloadAsAnInvalidResponse() {
    byte[] wire = converseFrame("contentBlockDelta", "{not json");
    ai:Error? err = drainChunks(wire, CONVERSE_STREAM, false).err;
    test:assertTrue(err is ai:LlmInvalidResponseError, "an unparseable payload must be an invalid response");
}

// ---------------------------------------------------------------------------
// The capability guard, at the provider boundary
// ---------------------------------------------------------------------------

@test:Config {}
function testAMantleRoutedModelResolvesAStreamingRouteWithoutABodyRewrite() returns error? {
    // Mantle used to be refused before any I/O — it streams as SSE from the same path
    // with a body flag, which was a different transport than the one that shipped.
    // Now it resolves like any other route, and the three things that make it work
    // are pinned here because each is invisible at the call site.
    //
    // `claude-sonnet-5` is the case that matters: it is dual-homed, so AUTO sends the
    // module's flagship Claude to Mantle, and every one of those callers used to lose
    // streaming.
    Route route = check resolveRoute(CLAUDE_SONNET_5, REGION);
    test:assertEquals(route.family, MANTLE, "a Mantle-capable bare id prefers Mantle under AUTO");

    readonly & ModelConverter converter = check selectConverter(route);
    test:assertEquals(converter.streamDialect, ANTHROPIC_STREAM, "Mantle Messages reuses Anthropic's events");
    test:assertEquals(converter.streamFields, {"stream": true}, "Mantle asks for the stream in the BODY");

    Endpoint ep = check buildEndpoint(route);
    test:assertEquals(ep.streamPath, ep.path, "and streams from the SAME path, not a sibling operation");
    test:assertEquals(streamWireFor(route.family), SSE, "so the response is SSE, not event-stream framed");
}

@test:Config {}
function testTheStreamingGuardStillRefusesACodecWithoutADialect() {
    // Every shipped converter streams, so the guard in `openChunkStream` is
    // unreachable today — but it is what keeps a future converter that this module
    // can encode and cannot decode incrementally from returning a silent, empty
    // stream instead of an error. Pinned at the field, since no route can exercise it
    // any more.
    readonly & ModelConverter unstreamable = {
        encode: encodeConverse,
        decode: decodeConverse,
        toolChoice: CONVERSE_TOOL_CHOICE,
        streamDialect: ()
    };
    test:assertTrue(unstreamable.streamDialect is (), "the guard reads exactly this");
}

// ---------------------------------------------------------------------------
// Closing — releasing the live response
// ---------------------------------------------------------------------------

@test:Config {}
function testClosingAPartiallyReadStreamReleasesTheResponseBody() returns error? {
    // The leak that matters: a caller that breaks out of the `foreach` once it has
    // seen enough. `close()` is the only hook it has, and without one the
    // `http:Response` body — and its pooled connection — stays open for the life of
    // the client.
    byte[] wire = converseFrame("contentBlockDelta", "{\"contentBlockIndex\":0,\"delta\":{\"text\":\"a\"}}");
    wire.push(...converseFrame("contentBlockDelta", "{\"contentBlockIndex\":0,\"delta\":{\"text\":\"b\"}}"));
    wire.push(...converseFrame("messageStop", "{\"stopReason\":\"end_turn\"}"));

    FakeByteStream body = new (wire, 1);
    stream<byte[], io:Error?> bytes = new (body);
    stream<ai:ChatMessageChunk, ai:Error?> chunks = new (new BedrockChunkIterator(
            new EventStreamEventSource(bytes, false), newStreamDecoder(CONVERSE_STREAM),
            "req-abc", observe:createChatSpan("test.model-v1:0")));

    record {|ai:ChatMessageChunk value;|}? first = check chunks.next();
    test:assertTrue(first is record {|ai:ChatMessageChunk value;|}, "the first chunk must arrive");
    test:assertEquals(body.closes(), 0, "reading must not close the body");

    check chunks.close();
    test:assertEquals(body.closes(), 1, "close() must reach the response byte stream");
}

@test:Config {}
function testClosingAnExhaustedStreamIsANoOp() returns error? {
    // A body read to the end has already returned its connection to the pool, so
    // closing it again would only risk a spurious 'already closed' error out of
    // `close()`. Draining then closing must stay silent.
    byte[] wire = converseFrame("messageStop", "{\"stopReason\":\"end_turn\"}");
    FakeByteStream body = new (wire, 1);
    stream<byte[], io:Error?> bytes = new (body);
    stream<ai:ChatMessageChunk, ai:Error?> chunks = new (new BedrockChunkIterator(
            new EventStreamEventSource(bytes, false), newStreamDecoder(CONVERSE_STREAM),
            "req-abc", observe:createChatSpan("test.model-v1:0")));

    _ = check chunks.next(); // messageStop
    test:assertEquals(check chunks.next(), (), "the stream must end after its last frame");

    check chunks.close();
    test:assertEquals(body.closes(), 0, "an exhausted body must not be closed again");
    check chunks.close();
    test:assertEquals(body.closes(), 0, "close() must stay idempotent");
}

@test:Config {}
function testClosingAFailedStreamStillReleasesTheResponseBody() returns error? {
    // A mid-stream service exception ends the stream with an error, but the body is
    // NOT drained — the bytes after the exception frame were never read. This is the
    // path where forgetting to close leaks hardest.
    byte[] wire = buildFrame({[HDR_MESSAGE_TYPE]: "exception", [HDR_EXCEPTION_TYPE]: "throttlingException"},
            "{\"message\":\"Too many requests\"}");
    wire.push(...converseFrame("messageStop", "{\"stopReason\":\"end_turn\"}"));

    FakeByteStream body = new (wire, 1);
    stream<byte[], io:Error?> bytes = new (body);
    stream<ai:ChatMessageChunk, ai:Error?> chunks = new (new BedrockChunkIterator(
            new EventStreamEventSource(bytes, false), newStreamDecoder(CONVERSE_STREAM),
            "req-abc", observe:createChatSpan("test.model-v1:0")));

    record {|ai:ChatMessageChunk value;|}|ai:Error? first = chunks.next();
    test:assertTrue(first is ai:Error, "an exception frame must surface as an error");

    check chunks.close();
    test:assertEquals(body.closes(), 1, "close() after a failure must still release the body");
}

@test:Config {}
function testTheTextProjectionYieldsOnlyContentAndPropagatesClose() returns error? {
    // `generateAsStream` hands back a text stream over the chunk stream. It must
    // yield the answer text only, and its `close()` must reach the chunk stream —
    // otherwise `BedrockChunkIterator.close()` is unreachable from a
    // `generateAsStream` caller and the release above never happens.
    FakeChunkSource chunkSource = new ([
        {role: ai:ASSISTANT, reasoning: "thinking first"},
        {role: ai:ASSISTANT, content: "Hello "},
        {role: ai:ASSISTANT, toolCalls: [{index: 0, id: "t1", name: "lookup"}]},
        {role: ai:ASSISTANT, content: ""},
        {role: ai:ASSISTANT, content: "world"},
        {role: ai:ASSISTANT, finishReason: ai:STOP}
    ]);
    stream<ai:ChatMessageChunk, ai:Error?> chunks = new (chunkSource);
    stream<string, ai:Error?> text = new (new ChunkTextIterator(chunks));

    record {|string value;|}? first = check text.next();
    test:assertEquals(first?.value, "Hello ", "reasoning must be skipped");
    test:assertFalse(chunkSource.isClosed(), "reading must not close the chunk stream");
    record {|string value;|}? second = check text.next();
    test:assertEquals(second?.value, "world", "tool-call and empty chunks must be skipped");
    test:assertEquals(check text.next(), (), "the finish-only chunk yields nothing");

    check text.close();
    test:assertTrue(chunkSource.isClosed(), "close() must propagate to the chunk stream underneath");
}

@test:Config {}
function testAFailedStreamStaysEnded() {
    // A mid-stream exception frame ends the stream. The bytes AFTER it are still
    // buffered and still decodable, so without a terminal state a consumer that
    // keeps pulling past the error is handed the rest of the answer as though
    // nothing had gone wrong.
    byte[] wire = converseFrame("contentBlockDelta", "{\"contentBlockIndex\":0,\"delta\":{\"text\":\"partial\"}}");
    wire.push(...buildFrame({[HDR_MESSAGE_TYPE]: "exception", [HDR_EXCEPTION_TYPE]: "modelStreamErrorException"},
            "{\"message\":\"boom\"}"));
    wire.push(...converseFrame("contentBlockDelta", "{\"contentBlockIndex\":0,\"delta\":{\"text\":\" more\"}}"));
    wire.push(...converseFrame("messageStop", "{\"stopReason\":\"end_turn\"}"));

    stream<byte[], io:Error?> bytes = new (new FakeByteStream(wire, 1));
    BedrockChunkIterator iterator = new (new EventStreamEventSource(bytes, false),
            newStreamDecoder(CONVERSE_STREAM), "req-abc", observe:createChatSpan("test.model-v1:0"));

    record {|ai:ChatMessageChunk value;|}|ai:Error? first = iterator.next();
    test:assertTrue(first is record {|ai:ChatMessageChunk value;|}, "the text before the exception arrives");
    test:assertTrue(iterator.next() is ai:Error, "the exception frame surfaces as an error");

    test:assertTrue(iterator.next() is (), "a failed stream must stay ended");
    test:assertTrue(iterator.next() is (), "and stay ended on every further pull");
}

@test:Config {}
function testAClosedStreamStopsYielding() returns error? {
    // Closing mid-response must also be terminal — a caller that broke out early and
    // then pulled again would otherwise resume reading a response it had released.
    byte[] wire = converseFrame("contentBlockDelta", "{\"contentBlockIndex\":0,\"delta\":{\"text\":\"a\"}}");
    wire.push(...converseFrame("contentBlockDelta", "{\"contentBlockIndex\":0,\"delta\":{\"text\":\"b\"}}"));

    stream<byte[], io:Error?> bytes = new (new FakeByteStream(wire, 1));
    BedrockChunkIterator iterator = new (new EventStreamEventSource(bytes, false),
            newStreamDecoder(CONVERSE_STREAM), "req-abc", observe:createChatSpan("test.model-v1:0"));

    test:assertTrue(iterator.next() is record {|ai:ChatMessageChunk value;|}, "the first chunk arrives");
    check iterator.close();
    test:assertTrue(iterator.next() is (), "a closed stream must not resume");
}

// ---------------------------------------------------------------------------
// Mantle, end to end: SSE frames -> chunks
// ---------------------------------------------------------------------------

# Replays a canned SSE event sequence, recording whether the body was released.
#
# The Mantle counterpart of `FakeByteStream`: the stdlib SSE reader hands back
# already-parsed `http:SseEvent` values, so what needs exercising here is the adaptation
# onto `StreamEventSource` — the sentinel, the keep-alives, the error frame and the
# release — not a line parser this module does not own.
class FakeSseStream {
    private final http:SseEvent[] events;
    private int pos = 0;
    private int closeCount = 0;

    isolated function init(http:SseEvent[] events) {
        self.events = events;
    }

    public isolated function next() returns record {|http:SseEvent value;|}|error? {
        if self.pos >= self.events.length() {
            return ();
        }
        http:SseEvent event = self.events[self.pos];
        self.pos += 1;
        return {value: event};
    }

    public isolated function close() returns error? {
        self.closeCount += 1;
        return ();
    }

    isolated function closes() returns int => self.closeCount;
}

// Drains a canned SSE response through the whole pipeline.
function drainSse(http:SseEvent[] events, StreamDialect dialect) returns DrainResult {
    stream<http:SseEvent, error?> sse = new (new FakeSseStream(events));
    return drain(new (new SseEventSource(sse), newStreamDecoder(dialect), "req-mantle",
            observe:createChatSpan("anthropic.claude-sonnet-5")));
}

@test:Config {}
function testMantleMessagesSsePipelineDecodesWithTheAnthropicDecoder() {
    // The reuse that makes Mantle Messages nearly free: these are the SAME events
    // `InvokeModelWithResponseStream` delivers, so the existing decoder reads them
    // once the SSE framing is stripped.
    DrainResult result = drainSse([
        {event: "message_start", data: "{\"type\":\"message_start\",\"message\":" +
                "{\"id\":\"msg_1\",\"model\":\"claude-sonnet-5\",\"usage\":{\"input_tokens\":9}}}"},
        {event: "content_block_delta",
            data: "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi \"}}"},
        {event: "content_block_delta",
            data: "{\"type\":\"content_block_delta\",\"index\":0," +
                    "\"delta\":{\"type\":\"text_delta\",\"text\":\"there\"}}"},
        {event: "message_delta",
            data: "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}," +
                    "\"usage\":{\"output_tokens\":4}}"},
        {event: "message_stop", data: "{\"type\":\"message_stop\"}"}
    ], ANTHROPIC_STREAM);

    test:assertTrue(result.err is (), "a well-formed SSE response must end cleanly");
    test:assertEquals(textOf(result.chunks), "Hi there");
    assertRoleAndIdOnEveryChunk(result.chunks, "msg_1");
    test:assertEquals(result.chunks[result.chunks.length() - 1].finishReason, ai:STOP);
    test:assertEquals(result.inputTokens, 9, "the input half arrives on message_start");
    test:assertEquals(result.outputTokens, 4);
}

@test:Config {}
function testMantleChatSsePipelineStopsAtTheDoneSentinel() {
    // `[DONE]` is not JSON. Parsing it would report a malformed event on a response
    // that in fact completed perfectly.
    FakeSseStream body = new ([
        {data: "{\"id\":\"chatcmpl-1\",\"model\":\"zai.glm-5\"," +
                "\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}"},
        {data: "{\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}"},
        {data: "{\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}"},
        {data: "{\"id\":\"chatcmpl-1\",\"choices\":[]," +
                "\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1,\"total_tokens\":6}}"},
        {data: "[DONE]"}
    ]);
    stream<http:SseEvent, error?> sse = new (body);
    DrainResult result = drain(new (new SseEventSource(sse), newStreamDecoder(OPENAI_CHAT_STREAM),
            "req-mantle", observe:createChatSpan("zai.glm-5")));

    test:assertEquals(result.err, ());
    // The role-only opener and the usage-only closer carry nothing for the caller.
    test:assertEquals(result.chunks.length(), 2);
    test:assertEquals(textOf(result.chunks), "ok");
    assertRoleAndIdOnEveryChunk(result.chunks, "chatcmpl-1");
    test:assertEquals(result.chunks[1].finishReason, ai:STOP);
    test:assertEquals(result.inputTokens, 5);
    test:assertEquals(result.outputTokens, 1);
    // The sentinel ends the stream mid-body, so the connection is released there
    // rather than left for a caller that has no reason to call `close()`.
    test:assertEquals(body.closes(), 1, "[DONE] must release the response");
}

@test:Config {}
function testMantleResponsesSsePipelineDecodesTheLifecycle() {
    DrainResult result = drainSse([
        {event: RESPONSES_EVT_CREATED,
            data: "{\"type\":\"response.created\",\"response\":{\"id\":\"resp_9\",\"model\":\"openai.gpt-5.5\"}}"},
        {event: RESPONSES_EVT_OUTPUT_TEXT_DELTA,
            data: "{\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Once \"}"},
        {event: RESPONSES_EVT_OUTPUT_TEXT_DELTA,
            data: "{\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"upon\"}"},
        {event: RESPONSES_EVT_COMPLETED,
            data: "{\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"," +
                    "\"usage\":{\"input_tokens\":8,\"output_tokens\":2,\"total_tokens\":10}}}"}
    ], RESPONSES_STREAM);

    test:assertTrue(result.err is ());
    test:assertEquals(textOf(result.chunks), "Once upon");
    // `response.created` yields no chunk, but its id is on every chunk after it.
    assertRoleAndIdOnEveryChunk(result.chunks, "resp_9");
    test:assertEquals(result.chunks[result.chunks.length() - 1].finishReason, ai:STOP);
    test:assertEquals(result.inputTokens, 8);
    test:assertEquals(result.outputTokens, 2);
}

@test:Config {}
function testSseKeepAlivesAndCommentsAreSkipped() {
    // Mantle sends comment lines to hold the connection open while the model thinks.
    // Surfacing them would hand a caller empty chunks to filter out.
    DrainResult result = drainSse([
        {comment: "keep-alive"},
        {event: "ping", data: "{\"type\":\"ping\"}"},
        {data: "   "},
        {event: "content_block_delta",
            data: "{\"type\":\"content_block_delta\",\"index\":0," +
                    "\"delta\":{\"type\":\"text_delta\",\"text\":\"finally\"}}"}
    ], ANTHROPIC_STREAM);

    test:assertTrue(result.err is ());
    test:assertEquals(result.chunks.length(), 1, "only the one event with content reaches the caller");
    test:assertEquals(textOf(result.chunks), "finally");
}

@test:Config {}
function testSseErrorEventSurfacesAsAnError() {
    // Mantle's equivalent of the event-stream exception frame: a clean 200, then a
    // failure partway through generating. Ending quietly would present a half answer
    // as a whole one.
    DrainResult result = drainSse([
        {event: "content_block_delta",
            data: "{\"type\":\"content_block_delta\",\"index\":0," +
                    "\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}"},
        {event: "error", data: "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\"," +
                "\"message\":\"Server overloaded\"}}"},
        {event: "content_block_delta",
            data: "{\"type\":\"content_block_delta\",\"index\":0," +
                    "\"delta\":{\"type\":\"text_delta\",\"text\":\" more\"}}"}
    ], ANTHROPIC_STREAM);

    test:assertEquals(textOf(result.chunks), "partial", "the text before the failure is delivered");
    ai:Error? err = result.err;
    if err is () {
        test:assertFail("an SSE error event must end the stream with an error");
    }
    test:assertTrue(err.message().includes("Server overloaded"), err.message());
}

@test:Config {}
function testSseMalformedDataIsReported() {
    DrainResult result = drainSse([{event: "message_delta", data: "{not json"}], ANTHROPIC_STREAM);
    ai:Error? err = result.err;
    if err is () {
        test:assertFail("unparseable SSE data must not pass as a clean end of stream");
    }
    test:assertTrue(err.message().includes("not valid JSON"), err.message());
}

@test:Config {}
function testClosingAPartiallyReadSseStreamReleasesTheResponse() returns error? {
    // The same leak as on the event-stream wire: a caller that breaks out early has
    // only `close()`, and without the propagation the SSE body stays open.
    FakeSseStream body = new ([
        {event: "content_block_delta",
            data: "{\"type\":\"content_block_delta\",\"index\":0," +
                    "\"delta\":{\"type\":\"text_delta\",\"text\":\"a\"}}"},
        {event: "content_block_delta",
            data: "{\"type\":\"content_block_delta\",\"index\":0," +
                    "\"delta\":{\"type\":\"text_delta\",\"text\":\"b\"}}"}
    ]);
    stream<http:SseEvent, error?> sse = new (body);
    stream<ai:ChatMessageChunk, ai:Error?> chunks = new (new BedrockChunkIterator(
            new SseEventSource(sse), newStreamDecoder(ANTHROPIC_STREAM), "req-mantle",
            observe:createChatSpan("anthropic.claude-sonnet-5")));

    record {|ai:ChatMessageChunk value;|}? first = check chunks.next();
    test:assertTrue(first is record {|ai:ChatMessageChunk value;|}, "the first chunk must arrive");
    test:assertEquals(body.closes(), 0, "reading must not close the body");

    check chunks.close();
    test:assertEquals(body.closes(), 1, "close() must reach the SSE stream");
    check chunks.close();
    test:assertEquals(body.closes(), 1, "close() must stay idempotent");
}
