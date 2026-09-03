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
# `ChunkTextIterator` propagation test.
class FakeChunkSource {
    private final ai:ChatCompletionChunk[] chunks;
    private int pos = 0;
    private boolean closed = false;

    isolated function init(ai:ChatCompletionChunk[] chunks) {
        self.chunks = chunks;
    }

    public isolated function next() returns record {|ai:ChatCompletionChunk value;|}|ai:Error? {
        if self.pos >= self.chunks.length() {
            return ();
        }
        ai:ChatCompletionChunk chunk = self.chunks[self.pos];
        self.pos += 1;
        return {value: chunk};
    }

    public isolated function close() returns ai:Error? {
        self.closed = true;
        return ();
    }

    isolated function isClosed() returns boolean => self.closed;
}

# What draining a canned response produced: the chunks delivered, and the error the
# stream ended with (`()` on a clean end).
type DrainResult record {|
    ai:ChatCompletionChunk[] chunks;
    ai:Error? err;
|};

// Drains the iterator over a canned response, collecting chunks until it ends.
function drainChunks(byte[] wire, StreamDialect dialect, boolean unwrapBytes, int readSize = 1)
        returns DrainResult {
    stream<byte[], io:Error?> bytes = new (new FakeByteStream(wire, readSize));
    BedrockChunkIterator iterator = new (new EventStreamEventSource(bytes, unwrapBytes),
            newStreamDecoder(dialect), "req-abc", "test.model-v1:0",
            observe:createChatSpan("test.model-v1:0"));
    ai:ChatCompletionChunk[] chunks = [];
    while true {
        record {|ai:ChatCompletionChunk value;|}|ai:Error? next = iterator.next();
        if next is () {
            return {chunks, err: ()};
        }
        if next is ai:Error {
            return {chunks, err: next};
        }
        chunks.push(next.value);
    }
}

// Wraps a vendor event the way InvokeModelWithResponseStream does.
function invokeFrame(string eventJson) returns byte[] {
    json wrapper = {"bytes": array:toBase64(eventJson.toBytes())};
    return buildFrame({[HDR_MESSAGE_TYPE]: "event", [HDR_EVENT_TYPE]: "chunk",
                ":content-type": "application/json"}, wrapper.toJsonString());
}

// Concatenates the text content across a whole chunk sequence.
function textOf(ai:ChatCompletionChunk[] chunks) returns string {
    string out = "";
    foreach ai:ChatCompletionChunk chunk in chunks {
        foreach ai:ChatCompletionChunkChoice choice in chunk.choices {
            out += choice.delta.content ?: "";
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Converse, end to end
// ---------------------------------------------------------------------------

@test:Config {}
function testConverseStreamPipelineYieldsTextThenStopThenUsage() {
    byte[] wire = converseFrame("messageStart", "{\"role\":\"assistant\"}");
    wire.push(...converseFrame("contentBlockDelta", "{\"contentBlockIndex\":0,\"delta\":{\"text\":\"Hello \"}}"));
    wire.push(...converseFrame("contentBlockDelta", "{\"contentBlockIndex\":0,\"delta\":{\"text\":\"world\"}}"));
    wire.push(...converseFrame("contentBlockStop", "{\"contentBlockIndex\":0}"));
    wire.push(...converseFrame("messageStop", "{\"stopReason\":\"end_turn\"}"));
    wire.push(...converseFrame("metadata",
            "{\"usage\":{\"inputTokens\":5,\"outputTokens\":2,\"totalTokens\":7}}"));

    DrainResult result = drainChunks(wire, CONVERSE_STREAM, false);
    ai:ChatCompletionChunk[] chunks = result.chunks;
    ai:Error? err = result.err;
    test:assertEquals(err, (), "a well-formed stream must end cleanly");

    // messageStart + two deltas + messageStop + metadata = 5. contentBlockStop is
    // skipped rather than surfaced as an empty chunk.
    test:assertEquals(chunks.length(), 5);
    test:assertEquals(textOf(chunks), "Hello world");
    test:assertEquals(chunks[0].choices[0].delta.role, ai:ASSISTANT);
    test:assertEquals(chunks[3].choices[0].finishReason, ai:STOP);
    test:assertEquals((<ai:CompletionTokenUsage>chunks[4]?.usage)?.totalTokens, 7);
}

@test:Config {}
function testStreamPipelineBackfillsIdAndModel() {
    // Converse events carry no completion id; the request id is the only value
    // stable across every chunk, which is what the contract asks `id` to be.
    byte[] wire = converseFrame("contentBlockDelta", "{\"contentBlockIndex\":0,\"delta\":{\"text\":\"x\"}}");
    ai:ChatCompletionChunk[] chunks = drainChunks(wire, CONVERSE_STREAM, false).chunks;
    test:assertEquals(chunks[0].id, "req-abc");
    test:assertEquals(chunks[0].model, "test.model-v1:0");
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
        ai:ChatCompletionChunk[] chunks = result.chunks;
        ai:Error? err = result.err;
        test:assertEquals(err, (), string `read size ${readSize} must not fail`);
        test:assertEquals(chunks.length(), 3, string `read size ${readSize} must yield 3 chunks`);
        test:assertEquals(textOf(chunks), "abc", string `read size ${readSize} must recover the text`);
    }
}

@test:Config {}
function testConverseStreamPipelineReassemblesToolCallArguments() {
    byte[] wire = converseFrame("messageStart", "{\"role\":\"assistant\"}");
    wire.push(...converseFrame("contentBlockStart",
            "{\"contentBlockIndex\":0,\"start\":{\"toolUse\":{\"toolUseId\":\"tu_7\",\"name\":\"get_time\"}}}"));
    wire.push(...converseFrame("contentBlockDelta",
            "{\"contentBlockIndex\":0,\"delta\":{\"toolUse\":{\"input\":\"{\\\"tz\\\":\"}}}"));
    wire.push(...converseFrame("contentBlockDelta",
            "{\"contentBlockIndex\":0,\"delta\":{\"toolUse\":{\"input\":\"\\\"UTC\\\"}\"}}}"));
    wire.push(...converseFrame("messageStop", "{\"stopReason\":\"tool_use\"}"));

    DrainResult result = drainChunks(wire, CONVERSE_STREAM, false);
    ai:ChatCompletionChunk[] chunks = result.chunks;
    ai:Error? err = result.err;
    test:assertEquals(err, ());

    // Accumulate exactly as a caller would: by tool-call index, across chunks.
    string name = "";
    string id = "";
    string arguments = "";
    foreach ai:ChatCompletionChunk chunk in chunks {
        ai:ToolCallChunk[]? calls = chunk.choices[0].delta.toolCalls;
        if calls is () {
            continue;
        }
        foreach ai:ToolCallChunk call in calls {
            test:assertEquals(call.index, 0);
            id = call?.id ?: id;
            name = call?.'function?.name ?: name;
            arguments += call?.'function?.arguments ?: "";
        }
    }
    test:assertEquals(id, "tu_7");
    test:assertEquals(name, "get_time");
    test:assertEquals(arguments, "{\"tz\":\"UTC\"}", "the argument fragments must reassemble into valid JSON");
    test:assertEquals(chunks[chunks.length() - 1].choices[0].finishReason, ai:TOOL_CALLS);
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

    DrainResult result = drainChunks(wire, ANTHROPIC_STREAM, true);
    ai:ChatCompletionChunk[] chunks = result.chunks;
    ai:Error? err = result.err;
    test:assertEquals(err, ());
    test:assertEquals(textOf(chunks), "Bonjour");
    // `ping` produced nothing: message_start + 2 deltas + message_delta = 4.
    test:assertEquals(chunks.length(), 4);
    // The dialect supplied its own id, so the request-id backfill must not override it.
    test:assertEquals(chunks[0].id, "msg_1");
    test:assertEquals(chunks[0].model, "claude-x");
    test:assertEquals(chunks[3].choices[0].finishReason, ai:STOP);
    // Both halves of usage land together on the final chunk, with the derived total.
    ai:CompletionTokenUsage usage = <ai:CompletionTokenUsage>chunks[3]?.usage;
    test:assertEquals(usage?.promptTokens, 11);
    test:assertEquals(usage?.completionTokens, 4);
    test:assertEquals(usage?.totalTokens, 15);
    test:assertEquals(chunks[0]?.usage, (), "the opening chunk carries no partial usage");
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
    ai:ChatCompletionChunk[] chunks = result.chunks;
    ai:Error? err = result.err;
    // The chunks that already arrived are still delivered; the failure lands after.
    test:assertEquals(textOf(chunks), "hi");
    if err !is ai:Error {
        test:assertFail("an exception frame must surface an error");
    }
    test:assertTrue(err.message().includes("modelStreamErrorException"), err.message());
    test:assertTrue(err.message().includes("upstream model failed"), err.message());
}

@test:Config {}
function testStreamPipelineRejectsAnInvokeFrameWithoutBytes() {
    // An Invoke frame is always `{"bytes": base64}`; anything else means the route
    // and the dialect have been paired wrongly.
    byte[] wire = buildFrame({[HDR_MESSAGE_TYPE]: "event", [HDR_EVENT_TYPE]: "chunk"},
            "{\"unexpected\":true}");
    ai:Error? err = drainChunks(wire, ANTHROPIC_STREAM, true).err;
    test:assertTrue(err is ai:Error, "a frame with no 'bytes' member must error");
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
    // Every shipped converter streams, so the guard in `runChatStream` is unreachable
    // today — but it is what keeps a future converter that this module can encode and
    // cannot decode incrementally from returning a silent, empty stream instead of an
    // error. Pinned at the field, since no route can exercise it any more.
    readonly & ModelConverter unstreamable = {
        encode: encodeConverse,
        decode: decodeConverse,
        toolChoice: CONVERSE_TOOL_CHOICE,
        streamDialect: ()
    };
    test:assertTrue(unstreamable.streamDialect is (), "the guard reads exactly this");
}

@test:Config {}
function testGenerateStreamRejectsANonStringTargetType() {
    // Structured output cannot stream: the typed value comes out of a forced tool
    // call whose arguments are only bindable once the whole JSON has arrived.
    AnthropicModelProvider provider = checkpanic new (CLAUDE_SONNET_5, TEST_CREDS, REGION,
            config = {apiFamily: CONVERSE});
    stream<FruitShape, ai:Error?>|ai:Error typed = provider->generateStream(`Name a fruit.`);
    if typed !is ai:Error {
        test:assertFail("generateStream with a record target must be refused");
    }
    test:assertTrue(typed.message().includes("only 'string'"), typed.message());
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
    stream<ai:ChatCompletionChunk, ai:Error?> chunks = new (new BedrockChunkIterator(
            new EventStreamEventSource(bytes, false), newStreamDecoder(CONVERSE_STREAM),
            "req-abc", "test.model-v1:0", observe:createChatSpan("test.model-v1:0")));

    record {|ai:ChatCompletionChunk value;|}? first = check chunks.next();
    test:assertTrue(first is record {|ai:ChatCompletionChunk value;|}, "the first chunk must arrive");
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
    stream<ai:ChatCompletionChunk, ai:Error?> chunks = new (new BedrockChunkIterator(
            new EventStreamEventSource(bytes, false), newStreamDecoder(CONVERSE_STREAM),
            "req-abc", "test.model-v1:0", observe:createChatSpan("test.model-v1:0")));

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
    stream<ai:ChatCompletionChunk, ai:Error?> chunks = new (new BedrockChunkIterator(
            new EventStreamEventSource(bytes, false), newStreamDecoder(CONVERSE_STREAM),
            "req-abc", "test.model-v1:0", observe:createChatSpan("test.model-v1:0")));

    record {|ai:ChatCompletionChunk value;|}|ai:Error? first = chunks.next();
    test:assertTrue(first is ai:Error, "an exception frame must surface as an error");

    check chunks.close();
    test:assertEquals(body.closes(), 1, "close() after a failure must still release the body");
}

@test:Config {}
function testClosingTheGenerateStreamTextStreamPropagates() returns error? {
    // `generateStream` hands back a text stream wrapping the chunk stream. Without a
    // `close()` on the projection, `BedrockChunkIterator.close()` is unreachable
    // from a `generateStream` caller and the release above never happens.
    FakeChunkSource chunkSource = new ([
        singleChoiceChunk({content: "Hello "}),
        singleChoiceChunk({content: "world"})
    ]);
    stream<ai:ChatCompletionChunk, ai:Error?> chunks = new (chunkSource);
    stream<string, ai:Error?> text = new (new ChunkTextIterator(chunks));

    record {|string value;|}? first = check text.next();
    test:assertEquals(first?.value, "Hello ");
    test:assertFalse(chunkSource.isClosed(), "reading must not close the chunk stream");

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
            newStreamDecoder(CONVERSE_STREAM), "req-abc", "test.model-v1:0",
            observe:createChatSpan("test.model-v1:0"));

    record {|ai:ChatCompletionChunk value;|}|ai:Error? first = iterator.next();
    test:assertTrue(first is record {|ai:ChatCompletionChunk value;|}, "the text before the exception arrives");
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
            newStreamDecoder(CONVERSE_STREAM), "req-abc", "test.model-v1:0",
            observe:createChatSpan("test.model-v1:0"));

    test:assertTrue(iterator.next() is record {|ai:ChatCompletionChunk value;|}, "the first chunk arrives");
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
    BedrockChunkIterator iterator = new (new SseEventSource(sse), newStreamDecoder(dialect),
            "req-mantle", "anthropic.claude-sonnet-5", observe:createChatSpan("anthropic.claude-sonnet-5"));
    ai:ChatCompletionChunk[] chunks = [];
    while true {
        record {|ai:ChatCompletionChunk value;|}|ai:Error? next = iterator.next();
        if next is () {
            return {chunks, err: ()};
        }
        if next is ai:Error {
            return {chunks, err: next};
        }
        chunks.push(next.value);
    }
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
    ai:ChatCompletionChunk last = result.chunks[result.chunks.length() - 1];
    test:assertEquals(last.choices[0].finishReason, ai:STOP);
    ai:CompletionTokenUsage usage = <ai:CompletionTokenUsage>last?.usage;
    test:assertEquals(usage.promptTokens, 9, "the prompt half is stashed from message_start");
    test:assertEquals(usage.completionTokens, 4);
}

@test:Config {}
function testMantleChatSsePipelineStopsAtTheDoneSentinel() {
    // `[DONE]` is not JSON. Parsing it would report a malformed event on a response
    // that in fact completed perfectly.
    FakeSseStream body = new ([
        {data: "{\"id\":\"chatcmpl-1\",\"model\":\"zai.glm-5\"," +
                "\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}"},
        {data: "{\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1,\"total_tokens\":6}}"},
        {data: "[DONE]"}
    ]);
    stream<http:SseEvent, error?> sse = new (body);
    BedrockChunkIterator iterator = new (new SseEventSource(sse), newStreamDecoder(OPENAI_CHAT_STREAM),
            "req-mantle", "zai.glm-5", observe:createChatSpan("zai.glm-5"));

    ai:ChatCompletionChunk[] chunks = [];
    while true {
        record {|ai:ChatCompletionChunk value;|}|ai:Error? next = iterator.next();
        if next is () {
            break;
        }
        if next is ai:Error {
            test:assertFail(next.message());
        }
        chunks.push(next.value);
    }
    test:assertEquals(textOf(chunks), "ok");
    ai:CompletionTokenUsage usage = <ai:CompletionTokenUsage>chunks[chunks.length() - 1]?.usage;
    test:assertEquals(usage.totalTokens, 6);
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
    test:assertEquals(result.chunks[0].id, "resp_9", "the dialect's own id survives the backfill");
    ai:ChatCompletionChunk last = result.chunks[result.chunks.length() - 1];
    test:assertEquals(last.choices[0].finishReason, ai:STOP);
    test:assertEquals((<ai:CompletionTokenUsage>last?.usage).totalTokens, 10);
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
    stream<ai:ChatCompletionChunk, ai:Error?> chunks = new (new BedrockChunkIterator(
            new SseEventSource(sse), newStreamDecoder(ANTHROPIC_STREAM), "req-mantle",
            "anthropic.claude-sonnet-5", observe:createChatSpan("anthropic.claude-sonnet-5")));

    record {|ai:ChatCompletionChunk value;|}? first = check chunks.next();
    test:assertTrue(first is record {|ai:ChatCompletionChunk value;|}, "the first chunk must arrive");
    test:assertEquals(body.closes(), 0, "reading must not close the body");

    check chunks.close();
    test:assertEquals(body.closes(), 1, "close() must reach the SSE stream");
    check chunks.close();
    test:assertEquals(body.closes(), 1, "close() must stay idempotent");
}
