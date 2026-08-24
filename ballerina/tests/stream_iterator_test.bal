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
    BedrockChunkIterator iterator = new (bytes, newStreamDecoder(dialect), unwrapBytes,
            "req-abc", "test.model-v1:0", observe:createChatSpan("test.model-v1:0"));
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
    // Production reads ONE byte at a time, but frame boundaries must not depend on
    // where the reads fall. The same wire decoded at four read sizes must agree.
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
function testChatStreamIsRefusedOnAMantleRoutedModel() {
    // Mantle streams as SSE with a body flag — a different transport entirely, and
    // not yet implemented. The refusal must happen BEFORE any I/O and must name the
    // escape hatch, exactly as the module's other capability guards do.
    AnthropicModelProvider mantle = checkpanic new (TEST_CREDS, CLAUDE_SONNET_5, REGION);
    stream<ai:ChatCompletionChunk, ai:Error?>|ai:Error result = mantle->chatStream({role: ai:USER, content: "Hi"});
    if result !is ai:Error {
        test:assertFail("chatStream on an AUTO (Mantle) route must be refused");
    }
    test:assertTrue(result.message().includes("Streaming is not supported"), result.message());
    test:assertTrue(result.message().includes("apiFamily = CONVERSE"), result.message());
}

@test:Config {}
function testGenerateStreamRejectsANonStringTargetType() {
    // Structured output cannot stream: the typed value comes out of a forced tool
    // call whose arguments are only bindable once the whole JSON has arrived.
    AnthropicModelProvider provider = checkpanic new (TEST_CREDS, CLAUDE_SONNET_5, REGION,
            config = {apiFamily: CONVERSE});
    stream<FruitShape, ai:Error?>|ai:Error typed = provider->generateStream(`Name a fruit.`);
    if typed !is ai:Error {
        test:assertFail("generateStream with a record target must be refused");
    }
    test:assertTrue(typed.message().includes("only 'string'"), typed.message());
}
