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
import ballerina/http;
import ballerina/test;

// `chatAsStream` and `generateAsStream` through a real provider, over a real socket,
// against a local stand-in for `bedrock-runtime` reached through `serviceUrl`.

const int STREAM_MOCK_PORT = 18801;
const int GENERATE_MOCK_PORT = 18804;
const int FAILING_MOCK_PORT = 18802;
// Nothing listens here.
const int CLOSED_PORT = 18803;
const string WIRE_MODEL = "anthropic.claude-sonnet-4-6";
const string WIRE_REQUEST_ID = "req-wire-1";

// One ConverseStream response mixing every kind of chunk: reasoning, text, a tool
// call split across fragments, a stop reason and a usage-only closer.
final readonly & byte[] MIXED_CONVERSE_STREAM = mixedConverseStream().cloneReadOnly();

function mixedConverseStream() returns byte[] {
    byte[] wire = converseFrame("messageStart", "{\"role\":\"assistant\"}");
    wire.push(...converseFrame("contentBlockDelta",
            "{\"contentBlockIndex\":0,\"delta\":{\"reasoningContent\":{\"text\":\"Thinking.\"}}}"));
    wire.push(...converseFrame("contentBlockStop", "{\"contentBlockIndex\":0}"));
    wire.push(...converseFrame("contentBlockDelta", "{\"contentBlockIndex\":1,\"delta\":{\"text\":\"Hel\"}}"));
    wire.push(...converseFrame("contentBlockDelta", "{\"contentBlockIndex\":1,\"delta\":{\"text\":\"lo\"}}"));
    wire.push(...converseFrame("contentBlockStop", "{\"contentBlockIndex\":1}"));
    wire.push(...converseFrame("contentBlockStart",
            "{\"contentBlockIndex\":2,\"start\":{\"toolUse\":{\"toolUseId\":\"tu_1\",\"name\":\"lookup\"}}}"));
    wire.push(...converseFrame("contentBlockDelta",
            "{\"contentBlockIndex\":2,\"delta\":{\"toolUse\":{\"input\":\"{\\\"q\\\":\"}}}"));
    wire.push(...converseFrame("contentBlockDelta",
            "{\"contentBlockIndex\":2,\"delta\":{\"toolUse\":{\"input\":\"\\\"x\\\"}\"}}}"));
    wire.push(...converseFrame("contentBlockStop", "{\"contentBlockIndex\":2}"));
    wire.push(...converseFrame("messageStop", "{\"stopReason\":\"tool_use\"}"));
    wire.push(...converseFrame("metadata", "{\"usage\":{\"inputTokens\":6,\"outputTokens\":9,\"totalTokens\":15}}"));
    return wire;
}

// The last request body the streaming mock received.
isolated json lastStreamRequest = ();

// Answers every POST with the canned event-stream response.
isolated service class StreamingBedrockMock {
    *http:Service;

    isolated resource function post [string... path](http:Request req) returns http:Response|error {
        json body = check req.getJsonPayload();
        lock {
            lastStreamRequest = body.clone();
        }
        http:Response res = new;
        res.setBinaryPayload(MIXED_CONVERSE_STREAM, "application/vnd.amazon.eventstream");
        res.setHeader("x-amzn-RequestId", WIRE_REQUEST_ID);
        return res;
    }
}

// Rejects every request the way Bedrock rejects a malformed one.
isolated service class FailingBedrockMock {
    *http:Service;

    isolated resource function post [string... path]() returns http:BadRequest {
        return {body: {message: "Malformed input request"}};
    }
}

function wireProvider(int port) returns AnthropicModelProvider|error =>
    new (WIRE_MODEL, TEST_CREDS, "us-east-1", serviceUrl = string `http://localhost:${port}`,
        config = {retryConfig: {maxRetries: 0}});

@test:Config {}
function testChatAsStreamOverTheWire() returns error? {
    http:Listener mockListener = check new (STREAM_MOCK_PORT);
    check mockListener.attach(new StreamingBedrockMock(), "/");
    check mockListener.'start();

    AnthropicModelProvider provider = check wireProvider(STREAM_MOCK_PORT);
    stream<ai:ChatMessageChunk, ai:Error?> chunks = check provider->chatAsStream(
            [{role: ai:USER, content: "hi"}], [{name: "lookup", description: "Looks something up"}]);
    ai:ChatMessageChunk[] received = check from ai:ChatMessageChunk chunk in chunks
        select chunk;
    check mockListener.gracefulStop();

    // reasoning, "Hel", "lo", tool opener, two argument fragments, stop = 7.
    test:assertEquals(received.length(), 7);
    foreach ai:ChatMessageChunk chunk in received {
        test:assertEquals(chunk.role, ai:ASSISTANT, "every chunk must carry the assistant role");
        test:assertEquals(chunk?.id, WIRE_REQUEST_ID, "Converse has no message id: the request id is stamped");
    }
    test:assertEquals(received[0].reasoning, "Thinking.");
    test:assertEquals(received[0].content, ());
    test:assertEquals(received[1].content, "Hel");
    test:assertEquals(received[2].content, "lo");
    test:assertEquals(received[3].toolCalls, [{index: 0, id: "tu_1", name: "lookup"}],
            "the tool call's index counts tool calls, not content blocks");
    test:assertEquals(received[4].toolCalls, [{index: 0, arguments: "{\"q\":"}]);
    test:assertEquals(received[5].toolCalls, [{index: 0, arguments: "\"x\"}"}]);
    test:assertEquals(received[6].finishReason, ai:TOOL_CALLS);
}

@test:Config {}
function testGenerateAsStreamYieldsOnlyTheAnswerText() returns error? {
    http:Listener mockListener = check new (GENERATE_MOCK_PORT);
    check mockListener.attach(new StreamingBedrockMock(), "/");
    check mockListener.'start();

    AnthropicModelProvider provider = check wireProvider(GENERATE_MOCK_PORT);
    string topic = "greetings";
    stream<string, ai:Error?> text = check provider->generateAsStream(`Say hello about ${topic}.`);
    string[] fragments = check from string fragment in text
        select fragment;
    check mockListener.gracefulStop();

    test:assertEquals(fragments, ["Hel", "lo"], "reasoning, tool-call and finish-only chunks must be skipped");

    // Built as `generate()` builds a text request: one user message carrying the
    // resolved prompt, and no system prompt or tools.
    json request;
    lock {
        request = lastStreamRequest.clone();
    }
    map<json> body = check request.ensureType();
    test:assertFalse(body.hasKey("toolConfig"), "generateAsStream must not send tools");
    test:assertFalse(body.hasKey("system"), "generateAsStream must not send a system prompt");
    json[] messages = check body["messages"].ensureType();
    test:assertEquals(messages.length(), 1);
    test:assertEquals(check messages[0].role, "user");
    test:assertEquals(check messages[0].content, <json>[{"text": "Say hello about greetings."}]);
}

@test:Config {}
function testStreamingReportsAConnectionFailure() returns error? {
    AnthropicModelProvider provider = check wireProvider(CLOSED_PORT);

    stream<ai:ChatMessageChunk, ai:Error?>|ai:Error chat = provider->chatAsStream([{role: ai:USER, content: "hi"}]);
    test:assertTrue(chat is ai:LlmConnectionError,
            "an unreachable endpoint must be an ai:LlmConnectionError from chatAsStream");

    stream<string, ai:Error?>|ai:Error text = provider->generateAsStream(`hi`);
    test:assertTrue(text is ai:LlmConnectionError,
            "an unreachable endpoint must be an ai:LlmConnectionError from generateAsStream");
}

@test:Config {}
function testStreamingReportsAnHttpError() returns error? {
    http:Listener mockListener = check new (FAILING_MOCK_PORT);
    check mockListener.attach(new FailingBedrockMock(), "/");
    check mockListener.'start();

    AnthropicModelProvider provider = check wireProvider(FAILING_MOCK_PORT);
    stream<ai:ChatMessageChunk, ai:Error?>|ai:Error chat = provider->chatAsStream([{role: ai:USER, content: "hi"}]);
    stream<string, ai:Error?>|ai:Error text = provider->generateAsStream(`hi`);
    check mockListener.gracefulStop();

    if chat !is ai:Error {
        test:assertFail("an HTTP 400 must be returned as an error, not a stream");
    }
    test:assertTrue(chat.message().includes("HTTP 400"), chat.message());
    test:assertTrue(chat.message().includes("Malformed input request"), chat.message());
    if text !is ai:Error {
        test:assertFail("an HTTP 400 must be returned as an error, not a text stream");
    }
    test:assertTrue(text.message().includes("HTTP 400"), text.message());
}
