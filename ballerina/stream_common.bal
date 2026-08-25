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

// The streaming spine — `chatStream` for every vendor facade, plus the
// `generateStream` text projection. Mirrors `runChat` in provider_common.bal: the
// facades stay thin, and all the logic lives here.

// How many bytes to pull per read from the response body.
//
// MEASURED, not guessed. `http:Response.getByteStream(n)` BLOCKS until it has n
// bytes or the stream ends — the array size is a latency knob, not a buffer hint.
// (That is observed behaviour, not a documented contract: the API doc says only
// "the size of the byte array". The stdlib's own SSE reader passes 1 with the note
// "the streaming party can decide to send one byte at a time".)
//
// Against a mock emitting 200-byte frames — a realistic ConverseStream frame is
// 12B prelude + ~78B of `:event-type`/`:content-type`/`:message-type` headers + a
// small JSON payload + 4B CRC — at 50 frames/s for 300 frames:
//
//     size | reads  | time to 1st decodable frame
//     -----+--------+----------------------------
//        1 | 60000  |  ~0.15s
//       16 |  3750  |  ~0.04s
//       64 |   938  |  ~0.04s
//
// All three keep pace with the sender overall; the difference is per-read overhead.
// One byte per read is the WORST of the three on latency, not the safest: reaching
// the first frame boundary costs 200 crossings of the stream/native boundary, which
// measurably delays the first token and burns ~16x the reads for no benefit.
//
// The ceiling is the SMALLEST frame Bedrock emits (~117B: a bare
// `{"contentBlockIndex":n}` plus headers). Above that, a read can span two frames
// and stall the first until the second arrives; a size at or above the whole
// response buffers everything and yields nothing until generation completes (8192
// measured a first read only at end-of-stream). 16 sits an order of magnitude below
// that ceiling.
//
// 16 over 64 — both measured the same latency — because a read can only ever strand
// bytes it has not filled: at most 15 here versus 63. That matters when the model
// PAUSES mid-response (between text and a tool call, or during thinking): whatever
// is buffered short of the array size waits for the next frame, so the trailing
// fragment of the last token surfaces late. End-of-stream always flushes it, so
// nothing is lost — only briefly delayed.
const int STREAM_READ_SIZE = 16;

// `:message-type` header value marking a frame that carries a service exception
// rather than an event.
const string EVENTSTREAM_MSG_TYPE_EXCEPTION = "exception";

// The whole `chatStream()` implementation, shared by every vendor facade.
//
// NOT `isolated`, unlike `runChat`. The returned stream is backed by an iterator
// holding mutable framing state (a partially-filled byte buffer, the tool-index
// map) that necessarily outlives this call. `ai:ModelProvider` does not declare
// `chatStream` isolated either, so the facades match the contract.
function runChatStream(string providerName, ApiFamily family, string wireModelId, string bareModelId,
        readonly & ModelCodec codec, BedrockTransport transport, map<string> & readonly extraHeaders,
        readonly & InferenceParams params, ai:ChatMessage[]|ai:ChatUserMessage messages,
        ai:ChatCompletionFunctions[] tools, string? stop)
        returns stream<ai:ChatCompletionChunk, ai:Error?>|ai:Error {
    // Refused BEFORE any I/O, naming the escape hatch — the same shape as the
    // module's other capability guards. `supportsStreaming` has been carried on
    // every codec since the module shipped; this is what finally reads it.
    if !codec.supportsStreaming {
        return error ai:Error(string `Streaming is not supported for model '${wireModelId}' on the ` +
            string `${family} route. Use 'apiFamily = CONVERSE' — ConverseStream is model-agnostic ` +
            string `and streams every vendor.`);
    }
    StreamDialect dialect = check selectStreamDialect(family, bareModelId);

    ai:ChatMessage[] msgs;
    if messages is ai:ChatUserMessage {
        msgs = [messages];
    } else {
        msgs = messages;
    }

    observe:ChatSpan span = observe:createChatSpan(wireModelId);
    span.addProvider(providerName);
    if stop is string {
        span.addStopSequence(stop);
    }
    decimal? spanTemperature = params?.temperature;
    if spanTemperature is decimal {
        span.addTemperature(spanTemperature);
    }
    span.addInputMessages(messagesForSpan(msgs));
    if tools.length() > 0 {
        span.addTools(tools);
    }

    // The request body is IDENTICAL to the non-streaming call — Bedrock selects
    // streaming by operation (`converse-stream`, `invoke-with-response-stream`),
    // never by a body flag — so the codec's encoder is reused verbatim.
    [ai:ChatSystemMessage?, ai:ChatMessage[]] [system, rest] = hoistSystem(msgs);
    RequestCodec encode = codec.encode;
    json|ai:Error encoded = encode(system, rest, tools, stop, params);
    if encoded is ai:Error {
        span.close(encoded);
        return encoded;
    }

    [http:Response, map<string>]|ai:Error opened = transport.executeStreaming(encoded, extraHeaders);
    if opened is ai:Error {
        span.close(opened);
        return opened;
    }
    [http:Response, map<string>] [response, responseHeaders] = opened;

    stream<byte[], io:Error?>|http:ClientError bytes = response.getByteStream(STREAM_READ_SIZE);
    if bytes is http:ClientError {
        ai:Error err = error ai:LlmConnectionError("Failed to open the Bedrock response byte stream", bytes);
        span.close(err);
        return err;
    }

    string? responseId = responseHeaders[REQUEST_ID_HEADER];
    if responseId is string {
        span.addResponseId(responseId);
    }
    // Assigned to an explicitly typed local before returning: `new (iterator)`
    // cannot infer the stream's type parameters when the function returns a UNION
    // (`stream<...>|ai:Error`).
    stream<ai:ChatCompletionChunk, ai:Error?> chunks = new (new BedrockChunkIterator(
            bytes, newStreamDecoder(dialect), family == INVOKE, responseId, wireModelId, span));
    return chunks;
}

# Turns the raw response bytes into normalized chunks: frame, unwrap, decode.
#
# Owns the observe span for the whole response, because a stream outlives the call
# that created it — `runChatStream` has already returned by the time the first chunk
# is pulled, so it cannot close the span itself.
class BedrockChunkIterator {
    private final stream<byte[], io:Error?> bytes;
    private final EventStreamFramer framer = new;
    private final StreamChunkDecoder decoder;
    // INVOKE wraps each vendor payload as `{"bytes": "<base64>"}`; Converse frames
    // carry the event JSON directly.
    private final boolean unwrapBytes;
    private final string? responseId;
    private final string wireModelId;
    private final observe:ChatSpan span;
    // Accumulated for the span, which wants the totals a non-streaming call reads
    // straight off the response body.
    private int promptTokens = 0;
    private int completionTokens = 0;
    private string finishReason = "";
    private boolean closed = false;

    isolated function init(stream<byte[], io:Error?> bytes, StreamChunkDecoder decoder, boolean unwrapBytes,
            string? responseId, string wireModelId, observe:ChatSpan span) {
        self.bytes = bytes;
        self.decoder = decoder;
        self.unwrapBytes = unwrapBytes;
        self.responseId = responseId;
        self.wireModelId = wireModelId;
        self.span = span;
    }

    public isolated function next() returns record {|ai:ChatCompletionChunk value;|}|ai:Error? {
        while true {
            EventStreamFrame|ai:Error? frame = self.framer.nextFrame();
            if frame is ai:Error {
                return self.failWith(frame);
            }
            if frame is EventStreamFrame {
                ai:ChatCompletionChunk|ai:Error? chunk = self.mapFrame(frame);
                if chunk is ai:Error {
                    return self.failWith(chunk);
                }
                if chunk is ai:ChatCompletionChunk {
                    self.recordForSpan(chunk);
                    return {value: chunk};
                }
                // An event with nothing to surface (contentBlockStop, ping, an event
                // type AWS added later). Pull the next frame rather than emitting an
                // empty chunk a caller would have to filter out.
                continue;
            }

            // No complete frame buffered — read more bytes.
            record {|byte[] value;|}|io:Error? next = self.bytes.next();
            if next is io:Error {
                return self.failWith(error ai:LlmConnectionError("Bedrock stream failed mid-response", next));
            }
            if next is () {
                // End of body. A leftover partial frame means the response was cut
                // short; reporting it beats presenting a truncated answer as complete.
                if self.framer.hasPartialFrame() {
                    return self.failWith(error ai:LlmInvalidResponseError(
                        "Bedrock stream ended mid-frame; the response was truncated"));
                }
                self.finish();
                return ();
            }
            self.framer.feed(next.value);
        }
    }

    // One frame -> at most one chunk.
    private isolated function mapFrame(EventStreamFrame frame) returns ai:ChatCompletionChunk|ai:Error? {
        // A service exception arrives as a FRAME, not an HTTP status: the response
        // was a clean 200 and the failure (throttling, a model error, a filtered
        // completion) happened partway through generating. Surfacing it as an error
        // is what stops a caller reading a half-answer as a whole one.
        if frame.headers[HDR_MESSAGE_TYPE] == EVENTSTREAM_MSG_TYPE_EXCEPTION {
            string kind = frame.headers[HDR_EXCEPTION_TYPE] ?: "unknown";
            json|ai:Error body = framePayloadAsJson(frame);
            string detail = "";
            if body is map<json> {
                detail = strField(body, "message") ?: strField(body, "Message") ?: "";
            }
            return error ai:LlmError(string `Bedrock stream ${kind}` + (detail == "" ? "" : string `: ${detail}`));
        }

        json payload = check framePayloadAsJson(frame);
        string eventType = frame.headers[HDR_EVENT_TYPE] ?: "";
        if self.unwrapBytes {
            payload = check unwrapInvokeChunk(payload);
            [eventType, payload] = unwrapNamedEvent(eventType, payload);
        }
        ai:ChatCompletionChunk? chunk = check self.decoder.decode(eventType, payload);
        if chunk is () {
            return ();
        }

        // Identity, filled in where the dialect did not supply it. Converse events
        // carry neither: the request id from the response header is the only value
        // stable across every chunk, which is what the contract asks `id` to be.
        ai:ChatCompletionChunk out = chunk;
        if out.id is () {
            string? id = self.responseId;
            if id is string {
                out.id = id;
            }
        }
        if out.model is () {
            out.model = self.wireModelId;
        }
        return out;
    }

    // Accumulates what the span reports at close.
    private isolated function recordForSpan(ai:ChatCompletionChunk chunk) {
        ai:CompletionTokenUsage? usage = chunk?.usage;
        if usage is ai:CompletionTokenUsage {
            self.promptTokens = usage?.promptTokens ?: self.promptTokens;
            self.completionTokens = usage?.completionTokens ?: self.completionTokens;
        }
        foreach ai:ChatCompletionChunkChoice choice in chunk.choices {
            ai:FinishReason? reason = choice.finishReason;
            if reason is ai:FinishReason {
                self.finishReason = reason;
            }
        }
    }

    // Closes the span once, on the success path.
    private isolated function finish() {
        if self.closed {
            return;
        }
        self.closed = true;
        self.span.addInputTokenCount(self.promptTokens);
        self.span.addOutputTokenCount(self.completionTokens);
        if self.finishReason != "" {
            self.span.addFinishReason(self.finishReason);
        }
        self.span.close();
    }

    // Closes the span with the error and returns it, so every failure path is one
    // statement and none can forget the span.
    private isolated function failWith(ai:Error err) returns ai:Error {
        if !self.closed {
            self.closed = true;
            self.span.close(err);
        }
        return err;
    }
}

// Re-derives the event type for a dialect that names its event in the PAYLOAD
// rather than in the frame header.
//
// FOUND LIVE, 2026-08-24, Nova Pro on `InvokeModelWithResponseStream`: the stream
// completed cleanly and produced ZERO chunks — no error, no text. On the Invoke
// route every frame's `:event-type` header is the constant `chunk`, so the event
// name has to come from somewhere else, and Nova puts it in the single top-level
// KEY of the payload:
//
//     ConverseStream (header):  :event-type: contentBlockDelta
//                               {"contentBlockIndex":0,"delta":{"text":"hi"}}
//
//     Nova on Invoke (key):     :event-type: chunk
//                               {"contentBlockDelta":{"contentBlockIndex":0,
//                                                     "delta":{"text":"hi"}}}
//
// The decoder matched `chunk` against its event names, found nothing, and skipped
// every frame — a silent empty stream, which is the worst possible failure shape.
//
// Anthropic-on-Invoke is untouched: its payload keys are `type`/`index`/`delta`/
// `message`/`usage`/`content_block`, none of which collide with a Converse event
// name, so it falls through to the header and its own decoder reads `type` itself.
// Falling back to the header keeps ConverseStream working unchanged.
//
// The lookup is over EVERY key, not just a single-key payload. Nova's terminal
// frame is the exception that proves it:
//
//     {"metadata": {"usage": {...}, "metrics": {}, "trace": {}},
//      "amazon-bedrock-invocationMetrics": {...}}
//
// Bedrock decorates the last frame with its own invocation metrics, so the payload
// carries TWO top-level keys. An arity guard drops it, and the usage that rides on
// `metadata` — the only usage a Converse-shaped stream ever reports — never reaches
// the caller. The event-name lookup is the discriminator; arity never was.
isolated function unwrapNamedEvent(string headerEventType, json payload) returns [string, json] {
    if payload is map<json> {
        foreach [string, json] [name, inner] in payload.entries() {
            if isConverseEventName(name) {
                return [name, inner];
            }
        }
    }
    return [headerEventType, payload];
}

// Whether a name is one of the `ConverseStream` event types.
isolated function isConverseEventName(string name) returns boolean =>
    name == CONVERSE_EVT_MESSAGE_START || name == CONVERSE_EVT_CONTENT_BLOCK_START ||
    name == CONVERSE_EVT_CONTENT_BLOCK_DELTA || name == CONVERSE_EVT_CONTENT_BLOCK_STOP ||
    name == CONVERSE_EVT_MESSAGE_STOP || name == CONVERSE_EVT_METADATA;

// Unwraps an InvokeModelWithResponseStream frame body.
//
// Every frame on the Invoke route is `:event-type: chunk` with a payload of
// `{"bytes": "<base64>"}`, whose decoded content is the VENDOR's own event JSON.
// Converse has no such wrapper — the frame body is the event.
isolated function unwrapInvokeChunk(json payload) returns json|ai:Error {
    if payload !is map<json> {
        return error ai:LlmInvalidResponseError("Bedrock Invoke stream frame was not a JSON object");
    }
    string? encoded = strField(payload, "bytes");
    if encoded is () {
        return error ai:LlmInvalidResponseError("Bedrock Invoke stream frame carried no 'bytes' member");
    }
    byte[]|error decoded = array:fromBase64(encoded);
    if decoded is error {
        return error ai:LlmInvalidResponseError("Bedrock Invoke stream frame 'bytes' was not valid base64", decoded);
    }
    string text = check bytesToString(decoded);
    json|error parsed = text.fromJsonString();
    if parsed is error {
        return error ai:LlmInvalidResponseError("Bedrock Invoke stream frame payload was not valid JSON", parsed);
    }
    return parsed;
}

// Builds the string stream behind the dependently-typed `generateStream`.
//
// Invoked by the `StreamGenerator` native shim. Regular (non-dependent) function:
// the Java boundary coerces the result to the caller's `td`.
//
// NOT `isolated` — it calls the non-isolated `chatStream`.
//
// + llmModel - The provider whose `chatStream` supplies the chunks
// + prompt - The prompt to run
// + td - The caller's expected type; only `string` is supported
// + return - A stream of text fragments, or an `ai:Error`
function generateLlmResponseStream(ai:ModelProvider llmModel, ai:Prompt prompt, typedesc<anydata> td)
        returns stream<string, ai:Error?>|ai:Error {
    // Structured output cannot stream: `generate()` gets its typed value out of a
    // FORCED TOOL CALL, whose arguments are only bindable once the whole JSON has
    // arrived — there is no partial record to hand back. A typed target is a clean
    // error rather than a stream that yields nothing until the end.
    if td !is typedesc<string> {
        return error ai:LlmInvalidGenerationError(
            "'generateStream' supports only 'string'; use 'generate' for structured types.");
    }
    stream<ai:ChatCompletionChunk, ai:Error?> chunks = check llmModel->chatStream({role: ai:USER, content: prompt});
    stream<string, ai:Error?> text = new (new ChunkTextIterator(chunks));
    return text;
}

# Projects a chunk stream onto its text fragments, skipping the chunks that carry
# no content — role-only openers, tool-call fragments, usage-only closers.
class ChunkTextIterator {
    private final stream<ai:ChatCompletionChunk, ai:Error?> chunks;

    isolated function init(stream<ai:ChatCompletionChunk, ai:Error?> chunks) {
        self.chunks = chunks;
    }

    public isolated function next() returns record {|string value;|}|ai:Error? {
        while true {
            record {|ai:ChatCompletionChunk value;|}|ai:Error? next = self.chunks.next();
            if next is ai:Error {
                return next;
            }
            if next is () {
                return ();
            }
            string text = "";
            foreach ai:ChatCompletionChunkChoice choice in next.value.choices {
                string? content = choice.delta.content;
                if content is string {
                    text += content;
                }
            }
            // An empty fragment is not end-of-stream; keep pulling. Emitting "" for
            // every tool-call fragment would flood a caller printing the stream.
            if text != "" {
                return {value: text};
            }
        }
    }
}
