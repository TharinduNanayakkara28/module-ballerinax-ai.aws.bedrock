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

// The streaming spine — `chatAsStream` and `generateAsStream` for every vendor
// facade. Mirrors `runChat` in provider_common.bal: the facades stay thin, and all
// the logic lives here.

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
//
// Applies to the event-stream wire only. Mantle's SSE body is read by the stdlib's
// own parser, which sizes its own reads.
const int STREAM_READ_SIZE = 16;

// The whole `chatAsStream()` implementation, shared by every vendor facade.
//
// NOT `isolated`, unlike `runChat`. The returned stream is backed by an iterator
// holding mutable framing state (a partially-filled byte buffer, the tool-index
// map) that necessarily outlives this call. `ai:ModelProvider` does not declare
// `chatAsStream` isolated either, so the facades match the contract.
function runChatStream(string providerName, ApiFamily family, string wireModelId,
        readonly & ModelConverter converter, BedrockTransport transport, map<string> & readonly extraHeaders,
        readonly & InferenceParams params, ai:ChatMessage[]|ai:ChatUserMessage messages,
        ai:ChatCompletionFunctions[] tools, string? stop)
        returns stream<ai:ChatMessageChunk, ai:Error?>|ai:Error {
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
    if tools.length() > 0 {
        span.addTools(tools);
    }

    // Resolve BEFORE encoding, exactly as `runChat` does: flattens each prompt to
    // parts and fetches any image URL, so the encoders stay pure. An image on a
    // dialect that cannot carry one fails HERE — before the stream is opened, which
    // is the only point at which a caller can still be handed a plain error rather
    // than a stream that immediately faults.
    [string?, ResolvedMessage[]]|ai:Error resolved = resolveMessages(msgs);
    if resolved is ai:Error {
        span.close(resolved);
        return resolved;
    }
    [string?, ResolvedMessage[]] [system, rest] = resolved;
    // Recorded from the RESOLVED form so an image becomes a placeholder rather than
    // shipping megabytes of user data to the telemetry backend.
    span.addInputMessages(messagesForSpan(system, rest));
    return openChunkStream(family, wireModelId, converter, transport, extraHeaders, params,
            system, rest, tools, stop, span);
}

// The whole `generateAsStream()` implementation, shared by every vendor facade.
//
// The request is built exactly as `generate()` builds it for a text answer (see
// `plainTextResponse`): the prompt resolved to parts as a single user message, no
// system prompt, no tools, on the provider's generate spine. Only the answer text
// reaches the caller — reasoning, tool-call and finish-only chunks are dropped.
//
// Streaming yields text only: a structured value has no valid intermediate state,
// so typed output stays with `generate()`.
function runGenerateStream(string providerName, ApiFamily family, string wireModelId,
        readonly & ModelConverter converter, BedrockTransport transport, map<string> & readonly extraHeaders,
        readonly & InferenceParams params, ai:Prompt prompt) returns stream<string, ai:Error?>|ai:Error {
    observe:GenerateContentSpan span = observe:createGenerateContentSpan(wireModelId);
    span.addProvider(providerName);
    decimal? spanTemperature = params?.temperature;
    if spanTemperature is decimal {
        span.addTemperature(spanTemperature);
    }
    ContentPart[]|ai:Error parts = contentToParts(prompt);
    if parts is ai:Error {
        span.close(parts);
        return parts;
    }
    ResolvedUserMessage userMsg = {parts};
    ResolvedMessage[] resolved = [userMsg];
    span.addInputMessages(messagesForSpan((), resolved));
    stream<ai:ChatMessageChunk, ai:Error?> chunks = check openChunkStream(family, wireModelId, converter,
            transport, extraHeaders, params, (), resolved, [], (), span);
    stream<string, ai:Error?> text = new (new ChunkTextIterator(chunks));
    return text;
}

// Encodes the resolved request, opens the live response and wraps it in a chunk
// stream that owns `span` from here on. Closes `span` itself on every failure
// before the stream exists.
function openChunkStream(ApiFamily family, string wireModelId, readonly & ModelConverter converter,
        BedrockTransport transport, map<string> & readonly extraHeaders, readonly & InferenceParams params,
        string? system, ResolvedMessage[] rest, ai:ChatCompletionFunctions[] tools, string? stop,
        StreamSpan span) returns stream<ai:ChatMessageChunk, ai:Error?>|ai:Error {
    // Refused BEFORE any I/O, naming the escape hatch — the same shape as the
    // module's other capability guards. The converter carries its own dialect, so the
    // capability check and the dialect it implies are one lookup and cannot
    // disagree — including on an opaque ARN, whose converter `selectConverter` resolves
    // from `modelSchema` rather than from the (ARN-valued) model id.
    //
    // Every converter the module ships now carries a dialect, so this is unreachable
    // today. It stays because the field is what makes streaming support a property
    // of the converter: a dialect AWS ships next that this module can encode but not
    // decode incrementally gets a clean refusal here rather than a silent empty
    // stream.
    StreamDialect? dialect = converter.streamDialect;
    if dialect is () {
        ai:Error err = error ai:Error(string `Streaming is not supported for model '${wireModelId}' on the ` +
            string `${family} route. Use 'apiFamily = CONVERSE' — ConverseStream is model-agnostic ` +
            string `and streams every vendor.`);
        span.close(err);
        return err;
    }
    StreamWire wire = streamWireFor(family);

    // The encoder is reused verbatim — a streaming request differs from a buffered
    // one only in how the route ASKS for the stream, never in what it asks for. On
    // `bedrock-runtime` that is a different operation and the body is byte-identical;
    // on Mantle it is `"stream": true` (plus `stream_options` where the dialect hides
    // usage behind it), which the converter carries as `streamFields`.
    RequestEncoder encode = converter.encode;
    json|ai:Error encoded = encode(system, rest, tools, stop, params);
    if encoded is ai:Error {
        span.close(encoded);
        return encoded;
    }
    // Mantle names the model in the BODY, not the path, so the same injection the
    // buffered call makes has to happen here — a streamed Mantle request without it
    // is rejected before a single chunk arrives.
    json addressed = family == MANTLE ? injectModel(encoded, wireModelId) : encoded;
    json|ai:Error body = withStreamFields(addressed, converter.streamFields);
    if body is ai:Error {
        span.close(body);
        return body;
    }

    [http:Response, map<string>]|ai:Error opened = transport.executeStreaming(body, extraHeaders, wire == SSE);
    if opened is ai:Error {
        span.close(opened);
        return opened;
    }
    [http:Response, map<string>] [response, responseHeaders] = opened;

    StreamEventSource|ai:Error events = openEventSource(response, wire, family == INVOKE);
    if events is ai:Error {
        span.close(events);
        return events;
    }

    string? requestId = responseHeaders[REQUEST_ID_HEADER];
    if requestId is string {
        span.addResponseId(requestId);
    }
    // Assigned to an explicitly typed local before returning: `new (iterator)`
    // cannot infer the stream's type parameters when the function returns a UNION
    // (`stream<...>|ai:Error`).
    stream<ai:ChatMessageChunk, ai:Error?> chunks = new (new BedrockChunkIterator(
            events, newStreamDecoder(dialect), requestId, span));
    return chunks;
}

// Adds the body fields that turn this route's request into a streaming one.
//
// A no-op on `bedrock-runtime`, where streaming is a different OPERATION and the
// body is untouched — which is why the field is optional rather than an empty map
// everywhere.
isolated function withStreamFields(json encoded, map<json>? streamFields) returns json|ai:Error {
    if streamFields is () {
        return encoded;
    }
    if encoded !is map<json> {
        // Unreachable with the shipped converters — every Mantle encoder builds an
        // object — but a converter that returned an array could not carry `"stream"`,
        // and silently sending a NON-streaming request would hang the caller on a
        // stream that yields one buffered answer at the very end.
        return error ai:LlmInvalidGenerationError(
            "This route's request body is not a JSON object, so it cannot carry the streaming flag");
    }
    map<json> body = encoded;
    foreach [string, json] [k, v] in streamFields.entries() {
        body[k] = v;
    }
    return body;
}

// Opens the right reader for the route's wire format.
//
// The SSE branch hands the body to the stdlib parser; the event-stream branch reads
// raw bytes, because `getSseEventStream()` hard-validates the content type against
// `text/event-stream` and `bedrock-runtime` answers
// `application/vnd.amazon.eventstream` (see stream_eventstream.bal).
isolated function openEventSource(http:Response response, StreamWire wire, boolean unwrapBytes)
        returns StreamEventSource|ai:Error {
    if wire == SSE {
        stream<http:SseEvent, error?>|http:ClientError events = response.getSseEventStream();
        if events is http:ClientError {
            // The stdlib reader validates the content type, so this is also what a
            // Mantle response that is NOT a stream looks like — the request reached
            // the model but never asked for one. Naming that cause matters: the only
            // way to get here is a converter (or a `routeOverrides` entry) whose
            // `streamFields` did not carry the flag, and the raw binding error says
            // nothing about it.
            return error ai:LlmConnectionError(
                "Failed to open the Bedrock SSE event stream: the response was not 'text/event-stream'. " +
                "The Mantle route asks for a stream with a body flag, so this usually means the request " +
                "was sent without one.", events);
        }
        return new SseEventSource(events);
    }
    stream<byte[], io:Error?>|http:ClientError bytes = response.getByteStream(STREAM_READ_SIZE);
    if bytes is http:ClientError {
        return error ai:LlmConnectionError("Failed to open the Bedrock response byte stream", bytes);
    }
    return new EventStreamEventSource(bytes, unwrapBytes);
}

# Turns a source of native events into `ai:ChatMessageChunk`s.
#
# Wire-format agnostic by construction: framing, the Invoke envelope, in-band
# service exceptions and truncation all live behind `StreamEventSource`, and the
# dialect's event model lives behind `StreamChunkDecoder`. What is left here is what
# is TRUE OF EVERY BEDROCK STREAM — stamping the response id, span accounting,
# latching a failure, and releasing the response.
#
# Owns the observe span for the whole response, because a stream outlives the call
# that created it — `openChunkStream` has already returned by the time the first
# chunk is pulled, so it cannot close the span itself.
class BedrockChunkIterator {
    private final StreamEventSource events;
    private final StreamChunkDecoder decoder;
    private final string? requestId;
    private final StreamSpan span;
    // The provider's own response id, once an event has named it.
    private string? responseId = ();
    // The id stamped on every chunk, fixed when the first chunk goes out so it stays
    // stable across the whole response.
    private string? chunkId = ();
    // Accumulated for the span, which wants the totals a non-streaming call reads
    // straight off the response body. Module-visible so tests can read them; the
    // span itself cannot be observed from outside `ai.observe`.
    int inputTokens = 0;
    int outputTokens = 0;
    string finishReason = "";
    private boolean closed = false;

    isolated function init(StreamEventSource events, StreamChunkDecoder decoder, string? requestId,
            StreamSpan span) {
        self.events = events;
        self.decoder = decoder;
        self.requestId = requestId;
        self.span = span;
    }

    public isolated function next() returns record {|ai:ChatMessageChunk value;|}|ai:Error? {
        // Once the stream has ended — cleanly, on an error, or because the caller
        // closed it — it stays ended. Without this, a consumer that keeps pulling
        // after a reported failure re-enters the read loop and can be handed MORE
        // chunks: exactly the half-answer-read-as-whole that surfacing the exception
        // frame exists to prevent.
        if self.closed {
            return ();
        }
        while true {
            StreamEvent|ai:Error? event = self.events.next();
            if event is ai:Error {
                return self.failWith(event);
            }
            if event is () {
                self.finish();
                return ();
            }
            StreamUpdate|ai:Error? update = self.decoder.decode(event.eventType, event.payload);
            if update is ai:Error {
                return self.failWith(update);
            }
            if update is () {
                continue;
            }
            self.accumulate(update);
            ai:ChatMessageChunk? chunk = update?.chunk;
            if chunk is () {
                // Pings, message-start markers, usage-only closers: nothing for the
                // caller, so pull the next event rather than emit an empty chunk.
                continue;
            }
            return {value: self.withId(chunk)};
        }
    }

    // Stamps the response id. The provider's own id wins; Converse events carry
    // none, and there the request id from the response header is the only value
    // stable across every chunk of a response.
    private isolated function withId(ai:ChatMessageChunk chunk) returns ai:ChatMessageChunk {
        string? id = self.chunkId ?: self.responseId ?: self.requestId;
        self.chunkId = id;
        if id is string {
            chunk.id = id;
        }
        return chunk;
    }

    // Accumulates what the span reports at close.
    private isolated function accumulate(StreamUpdate update) {
        string? responseId = update?.responseId;
        if responseId is string && self.responseId is () {
            self.responseId = responseId;
        }
        StreamUsage? usage = update?.usage;
        if usage is StreamUsage {
            self.inputTokens = usage?.inputTokens ?: self.inputTokens;
            self.outputTokens = usage?.outputTokens ?: self.outputTokens;
        }
        ai:ChatMessageChunk? chunk = update?.chunk;
        if chunk is ai:ChatMessageChunk {
            ai:FinishReason? reason = chunk.finishReason;
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
        self.span.addInputTokenCount(self.inputTokens);
        self.span.addOutputTokenCount(self.outputTokens);
        if self.finishReason != "" {
            self.span.addFinishReason(self.finishReason);
            // Paired with the finish reason, as in the reference iterator. `TEXT` is
            // the only honest value: `observe:OutputType` is TEXT|JSON, and a chat
            // stream is text deltas even when some of them carry tool-call
            // fragments — structured output never streams.
            self.span.addOutputType(observe:TEXT);
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

    // Releases the live response and closes the span.
    //
    // Closing the stream is the ONLY hook a consumer that stops early has — breaking
    // out of a `foreach` once it has seen enough, or abandoning the stream after a
    // mid-stream error. Without this the `http:Response` body is left open, leaking a
    // pooled connection per abandoned stream until the pool is exhausted, and the
    // span dangles unclosed.
    //
    // Idempotent on both halves: the span close guards on `closed` (and is already a
    // no-op after a failure), and the event source skips a body it already released.
    public isolated function close() returns ai:Error? {
        self.finish();
        return self.events.close();
    }
}

# Projects a chunk stream onto its answer text, yielding each non-empty `content`
# fragment and skipping reasoning, tool-call and finish-only chunks.
class ChunkTextIterator {
    private final stream<ai:ChatMessageChunk, ai:Error?> chunks;

    isolated function init(stream<ai:ChatMessageChunk, ai:Error?> chunks) {
        self.chunks = chunks;
    }

    public isolated function next() returns record {|string value;|}|ai:Error? {
        while true {
            record {|ai:ChatMessageChunk value;|}|ai:Error? next = self.chunks.next();
            if next !is record {|ai:ChatMessageChunk value;|} {
                return next;
            }
            string? content = next.value.content;
            if content is string && content != "" {
                return {value: content};
            }
        }
    }

    // Propagates the close to the chunk stream underneath — and through it to the
    // live HTTP response and the span. Without this, closing a `generateAsStream`
    // text stream early would release nothing.
    public isolated function close() returns ai:Error? {
        return self.chunks.close();
    }
}
