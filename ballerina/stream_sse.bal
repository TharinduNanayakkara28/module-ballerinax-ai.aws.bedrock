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

// Server-sent events -> `StreamEvent`. The `bedrock-mantle` wire format.
//
// WHY THIS EXISTS ALONGSIDE `EventStreamFramer` — the two Bedrock endpoints do not
// share a streaming protocol, and neither one is a variant of the other:
//
//   bedrock-runtime  ConverseStream / InvokeModelWithResponseStream answer
//                    `application/vnd.amazon.eventstream` — a BINARY framed format
//                    this module decodes itself (see stream_eventstream.bal), on a
//                    SIBLING operation path.
//   bedrock-mantle   serves the vendor APIs verbatim, so it streams the way those
//                    APIs do: `text/event-stream` on the SAME path, switched on by
//                    `"stream": true` in the request body.
//
// So the content-type validation inside `http:Response.getSseEventStream()` — the
// very thing that makes it unusable on the runtime endpoint — is what makes it the
// right reader here. Mantle streaming needs no hand-rolled parser at all; this class
// only adapts the stdlib's `http:SseEvent` onto `StreamEventSource`.
//
// https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html

// The sentinel OpenAI's two dialects put on the final `data:` line. It is NOT JSON,
// so it has to be recognised before the parse or it reads as a malformed event.
// Anthropic Messages never sends one — it ends the body after `message_stop` — which
// is why its absence is not treated as a truncated response.
const string SSE_DONE = "[DONE]";

// The SSE event name every Mantle dialect uses for a mid-stream failure.
const string SSE_EVT_ERROR = "error";

# Reads one Mantle SSE response.
class SseEventSource {
    *StreamEventSource;

    private final stream<http:SseEvent, error?> events;
    // Whether the response still holds a connection. Set once the body ends or the
    // terminal sentinel arrives, so `close()` can skip a stream already released.
    private boolean released = false;

    isolated function init(stream<http:SseEvent, error?> events) {
        self.events = events;
    }

    isolated function next() returns StreamEvent|ai:Error? {
        while true {
            record {|http:SseEvent value;|}|error? next = self.events.next();
            if next is error {
                return error ai:LlmConnectionError("Bedrock stream failed mid-response", next);
            }
            if next is () {
                // End of body. Unlike the event-stream wire there is nothing to
                // check for truncation: a half-written SSE event never completes a
                // `data:` line, so the parser simply never hands it over. The
                // dialect decoders are what notice a missing terminal event — a
                // caller that saw no finish reason knows the answer was cut short.
                self.released = true;
                return ();
            }

            http:SseEvent event = next.value;
            string? data = event.data;
            if data is () {
                // A comment, a bare `id:`, or a `retry:` — keep-alive traffic that
                // carries nothing. Pull the next one rather than surfacing it.
                continue;
            }
            string body = data.trim();
            if body == "" {
                continue;
            }
            if body == SSE_DONE {
                // The stream is over, but the body has not been drained — nothing
                // follows the sentinel, and waiting for the server to close would
                // hold a pooled connection for no reason. Release it here.
                self.release();
                return ();
            }

            json|error payload = body.fromJsonString();
            if payload is error {
                return error ai:LlmInvalidResponseError("Bedrock SSE event data was not valid JSON", payload);
            }
            string name = event.event ?: "";
            if name == SSE_EVT_ERROR {
                // A mid-stream failure. Surfaced as an error rather than skipped:
                // the generation has failed, and ending the stream quietly would
                // look to a caller like a clean, complete answer.
                return sseEventError(payload);
            }
            return {eventType: name, payload};
        }
    }

    isolated function close() returns ai:Error? {
        if self.released {
            return ();
        }
        self.released = true;
        error? err = self.events.close();
        if err is error {
            return error ai:LlmConnectionError("Failed to close the Bedrock SSE stream", err);
        }
        return ();
    }

    // Best-effort release used on the terminal sentinel, where a close failure is
    // not worth failing an otherwise complete response over.
    private isolated function release() {
        if self.released {
            return;
        }
        self.released = true;
        error? closeError = self.events.close();
        if closeError is error {
            // Intentionally ignored: the answer is complete and already delivered,
            // and the alternative is failing a good response on the way out.
        }
    }
}

// Builds the error for an `event: error` frame.
//
// The three Mantle dialects nest the detail differently — Anthropic sends
// `{"type":"error","error":{"type":…,"message":…}}`, the OpenAI dialects
// `{"error":{"message":…,"type":…}}`, and a bare `{"message":…}` is possible — so
// the message is looked for in each place rather than assumed.
isolated function sseEventError(json payload) returns ai:Error {
    map<json> p = payload is map<json> ? payload : {};
    map<json> err = mapField(p, "error") ?: p;
    string detail = strField(err, "message") ?: strField(p, "message") ?: "unknown";
    string kind = strField(err, "type") ?: strField(err, "code") ?: "error";
    return error ai:LlmError(string `Bedrock stream error (${kind}): ${detail}`);
}
