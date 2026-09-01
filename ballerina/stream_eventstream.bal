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

// AWS event-stream (`application/vnd.amazon.eventstream`) frame decoder.
//
// WHY THIS EXISTS AT ALL — the other `ai.*` provider modules consume streams with
// `http:Response.getSseEventStream()`. That is unavailable here: it hard-validates
// the response content type against `text/event-stream` and returns a
// `PayloadBindingClientError` for anything else, while `ConverseStream` and
// `InvokeModelWithResponseStream` both answer with
// `application/vnd.amazon.eventstream` — a BINARY framed protocol, not SSE. So the
// bytes are read raw and framed here.
//
// WIRE FORMAT — one message is:
//
//   +--------------------+--------------------+--------------------+
//   | totalLen  (u32 BE) | headersLen (u32 BE)| preludeCrc (u32 BE)|   12-byte prelude
//   +--------------------+--------------------+--------------------+
//   | headers ... headersLen bytes ...                             |
//   +--------------------------------------------------------------+
//   | payload ... totalLen - headersLen - 16 bytes ...              |
//   +--------------------------------------------------------------+
//   | messageCrc (u32 BE)                                           |
//   +--------------------------------------------------------------+
//
// and one header is `nameLen:u8, name, valueType:u8, value` where the value's width
// depends on the type tag.
//
// https://docs.aws.amazon.com/AmazonS3/latest/API/RESTSelectObjectAppendix.html
// (the encoding is shared service-wide; Bedrock does not restate it)

// Bytes of prelude (`totalLen`, `headersLen`, `preludeCrc`).
const int EVENTSTREAM_PRELUDE_LEN = 12;
// Prelude + trailing message CRC — the non-header, non-payload overhead.
const int EVENTSTREAM_OVERHEAD_LEN = 16;

// A defensive ceiling on a single frame. Bedrock frames are hundreds of bytes;
// anything near a megabyte means the byte offsets have desynchronised and we are
// reading a length out of the middle of a payload. Failing here surfaces that as a
// named error rather than an attempt to allocate a nonsense buffer.
const int EVENTSTREAM_MAX_FRAME_LEN = 1024 * 1024;

// Header value type tags. Only STRING is read — Bedrock puts `:event-type`,
// `:message-type`, `:content-type` and `:exception-type` in string headers — but
// every tag's WIDTH must still be known, because skipping one by the wrong number
// of bytes desynchronises the rest of the header block.
const int HDR_BOOL_TRUE = 0;
const int HDR_BOOL_FALSE = 1;
const int HDR_BYTE = 2;
const int HDR_SHORT = 3;
const int HDR_INT = 4;
const int HDR_LONG = 5;
const int HDR_BYTE_ARRAY = 6;
const int HDR_STRING = 7;
const int HDR_TIMESTAMP = 8;
const int HDR_UUID = 9;

// Header names Bedrock sets on every frame.
const string HDR_EVENT_TYPE = ":event-type";
const string HDR_MESSAGE_TYPE = ":message-type";
const string HDR_EXCEPTION_TYPE = ":exception-type";

// One decoded event-stream message: its string headers and its raw payload.
type EventStreamFrame record {|
    # String-valued headers, keyed by name. Non-string headers are skipped (Bedrock
    # sets none that we need).
    map<string> headers;
    # The frame body — JSON on Converse, `{"bytes": "<base64>"}` on Invoke.
    byte[] payload;
|};

# Incremental framer: bytes in, whole frames out.
#
# NOT isolated, and deliberately so — it exists to hold mutable buffer state across
# calls. It is constructed and consumed inside a single stream iterator, so the
# buffer is never shared between strands.
class EventStreamFramer {
    // Bytes received but not yet consumed by a complete frame. Frames are drained
    // as soon as they complete, so this stays on the order of one frame.
    private byte[] buf = [];

    # Appends freshly-read bytes to the buffer.
    isolated function feed(byte[] chunk) {
        // Spread, not a per-byte loop: with `STREAM_READ_SIZE` at 16 a loop crosses
        // the array-growth path once per BYTE for the whole response.
        self.buf.push(...chunk);
    }

    # Pops the next COMPLETE frame, or `()` when more bytes are needed.
    # An `ai:Error` means the stream is malformed and cannot be resynchronised.
    isolated function nextFrame() returns EventStreamFrame|ai:Error? {
        byte[] b = self.buf;
        if b.length() < EVENTSTREAM_PRELUDE_LEN {
            return (); // not even a prelude yet
        }
        int totalLen = readU32BE(b, 0);
        int headersLen = readU32BE(b, 4);
        if totalLen < EVENTSTREAM_OVERHEAD_LEN || totalLen > EVENTSTREAM_MAX_FRAME_LEN
                || headersLen < 0 || headersLen > totalLen - EVENTSTREAM_OVERHEAD_LEN {
            return error ai:LlmInvalidResponseError(
                string `Malformed Bedrock event-stream frame: totalLength=${totalLen}, ` +
                string `headersLength=${headersLen}`);
        }
        if b.length() < totalLen {
            return (); // frame still arriving
        }

        // The prelude and trailing CRC32s are NOT verified. Both guard against
        // corruption on the wire, which TLS already rules out, and the explicit
        // lengths are what framing actually needs. `crypto:crc32b` is available if a
        // reason to check them ever appears.
        int headersStart = EVENTSTREAM_PRELUDE_LEN;
        int headersEnd = headersStart + headersLen;
        map<string> headers = check parseHeaders(b, headersStart, headersEnd);
        // Payload runs from the end of the headers to the start of the message CRC.
        byte[] payload = b.slice(headersEnd, totalLen - 4);

        // Drop the consumed frame. `slice` copies, but only what is left over —
        // typically a partial next frame, so a few hundred bytes at most.
        self.buf = b.slice(totalLen);
        return {headers, payload};
    }

    # True when bytes remain that never formed a complete frame. Checked once the
    # byte stream ends, so a truncated response is reported rather than passed off
    # as a clean end of generation.
    isolated function hasPartialFrame() returns boolean => self.buf.length() > 0;
}

// Parses the header block into its string-valued entries. Non-string headers are
// stepped over by their type's width — see the tag constants above.
isolated function parseHeaders(byte[] b, int begin, int end) returns map<string>|ai:Error {
    map<string> headers = {};
    int p = begin;
    while p < end {
        int nameLen = b[p];
        p += 1;
        if p + nameLen > end {
            return error ai:LlmInvalidResponseError("Malformed event-stream header: name overruns the block");
        }
        string name = check bytesToString(b.slice(p, p + nameLen));
        p += nameLen;
        if p >= end {
            return error ai:LlmInvalidResponseError("Malformed event-stream header: missing value type");
        }
        int valueType = b[p];
        p += 1;

        match valueType {
            HDR_BOOL_TRUE|HDR_BOOL_FALSE => {
                // Value is carried by the tag itself; no value bytes follow.
            }
            HDR_BYTE => {
                p += 1;
            }
            HDR_SHORT => {
                p += 2;
            }
            HDR_INT => {
                p += 4;
            }
            HDR_LONG|HDR_TIMESTAMP => {
                p += 8;
            }
            HDR_UUID => {
                p += 16;
            }
            HDR_BYTE_ARRAY => {
                if p + 2 > end {
                    return error ai:LlmInvalidResponseError("Malformed event-stream header: truncated byte-array length");
                }
                p += 2 + readU16BE(b, p);
            }
            HDR_STRING => {
                if p + 2 > end {
                    return error ai:LlmInvalidResponseError("Malformed event-stream header: truncated string length");
                }
                int valueLen = readU16BE(b, p);
                p += 2;
                if p + valueLen > end {
                    return error ai:LlmInvalidResponseError("Malformed event-stream header: value overruns the block");
                }
                headers[name] = check bytesToString(b.slice(p, p + valueLen));
                p += valueLen;
            }
            _ => {
                // An unknown tag has an unknown width, so the rest of the block can
                // no longer be located. Stop rather than emit garbage headers.
                return error ai:LlmInvalidResponseError(
                    string `Unknown event-stream header value type '${valueType}' for header '${name}'`);
            }
        }
    }
    return headers;
}

// Big-endian u32. Bytes are unsigned in Ballerina, so no masking is needed.
isolated function readU32BE(byte[] b, int off) returns int
    => (b[off] * 16777216) + (b[off + 1] * 65536) + (b[off + 2] * 256) + b[off + 3];

// Big-endian u16.
isolated function readU16BE(byte[] b, int off) returns int => (b[off] * 256) + b[off + 1];

// UTF-8 bytes to string, as an `ai:Error` rather than a raw one.
isolated function bytesToString(byte[] b) returns string|ai:Error {
    string|error s = string:fromBytes(b);
    if s is error {
        return error ai:LlmInvalidResponseError("Bedrock event-stream frame contained invalid UTF-8", s);
    }
    return s;
}

// The frame's payload parsed as JSON.
isolated function framePayloadAsJson(EventStreamFrame frame) returns json|ai:Error {
    string text = check bytesToString(frame.payload);
    json|error parsed = text.fromJsonString();
    if parsed is error {
        return error ai:LlmInvalidResponseError("Bedrock event-stream frame payload was not valid JSON", parsed);
    }
    return parsed;
}
