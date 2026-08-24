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
import ballerina/crypto;
import ballerina/lang.array;
import ballerina/test;

// AWS event-stream framing. The decoder is the one piece of this module that reads
// BINARY off the wire, so it is tested against byte-exact fixtures built here by an
// independent encoder — if the two ever disagree about the layout, these fail.

// ---------------------------------------------------------------------------
// Fixture builder — the inverse of `EventStreamFramer`, written from the spec
// rather than from the decoder, so a shared misreading cannot cancel out.
// ---------------------------------------------------------------------------

// Encodes one string-valued header: nameLen:u8, name, type:u8(7), valueLen:u16, value.
function encodeHeader(string name, string value) returns byte[] {
    byte[] out = [];
    byte[] nameBytes = name.toBytes();
    byte[] valueBytes = value.toBytes();
    out.push(<byte>nameBytes.length());
    out.push(...nameBytes);
    out.push(<byte>HDR_STRING);
    out.push(...u16BE(valueBytes.length()));
    out.push(...valueBytes);
    return out;
}

function u32BE(int n) returns byte[] =>
    [<byte>(n / 16777216 % 256), <byte>(n / 65536 % 256), <byte>(n / 256 % 256), <byte>(n % 256)];

function u16BE(int n) returns byte[] => [<byte>(n / 256 % 256), <byte>(n % 256)];

// Real CRC32s, not zero padding. The decoder deliberately does not verify them, but
// building valid ones keeps these fixtures honest: if checking is ever added, the
// tests keep passing rather than all breaking at once.
function crc32Bytes(byte[] data) returns byte[] {
    string hex = crypto:crc32b(data);
    byte[] out = [];
    int i = 0;
    // `crc32b` returns lowercase hex, unpadded on the left for small values.
    string padded = hex;
    while padded.length() < 8 {
        padded = "0" + padded;
    }
    while i < 8 {
        out.push(<byte>(checkpanic int:fromHexString(padded.substring(i, i + 2))));
        i += 2;
    }
    return out;
}

// Builds a complete event-stream message.
function buildFrame(map<string> headers, string payload) returns byte[] {
    byte[] headerBytes = [];
    foreach [string, string] [k, v] in headers.entries() {
        headerBytes.push(...encodeHeader(k, v));
    }
    byte[] payloadBytes = payload.toBytes();
    int totalLen = EVENTSTREAM_OVERHEAD_LEN + headerBytes.length() + payloadBytes.length();

    byte[] prelude = [];
    prelude.push(...u32BE(totalLen));
    prelude.push(...u32BE(headerBytes.length()));

    byte[] frame = [];
    frame.push(...prelude);
    frame.push(...crc32Bytes(prelude));
    frame.push(...headerBytes);
    frame.push(...payloadBytes);
    frame.push(...crc32Bytes(frame));
    return frame;
}

// A Converse-shaped event frame.
function converseFrame(string eventType, string payload) returns byte[] =>
    buildFrame({[HDR_MESSAGE_TYPE]: "event", [HDR_EVENT_TYPE]: eventType, ":content-type": "application/json"},
        payload);

// ---------------------------------------------------------------------------
// Framing
// ---------------------------------------------------------------------------

@test:Config {}
function testFramerDecodesASingleFrame() {
    EventStreamFramer framer = new;
    framer.feed(converseFrame("contentBlockDelta", "{\"delta\":{\"text\":\"hi\"}}"));

    EventStreamFrame|ai:Error? frame = framer.nextFrame();
    if frame !is EventStreamFrame {
        test:assertFail("expected a complete frame to decode");
    }
    test:assertEquals(frame.headers[HDR_EVENT_TYPE], "contentBlockDelta");
    test:assertEquals(frame.headers[HDR_MESSAGE_TYPE], "event");
    test:assertEquals(checkpanic string:fromBytes(frame.payload), "{\"delta\":{\"text\":\"hi\"}}");
    test:assertFalse(framer.hasPartialFrame());
}

@test:Config {}
function testFramerReturnsNilUntilAFrameIsComplete() {
    byte[] frame = converseFrame("messageStart", "{\"role\":\"assistant\"}");
    EventStreamFramer framer = new;

    // Everything but the last byte: still incomplete.
    framer.feed(frame.slice(0, frame.length() - 1));
    EventStreamFrame|ai:Error? partial = framer.nextFrame();
    test:assertTrue(partial is (), "a frame missing its final byte must not decode");
    test:assertTrue(framer.hasPartialFrame());

    framer.feed([frame[frame.length() - 1]]);
    EventStreamFrame|ai:Error? complete = framer.nextFrame();
    test:assertTrue(complete is EventStreamFrame, "the frame must decode once its last byte arrives");
}

@test:Config {}
function testFramerHandlesByteAtATimeDelivery() {
    // The real read size is ONE byte (see STREAM_READ_SIZE), so frames arrive split
    // at arbitrary points. Feeding a byte at a time is the production path, not an
    // edge case.
    byte[] a = converseFrame("contentBlockDelta", "{\"delta\":{\"text\":\"one\"}}");
    byte[] b = converseFrame("contentBlockDelta", "{\"delta\":{\"text\":\"two\"}}");
    EventStreamFramer framer = new;

    byte[][] frames = [a, b];
    int decoded = 0;
    foreach byte[] wire in frames {
        foreach byte x in wire {
            framer.feed([x]);
            EventStreamFrame|ai:Error? frame = framer.nextFrame();
            if frame is EventStreamFrame {
                decoded += 1;
            }
        }
    }
    test:assertEquals(decoded, 2, "both frames must decode when fed one byte at a time");
    test:assertFalse(framer.hasPartialFrame());
}

@test:Config {}
function testFramerDecodesBackToBackFramesInOneFeed() {
    byte[] wire = converseFrame("messageStart", "{\"role\":\"assistant\"}");
    wire.push(...converseFrame("contentBlockDelta", "{\"delta\":{\"text\":\"x\"}}"));
    wire.push(...converseFrame("messageStop", "{\"stopReason\":\"end_turn\"}"));

    EventStreamFramer framer = new;
    framer.feed(wire);

    string[] seen = [];
    while true {
        EventStreamFrame|ai:Error? frame = framer.nextFrame();
        if frame !is EventStreamFrame {
            break;
        }
        seen.push(frame.headers[HDR_EVENT_TYPE] ?: "");
    }
    test:assertEquals(seen, ["messageStart", "contentBlockDelta", "messageStop"]);
}

@test:Config {}
function testFramerRejectsAnImpossibleLength() {
    // A totalLength below the fixed 16-byte overhead cannot describe a real frame.
    // Caught as a named error rather than propagating as a negative slice.
    byte[] wire = [];
    wire.push(...u32BE(4)); // totalLength
    wire.push(...u32BE(0)); // headersLength
    wire.push(0, 0, 0, 0); // prelude CRC placeholder — never reached, the length check fires first
    EventStreamFramer framer = new;
    framer.feed(wire);

    EventStreamFrame|ai:Error? frame = framer.nextFrame();
    test:assertTrue(frame is ai:Error, "a malformed prelude must surface as an error");
}

@test:Config {}
function testFramerSkipsNonStringHeadersWithoutDesynchronising() {
    // Only string headers are read, but every other type's WIDTH still has to be
    // right — a wrong skip would misalign the rest of the block and corrupt the
    // headers that DO matter. A bool (no value bytes) and an int (4) sit in front.
    byte[] boolHeader = [];
    byte[] boolName = ":flag".toBytes();
    boolHeader.push(<byte>boolName.length());
    boolHeader.push(...boolName);
    boolHeader.push(<byte>HDR_BOOL_TRUE);

    byte[] intHeader = [];
    byte[] intName = ":n".toBytes();
    intHeader.push(<byte>intName.length());
    intHeader.push(...intName);
    intHeader.push(<byte>HDR_INT);
    intHeader.push(...u32BE(7));

    byte[] headerBytes = [];
    headerBytes.push(...boolHeader);
    headerBytes.push(...intHeader);
    headerBytes.push(...encodeHeader(HDR_EVENT_TYPE, "metadata"));

    byte[] payloadBytes = "{}".toBytes();
    int totalLen = EVENTSTREAM_OVERHEAD_LEN + headerBytes.length() + payloadBytes.length();
    byte[] prelude = [];
    prelude.push(...u32BE(totalLen));
    prelude.push(...u32BE(headerBytes.length()));
    byte[] wire = [];
    wire.push(...prelude);
    wire.push(...crc32Bytes(prelude));
    wire.push(...headerBytes);
    wire.push(...payloadBytes);
    wire.push(...crc32Bytes(wire));

    EventStreamFramer framer = new;
    framer.feed(wire);
    EventStreamFrame|ai:Error? frame = framer.nextFrame();
    if frame !is EventStreamFrame {
        test:assertFail("expected a complete frame to decode");
    }
    test:assertEquals(frame.headers[HDR_EVENT_TYPE], "metadata",
            "the string header after a bool and an int must still be located correctly");
}

// ---------------------------------------------------------------------------
// Invoke `{"bytes": base64}` unwrapping
// ---------------------------------------------------------------------------

@test:Config {}
function testUnwrapInvokeChunkDecodesBase64Payload() {
    string inner = "{\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hey\"}}";
    json wrapper = {"bytes": array:toBase64(inner.toBytes())};

    json|ai:Error unwrapped = unwrapInvokeChunk(wrapper);
    if unwrapped !is map<json> {
        test:assertFail("expected the decoded vendor event");
    }
    test:assertEquals(unwrapped["type"], "content_block_delta");
}

@test:Config {}
function testUnwrapInvokeChunkRejectsAMissingBytesMember() {
    json|ai:Error unwrapped = unwrapInvokeChunk({"nope": "x"});
    test:assertTrue(unwrapped is ai:Error, "a frame with no 'bytes' member must error, not silently yield null");
}
