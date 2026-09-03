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
import ballerina/lang.array;
import ballerina/test;

// Multimodal image input.
//
// THE BUG THESE EXIST FOR: `contentToString` used to flatten every prompt insertion
// with `.toString()`, so an `ai:ImageDocument` arrived at the model as the literal
// text `{"type":"image","content":...}`. No error, HTTP 200, and a confident answer
// about an image the model never received. Silent-wrong-answer, so every path below
// asserts either a real image block or a NAMED error — never a quiet drop.

// An 8-byte PNG signature plus filler. Enough for the magic-byte sniff, which only
// reads the leading bytes.
final byte[] & readonly PNG_BYTES = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x01];
final byte[] & readonly JPEG_BYTES = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
final byte[] & readonly GIF_BYTES = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61];
final byte[] & readonly WEBP_BYTES = [0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50];

final InferenceParams IMG_PARAMS = {maxTokens: 100};

// ---- Converse: the native image ContentBlock (verified route) ----

@test:Config {}
function testConverseEmitsNativeImageBlockWithBareFormatToken() returns error? {
    map<json> body = check encodeConverse((), [userImage("image/png", PNG_BYTES)], [], (),
            IMG_PARAMS).ensureType();
    json[] messages = check body["messages"].ensureType();
    map<json> first = check messages[0].ensureType();
    json[] content = check first["content"].ensureType();
    // `format` is a BARE token, not the MIME type — Converse's ImageBlock constrains
    // it to exactly png|jpeg|gif|webp, so emitting "image/png" here is a 400.
    test:assertEquals(content[0], <json>{
                "image": {
                    "format": "png",
                    "source": {"bytes": array:toBase64(PNG_BYTES)}
                }
            });
}

@test:Config {}
function testConversePreservesTextAndImageOrderWithinOneTurn() returns error? {
    ResolvedUserMessage msg = {
        parts: [{text: "Before "}, {mimeType: "image/jpeg", data: JPEG_BYTES}, {text: " after"}]
    };
    map<json> body = check encodeConverse((), [msg], [], (), IMG_PARAMS).ensureType();
    json[] messages = check body["messages"].ensureType();
    map<json> first = check messages[0].ensureType();
    json[] content = check first["content"].ensureType();
    test:assertEquals(content.length(), 3, "text/image/text must stay three ordered blocks");
    test:assertEquals(content[0], <json>{"text": "Before "});
    test:assertEquals(content[2], <json>{"text": " after"});
}

// ---- Anthropic Messages: base64 source only (verified route) ----

@test:Config {}
function testAnthropicEmitsBase64SourceNeverAUrlSource() returns error? {
    map<json> body = check encodeInvokeAnthropic((), [userImage("image/webp", WEBP_BYTES)], [], (),
            IMG_PARAMS).ensureType();
    json[] messages = check body["messages"].ensureType();
    map<json> first = check messages[0].ensureType();
    json[] content = check first["content"].ensureType();
    // Anthropic's API models a `url` source, but AWS Bedrock does not support it —
    // "On Amazon Bedrock and Google Cloud, only base64-encoded sources are currently
    // available". Emitting a url source here would 400 on every Bedrock call.
    test:assertEquals(content[0], <json>{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/webp", // full MIME type here, unlike Converse
                    "data": array:toBase64(WEBP_BYTES)
                }
            });
}

// ---- MIME resolution ----

@test:Config {}
function testMimeTypeIsSniffedFromMagicBytesWhenMetadataOmitsIt() returns error? {
    // `metadata.mimeType` is OPTIONAL on ai:ImageDocument but `format`/`media_type`
    // are REQUIRED on the wire, so an unset type must be recovered, not defaulted.
    foreach [byte[], string] [bytes, expected] in [
        [PNG_BYTES, "image/png"],
        [JPEG_BYTES, "image/jpeg"],
        [GIF_BYTES, "image/gif"],
        [WEBP_BYTES, "image/webp"]
    ] {
        ai:ImageDocument doc = {content: bytes};
        ContentPart[] parts = check contentToParts(`Look: ${doc}`);
        ImagePart img = check parts.filter(p => p is ImagePart)[0].ensureType();
        test:assertEquals(img.mimeType, expected);
    }
}

@test:Config {}
function testExplicitMimeTypeWinsOverSniffingAndIsNormalized() returns error? {
    // `image/jpg` is not an IANA type and Bedrock rejects it; it must normalize.
    ai:ImageDocument doc = {content: JPEG_BYTES, metadata: {mimeType: "IMAGE/JPG; charset=binary"}};
    ContentPart[] parts = check contentToParts(`${doc}`);
    ImagePart img = check parts[0].ensureType();
    test:assertEquals(img.mimeType, "image/jpeg");
}

@test:Config {}
function testWildcardMimeTypeFallsThroughToSniffingRatherThanReachingTheWire() returns error? {
    // ai.openai and ai.azure default to `image/*`. That is a guaranteed 400 here:
    // neither Converse's `format` nor Anthropic's `media_type` has a wildcard.
    ai:ImageDocument doc = {content: PNG_BYTES, metadata: {mimeType: "image/*"}};
    ContentPart[] parts = check contentToParts(`${doc}`);
    ImagePart img = check parts[0].ensureType();
    test:assertEquals(img.mimeType, "image/png");
}

@test:Config {}
function testUnidentifiableImageFailsWithANamedError() returns error? {
    ai:ImageDocument doc = {content: [0x00, 0x01, 0x02, 0x03]};
    ContentPart[]|ai:Error parts = contentToParts(`${doc}`);
    test:assertTrue(parts is ai:Error);
    if parts is ai:Error {
        test:assertTrue(parts.message().includes("metadata.mimeType"), parts.message());
    }
}

@test:Config {}
function testUnsupportedImageTypeNamesTheFourAcceptedFormats() returns error? {
    ai:ImageDocument doc = {content: PNG_BYTES, metadata: {mimeType: "image/bmp"}};
    ContentPart[]|ai:Error parts = contentToParts(`${doc}`);
    test:assertTrue(parts is ai:Error);
    if parts is ai:Error {
        test:assertTrue(parts.message().includes("image/webp"), parts.message());
    }
}

// ---- Refusals: nothing may be silently dropped ----

@test:Config {}
function testImageInASystemMessageIsRefused() returns error? {
    // Converse's SystemContentBlock is text|guardContent|cachePoint, Anthropic's
    // `system` is text, Responses' `instructions` is a string. No route can carry it.
    ai:ImageDocument doc = {content: PNG_BYTES};
    ai:ChatMessage[] messages = [
        {role: ai:SYSTEM, content: `Use this: ${doc}`},
        {role: ai:USER, content: "hi"}
    ];
    [string?, ResolvedMessage[]]|ai:Error resolved = resolveMessages(messages);
    test:assertTrue(resolved is ai:Error);
    if resolved is ai:Error {
        test:assertTrue(resolved.message().includes("system message"), resolved.message());
    }
}

@test:Config {}
function testTextOnlyAndUnverifiedDialectsRefuseImagesByName() returns error? {
    ResolvedMessage[] withImage = [userImage("image/png", PNG_BYTES)];
    // Each of these either has no content-part array at all (the prompt dialects), or
    // no primary source confirming image support (the OpenAI-shaped ones). Refusing
    // is the whole point: a silent drop is the bug being fixed.
    json|ai:Error deepseek = encodeDeepSeekInvoke((), withImage, [], (), IMG_PARAMS);
    json|ai:Error mistralText = encodeMistralText((), withImage, [], (), IMG_PARAMS);
    json|ai:Error mistralChat = encodeMistralChat((), withImage, [], (), IMG_PARAMS);
    json|ai:Error openAIChat = encodeOpenAIChat((), withImage, [], (), IMG_PARAMS);
    json|ai:Error responses = encodeResponses((), withImage, [], (), IMG_PARAMS);

    foreach json|ai:Error result in [deepseek, mistralText, mistralChat, openAIChat, responses] {
        test:assertTrue(result is ai:Error, "an image must never be silently dropped");
        if result is ai:Error {
            test:assertTrue(result.message().includes("Image input is not supported"), result.message());
            // The error has to name a way forward, not just say no.
            test:assertTrue(result.message().includes("CONVERSE"), result.message());
        }
    }
}

@test:Config {}
function testNonImageDocumentsAreRefusedRatherThanStringified() returns error? {
    ai:AudioDocument audio = {content: [0x01, 0x02]};
    ContentPart[]|ai:Error parts = contentToParts(`Transcribe ${audio}`);
    test:assertTrue(parts is ai:Error);
    if parts is ai:Error {
        test:assertTrue(parts.message().includes("text and image"), parts.message());
    }
}

// ---- Text behaviour must be byte-identical to before ----

@test:Config {}
function testAPlainTextPromptStillProducesExactlyOneTextPart() returns error? {
    // The regression guard for the refactor: every existing golden body depends on a
    // text-only prompt collapsing to a single string, not to a parts array.
    ContentPart[] parts = check contentToParts(`Hello ${"world"}, ${42} times`);
    test:assertEquals(parts.length(), 1);
    TextPart text = check parts[0].ensureType();
    test:assertEquals(text.text, "Hello world, 42 times");
}

// ---- Observability must not carry the payload ----

@test:Config {}
function testSpanProjectionRedactsImageBytes() returns error? {
    // A span is exported to the caller's telemetry backend. Putting the image there
    // would bloat every trace and ship user data somewhere never meant to hold it.
    string projected = partsForSpan([{text: "look "}, {mimeType: "image/png", data: PNG_BYTES}]);
    test:assertEquals(projected, string `look [image image/png, ${PNG_BYTES.length()} bytes]`);
    test:assertFalse(projected.includes(array:toBase64(PNG_BYTES)), "bytes must never reach the span");
}

// ---- URL handling ----

@test:Config {}
function testNonHttpImageUrlIsRefusedBeforeAnyFetch() returns error? {
    // `s3://`, `file://` and friends all satisfy the ai:Url constraint. None of them
    // may reach an HTTP client — this connector holds AWS credentials.
    ai:ImageDocument doc = {content: "file:///etc/passwd"};
    ContentPart[]|ai:Error parts = contentToParts(`${doc}`);
    test:assertTrue(parts is ai:Error);
    if parts is ai:Error {
        test:assertTrue(parts.message().includes("http"), parts.message());
    }
}

// ---- Emitters for the not-yet-verified dialects ----
// These are wired but gated behind `enableUnverifiedImageRoutes`. The shapes are
// pinned here so that flipping the flag changes only WHETHER they run, never WHAT
// they emit — the live test then validates the exact bytes this module produces.

@test:Config {}
function testOpenAIContentStaysABareStringUntilAnImageAppears() returns error? {
    // Backward compatibility: a text-only turn must keep emitting `"content": "..."`.
    // Some Bedrock-hosted OpenAI-shaped models accept only a string, and every
    // existing golden body assumes it.
    test:assertEquals(openAIContentParts([{text: "just text"}]), <json>"just text");

    test:assertEquals(openAIContentParts([{text: "look "}, {mimeType: "image/png", data: PNG_BYTES}]),
            <json>[
                {"type": "text", "text": "look "},
                {
                    "type": "image_url",
                    "image_url": {"url": string `data:image/png;base64,${array:toBase64(PNG_BYTES)}`}
                }
            ]);
}

@test:Config {}
function testResponsesEmitsImageUrlAsABareStringNotAnObject() returns error? {
    // The Responses dialect differs from Chat-Completions here: `image_url` is a
    // plain string. Emitting the object form is a 400.
    test:assertEquals(responsesContentParts([{mimeType: "image/gif", data: GIF_BYTES}]),
            <json>[
                {"type": "input_image", "image_url": string `data:image/gif;base64,${array:toBase64(GIF_BYTES)}`}
            ]);
}
