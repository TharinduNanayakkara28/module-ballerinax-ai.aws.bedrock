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
import ballerina/lang.array;

// Multimodal content resolution.
//
// An `ai:Prompt` may carry `ai:Document`/`ai:Chunk` insertions, including images.
// This file flattens a prompt into ordered `ContentPart`s ONCE, before any converter
// runs, and every dialect then maps those parts onto its own wire shape.
//
// WHY A PRE-PASS RATHER THAN WORK INSIDE THE CONVERTERS: resolving an `ai:Url` image
// requires an HTTP GET. Converters are pure `RequestEncoder` functions that the
// golden-file tests drive with no network and no credentials, and that property is
// load-bearing. So all I/O happens here; the per-dialect emitters below are pure.
//
// NORMAL FORM IS BYTES + A CONCRETE MIME TYPE, not a data URL and not a URL:
//   - Converse's `format` is a bare token (`png`), so a data URL would have to be
//     re-parsed to recover it — a lossy round-trip.
//   - Converse has no URL member at all, and Anthropic-on-Bedrock accepts base64
//     only ("On Amazon Bedrock and Google Cloud, only base64-encoded sources are
//     currently available" — https://platform.claude.com/docs/en/build-with-claude/vision).
// Bytes+mime is a superset: every dialect's shape derives from it, none of them
// derive back.

# One part of a user turn after its `ai:Prompt` has been flattened.
type ContentPart TextPart|ImagePart;

# Literal text.
type TextPart record {|
    readonly "text" kind = "text";
    string text;
|};

# An image, always as raw bytes plus a concrete IANA type.
type ImagePart record {|
    readonly "image" kind = "image";
    # Concrete type — never a wildcard. Both Converse's `format` and Anthropic's
    # `media_type` are derived from this, and neither accepts `image/*`.
    string mimeType;
    # UNencoded bytes. Each emitter base64-encodes at its own wire boundary.
    byte[] data;
|};

# A user message whose content has been resolved to parts. Assistant and function
# messages are unchanged — neither can carry an image.
type ResolvedUserMessage record {|
    ai:USER role = ai:USER;
    ContentPart[] parts;
|};

type ResolvedMessage ResolvedUserMessage|ai:ChatAssistantMessage|ai:ChatFunctionMessage;

// The only image formats ANY Bedrock dialect accepts. Converse constrains `format`
// to exactly these four tokens, and Anthropic's `media_type` to their `image/*` form.
// https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageBlock.html
final readonly & map<string> MIME_TO_CONVERSE_FORMAT = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp"
};

const MAX_IMAGE_DOWNLOAD_BYTES = 20 * 1024 * 1024; // 20 MiB
const MAX_IMAGE_REDIRECTS = 5;

// ---------------------------------------------------------------------------
// Resolution (does I/O)
// ---------------------------------------------------------------------------

// Hoists system content (as text) and resolves every user turn to parts.
// Runs once per chat()/generate(), before the converter.
isolated function resolveMessages(ai:ChatMessage[] messages)
        returns [string?, ResolvedMessage[]]|ai:Error {
    string[] systemParts = [];
    ResolvedMessage[] rest = [];
    foreach ai:ChatMessage m in messages {
        if m is ai:ChatSystemMessage {
            // System is text-only on EVERY route: Converse's SystemContentBlock is
            // `text | guardContent | cachePoint`, Anthropic's `system` is a string or
            // text blocks, and Responses' `instructions` is a string. None has an
            // image member, so an image here must be named rather than flattened.
            systemParts.push(check contentToText(m.content, "a system message"));
        } else if m is ai:ChatUserMessage {
            rest.push({parts: check contentToParts(m.content)});
        } else if m is ai:ChatAssistantMessage|ai:ChatFunctionMessage {
            rest.push(m);
        }
    }
    string? system = systemParts.length() == 0 ? () : string:'join("\n\n", ...systemParts);
    return [system, rest];
}

// Flattens a user turn's content into ordered parts, fetching any image URL.
// Adjacent text is merged so a prompt with no documents yields exactly one TextPart
// and every dialect keeps emitting the same wire bytes it does today.
isolated function contentToParts(string|ai:Prompt content) returns ContentPart[]|ai:Error {
    if content is string {
        return content == "" ? [] : [{text: content}];
    }
    string[] & readonly strings = content.strings;
    anydata[] insertions = content.insertions;
    ContentPart[] parts = [];
    string text = strings.length() > 0 ? strings[0] : "";

    foreach int i in 0 ..< insertions.length() {
        anydata insertion = insertions[i];
        if insertion is ai:Document|ai:Chunk {
            text = flushText(text, parts);
            check appendDocument(insertion, parts);
        } else if insertion is (ai:Document|ai:Chunk)[] {
            text = flushText(text, parts);
            foreach ai:Document|ai:Chunk doc in insertion {
                check appendDocument(doc, parts);
            }
        } else {
            // A plain interpolation — unchanged from the previous behaviour.
            text += insertion.toString();
        }
        if i + 1 < strings.length() {
            text += strings[i + 1];
        }
    }
    _ = flushText(text, parts);
    return parts;
}

// Renders content to plain text, refusing anything that is not text. `sink` names
// the surface that cannot carry the image, so the error tells the caller where the
// problem is rather than just that there is one.
isolated function contentToText(string|ai:Prompt content, string sink) returns string|ai:Error {
    ContentPart[] parts = check contentToParts(content);
    string text = "";
    foreach ContentPart part in parts {
        if part is ImagePart {
            return error ai:Error(string `Images are not supported in ${sink}. ` +
                "Move the image into a user message on a Converse or Anthropic route.");
        }
        text += part.text;
    }
    return text;
}

// Appends a document/chunk as a part. Text and image only — Converse does model
// `document`/`video`/`audio` blocks, so this is a deliberate scope line rather than
// a Bedrock limitation.
isolated function appendDocument(ai:Document|ai:Chunk doc, ContentPart[] parts) returns ai:Error? {
    if doc is ai:TextDocument|ai:TextChunk {
        string text = doc.content;
        if text != "" {
            parts.push({text});
        }
        return;
    }
    if doc is ai:ImageDocument {
        parts.push(check toImagePart(doc));
        return;
    }
    return error ai:Error("Only text and image documents are supported.");
}

// Resolves an `ai:ImageDocument` to bytes + a concrete MIME type.
isolated function toImagePart(ai:ImageDocument doc) returns ImagePart|ai:Error {
    ai:Url|byte[] content = doc.content;
    byte[] data;
    string? mimeType = normalizeMimeType(doc.metadata?.mimeType);
    if content is ai:Url {
        // Bedrock never fetches on our behalf: Converse has no URL source and
        // Anthropic-on-Bedrock is base64-only. So the connector fetches, and the
        // behaviour is uniform across every dialect rather than working on some.
        [byte[], string?] [downloaded, contentType] = check downloadImage(content);
        data = downloaded;
        mimeType = mimeType ?: normalizeMimeType(contentType);
    } else {
        data = content;
    }
    // Sniff last: it is the only source that cannot be wrong, but an explicit
    // metadata.mimeType is the caller's stated intent and wins.
    string? resolved = mimeType ?: sniffImageMime(data);
    if resolved is () {
        return error ai:Error("Could not determine the image type. Set " +
            "'metadata.mimeType' to one of image/png, image/jpeg, image/gif, image/webp.");
    }
    if !MIME_TO_CONVERSE_FORMAT.hasKey(resolved) {
        return error ai:Error(string `Unsupported image type '${resolved}'. Bedrock ` +
            "accepts image/png, image/jpeg, image/gif and image/webp.");
    }
    return {mimeType: resolved, data};
}

// Merges accumulated text into a part and resets the accumulator.
isolated function flushText(string text, ContentPart[] parts) returns string {
    if text != "" {
        parts.push({text});
    }
    return "";
}

// ---------------------------------------------------------------------------
// MIME handling
// ---------------------------------------------------------------------------

// Lowercases, strips any `; charset=...` parameter, and maps the common non-IANA
// `image/jpg` onto `image/jpeg` (which is the only spelling Bedrock accepts).
isolated function normalizeMimeType(string? raw) returns string? {
    if raw is () {
        return ();
    }
    string value = raw.trim().toLowerAscii();
    int? semi = value.indexOf(";");
    if semi is int {
        value = value.substring(0, semi).trim();
    }
    if value == "" || value == "image/*" || value == "application/octet-stream" {
        // Not concrete enough for `format`/`media_type`; fall through to sniffing.
        return ();
    }
    return value == "image/jpg" ? "image/jpeg" : value;
}

// Identifies the four supported formats from their magic bytes. Total over that set,
// so the "could not determine" error only fires for genuinely unsupported data.
isolated function sniffImageMime(byte[] data) returns string? {
    if startsWithBytes(data, [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return "image/png";
    }
    if startsWithBytes(data, [0xFF, 0xD8, 0xFF]) {
        return "image/jpeg";
    }
    if startsWithBytes(data, [0x47, 0x49, 0x46, 0x38]) { // "GIF8"
        return "image/gif";
    }
    // WebP: "RIFF" .... "WEBP" — the size field sits between the two markers.
    if startsWithBytes(data, [0x52, 0x49, 0x46, 0x46]) && data.length() >= 12
        && data[8] == 0x57 && data[9] == 0x45 && data[10] == 0x42 && data[11] == 0x50 {
        return "image/webp";
    }
    return ();
}

isolated function startsWithBytes(byte[] data, int[] prefix) returns boolean {
    if data.length() < prefix.length() {
        return false;
    }
    foreach int i in 0 ..< prefix.length() {
        if <int>data[i] != prefix[i] {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Download (the only I/O here)
// ---------------------------------------------------------------------------

// Fetches an image URL, returning its bytes and any `Content-Type`.
//
// Redirects are followed MANUALLY so that every hop is scheme-checked. Letting the
// HTTP client follow them would check only the first URL, so a public origin could
// bounce the fetch to an internal address — and this connector holds AWS credentials,
// which makes it a more valuable SSRF target than most.
isolated function downloadImage(string url) returns [byte[], string?]|ai:Error {
    string target = url;
    int redirects = 0;
    while true {
        check validateDownloadTarget(target);
        [string, string] [origin, path] = check splitUrl(target);
        http:Client|error cl = new (origin, {followRedirects: {enabled: false}});
        if cl is error {
            return error ai:Error(string `Could not open a connection to '${origin}'`, cl);
        }
        http:Response|error resp = cl->get(path);
        if resp is error {
            return error ai:Error(string `Failed to download the image from '${target}'`, resp);
        }
        int status = resp.statusCode;
        if status >= 300 && status < 400 {
            if redirects >= MAX_IMAGE_REDIRECTS {
                return error ai:Error(string `Too many redirects (>${MAX_IMAGE_REDIRECTS}) ` +
                    string `while downloading '${url}'`);
            }
            string|error location = resp.getHeader("Location");
            if location is error {
                return error ai:Error(string `Redirect from '${target}' had no Location header`);
            }
            target = resolveRedirect(target, location);
            redirects += 1;
            continue;
        }
        if status < 200 || status >= 300 {
            return error ai:Error(string `Downloading '${target}' returned HTTP ${status}`);
        }
        byte[]|error payload = resp.getBinaryPayload();
        if payload is error {
            return error ai:Error(string `Could not read the image bytes from '${target}'`, payload);
        }
        if payload.length() > MAX_IMAGE_DOWNLOAD_BYTES {
            return error ai:Error(string `Image at '${target}' exceeds the ` +
                string `${MAX_IMAGE_DOWNLOAD_BYTES} byte download limit`);
        }
        string|error contentType = resp.getHeader("Content-Type");
        return [payload, contentType is string ? contentType : ()];
    }
}

// Only http/https may be fetched. `s3://`, `file://`, `gopher://` and friends are all
// valid `ai:Url` values, and none of them should reach an HTTP client.
isolated function validateDownloadTarget(string url) returns ai:Error? {
    string lower = url.toLowerAscii();
    if lower.startsWith("https://") || lower.startsWith("http://") {
        return;
    }
    return error ai:Error(string `Only http(s) image URLs can be downloaded; got '${url}'. ` +
        "Pass the image as a byte array instead.");
}

// Splits an absolute URL into [origin, path-with-query].
isolated function splitUrl(string url) returns [string, string]|ai:Error {
    int? schemeEnd = url.indexOf("://");
    if schemeEnd is () {
        return error ai:Error(string `Malformed image URL '${url}'`);
    }
    int hostStart = schemeEnd + 3;
    string rest = url.substring(hostStart);
    int? slash = rest.indexOf("/");
    if slash is () {
        return [url, "/"];
    }
    return [url.substring(0, hostStart + slash), rest.substring(slash)];
}

// Resolves a Location header against the URL it came from (absolute or root-relative).
isolated function resolveRedirect(string base, string location) returns string {
    string target = location.trim();
    if target.toLowerAscii().startsWith("http://") || target.toLowerAscii().startsWith("https://") {
        return target;
    }
    [string, string]|ai:Error parts = splitUrl(base);
    if parts is ai:Error {
        return target;
    }
    string origin = parts[0];
    return target.startsWith("/") ? origin + target : origin + "/" + target;
}

// ---------------------------------------------------------------------------
// Per-dialect emitters — PURE. This is the whole mapping table.
// ---------------------------------------------------------------------------

// Converse / Nova-Invoke content blocks.
// `{"image": {"format": "png", "source": {"bytes": "<base64>"}}}`
// https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageBlock.html
isolated function converseContentBlocks(ContentPart[] parts) returns json[] {
    json[] blocks = [];
    foreach ContentPart part in parts {
        if part is TextPart {
            blocks.push({"text": part.text});
        } else {
            blocks.push({
                "image": {
                    // A bare token here, NOT the full MIME type.
                    "format": MIME_TO_CONVERSE_FORMAT.get(part.mimeType),
                    "source": {"bytes": array:toBase64(part.data)}
                }
            });
        }
    }
    return blocks;
}

// Anthropic Messages content blocks (Invoke-Anthropic AND Mantle Messages).
// `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"..."}}`
// https://platform.claude.com/docs/en/api/messages
isolated function anthropicContentBlocks(ContentPart[] parts) returns json[] {
    json[] blocks = [];
    foreach ContentPart part in parts {
        if part is TextPart {
            blocks.push({"type": "text", "text": part.text});
        } else {
            blocks.push({
                "type": "image",
                "source": {
                    "type": "base64", // Bedrock does not accept the `url` source type.
                    "media_type": part.mimeType,
                    "data": array:toBase64(part.data)
                }
            });
        }
    }
    return blocks;
}

// OpenAI Chat-Completions content (also the Mistral chat dialect, which copies it).
//
// Returns a BARE STRING when there is no image, exactly as before, so every existing
// golden body stays byte-identical and text-only models that accept only a string
// keep working. The parts array appears only once an image is present.
// `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`
isolated function openAIContentParts(ContentPart[] parts) returns json {
    if !hasImage(parts) {
        return partsText(parts);
    }
    json[] out = [];
    foreach ContentPart part in parts {
        if part is TextPart {
            out.push({"type": "text", "text": part.text});
        } else {
            out.push({"type": "image_url", "image_url": {"url": dataUri(part)}});
        }
    }
    return out;
}

// OpenAI Responses content. Note `image_url` is a BARE STRING here, not the object
// the Chat-Completions dialect uses — the same asymmetry this module already encodes
// for tool choice.
isolated function responsesContentParts(ContentPart[] parts) returns json[] {
    json[] out = [];
    foreach ContentPart part in parts {
        if part is TextPart {
            out.push({"type": "input_text", "text": part.text});
        } else {
            out.push({"type": "input_image", "image_url": dataUri(part)});
        }
    }
    return out;
}

// `data:<mime>;base64,<data>` — the only way the OpenAI-shaped dialects take bytes.
isolated function dataUri(ImagePart part) returns string
    => string `data:${part.mimeType};base64,${array:toBase64(part.data)}`;

// True when any part is an image — used by the dialects whose image support is not
// yet confirmed, so they can refuse rather than silently drop it.
isolated function hasImage(ContentPart[] parts) returns boolean {
    foreach ContentPart part in parts {
        if part is ImagePart {
            return true;
        }
    }
    return false;
}

# Sends images on the dialects whose image support is NOT confirmed by any
# first-party source — the OpenAI-shaped Mantle/Invoke paths and Mistral's chat
# dialect. Defaults to `false`, which refuses them at construction.
#
# This exists so the behaviour can be verified END TO END through the module, against
# the exact bytes it emits, rather than against hand-written JSON that may not match.
# Set it in `Config.toml`, run the live image tests, and if the route accepts the
# request the default here flips permanently:
#
#     [ballerinax.ai.aws.bedrock]
#     enableUnverifiedImageRoutes = true
#
# Unsupported and experimental: AWS may reject these requests outright.
public configurable boolean enableUnverifiedImageRoutes = false;

// Refuses a request carrying an image on a dialect that cannot express one — either
// because it has no content-part array at all (the prompt-template dialects), or
// because its image support is NOT yet verified against a primary source.
//
// Called at the TOP of the affected encoders, so the refusal happens before any body
// is built and no image can be silently dropped downstream.
//
// `unverifiedOnly` marks the second class: those dialects DO have a content-part
// shape and an emitter ready, so `enableUnverifiedImageRoutes` lets a live test push
// real bytes through them. The prompt-template dialects pass `false` — no flag can
// make a single string carry an image.
isolated function rejectImagesIn(ResolvedMessage[] messages, string dialect,
        boolean unverifiedOnly = false) returns ai:Error? {
    if unverifiedOnly && enableUnverifiedImageRoutes {
        return;
    }
    foreach ResolvedMessage m in messages {
        if m is ResolvedUserMessage && hasImage(m.parts) {
            return error ai:Error(string `Image input is not supported on ${dialect}. ` +
                "Use 'apiFamily = CONVERSE', or an Anthropic model — both accept images.");
        }
    }
}

// Concatenates the text of already-image-free parts.
//
// INVARIANT: every caller runs `rejectImagesIn` first, so an `ImagePart` cannot reach
// here. Dropping one silently is the exact bug this file exists to fix, so the guard
// belongs at the encoder boundary rather than being re-checked per message.
isolated function partsText(ContentPart[] parts) returns string {
    string text = "";
    foreach ContentPart part in parts {
        if part is TextPart {
            text += part.text;
        }
    }
    return text;
}

// Redacted projection for the observe span. The span must never carry the payload:
// an image is potentially megabytes, and it is user data that would otherwise be
// shipped verbatim to whatever telemetry backend is configured.
isolated function partsForSpan(ContentPart[] parts) returns string {
    string text = "";
    foreach ContentPart part in parts {
        if part is TextPart {
            text += part.text;
        } else {
            text += string `[image ${part.mimeType}, ${part.data.length()} bytes]`;
        }
    }
    return text;
}
