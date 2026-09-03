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
import ballerina/uuid;

// Pure conversions between `ai:Chunk`/`ai:Document`/`ai:QueryMatch` and the Bedrock
// `KnowledgeBaseDocument` / `KnowledgeBaseRetrievalResult` wire shapes. No I/O —
// golden-file testable directly.

const string KB_CONTENT_TYPE_TEXT = "TEXT";

// Bedrock's own limits on `DocumentMetadata.inlineAttributes` — validated here so a
// caller sees a clear message instead of an opaque 400 from `IngestKnowledgeBaseDocuments`.
// https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_DocumentMetadata.html
// https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_MetadataAttribute.html
// https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_MetadataAttributeValue.html
const int MAX_INLINE_ATTRIBUTES = 50;
const int MAX_METADATA_KEY_LENGTH = 200;
const int MAX_STRING_VALUE_LENGTH = 2048;
const int MAX_STRING_LIST_LENGTH = 10;

// `ai:Chunk`/`ai:Document` -> a `KnowledgeBaseDocument` JSON fragment (the
// `documents[]` array element for `IngestKnowledgeBaseDocuments`), plus the
// `customDocumentIdentifier.id` it was given — the caller needs that id back to
// poll `GetKnowledgeBaseDocuments` for the document it just submitted.
//
// Only `ai:TextChunk`/`ai:TextDocument` are supported: a non-text chunk returns a
// clean `ai:Error` rather than being silently dropped from the batch, matching this
// module's existing multimodal stance (see content_parts.bal).
isolated function chunkToKnowledgeBaseDocument(ai:Chunk|ai:Document chunk) returns [json, string]|ai:Error {
    string content;
    if chunk is ai:TextChunk {
        content = chunk.content;
    } else if chunk is ai:TextDocument {
        content = chunk.content;
    } else {
        return error ai:Error(
            string `Unsupported document type '${chunk.'type}': BedrockManagedKnowledgeBase only ingests ` +
            "text content ('ai:TextChunk'/'ai:TextDocument'). Convert non-text content to text before ingesting.");
    }

    ai:Metadata? metadata = chunk.metadata;
    string documentId = documentIdFrom(metadata) ?: uuid:createRandomUuid();

    map<json> documentJson = {
        content: {
            dataSourceType: "CUSTOM",
            custom: {
                sourceType: "IN_LINE",
                customDocumentIdentifier: {id: documentId},
                inlineContent: {'type: "TEXT", textContent: {data: content}}
            }
        }
    };
    if metadata is ai:Metadata {
        json? metadataJson = check metadataToDocumentMetadata(metadata);
        if metadataJson is json {
            documentJson["metadata"] = metadataJson;
        }
    }
    return [documentJson, documentId];
}

// The document id to submit: `ai:Metadata.id` (an int field already on the shared
// `ai` module type) stringified when present, so a caller who wants deterministic,
// re-ingestable ids can supply one; otherwise a fresh UUID, mirroring the Azure
// knowledge base precedent's fallback.
isolated function documentIdFrom(ai:Metadata? metadata) returns string? {
    if metadata is () {
        return ();
    }
    int? id = metadata.id;
    return id is int ? id.toString() : ();
}

// `ai:Metadata` -> `DocumentMetadata` (`IN_LINE_ATTRIBUTE`). Returns `()` when there
// is nothing to send (an empty or all-`()` metadata record) so the caller omits the
// `metadata` field entirely.
isolated function metadataToDocumentMetadata(ai:Metadata metadata) returns json?|ai:Error {
    json[] attributes = [];
    foreach [string, json] [key, value] in metadata.entries() {
        if value is () {
            continue;
        }
        if key.length() > MAX_METADATA_KEY_LENGTH {
            return error ai:Error(
                string `Metadata key '${key}' exceeds Bedrock's ${MAX_METADATA_KEY_LENGTH}-character limit`);
        }
        attributes.push({key, value: check toMetadataAttributeValue(key, value)});
    }
    if attributes.length() == 0 {
        return ();
    }
    if attributes.length() > MAX_INLINE_ATTRIBUTES {
        return error ai:Error(
            string `${attributes.length()} metadata attributes exceed Bedrock's ` +
            string `${MAX_INLINE_ATTRIBUTES}-attribute-per-document limit`);
    }
    return {'type: "IN_LINE_ATTRIBUTE", inlineAttributes: attributes};
}

// One `ai:Metadata` value -> a typed `MetadataAttributeValue`. Bedrock supports
// exactly four shapes (`BOOLEAN`/`NUMBER`/`STRING`/`STRING_LIST`); anything else
// (nested objects, mixed-type arrays, ...) is a clear `ai:Error` rather than a
// silent drop or a lossy `toString()`.
isolated function toMetadataAttributeValue(string key, json value) returns json|ai:Error {
    if value is boolean {
        return {'type: "BOOLEAN", booleanValue: value};
    }
    if value is int || value is float || value is decimal {
        return {'type: "NUMBER", numberValue: <float>value};
    }
    if value is string {
        if value.length() > MAX_STRING_VALUE_LENGTH {
            return error ai:Error(string `Metadata value for key '${key}' exceeds Bedrock's ` +
                string `${MAX_STRING_VALUE_LENGTH}-character 'STRING' limit`);
        }
        return {'type: "STRING", stringValue: value};
    }
    if value is json[] {
        if value.length() > MAX_STRING_LIST_LENGTH {
            return error ai:Error(string `Metadata value for key '${key}' has ${value.length()} entries, ` +
                string `exceeding Bedrock's ${MAX_STRING_LIST_LENGTH}-entry 'STRING_LIST' limit`);
        }
        string[] items = [];
        foreach json item in value {
            if item !is string {
                return error ai:Error(
                    string `Metadata key '${key}' is an array containing a non-string element ` +
                    string `(${item.toJsonString()}); only string arrays map to Bedrock's 'STRING_LIST'`);
            }
            items.push(item);
        }
        return {'type: "STRING_LIST", stringListValue: items};
    }
    return error ai:Error(
        string `Metadata key '${key}' has a value Bedrock cannot represent as an inline attribute ` +
        string `(supported: boolean, number, string, string[]); got ${value.toJsonString()}`);
}

// A `KnowledgeBaseRetrievalResult` (one element of `Retrieve`'s `retrievalResults[]`)
// -> `ai:QueryMatch`. Only `content.type == "TEXT"` is supported — `IMAGE`/`ROW`/
// `AUDIO`/`VIDEO` have no `ai:TextChunk`-shaped representation, so they are a clean
// `ai:Error` rather than a silently empty or truncated chunk.
//
// `metadata` is passed through VERBATIM, including the six underscore-prefixed
// system attributes Bedrock injects (`_source_uri`, `_chunk_id`, `_data_source_id`,
// `_data_source_type`, `_file_type`, `_language_code`) — `_source_uri` in particular
// is what makes `deleteByFilter`'s probe possible at all (see knowledgebase_managed.bal).
isolated function retrievalResultToQueryMatch(json result) returns ai:QueryMatch|ai:Error {
    map<json> resultMap = result is map<json> ? result : {};
    json contentJson = resultMap["content"] ?: {};
    map<json> content = contentJson is map<json> ? contentJson : {};
    json contentTypeJson = content["type"] ?: KB_CONTENT_TYPE_TEXT;
    string contentType = contentTypeJson is string ? contentTypeJson : contentTypeJson.toString();
    if contentType != KB_CONTENT_TYPE_TEXT {
        return error ai:Error(
            string `BedrockManagedKnowledgeBase only supports TEXT retrieval results; got '${contentType}'. ` +
            "Non-text content (IMAGE/ROW/AUDIO/VIDEO) has no 'ai:TextChunk'-shaped representation.");
    }
    json textJson = content["text"] ?: "";
    string text = textJson is string ? textJson : textJson.toString();

    ai:Metadata metadata = {};
    json metadataJson = resultMap["metadata"] ?: {};
    if metadataJson is map<json> {
        foreach [string, json] [key, value] in metadataJson.entries() {
            metadata[key] = value;
        }
    }

    json scoreJson = resultMap["score"] ?: 0;
    float similarityScore = toFloatScore(scoreJson);

    ai:TextChunk chunk = {content: text, metadata};
    return {chunk, similarityScore};
}

isolated function toFloatScore(json value) returns float {
    if value is float {
        return value;
    }
    if value is int {
        return <float>value;
    }
    if value is decimal {
        return <float>value;
    }
    return 0.0;
}
