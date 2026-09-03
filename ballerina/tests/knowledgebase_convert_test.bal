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
import ballerina/test;

// Golden tests for `ai:Chunk`/`ai:Document` <-> Bedrock wire shapes. No I/O.

@test:Config {}
function testTextChunkEncodesAsCustomInLineTextDocument() returns error? {
    ai:TextChunk chunk = {content: "hello world"};
    [json, string] [wireDoc, id] = check chunkToKnowledgeBaseDocument(chunk);
    test:assertTrue(id.length() > 0, "a document id must always be assigned");
    map<json> content = <map<json>>(<map<json>>wireDoc)["content"];
    test:assertEquals(content["dataSourceType"], "CUSTOM");
    map<json> custom = <map<json>>content["custom"];
    test:assertEquals(custom["sourceType"], "IN_LINE");
    test:assertEquals((<map<json>>custom["customDocumentIdentifier"])["id"], id);
    map<json> inlineContent = <map<json>>custom["inlineContent"];
    test:assertEquals(inlineContent["type"], "TEXT");
    test:assertEquals((<map<json>>inlineContent["textContent"])["data"], "hello world");
    test:assertFalse((<map<json>>wireDoc).hasKey("metadata"), "no metadata on the chunk means no metadata field");
}

@test:Config {}
function testMetadataIntIdIsUsedAsTheDocumentId() returns error? {
    ai:TextChunk chunk = {content: "hi", metadata: {id: 42}};
    [json, string] [_, id] = check chunkToKnowledgeBaseDocument(chunk);
    test:assertEquals(id, "42");
}

@test:Config {}
function testNonTextDocumentReturnsACleanError() {
    ai:Document doc = {'type: "image", content: "base64-blob-not-text"};
    [json, string]|ai:Error result = chunkToKnowledgeBaseDocument(doc);
    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        test:assertTrue(result.message().includes("text"), result.message());
    }
}

@test:Config {}
function testEachMetadataAttributeValueTypeIsMappedCorrectly() returns error? {
    ai:TextChunk chunk = {
        content: "hi",
        metadata: {
            "flag": true,
            "count": 7,
            "price": 9.99,
            "label": "acme",
            "tags": ["a", "b", "c"]
        }
    };
    [json, string] [wireDoc, _] = check chunkToKnowledgeBaseDocument(chunk);
    map<json> metadata = <map<json>>(<map<json>>wireDoc)["metadata"];
    test:assertEquals(metadata["type"], "IN_LINE_ATTRIBUTE");
    json[] attributes = <json[]>metadata["inlineAttributes"];
    map<json> byKey = {};
    foreach json attr in attributes {
        map<json> a = <map<json>>attr;
        byKey[<string>a["key"]] = a["value"];
    }
    test:assertEquals(byKey["flag"], {'type: "BOOLEAN", booleanValue: true});
    test:assertEquals(byKey["count"], {'type: "NUMBER", numberValue: 7.0});
    test:assertEquals(byKey["label"], {'type: "STRING", stringValue: "acme"});
    test:assertEquals(byKey["tags"], {'type: "STRING_LIST", stringListValue: ["a", "b", "c"]});
}

@test:Config {}
function testMixedTypeArrayMetadataIsAClearError() {
    ai:TextChunk chunk = {content: "hi", metadata: {"bad": [1, "two", 3]}};
    [json, string]|ai:Error result = chunkToKnowledgeBaseDocument(chunk);
    test:assertTrue(result is ai:Error);
}

@test:Config {}
function testOversizedStringMetadataValueIsRejected() {
    string tooLong = "";
    foreach int i in 0 ..< 2049 {
        tooLong += "x";
    }
    ai:TextChunk chunk = {content: "hi", metadata: {"big": tooLong}};
    [json, string]|ai:Error result = chunkToKnowledgeBaseDocument(chunk);
    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        test:assertTrue(result.message().includes("2048"), result.message());
    }
}

@test:Config {}
function testMoreThanFiftyMetadataAttributesIsRejected() {
    ai:Metadata metadata = {};
    foreach int i in 0 ..< 51 {
        metadata["k" + i.toString()] = "v";
    }
    ai:TextChunk chunk = {content: "hi", metadata};
    [json, string]|ai:Error result = chunkToKnowledgeBaseDocument(chunk);
    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        test:assertTrue(result.message().includes("50"), result.message());
    }
}

// ---- retrieval result -> ai:QueryMatch ----

@test:Config {}
function testTextRetrievalResultMapsToATextChunkWithScoreAndMetadata() returns error? {
    json result = {
        content: {text: "the chunk text", 'type: "TEXT"},
        documentId: "doc-1",
        location: {'type: "CUSTOM", customDocumentLocation: {id: "doc-1"}},
        metadata: {"_source_uri": "doc-1", "tenant": "acme"},
        score: 0.87
    };
    ai:QueryMatch queryMatch = check retrievalResultToQueryMatch(result);
    test:assertEquals(queryMatch.similarityScore, 0.87);
    ai:Chunk chunk = queryMatch.chunk;
    test:assertTrue(chunk is ai:TextChunk);
    if chunk is ai:TextChunk {
        test:assertEquals(chunk.content, "the chunk text");
    }
    ai:Metadata? metadata = chunk.metadata;
    test:assertTrue(metadata is ai:Metadata);
    if metadata is ai:Metadata {
        test:assertEquals(metadata["_source_uri"], "doc-1");
        test:assertEquals(metadata["tenant"], "acme");
    }
}

@test:Config {}
function testNonTextRetrievalResultIsACleanError() {
    json result = {
        content: {'type: "IMAGE", byteContent: "data:image/jpeg;base64,xyz"},
        score: 0.5
    };
    ai:QueryMatch|ai:Error queryMatch = retrievalResultToQueryMatch(result);
    test:assertTrue(queryMatch is ai:Error);
    if queryMatch is ai:Error {
        test:assertTrue(queryMatch.message().includes("TEXT"), queryMatch.message());
    }
}
