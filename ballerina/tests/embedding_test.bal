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

// Embedding tests.

// ---- converter golden files ----

@test:Config {}
function testTitanEmitsInputTextAsAString() returns error? {
    // `inputText` is a STRING, not an array — the whole reason maxBatchSize is 1.
    map<json> body = check encodeTitanEmbed(["hello"], {}).ensureType();
    test:assertEquals(body["inputText"], "hello");
    test:assertTrue(body["inputText"] is string, "inputText must be a string, never an array");
    test:assertFalse(body.hasKey("texts"), "Titan must not emit Cohere's 'texts' field");
}

@test:Config {}
function testTitanForwardsDimensionsAndNormalize() returns error? {
    map<json> body = check encodeTitanEmbed(["hi"], {dimensions: 512, normalize: true}).ensureType();
    test:assertEquals(body["dimensions"], 512);
    test:assertEquals(body["normalize"], true);
}

@test:Config {}
function testTitanRejectsMoreThanOneTextPerCall() {
    json|ai:Error body = encodeTitanEmbed(["a", "b"], {});
    test:assertTrue(body is ai:Error, "Titan embeds exactly one text per call");
}

@test:Config {}
function testCohereEmitsTextsAsAnArrayAndAlwaysEmitsInputType() returns error? {
    // input_type is REQUIRED on every request.
    map<json> body = check encodeCohereEmbedV3(["a", "b"], {inputType: SEARCH_DOCUMENT}).ensureType();
    test:assertEquals(body["texts"], <json>["a", "b"]);
    test:assertTrue(body["texts"] is json[], "texts must be an array, never a string");
    test:assertEquals(body["input_type"], "search_document");
    test:assertFalse(body.hasKey("inputText"), "Cohere must not emit Titan's 'inputText' field");
}

@test:Config {}
function testCohereQueryInputTypeIsDistinctFromDocument() returns error? {
    // Getting these backwards silently degrades retrieval.
    map<json> query = check encodeCohereEmbedV3(["q"], {inputType: SEARCH_QUERY}).ensureType();
    test:assertEquals(query["input_type"], "search_query");
}

@test:Config {}
function testCohereEncodeFailsWithoutInputType() {
    json|ai:Error body = encodeCohereEmbedV3(["a"], {});
    test:assertTrue(body is ai:Error);
    if body is ai:Error {
        test:assertTrue(body.message().includes("input_type"), body.message());
    }
}

@test:Config {}
function testCohereRejectsOverSizedWindow() {
    string[] texts = [];
    foreach int i in 0 ..< 97 {
        texts.push(string `t${i}`);
    }
    json|ai:Error body = encodeCohereEmbedV3(texts, {inputType: SEARCH_DOCUMENT});
    test:assertTrue(body is ai:Error, "Cohere accepts at most 96 texts per call");
}

// ---- decode: the inputTokenCount asymmetry ----

@test:Config {}
function testTitanDecodePopulatesInputTokenCount() returns error? {
    json canned = {"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 5};
    DecodedEmbedding decoded = check decodeTitanEmbed(canned);
    test:assertEquals(decoded.embeddings.length(), 1);
    test:assertEquals(decoded.embeddings[0], <ai:Vector>[0.1, 0.2, 0.3]);
    test:assertEquals(decoded.inputTokenCount, 5, "Titan reports inputTextTokenCount");
    test:assertEquals(decoded.responseId, ());
}

@test:Config {}
function testCohereDecodeHasNoInputTokenCount() returns error? {
    // This is the assertion that catches a naive addInputTokenCount crash.
    json canned = {"embeddings": [[0.1, 0.2], [0.3, 0.4]], "id": "req-9", "response_type": "embeddings_floats"};
    DecodedEmbedding decoded = check decodeCohereEmbed(canned);
    test:assertEquals(decoded.embeddings.length(), 2);
    test:assertEquals(decoded.inputTokenCount, (), "Cohere's response has no token-count field at all");
    test:assertEquals(decoded.responseId, "req-9");
}

@test:Config {}
function testCohereDecodeAcceptsTypedEmbeddingsShape() returns error? {
    // Embed v4 with embedding_types returns {"embeddings": {"float": [[..]]}}.
    json canned = {"embeddings": {"float": [[0.5, 0.6]]}, "id": "req-10"};
    DecodedEmbedding decoded = check decodeCohereEmbed(canned);
    test:assertEquals(decoded.embeddings.length(), 1);
    test:assertEquals(decoded.embeddings[0], <ai:Vector>[0.5, 0.6]);
}

// ---- batching: the highest-value test ----

@test:Config {}
function testTitanBatchingIs100RequestsFor100Chunks() {
    string[] texts = [];
    foreach int i in 0 ..< 100 {
        texts.push(string `chunk-${i}`);
    }
    string[][] windows = partitionTexts(texts, TITAN_EMBED_CONVERTER.maxBatchSize);
    test:assertEquals(TITAN_EMBED_CONVERTER.maxBatchSize, 1);
    test:assertEquals(windows.length(), 100, "Titan has no batch input → 100 sequential requests");
    foreach string[] window in windows {
        test:assertEquals(window.length(), 1);
    }
}

@test:Config {}
function testCohereBatchingIs2RequestsFor100Chunks() {
    string[] texts = [];
    foreach int i in 0 ..< 100 {
        texts.push(string `chunk-${i}`);
    }
    string[][] windows = partitionTexts(texts, COHERE_EMBED_V3_CONVERTER.maxBatchSize);
    test:assertEquals(COHERE_EMBED_V3_CONVERTER.maxBatchSize, 96);
    test:assertEquals(windows.length(), 2, "ceil(100/96) == 2 requests");
    test:assertEquals(windows[0].length(), 96);
    test:assertEquals(windows[1].length(), 4);
}

@test:Config {}
function testPartitioningPreservesInputOrder() {
    // ORDER IS THE CONTRACT — assert with distinguishable values, not just lengths.
    string[] texts = [];
    foreach int i in 0 ..< 100 {
        texts.push(string `chunk-${i}`);
    }
    string[] flattened = [];
    foreach string[] window in partitionTexts(texts, 96) {
        foreach string text in window {
            flattened.push(text);
        }
    }
    test:assertEquals(flattened, texts, "windows must flatten back to the exact input order");
    test:assertEquals(flattened[0], "chunk-0");
    test:assertEquals(flattened[96], "chunk-96", "the first element of the second window");
    test:assertEquals(flattened[99], "chunk-99");
}

// ---- construction errors ----

@test:Config {}
function testTitanRejectsACohereModelId() {
    TitanEmbeddingProvider|ai:Error provider = new ("cohere.embed-english-v3", TEST_CREDS, REGION);
    test:assertTrue(provider is ai:Error);
    if provider is ai:Error {
        test:assertTrue(provider.message().includes("amazon.titan-embed"), provider.message());
    }
}

@test:Config {}
function testCohereRejectsATitanModelId() {
    CohereEmbeddingProvider|ai:Error provider = new ("amazon.titan-embed-text-v2:0", TEST_CREDS, REGION);
    test:assertTrue(provider is ai:Error);
    if provider is ai:Error {
        test:assertTrue(provider.message().includes("cohere.embed"), provider.message());
    }
}

@test:Config {}
function testEmbeddingArnIsRejectedAtConstruction() {
    // v1: ARNs are out of scope — an ARN carries no vendor prefix.
    TitanEmbeddingProvider|ai:Error provider = new (
        "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc", TEST_CREDS, REGION);
    test:assertTrue(provider is ai:Error);
    if provider is ai:Error {
        test:assertTrue(provider.message().includes("ARN"), provider.message());
    }
}

@test:Config {}
function testEmbeddingProvidersConstructAndNormalizeCris() returns error? {
    TitanEmbeddingProvider titan = check new (TITAN_EMBED_TEXT_V2, TEST_CREDS, REGION);
    CohereEmbeddingProvider cohere = check new (COHERE_EMBED_ENGLISH_V3, TEST_CREDS, REGION,
        inputType = SEARCH_QUERY);
    // Cohere Embed v4 is offered cross-region, so a CRIS-prefixed id must resolve.
    CohereEmbeddingProvider cris = check new ("us.cohere.embed-v4:0", TEST_CREDS, REGION);
    test:assertTrue(titan is TitanEmbeddingProvider);
    test:assertTrue(cohere is CohereEmbeddingProvider);
    test:assertTrue(cris is CohereEmbeddingProvider);
}

// ---- chunk type guard ----

@test:Config {}
function testNonTextChunkReturnsCleanErrorFromEmbedAndBatchEmbed() returns error? {
    // The guard fires before any I/O, so this needs no AWS.
    TitanEmbeddingProvider titan = check new (TITAN_EMBED_TEXT_V2, TEST_CREDS, REGION);
    ai:Chunk imageChunk = {'type: "image", content: "not-text"};

    ai:Embedding|ai:Error single = titan->embed(imageChunk);
    test:assertTrue(single is ai:Error);
    if single is ai:Error {
        test:assertTrue(single.message().includes("Unsupported chunk type"), single.message());
    }

    ai:Embedding[]|ai:Error batch = titan->batchEmbed([imageChunk]);
    test:assertTrue(batch is ai:Error);
    if batch is ai:Error {
        test:assertTrue(batch.message().includes("Unsupported chunk type"), batch.message());
    }
}

@test:Config {}
function testNonTextChunkRejectedByCohereToo() returns error? {
    CohereEmbeddingProvider cohere = check new (COHERE_EMBED_ENGLISH_V3, TEST_CREDS, REGION);
    ai:Chunk imageChunk = {'type: "image", content: "not-text"};
    ai:Embedding[]|ai:Error batch = cohere->batchEmbed([imageChunk]);
    test:assertTrue(batch is ai:Error);
}

// ---- Cohere: two request shapes under one vendor prefix ----

@test:Config {}
function testCohereV4UsesOutputDimensionNotDimensions() returns error? {
    // REGRESSION: the converter emitted `dimensions`, which v4 does not define. Cohere
    // would ignore it and return 1536-wide vectors while the caller believed they
    // had asked for 256 — a silent corpus-width mismatch, not an error.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html
    map<json> body = check encodeCohereEmbedV4(["hi"], {inputType: SEARCH_DOCUMENT, dimensions: 256}).ensureType();
    test:assertEquals(body["output_dimension"], <json>256);
    test:assertFalse(body.hasKey("dimensions"), "`dimensions` is not a v4 parameter");
}

@test:Config {}
function testCohereV3HasNoOutputSizeParameterAtAll() returns error? {
    // v3 defines no output-size field; `dimensions` is refused at construction, so
    // the converter must never invent one.
    map<json> body = check encodeCohereEmbedV3(["hi"], {inputType: SEARCH_DOCUMENT, dimensions: 256}).ensureType();
    test:assertFalse(body.hasKey("dimensions"));
    test:assertFalse(body.hasKey("output_dimension"));
}

@test:Config {}
function testCohereTruncateIsSpelledPerVersion() returns error? {
    // v3: NONE|START|END. v4 renamed them: NONE|LEFT|RIGHT. Same meaning.
    map<json> v3 = check encodeCohereEmbedV3(["hi"], {inputType: SEARCH_QUERY, truncate: TRUNCATE_START})
        .ensureType();
    test:assertEquals(v3["truncate"], <json>"START");
    map<json> v4 = check encodeCohereEmbedV4(["hi"], {inputType: SEARCH_QUERY, truncate: TRUNCATE_START})
        .ensureType();
    test:assertEquals(v4["truncate"], <json>"LEFT", "v4 spells START as LEFT");
    map<json> v4End = check encodeCohereEmbedV4(["hi"], {inputType: SEARCH_QUERY, truncate: TRUNCATE_END})
        .ensureType();
    test:assertEquals(v4End["truncate"], <json>"RIGHT", "v4 spells END as RIGHT");
    map<json> v4None = check encodeCohereEmbedV4(["hi"], {inputType: SEARCH_QUERY, truncate: TRUNCATE_NONE})
        .ensureType();
    test:assertEquals(v4None["truncate"], <json>"NONE", "NONE is unchanged across versions");
}

@test:Config {}
function testCohereV4StillEmitsInputTypeAndTextsArray() returns error? {
    map<json> body = check encodeCohereEmbedV4(["a", "b"], {inputType: SEARCH_DOCUMENT}).ensureType();
    test:assertEquals(body["input_type"], <json>"search_document");
    test:assertEquals(body["texts"], <json>["a", "b"]);
    test:assertFalse(body.hasKey("inputText"));
}

@test:Config {}
function testCohereVersionIsSelectedByModelId() {
    test:assertTrue(usesCohereEmbedV4("cohere.embed-v4:0"));
    test:assertFalse(usesCohereEmbedV4("cohere.embed-english-v3"));
    test:assertFalse(usesCohereEmbedV4("cohere.embed-multilingual-v3"));
}

// ---- dimensions is validated at construction, not at AWS ----

@test:Config {}
function testTitanRejectsAnUnsupportedDimensions() {
    TitanEmbeddingProvider|ai:Error provider = new (TITAN_EMBED_TEXT_V2, TEST_CREDS, "us-east-1", dimensions = 999);
    test:assertTrue(provider is ai:Error, "999 is not one of Titan V2's accepted widths");
}

@test:Config {}
function testTitanV1RejectsDimensionsEntirely() {
    TitanEmbeddingProvider|ai:Error provider = new (TITAN_EMBED_TEXT_V1, TEST_CREDS, "us-east-1", dimensions = 512);
    test:assertTrue(provider is ai:Error, "Titan V1 has no dimensions parameter");
}

@test:Config {}
function testTitanAcceptsItsThreeSupportedWidths() returns error? {
    foreach int d in [256, 512, 1024] {
        TitanEmbeddingProvider _ = check new (TITAN_EMBED_TEXT_V2, TEST_CREDS, "us-east-1", dimensions = d);
    }
}

@test:Config {}
function testCohereV3RejectsDimensionsBecauseTheModelHasNone() {
    CohereEmbeddingProvider|ai:Error provider = new (COHERE_EMBED_ENGLISH_V3, TEST_CREDS, "us-east-1",
            dimensions = 256);
    test:assertTrue(provider is ai:Error, "v3 has no output-size parameter — must not be silently dropped");
}

@test:Config {}
function testCohereV4RejectsAnUnsupportedDimensions() {
    CohereEmbeddingProvider|ai:Error provider = new (COHERE_EMBED_V4, TEST_CREDS, "us-east-1", dimensions = 999);
    test:assertTrue(provider is ai:Error);
}

@test:Config {}
function testCohereV4AcceptsItsSupportedWidths() returns error? {
    foreach int d in [256, 512, 1024, 1536] {
        CohereEmbeddingProvider _ = check new (COHERE_EMBED_V4, TEST_CREDS, "us-east-1", dimensions = d);
    }
}

@test:Config {}
function testCohereVersionSurvivesACrisGeoPrefix() {
    // REGRESSION: the provider passes the WIRE id (geo prefix restored) to the
    // version check. A raw startsWith("cohere.embed-v4") missed `us.`/`global.`
    // prefixed ids and silently selected the v3 converter — wrong truncate spelling and
    // no output_dimension, with no error. Cohere Embed v4 is the one embedding
    // model AWS gives Geo AND Global inference ids.
    test:assertTrue(usesCohereEmbedV4("us.cohere.embed-v4:0"));
    test:assertTrue(usesCohereEmbedV4("eu.cohere.embed-v4:0"));
    test:assertTrue(usesCohereEmbedV4("global.cohere.embed-v4:0"));
    test:assertFalse(usesCohereEmbedV4("us.cohere.embed-english-v3"));
}

@test:Config {}
function testTitanV1GuardSurvivesACrisGeoPrefix() {
    test:assertTrue(isTitanEmbedV1("amazon.titan-embed-text-v1"));
    test:assertTrue(isTitanEmbedV1("us.amazon.titan-embed-text-v1"));
    test:assertFalse(isTitanEmbedV1("amazon.titan-embed-text-v2:0"));
    test:assertFalse(isTitanEmbedV1("us.amazon.titan-embed-text-v2:0"));
}

@test:Config {}
function testCrisPrefixedCohereV4StillGetsTheV4Converter() returns error? {
    // End-to-end through construction: a geo-prefixed v4 id must accept `dimensions`
    // (v3 rejects it) and reach the wire as `output_dimension`.
    CohereEmbeddingProvider _ = check new ("us.cohere.embed-v4:0", TEST_CREDS, "us-east-1", dimensions = 256);
}
