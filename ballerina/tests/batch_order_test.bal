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

// batchEmbed's reassembly loop.
//
// COVERAGE GAP this closes: the embedding tests asserted `partitionTexts` (how many
// windows) but never the loop that stitches the windows back together — so "results
// come back in input order", the actual contract, was untested. A transposition
// there is invisible: every vector is present and well-formed, just attached to the
// wrong chunk, which surfaces later as quietly bad retrieval. It needs the transport
// intercepted, which is why it was skipped.

// Substitutes for `BedrockTransport`, recording each window and returning vectors
// that identify which text produced them.
isolated client class MockEmbedTransport {
    private final string vendor;
    private final int[] windowSizes = [];

    isolated function init(string vendor) {
        self.vendor = vendor;
    }

    // Mirrors BedrockTransport.execute's signature so runBatchEmbed can call it.
    isolated function execute(json body, map<string> extraHeaders = {}) returns TransportResponse|ai:Error {
        string[]|error parsed = textsFromBody(body, self.vendor);
        if parsed is error {
            return error ai:Error("mock transport could not read the request body", parsed);
        }
        string[] texts = parsed;
        lock {
            self.windowSizes.push(texts.length());
        }
        // Each vector's first element is its text's ordinal ("t7" -> 7.0), so a
        // misplaced result is caught by value, not just by count.
        json[] vectors = [];
        foreach string text in texts {
            float|error ordinal = ordinalOf(text);
            if ordinal is error {
                return error ai:Error("bad probe text", ordinal);
            }
            vectors.push(<json>[ordinal, 0.5]);
        }
        return {body: embedResponseBody(vectors, self.vendor), headers: {}};
    }

    isolated function requestCount() returns int {
        lock {
            return self.windowSizes.length();
        }
    }

    isolated function sizes() returns int[] {
        lock {
            return self.windowSizes.clone();
        }
    }
}

// Reads the texts back out of an encoded request body — which also pins the two
// vendors' request shapes: Titan sends a bare `inputText` STRING, Cohere a `texts`
// ARRAY.
isolated function textsFromBody(json body, string vendor) returns string[]|error {
    map<json> b = check body.ensureType();
    if vendor == "titan" {
        string inputText = check b["inputText"].ensureType();
        return [inputText];
    }
    json[] texts = check b["texts"].ensureType();
    return from json t in texts
        select check t.ensureType(string);
}

isolated function embedResponseBody(json[] vectors, string vendor) returns json {
    if vendor == "titan" {
        // Titan returns ONE embedding per call, plus a token count.
        return {"embedding": vectors[0], "inputTextTokenCount": 3};
    }
    return {"embeddings": vectors, "id": "resp-1"};
}

isolated function ordinalOf(string text) returns float|error {
    int n = check int:fromString(text.substring(1));
    return <float>n;
}

// "t0", "t1", ... "t{n-1}" — distinguishable, and their ordinal is their index.
isolated function probeTexts(int n) returns ai:Chunk[] {
    ai:Chunk[] chunks = [];
    foreach int i in 0 ..< n {
        chunks.push(<ai:TextChunk>{content: string `t${i}`});
    }
    return chunks;
}

// Asserts embedding[i] is the vector produced by text i, for every i.
isolated function assertInInputOrder(ai:Embedding[] embeddings, int expectedCount) {
    test:assertEquals(embeddings.length(), expectedCount);
    foreach int i in 0 ..< expectedCount {
        ai:Embedding e = embeddings[i];
        if e !is float[] {
            test:assertFail("embedding " + i.toString() + " was not a vector");
        }
        test:assertEquals(e[0], <float>i,
                string `embedding at index ${i} came from text t${<int>e[0]} — results are out of input order`);
    }
}

// ---- 100 chunks: Titan = 100 requests, Cohere = 2 (96 + 4), both in order ----

@test:Config {}
function testTitanBatchEmbedReturnsResultsInInputOrder() returns error? {
    MockEmbedTransport mock = new ("titan");
    ai:Embedding[] embeddings = check runBatchEmbed("Titan", "amazon.titan-embed-text-v2:0",
            TITAN_EMBED_CONVERTER, mock, {}, probeTexts(100));
    assertInInputOrder(embeddings, 100);
    test:assertEquals(mock.requestCount(), 100, "Titan embeds one text per request (maxBatchSize 1)");
}

@test:Config {}
function testCohereBatchEmbedReturnsResultsInInputOrderAcrossWindows() returns error? {
    // The interesting case: 100 texts span TWO windows, so the second window's
    // results must land at absolute indices 96..99, not back at 0.
    MockEmbedTransport mock = new ("cohere");
    ai:Embedding[] embeddings = check runBatchEmbed("Cohere", "cohere.embed-english-v3",
            COHERE_EMBED_V3_CONVERTER, mock, {inputType: SEARCH_DOCUMENT},
            probeTexts(100));
    assertInInputOrder(embeddings, 100);
    test:assertEquals(mock.requestCount(), 2, "Cohere batches 96 per request → ceil(100/96) = 2");
    test:assertEquals(mock.sizes(), [96, 4], "the tail window carries the remainder");
}

@test:Config {}
function testCohereBatchEmbedHandlesAnExactWindowBoundary() returns error? {
    // 96 must be ONE window, not 96 + an empty trailing one.
    MockEmbedTransport mock = new ("cohere");
    ai:Embedding[] embeddings = check runBatchEmbed("Cohere", "cohere.embed-english-v3",
            COHERE_EMBED_V3_CONVERTER, mock, {inputType: SEARCH_DOCUMENT},
            probeTexts(96));
    assertInInputOrder(embeddings, 96);
    test:assertEquals(mock.sizes(), [96]);
}
