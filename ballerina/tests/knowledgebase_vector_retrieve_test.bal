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
import ballerina/test;

// `retrieve()` against a stubbed bedrock-agent-runtime, asserting the request body
// actually put on the wire — the VECTOR search branch, paging, and the cap logic.

const string VRET_KB_ID = "KBVECRET01";
const string VRET_DS_ID = "DSVECRET01";

isolated json[] vectorRetrieveBodies = [];

isolated function recordVectorRetrieveBody(json body) {
    lock {
        vectorRetrieveBodies.push(body.clone());
    }
}

isolated function readVectorRetrieveBodies() returns json[] {
    lock {
        return vectorRetrieveBodies.clone();
    }
}

isolated function resetVectorRetrieveBodies() {
    lock {
        vectorRetrieveBodies.removeAll();
    }
}

isolated function vecRetResult(string id, float score) returns json => {
    content: {text: string `text for ${id}`, 'type: "TEXT"},
    documentId: id,
    location: {'type: "CUSTOM", customDocumentLocation: {id}},
    // Self-managed knowledge bases use the `x-amz-bedrock` reserved prefix, not the
    // managed `_` one.
    metadata: {"x-amz-bedrock-kb-source-uri": id},
    score
};

isolated service class VectorRetrieveMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${VRET_KB_ID}` {
            return {
                knowledgeBase: {
                    knowledgeBaseId: VRET_KB_ID,
                    name: "kb",
                    status: "ACTIVE",
                    knowledgeBaseConfiguration: {'type: "VECTOR"}
                }
            };
        }
        if p == string `/knowledgebases/${VRET_KB_ID}/datasources/${VRET_DS_ID}` {
            return {
                dataSource: {
                    dataSourceId: VRET_DS_ID,
                    status: "AVAILABLE",
                    dataSourceConfiguration: {'type: "CUSTOM"},
                    vectorIngestionConfiguration: {chunkingConfiguration: {chunkingStrategy: "FIXED_SIZE"}}
                }
            };
        }
        return error(string `unexpected GET ${p}`);
    }

    isolated resource function post [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${VRET_KB_ID}/datasources/` {
            return {dataSourceSummaries: [{dataSourceId: VRET_DS_ID, name: "ds"}]};
        }
        if p == string `/knowledgebases/${VRET_KB_ID}/retrieve` {
            json body = check req.getJsonPayload();
            recordVectorRetrieveBody(body);
            map<json> bodyMap = <map<json>>body;
            json? nextToken = bodyMap["nextToken"] ?: ();
            if nextToken is () {
                return {
                    retrievalResults: [vecRetResult("v1", 0.9), vecRetResult("v2", 0.8)],
                    nextToken: "page2"
                };
            }
            return {retrievalResults: [vecRetResult("v3", 0.7)]};
        }
        return error(string `unexpected POST ${p}`);
    }
}

// Not `isolated`: it builds a mutable `VectorKnowledgeBaseConfig` before handing it
// to `init`, which an isolated function may not do.
function newVectorRetrieveKb(int port, SearchType? overrideSearchType = (),
        VectorRerankingConfig? reranking = (), int? numberOfResults = ())
        returns BedrockVectorKnowledgeBase|ai:Error {
    VectorKnowledgeBaseConfig config = {};
    if overrideSearchType is SearchType {
        config.overrideSearchType = overrideSearchType;
    }
    if reranking is VectorRerankingConfig {
        config.rerankingConfiguration = reranking;
    }
    if numberOfResults is int {
        config.numberOfResults = numberOfResults;
    }
    return new (VRET_KB_ID, KB_TEST_CREDS, "us-east-1", string `http://localhost:${port}`, config);
}

// THE branch guard: a self-managed knowledge base must be queried through
// `vectorSearchConfiguration`. `managedSearchConfiguration` is the managed class's
// branch and carries different members.
// https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_KnowledgeBaseRetrievalConfiguration.html
@test:Config {}
function testRetrieveSendsVectorSearchConfigurationOnly() returns error? {
    final int port = 18691;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorRetrieveMock(), "/");
    check mockListener.'start();
    resetVectorRetrieveBodies();

    BedrockVectorKnowledgeBase kb = check newVectorRetrieveKb(port);
    ai:QueryMatch[] matches = check kb.retrieve("hello", 2);
    check mockListener.gracefulStop();

    test:assertEquals(matches.length(), 2);

    json[] bodies = readVectorRetrieveBodies();
    test:assertTrue(bodies.length() > 0);
    map<json> retrievalConfig = <map<json>>(<map<json>>bodies[0])["retrievalConfiguration"];
    test:assertTrue(retrievalConfig.hasKey("vectorSearchConfiguration"));
    test:assertFalse(retrievalConfig.hasKey("managedSearchConfiguration"),
        "a self-managed knowledge base must not be sent the managed search branch");
    test:assertFalse(bodies[0].toJsonString().includes("managedSearchConfiguration"));
}

// `overrideSearchType` 400s or silently degrades on most backends, so it must be
// absent unless the caller explicitly asked for it.
@test:Config {}
function testRetrieveOmitsOverrideSearchTypeByDefault() returns error? {
    final int port = 18692;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorRetrieveMock(), "/");
    check mockListener.'start();
    resetVectorRetrieveBodies();

    BedrockVectorKnowledgeBase kb = check newVectorRetrieveKb(port);
    ai:QueryMatch[] _ = check kb.retrieve("hello", 1);
    check mockListener.gracefulStop();

    map<json> vectorSearch = <map<json>>(<map<json>>(<map<json>>readVectorRetrieveBodies()[0])
        ["retrievalConfiguration"])["vectorSearchConfiguration"];
    test:assertFalse(vectorSearch.hasKey("overrideSearchType"));
    test:assertFalse(vectorSearch.hasKey("rerankingConfiguration"));
    // No filter was passed, so none must be emitted — not even a null one.
    test:assertFalse(vectorSearch.hasKey("filter"));
}

@test:Config {}
function testRetrieveEmitsOverrideSearchTypeAndRerankingWhenConfigured() returns error? {
    final int port = 18693;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorRetrieveMock(), "/");
    check mockListener.'start();
    resetVectorRetrieveBodies();

    BedrockVectorKnowledgeBase kb = check newVectorRetrieveKb(port, SEARCH_HYBRID,
        {modelArn: "arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0",
            numberOfRerankedResults: 3});
    ai:QueryMatch[] _ = check kb.retrieve("hello", 1);
    check mockListener.gracefulStop();

    map<json> vectorSearch = <map<json>>(<map<json>>(<map<json>>readVectorRetrieveBodies()[0])
        ["retrievalConfiguration"])["vectorSearchConfiguration"];
    test:assertEquals(vectorSearch["overrideSearchType"], "HYBRID");
    json expectedReranking = {
        'type: "BEDROCK_RERANKING_MODEL",
        bedrockRerankingConfiguration: {
            modelConfiguration: {modelArn: "arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0"},
            numberOfRerankedResults: 3
        }
    };
    test:assertEquals(vectorSearch["rerankingConfiguration"], expectedReranking);
}

@test:Config {}
function testRetrieveForwardsMetadataFilters() returns error? {
    final int port = 18694;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorRetrieveMock(), "/");
    check mockListener.'start();
    resetVectorRetrieveBodies();

    BedrockVectorKnowledgeBase kb = check newVectorRetrieveKb(port);
    ai:MetadataFilters filters = {filters: [{key: "tenant", operator: ai:EQUAL, value: "acme"}]};
    ai:QueryMatch[] _ = check kb.retrieve("hello", 1, filters);
    check mockListener.gracefulStop();

    map<json> vectorSearch = <map<json>>(<map<json>>(<map<json>>readVectorRetrieveBodies()[0])
        ["retrievalConfiguration"])["vectorSearchConfiguration"];
    test:assertEquals(vectorSearch["filter"], {'equals: {key: "tenant", value: "acme"}});
}

@test:Config {}
function testRetrievePagesThroughNextTokenAndPreservesOrder() returns error? {
    final int port = 18695;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorRetrieveMock(), "/");
    check mockListener.'start();
    resetVectorRetrieveBodies();

    BedrockVectorKnowledgeBase kb = check newVectorRetrieveKb(port);
    // maxLimit -1 -> no client-side limit, so both pages are consumed.
    ai:QueryMatch[] matches = check kb.retrieve("hello", -1);
    check mockListener.gracefulStop();

    test:assertEquals(matches.length(), 3);
    test:assertEquals(matches[0].chunk.content, "text for v1");
    test:assertEquals(matches[2].chunk.content, "text for v3");

    json[] bodies = readVectorRetrieveBodies();
    test:assertEquals(bodies.length(), 2, "expected exactly one follow-up page request");
    test:assertFalse((<map<json>>bodies[0]).hasKey("nextToken"));
    test:assertEquals((<map<json>>bodies[1])["nextToken"], "page2");
}

@test:Config {}
function testRetrieveCapsPerCallByMaxLimitAndNumberOfResults() returns error? {
    final int port = 18696;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorRetrieveMock(), "/");
    check mockListener.'start();
    resetVectorRetrieveBodies();

    // Configured cap 25; maxLimit 2 is smaller, so `numberOfResults` must be 2.
    BedrockVectorKnowledgeBase kb = check newVectorRetrieveKb(port, numberOfResults = 25);
    ai:QueryMatch[] matches = check kb.retrieve("hello", 2);
    check mockListener.gracefulStop();

    test:assertEquals(matches.length(), 2);
    map<json> vectorSearch = <map<json>>(<map<json>>(<map<json>>readVectorRetrieveBodies()[0])
        ["retrievalConfiguration"])["vectorSearchConfiguration"];
    test:assertEquals(vectorSearch["numberOfResults"], 2);
}

@test:Config {}
function testRetrieveRejectsNonPositiveMaxLimit() returns error? {
    final int port = 18697;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorRetrieveMock(), "/");
    check mockListener.'start();

    BedrockVectorKnowledgeBase kb = check newVectorRetrieveKb(port);
    ai:QueryMatch[]|ai:Error result = kb.retrieve("hello", 0);
    check mockListener.gracefulStop();

    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        test:assertTrue(result.message().includes("maxLimit"), result.message());
    }
}
