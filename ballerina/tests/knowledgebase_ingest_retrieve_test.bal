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

// End-to-end `ingest()`/`retrieve()` tests against a stubbed bedrock-agent /
// bedrock-agent-runtime pair on a local listener.

const string ING_KB_ID = "KBINGRET01";
const string ING_DS_ID = "DSINGRET01";

// Sentinel: a submitted document whose 'ai:Metadata.id' is 999 (wire id "999") is
// reported FAILED by the mock's getDocuments handler; every other id is INDEXED.
const string ING_FAIL_ID = "999";

isolated int[] ingestBatchSizes = [];
isolated int retrievePageCalls = 0;

isolated function recordIngestBatch(int size) {
    lock {
        ingestBatchSizes.push(size);
    }
}

isolated function readIngestBatchSizes() returns int[] {
    lock {
        return ingestBatchSizes.clone();
    }
}

isolated function recordRetrievePage() {
    lock {
        retrievePageCalls += 1;
    }
}

isolated function readRetrievePageCalls() returns int {
    lock {
        return retrievePageCalls;
    }
}

isolated function ingRetResult(string id, float score) returns json => {
    content: {text: string `text for ${id}`, 'type: "TEXT"},
    documentId: id,
    location: {'type: "CUSTOM", customDocumentLocation: {id}},
    metadata: {"_source_uri": id},
    score
};

isolated service class IngestRetrieveMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${ING_KB_ID}` {
            return {knowledgeBase: {knowledgeBaseId: ING_KB_ID, name: "kb", status: "ACTIVE"}};
        }
        if p == string `/knowledgebases/${ING_KB_ID}/datasources/${ING_DS_ID}` {
            return {
                dataSource: {
                    dataSourceConfiguration: {'type: "CUSTOM"},
                    vectorIngestionConfiguration: {chunkingConfiguration: {chunkingStrategy: "FIXED_SIZE"}}
                }
            };
        }
        return error(string `unexpected GET ${p}`);
    }

    isolated resource function put [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${ING_KB_ID}/datasources/${ING_DS_ID}/documents` {
            json body = check req.getJsonPayload();
            json[] documents = <json[]>(<map<json>>body)["documents"];
            recordIngestBatch(documents.length());
            // Non-terminal on purpose: the real signal ingest() waits on comes from
            // the follow-up getDocuments poll below, not this response.
            json[] details = documents.map(_doc => <json>{status: "STARTING"});
            return {documentDetails: details};
        }
        return error(string `unexpected PUT ${p}`);
    }

    isolated resource function post [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        json body = check req.getJsonPayload();

        if p == string `/knowledgebases/${ING_KB_ID}/datasources/${ING_DS_ID}/documents/getDocuments` {
            json[] identifiers = <json[]>(<map<json>>body)["documentIdentifiers"];
            json[] details = [];
            foreach json ident in identifiers {
                string id = <string>(<map<json>>(<map<json>>ident)["custom"])["id"];
                boolean failed = id == ING_FAIL_ID;
                map<json> detail = {
                    identifier: {dataSourceType: "CUSTOM", custom: {id}},
                    status: failed ? "FAILED" : "INDEXED"
                };
                if failed {
                    detail["statusReason"] = "simulated failure for test";
                }
                details.push(detail);
            }
            return {documentDetails: details};
        }

        if p == string `/knowledgebases/${ING_KB_ID}/retrieve` {
            recordRetrievePage();
            map<json> bodyMap = <map<json>>body;
            json? nextToken = bodyMap["nextToken"] ?: ();
            if nextToken is () {
                return {
                    retrievalResults: [ingRetResult("r1", 0.9), ingRetResult("r2", 0.8)],
                    nextToken: "page2"
                };
            }
            return {retrievalResults: [ingRetResult("r3", 0.7)]};
        }
        return error(string `unexpected POST ${p}`);
    }
}

@test:Config {}
function testIngestBatchesInGroupsOfTenAndSucceedsWhenAllIndex() returns error? {
    final int port = 18671;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new IngestRetrieveMock(), "/");
    check mockListener.'start();

    BedrockManagedKnowledgeBase kb = check new (
        ING_KB_ID, KB_TEST_CREDS, "us-east-1",
        serviceUrl = string `http://localhost:${port}`,
        dataSourceId = ING_DS_ID);

    ai:TextChunk[] chunks = [];
    foreach int i in 0 ..< 12 {
        chunks.push({content: string `chunk ${i}`});
    }
    ai:Error? result = kb.ingest(chunks);
    check mockListener.gracefulStop();

    test:assertTrue(result is (), result is ai:Error ? result.message() : "");
    // 12 documents, batch size 10 -> two PUT calls of 10 and 2.
    int[] batches = readIngestBatchSizes();
    test:assertEquals(batches.length(), 2);
    test:assertEquals(batches[0], 10);
    test:assertEquals(batches[1], 2);
}

@test:Config {}
function testIngestReportsAFailedDocumentByIdAndReason() returns error? {
    final int port = 18672;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new IngestRetrieveMock(), "/");
    check mockListener.'start();

    BedrockManagedKnowledgeBase kb = check new (
        ING_KB_ID, KB_TEST_CREDS, "us-east-1",
        serviceUrl = string `http://localhost:${port}`,
        dataSourceId = ING_DS_ID);

    ai:TextChunk okChunk = {content: "fine"};
    ai:TextChunk failChunk = {content: "will fail", metadata: {id: 999}};
    ai:Error? result = kb.ingest([okChunk, failChunk]);
    check mockListener.gracefulStop();

    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        string msg = result.message();
        test:assertTrue(msg.includes(ING_FAIL_ID), msg);
        test:assertTrue(msg.includes("FAILED"), msg);
        test:assertTrue(msg.includes("simulated failure for test"), msg);
    }
}

@test:Config {}
function testRetrievePaginatesAndMapsResultsInOrder() returns error? {
    final int port = 18673;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new IngestRetrieveMock(), "/");
    check mockListener.'start();

    BedrockManagedKnowledgeBase kb = check new (
        ING_KB_ID, KB_TEST_CREDS, "us-east-1",
        serviceUrl = string `http://localhost:${port}`,
        dataSourceId = ING_DS_ID,
        numberOfResults = 2);

    ai:QueryMatch[] all = check kb.retrieve("find things", 10);
    test:assertEquals(all.length(), 3, all.toJsonString());
    test:assertEquals(readRetrievePageCalls(), 2, "expected two pages to be fetched via nextToken");
    ai:Chunk firstChunk = all[0].chunk;
    test:assertTrue(firstChunk is ai:TextChunk);
    if firstChunk is ai:TextChunk {
        test:assertEquals(firstChunk.content, "text for r1");
    }
    test:assertEquals(all[0].similarityScore, 0.9);
    test:assertEquals(all[2].similarityScore, 0.7);

    // maxLimit truncates even mid-page, without erroring on the extra availability.
    ai:QueryMatch[] limited = check kb.retrieve("find things", 1);
    test:assertEquals(limited.length(), 1);

    check mockListener.gracefulStop();
}
