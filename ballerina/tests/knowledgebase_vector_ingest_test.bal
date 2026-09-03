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

// `ingest()` on the self-managed class. The implementation mirrors the managed one,
// so these guard against the two copies drifting apart: batching at 10, the terminal
// poll, and a FAILED document producing a named error rather than a silent success.

const string VING_KB_ID = "KBVECING01";
const string VING_DS_ID = "DSVECING01";

// Sentinel: a document whose wire id is "999" is reported FAILED by the mock.
const string VING_FAIL_ID = "999";

isolated int[] vectorIngestBatchSizes = [];

isolated function recordVectorIngestBatch(int size) {
    lock {
        vectorIngestBatchSizes.push(size);
    }
}

isolated function readVectorIngestBatchSizes() returns int[] {
    lock {
        return vectorIngestBatchSizes.clone();
    }
}

isolated function resetVectorIngestBatchSizes() {
    lock {
        vectorIngestBatchSizes.removeAll();
    }
}

isolated service class VectorIngestMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${VING_KB_ID}` {
            return {
                knowledgeBase: {
                    knowledgeBaseId: VING_KB_ID,
                    name: "kb",
                    status: "ACTIVE",
                    knowledgeBaseConfiguration: {'type: "VECTOR"}
                }
            };
        }
        if p == string `/knowledgebases/${VING_KB_ID}/datasources/${VING_DS_ID}` {
            return {
                dataSource: {
                    dataSourceId: VING_DS_ID,
                    status: "AVAILABLE",
                    dataSourceConfiguration: {'type: "CUSTOM"},
                    // FIXED_SIZE => Bedrock chunks server-side => chunker defaults to
                    // ai:DISABLE, so documents pass through one-for-one.
                    vectorIngestionConfiguration: {chunkingConfiguration: {chunkingStrategy: "FIXED_SIZE"}}
                }
            };
        }
        return error(string `unexpected GET ${p}`);
    }

    isolated resource function put [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${VING_KB_ID}/datasources/${VING_DS_ID}/documents` {
            json body = check req.getJsonPayload();
            json[] documents = <json[]>(<map<json>>body)["documents"];
            recordVectorIngestBatch(documents.length());
            // Non-terminal on purpose: what ingest() waits on is the getDocuments
            // poll below, not this 202.
            return {documentDetails: documents.map(_doc => <json>{status: "STARTING"})};
        }
        return error(string `unexpected PUT ${p}`);
    }

    isolated resource function post [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${VING_KB_ID}/datasources/` {
            return {dataSourceSummaries: [{dataSourceId: VING_DS_ID, name: "ds"}]};
        }
        if p == string `/knowledgebases/${VING_KB_ID}/datasources/${VING_DS_ID}/documents/getDocuments` {
            json body = check req.getJsonPayload();
            json[] identifiers = <json[]>(<map<json>>body)["documentIdentifiers"];
            json[] details = [];
            foreach json ident in identifiers {
                string id = <string>(<map<json>>(<map<json>>ident)["custom"])["id"];
                boolean failed = id == VING_FAIL_ID;
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
        return error(string `unexpected POST ${p}`);
    }
}

// `IngestKnowledgeBaseDocuments` accepts at most 10 documents per request
// (API_agent_IngestKnowledgeBaseDocuments.html: "Maximum number of 10 items").
@test:Config {}
function testVectorIngestBatchesInGroupsOfTen() returns error? {
    final int port = 18711;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorIngestMock(), "/");
    check mockListener.'start();
    resetVectorIngestBatchSizes();

    BedrockVectorKnowledgeBase kb = check new (VING_KB_ID, KB_TEST_CREDS, "us-east-1",
        string `http://localhost:${port}`);

    ai:TextDocument[] docs = [];
    foreach int i in 0 ..< 23 {
        docs.push({content: string `document ${i}`, metadata: {id: i}});
    }
    ai:Error? result = kb.ingest(docs);
    check mockListener.gracefulStop();

    if result is ai:Error {
        test:assertFail(result.message());
    }
    test:assertEquals(readVectorIngestBatchSizes(), [10, 10, 3]);
}

// ingest() must not report success for a document that later lands FAILED — the
// `ai:KnowledgeBase.ingest` contract returns a bare `Error?` with no job handle, so
// a caller has no other way to discover it.
@test:Config {}
function testVectorIngestReportsFailedDocuments() returns error? {
    final int port = 18712;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorIngestMock(), "/");
    check mockListener.'start();
    resetVectorIngestBatchSizes();

    BedrockVectorKnowledgeBase kb = check new (VING_KB_ID, KB_TEST_CREDS, "us-east-1",
        string `http://localhost:${port}`);
    ai:TextDocument[] docs = [
        {content: "fine", metadata: {id: 1}},
        {content: "doomed", metadata: {id: 999}}
    ];
    ai:Error? result = kb.ingest(docs);
    check mockListener.gracefulStop();

    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        string msg = result.message();
        test:assertTrue(msg.includes(VING_FAIL_ID), msg);
        test:assertTrue(msg.includes("FAILED"), msg);
        test:assertTrue(msg.includes("simulated failure for test"), msg);
    }
}

// Only text is supported. A non-text chunk must produce a named error, never be
// silently dropped.
@test:Config {}
function testVectorIngestRefusesNonTextContent() returns error? {
    final int port = 18713;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorIngestMock(), "/");
    check mockListener.'start();
    resetVectorIngestBatchSizes();

    BedrockVectorKnowledgeBase kb = check new (VING_KB_ID, KB_TEST_CREDS, "us-east-1",
        string `http://localhost:${port}`);
    ai:ImageDocument image = {content: "https://example.com/cat.png"};
    ai:Error? result = kb.ingest(image);
    check mockListener.gracefulStop();

    test:assertTrue(result is ai:Error);
    // Nothing may have been submitted.
    test:assertEquals(readVectorIngestBatchSizes(), []);
}
