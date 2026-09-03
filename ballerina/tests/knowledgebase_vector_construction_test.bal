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

// `BedrockVectorKnowledgeBase` construction against a stubbed bedrock-agent on a
// local listener: the knowledge base type guard, the find-or-create path, and the
// checks that fire before any I/O at all.

const string VEC_KB_ID = "KBVECTOR01";
const string VEC_DS_ID = "DSVECTOR01";

final OpenSearchServerlessStorage VEC_TEST_STORAGE = {
    collectionArn: "arn:aws:aoss:us-east-1:123456789012:collection/abcdefghij1234567890",
    vectorIndexName: "bedrock-index",
    fieldMapping: {
        vectorField: "embeddings",
        textField: "AMAZON_BEDROCK_TEXT_CHUNK",
        metadataField: "AMAZON_BEDROCK_METADATA"
    }
};

const string VEC_EMBEDDING_ARN = "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0";
const string VEC_ROLE_ARN = "arn:aws:iam::123456789012:role/service-role/AmazonBedrockExecutionRoleForKB_1";

isolated function vectorKbResponse(string kbType) returns json => {
    knowledgeBase: {
        knowledgeBaseId: VEC_KB_ID,
        name: "vec-kb",
        status: "ACTIVE",
        knowledgeBaseConfiguration: {'type: kbType}
    }
};

// ---- knowledge base type guard: attaching to a MANAGED knowledge base ----

isolated service class ManagedTypeKbMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json {
        return vectorKbResponse("MANAGED");
    }
}

@test:Config {}
function testVectorClassRefusesAManagedKnowledgeBase() returns error? {
    final int port = 18681;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new ManagedTypeKbMock(), "/");
    check mockListener.'start();

    BedrockVectorKnowledgeBase|ai:Error kb = new (VEC_KB_ID, KB_TEST_CREDS, "us-east-1",
        serviceUrl = string `http://localhost:${port}`);
    check mockListener.gracefulStop();

    test:assertTrue(kb is ai:Error);
    if kb is ai:Error {
        string msg = kb.message();
        test:assertTrue(msg.includes("MANAGED"), msg);
        test:assertTrue(msg.includes("BedrockVectorKnowledgeBase"), msg);
        // The message must point at the class that DOES support it.
        test:assertTrue(msg.includes("BedrockManagedKnowledgeBase"), msg);
    }
}

// ---- happy path: attaching to a VECTOR knowledge base with a CUSTOM data source ----

isolated service class VectorAttachMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${VEC_KB_ID}` {
            return vectorKbResponse("VECTOR");
        }
        if p == string `/knowledgebases/${VEC_KB_ID}/datasources/${VEC_DS_ID}` {
            // A self-managed data source uses the PLAIN CUSTOM type and reports its
            // chunking configuration back.
            return {
                dataSource: {
                    dataSourceId: VEC_DS_ID,
                    status: "AVAILABLE",
                    dataSourceConfiguration: {'type: "CUSTOM"},
                    vectorIngestionConfiguration: {chunkingConfiguration: {chunkingStrategy: "NONE"}}
                }
            };
        }
        return error(string `unexpected GET ${p}`);
    }

    isolated resource function post [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${VEC_KB_ID}/datasources/` {
            return {dataSourceSummaries: [{dataSourceId: VEC_DS_ID, name: "ds", status: "AVAILABLE"}]};
        }
        return error(string `unexpected POST ${p}`);
    }
}

@test:Config {}
function testVectorClassAttachesToAVectorKnowledgeBase() returns error? {
    final int port = 18682;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorAttachMock(), "/");
    check mockListener.'start();

    BedrockVectorKnowledgeBase|ai:Error kb = new (VEC_KB_ID, KB_TEST_CREDS, "us-east-1",
        serviceUrl = string `http://localhost:${port}`);
    check mockListener.gracefulStop();

    if kb is ai:Error {
        test:assertFail(kb.message());
    }
}

// A data source reporting `chunkingStrategy: NONE` means the caller owns chunk
// boundaries, so the detected default is `ai:AUTO` and an explicit chunker is
// allowed. On a self-managed knowledge base this detection actually works, unlike
// the managed path where `GetDataSource` may not echo `chunkingConfiguration` at all.
@test:Config {}
function testExplicitChunkerAllowedWhenDataSourceChunkingIsNone() returns error? {
    final int port = 18683;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorAttachMock(), "/");
    check mockListener.'start();

    BedrockVectorKnowledgeBase|ai:Error kb = new (VEC_KB_ID, KB_TEST_CREDS, "us-east-1",
        serviceUrl = string `http://localhost:${port}`, chunker = new ai:MarkdownChunker());
    check mockListener.gracefulStop();

    if kb is ai:Error {
        test:assertFail(kb.message());
    }
}

// ---- find-or-create: no name match creates both resources ----

// One variable, not three: Ballerina forbids a `lock` from touching more than one
// module-level isolated variable, so the captured bodies are keyed by request path.
isolated map<json> vectorCreateBodies = {};

isolated function recordVectorCreate(string path, json body) {
    lock {
        vectorCreateBodies[path] = body.clone();
    }
}

isolated function readVectorCreateBody(string path) returns json {
    lock {
        return (vectorCreateBodies[path] ?: ()).clone();
    }
}

isolated service class VectorCreateMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${VEC_KB_ID}` {
            return vectorKbResponse("VECTOR");
        }
        if p == string `/knowledgebases/${VEC_KB_ID}/datasources/${VEC_DS_ID}` {
            return {
                dataSource: {
                    dataSourceId: VEC_DS_ID,
                    status: "AVAILABLE",
                    dataSourceConfiguration: {'type: "CUSTOM"},
                    vectorIngestionConfiguration: {chunkingConfiguration: {chunkingStrategy: "FIXED_SIZE"}}
                }
            };
        }
        return error(string `unexpected GET ${p}`);
    }

    isolated resource function put [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        json body = check req.getJsonPayload();
        recordVectorCreate(p, body);
        if p == "/knowledgebases/" {
            return vectorKbResponse("VECTOR");
        }
        if p == string `/knowledgebases/${VEC_KB_ID}/datasources/` {
            return {dataSource: {dataSourceId: VEC_DS_ID, status: "AVAILABLE"}};
        }
        return error(string `unexpected PUT ${p}`);
    }

    isolated resource function post [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        // ListKnowledgeBases: no name match, so the definition path creates.
        if p == "/knowledgebases/" {
            return {knowledgeBaseSummaries: []};
        }
        return error(string `unexpected POST ${p}`);
    }
}

@test:Config {}
function testFindOrCreateSendsVectorBodiesOnTheWire() returns error? {
    final int port = 18684;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorCreateMock(), "/");
    check mockListener.'start();

    VectorKnowledgeBaseDefinition def = {
        name: "vec-kb",
        roleArn: VEC_ROLE_ARN,
        embeddingModelArn: VEC_EMBEDDING_ARN,
        storageConfiguration: VEC_TEST_STORAGE
    };
    BedrockVectorKnowledgeBase|ai:Error kb = new (def, KB_TEST_CREDS, "us-east-1",
        serviceUrl = string `http://localhost:${port}`);
    check mockListener.gracefulStop();

    if kb is ai:Error {
        test:assertFail(kb.message());
    }

    // CreateKnowledgeBase carried the VECTOR shape.
    map<json> kbBody = <map<json>>readVectorCreateBody("/knowledgebases/");
    map<json> kbConfig = <map<json>>kbBody["knowledgeBaseConfiguration"];
    test:assertEquals(kbConfig["type"], "VECTOR");
    test:assertEquals(
        (<map<json>>kbConfig["vectorKnowledgeBaseConfiguration"])["embeddingModelArn"], VEC_EMBEDDING_ARN);
    test:assertTrue(kbBody.hasKey("storageConfiguration"));
    test:assertEquals((<map<json>>kbBody["storageConfiguration"])["type"], "OPENSEARCH_SERVERLESS");

    // CreateDataSource carried the PLAIN CUSTOM shape — no connector wrapper.
    map<json> dsBody = <map<json>>readVectorCreateBody(string `/knowledgebases/${VEC_KB_ID}/datasources/`);
    test:assertEquals(<map<json>>dsBody["dataSourceConfiguration"], {'type: "CUSTOM"});
    test:assertFalse(dsBody.toJsonString().includes("MANAGED_KNOWLEDGE_BASE_CONNECTOR"), dsBody.toJsonString());
    test:assertTrue(dsBody.hasKey("vectorIngestionConfiguration"));
}

// ---- checks that fire before any I/O ----

// An unreachable port: reaching the network at all would surface as a connection
// error, so a clean validation message proves the check ran first.
@test:Config {}
function testInvalidS3VectorsStorageFailsBeforeAnyRequest() returns error? {
    VectorKnowledgeBaseDefinition def = {
        name: "vec-kb",
        roleArn: VEC_ROLE_ARN,
        embeddingModelArn: VEC_EMBEDDING_ARN,
        // Neither `indexArn` nor `vectorBucketArn` + `indexName`.
        storageConfiguration: <S3VectorsStorage>{
            vectorBucketArn: "arn:aws:s3vectors:us-east-1:123456789012:bucket/b"
        }
    };
    BedrockVectorKnowledgeBase|ai:Error kb = new (def, KB_TEST_CREDS, "us-east-1",
        serviceUrl = "http://localhost:1");

    test:assertTrue(kb is ai:Error);
    if kb is ai:Error {
        string msg = kb.message();
        test:assertTrue(msg.includes("indexArn"), msg);
        test:assertFalse(msg.includes("Connection refused"), msg);
    }
}

// The range/strategy validators are unit-tested directly elsewhere; these prove they
// are actually WIRED into construction. Without them, deleting the `check
// validate...` calls from `resolveVectorKbSpine` would leave those unit tests green.
// An unreachable port makes the point: a clean validation message can only mean the
// check ran before any request was attempted.
@test:Config {}
function testDataSourceValidationIsWiredIntoConstruction() returns error? {
    VectorKnowledgeBaseDefinition def = {
        name: "vec-kb",
        roleArn: VEC_ROLE_ARN,
        embeddingModelArn: VEC_EMBEDDING_ARN,
        storageConfiguration: VEC_TEST_STORAGE,
        dataSource: {name: "ds", maxTokens: 0}
    };
    BedrockVectorKnowledgeBase|ai:Error kb = new (def, KB_TEST_CREDS, "us-east-1",
        serviceUrl = "http://localhost:1");

    test:assertTrue(kb is ai:Error);
    if kb is ai:Error {
        test:assertTrue(kb.message().includes("maxTokens"), kb.message());
        test:assertFalse(kb.message().includes("Connection refused"), kb.message());
    }
}

@test:Config {}
function testUnsupportedChunkingStrategyIsWiredIntoConstruction() returns error? {
    VectorKnowledgeBaseDefinition def = {
        name: "vec-kb",
        roleArn: VEC_ROLE_ARN,
        embeddingModelArn: VEC_EMBEDDING_ARN,
        storageConfiguration: VEC_TEST_STORAGE,
        dataSource: {name: "ds", chunkingStrategy: HIERARCHICAL}
    };
    BedrockVectorKnowledgeBase|ai:Error kb = new (def, KB_TEST_CREDS, "us-east-1",
        serviceUrl = "http://localhost:1");

    test:assertTrue(kb is ai:Error);
    if kb is ai:Error {
        test:assertTrue(kb.message().includes("HIERARCHICAL"), kb.message());
    }
}

// Retrieval-side ranges are validated at construction too, not lazily on first use —
// and this path runs even when attaching by id, where there is no data source
// definition to validate.
@test:Config {}
function testRetrievalConfigValidationIsWiredIntoConstruction() returns error? {
    BedrockVectorKnowledgeBase|ai:Error kb = new (VEC_KB_ID, KB_TEST_CREDS, "us-east-1",
        serviceUrl = "http://localhost:1", numberOfResults = 0);

    test:assertTrue(kb is ai:Error);
    if kb is ai:Error {
        test:assertTrue(kb.message().includes("numberOfResults"), kb.message());
        test:assertFalse(kb.message().includes("Connection refused"), kb.message());
    }
}

// ---- ambiguous name ----

isolated service class VectorAmbiguousNameMock {
    *http:Service;

    isolated resource function post [string... path](http:Request req) returns json {
        return {
            knowledgeBaseSummaries: [
                {knowledgeBaseId: "KBVECAAAAA", name: "dup-vec-kb", status: "ACTIVE"},
                {knowledgeBaseId: "KBVECBBBBB", name: "dup-vec-kb", status: "ACTIVE"}
            ]
        };
    }
}

@test:Config {}
function testAmbiguousVectorKnowledgeBaseNameFailsAtConstruction() returns error? {
    final int port = 18685;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorAmbiguousNameMock(), "/");
    check mockListener.'start();

    VectorKnowledgeBaseDefinition def = {
        name: "dup-vec-kb",
        roleArn: VEC_ROLE_ARN,
        embeddingModelArn: VEC_EMBEDDING_ARN,
        storageConfiguration: VEC_TEST_STORAGE
    };
    BedrockVectorKnowledgeBase|ai:Error kb = new (def, KB_TEST_CREDS, "us-east-1",
        serviceUrl = string `http://localhost:${port}`);
    check mockListener.gracefulStop();

    test:assertTrue(kb is ai:Error);
    if kb is ai:Error {
        string msg = kb.message();
        test:assertTrue(msg.includes("KBVECAAAAA") && msg.includes("KBVECBBBBB"), msg);
    }
}

// ---- no CUSTOM data source to write into ----

isolated service class VectorNoCustomDataSourceMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${VEC_KB_ID}` {
            return vectorKbResponse("VECTOR");
        }
        if p.startsWith(string `/knowledgebases/${VEC_KB_ID}/datasources/`) {
            return {dataSource: {dataSourceId: "DSS3AAAAAA", dataSourceConfiguration: {'type: "S3"}}};
        }
        return error(string `unexpected GET ${p}`);
    }

    isolated resource function post [string... path](http:Request req) returns json|error {
        if req.rawPath == string `/knowledgebases/${VEC_KB_ID}/datasources/` {
            return {dataSourceSummaries: [{dataSourceId: "DSS3AAAAAA", name: "s3-ds"}]};
        }
        return error(string `unexpected POST ${req.rawPath}`);
    }
}

@test:Config {}
function testVectorKnowledgeBaseWithNoCustomDataSourceFailsAtConstruction() returns error? {
    final int port = 18686;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorNoCustomDataSourceMock(), "/");
    check mockListener.'start();

    BedrockVectorKnowledgeBase|ai:Error kb = new (VEC_KB_ID, KB_TEST_CREDS, "us-east-1",
        serviceUrl = string `http://localhost:${port}`);
    check mockListener.gracefulStop();

    test:assertTrue(kb is ai:Error);
    if kb is ai:Error {
        test:assertTrue(kb.message().includes("CUSTOM"), kb.message());
    }
}

// ---- a non-ACTIVE knowledge base ----

isolated service class VectorCreatingKbMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json {
        return {
            knowledgeBase: {
                knowledgeBaseId: VEC_KB_ID,
                name: "vec-kb",
                status: "CREATING",
                knowledgeBaseConfiguration: {'type: "VECTOR"}
            }
        };
    }
}

@test:Config {}
function testNonActiveVectorKnowledgeBaseFailsAtConstruction() returns error? {
    final int port = 18687;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorCreatingKbMock(), "/");
    check mockListener.'start();

    BedrockVectorKnowledgeBase|ai:Error kb = new (VEC_KB_ID, KB_TEST_CREDS, "us-east-1",
        serviceUrl = string `http://localhost:${port}`);
    check mockListener.gracefulStop();

    test:assertTrue(kb is ai:Error);
    if kb is ai:Error {
        test:assertTrue(kb.message().includes("CREATING"), kb.message());
    }
}
