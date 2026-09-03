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

// `BedrockManagedKnowledgeBase` construction-error scenarios, each against a small
// stubbed bedrock-agent on its own local listener: an ambiguous knowledge base
// name, a knowledge base with no CUSTOM data source, and one with several.

// Separate from the model providers' `TEST_CREDS`, which is a `BedrockCredentials`
// and so admits a `BearerToken`. The knowledge base classes take
// `KnowledgeBaseCredentials` (SigV4 only) because Bedrock API keys do not work on
// the agent planes — passing `TEST_CREDS` here must NOT compile.
final KnowledgeBaseCredentials KB_TEST_CREDS = {accessKeyId: "AKIATEST", secretAccessKey: "secret"};

// ---- ambiguous name: two knowledge bases share it ----

isolated service class AmbiguousNameMock {
    *http:Service;

    isolated resource function post [string... path](http:Request req) returns json {
        return {
            knowledgeBaseSummaries: [
                {knowledgeBaseId: "KBAAAAAAAA", name: "dup-kb", status: "ACTIVE", updatedAt: "2026-08-13T00:00:00Z"},
                {knowledgeBaseId: "KBBBBBBBBB", name: "dup-kb", status: "ACTIVE", updatedAt: "2026-08-13T00:00:00Z"}
            ]
        };
    }
}

@test:Config {}
function testAmbiguousKnowledgeBaseNameFailsAtConstruction() returns error? {
    final int port = 18661;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new AmbiguousNameMock(), "/");
    check mockListener.'start();

    BedrockManagedKnowledgeBase|ai:Error kb = new (
        {name: "dup-kb", roleArn: "arn:aws:iam::123456789012:role/service-role/bedrock-kb"},
        KB_TEST_CREDS, "us-east-1", serviceUrl = string `http://localhost:${port}`);
    check mockListener.gracefulStop();

    test:assertTrue(kb is ai:Error);
    if kb is ai:Error {
        string msg = kb.message();
        test:assertTrue(msg.includes("KBAAAAAAAA") && msg.includes("KBBBBBBBBB"), msg);
    }
}

// ---- no CUSTOM data source ----

const string NO_CUSTOM_KB_ID = "KBNOCUSTOM";

isolated service class NoCustomDataSourceMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json|error {
        if req.rawPath == string `/knowledgebases/${NO_CUSTOM_KB_ID}` {
            return {knowledgeBase: {knowledgeBaseId: NO_CUSTOM_KB_ID, name: "kb", status: "ACTIVE"}};
        }
        return error(string `unexpected GET ${req.rawPath}`);
    }

    isolated resource function post [string... path](http:Request req) returns json|error {
        if req.rawPath == string `/knowledgebases/${NO_CUSTOM_KB_ID}/datasources/` {
            return {
                dataSourceSummaries: [
                    {
                        knowledgeBaseId: NO_CUSTOM_KB_ID,
                        dataSourceId: "DSSHAREPT1",
                        name: "sp-src",
                        status: "AVAILABLE",
                        updatedAt: "2026-08-13T00:00:00Z"
                    }
                ]
            };
        }
        return error(string `unexpected POST ${req.rawPath}`);
    }
}

@test:Config {}
function testNoCustomDataSourceFailsAtConstruction() returns error? {
    final int port = 18662;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new NoCustomDataSourceMock(), "/");
    check mockListener.'start();

    BedrockManagedKnowledgeBase|ai:Error kb =
        new (NO_CUSTOM_KB_ID, KB_TEST_CREDS, "us-east-1", serviceUrl = string `http://localhost:${port}`);
    check mockListener.gracefulStop();

    test:assertTrue(kb is ai:Error);
    if kb is ai:Error {
        test:assertTrue(kb.message().includes("no 'CUSTOM' data source") || kb.message().includes("CUSTOM"),
            kb.message());
    }
}

// ---- several CUSTOM data sources ----

const string MULTI_CUSTOM_KB_ID = "KBMULTICUS";

isolated service class MultipleCustomDataSourcesMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${MULTI_CUSTOM_KB_ID}` {
            return {knowledgeBase: {knowledgeBaseId: MULTI_CUSTOM_KB_ID, name: "kb", status: "ACTIVE"}};
        }
        if p == string `/knowledgebases/${MULTI_CUSTOM_KB_ID}/datasources/DSCUSTOMA1`
                || p == string `/knowledgebases/${MULTI_CUSTOM_KB_ID}/datasources/DSCUSTOMB1` {
            return {dataSource: {dataSourceConfiguration: {'type: "CUSTOM"}}};
        }
        return error(string `unexpected GET ${p}`);
    }

    isolated resource function post [string... path](http:Request req) returns json|error {
        if req.rawPath == string `/knowledgebases/${MULTI_CUSTOM_KB_ID}/datasources/` {
            return {
                dataSourceSummaries: [
                    {
                        knowledgeBaseId: MULTI_CUSTOM_KB_ID,
                        dataSourceId: "DSCUSTOMA1",
                        name: "custom-a",
                        status: "AVAILABLE",
                        updatedAt: "2026-08-13T00:00:00Z"
                    },
                    {
                        knowledgeBaseId: MULTI_CUSTOM_KB_ID,
                        dataSourceId: "DSCUSTOMB1",
                        name: "custom-b",
                        status: "AVAILABLE",
                        updatedAt: "2026-08-13T00:00:00Z"
                    }
                ]
            };
        }
        return error(string `unexpected POST ${req.rawPath}`);
    }
}

@test:Config {}
function testMultipleCustomDataSourcesFailAtConstruction() returns error? {
    final int port = 18663;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new MultipleCustomDataSourcesMock(), "/");
    check mockListener.'start();

    BedrockManagedKnowledgeBase|ai:Error kb =
        new (MULTI_CUSTOM_KB_ID, KB_TEST_CREDS, "us-east-1", serviceUrl = string `http://localhost:${port}`);
    check mockListener.gracefulStop();

    test:assertTrue(kb is ai:Error);
    if kb is ai:Error {
        string msg = kb.message();
        test:assertTrue(msg.includes("DSCUSTOMA1") && msg.includes("DSCUSTOMB1"), msg);
        test:assertTrue(msg.includes("dataSourceId"), msg);
    }
}

// ---- a VECTOR (customer-owned vector store) knowledge base ----

const string VECTOR_KB_ID = "KBVECTOR01";

// A VECTOR knowledge base is ACTIVE and otherwise perfectly well-formed — it is only
// the `knowledgeBaseConfiguration.type` that makes it unusable here. The data source
// endpoints are deliberately NOT stubbed: construction must fail on the type check
// before it ever reaches them, so any request past that point surfaces as an
// 'unexpected GET' rather than passing quietly.
isolated service class VectorKnowledgeBaseMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json|error {
        if req.rawPath == string `/knowledgebases/${VECTOR_KB_ID}` {
            return {
                knowledgeBase: {
                    knowledgeBaseId: VECTOR_KB_ID,
                    name: "self-managed-kb",
                    status: "ACTIVE",
                    knowledgeBaseConfiguration: {
                        'type: "VECTOR",
                        vectorKnowledgeBaseConfiguration: {
                            embeddingModelArn: "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0"
                        }
                    }
                }
            };
        }
        return error(string `unexpected GET ${req.rawPath}`);
    }
}

@test:Config {}
function testVectorKnowledgeBaseIsRefusedAtConstruction() returns error? {
    final int port = 18664;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorKnowledgeBaseMock(), "/");
    check mockListener.'start();

    BedrockManagedKnowledgeBase|ai:Error kb =
        new (VECTOR_KB_ID, KB_TEST_CREDS, "us-east-1", serviceUrl = string `http://localhost:${port}`);
    check mockListener.gracefulStop();

    // Everything this class does was measured on the managed search branch against
    // Bedrock's own vector store; a VECTOR knowledge base is served by a different
    // branch and a different ranking engine, where `deleteByFilter`'s pinned probe
    // has never been shown sound. Refusing at construction is the whole point — a
    // half-working retrieve() with a silently under-deleting deleteByFilter is worse
    // than no support at all.
    test:assertTrue(kb is ai:Error);
    if kb is ai:Error {
        string msg = kb.message();
        test:assertTrue(msg.includes("VECTOR"), msg);
        test:assertTrue(msg.includes("MANAGED"), msg);
    }
}

// ---- CreateKnowledgeBase embedding model / KMS ----
//
// AWS is strict about which fields accompany which embedding type: "When using
// MANAGED, you must not specify embeddingModelArn or embeddingModelConfiguration.
// When using CUSTOM, both fields are required."
// https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-create.html

const string TITAN_EMBED_ARN = "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0";

isolated function managedKbConfigOf(KnowledgeBaseDefinition def) returns map<json> {
    map<json> body = createKnowledgeBaseRequestBody(def);
    return <map<json>>(<map<json>>body["knowledgeBaseConfiguration"])["managedKnowledgeBaseConfiguration"];
}

@test:Config {}
function testDefaultUsesServiceManagedEmbeddingAndSendsNoArn() {
    map<json> managed = managedKbConfigOf({name: "kb", roleArn: "arn:aws:iam::1:role/r"});
    test:assertEquals(managed["embeddingModelType"], "MANAGED");
    // Sending either of these alongside MANAGED is rejected by AWS.
    test:assertFalse(managed.hasKey("embeddingModelArn"));
    test:assertFalse(managed.hasKey("embeddingModelConfiguration"));
}

@test:Config {}
function testCallerSuppliedEmbeddingModelSendsArnAndConfiguration() {
    map<json> managed = managedKbConfigOf({
        name: "kb",
        roleArn: "arn:aws:iam::1:role/r",
        embeddingModel: {embeddingModelArn: TITAN_EMBED_ARN}
    });
    test:assertEquals(managed["embeddingModelType"], "CUSTOM");
    test:assertEquals(managed["embeddingModelArn"], TITAN_EMBED_ARN);
    map<json> bedrockConfig = <map<json>>(<map<json>>managed["embeddingModelConfiguration"])
        ["bedrockEmbeddingModelConfiguration"];
    // AWS requires 1024 dimensions and float32 on a managed knowledge base.
    test:assertEquals(bedrockConfig["dimensions"], 1024);
    test:assertEquals(bedrockConfig["embeddingDataType"], "FLOAT32");
}

@test:Config {}
function testKmsKeyIsSentOnlyWhenConfigured() {
    map<json> without = managedKbConfigOf({name: "kb", roleArn: "arn:aws:iam::1:role/r"});
    test:assertFalse(without.hasKey("serverSideEncryptionConfiguration"));
    map<json> with = managedKbConfigOf({
        name: "kb",
        roleArn: "arn:aws:iam::1:role/r",
        kmsKeyArn: "arn:aws:kms:us-east-1:1:key/abc"
    });
    map<json> sse = <map<json>>with["serverSideEncryptionConfiguration"];
    test:assertEquals(sse["kmsKeyArn"], "arn:aws:kms:us-east-1:1:key/abc");
}

// ---- managed reranker vs caller-supplied embedding model ----

@test:Config {}
function testManagedRerankerWithCustomEmbeddingModelIsRefused() {
    KnowledgeBaseDefinition def = {
        name: "kb",
        roleArn: "arn:aws:iam::1:role/r",
        embeddingModel: {embeddingModelArn: TITAN_EMBED_ARN}
    };
    // Both choices are permanent at creation time, so this must fail before any I/O —
    // learning it from a failed retrieve() means the KB is already built wrong.
    ai:Error? guard = guardEmbeddingModelAgainstReranker(def, RERANKING_MANAGED);
    test:assertTrue(guard is ai:Error);
    if guard is ai:Error {
        test:assertTrue(guard.message().includes("embeddingModel"), guard.message());
        test:assertTrue(guard.message().includes("managed reranker"), guard.message());
    }
}

@test:Config {}
function testRerankerGuardAllowsEveryOtherCombination() {
    KnowledgeBaseDefinition custom = {
        name: "kb",
        roleArn: "arn:aws:iam::1:role/r",
        embeddingModel: {embeddingModelArn: TITAN_EMBED_ARN}
    };
    KnowledgeBaseDefinition managed = {name: "kb", roleArn: "arn:aws:iam::1:role/r"};
    test:assertTrue(guardEmbeddingModelAgainstReranker(custom, RERANKING_NONE) is ());
    test:assertTrue(guardEmbeddingModelAgainstReranker(custom, ()) is ());
    test:assertTrue(guardEmbeddingModelAgainstReranker(managed, RERANKING_MANAGED) is ());
    // Attaching by id says nothing about how the KB was configured, so no guard.
    test:assertTrue(guardEmbeddingModelAgainstReranker("KBEXISTING", RERANKING_MANAGED) is ());
}
