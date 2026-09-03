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

// End-to-end test of `deleteByFilter`'s probe algorithm against a stubbed
// bedrock-agent/bedrock-agent-runtime pair on a local listener. Both agent planes
// resolve to the SAME `serviceUrl` here (a concrete `http://localhost:...` carries
// no `{endpoint}` placeholder to vary), so one mock serves both.
//
// Scenario: one CUSTOM data source with three INDEXED documents —
//   - 'doc-match': the probe (userFilter AND _source_uri==id) finds it -> deleted.
//   - 'doc-skip-a'/'doc-skip-b': the probe finds nothing -> genuinely excluded by
//     the filter -> left alone, and NOT reported as a problem.
// Plus one SHAREPOINT data source, which `DocumentIdentifier` cannot address at all
// -> reported as undeletable, never silently ignored.
//
// A zero-hit probe is unambiguous BECAUSE the filter is pinned to one document:
// with the candidate set narrowed to that document's chunks, nothing can crowd it
// out and no relevance floor can hide it (measured against the live API).
// So there is exactly ONE probe per candidate document, asserted below.

const string DEL_KB_ID = "KBDELTEST1";
const string DEL_DS_CUSTOM = "DSCUSTOM01";
const string DEL_DS_SHAREPOINT = "DSSHAREPT1";

isolated json[] deletedIdentifiers = [];
isolated int retrieveProbeCount = 0;

isolated function recordDeletedIdentifiers(json[] identifiers) {
    lock {
        deletedIdentifiers.push(...identifiers.clone());
    }
}

isolated function readDeletedIdentifiers() returns json[] {
    lock {
        return deletedIdentifiers.clone();
    }
}

isolated function recordRetrieveProbe() {
    lock {
        retrieveProbeCount += 1;
    }
}

isolated function readRetrieveProbeCount() returns int {
    lock {
        return retrieveProbeCount;
    }
}

isolated function deleteTestDocDetail(string id) returns json => {
    knowledgeBaseId: DEL_KB_ID,
    dataSourceId: DEL_DS_CUSTOM,
    identifier: {dataSourceType: "CUSTOM", custom: {id}},
    status: "INDEXED",
    updatedAt: "2026-08-13T00:00:00Z"
};

isolated function deleteTestRetrievalResult(string id) returns json => {
    content: {text: string `chunk text for ${id}`, 'type: "TEXT"},
    documentId: id,
    location: {'type: "CUSTOM", customDocumentLocation: {id}},
    metadata: {"_source_uri": id},
    score: 0.9
};

isolated service class DeleteTestMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${DEL_KB_ID}` {
            return {
                knowledgeBase: {
                    knowledgeBaseId: DEL_KB_ID,
                    name: "kb",
                    status: "ACTIVE",
                    knowledgeBaseConfiguration: {'type: "MANAGED"}
                }
            };
        }
        if p == string `/knowledgebases/${DEL_KB_ID}/datasources/${DEL_DS_CUSTOM}` {
            return {
                dataSource: {
                    knowledgeBaseId: DEL_KB_ID,
                    dataSourceId: DEL_DS_CUSTOM,
                    name: "custom-src",
                    status: "AVAILABLE",
                    dataSourceConfiguration: {'type: "CUSTOM"},
                    vectorIngestionConfiguration: {chunkingConfiguration: {chunkingStrategy: "FIXED_SIZE"}}
                }
            };
        }
        if p == string `/knowledgebases/${DEL_KB_ID}/datasources/${DEL_DS_SHAREPOINT}` {
            return {
                dataSource: {
                    knowledgeBaseId: DEL_KB_ID,
                    dataSourceId: DEL_DS_SHAREPOINT,
                    name: "sp-src",
                    status: "AVAILABLE",
                    dataSourceConfiguration: {'type: "SHAREPOINT"}
                }
            };
        }
        return error(string `unexpected GET ${p}`);
    }

    isolated resource function post [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        json body = check req.getJsonPayload();

        if p == string `/knowledgebases/${DEL_KB_ID}/datasources/` {
            return {
                dataSourceSummaries: [
                    {
                        knowledgeBaseId: DEL_KB_ID,
                        dataSourceId: DEL_DS_CUSTOM,
                        name: "custom-src",
                        status: "AVAILABLE",
                        updatedAt: "2026-08-13T00:00:00Z"
                    },
                    {
                        knowledgeBaseId: DEL_KB_ID,
                        dataSourceId: DEL_DS_SHAREPOINT,
                        name: "sp-src",
                        status: "AVAILABLE",
                        updatedAt: "2026-08-13T00:00:00Z"
                    }
                ]
            };
        }
        if p == string `/knowledgebases/${DEL_KB_ID}/datasources/${DEL_DS_CUSTOM}/documents` {
            return {
                documentDetails: [
                    deleteTestDocDetail("doc-match"),
                    deleteTestDocDetail("doc-skip-a"),
                    deleteTestDocDetail("doc-skip-b")
                ]
            };
        }
        if p == string `/knowledgebases/${DEL_KB_ID}/datasources/${DEL_DS_CUSTOM}/documents/deleteDocuments` {
            map<json> bodyMap = <map<json>>body;
            json[] identifiers = <json[]>bodyMap["documentIdentifiers"];
            recordDeletedIdentifiers(identifiers);
            return {documentDetails: []};
        }
        if p == string `/knowledgebases/${DEL_KB_ID}/retrieve` {
            return deleteTestMockRetrieve(body);
        }
        return error(string `unexpected POST ${p}`);
    }
}

// The probe dispatcher: decides the canned `retrievalResults[]` from which document
// id the filter mentions. Every probe must be the COMBINED form (user filter AND the
// id leaf — an 'andAll') and must ask for exactly one result; both are asserted here
// rather than in the test body, so a regression fails at the request that made it.
isolated function deleteTestMockRetrieve(json body) returns json|error {
    string bodyStr = body.toJsonString();
    recordRetrieveProbe();
    if !bodyStr.includes("andAll") || !bodyStr.includes("\"numberOfResults\":1") {
        return error(string `probe was not a pinned single-result query: ${bodyStr}`);
    }
    if bodyStr.includes("doc-match") {
        return {retrievalResults: [deleteTestRetrievalResult("doc-match")]};
    }
    // 'doc-skip-a'/'doc-skip-b': zero hits under the pinned filter. Because the pin
    // rules out the relevance floor, that means "genuinely excluded" — not
    // "undeterminable" — so they are simply left alone and never reported.
    return {retrievalResults: []};
}

@test:Config {}
function testDeleteByFilterDeletesMatchesSkipsExclusionsAndReportsUndeletableSources() returns error? {
    final int port = 18651;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new DeleteTestMock(), "/");
    check mockListener.'start();

    BedrockManagedKnowledgeBase kb = check new (
        DEL_KB_ID, KB_TEST_CREDS, "us-east-1",
        serviceUrl = string `http://localhost:${port}`,
        dataSourceId = DEL_DS_CUSTOM);

    ai:MetadataFilters filters = {filters: [{key: "tenant", operator: ai:EQUAL, value: "acme"}]};
    ai:Error? result = kb.deleteByFilter(filters);
    check mockListener.gracefulStop();

    // The confirmed match was deleted regardless of the undeletable data source
    // elsewhere — a partial failure must not withhold the deletes it COULD make.
    json[] deleted = readDeletedIdentifiers();
    test:assertEquals(deleted.length(), 1, deleted.toJsonString());
    map<json> deletedIdentifier = <map<json>>deleted[0];
    test:assertEquals((<map<json>>deletedIdentifier["custom"])["id"], "doc-match");

    // ONE probe per candidate document, never two. The second (id-alone
    // reachability) probe existed only to tell "hidden by the relevance floor" from
    // "excluded by the filter"; pinning makes that distinction impossible to need,
    // so a reappearance of 2N probing is a regression.
    test:assertEquals(readRetrieveProbeCount(), 3, "expected exactly one probe per candidate document");

    // The undeletable data source is reported; the two excluded documents are not —
    // exclusion is now a sound conclusion, not an unresolved one.
    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        string msg = result.message();
        test:assertTrue(msg.includes(DEL_DS_SHAREPOINT) || msg.includes("SHAREPOINT"), msg);
        test:assertFalse(msg.includes("doc-skip"), msg);
    }
}
