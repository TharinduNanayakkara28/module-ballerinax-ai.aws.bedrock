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

// `deleteByFilter()` against a stubbed agent pair. The load-bearing assertion is
// that the per-document probe filters on `x-amz-bedrock-kb-source-uri` and NOT on
// the managed `_source_uri` — the two knowledge base types use different reserved
// metadata prefixes, and using the managed spelling here would match nothing, so
// nothing would be deleted and no error would be raised.
// Prefix rule: https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-config.html
// The exact key, under "Auto-created fields":
// https://docs.aws.amazon.com/bedrock/latest/userguide/kb-multimodal-test-and-query.html

const string VDEL_KB_ID = "KBVECDEL01";
const string VDEL_DS_ID = "DSVECDEL01";

// `probe-match` matches the caller's filter; `probe-miss` does not but IS reachable
// on its own; `probe-hidden` is reachable under neither, i.e. indeterminate.
const string VDEL_MATCH_ID = "probe-match";
const string VDEL_MISS_ID = "probe-miss";
const string VDEL_HIDDEN_ID = "probe-hidden";

isolated json[] vectorProbeFilters = [];
isolated json[] vectorDeletePayloads = [];

isolated function recordVectorProbe(json filter) {
    lock {
        vectorProbeFilters.push(filter.clone());
    }
}

isolated function readVectorProbes() returns json[] {
    lock {
        return vectorProbeFilters.clone();
    }
}

isolated function recordVectorDelete(json payload) {
    lock {
        vectorDeletePayloads.push(payload.clone());
    }
}

isolated function readVectorDeletes() returns json[] {
    lock {
        return vectorDeletePayloads.clone();
    }
}

// Reads the pinned document id out of a probe filter, whichever shape it took.
isolated function pinnedIdOf(json filter) returns string? {
    map<json> f = <map<json>>filter;
    if f.hasKey("equals") {
        return <string>(<map<json>>f["equals"])["value"];
    }
    json[] clauses = <json[]>f["andAll"];
    foreach json clause in clauses {
        map<json> c = <map<json>>clause;
        if c.hasKey("equals") {
            map<json> leaf = <map<json>>c["equals"];
            if leaf["key"] == VECTOR_SOURCE_URI_METADATA_KEY {
                return <string>leaf["value"];
            }
        }
    }
    return ();
}

isolated function isPinnedOnlyProbe(json filter) returns boolean => (<map<json>>filter).hasKey("equals");

isolated service class VectorDeleteMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${VDEL_KB_ID}` {
            return {
                knowledgeBase: {
                    knowledgeBaseId: VDEL_KB_ID,
                    name: "kb",
                    status: "ACTIVE",
                    knowledgeBaseConfiguration: {'type: "VECTOR"}
                }
            };
        }
        if p == string `/knowledgebases/${VDEL_KB_ID}/datasources/${VDEL_DS_ID}` {
            return {
                dataSource: {
                    dataSourceId: VDEL_DS_ID,
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

        if p == string `/knowledgebases/${VDEL_KB_ID}/datasources/` {
            return {dataSourceSummaries: [{dataSourceId: VDEL_DS_ID, name: "ds"}]};
        }

        // ListKnowledgeBaseDocuments — exhaustive, and carries no metadata at all,
        // which is exactly why the probe below has to exist.
        if p == string `/knowledgebases/${VDEL_KB_ID}/datasources/${VDEL_DS_ID}/documents` {
            return {
                documentDetails: [VDEL_MATCH_ID, VDEL_MISS_ID, VDEL_HIDDEN_ID].map(id => <json>{
                    identifier: {dataSourceType: "CUSTOM", custom: {id}},
                    status: "INDEXED"
                })
            };
        }

        if p == string `/knowledgebases/${VDEL_KB_ID}/datasources/${VDEL_DS_ID}/documents/deleteDocuments` {
            json body = check req.getJsonPayload();
            recordVectorDelete((<map<json>>body)["documentIdentifiers"] ?: []);
            return {documentDetails: []};
        }

        if p == string `/knowledgebases/${VDEL_KB_ID}/retrieve` {
            json body = check req.getJsonPayload();
            map<json> vectorSearch = <map<json>>(<map<json>>(<map<json>>body)["retrievalConfiguration"])
                ["vectorSearchConfiguration"];
            json filter = vectorSearch["filter"] ?: ();
            recordVectorProbe(filter);

            string? pinned = pinnedIdOf(filter);
            boolean pinnedOnly = isPinnedOnlyProbe(filter);
            // probe-match: hits under the caller's filter.
            if pinned == VDEL_MATCH_ID {
                return {retrievalResults: [vecDelResult(VDEL_MATCH_ID)]};
            }
            // probe-miss: no hit under the caller's filter, but reachable alone —
            // genuinely excluded, so it must be skipped silently.
            if pinned == VDEL_MISS_ID {
                return pinnedOnly ? {retrievalResults: [vecDelResult(VDEL_MISS_ID)]} : {retrievalResults: []};
            }
            // probe-hidden: unreachable either way — indeterminate, must be reported.
            return {retrievalResults: []};
        }
        return error(string `unexpected POST ${p}`);
    }
}

isolated function vecDelResult(string id) returns json => {
    content: {text: string `text for ${id}`, 'type: "TEXT"},
    documentId: id,
    location: {'type: "CUSTOM", customDocumentLocation: {id}},
    metadata: {"x-amz-bedrock-kb-source-uri": id},
    score: 0.95
};

// `retrievalResultIdentifies` unit tests. The mock results above populate the
// metadata key, `customDocumentLocation.id` AND `documentId` together, so on their
// own they would keep passing with two of the three branches deleted. These isolate
// each branch, and pin the deliberate REJECTION of `documentId`.
@test:Config {}
function testRetrievalResultIdentifiesByMetadataKeyAlone() {
    json result = {content: {text: "t", 'type: "TEXT"}, metadata: {"x-amz-bedrock-kb-source-uri": "doc-1"}};
    test:assertTrue(retrievalResultIdentifies(result, "doc-1"));
    test:assertFalse(retrievalResultIdentifies(result, "doc-2"));
}

@test:Config {}
function testRetrievalResultIdentifiesByCustomLocationAlone() {
    json result = {content: {text: "t", 'type: "TEXT"},
        location: {'type: "CUSTOM", customDocumentLocation: {id: "doc-1"}}};
    test:assertTrue(retrievalResultIdentifies(result, "doc-1"));
    test:assertFalse(retrievalResultIdentifies(result, "doc-2"));
}

// The S3 branch — `listDeletableDocuments` uses the S3 URI as `sourceValue`, so this
// is the only thing that can identify a document on an S3 data source.
@test:Config {}
function testRetrievalResultIdentifiesByS3LocationAlone() {
    json result = {content: {text: "t", 'type: "TEXT"},
        location: {'type: "S3", s3Location: {uri: "s3://bucket/docs/a.txt"}}};
    test:assertTrue(retrievalResultIdentifies(result, "s3://bucket/docs/a.txt"));
    test:assertFalse(retrievalResultIdentifies(result, "s3://bucket/docs/b.txt"));
}

// `documentId` is a service-side id with no documented equality to the custom
// identifier or the S3 URI, so it must NOT count as proof of identity — accepting it
// would admit a relation AWS never promised, and a false positive here deletes.
@test:Config {}
function testRetrievalResultDoesNotIdentifyByDocumentIdAlone() {
    json result = {content: {text: "t", 'type: "TEXT"}, documentId: "doc-1"};
    test:assertFalse(retrievalResultIdentifies(result, "doc-1"),
        "documentId must not be accepted as proof of document identity");
}

// A result carrying no identity information at all cannot identify anything — the
// caller must treat it as unreachable, never as a match.
@test:Config {}
function testRetrievalResultWithNoIdentityFieldsIdentifiesNothing() {
    json result = {content: {text: "t", 'type: "TEXT"}, score: 0.99};
    test:assertFalse(retrievalResultIdentifies(result, "doc-1"));
}

// THE regression guard for the reserved-prefix split. If this ever asserts
// `_source_uri`, deleteByFilter on a self-managed knowledge base silently deletes
// nothing.
@test:Config {}
function testDeleteByFilterProbesOnTheVectorSourceUriKey() returns error? {
    final int port = 18701;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorDeleteMock(), "/");
    check mockListener.'start();
    lock {
        vectorProbeFilters.removeAll();
    }
    lock {
        vectorDeletePayloads.removeAll();
    }

    BedrockVectorKnowledgeBase kb = check new (VDEL_KB_ID, KB_TEST_CREDS, "us-east-1",
        string `http://localhost:${port}`);
    ai:MetadataFilters filters = {filters: [{key: "tenant", operator: ai:EQUAL, value: "acme"}]};
    ai:Error? result = kb.deleteByFilter(filters);
    check mockListener.gracefulStop();

    json[] probes = readVectorProbes();
    test:assertTrue(probes.length() > 0, "no probe was issued");
    foreach json probe in probes {
        string serialized = probe.toJsonString();
        test:assertTrue(serialized.includes("x-amz-bedrock-kb-source-uri"),
            string `probe did not pin on the self-managed key: ${serialized}`);
        test:assertFalse(serialized.includes("_source_uri"),
            string `probe used the MANAGED reserved key: ${serialized}`);
    }

    // The first probe ANDs the caller's filter with the pin.
    json expectedFirst = {
        andAll: [
            {'equals: {key: "tenant", value: "acme"}},
            {'equals: {key: "x-amz-bedrock-kb-source-uri", value: VDEL_MATCH_ID}}
        ]
    };
    test:assertEquals(probes[0], expectedFirst);

    // Only the matching document is deleted.
    json[] deletes = readVectorDeletes();
    test:assertEquals(deletes.length(), 1);
    json[] identifiers = <json[]>deletes[0];
    test:assertEquals(identifiers.length(), 1);
    test:assertEquals(identifiers[0], {dataSourceType: "CUSTOM", custom: {id: VDEL_MATCH_ID}});

    // `probe-hidden` was reachable under neither probe, so it is reported rather
    // than silently skipped.
    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        string msg = result.message();
        test:assertTrue(msg.includes(VDEL_HIDDEN_ID), msg);
        test:assertFalse(msg.includes(VDEL_MISS_ID),
            string `a document genuinely excluded by the filter must not be reported: ${msg}`);
    }
}

// A zero-hit first probe is re-probed with the pin ALONE, to tell "excluded by the
// filter" from "the store's relevance floor hid it". The managed class dropped this
// second probe on the strength of a measurement taken against Bedrock's own vector
// store; that measurement does not transfer to a customer-owned one.
@test:Config {}
function testDeleteByFilterReprobesWithThePinAloneOnAZeroHit() returns error? {
    final int port = 18702;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorDeleteMock(), "/");
    check mockListener.'start();
    lock {
        vectorProbeFilters.removeAll();
    }

    BedrockVectorKnowledgeBase kb = check new (VDEL_KB_ID, KB_TEST_CREDS, "us-east-1",
        string `http://localhost:${port}`);
    // The returned error is asserted by the sibling test; here only the probe
    // sequence matters.
    ai:Error? ignored = kb.deleteByFilter({filters: [{key: "tenant", operator: ai:EQUAL, value: "acme"}]});
    test:assertTrue(ignored is ai:Error || ignored is ());
    check mockListener.gracefulStop();

    json[] probes = readVectorProbes();
    // probe-match hits on the first probe (1). probe-miss and probe-hidden each miss
    // and are re-probed (2 each) — five in total.
    test:assertEquals(probes.length(), 5, probes.toJsonString());

    json[] pinnedOnly = probes.filter(isPinnedOnlyProbe);
    test:assertEquals(pinnedOnly.length(), 2, "expected one bare-pin re-probe per zero-hit document");
    foreach json probe in pinnedOnly {
        test:assertEquals((<map<json>>(<map<json>>probe)["equals"])["key"], VECTOR_SOURCE_URI_METADATA_KEY);
    }
}

// A store that IGNORES the metadata filter must not cause a mass delete. This mock
// answers every probe with the same top-scoring chunk, which is what an unfiltered
// top-1 query looks like — AWS documents exactly this for MongoDB Atlas, where
// "Metadata filtering doesn't work by default". Counting results would mark every
// document a match and wipe the knowledge base; checking identity must not.
isolated service class VectorIgnoresFilterMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${VDEL_KB_ID}` {
            return {
                knowledgeBase: {
                    knowledgeBaseId: VDEL_KB_ID,
                    name: "kb",
                    status: "ACTIVE",
                    knowledgeBaseConfiguration: {'type: "VECTOR"}
                }
            };
        }
        if p == string `/knowledgebases/${VDEL_KB_ID}/datasources/${VDEL_DS_ID}` {
            return {
                dataSource: {
                    dataSourceId: VDEL_DS_ID,
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
        if p == string `/knowledgebases/${VDEL_KB_ID}/datasources/` {
            return {dataSourceSummaries: [{dataSourceId: VDEL_DS_ID, name: "ds"}]};
        }
        if p == string `/knowledgebases/${VDEL_KB_ID}/datasources/${VDEL_DS_ID}/documents` {
            return {
                documentDetails: [VDEL_MATCH_ID, VDEL_MISS_ID, VDEL_HIDDEN_ID].map(id => <json>{
                    identifier: {dataSourceType: "CUSTOM", custom: {id}},
                    status: "INDEXED"
                })
            };
        }
        if p == string `/knowledgebases/${VDEL_KB_ID}/datasources/${VDEL_DS_ID}/documents/deleteDocuments` {
            json body = check req.getJsonPayload();
            recordVectorDelete((<map<json>>body)["documentIdentifiers"] ?: []);
            return {documentDetails: []};
        }
        if p == string `/knowledgebases/${VDEL_KB_ID}/retrieve` {
            // Always the same unrelated document, whatever the filter said.
            return {retrievalResults: [vecDelResult("some-other-document")]};
        }
        return error(string `unexpected POST ${p}`);
    }
}

@test:Config {}
function testDeleteByFilterDoesNotMassDeleteWhenTheStoreIgnoresTheFilter() returns error? {
    final int port = 18704;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorIgnoresFilterMock(), "/");
    check mockListener.'start();
    lock {
        vectorDeletePayloads.removeAll();
    }

    BedrockVectorKnowledgeBase kb = check new (VDEL_KB_ID, KB_TEST_CREDS, "us-east-1",
        string `http://localhost:${port}`);
    ai:Error? result = kb.deleteByFilter({filters: [{key: "tenant", operator: ai:EQUAL, value: "acme"}]});
    check mockListener.gracefulStop();

    // NOTHING may be deleted: no probe ever returned the document it pinned.
    test:assertEquals(readVectorDeletes().length(), 0,
        "a store that ignores the filter must not cause any deletion");

    // And the caller must be told, not left thinking it succeeded.
    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        string msg = result.message();
        foreach string id in [VDEL_MATCH_ID, VDEL_MISS_ID, VDEL_HIDDEN_ID] {
            test:assertTrue(msg.includes(id), string `${id} was not reported as indeterminate: ${msg}`);
        }
    }
}

// An empty `ai:MetadataFilters` maps to no filter at all. Left unguarded, every
// probe becomes "does this document exist" and the whole knowledge base is deleted.
// A group made only of EMPTY sub-groups is not nil — `metadataFiltersToRetrievalFilter`
// yields `{"andAll": [null, null]}` for it — so a plain nil check would let it through
// and it would select every document.
@test:Config {}
function testDeleteByFilterRefusesNestedEmptyFilterGroups() returns error? {
    final int port = 18706;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorDeleteMock(), "/");
    check mockListener.'start();
    lock {
        vectorDeletePayloads.removeAll();
    }

    BedrockVectorKnowledgeBase kb = check new (VDEL_KB_ID, KB_TEST_CREDS, "us-east-1",
        string `http://localhost:${port}`);
    ai:MetadataFilters nestedEmpty = {filters: [{filters: []}, {filters: []}]};
    // Confirm the premise: this really is non-nil, so the nil check alone cannot
    // catch it. If Ballerina's filter mapping ever starts returning () here, this
    // assertion fails and the guard can be simplified.
    json? mapped = check metadataFiltersToRetrievalFilter(nestedEmpty);
    test:assertTrue(mapped !is (), "premise broken: nested empty groups now map to nil");

    ai:Error? result = kb.deleteByFilter(nestedEmpty);
    check mockListener.gracefulStop();

    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        test:assertTrue(result.message().includes("at least one"), result.message());
    }
    test:assertEquals(readVectorDeletes().length(), 0, "a filter with no leaf predicates must delete nothing");
}

@test:Config {}
function testDeleteByFilterRefusesAnEmptyFilterSet() returns error? {
    final int port = 18705;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorDeleteMock(), "/");
    check mockListener.'start();
    lock {
        vectorDeletePayloads.removeAll();
    }

    BedrockVectorKnowledgeBase kb = check new (VDEL_KB_ID, KB_TEST_CREDS, "us-east-1",
        string `http://localhost:${port}`);
    ai:Error? result = kb.deleteByFilter({filters: []});
    check mockListener.gracefulStop();

    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        test:assertTrue(result.message().includes("at least one"), result.message());
    }
    test:assertEquals(readVectorDeletes().length(), 0, "an empty filter set must delete nothing");
}

// Data sources other than CUSTOM/S3 cannot be deleted through this API at all —
// `DocumentIdentifier.dataSourceType` has only those two members. They must be
// named in the error rather than silently skipped.
isolated service class VectorUndeletableSourceMock {
    *http:Service;

    isolated resource function get [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${VDEL_KB_ID}` {
            return {
                knowledgeBase: {
                    knowledgeBaseId: VDEL_KB_ID,
                    name: "kb",
                    status: "ACTIVE",
                    knowledgeBaseConfiguration: {'type: "VECTOR"}
                }
            };
        }
        if p == string `/knowledgebases/${VDEL_KB_ID}/datasources/${VDEL_DS_ID}` {
            return {
                dataSource: {
                    dataSourceId: VDEL_DS_ID,
                    status: "AVAILABLE",
                    dataSourceConfiguration: {'type: "CUSTOM"},
                    vectorIngestionConfiguration: {chunkingConfiguration: {chunkingStrategy: "FIXED_SIZE"}}
                }
            };
        }
        if p == string `/knowledgebases/${VDEL_KB_ID}/datasources/DSSHAREPT1` {
            return {
                dataSource: {
                    dataSourceId: "DSSHAREPT1",
                    dataSourceConfiguration: {'type: "SHAREPOINT"}
                }
            };
        }
        return error(string `unexpected GET ${p}`);
    }

    isolated resource function post [string... path](http:Request req) returns json|error {
        string p = req.rawPath;
        if p == string `/knowledgebases/${VDEL_KB_ID}/datasources/` {
            return {
                dataSourceSummaries: [
                    {dataSourceId: VDEL_DS_ID, name: "custom-ds"},
                    {dataSourceId: "DSSHAREPT1", name: "sharepoint-ds"}
                ]
            };
        }
        if p == string `/knowledgebases/${VDEL_KB_ID}/datasources/${VDEL_DS_ID}/documents` {
            return {documentDetails: []};
        }
        return error(string `unexpected POST ${p}`);
    }
}

@test:Config {}
function testDeleteByFilterNamesUndeletableDataSources() returns error? {
    final int port = 18703;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new VectorUndeletableSourceMock(), "/");
    check mockListener.'start();

    BedrockVectorKnowledgeBase kb = check new (VDEL_KB_ID, KB_TEST_CREDS, "us-east-1",
        string `http://localhost:${port}`);
    ai:Error? result = kb.deleteByFilter({filters: [{key: "tenant", operator: ai:EQUAL, value: "acme"}]});
    check mockListener.gracefulStop();

    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        string msg = result.message();
        test:assertTrue(msg.includes("DSSHAREPT1"), msg);
        test:assertTrue(msg.includes("SHAREPOINT"), msg);
        test:assertTrue(msg.includes("CUSTOM/S3"), msg);
    }
}
