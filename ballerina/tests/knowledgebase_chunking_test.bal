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

// Pure tests: chunker-default detection, the connectorParameters unwrap, and the
// CreateDataSource chunking-strategy body. No I/O.

@test:Config {}
function testChunkerDefaultsToDisableOnAServerChunkingStrategy() returns error? {
    ai:Chunker|ai:AUTO|ai:DISABLE|ai:Error result = resolveChunker((), FIXED_SIZE);
    test:assertTrue(result is ai:DISABLE, result is ai:Error ? result.message() : result.toString());
}

@test:Config {}
function testChunkerDefaultsToAutoWhenTheStrategyIsNone() returns error? {
    ai:Chunker|ai:AUTO|ai:DISABLE|ai:Error result = resolveChunker((), NONE);
    test:assertTrue(result is ai:AUTO, result is ai:Error ? result.message() : result.toString());
}

@test:Config {}
function testExplicitChunkerAgainstAServerChunkingStrategyIsAConstructionError() {
    ai:Chunker|ai:AUTO|ai:DISABLE|ai:Error result = resolveChunker(ai:AUTO, FIXED_SIZE);
    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        test:assertTrue(result.message().includes("double-chunk"), result.message());
    }

    ai:Chunker|ai:AUTO|ai:DISABLE|ai:Error withRealChunker = resolveChunker(new ai:GenericRecursiveChunker(), HIERARCHICAL);
    test:assertTrue(withRealChunker is ai:Error);
}

@test:Config {}
function testExplicitDisableIsAlwaysAcceptedRegardlessOfStrategy() returns error? {
    ai:Chunker|ai:AUTO|ai:DISABLE|ai:Error onFixed = resolveChunker(ai:DISABLE, FIXED_SIZE);
    test:assertTrue(onFixed is ai:DISABLE);
    ai:Chunker|ai:AUTO|ai:DISABLE|ai:Error onNone = resolveChunker(ai:DISABLE, NONE);
    test:assertTrue(onNone is ai:DISABLE);
}

@test:Config {}
function testExplicitChunkerIsAcceptedWhenTheStrategyIsNone() returns error? {
    ai:Chunker chunker = new ai:GenericRecursiveChunker();
    ai:Chunker|ai:AUTO|ai:DISABLE|ai:Error result = resolveChunker(chunker, NONE);
    test:assertTrue(result is ai:Chunker);
}

// ---- data-source effective type: MANAGED_KNOWLEDGE_BASE_CONNECTOR unwrap ----

@test:Config {}
function testPlainCustomDataSourceTypePassesThrough() {
    map<json> dataSource = {dataSourceConfiguration: {'type: "CUSTOM"}};
    test:assertEquals(effectiveDataSourceType(dataSource), "CUSTOM");
}

@test:Config {}
function testManagedConnectorWithStringEncodedParametersUnwraps() {
    // Probe-measured: the live response returns connectorParameters as a
    // JSON-ENCODED STRING, not an object, despite the service model declaring it a
    // free-form Document on both directions.
    map<json> dataSource = {
        dataSourceConfiguration: {
            'type: "MANAGED_KNOWLEDGE_BASE_CONNECTOR",
            managedKnowledgeBaseConnectorConfiguration: {
                connectorParameters: "{\"type\":\"CUSTOM\",\"aclEnabled\":false}"
            }
        }
    };
    test:assertEquals(effectiveDataSourceType(dataSource), "CUSTOM");
}

@test:Config {}
function testManagedConnectorWithObjectParametersAlsoUnwraps() {
    // Defensive: handle the object shape too, in case AWS fixes the asymmetry.
    map<json> dataSource = {
        dataSourceConfiguration: {
            'type: "MANAGED_KNOWLEDGE_BASE_CONNECTOR",
            managedKnowledgeBaseConnectorConfiguration: {
                connectorParameters: {'type: "S3"}
            }
        }
    };
    test:assertEquals(effectiveDataSourceType(dataSource), "S3");
}

@test:Config {}
function testSharePointDataSourceIsNotCustomOrS3() {
    map<json> dataSource = {dataSourceConfiguration: {'type: "SHAREPOINT"}};
    string effectiveType = effectiveDataSourceType(dataSource);
    test:assertNotEquals(effectiveType, "CUSTOM");
    test:assertNotEquals(effectiveType, "S3");
}

// ---- CreateDataSource body ----
//
// These replace the former `chunkingConfigurationJson` tests. That function built a
// `chunkingConfiguration` the managed-KB API rejects outright — 400 "A chunking
// strategy cannot be specified with a managed embedding model" for NONE, FIXED_SIZE
// and SEMANTIC alike. It was unreachable
// correctness, so it and its tests are gone; what matters now is the two things the
// live API actually demands.

@test:Config {}
function testCreateDataSourceBodyCarriesConnectorVersion() returns error? {
    map<json> body = check createDataSourceRequestBody({name: "ds"});
    map<json> connector = <map<json>>(<map<json>>(<map<json>>body["dataSourceConfiguration"])
        ["managedKnowledgeBaseConnectorConfiguration"])["connectorParameters"];
    // Omitting `version` returns 400 "The 'version' field is required in connector
    // parameters." — measured, and documented as "a required version field set to 1".
    test:assertEquals(connector["type"], "CUSTOM");
    test:assertEquals(connector["version"], "1");
}

@test:Config {}
function testCreateDataSourceBodyOmitsVectorIngestionConfiguration() returns error? {
    map<json> body = check createDataSourceRequestBody({name: "ds"});
    // A managed embedding model rejects ANY chunkingConfiguration, and SMART_PARSING
    // is both the only supported parsing strategy and the default — so the whole
    // field must be absent, not empty.
    test:assertFalse(body.hasKey("vectorIngestionConfiguration"),
        "vectorIngestionConfiguration must not be sent on a managed knowledge base");
}

@test:Config {}
function testCreateDataSourceBodyCarriesDescriptionOnlyWhenSet() returns error? {
    map<json> without = check createDataSourceRequestBody({name: "ds"});
    test:assertFalse(without.hasKey("description"));
    map<json> with = check createDataSourceRequestBody({name: "ds", description: "notes"});
    test:assertEquals(with["description"], "notes");
}
