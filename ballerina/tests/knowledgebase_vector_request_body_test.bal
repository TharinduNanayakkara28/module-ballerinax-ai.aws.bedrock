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

// Golden-file tests for the PURE request-body builders in
// knowledgebase_vector_common.bal. No listener, no transport: these assert the exact
// JSON the module would put on the wire, so every shape is checkable without AWS.
//
// Required members per backend come from each type's own API reference page,
// cross-checked against the `required` arrays in the `bedrock-agent` service model
// (botocore `service-2.json`).

final OpenSearchFieldMapping OSS_FIELDS = {
    vectorField: "embeddings",
    textField: "AMAZON_BEDROCK_TEXT_CHUNK",
    metadataField: "AMAZON_BEDROCK_METADATA"
};

// ---- storageConfigurationJson: one case per backend ----

@test:Config {}
function testStorageConfigOpenSearchServerless() {
    OpenSearchServerlessStorage storage = {
        collectionArn: "arn:aws:aoss:us-east-1:123456789012:collection/abcdefghij1234567890",
        vectorIndexName: "bedrock-index",
        fieldMapping: OSS_FIELDS
    };
    json expected = {
        'type: "OPENSEARCH_SERVERLESS",
        opensearchServerlessConfiguration: {
            collectionArn: "arn:aws:aoss:us-east-1:123456789012:collection/abcdefghij1234567890",
            vectorIndexName: "bedrock-index",
            fieldMapping: {
                vectorField: "embeddings",
                textField: "AMAZON_BEDROCK_TEXT_CHUNK",
                metadataField: "AMAZON_BEDROCK_METADATA"
            }
        }
    };
    test:assertEquals(storageConfigurationJson(storage), expected);
}

@test:Config {}
function testStorageConfigOpenSearchManagedCluster() {
    OpenSearchManagedClusterStorage storage = {
        domainEndpoint: "https://search-mydomain.us-east-1.es.amazonaws.com",
        domainArn: "arn:aws:es:us-east-1:123456789012:domain/mydomain",
        vectorIndexName: "bedrock-index",
        fieldMapping: OSS_FIELDS
    };
    json expected = {
        'type: "OPENSEARCH_MANAGED_CLUSTER",
        opensearchManagedClusterConfiguration: {
            domainEndpoint: "https://search-mydomain.us-east-1.es.amazonaws.com",
            domainArn: "arn:aws:es:us-east-1:123456789012:domain/mydomain",
            vectorIndexName: "bedrock-index",
            fieldMapping: {
                vectorField: "embeddings",
                textField: "AMAZON_BEDROCK_TEXT_CHUNK",
                metadataField: "AMAZON_BEDROCK_METADATA"
            }
        }
    };
    test:assertEquals(storageConfigurationJson(storage), expected);
}

@test:Config {}
function testStorageConfigS3VectorsWithBucketAndIndexName() {
    S3VectorsStorage storage = {
        vectorBucketArn: "arn:aws:s3vectors:us-east-1:123456789012:bucket/my-vectors",
        indexName: "bedrock-index"
    };
    json expected = {
        'type: "S3_VECTORS",
        s3VectorsConfiguration: {
            vectorBucketArn: "arn:aws:s3vectors:us-east-1:123456789012:bucket/my-vectors",
            indexName: "bedrock-index"
        }
    };
    test:assertEquals(storageConfigurationJson(storage), expected);
}

@test:Config {}
function testStorageConfigS3VectorsWithIndexArnOnly() {
    S3VectorsStorage storage = {
        indexArn: "arn:aws:s3vectors:us-east-1:123456789012:bucket/my-vectors/index/bedrock-index"
    };
    json expected = {
        'type: "S3_VECTORS",
        s3VectorsConfiguration: {
            indexArn: "arn:aws:s3vectors:us-east-1:123456789012:bucket/my-vectors/index/bedrock-index"
        }
    };
    test:assertEquals(storageConfigurationJson(storage), expected);
}

@test:Config {}
function testStorageConfigRdsIncludesPrimaryKeyAndOmitsUnsetCustomMetadata() {
    RdsStorage storage = {
        resourceArn: "arn:aws:rds:us-east-1:123456789012:cluster:pgvector-1",
        credentialsSecretArn: "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds-creds",
        databaseName: "bedrock_db",
        tableName: "bedrock_integration.bedrock_kb",
        fieldMapping: {
            primaryKeyField: "id",
            vectorField: "embedding",
            textField: "chunks",
            metadataField: "metadata"
        }
    };
    json expected = {
        'type: "RDS",
        rdsConfiguration: {
            resourceArn: "arn:aws:rds:us-east-1:123456789012:cluster:pgvector-1",
            credentialsSecretArn: "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds-creds",
            databaseName: "bedrock_db",
            tableName: "bedrock_integration.bedrock_kb",
            fieldMapping: {
                primaryKeyField: "id",
                vectorField: "embedding",
                textField: "chunks",
                metadataField: "metadata"
            }
        }
    };
    test:assertEquals(storageConfigurationJson(storage), expected);
}

@test:Config {}
function testStorageConfigRdsEmitsCustomMetadataFieldWhenSet() {
    RdsStorage storage = {
        resourceArn: "arn:aws:rds:us-east-1:123456789012:cluster:pgvector-1",
        credentialsSecretArn: "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds-creds",
        databaseName: "bedrock_db",
        tableName: "bedrock_kb",
        fieldMapping: {
            primaryKeyField: "id",
            vectorField: "embedding",
            textField: "chunks",
            metadataField: "metadata",
            customMetadataField: "custom_metadata"
        }
    };
    map<json> body = <map<json>>storageConfigurationJson(storage);
    map<json> rds = <map<json>>body["rdsConfiguration"];
    map<json> fieldMapping = <map<json>>rds["fieldMapping"];
    test:assertEquals(fieldMapping["customMetadataField"], "custom_metadata");
}

// Neptune Analytics and Pinecone have NO `vectorField` in their field mappings
// (unlike every other backend). Emitting one would be rejected.
@test:Config {}
function testStorageConfigNeptuneAnalyticsHasNoVectorField() {
    NeptuneAnalyticsStorage storage = {
        graphArn: "arn:aws:neptune-graph:us-east-1:123456789012:graph/g-abc123",
        fieldMapping: {textField: "text", metadataField: "metadata"}
    };
    json expected = {
        'type: "NEPTUNE_ANALYTICS",
        neptuneAnalyticsConfiguration: {
            graphArn: "arn:aws:neptune-graph:us-east-1:123456789012:graph/g-abc123",
            fieldMapping: {textField: "text", metadataField: "metadata"}
        }
    };
    test:assertEquals(storageConfigurationJson(storage), expected);

    // Assert against the ENCODER's output, not the literal above.
    map<json> encoded = <map<json>>storageConfigurationJson(storage);
    map<json> neptune = <map<json>>encoded["neptuneAnalyticsConfiguration"];
    test:assertFalse((<map<json>>neptune["fieldMapping"]).hasKey("vectorField"));
}

@test:Config {}
function testStorageConfigPineconeHasNoVectorFieldAndOmitsUnsetNamespace() {
    PineconeStorage storage = {
        connectionString: "https://my-index-abc123.svc.us-east-1.pinecone.io",
        credentialsSecretArn: "arn:aws:secretsmanager:us-east-1:123456789012:secret:pinecone",
        fieldMapping: {textField: "text", metadataField: "metadata"}
    };
    json expected = {
        'type: "PINECONE",
        pineconeConfiguration: {
            connectionString: "https://my-index-abc123.svc.us-east-1.pinecone.io",
            credentialsSecretArn: "arn:aws:secretsmanager:us-east-1:123456789012:secret:pinecone",
            fieldMapping: {textField: "text", metadataField: "metadata"}
        }
    };
    test:assertEquals(storageConfigurationJson(storage), expected);
}

@test:Config {}
function testStorageConfigPineconeEmitsNamespaceWhenSet() {
    PineconeStorage storage = {
        connectionString: "https://my-index-abc123.svc.us-east-1.pinecone.io",
        credentialsSecretArn: "arn:aws:secretsmanager:us-east-1:123456789012:secret:pinecone",
        namespace: "tenant-a",
        fieldMapping: {textField: "text", metadataField: "metadata"}
    };
    map<json> body = <map<json>>storageConfigurationJson(storage);
    test:assertEquals((<map<json>>body["pineconeConfiguration"])["namespace"], "tenant-a");
}

@test:Config {}
function testStorageConfigRedisEnterpriseCloud() {
    RedisEnterpriseCloudStorage storage = {
        endpoint: "https://redis-12345.c1.us-east-1-2.ec2.cloud.redislabs.com:12345",
        vectorIndexName: "bedrock-index",
        credentialsSecretArn: "arn:aws:secretsmanager:us-east-1:123456789012:secret:redis",
        fieldMapping: {vectorField: "vector", textField: "text", metadataField: "metadata"}
    };
    json expected = {
        'type: "REDIS_ENTERPRISE_CLOUD",
        redisEnterpriseCloudConfiguration: {
            endpoint: "https://redis-12345.c1.us-east-1-2.ec2.cloud.redislabs.com:12345",
            vectorIndexName: "bedrock-index",
            credentialsSecretArn: "arn:aws:secretsmanager:us-east-1:123456789012:secret:redis",
            fieldMapping: {vectorField: "vector", textField: "text", metadataField: "metadata"}
        }
    };
    test:assertEquals(storageConfigurationJson(storage), expected);
}

@test:Config {}
function testStorageConfigMongoDbAtlasOmitsUnsetOptionals() {
    MongoDbAtlasStorage storage = {
        endpoint: "https://cluster0.abcde.mongodb.net",
        databaseName: "bedrock",
        collectionName: "chunks",
        vectorIndexName: "vector_index",
        credentialsSecretArn: "arn:aws:secretsmanager:us-east-1:123456789012:secret:mongo",
        fieldMapping: {vectorField: "embedding", textField: "text", metadataField: "metadata"}
    };
    map<json> body = <map<json>>storageConfigurationJson(storage);
    map<json> mongo = <map<json>>body["mongoDbAtlasConfiguration"];
    test:assertFalse(mongo.hasKey("endpointServiceName"));
    test:assertFalse(mongo.hasKey("textIndexName"));
}

@test:Config {}
function testStorageConfigMongoDbAtlasEmitsOptionalsWhenSet() {
    MongoDbAtlasStorage storage = {
        endpoint: "https://cluster0.abcde.mongodb.net",
        databaseName: "bedrock",
        collectionName: "chunks",
        vectorIndexName: "vector_index",
        credentialsSecretArn: "arn:aws:secretsmanager:us-east-1:123456789012:secret:mongo",
        endpointServiceName: "com.amazonaws.vpce.us-east-1.vpce-svc-123",
        textIndexName: "text_index",
        fieldMapping: {vectorField: "embedding", textField: "text", metadataField: "metadata"}
    };
    map<json> mongo = <map<json>>(<map<json>>storageConfigurationJson(storage))["mongoDbAtlasConfiguration"];
    test:assertEquals(mongo["endpointServiceName"], "com.amazonaws.vpce.us-east-1.vpce-svc-123");
    test:assertEquals(mongo["textIndexName"], "text_index");
}

// `StorageConfiguration.type` is `Required: Yes` in the API reference even though the
// worked example in knowledge-base-create.html omits it. Assert it on ALL eight.
@test:Config {}
function testStorageConfigAlwaysEmitsType() {
    StorageConfiguration[] all = [
        {collectionArn: "arn:aws:aoss:us-east-1:123456789012:collection/c1", vectorIndexName: "i",
            fieldMapping: OSS_FIELDS},
        {domainEndpoint: "https://d.es.amazonaws.com", domainArn: "arn:aws:es:us-east-1:123456789012:domain/d",
            vectorIndexName: "i", fieldMapping: OSS_FIELDS},
        {indexArn: "arn:aws:s3vectors:us-east-1:123456789012:bucket/b/index/i"},
        {resourceArn: "arn:aws:rds:us-east-1:123456789012:cluster:c", credentialsSecretArn: "arn:s",
            databaseName: "d", tableName: "t",
            fieldMapping: {primaryKeyField: "id", vectorField: "v", textField: "t", metadataField: "m"}},
        {graphArn: "arn:aws:neptune-graph:us-east-1:123456789012:graph/g",
            fieldMapping: {textField: "t", metadataField: "m"}},
        {connectionString: "https://p.pinecone.io", credentialsSecretArn: "arn:s",
            fieldMapping: {textField: "t", metadataField: "m"}},
        {endpoint: "https://r.redislabs.com:1", vectorIndexName: "i", credentialsSecretArn: "arn:s",
            fieldMapping: {vectorField: "v", textField: "t", metadataField: "m"}},
        {endpoint: "https://m.mongodb.net", databaseName: "d", collectionName: "c", vectorIndexName: "i",
            credentialsSecretArn: "arn:s",
            fieldMapping: {vectorField: "v", textField: "t", metadataField: "m"}}
    ];
    string[] expectedTypes = [
        "OPENSEARCH_SERVERLESS", "OPENSEARCH_MANAGED_CLUSTER", "S3_VECTORS", "RDS",
        "NEPTUNE_ANALYTICS", "PINECONE", "REDIS_ENTERPRISE_CLOUD", "MONGO_DB_ATLAS"
    ];
    test:assertEquals(all.length(), expectedTypes.length());
    foreach int i in 0 ..< all.length() {
        map<json> body = <map<json>>storageConfigurationJson(all[i]);
        test:assertTrue(body.hasKey("type"), string `backend ${expectedTypes[i]} omitted 'type'`);
        test:assertEquals(body["type"], expectedTypes[i]);
    }
}

// ---- validateStorageConfiguration ----

@test:Config {}
function testS3VectorsWithoutAnyIndexIsAConstructionError() {
    S3VectorsStorage storage = {vectorBucketArn: "arn:aws:s3vectors:us-east-1:123456789012:bucket/b"};
    ai:Error? result = validateStorageConfiguration(storage);
    test:assertTrue(result is ai:Error);
    if result is ai:Error {
        test:assertTrue(result.message().includes("indexArn"), result.message());
        test:assertTrue(result.message().includes("indexName"), result.message());
    }
}

@test:Config {}
function testS3VectorsAcceptsEitherValidCombination() {
    S3VectorsStorage byArn = {indexArn: "arn:aws:s3vectors:us-east-1:123456789012:bucket/b/index/i"};
    S3VectorsStorage byBucketAndName = {
        vectorBucketArn: "arn:aws:s3vectors:us-east-1:123456789012:bucket/b",
        indexName: "i"
    };
    test:assertTrue(validateStorageConfiguration(byArn) is ());
    test:assertTrue(validateStorageConfiguration(byBucketAndName) is ());
}

// ---- createVectorKnowledgeBaseRequestBody ----

final OpenSearchServerlessStorage TEST_STORAGE = {
    collectionArn: "arn:aws:aoss:us-east-1:123456789012:collection/abcdefghij1234567890",
    vectorIndexName: "bedrock-index",
    fieldMapping: OSS_FIELDS
};

const string TEST_EMBEDDING_ARN = "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0";

@test:Config {}
function testCreateVectorKnowledgeBaseBodyShape() {
    VectorKnowledgeBaseDefinition def = {
        name: "my-vector-kb",
        roleArn: "arn:aws:iam::123456789012:role/service-role/AmazonBedrockExecutionRoleForKnowledgeBase_1",
        embeddingModelArn: TEST_EMBEDDING_ARN,
        storageConfiguration: TEST_STORAGE
    };
    map<json> body = createVectorKnowledgeBaseRequestBody(def);

    map<json> kbConfig = <map<json>>body["knowledgeBaseConfiguration"];
    test:assertEquals(kbConfig["type"], "VECTOR");
    test:assertFalse(kbConfig.hasKey("managedKnowledgeBaseConfiguration"));

    map<json> vectorConfig = <map<json>>kbConfig["vectorKnowledgeBaseConfiguration"];
    test:assertEquals(vectorConfig["embeddingModelArn"], TEST_EMBEDDING_ARN);
    // Unset embedding tuning must not produce an empty wrapper.
    test:assertFalse(vectorConfig.hasKey("embeddingModelConfiguration"));

    // `storageConfiguration` is a TOP-LEVEL sibling of `knowledgeBaseConfiguration`,
    // not nested inside it.
    test:assertTrue(body.hasKey("storageConfiguration"));
    test:assertFalse(kbConfig.hasKey("storageConfiguration"));
    test:assertEquals(body["storageConfiguration"], storageConfigurationJson(TEST_STORAGE));

    // Unset description is omitted rather than sent as null.
    test:assertFalse(body.hasKey("description"));
}

@test:Config {}
function testCreateVectorKnowledgeBaseBodyCarriesEmbeddingModelConfiguration() {
    VectorKnowledgeBaseDefinition def = {
        name: "my-vector-kb",
        roleArn: "arn:aws:iam::123456789012:role/service-role/kb",
        embeddingModelArn: TEST_EMBEDDING_ARN,
        embeddingModel: {dimensions: 1024, embeddingDataType: EMBEDDING_BINARY},
        storageConfiguration: TEST_STORAGE,
        description: "a description"
    };
    map<json> body = createVectorKnowledgeBaseRequestBody(def);
    map<json> vectorConfig =
        <map<json>>(<map<json>>body["knowledgeBaseConfiguration"])["vectorKnowledgeBaseConfiguration"];
    json expectedEmbedding = {
        bedrockEmbeddingModelConfiguration: {dimensions: 1024, embeddingDataType: "BINARY"}
    };
    test:assertEquals(vectorConfig["embeddingModelConfiguration"], expectedEmbedding);
    test:assertEquals(body["description"], "a description");
}

// ---- createVectorDataSourceRequestBody ----

// The divergence most likely to be broken by copying the managed path: a managed
// knowledge base needs the MANAGED_KNOWLEDGE_BASE_CONNECTOR wrapper, a self-managed
// one takes the plain form and REJECTS the wrapper.
@test:Config {}
function testCreateVectorDataSourceUsesPlainCustomTypeWithNoConnectorWrapper() returns error? {
    map<json> body = check createVectorDataSourceRequestBody({name: "ballerina-custom-source"});

    map<json> dsConfig = <map<json>>body["dataSourceConfiguration"];
    test:assertEquals(dsConfig["type"], "CUSTOM");
    test:assertEquals(dsConfig.length(), 1, "dataSourceConfiguration must carry ONLY 'type'");
    test:assertFalse(dsConfig.hasKey("managedKnowledgeBaseConnectorConfiguration"));

    string serialized = body.toJsonString();
    test:assertFalse(serialized.includes("MANAGED_KNOWLEDGE_BASE_CONNECTOR"), serialized);
    test:assertFalse(serialized.includes("connectorParameters"), serialized);
}

// Unlike the managed path — where Bedrock rejects chunkingConfiguration against a
// service-managed embedding model — a self-managed data source DOES take it.
@test:Config {}
function testCreateVectorDataSourceSendsChunkingConfiguration() returns error? {
    map<json> body = check createVectorDataSourceRequestBody({name: "ds"});
    map<json> ingestion = <map<json>>body["vectorIngestionConfiguration"];
    json expected = {
        chunkingStrategy: "FIXED_SIZE",
        fixedSizeChunkingConfiguration: {maxTokens: 300, overlapPercentage: 20}
    };
    test:assertEquals(ingestion["chunkingConfiguration"], expected);
}

// Only FIXED_SIZE and NONE are encodable — the other two are rejected at
// construction by `validateVectorDataSource`, so no wire shape is asserted for them.
@test:Config {}
function testVectorChunkingConfigurationPerStrategy() returns error? {
    json fixedSize = check vectorChunkingConfigurationJson({name: "d", chunkingStrategy: FIXED_SIZE, maxTokens: 512,
        overlapPercentage: 10});
    test:assertEquals(fixedSize, {
        chunkingStrategy: "FIXED_SIZE",
        fixedSizeChunkingConfiguration: {maxTokens: 512, overlapPercentage: 10}
    });
    test:assertEquals(check vectorChunkingConfigurationJson({name: "d", chunkingStrategy: NONE}),
        {chunkingStrategy: "NONE"});
}

// HIERARCHICAL and SEMANTIC each need a tuning sub-object with required members and
// no service-side default, which `VectorDataSourceDefinition` cannot express. Sending
// the bare strategy would be an incomplete body, so they are refused up front.
@test:Config {}
function testUnsupportedChunkingStrategiesAreRejected() {
    ChunkingStrategy[] unsupported = [HIERARCHICAL, SEMANTIC];
    foreach ChunkingStrategy strategy in unsupported {
        ai:Error? result = validateVectorDataSource({name: "d", chunkingStrategy: strategy});
        test:assertTrue(result is ai:Error, string `${strategy} should be rejected`);
        if result is ai:Error {
            test:assertTrue(result.message().includes(strategy.toString()), result.message());
            // The message must say what to do instead.
            test:assertTrue(result.message().includes("FIXED_SIZE"), result.message());
        }
    }
    test:assertTrue(validateVectorDataSource({name: "d", chunkingStrategy: FIXED_SIZE}) is ());
    test:assertTrue(validateVectorDataSource({name: "d", chunkingStrategy: NONE}) is ());
}

// The service model's own integer bounds. `overlapPercentage` is min 1, NOT 0 —
// "no overlap" is not expressible on FIXED_SIZE.
@test:Config {}
function testChunkingRangesAreValidated() {
    ai:Error? zeroOverlap = validateVectorDataSource({name: "d", overlapPercentage: 0});
    test:assertTrue(zeroOverlap is ai:Error);
    if zeroOverlap is ai:Error {
        test:assertTrue(zeroOverlap.message().includes("overlapPercentage"), zeroOverlap.message());
    }
    test:assertTrue(validateVectorDataSource({name: "d", overlapPercentage: 100}) is ai:Error);
    test:assertTrue(validateVectorDataSource({name: "d", maxTokens: 0}) is ai:Error);
    test:assertTrue(validateVectorDataSource({name: "d", maxTokens: 1, overlapPercentage: 1}) is ());
    test:assertTrue(validateVectorDataSource({name: "d", overlapPercentage: 99}) is ());
}

@test:Config {}
function testRetrievalConfigRangesAreValidated() {
    test:assertTrue(validateVectorRetrievalConfig({numberOfResults: 0}) is ai:Error);
    test:assertTrue(validateVectorRetrievalConfig({numberOfResults: 101}) is ai:Error);
    test:assertTrue(validateVectorRetrievalConfig({numberOfResults: 100}) is ());
    test:assertTrue(validateVectorRetrievalConfig({
        rerankingConfiguration: {modelArn: "arn:model", numberOfRerankedResults: 0}
    }) is ai:Error);
    test:assertTrue(validateVectorRetrievalConfig({
        rerankingConfiguration: {modelArn: "arn:model", numberOfRerankedResults: 100}
    }) is ());
    test:assertTrue(validateVectorRetrievalConfig({}) is ());
}

// ---- vectorSearchConfigJson / rerankingConfigJson ----

@test:Config {}
function testVectorSearchConfigOmitsUnsetMembers() {
    json config = vectorSearchConfigJson((), 10, (), ());
    test:assertEquals(config, {numberOfResults: 10});
    map<json> configMap = <map<json>>config;
    // A null filter must never reach the wire — `filter is json` would not reject it.
    test:assertFalse(configMap.hasKey("filter"));
    // Defaulting overrideSearchType would 400 or silently degrade on most backends.
    test:assertFalse(configMap.hasKey("overrideSearchType"));
    test:assertFalse(configMap.hasKey("rerankingConfiguration"));
}

@test:Config {}
function testVectorSearchConfigEmitsSetMembers() {
    json filter = {'equals: {key: "tenant", value: "acme"}};
    json config = vectorSearchConfigJson(filter, 25, SEARCH_HYBRID,
        {modelArn: "arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0"});
    json expected = {
        numberOfResults: 25,
        filter: {'equals: {key: "tenant", value: "acme"}},
        overrideSearchType: "HYBRID",
        rerankingConfiguration: {
            'type: "BEDROCK_RERANKING_MODEL",
            bedrockRerankingConfiguration: {
                modelConfiguration: {modelArn: "arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0"}
            }
        }
    };
    test:assertEquals(config, expected);
}

// `rerankingModelType` is the MANAGED branch's shortcut and is not a member of
// vectorSearchConfiguration — it must never appear here.
@test:Config {}
function testVectorSearchConfigNeverEmitsRerankingModelType() {
    json config = vectorSearchConfigJson((), 5, SEARCH_SEMANTIC,
        {modelArn: "arn:aws:bedrock:us-east-1::foundation-model/amazon.rerank-v1:0",
            numberOfRerankedResults: 3});
    test:assertFalse(config.toJsonString().includes("rerankingModelType"), config.toJsonString());
}

@test:Config {}
function testRerankingConfigEmitsNumberOfRerankedResultsWhenSet() {
    json config = rerankingConfigJson({modelArn: "arn:model", numberOfRerankedResults: 7});
    map<json> bedrockReranking =
        <map<json>>(<map<json>>config)["bedrockRerankingConfiguration"];
    test:assertEquals(bedrockReranking["numberOfRerankedResults"], 7);
    test:assertEquals((<map<json>>config)["type"], "BEDROCK_RERANKING_MODEL");
}

// ---- withVectorSourceUriFilter ----

// THE REGRESSION GUARD for the reserved-prefix split: self-managed knowledge bases
// use `x-amz-bedrock-kb-source-uri`, managed ones use `_source_uri`. Reusing the
// managed key here would produce a deleteByFilter that matches nothing and so
// deletes nothing, silently.
// Prefix rule: https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-config.html
// The exact key, under "Auto-created fields":
// https://docs.aws.amazon.com/bedrock/latest/userguide/kb-multimodal-test-and-query.html
@test:Config {}
function testVectorSourceUriFilterUsesTheXAmzBedrockKey() {
    json bare = withVectorSourceUriFilter((), "doc-1");
    test:assertEquals(bare, {'equals: {key: "x-amz-bedrock-kb-source-uri", value: "doc-1"}});
    test:assertFalse(bare.toJsonString().includes("_source_uri"), bare.toJsonString());
}

@test:Config {}
function testVectorSourceUriFilterAndsWithTheUserFilter() {
    json userFilter = {'equals: {key: "tenant", value: "acme"}};
    json combined = withVectorSourceUriFilter(userFilter, "doc-1");
    json expected = {
        andAll: [
            {'equals: {key: "tenant", value: "acme"}},
            {'equals: {key: "x-amz-bedrock-kb-source-uri", value: "doc-1"}}
        ]
    };
    test:assertEquals(combined, expected);
}

// The managed spelling must stay on the managed side — asserting both here means a
// future refactor that unifies them cannot silently flip either.
@test:Config {}
function testManagedAndVectorSourceUriKeysAreDistinct() {
    test:assertEquals(SOURCE_URI_METADATA_KEY, "_source_uri");
    test:assertEquals(VECTOR_SOURCE_URI_METADATA_KEY, "x-amz-bedrock-kb-source-uri");
    test:assertNotEquals(SOURCE_URI_METADATA_KEY, VECTOR_SOURCE_URI_METADATA_KEY);
}
