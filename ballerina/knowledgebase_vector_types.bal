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

// Public surface for `BedrockVectorKnowledgeBase` — the SELF-MANAGED knowledge base
// (`KnowledgeBaseConfiguration.type = VECTOR`), where the vector store belongs to the
// caller rather than to Bedrock. See knowledgebase_vector_common.bal for the
// resolution spine and knowledgebase_vector.bal for the public class.
//
// Deliberately kept separate from knowledgebase_types.bal: the two knowledge base
// types share no configuration record. MANAGED has no `storageConfiguration` and no
// `embeddingModelArn`; VECTOR requires both, has no `rerankingModelType` shortcut,
// and gains `overrideSearchType`.

// ============================================================================
// Field mappings.
//
// One record per backend, mirroring the service model 1:1 even where two backends
// currently agree. They are NOT interchangeable — `PineconeFieldMapping` and
// `NeptuneAnalyticsFieldMapping` have NO `vectorField`, and `RdsFieldMapping` adds
// `primaryKeyField` plus an optional `customMetadataField`. A single shared record
// would either reject valid Pinecone/Neptune configurations or send fields AWS
// rejects. Required members below are the `required` arrays in the `bedrock-agent`
// service model (botocore `service-2.json`), cross-checked against each type's own
// API reference page.
// ============================================================================

# Field mapping for an Amazon OpenSearch vector index — Serverless and Managed
# Cluster declare identical shapes.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_OpenSearchServerlessFieldMapping.html
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_OpenSearchManagedClusterFieldMapping.html
public type OpenSearchFieldMapping record {|
    # Field holding the vector embeddings.
    string vectorField;
    # Field holding the raw text chunk.
    string textField;
    # Field holding the metadata Bedrock manages.
    string metadataField;
|};

# Field mapping for a Pinecone index. Note there is NO `vectorField` — Pinecone
# stores the vector natively rather than in a named field.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_PineconeFieldMapping.html
public type PineconeFieldMapping record {|
    # Field holding the raw text chunk.
    string textField;
    # Field holding the metadata Bedrock manages.
    string metadataField;
|};

# Field mapping for a Neptune Analytics graph. Like Pinecone, NO `vectorField`.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_NeptuneAnalyticsFieldMapping.html
public type NeptuneAnalyticsFieldMapping record {|
    # Field holding the raw text chunk.
    string textField;
    # Field holding the metadata Bedrock manages.
    string metadataField;
|};

# Field mapping for a MongoDB Atlas collection.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_MongoDbAtlasFieldMapping.html
public type MongoDbAtlasFieldMapping record {|
    # Field holding the vector embeddings.
    string vectorField;
    # Field holding the raw text chunk.
    string textField;
    # Field holding the metadata Bedrock manages.
    string metadataField;
|};

# Field mapping for a Redis Enterprise Cloud index.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_RedisEnterpriseCloudFieldMapping.html
public type RedisEnterpriseCloudFieldMapping record {|
    # Field holding the vector embeddings.
    string vectorField;
    # Field holding the raw text chunk.
    string textField;
    # Field holding the metadata Bedrock manages.
    string metadataField;
|};

# Column mapping for an Amazon Aurora/RDS table. These are COLUMN names, and this
# is the only backend with a primary key and an optional custom-metadata column.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_RdsFieldMapping.html
public type RdsFieldMapping record {|
    # Primary key column.
    string primaryKeyField;
    # Column holding the vector embeddings.
    string vectorField;
    # Column holding the raw text chunk.
    string textField;
    # Column holding the metadata Bedrock manages.
    string metadataField;
    # Column holding YOUR metadata attributes, as a single `jsonb` value. Without it
    # you must add one typed column per metadata attribute instead. AWS requires a
    # GIN index on this column for metadata filtering to work.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-setup.html
    string customMetadataField?;
|};

// ============================================================================
// Storage configurations — one per vector store backend.
//
// A closed tagged union discriminated by the singleton `'type` field, so
// `storageConfigurationJson` is an exhaustive match the compiler checks.
// `StorageConfiguration.type` is `Required: Yes` in the API reference even though
// the user-guide example omits it, so it is always emitted.
// https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_StorageConfiguration.html
//
// NOTE ON PROVISIONING: none of these can be created by this module. The Bedrock
// API accepts only a storage configuration naming an ALREADY EXISTING store — the
// console's "Quick create a new vector store" is console-only ("If you prefer to
// let Amazon Bedrock create and manage a vector store for you, use the console",
// https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html).
// ============================================================================

# Amazon OpenSearch Serverless. The only backend Bedrock supports `startsWith`
# filters on, and (with a filterable text field) the one every AWS source agrees
# supports `SEARCH_HYBRID`. Note this module cannot emit `startsWith` regardless:
# `ai:MetadataFilterOperator` has no operator that means "starts with".
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_OpenSearchServerlessConfiguration.html
public type OpenSearchServerlessStorage record {|
    # Discriminator. Always emitted on the wire.
    readonly "OPENSEARCH_SERVERLESS" 'type = "OPENSEARCH_SERVERLESS";
    # ARN of the vector search collection.
    string collectionArn;
    # Name of the vector index inside the collection. AWS requires the `faiss`
    # engine — with `nmslib`, metadata filtering does not work and the index must be
    # rebuilt.
    string vectorIndexName;
    # Names of the fields Bedrock reads and writes.
    OpenSearchFieldMapping fieldMapping;
|};

# Amazon OpenSearch Service Managed Cluster. AWS requires a PUBLIC-access domain —
# domains behind a VPC are not supported for knowledge bases — and engine 2.13+ for
# a k-NN index (2.16+ for binary vectors).
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_OpenSearchManagedClusterConfiguration.html
public type OpenSearchManagedClusterStorage record {|
    # Discriminator. Always emitted on the wire.
    readonly "OPENSEARCH_MANAGED_CLUSTER" 'type = "OPENSEARCH_MANAGED_CLUSTER";
    # Endpoint of the OpenSearch domain.
    string domainEndpoint;
    # ARN of the OpenSearch domain.
    string domainArn;
    # Name of the vector index inside the domain.
    string vectorIndexName;
    # Names of the fields Bedrock reads and writes.
    OpenSearchFieldMapping fieldMapping;
|};

# Amazon S3 Vectors — the cheapest backend, at the cost of three limits AWS
# documents: SEMANTIC search only (no `SEARCH_HYBRID`), floating-point vectors only
# (no binary), and at most 1 KB of custom metadata across 35 keys per vector, which
# hierarchical chunking can exceed and fail ingestion. `startsWith` and
# `stringContains` filters are not supported.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_S3VectorsConfiguration.html
# https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-bedrock-kb.html
public type S3VectorsStorage record {|
    # Discriminator. Always emitted on the wire.
    readonly "S3_VECTORS" 'type = "S3_VECTORS";
    # ARN of the S3 vector bucket. Pair with `indexName`.
    string vectorBucketArn?;
    # ARN of the vector index. An alternative to `vectorBucketArn` + `indexName`.
    string indexArn?;
    # Name of the vector index inside the bucket. Pair with `vectorBucketArn`.
    string indexName?;
|};

# Amazon Aurora PostgreSQL (RDS). The cluster must live in the SAME AWS account as
# the knowledge base. If you filter on metadata, AWS recommends enabling HNSW
# iterative index scans (pgvector 0.8.0+) — without them, selective filters
# silently return fewer results than they should rather than erroring.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_RdsConfiguration.html
public type RdsStorage record {|
    # Discriminator. Always emitted on the wire.
    readonly "RDS" 'type = "RDS";
    # ARN of the Aurora DB cluster.
    string resourceArn;
    # ARN of the Secrets Manager secret holding the database credentials.
    string credentialsSecretArn;
    # Database name.
    string databaseName;
    # Table holding the vectors.
    string tableName;
    # Names of the COLUMNS Bedrock reads and writes.
    RdsFieldMapping fieldMapping;
|};

# Amazon Neptune Analytics (GraphRAG). The vector search index can only be created
# when the graph is created, and its dimension must match the embedding model.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_NeptuneAnalyticsConfiguration.html
public type NeptuneAnalyticsStorage record {|
    # Discriminator. Always emitted on the wire.
    readonly "NEPTUNE_ANALYTICS" 'type = "NEPTUNE_ANALYTICS";
    # ARN of the Neptune Analytics graph.
    string graphArn;
    # Names of the fields Bedrock reads and writes.
    NeptuneAnalyticsFieldMapping fieldMapping;
|};

# Pinecone. Using it means authorizing AWS to access a third-party service on your
# behalf.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_PineconeConfiguration.html
public type PineconeStorage record {|
    # Discriminator. Always emitted on the wire.
    readonly "PINECONE" 'type = "PINECONE";
    # Endpoint URL of the index management page.
    string connectionString;
    # ARN of the Secrets Manager secret holding the Pinecone API key, under the key
    # `apiKey`.
    string credentialsSecretArn;
    # Namespace new data is written to. Unset writes to the default namespace.
    string namespace?;
    # Names of the fields Bedrock reads and writes.
    PineconeFieldMapping fieldMapping;
|};

# Redis Enterprise Cloud. TLS must be enabled, and the Secrets Manager secret needs
# five specific keys: `username`, `password`, `serverCertificate`,
# `clientPrivateKey`, `clientCertificate`.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_RedisEnterpriseCloudConfiguration.html
public type RedisEnterpriseCloudStorage record {|
    # Discriminator. Always emitted on the wire.
    readonly "REDIS_ENTERPRISE_CLOUD" 'type = "REDIS_ENTERPRISE_CLOUD";
    # Public endpoint URL of the database.
    string endpoint;
    # Name of the vector index.
    string vectorIndexName;
    # ARN of the Secrets Manager secret holding the credentials and certificates.
    string credentialsSecretArn;
    # Names of the fields Bedrock reads and writes.
    RedisEnterpriseCloudFieldMapping fieldMapping;
|};

# MongoDB Atlas. **Metadata filtering does not work by default** — AWS requires
# filters to be configured explicitly in the Atlas vector index first.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_MongoDbAtlasConfiguration.html
public type MongoDbAtlasStorage record {|
    # Discriminator. Always emitted on the wire.
    readonly "MONGO_DB_ATLAS" 'type = "MONGO_DB_ATLAS";
    # Endpoint URL of the Atlas cluster.
    string endpoint;
    # Database name in the cluster.
    string databaseName;
    # Collection name in the database.
    string collectionName;
    # Name of the Atlas vector search index.
    string vectorIndexName;
    # ARN of the Secrets Manager secret holding `username` and `password`.
    string credentialsSecretArn;
    # Name of the VPC endpoint service connected to the cluster, when reaching Atlas
    # over AWS PrivateLink.
    string endpointServiceName?;
    # Name of the Atlas text search index. REQUIRED for `SEARCH_HYBRID` on this
    # backend.
    string textIndexName?;
    # Names of the fields Bedrock reads and writes.
    MongoDbAtlasFieldMapping fieldMapping;
|};

# The customer-provisioned vector store backing a self-managed knowledge base.
# Every member must already exist — this module never creates one.
public type StorageConfiguration OpenSearchServerlessStorage|OpenSearchManagedClusterStorage|
    S3VectorsStorage|RdsStorage|NeptuneAnalyticsStorage|PineconeStorage|
    RedisEnterpriseCloudStorage|MongoDbAtlasStorage;

// ============================================================================
// Embedding model, search type, reranking.
// ============================================================================

# Vector data type for the embedding model.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_BedrockEmbeddingModelConfiguration.html
public enum EmbeddingDataType {
    # Floating-point vectors. The default, and the only type S3 Vectors supports.
    EMBEDDING_FLOAT32 = "FLOAT32",
    # Binary vectors — cheaper and less precise. Supported ONLY on OpenSearch
    # Serverless and OpenSearch Managed Cluster ("the only vector stores that
    # support storing binary vectors"), and Managed Cluster additionally needs
    # engine version 2.16 or later.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-setup.html
    EMBEDDING_BINARY = "BINARY"
}

# Tuning for the embedding model on a self-managed knowledge base.
#
# `dimensions` MUST match the dimension the vector index was created with —
# Bedrock does not reconcile them, and a mismatch fails ingestion with an opaque
# error. This module cannot check it: verifying would mean querying your vector
# store, which needs credentials and network reach the calling application does not
# have (those permissions belong to the knowledge base's own service role).
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_BedrockEmbeddingModelConfiguration.html
public type VectorEmbeddingModelConfig record {|
    # Vector dimensions (0-4096). Must equal the vector index's dimension.
    int dimensions?;
    # Vector data type. Read `EmbeddingDataType` before setting `EMBEDDING_BINARY`.
    EmbeddingDataType embeddingDataType?;
|};

# Search strategy override for `retrieve()`.
#
# LEAVE IT UNSET unless you know your backend supports the value. Unset means
# Bedrock picks a strategy suited to the store, which is correct everywhere.
#
# ## Two AWS sources disagree on where `SEARCH_HYBRID` works
#
# - The API reference names ONLY OpenSearch Serverless with a filterable text field.
#   https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_KnowledgeBaseVectorSearchConfiguration.html
# - The user guide names three: "Hybrid search is only supported for Amazon RDS,
#   Amazon OpenSearch Serverless, and MongoDB vector stores that contain a
#   filterable text field."
#   https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-config.html
#
# Both agree it is unavailable on S3 Vectors, Neptune Analytics, Pinecone, and
# Redis. Neither is treated as authoritative here, which is why nothing is defaulted.
public enum SearchType {
    # Combine vector embeddings with raw-text search. Backend-dependent, see above.
    SEARCH_HYBRID = "HYBRID",
    # Vector embeddings only. Available on every backend.
    SEARCH_SEMANTIC = "SEMANTIC"
}

# Reranking for `retrieve()` on a self-managed knowledge base.
#
# Unlike `BedrockManagedKnowledgeBase`, which selects a reranker with the single
# `RerankingModelType` enum, `vectorSearchConfiguration` has no such shortcut — its
# only reranking member is the full `rerankingConfiguration` object. This record is
# therefore the ONLY way to rerank on this class.
#
# Reranking applies its own relevance cut, so it can return FEWER results than
# `numberOfResults` asked for.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_VectorSearchBedrockRerankingConfiguration.html
public type VectorRerankingConfig record {|
    # ARN of the Bedrock reranker model.
    string modelArn;
    # How many results to return after reranking (1-100). Unset leaves it to Bedrock.
    int numberOfRerankedResults?;
|};

// ============================================================================
// Definition and configuration.
// ============================================================================

# The `CUSTOM` direct-ingestion data source created alongside a self-managed
# knowledge base.
#
# Separate from `DataSourceDefinition` (the managed one) because chunking diverges:
# a managed knowledge base REJECTS `chunkingConfiguration` outright on a
# service-managed embedding model, so that record carries no chunking fields. A
# self-managed knowledge base always supplies its own embedding model, so chunking
# IS configurable here — and `NONE` is what makes a client-side `ai:Chunker` usable.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_ChunkingConfiguration.html
public type VectorDataSourceDefinition record {|
    # Data source name.
    string name;
    # Data source description.
    string description?;
    # How Bedrock chunks documents submitted through this data source. FIXED for the
    # life of the data source — `chunkingConfiguration` cannot be changed after
    # `CreateDataSource`. Set `NONE` to chunk client-side with an `ai:Chunker`.
    #
    # Only `FIXED_SIZE` and `NONE` are accepted here. `HIERARCHICAL` and `SEMANTIC`
    # are construction errors: each needs a tuning sub-object with required members
    # and no service-side default, which this record cannot express. To use one,
    # create the data source in the AWS console and attach by knowledge base id.
    ChunkingStrategy chunkingStrategy = FIXED_SIZE;
    # `FIXED_SIZE` tuning: approximate tokens per chunk. Must be **1-8192**. Ignored
    # on other strategies.
    int maxTokens = 300;
    # `FIXED_SIZE` tuning: percentage overlap between adjacent chunks. Must be
    # **1-99** — Bedrock rejects 0, so there is no "no overlap" value here; use
    # `chunkingStrategy = NONE` if you do not want Bedrock to chunk at all. Ignored on
    # other strategies.
    int overlapPercentage = 20;
|};

# A self-managed knowledge base to find-or-create by name, with all content flowing
# through this module.
#
# **The vector store named by `storageConfiguration` must already exist.** The
# Bedrock API has no equivalent of the console's "Quick create a new vector store";
# it accepts only a configuration naming an existing collection, cluster, table, or
# bucket. Provision it with Terraform/CDK/the console first, then pass its ARNs here.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_CreateKnowledgeBase.html
public type VectorKnowledgeBaseDefinition record {|
    # Knowledge base name. Must match `([0-9a-zA-Z][_-]?){1,100}` — dots are
    # rejected. Also the find-or-create lookup key.
    string name;
    # IAM role Bedrock assumes to manage the knowledge base. It needs permissions on
    # the vector store as well as on Bedrock — `aoss:APIAccessAll`, `es:ESHttp*`,
    # `rds-data:*`, `neptune-graph:*`, `s3vectors:*`, or
    # `secretsmanager:GetSecretValue`, depending on the backend. Console-created
    # roles live under the `/service-role/` path.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/kb-permissions.html
    string roleArn;
    # Knowledge base description.
    string description?;
    # ARN of the embedding model. REQUIRED — unlike a managed knowledge base, there
    # is no service-managed embedding model on this path. The role above must hold
    # `bedrock:InvokeModel` on it.
    string embeddingModelArn;
    # Embedding model tuning. Leave unset for the model's own defaults.
    VectorEmbeddingModelConfig embeddingModel?;
    # The customer-provisioned vector store. Must already exist.
    StorageConfiguration storageConfiguration;
    # The `CUSTOM` direct-ingestion data source created alongside the knowledge base.
    VectorDataSourceDefinition dataSource = {name: "ballerina-custom-source"};
    # How long `init` waits for the knowledge base and data source to leave their
    # transient `CREATING` states (~83s measured for a managed knowledge base; a
    # self-managed one also has to attach to your store).
    decimal readyTimeout = 300;
|};

# Configuration for `BedrockVectorKnowledgeBase`.
public type VectorKnowledgeBaseConfig record {|
    # The `CUSTOM` data source to ingest into / delete from. Resolved automatically
    # when omitted — which requires exactly one `CUSTOM` data source on the knowledge
    # base; construction errors naming the candidates when there are none or several.
    string dataSourceId?;
    # Client-side chunking before `ingest()`. Leave unset (the default) to DETECT it
    # from the resolved data source's actual `chunkingStrategy`: `ai:DISABLE` when
    # Bedrock chunks server-side (every strategy but `NONE`), `ai:AUTO` when it is
    # `NONE`. Passing an explicit `ai:Chunker` against a server-chunking data source
    # is a construction error — it would double-chunk, silently corrupting the
    # boundaries a chunker just computed.
    ai:Chunker|ai:AUTO|ai:DISABLE chunker?;
    # How long `ingest()` polls for submitted documents to leave their transient
    # states (`PENDING`/`STARTING`/`IN_PROGRESS`) and reach a terminal one.
    decimal ingestTimeout = 300;
    # Default `numberOfResults` for `retrieve()` (1-100), used when `maxLimit` does
    # not already imply a smaller cap. Bedrock's own default is 5.
    int numberOfResults?;
    # Search strategy override. Leave unset — read `SearchType` first, it is
    # backend-dependent and two AWS sources disagree on where `SEARCH_HYBRID` works.
    SearchType overrideSearchType?;
    # Reranking for `retrieve()`. Unset applies no reranking.
    VectorRerankingConfig rerankingConfiguration?;
    # Underlying HTTP client configuration, shared by both agent-plane clients.
    http:ClientConfiguration httpConfig?;
    # Retry policy, shared by both agent-plane clients.
    RetryConfig retryConfig?;
    # Use the FIPS 140-validated endpoint variant. Ignored when `serviceUrl` is a
    # concrete URL.
    boolean fips = false;
|};

// `implicitFilterConfiguration` is deliberately NOT exposed. It is a real member of
// `vectorSearchConfiguration` (a model generates a metadata filter from the user's
// prompt), but it requires a `modelArn` plus a 1-25 entry `MetadataAttributeSchema`
// array describing every filterable attribute, it has no counterpart anywhere in the
// `ai` module's contract, and how it composes with an explicit `filter` is not
// documented. Adding it later is additive — a new optional field on
// `VectorKnowledgeBaseConfig` — so nothing here forecloses it.
// https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_ImplicitFilterConfiguration.html
