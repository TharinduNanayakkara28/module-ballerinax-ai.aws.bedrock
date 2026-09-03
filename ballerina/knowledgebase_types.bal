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
import ballerinax/aws.auth;

// Public surface for `BedrockManagedKnowledgeBase`: configuration, the
// find-or-create definition, and the chunking-strategy enum. See
// knowledgebase_common.bal for the resolution spine and knowledgebase_managed.bal
// for the public class.

# Credentials for the two Bedrock agent planes. SigV4 only — deliberately NOT
# `BedrockCredentials`, which also admits a `BearerToken`.
#
# Amazon Bedrock API keys "are limited to Amazon Bedrock and Amazon Bedrock Runtime
# actions" and explicitly cannot be used with "Agents for Amazon Bedrock or Agents
# for Amazon Bedrock Runtime API operations" — and both knowledge base planes
# (`bedrock-agent` for the control calls, `bedrock-agent-runtime` for `Retrieve`)
# are exactly those. Admitting a `BearerToken` here would compile and then fail at
# runtime with an opaque 403, so it is excluded at the type level instead.
# https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-use.html
#
# Widen this alias if AWS ever extends API keys to the Agents planes.
public type KnowledgeBaseCredentials auth:AuthConfig;

# How Bedrock splits ingested documents into retrievable chunks. Set on
# `CreateDataSource` and FIXED for the life of the data source — it cannot be
# changed afterward.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_ChunkingConfiguration.html
#
# ## UNRESOLVED: a managed knowledge base may not report its chunking back
#
# `GetDataSource` on a managed knowledge base's data source returns
# `vectorIngestionConfiguration` carrying ONLY `parsingConfiguration`
# (`{parsingStrategy: SMART_PARSING}`) — no `chunkingConfiguration` field at all,
# even though the console exposes a "Text chunking strategy" selector for the same
# data source (measured against the live API on 2026-08-14).
#
# The data source measured was created THROUGH THE CONSOLE with "Default chunking"
# selected, so its silence is ambiguous between two readings:
#
# 1. a managed knowledge base never stores or echoes `chunkingConfiguration`, so
#    the strategy can never be read back; or
# 2. it echoes only an EXPLICITLY set config, and that one never had one.
#
# Only creating a data source with an explicit `chunkingStrategy` and reading it
# back distinguishes them, and that has not been done. Until it is,
# `detectChunkingStrategy` treats an absent `chunkingConfiguration` as
# server-side chunking, which is the safe direction: it yields `ai:DISABLE` and
# risks UNDER-chunking a `NONE` data source, rather than `ai:AUTO` double-chunking
# a server-chunking one and destroying boundaries a chunker just computed.
#
# Consequence under reading 1: on a data source this module did not create,
# `NONE` is undetectable and an `ai:Chunker` must be passed explicitly.
public enum ChunkingStrategy {
    # Bedrock splits each submitted document into chunks of the approximate size set
    # by `maxTokens`/`overlapPercentage`. The default: safe out of the box, but a
    # client-supplied `ai:Chunker` against a data source using this strategy would
    # double-chunk — Bedrock re-splits whatever is submitted — so `chunker` defaults
    # to `ai:DISABLE` on it.
    FIXED_SIZE,
    # Two-layer chunking: large parent chunks, smaller child chunks derived from them.
    HIERARCHICAL,
    # Chunks by grouping semantically similar content (NLP-derived boundaries).
    SEMANTIC,
    # Bedrock treats each submitted document as exactly ONE chunk — the client owns
    # chunk boundaries. This is what makes an `ai:Chunker` meaningful: `chunker`
    # defaults to `ai:AUTO` on a data source using this strategy.
    NONE
}

# Reranking model selection for `retrieve()` on a managed knowledge base.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_ManagedSearchConfiguration.html
public enum RerankingModelType {
    # No reranking pass.
    RERANKING_NONE = "NONE",
    # Bedrock's own managed reranking model.
    RERANKING_MANAGED = "MANAGED"
}

# The `CUSTOM` direct-ingestion data source created alongside a knowledge base in
# the find-or-create (`KnowledgeBaseDefinition`) path. This is the data source
# `ingest()` writes into — see the module doc on why a `CUSTOM` source, specifically,
# is required.
public type DataSourceDefinition record {|
    # Data source name.
    string name;
    # Data source description.
    string description?;
|};

# A caller-supplied Bedrock embedding model for a knowledge base created through
# `KnowledgeBaseDefinition`, replacing Bedrock's service-managed one.
#
# Three consequences, all permanent — `embeddingModelType` CANNOT be changed after
# creation, so switching means building a new knowledge base:
#
# 1. **Metered separately.** The service-managed model is included at no additional
#    cost; a caller-supplied one bills per use.
# 2. **The managed reranker becomes unavailable.** `RERANKING_MANAGED` and this
#    record are mutually exclusive — setting both is a construction error.
# 3. **The knowledge base's `roleArn` must be able to invoke the model.** It needs
#    `bedrock:InvokeModel` on `embeddingModelArn`; without it `CreateKnowledgeBase`
#    fails with AWS's opaque "Unable to verify the specified embedding model".
#
# https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-create.html#kb-managed-embedding-models
public type ManagedEmbeddingModel record {|
    # Embedding model ARN. AWS supports Amazon Titan Text Embeddings V2, Cohere Embed
    # English v3, Cohere Embed Multilingual v3, Cohere Embed v4, and Amazon Nova
    # Multimodal Embeddings on a managed knowledge base.
    string embeddingModelArn;
    # Vector dimensions. AWS requires 1024 on a managed knowledge base.
    int dimensions = 1024;
    # Vector data type. AWS requires float32 on a managed knowledge base.
    string embeddingDataType = "FLOAT32";
|};

# A knowledge base to find-or-create by name, with all content flowing through this
# module. `CreateKnowledgeBase` has no upsert and knowledge base names
# are not unique per account, so `init` first searches `ListKnowledgeBases` for an
# exact name match: exactly one match → attach to it (no write); no match → create
# it; more than one match → a construction error naming the candidate ids.
public type KnowledgeBaseDefinition record {|
    # Knowledge base name. Must match `([0-9a-zA-Z][_-]?){1,100}` — dots are
    # rejected. Also the find-or-create lookup key.
    string name;
    # IAM role Bedrock assumes to manage the knowledge base. Console-created roles
    # live under the `/service-role/` path — a hand-written `roleArn` omitting that
    # segment will not match one created through the console.
    string roleArn;
    # Knowledge base description.
    string description?;
    # The `CUSTOM` direct-ingestion data source created alongside the knowledge base.
    DataSourceDefinition dataSource = {name: "ballerina-custom-source"};
    # Embedding model. Leave unset for Bedrock's service-managed model: no additional
    # cost, no IAM beyond the base role, and `RERANKING_MANAGED` stays available — but
    # chunking is then FIXED at 300 tokens / 20% overlap and cannot be configured
    # (measured: Bedrock rejects any `chunkingConfiguration` on a managed embedding
    # model), which is why an `ai:Chunker` is unusable on a knowledge base created
    # this way. Set it to choose your own model — read `ManagedEmbeddingModel` first,
    # every consequence is permanent.
    ManagedEmbeddingModel embeddingModel?;
    # Customer-managed KMS key for the managed vector store. Unset uses an AWS-owned
    # key.
    string kmsKeyArn?;
    # How long `init` waits for the knowledge base and data source to leave their
    # transient `CREATING` states (~83s measured for the knowledge base alone).
    decimal readyTimeout = 300;
|};

# Configuration for `BedrockManagedKnowledgeBase`.
public type ManagedKnowledgeBaseConfig record {|
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
    #
    # NOTE: detection is only as good as what `GetDataSource` reports, and a managed
    # knowledge base may not report `chunkingConfiguration` at all — see the
    # `ChunkingStrategy` doc comment. When it does not, detection yields `ai:DISABLE`
    # (server chunks); pass an explicit `ai:Chunker` if you know the data source was
    # created with `NONE`.
    ai:Chunker|ai:AUTO|ai:DISABLE chunker?;
    # How long `ingest()` polls for submitted documents to leave their transient
    # states (`PENDING`/`STARTING`/`IN_PROGRESS`) and reach a terminal one.
    # Ingestion is inherently slow: ~14s measured for a single small document, ~47s
    # for a 97KB one that split into 69 chunks.
    decimal ingestTimeout = 300;
    # Default `numberOfResults` for `retrieve()` (1-100), used when `maxLimit` does
    # not already imply a smaller cap.
    int numberOfResults?;
    # Reranking model for `retrieve()`. Unset leaves it to Bedrock's own default.
    # `RERANKING_MANAGED` applies its own relevance cut, so it can return FEWER
    # results than `numberOfResults` asked for (measured: 4 where `RERANKING_NONE`
    # returned 5).
    RerankingModelType rerankingModelType?;
    # Underlying HTTP client configuration, shared by both agent-plane clients.
    http:ClientConfiguration httpConfig?;
    # Retry policy, shared by both agent-plane clients.
    RetryConfig retryConfig?;
    # Use the FIPS 140-validated endpoint variant. Ignored when `serviceUrl` is a
    # concrete URL.
    boolean fips = false;
|};
