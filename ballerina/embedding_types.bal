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

// Embedding types; the public class is split by vendor.
// Embeddings are InvokeModel-ONLY — there is no Converse equivalent
// and no streaming, so the model provider's routing ladder collapses entirely.
// Credentials, transport, SigV4, retry, and error mapping are reused unchanged.

# Well-known Amazon Titan text-embedding model ids.
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
public enum TitanEmbeddingModel {
    TITAN_EMBED_TEXT_V2 = "amazon.titan-embed-text-v2:0",
    TITAN_EMBED_TEXT_V1 = "amazon.titan-embed-text-v1"
}

# Well-known Cohere Embed model ids.
# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html
public enum CohereEmbeddingModel {
    COHERE_EMBED_ENGLISH_V3 = "cohere.embed-english-v3",
    COHERE_EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3",
    COHERE_EMBED_V4 = "cohere.embed-v4:0"
}

# Cohere's REQUIRED `input_type`. Getting this wrong silently degrades retrieval
# — there is no error, just worse results. Embed your corpus with
# `SEARCH_DOCUMENT` and your queries with `SEARCH_QUERY`.
public enum CohereInputType {
    SEARCH_DOCUMENT = "search_document",
    SEARCH_QUERY = "search_query",
    CLASSIFICATION = "classification",
    CLUSTERING = "clustering"
}

# Cohere's `truncate` behaviour for over-long inputs.
# Members are prefixed because a bare `NONE` would collide with `GuardrailAction`.
public enum Truncate {
    TRUNCATE_NONE = "NONE",
    TRUNCATE_START = "START",
    TRUNCATE_END = "END"
}

# Titan-specific embedding configuration.
public type TitanEmbeddingConfig record {|
    # Output vector size — Titan V2 accepts 256 | 512 | 1024.
    int dimensions?;
    # Whether to L2-normalize the returned vector (Titan only).
    boolean normalize?;
    # Escape hatch, mirroring the model provider's passthrough.
    AdditionalRequestFields additionalModelRequestFields?;
    # Use the FIPS 140-validated endpoint variant (`bedrock-runtime-fips.{region}...`),
    # resolved from AWS SDK endpoint metadata. Required for FedRAMP and GovCloud.
    # Changes only which HOST is dialled — never the SigV4 signing scope.
    boolean fips = false;
    # Retry policy.
    RetryConfig retryConfig?;
    # Underlying HTTP client configuration.
    http:ClientConfiguration httpConfig?;
|};

# Cohere-specific embedding configuration. `inputType` lives ONLY here, which is
# why the model provider's "inputType set on Titan → error" check is gone: with
# the vendor split it is unrepresentable.
public type CohereEmbeddingConfig record {|
    # REQUIRED on the wire. Defaults to `SEARCH_DOCUMENT` — the ingest path is the
    # higher-volume one. Construct a second provider with `SEARCH_QUERY` for the
    # query side.
    CohereInputType inputType = SEARCH_DOCUMENT;

    # Truncation behaviour for over-long inputs.
    Truncate truncate?;
    # Output vector size — Cohere Embed v4 accepts 256..1536.
    int dimensions?;
    # Escape hatch, mirroring the model provider's passthrough.
    AdditionalRequestFields additionalModelRequestFields?;
    # Use the FIPS 140-validated endpoint variant (`bedrock-runtime-fips.{region}...`),
    # resolved from AWS SDK endpoint metadata. Required for FedRAMP and GovCloud.
    # Changes only which HOST is dialled — never the SigV4 signing scope.
    boolean fips = false;
    # Retry policy.
    RetryConfig retryConfig?;
    # Underlying HTTP client configuration.
    http:ClientConfiguration httpConfig?;
|};

# Resolved embedding parameters, fixed at construction.
# Module-private: built at construction and consumed only by the internal converters.
type EmbeddingParams record {|
    # Output vector size.
    int dimensions?;
    # Titan only.
    boolean normalize?;
    # Cohere only — required on the wire.
    CohereInputType inputType?;
    # Cohere only.
    Truncate truncate?;
    # Escape-hatch passthrough, mirroring the model provider's passthrough.
    AdditionalRequestFields additionalModelRequestFields?;
|};

# What an embedding `decode` produces — NOT a bare vector.
# Cohere's response carries no token count at all, so `inputTokenCount` is
# optional and the observe-span call MUST be guarded. Module-private.
type DecodedEmbedding record {|
    # One embedding per input text, in input order.
    ai:Embedding[] embeddings;
    # Titan: `inputTextTokenCount`. Cohere: absent → `()`.
    int? inputTokenCount;
    # Cohere: `id`. Titan: absent → `()`.
    string? responseId;
|};

# Encodes a window of texts into a request body.
# Module-private converter plumbing.
type EncodeEmbedRequest isolated function (string[] texts, EmbeddingParams params)
    returns json|ai:Error;

# Decodes an embedding response. Module-private converter plumbing.
type DecodeEmbedResponse isolated function (json response) returns DecodedEmbedding|ai:Error;

# An embedding converter. `maxBatchSize` is the WIRE limit, not
# a tuning knob: Titan's `inputText` is a single string (1), Cohere's `texts` is
# an array of up to 96. Module-private converter registry record.
type EmbeddingConverter record {|
    # Texts per request the wire allows — Titan 1, Cohere 96. One window == one call.
    int maxBatchSize;
    # Texts → request body.
    EncodeEmbedRequest encode;
    # Wire JSON → `DecodedEmbedding`.
    DecodeEmbedResponse decode;
|};
