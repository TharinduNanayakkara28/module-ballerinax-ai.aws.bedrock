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

// Amazon Titan text embeddings.
// Request:  {"inputText": string, "dimensions": int, "normalize": bool}
// Response: {"embedding": [floats], "inputTextTokenCount": int}
// `inputText` is a STRING, not an array — Titan embeds exactly ONE text per call,
// which is why maxBatchSize is 1 and batchEmbed becomes n sequential calls.
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html

// Titan embeds exactly one text per InvokeModel call.
const int TITAN_MAX_BATCH = 1;

// Titan Embed V1 ids, which have NO `dimensions` parameter (V2 added it). Strips
// any CRIS geo prefix first, for the same reason as `usesCohereEmbedV4`.
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
isolated function isTitanEmbedV1(string modelId) returns boolean {
    [string, string?] [bareId, _] = normalizeModelId(modelId);
    return bareId.startsWith("amazon.titan-embed-text-v1");
}

// The Titan embedding converter.
final readonly & EmbeddingConverter TITAN_EMBED_CONVERTER = {
    maxBatchSize: TITAN_MAX_BATCH,
    encode: encodeTitanEmbed,
    decode: decodeTitanEmbed
};

// Encodes a Titan embedding request. `texts` must hold exactly one element —
// `inputText` is a single string on the wire.
isolated function encodeTitanEmbed(string[] texts, EmbeddingParams params) returns json|ai:Error {
    if texts.length() != 1 {
        return error ai:Error(string `Titan embeds exactly one text per call; got ${texts.length()}. ` +
            string `This is a batching bug — windows must be sized by converter.maxBatchSize.`);
    }
    map<json> body = {"inputText": texts[0]}; // a STRING, never an array
    int? dimensions = params?.dimensions;
    if dimensions is int {
        body["dimensions"] = dimensions;
    }
    boolean? normalize = params?.normalize;
    if normalize is boolean {
        body["normalize"] = normalize;
    }
    map<json>? extra = additionalFieldsToJson(params?.additionalModelRequestFields);
    if extra is map<json> {
        foreach [string, json] [k, v] in extra.entries() {
            body[k] = v;
        }
    }
    return body;
}

// Decodes a Titan embedding response. Titan DOES report an
// input token count.
isolated function decodeTitanEmbed(json response) returns DecodedEmbedding|ai:Error {
    map<json>|error rr = response.ensureType();
    if rr is error {
        return error ai:LlmInvalidResponseError("Titan embedding response was not a JSON object", rr);
    }
    map<json> r = rr;
    json[]? raw = arrField(r, "embedding");
    if raw is () {
        return error ai:LlmInvalidResponseError("Titan embedding response had no 'embedding' field");
    }
    ai:Vector|ai:Error vector = toVector(raw);
    if vector is ai:Error {
        return vector;
    }
    return {
        embeddings: [vector],
        inputTokenCount: intField(r, "inputTextTokenCount"),
        responseId: () // Titan returns no id
    };
}
