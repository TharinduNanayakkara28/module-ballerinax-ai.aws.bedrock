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

// Amazon Nova on the InvokeModel route. Nova is Converse-shaped but
// the request body REQUIRES `"schemaVersion": "messages-v1"` — omit it and the
// request fails validation. The response shape matches Converse, so `decode` is
// shared with the Converse converter.

// Encodes a Nova InvokeModel request body. Reuses Converse message/
// tool mapping and prepends the mandatory `schemaVersion`.
isolated function encodeNovaInvoke(string? system, ResolvedMessage[] messages,
        ai:ChatCompletionFunctions[] tools, string? stop, InferenceParams params) returns json|ai:Error {
    json[] wire = [];
    foreach ResolvedMessage m in messages {
        wire.push(converseMessage(m)); // Nova uses the Converse content-block shape
    }

    map<json> inferenceConfig = {"maxTokens": params.maxTokens};
    setTemperature(inferenceConfig, params);
    string[]? stops = params.stopSequences;
    if stop is string {
        stops = [stop];
    }
    if stops is string[] && stops.length() > 0 {
        inferenceConfig["stopSequences"] = stops;
    }

    // mandatory schema version; validation fails without it.
    map<json> body = {"schemaVersion": "messages-v1", "messages": wire, "inferenceConfig": inferenceConfig};
    if system is string {
        body["system"] = [{"text": system}]; // top-level, never a message
    }
    if tools.length() > 0 {
        json[] toolSpecs = [];
        foreach ai:ChatCompletionFunctions t in tools {
            toolSpecs.push({
                "toolSpec": {"name": t.name, "description": t.description, "inputSchema": {"json": toolParameters(t)}}
            });
        }
        body["toolConfig"] = {"tools": toolSpecs};
    }
    // Nova reasoningConfig etc. ride additionalModelRequestFields.
    map<json>? extra = additionalFieldsToJson(params?.additionalModelRequestFields);
    if extra is map<json> {
        foreach [string, json] [k, v] in extra.entries() {
            // Nova is the one converter that nests inference knobs under a body key it
            // also builds itself. A passthrough `{"inferenceConfig": {"topK": 20}}`
            // must MERGE — a plain assignment would drop the maxTokens, temperature
            // and stopSequences resolved above, silently ignoring the caller's
            // inference settings. Every other key overwrites, as elsewhere.
            if k == "inferenceConfig" && v is map<json> {
                foreach [string, json] [nestedKey, nestedValue] in v.entries() {
                    inferenceConfig[nestedKey] = nestedValue;
                }
                body["inferenceConfig"] = inferenceConfig;
                continue;
            }
            body[k] = v;
        }
    }
    return body;
}
