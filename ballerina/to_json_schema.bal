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

// typedesc -> JSON schema for generate()'s expected response type.
// Ported from the reference module module-ballerinax-ai.openai (`to_json_schema.bal`)
// so this module owns its schema generation, as its siblings do.
//
// Resolution order:
//   1. the `@ai:JsonSchema` annotation ballerina/ai's compiler plugin attaches at
//      the generate() call site (records),
//   2. runtime generation in this module's `Native` (arrays, unions, simple types).
// A type neither can express is an `ai:Error` — never a silently empty schema.
//
// The reference carries a THIRD step, a pure-Ballerina fallback gated on `Native`
// returning nil. That step is not reproduced here because it cannot run: `Native`
// either returns a schema or raises, and it already covers a strict superset of what
// the fallback expressed (recursive arrays and unions, versus simple types and
// simple-member arrays only). Keeping it would mean ~70 lines that read like a
// safety net while being unreachable.

import ballerina/ai;
import ballerina/jballerina.java;

isolated function generateJsonSchemaForTypedescAsJson(typedesc<json> expectedResponseTypedesc)
        returns map<json>|ai:Error =>
    let map<json>? ann = expectedResponseTypedesc.@ai:JsonSchema in ann
                ?: check generateJsonSchemaForTypedescNative(expectedResponseTypedesc);

isolated function generateJsonSchemaForTypedescNative(typedesc<anydata> td) returns map<json>|ai:Error = @java:Method {
    'class: "io.ballerina.lib.ai.aws.bedrock.Native"
} external;
