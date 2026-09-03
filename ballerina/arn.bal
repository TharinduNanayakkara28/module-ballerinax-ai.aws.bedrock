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

// A parsed Bedrock ARN. The resource-type token is the free dispatch signal, and
// the region/partition segments are authoritative over `config.region`.
// Format: `arn:partition:service:region:account-id:resource-type/resource-id`.
type ParsedArn record {|
    # `aws` | `aws-cn` | `aws-us-gov`.
    string partition;
    # e.g. `bedrock`.
    string 'service;

    # Authoritative region — overrides `config.region`. MAY be empty:
    # foundation-model ARNs are often written globally, e.g.
    # `arn:aws:bedrock::123456789012:foundation-model/anthropic.claude-v2`.
    # `resolveArn` falls back to the caller's region in that case.
    string region;

    string accountId;
    # e.g. `imported-model`, `provisioned-model`, `inference-profile`.
    string resourceType;
    # The opaque id after the `/` (or `:`) delimiter; may be empty.
    string resourceId;
|};

// `true` if `model` is an ARN.
isolated function isArn(string model) returns boolean => model.startsWith("arn:");

// Parses a Bedrock ARN into its segments (pure). The first five `:`
// segments are structural; everything after the fifth colon is the resource,
// which itself uses `/` (or, rarely, `:`) between type and id.
isolated function parseArn(string arn) returns ParsedArn|error {
    if !arn.startsWith("arn:") {
        return error(string `not an ARN: ${arn}`);
    }
    string rest = arn;
    string[] fields = [];
    int count = 0;
    while count < 5 {
        int? idx = rest.indexOf(":");
        if idx is () {
            return error(string `malformed ARN (fewer than 6 segments): ${arn}`);
        }
        fields.push(rest.substring(0, idx));
        rest = rest.substring(idx + 1);
        count += 1;
    }
    // fields = ["arn", partition, service, region, account]; `rest` = resource.
    // Partition and service are structural — an empty one is a malformed ARN and
    // must be rejected here rather than producing a nonsense host downstream.
    // (`region` is legitimately empty on global ARNs; `resolveArn` substitutes the
    // caller's region for those. `accountId` is empty on AWS-owned ARNs.)
    if fields[1] == "" {
        return error(string `malformed ARN (empty partition segment): ${arn}`);
    }
    if fields[2] == "" {
        return error(string `malformed ARN (empty service segment): ${arn}`);
    }
    string res = rest;
    string resourceType;
    string resourceId;
    int? slash = res.indexOf("/");
    int? colon = res.indexOf(":");
    if slash is int {
        resourceType = res.substring(0, slash);
        resourceId = res.substring(slash + 1);
    } else if colon is int {
        resourceType = res.substring(0, colon);
        resourceId = res.substring(colon + 1);
    } else {
        resourceType = res;
        resourceId = "";
    }
    return {
        partition: fields[1],
        'service: fields[2],
        region: fields[3],
        accountId: fields[4],
        resourceType,
        resourceId
    };
}
