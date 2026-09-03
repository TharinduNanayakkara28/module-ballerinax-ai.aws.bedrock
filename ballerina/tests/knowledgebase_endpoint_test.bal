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

import ballerina/test;

// `buildAgentEndpoint` — the knowledge-base control/data plane hosts. No I/O.

@test:Config {}
function testControlPlaneHostAndSigningService() returns error? {
    Endpoint ep = check buildAgentEndpoint(AGENT_CONTROL, "us-east-1");
    test:assertEquals(ep.host, "bedrock-agent.us-east-1.amazonaws.com");
    test:assertEquals(ep.baseUrl, "https://bedrock-agent.us-east-1.amazonaws.com");
    // BOTH agent planes sign as SigV4 service 'bedrock' — NOT their own hostname.
    // Per the `signingName` field in both botocore service models
    // (bedrock-agent/2023-06-05 and bedrock-agent-runtime/2023-07-26).
    test:assertEquals(ep.signingService, SIGNING_BEDROCK);
    // Paths are per-call for the KB spine, never fixed at construction.
    test:assertEquals(ep.path, "");
}

@test:Config {}
function testDataPlaneHostAndSigningService() returns error? {
    Endpoint ep = check buildAgentEndpoint(AGENT_DATA, "us-east-1");
    test:assertEquals(ep.host, "bedrock-agent-runtime.us-east-1.amazonaws.com");
    test:assertEquals(ep.signingService, SIGNING_BEDROCK);
}

@test:Config {}
function testAgentEndpointsAreRegionParameterized() returns error? {
    Endpoint control = check buildAgentEndpoint(AGENT_CONTROL, "eu-west-1");
    test:assertEquals(control.host, "bedrock-agent.eu-west-1.amazonaws.com");

    Endpoint data = check buildAgentEndpoint(AGENT_DATA, "ap-southeast-2");
    test:assertEquals(data.host, "bedrock-agent-runtime.ap-southeast-2.amazonaws.com");
}

@test:Config {}
function testAgentEndpointOnChinaPartitionUsesTheChinaSuffix() returns error? {
    Endpoint ep = check buildAgentEndpoint(AGENT_CONTROL, "cn-north-1");
    test:assertTrue(ep.host.endsWith("amazonaws.com.cn"), ep.host);
    test:assertEquals(ep.signingService, SIGNING_BEDROCK);
}

@test:Config {}
function testAgentEndpointHonoursACustomServiceUrl() returns error? {
    Endpoint ep = check buildAgentEndpoint(AGENT_DATA, "us-east-1", "http://localhost:4566");
    test:assertEquals(ep.baseUrl, "http://localhost:4566");
    test:assertEquals(ep.host, "localhost:4566");
    test:assertEquals(ep.signingService, SIGNING_BEDROCK);
}
