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

import ballerina/http;
import ballerina/test;

// End-to-end wire test for `serviceUrl` against a LOCAL server.
//
// The other serviceUrl tests are pure string resolution — they prove `buildEndpoint`
// computes the right origin, but NOT that the origin is actually honoured by the
// transport. This one sends a real signed request over a real socket and inspects
// what arrived, which is the only way to catch "the URL resolved fine but the client
// ignored it" without AWS credentials.
//
// It cannot prove that a FIPS or PrivateLink host EXISTS — that is DNS's answer, not
// a unit test's. What it does prove is the mechanism: a custom origin is used
// verbatim, the route-derived path is still appended, the `Host` header follows the
// override, and SigV4 still signs the route's own scope rather than the new host's.

// One isolated variable, not three: Ballerina permits access to at most one
// lock-restricted variable per `lock` statement.
isolated map<string> captured = {};

// A stand-in Bedrock: records what it received and returns a minimal Converse body.
isolated service class MockBedrock {
    *http:Service;

    isolated resource function post [string... path](http:Request req) returns json|error {
        string|error hostHeader = req.getHeader("Host");
        string|error auth = req.getHeader("Authorization");
        map<string> seen = {
            path: req.rawPath,
            host: hostHeader is string ? hostHeader : "",
            auth: auth is string ? auth : ""
        };
        lock {
            captured = seen.clone();
        }
        return {
            output: {message: {role: "assistant", content: [{text: "ok"}]}},
            stopReason: "end_turn",
            usage: {inputTokens: 3, outputTokens: 1}
        };
    }
}

@test:Config {}
function testCustomServiceUrlIsHonouredOnTheWire() returns error? {
    // A fixed port: Ballerina's http:Listener does not accept 0 ("pick one for me").
    final int port = 18099;
    http:Listener mockListener = check new (port);
    check mockListener.attach(new MockBedrock(), "/");
    check mockListener.'start();

    // Point the provider at the local server. `http://` (not https) also proves the
    // override is taken verbatim rather than forced onto a scheme.
    AnthropicModelProvider provider = check new (
            "anthropic.claude-sonnet-4-6", TEST_CREDS, "us-east-1",
            serviceUrl = string `http://localhost:${port}`);
    _ = check provider->chat([{role: "user", content: "hi"}]);
    check mockListener.gracefulStop();

    map<string> seen;
    lock {
        seen = captured.clone();
    }
    string path = seen["path"] ?: "";
    string host = seen["host"] ?: "";
    string auth = seen["auth"] ?: "";

    // 1. The route-derived path survives the origin override.
    test:assertEquals(path, "/model/anthropic.claude-sonnet-4-6/converse",
            "serviceUrl replaces the origin only — the path stays route-derived");

    // 2. The Host header follows the override, not the AWS default. If this said
    //    `bedrock-runtime.us-east-1.amazonaws.com` the signature would be computed
    //    over a host we never contacted, which is a SignatureDoesNotMatch in prod.
    test:assertTrue(host.startsWith("localhost:"),
            string `Host header must follow the override, got '${host}'`);

    // 3. SigV4 still scopes to the ROUTE's region and service, not the new host.
    //    This is the invariant a FIPS/VPCE/gateway user depends on.
    test:assertTrue(auth.startsWith("AWS4-HMAC-SHA256 "), auth);
    test:assertTrue(auth.includes("/us-east-1/bedrock/aws4_request"),
            string `signing scope must stay us-east-1/bedrock, got '${auth}'`);
}
