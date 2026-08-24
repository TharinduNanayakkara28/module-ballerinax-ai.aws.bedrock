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

// Endpoint construction — host · path · partition · signing name (design §9.1,
// §9.2, §9.4). L2: runs once, at construction. The model-id path segment is
// URL-encoded HERE (single-encode): structural `/` stay literal, but the model
// id's `:`/`/` (ARNs, `-v1:0` ids) become `%3A`/`%2F`. This is the WIRE path; the
// transport double-encodes it for the SigV4 canonical URI (SigV4 non-S3 rule).

const SIGNING_BEDROCK = "bedrock";               // Converse / Invoke — §9.4
const SIGNING_BEDROCK_MANTLE = "bedrock-mantle"; // Mantle — §9.4

// The resolved wire endpoint (design §9). `signingService` is the SigV4 scope,
// not the IAM namespace — they differ inside this service family (§9.4).
type Endpoint record {|
    # Host, e.g. `bedrock-runtime.us-east-1.amazonaws.com`.
    string host;
    # Wire request path with the model-id segment single-encoded (§9.1).
    string path;
    # The STREAMING counterpart of `path`, or `()` where the route has none.
    #
    # A separate member rather than surgery on `path` at call time: the model-id
    # segment is single-encoded exactly once, here, and deriving the stream path by
    # string-replacing a suffix on an already-encoded path would put that invariant
    # in two places. Mantle has no entry — its streaming surface is SSE on the same
    # path with `"stream": true` in the body, which is not yet implemented.
    string? streamPath = ();
    # SigV4 signing name for this route.
    string signingService;
|};

// Builds the endpoint for a resolved route (design §9.1-9.4). Pure. Fails before
// any I/O for the one host-shape AWS cannot template: Mantle on a non-`aws`
// partition (design §9.2, open item #2).
isolated function buildEndpoint(Route route) returns Endpoint|error {
    if route.family == MANTLE {
        // Mantle uses the `api.aws` suffix and is NOT partition-templated (§9.2).
        if route.partition != "aws" {
            return error(string `Mantle is not available on partition '${route.partition}': the ` +
                string `bedrock-mantle 'api.aws' host is not partition-templated (design §9.2). ` +
                string `Use a commercial ('aws') region.`);
        }
        MantleEntry entry = check route.mantleEntry.ensureType();
        return {
            host: string `bedrock-mantle.${route.region}.api.aws`,
            path: entry.path,
            // Mantle streams as SSE on the SAME path, switched on by a `"stream":
            // true` body field rather than a different endpoint — so there is no
            // second path to record. Left unset until that dialect is implemented.
            streamPath: (),
            signingService: SIGNING_BEDROCK_MANTLE
        };
    }

    // Converse / Invoke on `bedrock-runtime`, partition-aware domain (§9.2).
    string host = string `bedrock-runtime.${route.region}.${awsDomain(route.partition)}`;
    // Single-encode the model-id segment (ARNs/`-v1:0` ids carry `:` and `/`).
    string encodedId = encodePathSegment(route.effectiveModelId);
    string path = route.family == CONVERSE
        ? string `/model/${encodedId}/converse` // §9.1
        : string `/model/${encodedId}/invoke`;   // §9.1
    // Bedrock puts streaming on a sibling operation, not a query flag: `Converse`
    // pairs with `ConverseStream`, `InvokeModel` with `InvokeModelWithResponseStream`.
    // https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
    string streamPath = route.family == CONVERSE
        ? string `/model/${encodedId}/converse-stream`
        : string `/model/${encodedId}/invoke-with-response-stream`;
    return {host, path, streamPath, signingService: SIGNING_BEDROCK};
}

// RFC 3986 unreserved set — the ONLY characters SigV4 leaves literal.
// https://datatracker.ietf.org/doc/html/rfc3986#section-2.3
const string RFC3986_UNRESERVED =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";

final readonly & string[] HEX_DIGITS =
    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F"];

// Percent-encodes one path segment per RFC 3986, which is what SigV4 requires.
//
// `url:encode` is NOT a path-segment encoder — it applies
// `application/x-www-form-urlencoded` rules, which differ from SigV4's on exactly
// the characters a model id can contain: a space becomes `+` instead of `%20`, and
// `~` is escaped even though it is unreserved. Since `getCanonicalUri` re-applies
// this same function to build the signing input, any such character produced a
// canonical URI AWS could not reconstruct — a `SignatureDoesNotMatch` on every
// request, with nothing in the message pointing at the encoder.
//
// Total by construction: every byte either passes through or becomes `%XX`, so
// there is no failure mode for a caller to handle.
// https://docs.aws.amazon.com/IAM/latest/UserGuide/create-signed-request.html
isolated function encodePathSegment(string segment) returns string {
    string encoded = "";
    foreach string:Char ch in segment {
        if RFC3986_UNRESERVED.includes(ch) {
            encoded += ch;
            continue;
        }
        // Percent-encode each UTF-8 byte, uppercase hex (SigV4 requires uppercase).
        foreach byte b in ch.toBytes() {
            int value = <int>b;
            encoded += "%" + HEX_DIGITS[value / 16] + HEX_DIGITS[value % 16];
        }
    }
    return encoded;
}

// Partition-aware DNS suffix (design §9.2). Route every host through here — n8n's
// hardcoded `.amazonaws.com` is a real China bug (§9.2).
isolated function awsDomain(string partition) returns string
    => partition == "aws-cn" ? "amazonaws.com.cn" : "amazonaws.com";
