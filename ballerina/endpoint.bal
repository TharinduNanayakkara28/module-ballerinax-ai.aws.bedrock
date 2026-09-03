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

import ballerinax/aws;

// Endpoint construction — host · path · partition · signing name.
// L2: runs once, at construction. The model-id path segment is
// URL-encoded HERE (single-encode): structural `/` stay literal, but the model
// id's `:`/`/` (ARNs, `-v1:0` ids) become `%3A`/`%2F`. This is the WIRE path; the
// transport double-encodes it for the SigV4 canonical URI (SigV4 non-S3 rule).

const SIGNING_BEDROCK = "bedrock";               // Converse / Invoke
const SIGNING_BEDROCK_MANTLE = "bedrock-mantle"; // Mantle

// SDK endpoint-metadata service prefixes. Mantle is served from the
// partition-neutral `api.aws` suffix in every partition that has it.
// https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
const RUNTIME_ENDPOINT_PREFIX = "bedrock-runtime";
const MANTLE_ENDPOINT_PREFIX = "bedrock-mantle";

// Knowledge-base control/data planes. BOTH sign as SigV4 service `bedrock` — same as
// Converse/Invoke, NOT their own hostname — per the `signingName` field in both
// botocore service models (bedrock-agent/2023-06-05 and
// bedrock-agent-runtime/2023-07-26/service-2.json). The endpoint PREFIX (hostname)
// and the SIGNING service are two different things in this service family; only the
// former differs here.
const AGENT_ENDPOINT_PREFIX = "bedrock-agent";
const AGENT_RUNTIME_ENDPOINT_PREFIX = "bedrock-agent-runtime";

// The resolved wire endpoint. `signingService` is the SigV4 scope,
// not the IAM namespace — they differ inside this service family.
type Endpoint record {|
    # Origin, e.g. `https://bedrock-runtime.us-east-1.amazonaws.com`.
    string baseUrl;
    # Host header / SigV4 canonical host, e.g. `bedrock-runtime.us-east-1.amazonaws.com`.
    string host;
    # Wire request path with the model-id segment single-encoded.
    string path;
    # The STREAMING counterpart of `path`, or `()` where the route has none.
    #
    # A separate member rather than surgery on `path` at call time: the model-id
    # segment is single-encoded exactly once, here, and deriving the stream path by
    # string-replacing a suffix on an already-encoded path would put that invariant
    # in two places. On Mantle it is EQUAL to `path` — that route streams from the
    # same URL, switched on by a body field — and the member stays optional so a
    # future route with no streaming surface at all can say so.
    string? streamPath = ();
    # SigV4 signing name for this route.
    string signingService;
|};

// The default `serviceUrl`. A TEMPLATE, not a constant host, because the origin is
// decided by the resolved route: `bedrock-runtime.{region}.amazonaws.com` on
// Converse/Invoke, `bedrock-mantle.{region}.api.aws` on Mantle, and
// `amazonaws.com.cn` in China. One pattern spans all three.
//
// Substitution is a no-op on a string containing no placeholders, so a caller who
// passes a concrete URL needs no sentinel and no "did they override it?" check —
// their URL simply passes through untouched.
public const DEFAULT_SERVICE_URL = "https://bedrock-{endpoint}.{region}.{domain}";

// Resolves `serviceUrl` against a route: the default template defers wholly to AWS
// SDK endpoint metadata, a concrete URL passes through unchanged, and a custom
// template is substituted with the `{domain}` taken from that same metadata.
//
// `{region}` comes from the RESOLVED ROUTE, not the raw `region` argument, so an ARN
// whose region segment overrides `region` still lands correctly. The signing region
// and signing name are NOT derived from the result — a VPCE, FIPS or gateway host
// still signs the route's own region/service scope.
isolated function resolveServiceUrl(string serviceUrl, Route route, boolean fips) returns string|error {
    boolean mantle = route.family == MANTLE;
    string serviceName = mantle ? MANTLE_ENDPOINT_PREFIX : RUNTIME_ENDPOINT_PREFIX;
    string endpointName = mantle ? "mantle" : (fips ? "runtime-fips" : "runtime");
    // Mantle is served from the partition-neutral `api.aws` suffix, which the SDK
    // models as the DUALSTACK variant — without this flag the metadata falls back to
    // `bedrock-mantle.{region}.amazonaws.com`, which does not resolve.
    // Verified 2026-08-09 against ballerinax/aws 1.0.1.
    return resolveServiceUrlCore(serviceName, endpointName, route.region, serviceUrl, fips, mantle);
}

// The knowledge-base agent planes: standard regional hosts, like `bedrock-runtime` —
// no Mantle-style dualstack suffix.
isolated function resolveAgentServiceUrl(string serviceUrl, string serviceName, string endpointName,
        string region, boolean fips) returns string|error
    => resolveServiceUrlCore(serviceName, endpointName, region, serviceUrl, fips, false);

// The pure core behind both `resolveServiceUrl` and `resolveAgentServiceUrl`: the
// default template defers wholly to AWS SDK endpoint metadata, a concrete URL passes
// through unchanged, and a custom template is substituted with the `{domain}` taken
// from that same metadata.
isolated function resolveServiceUrlCore(string serviceName, string endpointName, string region,
        string serviceUrl, boolean fips, boolean dualstack) returns string|error {
    aws:EndpointConfig endpointConfig = {fips, dualstack};

    // The default: the SDK owns the whole origin (all partitions, FIPS/dualstack
    // variants, per-service exceptions, and a standard-pattern fallback for regions
    // newer than the bundled metadata).
    if serviceUrl == DEFAULT_SERVICE_URL {
        return aws:resolveEndpoint(serviceName, region, endpointConfig);
    }
    // A concrete URL (PrivateLink, gateway, LocalStack) passes through untouched.
    if !serviceUrl.includes("{") {
        return trimTrailingSlash(serviceUrl);
    }

    // A custom template still needs the placeholder VALUES; take `{domain}` from the
    // same metadata rather than hardcoding a suffix table.
    string host = aws:resolveEndpointHost(serviceName, region, endpointConfig);
    string prefix = string `bedrock-${endpointName}.${region}.`;
    if !host.startsWith(prefix) {
        // The SDK returned a host this template cannot express (an endpoint exception
        // AWS added later). Trust the metadata over the template.
        return string `https://${host}`;
    }
    string url = re `\{endpoint\}`.replaceAll(serviceUrl, endpointName);
    url = re `\{region\}`.replaceAll(url, region);
    url = re `\{domain\}`.replaceAll(url, host.substring(prefix.length()));

    // A surviving brace is ALWAYS a typo (`{regoin}`), never a legal host: braces are
    // not valid in DNS names. Catching it here turns a silent DNS failure into a
    // construction error that names the mistake.
    if url.includes("{") || url.includes("}") {
        return error(string `unresolved placeholder in serviceUrl '${url}'; ` +
            string `supported placeholders are {endpoint}, {region} and {domain}`);
    }
    return trimTrailingSlash(url);
}

// Trailing slash would double up against the route-derived path.
isolated function trimTrailingSlash(string url) returns string
    => url.endsWith("/") ? url.substring(0, url.length() - 1) : url;

// Extracts the host from an origin for the `Host` header / SigV4 canonical host.
isolated function hostOf(string baseUrl) returns string {
    string rest = baseUrl;
    foreach string scheme in ["https://", "http://"] {
        if rest.startsWith(scheme) {
            rest = rest.substring(scheme.length());
            break;
        }
    }
    int? slash = rest.indexOf("/");
    return slash is int ? rest.substring(0, slash) : rest;
}

// Builds the endpoint for a resolved route. Pure. Fails before
// any I/O for the one host-shape AWS cannot template: Mantle on a non-`aws`
// partition.
//
// `serviceUrl` replaces the ORIGIN only — the route-derived path is still appended,
// because that path differs per family (`/model/{id}/converse` vs
// `/anthropic/v1/messages`) and is not the caller's to choose.
isolated function buildEndpoint(Route route, string serviceUrl = DEFAULT_SERVICE_URL,
        boolean fips = false) returns Endpoint|error {
    if route.family == MANTLE {
        if fips {
            // No `bedrock-mantle-fips` host exists; the SDK fallback would happily
            // synthesise one and fail at DNS. Name the mistake here instead.
            return error("'fips' is not available on the Mantle route: there is no " +
                "bedrock-mantle FIPS endpoint. Use 'apiFamily = CONVERSE' or 'INVOKE' " +
                "for a FIPS-compliant Bedrock call.");
        }
        // Mantle is served from the partition-neutral `api.aws` suffix. That suffix
        // exists in the commercial AND GovCloud partitions — `bedrock-mantle.us-gov-west-1.api.aws`
        // is real — but has no China analogue: `aws-cn` uses `amazonaws.com.cn`
        // throughout, so no bedrock-mantle host can be formed there at all.
        //
        // This is a HOST-SHAPE guard, not an availability oracle. Within an allowed
        // partition Mantle ships in only a SUBSET of regions (us-west-1, ca-central-1
        // and us-gov-east-1 are `bedrock-runtime`-only today), and that subset grows as
        // AWS expands. Encoding the region list here would reject a newly-added Mantle
        // region until our next release — the exact staleness the routing escape
        // hatches exist to avoid — so a well-formed but not-yet-served region is left
        // for AWS to reject at call time with its own diagnosis.
        // https://docs.aws.amazon.com/bedrock/latest/userguide/endpoints-region-availability.html
        if route.partition != "aws" && route.partition != "aws-us-gov" {
            return error(string `Mantle is not available on partition '${route.partition}': the ` +
                string `bedrock-mantle 'api.aws' host has no '${route.partition}' analogue. ` +
                string `Use a commercial ('aws') or GovCloud ('aws-us-gov') region.`);
        }
        MantleEntry entry = check route.mantleEntry.ensureType();
        string mantleBase = check resolveServiceUrl(serviceUrl, route, fips);
        return {
            baseUrl: mantleBase,
            host: hostOf(mantleBase),
            path: entry.path,
            // Mantle streams as SSE on the SAME path, switched on by a `"stream":
            // true` body field (carried by the converter's `streamFields`) rather than
            // by a different endpoint — so the stream path IS the path. The two
            // members are kept distinct anyway: `runChatStream` posts to
            // `streamPath` on every route, and collapsing them here is what keeps
            // that one call site free of a Mantle special case.
            // https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
            streamPath: entry.path,
            signingService: SIGNING_BEDROCK_MANTLE
        };
    }

    // Converse / Invoke on `bedrock-runtime`, partition-aware domain.
    string base = check resolveServiceUrl(serviceUrl, route, fips);
    // Single-encode the model-id segment (ARNs/`-v1:0` ids carry `:` and `/`).
    string encodedId = encodePathSegment(route.effectiveModelId);
    string path = route.family == CONVERSE
        ? string `/model/${encodedId}/converse`
        : string `/model/${encodedId}/invoke`;
    // Bedrock puts streaming on a sibling operation, not a query flag: `Converse`
    // pairs with `ConverseStream`, `InvokeModel` with `InvokeModelWithResponseStream`.
    // The ORIGIN is whatever `serviceUrl` resolved to, exactly as the buffered path —
    // a custom endpoint serves both or neither.
    // https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
    string streamPath = route.family == CONVERSE
        ? string `/model/${encodedId}/converse-stream`
        : string `/model/${encodedId}/invoke-with-response-stream`;
    return {baseUrl: base, host: hostOf(base), path, streamPath, signingService: SIGNING_BEDROCK};
}

// Which bedrock-agent plane an endpoint is for. Module-private: only the knowledge
// base spine needs this distinction.
enum AgentPlane {
    // Control plane (`bedrock-agent`): CreateKnowledgeBase, CreateDataSource,
    // IngestKnowledgeBaseDocuments, List/Get/DeleteKnowledgeBaseDocuments, ...
    AGENT_CONTROL,
    // Data plane (`bedrock-agent-runtime`): Retrieve.
    AGENT_DATA
}

// Builds the endpoint for a knowledge-base agent plane. Unlike `buildEndpoint`, the
// request PATH is not fixed at construction — a single `BedrockTransport` for a
// plane serves many paths (`/knowledgebases/`, `/knowledgebases/{id}/retrieve`,
// `/knowledgebases/{id}/datasources/{id}/documents`, …) — so `path` is left empty
// and every call site of `BedrockTransport.executeRequest` supplies its own.
isolated function buildAgentEndpoint(AgentPlane plane, string region, string serviceUrl = DEFAULT_SERVICE_URL,
        boolean fips = false) returns Endpoint|error {
    string serviceName = plane == AGENT_DATA ? AGENT_RUNTIME_ENDPOINT_PREFIX : AGENT_ENDPOINT_PREFIX;
    string endpointName = plane == AGENT_DATA ? "agent-runtime" : "agent";
    string base = check resolveAgentServiceUrl(serviceUrl, serviceName, endpointName, region, fips);
    return {baseUrl: base, host: hostOf(base), path: "", signingService: SIGNING_BEDROCK};
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
