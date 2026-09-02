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
import ballerina/crypto;
import ballerina/http;
import ballerina/lang.array;
import ballerina/lang.runtime;
import ballerina/time;

// SigV4 transport — per-route signing (§9.4), retry (§9.5), error mapping (§9.5).
// SigV4 scaffolding (canonical request, signing-key derivation, base16 hex)
// mirrors the ballerinax/aws.dynamodb connector's proven `utils.bal`.

const APPLICATION_JSON = "application/json";
const AWS4_HMAC_SHA256 = "AWS4-HMAC-SHA256";
const AWS4_REQUEST = "aws4_request";

// Wraps SigV4 signing + an HTTP client + retry/error mapping for one resolved
// route (design §9.4-9.5). Resolve-once: host, path, and signing scope are fixed.
isolated client class BedrockTransport {
    private final readonly & BedrockCredentials credentials;
    private final string region;
    private final string signingService;
    private final string host;
    private final string wirePath; // model-id segment single-encoded (from buildEndpoint)
    // The streaming sibling of `wirePath` (`converse-stream` /
    // `invoke-with-response-stream`), or `()` on a route that has none.
    private final string? streamPath;
    private final http:Client httpClient;
    private final readonly & RetryConfig retryConfig;
    // Whether the RESOLVED ROUTE is Mantle. Derived from the route's own signing
    // name, never from the `signingServiceName` override — otherwise a user of that
    // escape hatch silently loses the `bedrock-mantle:CreateInference` hint on a
    // 403, which is the difference between a solvable and an unsolvable error.
    private final boolean isMantleRoute;

    isolated function init(BedrockCredentials credentials, string region, Endpoint ep,
            string? signingServiceName = (), http:ClientConfiguration? httpConfig = (),
            RetryConfig? retryConfig = ()) returns error? {
        self.credentials = credentials.cloneReadOnly();
        self.region = region;
        // Signing name defaults per route, overridable without a release (§9.4).
        self.signingService = signingServiceName ?: ep.signingService;
        self.isMantleRoute = ep.signingService == SIGNING_BEDROCK_MANTLE;
        self.host = ep.host;
        self.wirePath = ep.path;
        self.streamPath = ep.streamPath;
        self.httpClient = check new (string `https://${ep.host}`, httpConfig ?: {});
        RetryConfig rc = retryConfig ?: {};
        self.retryConfig = rc.cloneReadOnly();
    }

    // POSTs a signed request and returns the JSON response plus the response
    // headers we care about (design §9.5), retrying transient errors with
    // exponential backoff. `extraHeaders` carry route-specific headers (guardrail,
    // anthropic-version, workspace, api-key) — all of them are signed.
    isolated function execute(json body, map<string> extraHeaders = {}) returns TransportResponse|ai:Error {
        RetryConfig rc = self.retryConfig;
        int attempt = 0;
        decimal delay = rc.initialDelay;
        while true {
            TransportResponse|RetryableError|ai:Error result = self.executeOnce(body, extraHeaders);
            if result is TransportResponse {
                return result; // success
            }
            if result is ai:Error {
                return result; // non-retryable
            }
            if result is RetryableError {
                // Back off unless attempts are exhausted.
                if attempt >= rc.maxRetries {
                    return error ai:LlmConnectionError(
                        string `${result.message()} (retries exhausted after ${rc.maxRetries} attempts)`, result.cause());
                }
                runtime:sleep(delay);
                delay = decimal:min(delay * rc.backoffFactor, rc.maxDelay);
                attempt += 1;
            }
        }
    }

    // A single signed round-trip (design §9.5). The wire path is sent single-
    // encoded; the canonical URI is double-encoded for the signature (§9.4).
    isolated function executeOnce(json body, map<string> extraHeaders)
            returns TransportResponse|RetryableError|ai:Error {
        string payload = body.toJsonString();
        map<string>|error headers = self.signedHeaders(payload, extraHeaders);
        if headers is error {
            return error ai:Error("Failed to sign the Bedrock request", headers);
        }
        http:Request req = new;
        req.setTextPayload(payload, contentType = APPLICATION_JSON);
        foreach [string, string] [k, v] in headers.entries() {
            req.setHeader(k, v);
        }
        // Wire path is already single-encoded by buildEndpoint; send it verbatim.
        http:Response|error resp = self.httpClient->post(self.wirePath, req);
        if resp is error {
            // Transport-level failure (DNS, TLS, socket) — treat as retryable.
            return error RetryableError("Connection error while calling Bedrock", resp);
        }
        return self.mapResponse(resp);
    }

    // POSTs a signed request to the STREAMING sibling path and hands back the live
    // response with its body unread, for the caller to frame incrementally.
    //
    // SIGNING IS UNCHANGED from the buffered path, and deliberately so: Bedrock
    // streams the RESPONSE, not the request. The request is an ordinary signed POST
    // carrying a complete JSON body, so the payload hash is over the whole body and
    // none of SigV4's chunked-upload machinery (an S3 concern) applies.
    //
    // RETRY COVERS ONLY THE HANDSHAKE — the connection and the response status. Once
    // a 2xx is in hand the response is returned live, and a failure after that
    // cannot be retried: chunks have already been delivered to the caller, and
    // re-sending would duplicate the answer rather than resume it.
    //
    // + sse - Whether the route answers `text/event-stream` (Mantle) rather than
    //         AWS's binary event-stream. Adds the matching `Accept`, which is
    //         SIGNED like every other header we send
    isolated function executeStreaming(json body, map<string> extraHeaders = {}, boolean sse = false)
            returns [http:Response, map<string>]|ai:Error {
        string? path = self.streamPath;
        if path is () {
            // Reached only if a codec carries a `streamDialect` without a stream
            // path being built for its route — a wiring bug.
            return error ai:Error("This Bedrock route has no streaming endpoint");
        }
        map<string> headers = extraHeaders;
        if sse {
            // Copied entry by entry, NOT with `clone()`: the facades pass a
            // `readonly &` header map, and cloning an immutable value hands back the
            // same immutable value — so the `Accept` insertion panicked with
            // "modification not allowed on readonly value" on the first Mantle
            // stream.
            map<string> withAccept = {};
            foreach [string, string] [name, value] in extraHeaders.entries() {
                withAccept[name] = value;
            }
            withAccept[ACCEPT_HEADER] = TEXT_EVENT_STREAM;
            headers = withAccept;
        }
        RetryConfig rc = self.retryConfig;
        int attempt = 0;
        decimal delay = rc.initialDelay;
        while true {
            [http:Response, map<string>]|RetryableError|ai:Error result =
                self.executeStreamingOnce(path, body, headers);
            if result is [http:Response, map<string>] {
                return result;
            }
            if result is ai:Error {
                return result;
            }
            if result is RetryableError {
                if attempt >= rc.maxRetries {
                    return error ai:LlmConnectionError(
                        string `${result.message()} (retries exhausted after ${rc.maxRetries} attempts)`, result.cause());
                }
                runtime:sleep(delay);
                delay = decimal:min(delay * rc.backoffFactor, rc.maxDelay);
                attempt += 1;
            }
        }
    }

    // One signed streaming round-trip. Returns the response with its body UNREAD —
    // calling `getJsonPayload`/`getTextPayload` here would buffer the whole stream
    // and defeat the point. Only the error paths read the body, and only after the
    // status has already ruled out a stream.
    isolated function executeStreamingOnce(string path, json body, map<string> extraHeaders)
            returns [http:Response, map<string>]|RetryableError|ai:Error {
        string payload = body.toJsonString();
        map<string>|error headers = self.signedHeadersFor(path, payload, extraHeaders);
        if headers is error {
            return error ai:Error("Failed to sign the Bedrock streaming request", headers);
        }
        http:Request req = new;
        req.setTextPayload(payload, contentType = APPLICATION_JSON);
        foreach [string, string] [k, v] in headers.entries() {
            req.setHeader(k, v);
        }
        http:Response|error resp = self.httpClient->post(path, req);
        if resp is error {
            return error RetryableError("Connection error while opening the Bedrock stream", resp);
        }
        http:Response response = resp;
        int status = response.statusCode;
        if status < 200 || status >= 300 {
            return self.mapErrorStatus(response, streaming = true);
        }
        // The request id is the only response header the stream needs: the event
        // payloads carry no completion id of their own, and the `ai` contract wants
        // one that is stable across every chunk of a response.
        map<string> responseHeaders = {};
        string? requestId = optionalHeader(response, "x-amzn-RequestId");
        if requestId is string {
            responseHeaders[REQUEST_ID_HEADER] = requestId;
        }
        return [response, responseHeaders];
    }

    // Maps an HTTP response to a `TransportResponse` or a typed error (§9.5 table).
    isolated function mapResponse(http:Response resp) returns TransportResponse|RetryableError|ai:Error {
        int status = resp.statusCode;
        if status >= 200 && status < 300 {
            json|error jsonBody = resp.getJsonPayload();
            if jsonBody is error {
                return error ai:LlmInvalidResponseError("Bedrock response was not valid JSON", jsonBody);
            }
            // Capture the response headers the decoder/provider needs (§9.5).
            //
            // The guardrail-fired signal is NOT here: it is a response BODY field
            // (`amazon-bedrock-guardrailAction`), read by each Invoke codec via
            // `invokeGuardrailAction`. InvokeModel documents only three response
            // headers, and no guardrail among them — the `X-Amzn-Bedrock-Guardrail*`
            // headers are request-only.
            // https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html
            map<string> responseHeaders = {};
            string? requestId = optionalHeader(resp, "x-amzn-RequestId");
            if requestId is string {
                responseHeaders[REQUEST_ID_HEADER] = requestId;
            }
            return {body: jsonBody, headers: responseHeaders};
        }
        return self.mapErrorStatus(resp);
    }

    // Non-2xx -> a typed error. EXTRACTED from `mapResponse` so the streaming path
    // shares the exact same status table — duplicating it would let the two drift.
    // `streaming` only selects the 403 hint, where the two routes genuinely differ:
    // the missing IAM action is `bedrock:InvokeModelWithResponseStream`, not
    // `bedrock:InvokeModel`.
    isolated function mapErrorStatus(http:Response resp, boolean streaming = false) returns RetryableError|ai:Error {
        int status = resp.statusCode;
        string detail = self.errorDetail(resp);
        boolean mantle = self.isMantleRoute;
        match status {
            // 502/504 come from the load balancers fronting Bedrock rather than the
            // service itself, so they carry no Bedrock error code — but they are just
            // as transient as a ThrottlingException and must be retried, not surfaced.
            429|408|500|502|503|504 => {
                return error RetryableError(string `Bedrock transient error (HTTP ${status}): ${detail}`);
            }
            400 => {
                return error ai:Error(string `Bedrock ValidationException (HTTP 400): ${detail}. ` +
                    string `The model may not support this route; try 'apiFamily = INVOKE' (or CONVERSE).`);
            }
            403 => {
                // The likely missing action differs per route, and naming the wrong
                // one sends the reader to the wrong policy. Streaming is its OWN IAM
                // action on both runtime operations — `ConverseStream` included,
                // despite being authorized separately from `Converse` — so a role
                // that calls `chat` fine can still be denied `chatStream`.
                string hint;
                if mantle {
                    hint = " Mantle needs the separate 'bedrock-mantle:CreateInference' IAM action — " +
                        "'bedrock:InvokeModel' permissions are NOT sufficient.";
                } else if streaming {
                    hint = " Streaming needs the separate 'bedrock:InvokeModelWithResponseStream' IAM " +
                        "action — 'bedrock:InvokeModel' alone covers 'chat' but NOT 'chatStream'.";
                } else {
                    hint = "";
                }
                return error ai:Error(string `Bedrock AccessDeniedException (HTTP 403): ${detail}.${hint}`);
            }
            404 => {
                return error ai:Error(string `Bedrock ResourceNotFoundException (HTTP 404): ${detail}. ` +
                    string `Check the model id and region.`);
            }
            424 => {
                return error ai:LlmError(string `Bedrock ModelErrorException (HTTP 424): ${detail}`);
            }
            _ => {
                return error ai:LlmError(string `Bedrock error (HTTP ${status}): ${detail}`);
            }
        }
    }

    // Best-effort extraction of Bedrock's error message from the response body.
    isolated function errorDetail(http:Response resp) returns string {
        json|error j = resp.getJsonPayload();
        if j is map<json> {
            json? msg = j["message"] ?: j["Message"];
            if msg is string {
                return msg;
            }
        }
        return string `status ${resp.statusCode}`;
    }

    // Builds the SigV4 (or bearer) headers for a request against the fixed
    // `wirePath`. A thin wrapper over `signedHeadersFor`, kept so the golden signing
    // tests — pinned to this exact signature — need no change.
    //
    // `fixedClock` exists ONLY for tests: signing is otherwise unobservable without
    // live AWS, and a wall clock makes the output unassertable. Production callers
    // omit it and get `amzTimestamps()`.
    isolated function signedHeaders(string payload, map<string> extraHeaders,
            [string, string]? fixedClock = ()) returns map<string>|error
        => self.signedHeadersFor(self.wirePath, payload, extraHeaders, fixedClock);

    // The path-parameterised form. Streaming signs a DIFFERENT path from the
    // buffered call (`converse-stream` rather than `converse`), and the path is
    // baked into the canonical URI, so it cannot stay fixed at `self.wirePath` —
    // signing the wrong one yields `SignatureDoesNotMatch`, not a 404.
    isolated function signedHeadersFor(string path, string payload, map<string> extraHeaders,
            [string, string]? fixedClock = ()) returns map<string>|error {
        map<string> headers = {};
        foreach [string, string] [k, v] in extraHeaders.entries() {
            headers[k] = v;
        }
        BedrockCredentials creds = self.credentials;

        // Bedrock API key (bearer) — first-class on both endpoints (§9.5): skip SigV4.
        if creds is BearerToken {
            // Anthropic's Mantle surface REJECTS a request carrying BOTH `Authorization`
            // and `x-api-key` (verified live 2026-08-03: either header alone -> 200, both
            // -> 401 `authentication_error: "request must not include both 'authorization'
            // and 'x-api-key' headers"`). When the caller already attached `x-api-key` — an
            // X_API_KEY Mantle route, set by addMantleApiKeyHeader — that header is the sole
            // authenticator and Bearer must be SUPPRESSED. The check is case-insensitive so
            // a routeOverrides entry using a different casing (e.g. `X-Api-Key`) cannot
            // slip past and resurrect the collision.
            if !hasApiKeyHeader(headers) {
                headers["Authorization"] = string `Bearer ${creds.apiKey}`;
            }
            headers["Content-Type"] = APPLICATION_JSON;
            return headers;
        }

        // ---- SigV4 (static / STS) ----
        [string, string] [amzDate, dateStamp] = fixedClock ?: check amzTimestamps();
        // Canonical URI is the DOUBLE-encoded wire path (SigV4 non-S3 rule §9.4):
        // the server re-encodes the received (single-encoded) path once to match.
        string canonicalUri = getCanonicalUri(path);
        string payloadHash = array:toBase16(crypto:hashSha256(payload.toBytes())).toLowerAscii();

        string accessKey = creds.accessKeyId;
        string secretKey = creds.secretAccessKey;
        string? sessionToken = creds is StsCredentials ? creds.sessionToken : ();

        // Sign EVERY header we send (§9.5), sorted by lowercased name.
        map<string> toSign = {"content-type": APPLICATION_JSON, "host": self.host, "x-amz-date": amzDate};
        if sessionToken is string {
            toSign["x-amz-security-token"] = sessionToken;
        }
        foreach [string, string] [name, value] in extraHeaders.entries() {
            toSign[name.toLowerAscii()] = value;
        }
        string[] sortedNames = toSign.keys().sort();
        string canonicalHeaders = "";
        foreach string name in sortedNames {
            canonicalHeaders += name + ":" + (toSign[name] ?: "").trim() + "\n";
        }
        string signedHeaderList = string:'join(";", ...sortedNames);

        string canonicalRequest = "POST" + "\n" + canonicalUri + "\n" + "" + "\n" +
            canonicalHeaders + "\n" + signedHeaderList + "\n" + payloadHash;
        string credentialScope = string `${dateStamp}/${self.region}/${self.signingService}/${AWS4_REQUEST}`;
        string stringToSign = AWS4_HMAC_SHA256 + "\n" + amzDate + "\n" + credentialScope + "\n" +
            array:toBase16(crypto:hashSha256(canonicalRequest.toBytes())).toLowerAscii();

        byte[] signingKey = check getSignatureKey(secretKey, dateStamp, self.region, self.signingService);
        string signature = array:toBase16(check crypto:hmacSha256(stringToSign.toBytes(), signingKey)).toLowerAscii();

        headers["Content-Type"] = APPLICATION_JSON;
        headers["X-Amz-Date"] = amzDate;
        if sessionToken is string {
            headers["X-Amz-Security-Token"] = sessionToken;
        }
        headers["Authorization"] = AWS4_HMAC_SHA256 + " " +
            string `Credential=${accessKey}/${credentialScope}, ` +
            string `SignedHeaders=${signedHeaderList}, Signature=${signature}`;
        return headers;
    }
}

// True if `headers` already carries an `x-api-key` (any casing). Used to decide
// whether the transport's default `Authorization: Bearer` must be SUPPRESSED: an
// X_API_KEY Mantle route attaches x-api-key, and Anthropic's Mantle surface 401s on a
// request that includes both auth headers (see the BearerToken branch above).
isolated function hasApiKeyHeader(map<string> headers) returns boolean {
    foreach string name in headers.keys() {
        if name.toLowerAscii() == "x-api-key" {
            return true;
        }
    }
    return false;
}

// A successful transport round-trip: the JSON body plus the selected response
// headers the decoder/provider needs (design §9.5).
type TransportResponse record {|
    json body;
    map<string> headers;
|};

// Response-header keys captured into `TransportResponse.headers` (design §9.5).
const REQUEST_ID_HEADER = "requestId";

// Sent on a Mantle streaming request. The vendor APIs answer SSE either way, but
// `http:Response.getSseEventStream()` refuses anything that is not
// `text/event-stream`, so asking for it explicitly is what keeps a proxy or a future
// content negotiation from turning a good stream into a binding error.
const ACCEPT_HEADER = "Accept";
const TEXT_EVENT_STREAM = "text/event-stream";

// A retryable transport outcome (408/429/500/502/503/504 or a connection failure — §9.5).
// A `distinct error` so it narrows cleanly against `json` and `ai:Error`.
type RetryableError distinct error;

// Returns a response header value, or `()` if absent.
isolated function optionalHeader(http:Response resp, string name) returns string? {
    string|error value = resp.getHeader(name);
    return value is string ? value : ();
}

// `[amzDate (ISO8601 basic), dateStamp (YYYYMMDD)]` for NOW, in UTC.
isolated function amzTimestamps() returns [string, string]|error {
    return formatAmzTimestamps(time:utcToCivil(time:utcNow()));
}

// The pure formatter behind `amzTimestamps` — split out so the format can be
// tested at a fixed clock, including the second-59 boundary that a wall-clock test
// would only hit once a minute (and only half the time).
isolated function formatAmzTimestamps(time:Civil c) returns [string, string]|error {
    string y = pad(c.year, 4);
    string mo = pad(c.month, 2);
    string d = pad(c.day, 2);
    string h = pad(c.hour, 2);
    string mi = pad(c.minute, 2);
    // `time:Civil.second` is a decimal carrying sub-second precision. `<int>` on a
    // decimal ROUNDS (half-to-even) in Ballerina — it does not truncate — so
    // `<int>59.7d` is 60, and second 59 with a fraction >= 0.5 would emit
    // "...T235960Z". That is not a valid ISO 8601 basic timestamp; AWS rejects the
    // X-Amz-Date header, and the resulting 403 is NOT retryable. Roughly 1 request
    // in 120 (P(second==59) * P(frac>=0.5)). `.floor()` truncates, which is what
    // SigV4 wants: whole seconds, no milliseconds.
    // https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv-signing-elements.html
    decimal secDec = c.second ?: 0;
    string s = pad(<int>secDec.floor(), 2);
    string amzDate = string `${y}${mo}${d}T${h}${mi}${s}Z`;
    string dateStamp = string `${y}${mo}${d}`;
    return [amzDate, dateStamp];
}

// Zero-pads `n` to `width` digits.
isolated function pad(int n, int width) returns string {
    string s = n.toString();
    while s.length() < width {
        s = "0" + s;
    }
    return s;
}

// Double-encodes the (already single-encoded) wire path for the SigV4 canonical
// URI (non-S3 rule §9.4): encode again, then restore structural `/` separators
// (their `%2F` maps back to `/`, while a model-id's internal `%2F`→`%252F` stays
// double-encoded).
//
// Uses the SAME RFC 3986 encoder as `buildEndpoint` — the two must agree
// character-for-character or the server cannot reconstruct what we signed. Total:
// `encodePathSegment` has no failure mode, so neither does this.
isolated function getCanonicalUri(string wirePath) returns string {
    return re `%2F`.replaceAll(encodePathSegment(wirePath), "/");
}

// SigV4 signing-key derivation (design §9.4). Identical to aws.dynamodb.
isolated function getSignatureKey(string secretKey, string dateStamp, string region, string serviceName)
        returns byte[]|error {
    byte[] kDate = check crypto:hmacSha256(dateStamp.toBytes(), ("AWS4" + secretKey).toBytes());
    byte[] kRegion = check crypto:hmacSha256(region.toBytes(), kDate);
    byte[] kService = check crypto:hmacSha256(serviceName.toBytes(), kRegion);
    return crypto:hmacSha256(AWS4_REQUEST.toBytes(), kService);
}
