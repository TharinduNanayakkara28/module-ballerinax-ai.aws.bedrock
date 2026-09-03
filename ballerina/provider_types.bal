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
import ballerinax/aws.auth;

// ============================================================================
// Credentials. SigV4 sources come from `ballerinax/aws.auth`; the bearer
// (Bedrock API key) is module-local because `auth:AuthConfig` is SigV4-only.
// (https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html)
// ============================================================================

# A Bedrock API key (bearer token) — first-class on both endpoints.
#
# Module-local rather than an `auth:AuthConfig` member: every `auth:AuthConfig`
# variant resolves to `auth:Credentials` (access key + secret + optional session
# token), which cannot carry an opaque key. A bearer bypasses SigV4 entirely.
public type BearerToken record {|
    # The Bedrock API key, sent as `Authorization: Bearer` — or as `x-api-key` on a
    # Mantle Messages path, where the two headers are mutually exclusive.
    string apiKey;
|};

# The credential union accepted by every provider.
#
# `auth:AuthConfig` covers static keys (`auth:StaticAuthConfig`, whose optional
# `sessionToken` also carries temporary STS credentials), `auth:AssumeRoleConfig`
# for cross-account, `auth:WebIdentityConfig` for EKS IRSA, plus SSO, named
# profiles and `credential_process`. Its default member, `auth:DEFAULT_CREDENTIALS`,
# walks the full chain — env vars, web identity, SSO, shared config, external
# process, ECS container credentials, EC2 IMDSv2 — so nothing needs configuring on
# EC2, ECS, EKS or Lambda. Expiry and refresh are handled by `auth:CredentialProvider`.
public type BedrockCredentials auth:AuthConfig|BearerToken;

// ============================================================================
// Guardrails / retry.
// ============================================================================

# Guardrail configuration. Placement is route-specific: Converse body field,
# Invoke headers, and a construction error on Mantle.
public type GuardrailConfig record {|
    # `guardrailIdentifier` (Converse body / `X-Amzn-Bedrock-GuardrailIdentifier`).
    string guardrailIdentifier;
    # `guardrailVersion`.
    string guardrailVersion;
|};

# Retry policy for the transport's throttling/warm-up backoff.
public type RetryConfig record {|
    # Max retry attempts for retryable errors (408/429/500/502/503/504).
    int maxRetries = 3;
    # Initial backoff delay, seconds.
    decimal initialDelay = 1.0;
    # Backoff ceiling, seconds.
    decimal maxDelay = 20.0;
    # Exponential backoff multiplier.
    decimal backoffFactor = 2.0;
|};

// ============================================================================
// Shared model config. Each vendor `*Config` includes this and adds
// only its vendor-specific fields.
// ============================================================================

# Everything that is not the model's identity, shared across vendors.
public type CommonModelConfig record {|
    // --- Routing ---
    # Route selection: `AUTO` (default) runs the resolver; `CONVERSE`/`INVOKE`/
    # `MANTLE` force that family. The escape hatch — outranks every heuristic.
    ApiFamily apiFamily = AUTO;

    # Use the FIPS 140-validated endpoint variant (`bedrock-runtime-fips.{region}...`),
    # resolved from AWS SDK endpoint metadata. Required for FedRAMP and GovCloud.
    # Changes only which HOST is dialled — never the SigV4 signing scope, the request
    # path, or the body. Ignored when `serviceUrl` is a concrete URL, and rejected at
    # construction on a MANTLE route (no `bedrock-mantle-fips` host exists).
    boolean fips = false;

    // --- Inference ---
    # Provider-level stop sequences; a per-call `stop` overrides these.
    string[] stopSequences?;

    // --- Converse passthrough ---
    # Forwarded verbatim on Converse; ignored elsewhere.
    # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
    AdditionalRequestFields additionalModelRequestFields?;

    # `serviceTier`.
    ServiceTier serviceTier?;

    # Request latency-optimized inference on Converse: routes the call onto AWS's
    # faster serving path (custom silicon / reserved capacity) for a lower
    # time-to-first-token, at a higher price. Same output — a speed/cost dial only.
    # Support is per model and region; unsupported combinations are rejected by AWS.
    boolean latencyOptimized?;

    // --- Cross-cutting ---
    # Guardrail; construction error on a MANTLE route.
    GuardrailConfig guardrail?;
    # Retry policy.
    RetryConfig retryConfig?;
    # Underlying HTTP client configuration.
    http:ClientConfiguration httpConfig?;
|};

// ============================================================================
// Inference params — resolved once at construction. Carries the
// Converse-body passthrough so the fixed converter signature can emit it; non-Converse
// converters ignore the extra fields.
// ============================================================================

# Resolved inference parameters plus Converse-body passthrough.
# Module-private: built at construction and consumed only by the internal converters.
type InferenceParams record {|
    # Sampling temperature. OPTIONAL: when unset the field is omitted from the
    # request body entirely and the model's own default applies.
    decimal temperature?;
    # Maximum tokens to generate.
    int maxTokens;
    # Provider-level stop sequences; a per-call `stop` overrides these.
    string[] stopSequences?;
    # Converse `additionalModelRequestFields` passthrough.
    AdditionalRequestFields additionalModelRequestFields?;
    # Converse `serviceTier`.
    ServiceTier serviceTier?;
    # Converse `performanceConfig.latency = "optimized"` when set.
    boolean latencyOptimized?;
    # Claude thinking. Emitted as a top-level `thinking` body field on the Anthropic
    # Messages dialects, and through `additionalModelRequestFields` on Converse
    # (which does not model it natively).
    ThinkingConfig thinking?;
    # `output_config.effort` — a SIBLING of `thinking`, never nested inside it.
    Effort effort?;
    # Converse `guardrailConfig` body field; Invoke uses headers instead.
    GuardrailConfig guardrail?;
|};

// ============================================================================
// Converter contract. `decode` returns a record, never a bare message,
// because the span/guardrail/retry all need `usage` + `stopReason`.
// ============================================================================

# Normalized token usage. Module-private — only reachable via `DecodedResponse`.
type TokenUsage record {|
    # Prompt tokens consumed.
    int inputTokens;
    # Completion tokens generated.
    int outputTokens;
|};

# What `decode` produces — more than the module-boundary message.
# Module-private: the span, guardrail signal, and retry loop read it.
type DecodedResponse record {|
    # The module invariant.
    ai:ChatAssistantMessage message;
    # Span input.
    TokenUsage usage;
    # Span + guardrail + retry signal.
    string stopReason;
    # Span input.
    string? responseId;
    # INTERVENED | NONE.
    GuardrailAction? guardrailAction;
|};

# Encode: system is hoisted out of `messages` into the signature so
# no converter can emit it as a `role: system` message. Module-private converter plumbing.
#
# Messages arrive ALREADY RESOLVED (`ResolvedMessage`): any `ai:Prompt` has been
# flattened to `ContentPart`s and any image URL fetched, by `resolveMessages`. That
# keeps every encoder pure — no network, no credentials — so the golden-file tests can
# drive them directly. See the header of content_parts.bal.
type RequestEncoder isolated function (
        string? system,
        ResolvedMessage[] messages,
        ai:ChatCompletionFunctions[] tools,
        string? stop,
        InferenceParams params) returns json|ai:Error;

# Decode: wire JSON → `DecodedResponse`. Module-private converter plumbing.
type ResponseDecoder isolated function (json response) returns DecodedResponse|ai:Error;

# An encode/decode pair. Module-private converter registry record.
type ModelConverter record {|
    # Messages → request body.
    RequestEncoder encode;
    # Wire JSON → `DecodedResponse`.
    ResponseDecoder decode;
    # How structured generation forces the single result tool on this dialect.
    # Lives on the converter because tool-choice tracks the wire shape, not the route family.
    ToolChoiceStyle toolChoice;
    # The native stream dialect this converter's route speaks, or `()` where the route
    # has no streaming decoder. Read by `runChatStream`, which refuses `chatStream`
    # before any I/O when it is `()`.
    #
    # ONE field, replacing the write-only `supportsStreaming` boolean this record
    # carried before. A flag and an id-based dialect lookup are derived independently
    # and can disagree: the lookup answered `()` for an opaque ARN — whose
    # `bareModelId` IS the ARN string, matching no vendor prefix — while the flag
    # still said `true`, so `chatStream` failed on every ARN route that `chat` served
    # fine. Hanging the dialect off the converter makes that unrepresentable:
    # `selectConverter` already resolves an ARN's dialect from `modelSchema`, and
    # whatever it picks carries its own answer.
    StreamDialect? streamDialect;
    # Body fields that turn this route's request into a STREAMING request, merged
    # into the encoded body by `runChatStream`. `()` on `bedrock-runtime`, where
    # streaming is a separate OPERATION and the body is byte-identical to the
    # buffered call; `{"stream": true}` (and whatever else the dialect hides usage
    # behind) on Mantle, which streams from the SAME path.
    #
    # On the converter rather than the endpoint because it is a property of the WIRE
    # DIALECT, not of the URL: all three Mantle converters post to `bedrock-mantle`,
    # but only the Chat Completions one asks for usage with `stream_options`.
    map<json>? streamFields = ();
|};
