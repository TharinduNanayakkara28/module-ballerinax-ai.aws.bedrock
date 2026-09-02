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

// ============================================================================
// Credentials — design §9.5. A union; bearer (Bedrock API key) is first-class
// on both endpoints (https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html).
// ============================================================================

# Long-lived IAM access keys.
public type StaticCredentials record {|
    # AWS access key id.
    string accessKeyId;
    # AWS secret access key.
    string secretAccessKey;
|};

# Temporary STS credentials — the required `sessionToken` distinguishes this from
# `StaticCredentials` and is emitted as `X-Amz-Security-Token` (design §9.5).
public type StsCredentials record {|
    # AWS access key id.
    string accessKeyId;
    # AWS secret access key.
    string secretAccessKey;
    # STS session token, sent as `X-Amz-Security-Token`.
    string sessionToken;
|};

# A Bedrock API key (bearer token) — first-class on both endpoints (§9.5).
public type BearerToken record {|
    # The Bedrock API key, sent as `Authorization: Bearer`.
    string apiKey;
|};

# The credential union accepted by every provider (design §9.5).
public type BedrockCredentials StaticCredentials|StsCredentials|BearerToken;

// ============================================================================
// Guardrails / retry — design §9.5.
// ============================================================================

# Guardrail configuration. Placement is route-specific (design §9.5): Converse
# body field, Invoke headers, and a construction error on Mantle.
public type GuardrailConfig record {|
    # `guardrailIdentifier` (Converse body / `X-Amzn-Bedrock-GuardrailIdentifier`).
    string guardrailIdentifier;
    # `guardrailVersion`.
    string guardrailVersion;
    # Optional `trace` mode (`enabled` | `disabled` | `enabled_full`).
    string trace?;
|};

# Retry policy for the transport's throttling/warm-up backoff (design §9.5).
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
// Shared model config — design §10. Each vendor `*Config` includes this and adds
// only its vendor-specific fields (CLAUDE.md §3).
// ============================================================================

# Everything that is not the model's identity, shared across vendors (design §10).
public type CommonModelConfig record {|
    // --- Routing (§5) ---
    # Route selection (amendment): `AUTO` (default) runs the resolver; `CONVERSE`/
    # `INVOKE`/`MANTLE` force that family. The escape hatch — outranks every heuristic.
    ApiFamily apiFamily = AUTO;
    # REQUIRED for `imported-model/` ARNs.
    ModelSchema modelSchema?;
    # Extends the built-in routing tables without waiting for a module release.
    #
    # AWS adds models faster than this module can ship. When Bedrock exposes a model
    # the tables in `constants.bal` do not know about, `resolveRoute` consults this
    # map at step 3 — before every built-in table — so a new model becomes reachable
    # with no code change.
    #
    # The KEY is the bare model id, after any cross-region-inference geo prefix is
    # stripped: use `anthropic.claude-x`, never `us.anthropic.claude-x`. For an ARN,
    # the key is the full ARN string. The VALUE takes one of two shapes:
    #
    # - An `ApiFamily` (`CONVERSE` | `INVOKE`) routes the model to `bedrock-runtime`
    #   on that family:
    #   `routeOverrides = {"anthropic.claude-x": CONVERSE}`
    #
    # - A `MantleEntry` routes it to `bedrock-mantle`, and must carry the per-model
    #   path, because a Mantle path is DATA — it is not derivable from the vendor
    #   prefix (GPT-5.5 is on `/openai/v1/responses` while other OpenAI models are
    #   not). `authHeader` selects `x-api-key` or `Authorization: Bearer`; `codec`
    #   selects the request/response dialect for that path:
    #   `routeOverrides = {"openai.gpt-x": {path: "/openai/v1/responses",
    #                                       authHeader: BEARER, codec: RESPONSES_CODEC}}`
    #
    # This outranks `MANTLE_CAPABLE` and the vendor prefix tables, but NOT `apiFamily`
    # — an explicit `apiFamily` is step 1 and still wins. Note that a Mantle-routed
    # model has `supportsStructuredOutput = false`, so a typed `generate()` errors on
    # it; see the `apiFamily` docs above.
    map<ApiFamily|MantleEntry> routeOverrides?;
    # Per-route SigV4 signing name override (§9.4).
    string signingServiceName?;

    // --- Inference (§7) ---
    # Provider-level stop sequences; a per-call `stop` overrides these (§7).
    string[] stopSequences?;

    // --- Converse passthrough (§9.3) ---
    # Forwarded verbatim on Converse; ignored elsewhere.
    json additionalModelRequestFields?;
    # JSON Pointers (max 10) whose values return in `additionalModelResponseFields`.
    string[] additionalModelResponseFieldPaths?;
    # `serviceTier`.
    ServiceTier serviceTier?;
    # Request latency-optimized inference on Converse: routes the call onto AWS's
    # faster serving path (custom silicon / reserved capacity) for a lower
    # time-to-first-token, at a higher price. Same output — a speed/cost dial only.
    # Emitted as `performanceConfig.latency = "optimized"`; unset means `standard`.
    # Support is per model and region; unsupported combinations are rejected by AWS.
    # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
    boolean latencyOptimized?;
    # `requestMetadata` (max 16 pairs).
    map<string> requestMetadata?;

    // --- Cross-cutting (§9.5) ---
    # Guardrail; construction error on a MANTLE route (§9.5).
    GuardrailConfig guardrail?;
    # Retry policy.
    RetryConfig retryConfig?;
    # Underlying HTTP client configuration.
    http:ClientConfiguration httpConfig?;
|};

// ============================================================================
// Inference params — resolved once at construction (design §6, §7). Carries the
// Converse-body passthrough so the fixed codec signature can emit it; non-Converse
// codecs ignore the extra fields.
// ============================================================================

# Resolved inference parameters plus Converse-body passthrough (design §7, §9.3).
# Module-private: built at construction and consumed only by the internal codecs.
type InferenceParams record {|
    # Sampling temperature. OPTIONAL: when unset the field is omitted from the
    # request body entirely and the model's own default applies (§7).
    decimal temperature?;
    # Maximum tokens to generate.
    int maxTokens;
    # Provider-level stop sequences; a per-call `stop` overrides these (§7).
    string[] stopSequences?;
    # Converse `additionalModelRequestFields` passthrough (§9.3).
    json additionalModelRequestFields?;
    # Converse `additionalModelResponseFieldPaths` (§9.3).
    string[] additionalModelResponseFieldPaths?;
    # Converse `serviceTier` (§9.3).
    ServiceTier serviceTier?;
    # Converse `performanceConfig.latency = "optimized"` when set (§9.3).
    boolean latencyOptimized?;
    # Converse `requestMetadata`, max 16 pairs (§9.3).
    map<string> requestMetadata?;
    # Converse `guardrailConfig` body field (§9.5); Invoke uses headers instead.
    GuardrailConfig guardrail?;
|};

// ============================================================================
// Codec contract — design §7. `decode` returns a record, never a bare message,
// because the span/guardrail/retry all need `usage` + `stopReason` (§3.2, §7).
// ============================================================================

# Normalized token usage (design §7). Module-private — only reachable via `DecodedResponse`.
type TokenUsage record {|
    # Prompt tokens consumed.
    int inputTokens;
    # Completion tokens generated.
    int outputTokens;
|};

# What `decode` produces — more than the module-boundary message (design §7).
# Module-private (CLAUDE.md §3): the span, guardrail signal, and retry loop read it.
type DecodedResponse record {|
    # The module invariant (§3).
    ai:ChatAssistantMessage message;
    # Span input (§3.2).
    TokenUsage usage;
    # Span + guardrail (§9.5) + retry (§8) signal.
    string stopReason;
    # Span input.
    string? responseId;
    # INTERVENED | NONE (§9.5).
    GuardrailAction? guardrailAction;
    # Converse-only response passthrough (§9.3).
    json? additionalModelResponseFields;
|};

# Encode: system is hoisted out of `messages` into the signature (design §7.1) so
# no codec can emit it as a `role: system` message. Module-private codec plumbing.
type RequestCodec isolated function (
        ai:ChatSystemMessage? system,
        ai:ChatMessage[] messages,
        ai:ChatCompletionFunctions[] tools,
        string? stop,
        InferenceParams params) returns json|ai:Error;

# Decode: wire JSON → `DecodedResponse` (design §7). Module-private codec plumbing.
type ResponseCodec isolated function (json response) returns DecodedResponse|ai:Error;

# An encode/decode pair (design §7). Module-private codec registry record.
type ModelCodec record {|
    # Messages → request body.
    RequestCodec encode;
    # Wire JSON → `DecodedResponse`.
    ResponseCodec decode;
    # How structured generation forces the single result tool on this dialect (§8).
    # Lives on the codec because tool-choice tracks the wire shape, not the route family.
    ToolChoiceStyle toolChoice;
    # The native stream dialect this codec's route speaks, or `()` where the route
    # has no streaming decoder. Read by `runChatStream`, which refuses `chatStream`
    # before any I/O when it is `()`.
    #
    # ONE field rather than a `supportsStreaming` flag beside a separate id-based
    # dialect lookup. The two were derived independently and could disagree: a
    # lookup keyed on the model id answered `()` for an opaque ARN — whose
    # `bareModelId` IS the ARN string, matching no vendor prefix — while the flag
    # still said `true`, so `chatStream` failed on every ARN route that `chat`
    # served fine. Hanging the dialect off the codec makes that state
    # unrepresentable: `selectCodec` already resolves an ARN's dialect from
    # `modelSchema`, and whatever it picks carries its own answer.
    StreamDialect? streamDialect;
    # Body fields that turn this route's request into a STREAMING request, merged
    # into the encoded body by `runChatStream`. `()` on `bedrock-runtime`, where
    # streaming is a separate OPERATION and the body is byte-identical to the
    # buffered call; `{"stream": true}` (and whatever else the dialect hides usage
    # behind) on Mantle, which streams from the SAME path.
    #
    # On the codec rather than the endpoint because it is a property of the WIRE
    # DIALECT, not of the URL: all three Mantle codecs post to `bedrock-mantle`, but
    # only the Chat Completions one has to ask for usage with `stream_options`.
    map<json>? streamFields = ();
|};
