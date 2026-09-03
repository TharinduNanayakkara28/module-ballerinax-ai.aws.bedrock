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

// ============================================================================
// Public enums.
// ============================================================================

# Route selection on the common config, plus the resolved wire family.
# `AUTO` (the config default) runs the resolver ladder; `CONVERSE`,
# `INVOKE`, `MANTLE` force that family. A resolved `Route.family` is never `AUTO`.
public enum ApiFamily {
    AUTO,
    CONVERSE,
    INVOKE,
    MANTLE
}

# A CONCRETE wire family — `ApiFamily` minus `AUTO`. `AUTO` is an instruction to the
# resolver ("pick one"), not a destination, so it is unrepresentable everywhere a
# family has already been decided. Module-private: a resolved `Route.family` is the
# only place a decided family is stored, and that record is internal.
type RouteFamily CONVERSE|INVOKE|MANTLE;

# How a wire dialect forces a single named tool. This is a property of
# the CONVERTER, not the route: Nova on InvokeModel is Converse-shaped, and Mistral's
# chat dialect looks OpenAI-shaped but forces tools with the bare string `"any"`.
# Deriving it from `ApiFamily` silently emits the wrong field for those dialects.
# Module-private: it lives on the internal `ModelConverter`, never on user config.
enum ToolChoiceStyle {
    # Converse: `toolConfig.toolChoice = {"tool": {"name": ...}}`.
    CONVERSE_TOOL_CHOICE,
    # Anthropic Messages: `tool_choice = {"type": "tool", "name": ...}`.
    ANTHROPIC_TOOL_CHOICE,
    # OpenAI Chat Completions: `tool_choice = {"type": "function", "function": {"name": ...}}`
    # — the function name NESTED, matching that dialect's nested `tools` entries.
    # https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_named_tool_choice_param.py
    OPENAI_CHAT_TOOL_CHOICE,
    # OpenAI Responses: `tool_choice = {"type": "function", "name": ...}` — FLAT, and
    # a different shape from Chat Completions above, mirroring this dialect's flat
    # `tools` entries. Sending the nested form here leaves the tool unforced: the
    # model answers in prose and generate() fails with "no tool call".
    # https://github.com/openai/openai-python/blob/main/src/openai/types/responses/tool_choice_function.py
    RESPONSES_TOOL_CHOICE,
    # Mistral chat completion: `tool_choice = "any"` — a bare string, and it cannot
    # name the tool, so forcing works only when exactly one tool is supplied.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-chat-completion.html
    MISTRAL_TOOL_CHOICE,
    # The dialect has no tool-calling at all (Mistral text completion), so
    # structured output is impossible on it.
    NO_TOOL_CHOICE
}

# Fields forwarded verbatim to the model — the escape hatch for anything Bedrock
# exposes that this module does not model (`top_p`, `top_k`, Nova `reasoningConfig`,
# `anthropic_beta`, prompt-caching `cache_control`, …).
#
# An OPEN record, deliberately: a field AWS ships after this release works today,
# with no module update. A closed record would make an unknown field a compile error,
# which is the opposite of what a passthrough is for.
#
# The rest type is `anydata` (the `record {}` default), so the value is NOT directly
# assignable to a `json` request body — every wire boundary goes through
# `additionalFieldsToJson`. That conversion is total: `toJson` deep-converts anydata
# that is not already json (an `xml` value becomes its string form) rather than
# failing, so there is no error path for a caller to handle.
#
# Keys for undeclared fields must be QUOTED string literals — `{"top_p": 0.9}`.
# `{top_p: 0.9}` is a compile error: identifiers cannot be used as rest-field keys.
public type AdditionalRequestFields record {
};

# How Claude allocates internal reasoning before answering.
#
# A closed, AWS-documented shape, which is why it is a record and not raw `json`:
# the wire spelling (`type`, `budget_tokens`) and the mode/budget pairing rules are
# all things a typed field can enforce at construction instead of surfacing as a 400.
public enum ThinkingMode {
    # Claude decides when and how much to think. The recommended mode, and the ONLY
    # one supported by Claude Mythos 5, Fable 5, Opus 4.7 and Mythos Preview —
    # `ENABLED`/`DISABLED` return a 400 on those. Pair with `effort` to steer depth.
    ADAPTIVE = "adaptive",
    # Manual budget via `budgetTokens`. Deprecated on Opus 4.6 / Sonnet 4.6 and
    # unsupported on the adaptive-only models above.
    ENABLED = "enabled",
    # No extended thinking.
    DISABLED = "disabled"
}

# Extended/adaptive thinking configuration for Claude.
public type ThinkingConfig record {|
    # Thinking mode. Defaults to `ADAPTIVE`, which every current Claude accepts.
    ThinkingMode mode = ADAPTIVE;
    # Reasoning-token budget. Valid ONLY with `ENABLED`, minimum 1024, and must be
    # less than `maxTokens`. All three rules are checked at construction.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html
    int budgetTokens?;
|};

# How much reasoning the model should spend, emitted as `output_config.effort`.
#
# This is a SIBLING of `thinking` on the wire, never a field inside it — AWS
# documents that nesting it under `thinking` returns a `ValidationException`. Keeping
# it a separate config field makes that mistake unrepresentable.
#
# It is also the only depth control available on the adaptive-only models
# (Mythos 5, Fable 5, Opus 4.7, Mythos Preview), where `budgetTokens` is a 400.
# https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-adaptive-thinking.html
public enum Effort {
    # Minimises thinking; may skip it entirely on simple tasks.
    EFFORT_LOW = "low",
    # Moderate thinking.
    EFFORT_MEDIUM = "medium",
    # The default. Claude always thinks.
    EFFORT_HIGH = "high",
    # Extended depth. Claude Opus 5 and Opus 4.6 ONLY.
    EFFORT_XHIGH = "xhigh",
    # No constraint on depth. Claude Opus 5 and Opus 4.6 ONLY.
    EFFORT_MAX = "max"
}

# Converse `serviceTier` passthrough.
#
# Values are AWS's, verbatim. Note there is no "standard" tier — the baseline is
# spelled `default`. The member names carry a `TIER_` prefix because a bare
# `DEFAULT` reads as a language keyword at the call site.
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html
public enum ServiceTier {
    # Baseline pay-per-token processing.
    TIER_DEFAULT = "default",
    # Higher throughput, time-based commitment.
    TIER_PRIORITY = "priority",
    # Lower cost for non-time-sensitive work.
    TIER_FLEX = "flex",
    # Dedicated throughput, term commitment.
    TIER_RESERVED = "reserved"
}

# Whether a guardrail intervened on a response.
# Module-private: only reachable via the internal `DecodedResponse`.
enum GuardrailAction {
    INTERVENED,
    NONE
}

// ============================================================================
// Route resolution result.
// ============================================================================

# A single Mantle model's wire contract. Module-private routing-table data.
#
# The PATH is the only per-model fact here. Converter and auth-header style used to be
# stored alongside it and are now DERIVED from the path (see `mantleConverterForPath`
# and `usesApiKeyHeader`), because path → dialect is 1:1 across every model AWS
# serves on Mantle. They are not derivable from the VENDOR prefix, which is the
# mistake this shape invites: `google.gemma-3-*` speaks Chat Completions on `/v1`
# while `google.gemma-4-*` speaks Responses on `/openai/v1` — one prefix, two
# dialects. Keying off the path keeps that distinction intact with no extra fields.
type MantleEntry record {|
    # Request path on the `bedrock-mantle` host, e.g. `/anthropic/v1/messages`.
    string path;

    # Whether this model is ALSO served on `bedrock-runtime` (Converse/InvokeModel).
    #
    # Drives the `generate()` fallback: under `AUTO` a Mantle-capable model routes
    # chat to Mantle, which has no structured output — but when the same model is on
    # `bedrock-runtime`, a typed `generate()` can quietly use Converse instead of
    # failing. A Mantle-ONLY model (GPT-5.x, Mythos, Gemma 4) has no such route, so it
    # keeps the clean error rather than being sent to an endpoint that does not serve
    # it. Verified per model against AWS's API-compatibility matrix.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/models-api-compatibility.html
    boolean onRuntime = false;

    # The id to put on the wire when it DIFFERS from the `bedrock-runtime` id.
    #
    # For most models the two endpoints share an id, and this is omitted. But some
    # models are published under different ids per endpoint — gpt-oss is
    # `openai.gpt-oss-120b-1:0` on `bedrock-runtime` and `openai.gpt-oss-120b` on
    # `bedrock-mantle`. Without this, forcing Mantle sends the runtime id and fails.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-120b.html
    string modelId?;
|};

# The fully resolved route — produced once by `resolveRoute` at construction.
# Everything downstream (endpoint, converter, transport) reads from this.
# Module-private: the resolver's output, mirroring the private `Endpoint`.
type Route record {|
    # The resolved wire dialect. Never `AUTO` — see `RouteFamily`.
    RouteFamily family;
    # Lookup key with any CRIS geo prefix stripped, e.g. `anthropic.claude-opus-4-8`.
    string bareModelId;
    # The stripped CRIS geo prefix, re-applied per family on the wire.
    string? geoPrefix;
    # The id that goes on the wire — family-specific (CRIS-prefixed for Converse/
    # Invoke, bare for Mantle, or the raw ARN for opaque ARNs).
    string effectiveModelId;
    # Region. An ARN's region segment overrides `config.region`.
    string region;
    # Partition: `aws` | `aws-cn` | `aws-us-gov`.
    string partition;
    # Present only for MANTLE routes.
    MantleEntry? mantleEntry;
|};

// ============================================================================
// Internal route-resolution input — the routing slice of a vendor config.
// ============================================================================

# The subset of a vendor `*Config` that `resolveRoute` (pure) needs. Each vendor
# `init` builds this from its own config record, keeping the resolver decoupled
# from the seven per-vendor config shapes. Module-private input.
type RouteConfig record {|
    # Explicit route override — outranks every heuristic.
    ApiFamily apiFamily?;
|};
