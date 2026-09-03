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

// Routing tables — DATA, not code. A new model for a listed vendor is reachable by
// extending these tables with no other code change.
//
// A new CONVERSE/INVOKE model needs no table entry at all: pass its id as a string
// and the resolver sinks it to Converse. Only a new MANTLE model needs a row here,
// because its path cannot be derived — and a module release is the cost of that,
// same as every other Ballerina model provider.

// There is deliberately NO default temperature. Anthropic deprecated sampling
// parameters on Claude 4.7 and later, and OpenAI's GPT-5.x reasoning models never
// accepted them — on those models any value at all is a 400, so a module default
// would make seven of this package's flagship model ids unusable out of the box.
// Unset means the key is absent from the request and the model's own default
// applies. See `setTemperature` in converter_common.bal.

// 512 was too low to be a safe default: on adaptive-thinking models (Claude 4.7+,
// Sonnet 5, Opus 5) thinking tokens count against this ceiling, so the response
// routinely stopped with `max_tokens` before producing any text — which reads as a
// module bug, not a config problem. 4096 leaves room for a thinking pass plus an
// answer while staying under the tightest per-model output cap in the supported set
// (Nova Pro/Lite/Micro are capped at 5K output tokens, so 8192 would be rejected
// outright on AmazonModelProvider).
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-pro.html
const int DEFAULT_MAX_TOKEN_COUNT = 4096;

// Cross-region-inference geo prefixes, stripped for lookup then re-applied per
// family on the wire. List copied from LiteLLM's cross-region
// inference regions helper — includes `us-gov`, which a hand-rolled list would miss.
// https://docs.aws.amazon.com/bedrock/latest/userguide/global-cross-region-inference.html
// Anthropic's documented floor for a manual thinking budget.
// https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html
const int MIN_THINKING_BUDGET_TOKENS = 1024;

final readonly & string[] CRIS_PREFIXES = ["global", "us", "eu", "apac", "jp", "au", "us-gov"];

// WHAT a model needs to speak Mantle — every Mantle-capable model, dual-endpoint
// or not. This is now THE table that drives AUTO routing: the preference order is
// MANTLE → CONVERSE → INVOKE, so membership here
// means a bare id resolves to Mantle by default (as well as when forced). Because
// membership requires a verified path/auth/converter, an unknown model is still absent
// and sinks to Converse — never Mantle by elimination.
// The PATH is the per-model datum; deriving it from the `openai.` prefix would
// break the moment AWS ships an `openai.*` model on a different path. Converter and
// auth-header style are DERIVED from the path (`mantleConverterForPath` /
// `usesApiKeyHeader`) because path → dialect is 1:1.
// https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
final readonly & map<MantleEntry> MANTLE_CAPABLE = {
    // GPT-5.x use `/openai/v1/responses`, distinct from the `/v1/responses` other
    // models use — every card below states this in its own Note.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-55.html
    "openai.gpt-5.5": {path: "/openai/v1/responses"},
    "openai.gpt-5.4": {path: "/openai/v1/responses"},
    // GPT-5.6 (launched 2026-07-13). Each id verified against its own card: Mantle
    // only, Responses YES / Chat Completions NO, path `/openai/v1`.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html
    "openai.gpt-5.6-sol": {path: "/openai/v1/responses"},
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-terra.html
    "openai.gpt-5.6-terra": {path: "/openai/v1/responses"},
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-luna.html
    "openai.gpt-5.6-luna": {path: "/openai/v1/responses"},
    // Anthropic Messages on Mantle: `anthropic-version: 2023-06-01` header.
    // Default auth header X_API_KEY per AWS's documented curl.
    "anthropic.claude-mythos-preview": {path: "/anthropic/v1/messages"},
    // Mantle-only, Messages API (Converse/Invoke/Responses all NO).
    // NOTE: this card's sample uses the Anthropic SDK with AWS_BEARER_TOKEN_BEDROCK,
    // which does not settle the wire header — so it keeps X_API_KEY for consistency
    // with mythos-preview above. Still unresolved; one live call settles it for both.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-mythos-5.html
    "anthropic.claude-mythos-5": {path: "/anthropic/v1/messages"},
    "anthropic.claude-haiku-4-5": {onRuntime: true, path: "/anthropic/v1/messages"},
    "zai.glm-5": {onRuntime: true, path: "/v1/chat/completions"},
    // --- Dual-endpoint models (bedrock-runtime YES + bedrock-mantle YES). ---
    // These DEFAULT to Mantle under AUTO (preference MANTLE →
    // CONVERSE → INVOKE); pass `apiFamily = CONVERSE` (or a `converse/` prefix) to use
    // the runtime surface instead — which `generate()` with a typed target requires,
    // since Mantle has no structured output.
    //
    // Card: bedrock-runtime YES + bedrock-mantle YES; Messages API YES,
    // Responses/Chat Completions NO; Mantle URL `/anthropic/v1/messages`.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-8.html
    "anthropic.claude-opus-4-8": {onRuntime: true, path: "/anthropic/v1/messages"},
    // Opus 5 (launched 2026-07-24) is dual-homed: bedrock-runtime YES +
    // bedrock-mantle YES; Messages YES, Responses NO, Chat Completions NO. Same id on
    // both endpoints, Mantle URL `/anthropic/v1/messages`. X_API_KEY for consistency
    // with the Anthropic entries above (the card's sample uses the Anthropic SDK with
    // AWS_BEARER_TOKEN_BEDROCK, which does not state the wire header).
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-5.html
    "anthropic.claude-opus-5": {onRuntime: true, path: "/anthropic/v1/messages"},
    // Sonnet 5 is dual-homed like opus-4-8: bedrock-runtime YES + bedrock-mantle YES,
    // Messages API on `/anthropic/v1/messages`. Like every entry in this block it
    // resolves to Mantle under AUTO; pass `apiFamily = CONVERSE` for
    // the runtime surface.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5.html
    "anthropic.claude-sonnet-5": {onRuntime: true, path: "/anthropic/v1/messages"},
    // gpt-oss is published under DIFFERENT IDS PER ENDPOINT — `-1:0` on
    // bedrock-runtime, bare on bedrock-mantle — hence `modelId`. Mantle base URL is
    // `/v1` (not `/openai/v1` like GPT-5.x), and it serves both Responses and Chat
    // Completions; we take Chat Completions.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-120b.html
    "openai.gpt-oss-120b-1:0": {
        onRuntime: true,
        path: "/v1/chat/completions",
        modelId: "openai.gpt-oss-120b"
    },
    // DeepSeek V3.2 — dual-homed; Mantle serves Chat Completions on `/v1`, same id on
    // both endpoints, BEARER (card sample sets OPENAI_API_KEY against the `/v1` base).
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html
    "deepseek.v3.2": {onRuntime: true, path: "/v1/chat/completions"},
    // Mistral Large 3 — dual-homed; Mantle Chat Completions on `/v1`, same id both
    // endpoints, BEARER.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-mistral-large-3.html
    "mistral.mistral-large-3-675b-instruct":
        {onRuntime: true, path: "/v1/chat/completions"},
    // Qwen3 Coder 480B — dual-homed; DIFFERENT id per endpoint (`-v1:0` on runtime,
    // `-instruct` on Mantle, like gpt-oss), so `modelId` overrides the wire id. Mantle
    // Chat Completions on `/v1`, BEARER.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-qwen-qwen3-coder-480b-a35b-instruct.html
    "qwen.qwen3-coder-480b-a35b-v1:0": {
        onRuntime: true,
        path: "/v1/chat/completions",
        modelId: "qwen.qwen3-coder-480b-a35b-instruct"
    },
    // Qwen3 32B — dual-homed; DIFFERENT id per endpoint (`-v1:0` on runtime, bare
    // `qwen.qwen3-32b` on Mantle), Chat Completions on `/v1`, BEARER. In-Region YES.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-qwen-qwen3-32b.html
    "qwen.qwen3-32b-v1:0": {
        onRuntime: true,
        path: "/v1/chat/completions",
        modelId: "qwen.qwen3-32b"
    },
    // Gemma 4 is Mantle-ONLY. Its card's support matrix marks bedrock-runtime,
    // Converse, Invoke and Messages all NO, and states: "Gemma 4 models are
    // available only on the `bedrock-mantle` endpoint. This model is available on
    // the `openai/v1/responses` path ... different from the `v1/responses` path
    // used by other models." Its Programmatic Access table gives exactly one row:
    // bedrock-mantle | google.gemma-4-31b | https://bedrock-mantle.{region}.api.aws/openai/v1
    // Auth is BEARER: the card's sample sets OPENAI_API_KEY against that base URL.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-4-31b.html
    // Each id below was checked against its OWN card — same matrix, same Note, and
    // a Programmatic Access table with exactly one row (bedrock-mantle, /openai/v1).
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-4-31b.html
    "google.gemma-4-31b": {path: "/openai/v1/responses"},
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-4-e2b.html
    "google.gemma-4-e2b": {path: "/openai/v1/responses"},
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-4-26b-a4b.html
    "google.gemma-4-26b-a4b": {path: "/openai/v1/responses"},
    // Gemma 3 (dual-homed) — CRUCIALLY DIFFERENT from Gemma 4: its Mantle rows sit on
    // `/v1` with Chat Completions, NOT `/openai/v1` + Responses. So one vendor prefix
    // (`google.`) spans two Mantle path families, which is exactly why the path is
    // per-model table data and never derived from the prefix. Each id verified against
    // its own card: same id on both endpoints, `/v1` Chat Completions, BEARER,
    // In-Region YES in us-east-1.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-3-27b-pt.html
    "google.gemma-3-27b-it": {onRuntime: true, path: "/v1/chat/completions"},
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-3-12b-it.html
    "google.gemma-3-12b-it": {onRuntime: true, path: "/v1/chat/completions"},
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-3-4b-it.html
    "google.gemma-3-4b-it": {onRuntime: true, path: "/v1/chat/completions"}
};

// NOTE: there is no separate `MANTLE_DEFAULT` list. It
// once held only the Mantle-ONLY models, because dual-endpoint models defaulted to
// Converse. That default is now flipped — any Mantle-CAPABLE model prefers Mantle
// under AUTO — so `MANTLE_CAPABLE` membership alone now decides the default, and a
// second list would only drift out of sync. There is likewise no `CONVERSE_MODELS`
// allowlist: everything absent from `MANTLE_CAPABLE` (and every geo-prefixed id)
// sinks to Converse, so an unknown id can never reach Mantle by elimination.
