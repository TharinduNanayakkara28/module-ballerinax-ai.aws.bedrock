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
import ballerina/jballerina.java;

// GoogleModelProvider — Gemma (open-weight). Gemini is NOT on Bedrock (CLAUDE.md §3).
//
// CLAUDE.md §3 asked which endpoint serves Gemma. The answer is BOTH, and it splits
// by generation — the per-model cards are the only authority:
//
//   Gemma 3 — dual-homed. Its cards tick bedrock-runtime AND bedrock-mantle, so
//             the module routes it to Converse (the richer surface). Note AWS's
//             cards say "whenever possible, we recommend you use the bedrock-mantle
//             endpoint"; Converse is supported, so this is a deliberate choice.
//   Gemma 4 — bedrock-mantle ONLY. Its card marks bedrock-runtime / Converse /
//             Invoke / Messages all NO, and serves it from `/openai/v1/responses`.
//             So it signs `bedrock-mantle`, and it CANNOT do structured output.
//
// The routing lives in MANTLE_CAPABLE (constants.bal), not here.
//
// HISTORY: this file previously claimed Gemma was on bedrock-runtime because it
// appeared "in no Mantle table". That is routing by elimination — the very
// inference the design forbids (§11 principle 2) — and it was wrong for Gemma 4.
// A model's absence from a table is never evidence of its endpoint.
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-4-31b.html
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-google-gemma-3-27b-pt.html

# Well-known Google Gemma model ids. Any newer id can be passed as a `string`.
public enum GoogleModel {
    # Converse on `bedrock-runtime` (also served on Mantle).
    GEMMA_3_4B_IT = "google.gemma-3-4b-it",
    # Converse on `bedrock-runtime` (also served on Mantle).
    GEMMA_3_12B_IT = "google.gemma-3-12b-it",
    # Converse on `bedrock-runtime` (also served on Mantle). AWS titles this card
    # "Gemma 3 27B PT" but its id really is `-it` — do not "correct" this.
    GEMMA_3_27B_IT = "google.gemma-3-27b-it",
    # `bedrock-mantle` ONLY — structured output is not available on this route.
    GEMMA_4_E2B = "google.gemma-4-e2b",
    # `bedrock-mantle` ONLY — structured output is not available on this route.
    GEMMA_4_26B_A4B = "google.gemma-4-26b-a4b",
    # `bedrock-mantle` ONLY — structured output is not available on this route.
    GEMMA_4_31B = "google.gemma-4-31b"
}

# Google-specific configuration (CLAUDE.md §3).
public type GoogleConfig record {|
    *CommonModelConfig;
|};

# Google Gemma models on AWS Bedrock. Gemma 3 routes to Converse on
# `bedrock-runtime`; Gemma 4 is served only on `bedrock-mantle` and therefore
# supports neither structured output nor guardrails.
public isolated distinct client class GoogleModelProvider {
    *ai:ModelProvider;

    private final ApiFamily family;
    private final string wireModelId;
    // Kept alongside `wireModelId` for streaming: the stream DIALECT is chosen from
    // the bare (geo-prefix-stripped) id, which `wireModelId` no longer carries.
    private final string bareModelId;
    private final readonly & ModelCodec codec;
    private final BedrockTransport transport;
    private final readonly & InferenceParams params;
    private final map<string> & readonly extraHeaders;
    private final boolean supportsStructuredOutput;

    # + credentials - Static keys, STS, or a Bedrock API key (§9.5)
    # + model - A Gemma id (bare, CRIS-prefixed, ARN, or route-prefixed)
    # + region - Default region; an ARN `model`'s region segment overrides it (§5.2)
    # + maxTokens - Maximum tokens to generate
    # + temperature - Sampling temperature. Leave unset (the default) to omit the
    #                 field entirely and use the model's own default — several current
    #                 models reject it outright
    # + config - Routing overrides, guardrails, Converse passthrough
    # + return - `nil` on success; otherwise an `ai:Error`
    public isolated function init(
            @display {label: "AWS Credentials"} BedrockCredentials credentials,
            @display {label: "Model"} GoogleModel|string model,
            @display {label: "Region"} string region,
            @display {label: "Maximum Tokens"} int? maxTokens = DEFAULT_MAX_TOKEN_COUNT,
            @display {label: "Temperature"} decimal? temperature = (),
            @display {label: "Configuration"} *GoogleConfig config)
            returns ai:Error? {
        RouteConfig routeConfig = {
            apiFamily: config.apiFamily,
            modelSchema: config.modelSchema,
            routeOverrides: config.routeOverrides
        };
        [Route, readonly & ModelCodec, BedrockTransport] [route, codec, transport] =
            check resolveSpine("GoogleModelProvider", credentials, model, region, routeConfig,
                config?.signingServiceName, config?.httpConfig, config?.retryConfig, config?.guardrail);

        self.family = route.family;
        self.wireModelId = route.effectiveModelId;
        self.bareModelId = route.bareModelId;
        self.codec = codec;
        self.transport = transport;
        self.supportsStructuredOutput = route.family != MANTLE; // amendment
        self.params = buildInferenceParams(maxTokens, temperature, config?.stopSequences,
            config?.additionalModelRequestFields, config?.additionalModelResponseFieldPaths,
            config?.serviceTier, config?.latencyOptimized, config?.requestMetadata, config?.guardrail);
        self.extraHeaders = commonExtraHeaders(route, config?.guardrail, credentials).cloneReadOnly();
    }

    # + messages - Chat messages or a single user message
    # + tools - Tool definitions for function calling
    # + stop - Stop sequence; overrides configured `stopSequences` (§7)
    # + return - The assistant message, or an `ai:Error`
    isolated remote function chat(ai:ChatMessage[]|ai:ChatUserMessage messages,
            ai:ChatCompletionFunctions[] tools = [], string? stop = ())
            returns ai:ChatAssistantMessage|ai:Error
        => runChat("Google", self.family, self.wireModelId, self.codec, self.transport,
            self.extraHeaders, self.params, messages, tools, stop);

    # + prompt - The prompt to use in the chat request
    # + td - Type descriptor of the expected return type
    # + return - A value of the expected type, or an `ai:Error`
    isolated remote function generate(ai:Prompt prompt,
            @display {label: "Expected type"} typedesc<anydata> td = <>)
            returns td|ai:Error = @java:Method {
        'class: "io.ballerina.lib.ai.aws.bedrock.Generator"
    } external;

    # Sends a chat request and streams the reply as normalized chunks.
    #
    # Supported on the Converse route for every vendor (`ConverseStream` is
    # model-agnostic) and on the Invoke route for Claude and Nova. Any other route
    # returns an `ai:Error` naming `apiFamily = CONVERSE` as the remedy.
    #
    # NOT `isolated`, unlike `chat`: the returned stream is backed by an iterator
    # carrying mutable framing state. `ai:ModelProvider` does not declare it
    # isolated either.
    #
    # + messages - Chat messages or a single user message
    # + tools - Tool definitions for function calling
    # + stop - Stop sequence; overrides configured `stopSequences`
    # + return - A stream of response chunks, or an `ai:Error`
    remote function chatStream(ai:ChatMessage[]|ai:ChatUserMessage messages,
            ai:ChatCompletionFunctions[] tools = [], string? stop = ())
            returns stream<ai:ChatCompletionChunk, ai:Error?>|ai:Error
        => runChatStream("Google", self.family, self.wireModelId, self.bareModelId, self.codec,
            self.transport, self.extraHeaders, self.params, messages, tools, stop);

    # Streams a generated value as it is produced. Only a `string` target type is
    # supported — structured output is obtained by forcing a tool call, whose
    # arguments cannot be bound until the whole JSON has arrived.
    #
    # External Java per the platform convention, as `generate`: a dependently typed
    # function must be external. The shim calls back into
    # `generateLlmResponseStream`, which projects `chatStream`'s chunks onto text.
    #
    # + prompt - The prompt to use in the chat request
    # + td - Type descriptor of the expected return type
    # + return - A stream of text fragments, or an `ai:Error`
    remote function generateStream(ai:Prompt prompt,
            @display {label: "Expected type"} typedesc<anydata> td = <>)
            returns stream<td, ai:Error?>|ai:Error = @java:Method {
        'class: "io.ballerina.lib.ai.aws.bedrock.StreamGenerator"
    } external;
}
