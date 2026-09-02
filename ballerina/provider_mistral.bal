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

// MistralModelProvider — Converse (default) + Invoke-Mistral (CLAUDE.md §3).
//
// On the INVOKE route Mistral speaks two incompatible dialects picked by model id
// (see `usesMistralTextDialect` in codecs.bal). The Converse default hides this;
// it only matters when forcing `apiFamily = INVOKE`.

# Well-known Mistral model ids. Any newer id can be passed as a `string`.
public enum MistralModel {
    # Mistral Large 3 — the current flagship (675B, coding/reasoning/multilingual,
    # 256K context). Chat-completion dialect on InvokeModel (`messages`/`choices`)
    # and Converse — the same dialect as 24.07, NOT the 24.02 text template. In-Region
    # callable (us-east-1 etc.); Geo/Global not supported. Same id on both endpoints;
    # dual-homed with bedrock-mantle, defaults to Converse here.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-mistral-ai-mistral-large-3.html
    MISTRAL_LARGE_3 = "mistral.mistral-large-3-675b-instruct",
    # Chat-completion dialect on InvokeModel (`messages`/`choices`), and Converse.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-large-2407.html
    MISTRAL_LARGE_2407 = "mistral.mistral-large-2407-v1:0",
    # TEXT-completion dialect on InvokeModel (`prompt`/`outputs`) — note this is the
    # opposite dialect to its 24.07 sibling above, despite the shared family name.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_MistralAi_section.html
    MISTRAL_LARGE_2402 = "mistral.mistral-large-2402-v1:0",
    # TEXT-completion dialect on InvokeModel; no tool-calling.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-text-completion.html
    MISTRAL_7B_INSTRUCT = "mistral.mistral-7b-instruct-v0:2"
}

# Mistral-specific configuration (CLAUDE.md §3).
public type MistralConfig record {|
    *CommonModelConfig;
|};

# Mistral models on AWS Bedrock.
public isolated distinct client class MistralModelProvider {
    *ai:ModelProvider;

    private final ApiFamily family;
    private final string wireModelId;
    private final readonly & ModelCodec codec;
    private final BedrockTransport transport;
    private final readonly & InferenceParams params;
    private final map<string> & readonly extraHeaders;
    private final boolean supportsStructuredOutput;

    # + credentials - Static keys, STS, or a Bedrock API key (§9.5)
    # + model - A Mistral id (bare, CRIS-prefixed, ARN, or route-prefixed)
    # + region - Default region; an ARN `model`'s region segment overrides it (§5.2)
    # + maxTokens - Maximum tokens to generate
    # + temperature - Sampling temperature. Leave unset (the default) to omit the
    #                 field entirely and use the model's own default — several current
    #                 models reject it outright
    # + config - Routing overrides, guardrails, Converse passthrough
    # + return - `nil` on success; otherwise an `ai:Error`
    public isolated function init(
            @display {label: "AWS Credentials"} BedrockCredentials credentials,
            @display {label: "Model"} MistralModel|string model,
            @display {label: "Region"} string region,
            @display {label: "Maximum Tokens"} int? maxTokens = DEFAULT_MAX_TOKEN_COUNT,
            @display {label: "Temperature"} decimal? temperature = (),
            @display {label: "Configuration"} *MistralConfig config)
            returns ai:Error? {
        RouteConfig routeConfig = {
            apiFamily: config.apiFamily,
            modelSchema: config.modelSchema,
            routeOverrides: config.routeOverrides
        };
        [Route, readonly & ModelCodec, BedrockTransport] [route, codec, transport] =
            check resolveSpine("MistralModelProvider", credentials, model, region, routeConfig,
                config?.signingServiceName, config?.httpConfig, config?.retryConfig, config?.guardrail);

        self.family = route.family;
        self.wireModelId = route.effectiveModelId;
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
        => runChat("Mistral", self.family, self.wireModelId, self.codec, self.transport,
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
    # Supported on EVERY route this module resolves — the wire differs, the contract
    # does not: `ConverseStream` on Converse, `InvokeModelWithResponseStream` on
    # Invoke (both AWS event-stream framed), and SSE on Mantle, where the stream is
    # asked for with a body flag on the same path.
    #
    # Streaming on `bedrock-runtime` needs the SEPARATE
    # `bedrock:InvokeModelWithResponseStream` IAM action — `ConverseStream` included —
    # so a role that can `chat()` may still be denied here. On Mantle the usual
    # `bedrock-mantle:CreateInference` covers it.
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
        => runChatStream("Mistral", self.family, self.wireModelId, self.codec,
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
