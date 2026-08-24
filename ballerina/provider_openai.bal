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

// OpenAIModelProvider — GPT-5.x via Mantle Responses (`/openai/v1/responses`),
// GPT-OSS via Converse/Invoke (CLAUDE.md §3).

# Well-known OpenAI model ids on Bedrock. Any newer id can be passed as a `string`.
public enum OpenAIModel {
    // Mantle-only (design §5.5) — these do not exist on bedrock-runtime.
    # `bedrock-mantle` only, Responses API. No structured output.
    GPT_5_6_SOL = "openai.gpt-5.6-sol",
    # `bedrock-mantle` only, Responses API. No structured output.
    GPT_5_6_TERRA = "openai.gpt-5.6-terra",
    # `bedrock-mantle` only, Responses API. No structured output.
    GPT_5_6_LUNA = "openai.gpt-5.6-luna",
    # `bedrock-mantle` only, Responses API. No structured output.
    GPT_5_5 = "openai.gpt-5.5",
    # `bedrock-mantle` only, Responses API. No structured output.
    GPT_5_4 = "openai.gpt-5.4",
    // Open-weight, on bedrock-runtime.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-120b.html
    GPT_OSS_120B = "openai.gpt-oss-120b-1:0",
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-oss-20b.html
    GPT_OSS_20B = "openai.gpt-oss-20b-1:0"
}

# OpenAI-specific configuration (CLAUDE.md §3).
public type OpenAIConfig record {|
    *CommonModelConfig;
    # `reasoning_effort` — trades latency and token cost against reasoning depth.
    # Forwarded verbatim via the §9.3 passthrough, so the ACCEPTED VALUES ARE NOT
    # THE SAME for the two families this provider serves, and an unsupported value
    # is rejected by the endpoint rather than caught here:
    #
    # - GPT-OSS on `bedrock-runtime` (`GPT_OSS_120B`, `GPT_OSS_20B`):
    #   `low` | `medium` | `high`.
    # - GPT-5.x on `bedrock-mantle` (`GPT_5_4`, `GPT_5_5`, `GPT_5_6_*`): the
    #   Responses API set, which also includes `minimal` on some cards and drops
    #   values on others — check the model card for the id you are using.
    #
    # Left as a `string` rather than an enum precisely because the two sets differ
    # and both move; leave it unset to use the model's default.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-openai.html
    string reasoningEffort?;
|};

# OpenAI models on AWS Bedrock (GPT-5.x on Mantle, GPT-OSS on bedrock-runtime).
public isolated distinct client class OpenAIModelProvider {
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
    # + model - An OpenAI id (bare, CRIS-prefixed, ARN, or route-prefixed)
    # + region - Default region; an ARN `model`'s region segment overrides it (§5.2)
    # + maxTokens - Maximum tokens to generate
    # + temperature - Sampling temperature. Leave unset (the default) to omit the
    #                 field entirely and use the model's own default — several current
    #                 models reject it outright
    # + config - Routing overrides, guardrails, passthrough, OpenAI knobs
    # + return - `nil` on success; otherwise an `ai:Error`
    public isolated function init(
            @display {label: "AWS Credentials"} BedrockCredentials credentials,
            @display {label: "Model"} OpenAIModel|string model,
            @display {label: "Region"} string region,
            @display {label: "Maximum Tokens"} int? maxTokens = DEFAULT_MAX_TOKEN_COUNT,
            @display {label: "Temperature"} decimal? temperature = (),
            @display {label: "Configuration"} *OpenAIConfig config)
            returns ai:Error? {
        RouteConfig routeConfig = {
            apiFamily: config.apiFamily,
            modelSchema: config.modelSchema,
            routeOverrides: config.routeOverrides
        };
        [Route, readonly & ModelCodec, BedrockTransport] [route, codec, transport] =
            check resolveSpine("OpenAIModelProvider", credentials, model, region, routeConfig,
                config?.signingServiceName, config?.httpConfig, config?.retryConfig, config?.guardrail);

        self.family = route.family;
        self.wireModelId = route.effectiveModelId;
        self.bareModelId = route.bareModelId;
        self.codec = codec;
        self.transport = transport;
        self.supportsStructuredOutput = route.family != MANTLE; // amendment
        self.params = openAIParams(maxTokens, temperature, config);
        self.extraHeaders = commonExtraHeaders(route, config?.guardrail, credentials).cloneReadOnly();
    }

    # + messages - Chat messages or a single user message
    # + tools - Tool definitions for function calling
    # + stop - Stop sequence; overrides configured `stopSequences` (§7)
    # + return - The assistant message, or an `ai:Error`
    isolated remote function chat(ai:ChatMessage[]|ai:ChatUserMessage messages,
            ai:ChatCompletionFunctions[] tools = [], string? stop = ())
            returns ai:ChatAssistantMessage|ai:Error
        => runChat("OpenAI", self.family, self.wireModelId, self.codec, self.transport,
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
        => runChatStream("OpenAI", self.family, self.wireModelId, self.bareModelId, self.codec,
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

// Folds the OpenAI `reasoningEffort` knob into the §9.3 passthrough.
isolated function openAIParams(int? maxTokens, decimal? temperature, OpenAIConfig config)
        returns readonly & InferenceParams {
    map<json> extras = {};
    string? reasoningEffort = config?.reasoningEffort;
    if reasoningEffort is string {
        extras["reasoning_effort"] = reasoningEffort;
    }
    json additional = foldRequestFields(config?.additionalModelRequestFields, extras);
    return buildInferenceParams(maxTokens, temperature, config?.stopSequences,
        additional, config?.additionalModelResponseFieldPaths, config?.serviceTier, config?.latencyOptimized,
        config?.requestMetadata, config?.guardrail);
}
