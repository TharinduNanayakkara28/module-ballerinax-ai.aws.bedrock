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

// AnthropicModelProvider — a thin typed facade over the shared spine (CLAUDE.md
// §3). Claude on Converse (default) / Mantle (Mythos, Haiku) / Invoke-Anthropic.

# Well-known Claude model ids. Any newer Claude is reachable by passing its id as
# a `string` (design §5.5, principle 4).
public enum AnthropicModel {
    # Claude Opus 5 — Anthropic's most advanced Opus (1M context, 128K max output,
    # adaptive thinking on by default). Converse + Invoke + Messages; dual-homed
    # (bedrock-runtime and bedrock-mantle), so under `AUTO` it resolves to Mantle —
    # pass `apiFamily = CONVERSE` for typed `generate()`. In-Region callable in
    # us-east-1, eu-north-1, eu-west-1 and ap-southeast-4 only; elsewhere use a geo
    # (`us.`/`eu.`/`au.`) or `global.` profile.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-5.html
    CLAUDE_OPUS_5 = "anthropic.claude-opus-5",
    CLAUDE_OPUS_4_8 = "anthropic.claude-opus-4-8",
    # Claude Sonnet 5 — the current flagship Sonnet (1M context, adaptive thinking
    # always on). Converse + Invoke + Messages; dual-homed (bedrock-runtime and
    # bedrock-mantle), so under `AUTO` it resolves to Mantle — pass
    # `apiFamily = CONVERSE` for typed `generate()`.
    # In-Region callable in us-east-1 (not every region — some are Geo/Global only);
    # geo profiles `us.`/`eu.`/`au.` and `global.` also work.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5.html
    CLAUDE_SONNET_5 = "anthropic.claude-sonnet-5",
    CLAUDE_SONNET_4_6 = "anthropic.claude-sonnet-4-6",
    CLAUDE_HAIKU_4_5 = "anthropic.claude-haiku-4-5",
    CLAUDE_MYTHOS_PREVIEW = "anthropic.claude-mythos-preview",
    # `bedrock-mantle` only (Messages API). No structured output.
    #
    # This model rejects sampling parameters (as do Opus 4.7+, Opus 5 and Sonnet 5):
    # passing `temperature` at all returns a 400. Leave it unset — the default.
    # It also requires opting in to provider data sharing.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-mythos-5.html
    CLAUDE_MYTHOS_5 = "anthropic.claude-mythos-5"
}

# Anthropic-specific configuration (CLAUDE.md §3). Includes the shared
# `CommonModelConfig` and adds Claude-only knobs. `thinking` is folded
# into the Converse `additionalModelRequestFields` passthrough (design §9.3).
public type AnthropicConfig record {|
    *CommonModelConfig;
    # Extended-thinking config, forwarded verbatim (§9.3).
    json thinking?;
    # `anthropic-workspace` header for per-application cost scoping on Mantle (§7.3).
    string anthropicWorkspaceId?;
|};

# Claude on AWS Bedrock. Routes to Converse, Mantle (Messages), or Invoke-Anthropic
# at construction (design §5, §6).
public isolated distinct client class AnthropicModelProvider {
    *ai:ModelProvider;

    private final ApiFamily family;
    private final string wireModelId;
    private final readonly & ModelCodec codec;
    private final BedrockTransport transport;
    private final readonly & InferenceParams params;
    private final map<string> & readonly extraHeaders;
    // Structured output (generate() typed target) is unavailable on Mantle (amendment).
    private final boolean supportsStructuredOutput;

    # + credentials - Static keys, STS, or a Bedrock API key (§9.5)
    # + model - A Claude id (bare, CRIS-prefixed, ARN, or `mantle/|converse/|invoke/` prefixed)
    # + region - Default region; an ARN `model`'s region segment overrides it (§5.2)
    # + maxTokens - Maximum tokens to generate
    # + temperature - Sampling temperature. Leave unset (the default) to omit the
    #                 field entirely and use the model's own default — several current
    #                 models reject it outright
    # + config - Routing overrides, guardrails, Converse passthrough, Claude knobs
    # + return - `nil` on success; otherwise an `ai:Error`
    public isolated function init(
            @display {label: "AWS Credentials"} BedrockCredentials credentials,
            @display {label: "Model"} AnthropicModel|string model,
            @display {label: "Region"} string region,
            @display {label: "Maximum Tokens"} int? maxTokens = DEFAULT_MAX_TOKEN_COUNT,
            @display {label: "Temperature"} decimal? temperature = (),
            @display {label: "Configuration"} *AnthropicConfig config)
            returns ai:Error? {
        // ---- shared spine (identical in every vendor provider) ----
        RouteConfig routeConfig = {
            apiFamily: config.apiFamily,
            modelSchema: config.modelSchema,
            routeOverrides: config.routeOverrides
        };
        [Route, readonly & ModelCodec, BedrockTransport] [route, codec, transport] =
            check resolveSpine("AnthropicModelProvider", credentials, model, region, routeConfig,
                config?.signingServiceName, config?.httpConfig, config?.retryConfig, config?.guardrail);

        self.family = route.family;
        self.wireModelId = route.effectiveModelId;
        self.codec = codec;
        self.transport = transport;
        self.supportsStructuredOutput = route.family != MANTLE; // amendment
        self.params = resolveParams(maxTokens, temperature, config);
        self.extraHeaders = buildExtraHeaders(route, config, credentials).cloneReadOnly();
    }

    # Sends a chat request (design §3, §7). Opens an observe span and closes it on
    # every path (§3.2).
    #
    # + messages - Chat messages or a single user message
    # + tools - Tool definitions for function calling
    # + stop - Stop sequence; overrides configured `stopSequences` (§7)
    # + return - The assistant message, or an `ai:Error`
    isolated remote function chat(ai:ChatMessage[]|ai:ChatUserMessage messages,
            ai:ChatCompletionFunctions[] tools = [], string? stop = ())
            returns ai:ChatAssistantMessage|ai:Error
        => runChat("Anthropic", self.family, self.wireModelId, self.codec, self.transport,
            self.extraHeaders, self.params, messages, tools, stop);

    # Generates a value of the expected type by forcing a single tool whose schema
    # is that type (design §8). Available on the Converse and Invoke routes; a
    # Mantle-routed model returns an `ai:Error` for any target type other than
    # `string`. External Java per the platform convention (§3); the shim calls back
    # into `generateLlmResponse`, passing this provider's resolved state.
    #
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
        => runChatStream("Anthropic", self.family, self.wireModelId, self.codec,
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

// Resolves inference params once at construction (design §6, §7). Folds the
// Claude `thinking` knob into the `additionalModelRequestFields`
// passthrough (§9.3), which every Anthropic codec forwards.
isolated function resolveParams(int? maxTokens, decimal? temperature, AnthropicConfig config)
        returns readonly & InferenceParams {
    map<json> extras = {};
    json thinking = config?.thinking;
    if thinking != () {
        extras["thinking"] = thinking;
    }
    json additional = foldRequestFields(config?.additionalModelRequestFields, extras);
    return buildInferenceParams(maxTokens, temperature, config?.stopSequences,
        additional, config?.additionalModelResponseFieldPaths, config?.serviceTier, config?.latencyOptimized,
        config?.requestMetadata, config?.guardrail);
}

// Route-specific headers computed once (design §7.3, §9.5): the common Invoke
// guardrail and Mantle api-key headers, plus Anthropic's own Mantle Messages
// version/workspace headers.
isolated function buildExtraHeaders(Route route, AnthropicConfig config, BedrockCredentials creds)
        returns map<string> {
    // `x-api-key` is emitted by `commonExtraHeaders` for every vendor whose Mantle
    // entry declares X_API_KEY — it is table data, not an Anthropic special case.
    map<string> headers = commonExtraHeaders(route, config?.guardrail, creds);
    MantleEntry? entry = route.mantleEntry;
    if route.family == MANTLE && entry is MantleEntry {
        if entry.codec == MESSAGES_CODEC {
            // Different value AND mechanism from the Invoke body field (§7.3).
            headers["anthropic-version"] = "2023-06-01";
        }
        string? workspace = config?.anthropicWorkspaceId;
        if workspace is string {
            headers["anthropic-workspace"] = workspace;
        }
    }
    return headers;
}
