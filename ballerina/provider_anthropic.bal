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
import ballerinax/aws.auth;
import ballerina/jballerina.java;

// AnthropicModelProvider — a thin typed facade over the shared spine.
// Claude on Converse (default) / Mantle (Mythos, Haiku) / Invoke-Anthropic.

# Well-known Claude model ids. Any newer Claude is reachable by passing its id as
# a `string`.
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

# Anthropic-specific configuration. Includes the shared `CommonModelConfig` and
# adds Claude-only knobs.
public type AnthropicConfig record {|
    *CommonModelConfig;
    # Extended/adaptive thinking. A typed record rather than raw `json`: the wire
    # spelling and the mode/budget pairing rules are enforced at construction.
    ThinkingConfig thinking?;
    # Reasoning depth, emitted as `output_config.effort`. The ONLY depth control on
    # the adaptive-only models (Mythos 5, Fable 5, Opus 4.7, Mythos Preview).
    Effort effort?;
|};

# Claude on AWS Bedrock. Routes to Converse, Mantle (Messages), or Invoke-Anthropic
# at construction.
public isolated distinct client class AnthropicModelProvider {
    *ai:ModelProvider;

    private final ApiFamily family;
    private final string wireModelId;
    private final readonly & ModelConverter converter;
    private final BedrockTransport transport;
    private final readonly & InferenceParams params;
    private final map<string> & readonly extraHeaders;
    // Structured output (generate() typed target) is unavailable on Mantle.
    // The spine generate() uses. Same objects as the chat spine EXCEPT when AUTO
    // sent chat to Mantle and the model is also on bedrock-runtime — then these hold
    // a Converse spine so a typed generate() works instead of erroring.
    private final ApiFamily genFamily;
    private final string genModelId;
    private final readonly & ModelConverter genConverter;
    private final BedrockTransport genTransport;
    private final map<string> & readonly genHeaders;
    private final boolean supportsStructuredOutput;

    # + model - A Claude id (bare, CRIS-prefixed, ARN, or `mantle/|converse/|invoke/` prefixed)
    # + credentials - Defaults to the full AWS credential chain (env vars, EKS IRSA,
    #                 SSO, shared config, `credential_process`, ECS container credentials,
    #                 EC2 IMDSv2), so nothing needs configuring on AWS compute. Pass an
    #                 `auth:StaticAuthConfig`, `auth:AssumeRoleConfig`, ... for an explicit
    #                 source, or a `BearerToken` for a Bedrock API key
    # + region - Defaults to AWS_REGION/AWS_DEFAULT_REGION. An ARN `model`'s region
    #            segment overrides it. Also the SigV4 signing scope, which a custom
    #            `serviceUrl` does NOT change
    # + serviceUrl - Endpoint origin. The default template resolves per route from AWS
    #                SDK endpoint metadata, which already covers every partition
    #                (`amazonaws.com`, `amazonaws.com.cn`) and Mantle's `api.aws`.
    #                Override it only for a host AWS cannot derive:
    #                `https://vpce-0abc123.bedrock-runtime.us-east-1.vpce.amazonaws.com`
    #                (PrivateLink / VPC endpoint), `https://bedrock-gw.internal.corp`
    #                (an egress gateway or proxy), or `http://localhost:4566`
    #                (LocalStack, a mock server, or a recorded fixture in tests).
    #                The placeholders `{endpoint}` (`runtime`|`mantle`), `{region}` and
    #                `{domain}` are substituted, so a partial override such as
    #                `https://bedrock-{endpoint}.{region}.{domain}` keeps region and
    #                domain automatic. It replaces the ORIGIN only — the route-derived
    #                request path is still appended — and never changes the SigV4
    #                signing scope. For FIPS use `config.fips`, not a hand-written host
    # + maxTokens - Maximum tokens to generate
    # + temperature - Sampling temperature. Leave unset (the default) to omit the
    #                 field entirely and use the model's own default — several current
    #                 models reject it outright
    # + config - Routing overrides, guardrails, Converse passthrough, Claude knobs
    # + return - `nil` on success; otherwise an `ai:Error`
    public isolated function init(
            @display {label: "Model"} AnthropicModel|string model,
            @display {label: "AWS Credentials"} BedrockCredentials credentials = auth:DEFAULT_CREDENTIALS,
            @display {label: "Region"} string region = defaultRegion(),
            @display {label: "Service URL"} string serviceUrl = DEFAULT_SERVICE_URL,
            @display {label: "Maximum Tokens"} int? maxTokens = DEFAULT_MAX_TOKEN_COUNT,
            @display {label: "Temperature"} decimal? temperature = (),
            @display {label: "Configuration"} *AnthropicConfig config)
            returns ai:Error? {
        // ---- shared spine (identical in every vendor provider) ----
        RouteConfig routeConfig = {apiFamily: config.apiFamily};
        [Route, readonly & ModelConverter, BedrockTransport] [route, converter, transport] =
            check resolveSpine("AnthropicModelProvider", credentials, model, region, serviceUrl, routeConfig,
                config?.httpConfig, config?.retryConfig, config?.guardrail, config.fips);

        self.family = route.family;
        self.wireModelId = route.effectiveModelId;
        self.converter = converter;
        self.transport = transport;
        map<string> chatHeaders = buildExtraHeaders(route, config, credentials);
        self.extraHeaders = chatHeaders.cloneReadOnly();
        [ApiFamily, string, readonly & ModelConverter, BedrockTransport, map<string>]
            [genFamily, genModelId, genConverter, genTransport, genHeaders] =
            check resolveGenerateSpine("AnthropicModelProvider", credentials, model, region, serviceUrl,
                routeConfig, config?.httpConfig, config?.retryConfig, config?.guardrail,
                route, converter, transport, chatHeaders, config.fips);
        self.genFamily = genFamily;
        self.genModelId = genModelId;
        self.genConverter = genConverter;
        self.genTransport = genTransport;
        self.genHeaders = genHeaders.cloneReadOnly();
        self.supportsStructuredOutput = genFamily != MANTLE;
        self.params = check resolveParams(maxTokens, temperature, config);
    }

    # Sends a chat request. Opens an observe span and closes it on every path.
    #
    # + messages - Chat messages or a single user message
    # + tools - Tool definitions for function calling
    # + stop - Stop sequence; overrides configured `stopSequences`
    # + return - The assistant message, or an `ai:Error`
    isolated remote function chat(ai:ChatMessage[]|ai:ChatUserMessage messages,
            ai:ChatCompletionFunctions[] tools = [], string? stop = ())
            returns ai:ChatAssistantMessage|ai:Error
        => runChat("Anthropic", self.family, self.wireModelId, self.converter, self.transport,
            self.extraHeaders, self.params, messages, tools, stop);

    # Generates a value of the expected type by forcing a single tool whose schema
    # is that type. Available on the Converse and Invoke routes; a Mantle-routed
    # model returns an `ai:Error` for any target type other than `string`.
    # External Java per the platform convention; the shim calls back into
    # `generateLlmResponse`, passing this provider's resolved state.
    #
    # Uses the generate spine, which differs from the chat spine when `AUTO` routed
    # chat to Mantle and the model is also served on `bedrock-runtime`.
    #
    # + prompt - The prompt to use in the chat request
    # + td - Type descriptor of the expected return type
    # + return - A value of the expected type, or an `ai:Error`
    isolated remote function generate(ai:Prompt prompt,
            @display {label: "Expected type"} typedesc<anydata> td = <>)
            returns td|ai:Error = @java:Method {
        'class: "io.ballerina.lib.ai.aws.bedrock.Generator"
    } external;

    # Sends a chat request and streams the reply as `ai:ChatMessageChunk`s.
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
    # + return - A stream of assistant message chunks, or an `ai:Error`
    remote function chatAsStream(ai:ChatMessage[]|ai:ChatUserMessage messages,
            ai:ChatCompletionFunctions[] tools = [], string? stop = ())
            returns stream<ai:ChatMessageChunk, ai:Error?>|ai:Error
        => runChatStream("Anthropic", self.family, self.wireModelId, self.converter,
            self.transport, self.extraHeaders, self.params, messages, tools, stop);

    # Streams the answer to a prompt as text fragments. The request is built as
    # `generate` builds it for a `string` result, on the same route.
    #
    # Streaming produces text only: structured types have no valid intermediate
    # state, so use `generate` for structured output.
    #
    # + prompt - The prompt to use in the chat request
    # + return - A stream of text fragments, or an `ai:Error`
    remote function generateAsStream(ai:Prompt prompt) returns stream<string, ai:Error?>|ai:Error
        => runGenerateStream("Anthropic", self.genFamily, self.genModelId, self.genConverter,
            self.genTransport, self.genHeaders, self.params, prompt);
}

// Resolves inference params once at construction. Folds the
// Claude `thinking` knob into the `additionalModelRequestFields`
// passthrough, which every Anthropic converter forwards.
isolated function resolveParams(int? maxTokens, decimal? temperature, AnthropicConfig config)
        returns readonly & InferenceParams|ai:Error {
    int resolvedMaxTokens = maxTokens ?: DEFAULT_MAX_TOKEN_COUNT;
    ThinkingConfig? thinking = config?.thinking;
    if thinking is ThinkingConfig {
        check validateThinking(thinking, resolvedMaxTokens);
    }
    return buildInferenceParams(maxTokens, temperature, config?.stopSequences,
        config?.additionalModelRequestFields, config?.serviceTier, config?.latencyOptimized,
        config?.guardrail, thinking, config?.effort);
}

// The budget rules AWS enforces with a 400. Checked before any I/O so the message
// names the actual mistake instead of surfacing as an opaque ValidationException.
// https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html
isolated function validateThinking(ThinkingConfig thinking, int maxTokens) returns ai:Error? {
    int? budget = thinking?.budgetTokens;
    if thinking.mode != ENABLED {
        if budget is int {
            return error ai:Error(string `'budgetTokens' is only valid with 'mode = ENABLED'; mode is ` +
                string `'${thinking.mode}'. Adaptive thinking is steered with 'effort' instead.`);
        }
        return;
    }
    if budget is () {
        return error ai:Error("'mode = ENABLED' requires 'budgetTokens' — manual extended thinking " +
            "has no default budget. Use 'mode = ADAPTIVE' to let the model decide.");
    }
    if budget < MIN_THINKING_BUDGET_TOKENS {
        return error ai:Error(string `'budgetTokens' must be at least ` +
            string `${MIN_THINKING_BUDGET_TOKENS}; got ${budget}.`);
    }
    if budget >= maxTokens {
        return error ai:Error(string `'budgetTokens' (${budget}) must be less than 'maxTokens' ` +
            string `(${maxTokens}) — the thinking budget is drawn from the same ceiling.`);
    }
}

// Route-specific headers computed once: the common Invoke
// guardrail and Mantle api-key headers, plus Anthropic's own Mantle Messages
// version header.
isolated function buildExtraHeaders(Route route, AnthropicConfig config, BedrockCredentials creds)
        returns map<string> {
    // `x-api-key` is emitted by `commonExtraHeaders` for every Mantle Messages path —
    // it follows the path, not the vendor.
    map<string> headers = commonExtraHeaders(route, config?.guardrail, creds);
    MantleEntry? entry = route.mantleEntry;
    if route.family == MANTLE && entry is MantleEntry && usesApiKeyHeader(entry.path) {
        // Different value AND mechanism from the Invoke body field.
        // https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
        headers["anthropic-version"] = "2023-06-01";
    }
    return headers;
}
