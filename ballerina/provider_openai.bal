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

// OpenAIModelProvider — GPT-5.x via Mantle Responses (`/openai/v1/responses`),
// GPT-OSS via Converse/Invoke.

# Well-known OpenAI model ids on Bedrock. Any newer id can be passed as a `string`.
public enum OpenAIModel {
    // Mantle-only — these do not exist on bedrock-runtime.
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

# OpenAI-specific configuration.
public type OpenAIConfig record {|
    *CommonModelConfig;

    # `reasoning_effort` — trades latency and token cost against reasoning depth.
    # Forwarded verbatim via the passthrough, so the ACCEPTED VALUES ARE NOT
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
    private final readonly & ModelConverter converter;
    private final BedrockTransport transport;
    private final readonly & InferenceParams params;
    private final map<string> & readonly extraHeaders;
    // The spine generate() uses. Same objects as the chat spine EXCEPT when AUTO
    // sent chat to Mantle and the model is also on bedrock-runtime — then these hold
    // a Converse spine so a typed generate() works instead of erroring.
    private final ApiFamily genFamily;
    private final string genModelId;
    private final readonly & ModelConverter genConverter;
    private final BedrockTransport genTransport;
    private final map<string> & readonly genHeaders;
    private final boolean supportsStructuredOutput;

    # + model - An OpenAI id (bare, CRIS-prefixed, ARN, or route-prefixed)
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
    # + config - Routing overrides, guardrails, passthrough, OpenAI knobs
    # + return - `nil` on success; otherwise an `ai:Error`
    public isolated function init(
            @display {label: "Model"} OpenAIModel|string model,
            @display {label: "AWS Credentials"} BedrockCredentials credentials = auth:DEFAULT_CREDENTIALS,
            @display {label: "Region"} string region = defaultRegion(),
            @display {label: "Service URL"} string serviceUrl = DEFAULT_SERVICE_URL,
            @display {label: "Maximum Tokens"} int? maxTokens = DEFAULT_MAX_TOKEN_COUNT,
            @display {label: "Temperature"} decimal? temperature = (),
            @display {label: "Configuration"} *OpenAIConfig config)
            returns ai:Error? {
        RouteConfig routeConfig = {apiFamily: config.apiFamily};
        [Route, readonly & ModelConverter, BedrockTransport] [route, converter, transport] =
            check resolveSpine("OpenAIModelProvider", credentials, model, region, serviceUrl, routeConfig,
                config?.httpConfig, config?.retryConfig, config?.guardrail, config.fips);

        self.family = route.family;
        self.wireModelId = route.effectiveModelId;
        self.converter = converter;
        self.transport = transport;
        map<string> chatHeaders = commonExtraHeaders(route, config?.guardrail, credentials);
        self.extraHeaders = chatHeaders.cloneReadOnly();
        [ApiFamily, string, readonly & ModelConverter, BedrockTransport, map<string>]
            [genFamily, genModelId, genConverter, genTransport, genHeaders] =
            check resolveGenerateSpine("OpenAIModelProvider", credentials, model, region, serviceUrl,
                routeConfig, config?.httpConfig, config?.retryConfig, config?.guardrail,
                route, converter, transport, chatHeaders, config.fips);
        self.genFamily = genFamily;
        self.genModelId = genModelId;
        self.genConverter = genConverter;
        self.genTransport = genTransport;
        self.genHeaders = genHeaders.cloneReadOnly();
        self.supportsStructuredOutput = genFamily != MANTLE;
        self.params = openAIParams(maxTokens, temperature, config);
    }

    # + messages - Chat messages or a single user message
    # + tools - Tool definitions for function calling
    # + stop - Stop sequence; overrides configured `stopSequences`
    # + return - The assistant message, or an `ai:Error`
    isolated remote function chat(ai:ChatMessage[]|ai:ChatUserMessage messages,
            ai:ChatCompletionFunctions[] tools = [], string? stop = ())
            returns ai:ChatAssistantMessage|ai:Error
        => runChat("OpenAI", self.family, self.wireModelId, self.converter, self.transport,
            self.extraHeaders, self.params, messages, tools, stop);

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
        => runChatStream("OpenAI", self.family, self.wireModelId, self.converter,
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
        => runGenerateStream("OpenAI", self.genFamily, self.genModelId, self.genConverter,
            self.genTransport, self.genHeaders, self.params, prompt);
}

// Folds the OpenAI `reasoningEffort` knob into the passthrough.
isolated function openAIParams(int? maxTokens, decimal? temperature, OpenAIConfig config)
        returns readonly & InferenceParams {
    map<json> extras = {};
    string? reasoningEffort = config?.reasoningEffort;
    if reasoningEffort is string {
        extras["reasoning_effort"] = reasoningEffort;
    }
    AdditionalRequestFields? additional = foldRequestFields(config?.additionalModelRequestFields, extras);
    return buildInferenceParams(maxTokens, temperature, config?.stopSequences,
        additional, config?.serviceTier, config?.latencyOptimized, config?.guardrail);
}
