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

// DeepSeekModelProvider — Converse + Invoke.
//
// NOTE: DeepSeek requires a cross-region inference-profile id, not a
// bare model id — e.g. `us.deepseek.r1-v1:0`. We deliberately do NOT hard-fail on a
// bare id: AWS answers that with a `ValidationException` naming the real problem,
// and construction errors are reserved for what AWS cannot tell us. The
// resolver strips/re-applies the geo prefix for you.

# Well-known DeepSeek model ids. Any newer id can be passed as a `string`.
public enum DeepSeekModel {
    # The CRIS (cross-region) profile id, NOT the bare `deepseek.r1-v1:0`.
    #
    # This is deliberate: the card's Regional Availability table marks In-Region as
    # NO in *every* region and Geo as YES, so the bare id is not callable anywhere —
    # `us.` is the only form that resolves. US is also the only geo AWS lists for
    # this model. Pass a raw string if you need a different profile.
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-r1.html
    DEEPSEEK_R1 = "us.deepseek.r1-v1:0",
    # DeepSeek V3.2 — the current flagship (MoE, reasoning/coding). UNLIKE R1, the
    # BARE id IS callable: the card marks In-Region YES in us-east-1 and elsewhere,
    # Geo/Global not supported — so no CRIS prefix. Converse + Invoke on
    # bedrock-runtime (dual-homed with bedrock-mantle; defaults to Converse here).
    # https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html
    DEEPSEEK_V3_2 = "deepseek.v3.2"
}

# DeepSeek-specific configuration.
public type DeepSeekConfig record {|
    *CommonModelConfig;
|};

# DeepSeek models on AWS Bedrock.
public isolated distinct client class DeepSeekModelProvider {
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

    # + model - A DeepSeek id; use a CRIS inference-profile id (e.g. `us.deepseek.r1-v1:0`)
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
    # + config - Routing overrides, guardrails, Converse passthrough
    # + return - `nil` on success; otherwise an `ai:Error`
    public isolated function init(
            @display {label: "Model"} DeepSeekModel|string model,
            @display {label: "AWS Credentials"} BedrockCredentials credentials = auth:DEFAULT_CREDENTIALS,
            @display {label: "Region"} string region = defaultRegion(),
            @display {label: "Service URL"} string serviceUrl = DEFAULT_SERVICE_URL,
            @display {label: "Maximum Tokens"} int? maxTokens = DEFAULT_MAX_TOKEN_COUNT,
            @display {label: "Temperature"} decimal? temperature = (),
            @display {label: "Configuration"} *DeepSeekConfig config)
            returns ai:Error? {
        RouteConfig routeConfig = {apiFamily: config.apiFamily};
        [Route, readonly & ModelConverter, BedrockTransport] [route, converter, transport] =
            check resolveSpine("DeepSeekModelProvider", credentials, model, region, serviceUrl, routeConfig,
                config?.httpConfig, config?.retryConfig, config?.guardrail, config.fips);

        self.family = route.family;
        self.wireModelId = route.effectiveModelId;
        self.converter = converter;
        self.transport = transport;
        map<string> chatHeaders = commonExtraHeaders(route, config?.guardrail, credentials);
        self.extraHeaders = chatHeaders.cloneReadOnly();
        [ApiFamily, string, readonly & ModelConverter, BedrockTransport, map<string>]
            [genFamily, genModelId, genConverter, genTransport, genHeaders] =
            check resolveGenerateSpine("DeepSeekModelProvider", credentials, model, region, serviceUrl,
                routeConfig, config?.httpConfig, config?.retryConfig, config?.guardrail,
                route, converter, transport, chatHeaders, config.fips);
        self.genFamily = genFamily;
        self.genModelId = genModelId;
        self.genConverter = genConverter;
        self.genTransport = genTransport;
        self.genHeaders = genHeaders.cloneReadOnly();
        self.supportsStructuredOutput = genFamily != MANTLE;
        self.params = buildInferenceParams(maxTokens, temperature, config?.stopSequences,
            config?.additionalModelRequestFields, config?.serviceTier,
            config?.latencyOptimized, config?.guardrail);
    }

    # + messages - Chat messages or a single user message
    # + tools - Tool definitions for function calling
    # + stop - Stop sequence; overrides configured `stopSequences`
    # + return - The assistant message, or an `ai:Error`
    isolated remote function chat(ai:ChatMessage[]|ai:ChatUserMessage messages,
            ai:ChatCompletionFunctions[] tools = [], string? stop = ())
            returns ai:ChatAssistantMessage|ai:Error
        => runChat("DeepSeek", self.family, self.wireModelId, self.converter, self.transport,
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
        => runChatStream("DeepSeek", self.family, self.wireModelId, self.converter,
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
