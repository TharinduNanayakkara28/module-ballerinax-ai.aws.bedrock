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
import ballerina/ai.observe;
import ballerina/http;
import ballerina/os;

// Shared facade machinery: every vendor provider is a thin class
// over these. `runChat` is the whole `chat()` body; `buildInferenceParams`
// assembles the resolved `InferenceParams`; `commonExtraHeaders` builds the
// route-specific headers common to all vendors. Only the model enum, config
// extras (folded into `additionalModelRequestFields`), and Invoke-converter choice
// differ per vendor.

// The default `region` for every provider. `ballerinax/aws` does not resolve a
// region — both `CredentialProvider` and `resolveEndpoint` take it as an argument —
// so mirror the SDK's own environment lookup here. Returns "" when neither is set,
// which `resolveSpine` turns into a named construction error UNLESS the model is an
// ARN carrying its own region.
isolated function defaultRegion() returns string {
    string region = os:getEnv("AWS_REGION");
    return region != "" ? region : os:getEnv("AWS_DEFAULT_REGION");
}

// The full `chat()` implementation, shared by every vendor facade.
// Opens an observe span and closes it on every path.
isolated function runChat(string providerName, ApiFamily family, string wireModelId,
        readonly & ModelConverter converter, BedrockTransport transport, map<string> & readonly extraHeaders,
        readonly & InferenceParams params, ai:ChatMessage[]|ai:ChatUserMessage messages,
        ai:ChatCompletionFunctions[] tools, string? stop) returns ai:ChatAssistantMessage|ai:Error {
    ai:ChatMessage[] msgs;
    if messages is ai:ChatUserMessage {
        msgs = [messages];
    } else {
        msgs = messages;
    }

    observe:ChatSpan span = observe:createChatSpan(wireModelId);
    span.addProvider(providerName);
    if stop is string {
        span.addStopSequence(stop);
    }
    decimal? spanTemperature = params?.temperature;
    if spanTemperature is decimal {
        // Only report a temperature the caller actually set — recording an invented
        // default would misreport what went on the wire.
        span.addTemperature(spanTemperature);
    }
    if tools.length() > 0 {
        span.addTools(tools);
    }

    // Resolve BEFORE encoding: flattens each prompt to parts and fetches any image
    // URL, so the encoders below stay pure. Any image on a dialect that cannot carry
    // one fails here, before the request is built.
    [string?, ResolvedMessage[]]|ai:Error resolved = resolveMessages(msgs);
    if resolved is ai:Error {
        span.close(resolved);
        return resolved;
    }
    [string?, ResolvedMessage[]] [system, rest] = resolved;
    // Recorded from the RESOLVED form so images become a placeholder. The raw form
    // would put the whole image — potentially megabytes of user data — into the span
    // and ship it to whatever telemetry backend is configured.
    span.addInputMessages(messagesForSpan(system, rest));
    RequestEncoder encode = converter.encode;
    json|ai:Error encoded = encode(system, rest, tools, stop, params);
    if encoded is ai:Error {
        span.close(encoded);
        return encoded;
    }
    json body = family == MANTLE ? injectModel(encoded, wireModelId) : encoded;

    TransportResponse|ai:Error response = transport.execute(body, extraHeaders);
    if response is ai:Error {
        span.close(response);
        return response;
    }

    ResponseDecoder decode = converter.decode;
    DecodedResponse|ai:Error decoded = decode(response.body);
    if decoded is ai:Error {
        span.close(decoded);
        return decoded;
    }
    // Surface the request id from the response headers.
    augmentFromHeaders(decoded, response.headers);

    span.addInputTokenCount(decoded.usage.inputTokens);
    span.addOutputTokenCount(decoded.usage.outputTokens);
    // A fired guardrail must never be silently dropped — that is the whole
    // reason `decode` returns a record rather than a bare message.
    //
    // The `ai:ModelProvider` contract has nowhere to put this: `chat()` returns an
    // `ai:ChatAssistantMessage`, which carries no guardrail field, and turning an
    // intervention into an error would break callers who guardrail every request by
    // policy and expect the blocked-content message back. So the signal goes to the
    // span's finish reason — the one channel that both survives to the caller's
    // observability backend and is already keyed on "why did generation stop".
    //
    // Converse already reports `guardrail_intervened` as its stopReason, so this
    // only changes behaviour on the Invoke route, where the body field is the
    // only source.
    if decoded.guardrailAction == INTERVENED && decoded.stopReason != "guardrail_intervened" {
        span.addFinishReason("guardrail_intervened");
    } else {
        span.addFinishReason(decoded.stopReason);
    }
    string? responseId = decoded.responseId;
    if responseId is string {
        span.addResponseId(responseId);
    }
    span.addOutputMessages(decoded.message);
    span.close();
    return decoded.message;
}

// Assembles the resolved `InferenceParams` once at construction.
// `additionalModelRequestFields` already carries any vendor extras the facade
// folded in (Claude `thinking`, Nova `reasoningConfig`, Qwen thinking…).
isolated function buildInferenceParams(int? maxTokens, decimal? temperature,
        string[]? stopSequences, AdditionalRequestFields? additionalModelRequestFields,
        ServiceTier? serviceTier,
        boolean? latencyOptimized, GuardrailConfig? guardrail,
        ThinkingConfig? thinking = (), Effort? effort = ())
        returns readonly & InferenceParams {
    InferenceParams params = {maxTokens: maxTokens ?: DEFAULT_MAX_TOKEN_COUNT};
    // No default: an unset temperature stays unset all the way to the wire, so the
    // model applies its own. `?:` here would make it impossible for a caller to
    // OMIT the field, which is a hard 400 on every sampling-deprecated model — see
    // `setTemperature` in converter_common.bal.
    if temperature is decimal {
        params.temperature = temperature;
    }
    if stopSequences is string[] {
        params.stopSequences = stopSequences;
    }
    if additionalModelRequestFields != () {
        params.additionalModelRequestFields = additionalModelRequestFields;
    }
    if serviceTier is ServiceTier {
        params.serviceTier = serviceTier;
    }
    if latencyOptimized is boolean {
        params.latencyOptimized = latencyOptimized;
    }
    if guardrail is GuardrailConfig {
        params.guardrail = guardrail;
    }
    // Claude-only today, hence defaulted: the other six vendors never pass them.
    if thinking is ThinkingConfig {
        params.thinking = thinking;
    }
    if effort is Effort {
        params.effort = effort;
    }
    return params.cloneReadOnly();
}

// Route-specific headers common to all vendors: Invoke guardrail
// headers. Vendor facades merge their own Mantle headers on top.
isolated function commonExtraHeaders(Route route, GuardrailConfig? guardrail, BedrockCredentials creds)
        returns map<string> {
    map<string> headers = {};
    if route.family == INVOKE && guardrail is GuardrailConfig {
        headers["X-Amzn-Bedrock-GuardrailIdentifier"] = guardrail.guardrailIdentifier;
        headers["X-Amzn-Bedrock-GuardrailVersion"] = guardrail.guardrailVersion;
    }
    addMantleApiKeyHeader(headers, route, creds);
    return headers;
}

// `x-api-key` for a Mantle path that authenticates with it (the Anthropic Messages
// surface). Derived from the path via `usesApiKeyHeader`, not stored per model.
//
// Shared, not per-vendor: this previously lived in the Anthropic facade alone, which
// meant the same rule was honoured there and silently ignored everywhere else.
//
// RESOLVED (verified live 2026-08-03): Anthropic's Mantle surface
// REJECTS a request that carries BOTH `Authorization` and `x-api-key` — it returns
// 401 `authentication_error: "request must not include both 'authorization' and
// 'x-api-key' headers"`. Either header ALONE returns 200, so the per-model
// path now SELECTS the header rather than hedging with both. AWS's documented
// curl uses `x-api-key`, so a Messages path sends exactly that, and the transport
// SUPPRESSES its default `Authorization: Bearer` whenever x-api-key is present
// (see the BearerToken branch in transport.bal). Do NOT re-add a second header here:
// the old "both carry the same key, so whichever the service reads it succeeds"
// assumption is known-false and was the exact cause of the 401.
//
// Only a BearerToken can populate it: with SigV4 credentials there is no api key,
// and the signature alone must authenticate the request.
isolated function addMantleApiKeyHeader(map<string> headers, Route route, BedrockCredentials creds) {
    MantleEntry? entry = route.mantleEntry;
    if route.family == MANTLE && entry is MantleEntry && usesApiKeyHeader(entry.path)
            && creds is BearerToken {
        headers["x-api-key"] = creds.apiKey;
    }
}

// No-region construction guard, shared by every facade. Fires only when the region
// is absent from BOTH the argument (or the environment behind `defaultRegion`) and
// the model ARN — an empty region would otherwise build the host
// `bedrock-runtime..amazonaws.com` and surface as an opaque DNS failure.
isolated function guardRegion(string region) returns ai:Error? {
    if region == "" {
        return error ai:Error("No AWS region: pass 'region', set AWS_REGION (or " +
            "AWS_DEFAULT_REGION) in the environment, or use a model ARN that carries " +
            "its own region.");
    }
}

// Guardrail-on-Mantle construction guard, shared by every facade.
isolated function guardMantleGuardrail(ApiFamily family, GuardrailConfig? guardrail) returns ai:Error? {
    if family == MANTLE && guardrail is GuardrailConfig {
        return error ai:Error("Guardrails are not supported on the Mantle route; " +
            "apply the standalone ApplyGuardrail API instead");
    }
}

// The shared construction spine: resolve → guard → endpoint →
// converter → transport. Every failure AWS cannot diagnose surfaces here, before any
// I/O. Returns the resolved route (for header/param assembly), the
// converter, and the transport.
isolated function resolveSpine(string providerName, BedrockCredentials credentials, string model,
        string region, string serviceUrl, RouteConfig routeConfig,
        http:ClientConfiguration? httpConfig, RetryConfig? retryConfig, GuardrailConfig? guardrail,
        boolean fips = false)
        returns [Route, readonly & ModelConverter, BedrockTransport]|ai:Error {
    do {
        Route route = check resolveRoute(model, region, routeConfig); // L1, pure
        // Validate the RESOLVED region, not the argument: an ARN's region segment
        // legitimately supplies it, so an ARN model with no `region` and no
        // AWS_REGION in the environment is well-formed and must not be rejected.
        check guardRegion(route.region);
        check guardMantleGuardrail(route.family, guardrail);
        Endpoint ep = check buildEndpoint(route, serviceUrl, fips);   // L2
        readonly & ModelConverter converter = check selectConverter(route);
        BedrockTransport transport =
            check new (credentials, route.region, ep, httpConfig, retryConfig);
        return [route, converter, transport];
    } on fail error e {
        if e is ai:Error {
            return e;
        }
        return error ai:Error(string `Failed to initialize ${providerName}: ${e.message()}`, e);
    }
}

// `AdditionalRequestFields` -> a JSON object ready for a request body.
//
// `AdditionalRequestFields` is `record {}`, whose rest type is `anydata` and is
// therefore NOT assignable to `json`. Every wire boundary funnels through here so
// that conversion exists exactly once. `toJson` is TOTAL — it deep-converts anydata
// values that are not already json instead of failing — so this returns no error.
isolated function additionalFieldsToJson(AdditionalRequestFields? fields) returns map<json>? {
    if fields is () {
        return ();
    }
    json converted = fields.toJson();
    return converted is map<json> && converted.length() > 0 ? converted : ();
}

// Picks the spine `generate()` should use, which is NOT always the spine `chat()`
// uses.
//
// Under `AUTO` a Mantle-capable model routes to Mantle, and Mantle has no structured
// output — so a typed `generate()` would fail on exactly the flagship models. When
// the same model is ALSO served on `bedrock-runtime` (`MantleEntry.onRuntime`), this
// resolves a second, Converse spine and `generate()` uses that instead. `chat()` is
// untouched and stays on Mantle.
//
// Deliberately limited to AUTO: an EXPLICIT `apiFamily = MANTLE` is the caller
// naming a destination, and silently going somewhere else would break the one
// guarantee an explicit override exists to provide. A Mantle-ONLY model has no
// fallback either, and keeps the clean "no structured output" error.
//
// CONSEQUENCE, documented in the README: a provider in this state talks to two
// endpoints, which need two different IAM permissions —
// `bedrock-mantle:CreateInference` for `chat()` and `bedrock:InvokeModel` for a typed
// `generate()`. Credentials holding only one will see the other path 403.
isolated function resolveGenerateSpine(string providerName, BedrockCredentials credentials,
        string model, string region, string serviceUrl, RouteConfig routeConfig,
        http:ClientConfiguration? httpConfig, RetryConfig? retryConfig, GuardrailConfig? guardrail,
        Route chatRoute, readonly & ModelConverter chatConverter, BedrockTransport chatTransport,
        map<string> chatHeaders, boolean fips = false)
        returns [ApiFamily, string, readonly & ModelConverter, BedrockTransport, map<string>]|ai:Error {
    MantleEntry? entry = chatRoute.mantleEntry;
    boolean fallbackApplies = chatRoute.family == MANTLE
        && (routeConfig?.apiFamily ?: AUTO) == AUTO
        && entry is MantleEntry && entry.onRuntime;
    if !fallbackApplies {
        return [chatRoute.family, chatRoute.effectiveModelId, chatConverter, chatTransport, chatHeaders];
    }
    RouteConfig converseConfig = {apiFamily: CONVERSE};
    [Route, readonly & ModelConverter, BedrockTransport] [route, converter, transport] =
        check resolveSpine(providerName, credentials, model, region, serviceUrl, converseConfig,
            httpConfig, retryConfig, guardrail, fips);
    return [route.family, route.effectiveModelId, converter, transport,
        commonExtraHeaders(route, guardrail, credentials)];
}

// Folds a vendor's extra request fields into `additionalModelRequestFields`.
// Returns `()` when there is nothing to forward.
isolated function foldRequestFields(AdditionalRequestFields? base, map<json> extras)
        returns AdditionalRequestFields? {
    AdditionalRequestFields merged = {};
    if base is AdditionalRequestFields {
        foreach [string, anydata] [k, v] in base.entries() {
            merged[k] = v;
        }
    }
    foreach [string, json] [k, v] in extras.entries() {
        merged[k] = v;
    }
    return merged.length() > 0 ? merged : ();
}

// Injects the bare model id into a Mantle request body. Converse/
// Invoke carry the model id in the URL path instead.
isolated function injectModel(json body, string modelId) returns json {
    if body is map<json> {
        map<json> withModel = body.clone();
        withModel["model"] = modelId;
        return withModel;
    }
    return body;
}

// Simplified message projection for the observe span (avoids `Prompt` objects,
// which are not `anydata`).
//
// Image parts are REDACTED to `[image <mime>, <n> bytes]`. A span is shipped to the
// caller's telemetry backend, so putting the payload there would both bloat every
// trace and export user image data to a system that was never meant to hold it.
isolated function messagesForSpan(string? system, ResolvedMessage[] messages) returns json {
    json[] out = [];
    if system is string {
        out.push({role: ai:SYSTEM, content: system});
    }
    foreach ResolvedMessage m in messages {
        if m is ResolvedUserMessage {
            out.push({role: m.role, content: partsForSpan(m.parts)});
        } else if m is ai:ChatAssistantMessage {
            out.push({role: m.role, content: m.content});
        } else {
            out.push({role: m.role, content: m.content, name: m.name});
        }
    }
    return out;
}
