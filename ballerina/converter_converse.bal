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

// Converse converter — the model-agnostic normalized surface.
// https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html

// Encodes a Converse request body. `system` is the top-level `system` field, never
// a message. Forwards the passthrough verbatim.
isolated function encodeConverse(string? system, ResolvedMessage[] messages,
        ai:ChatCompletionFunctions[] tools, string? stop, InferenceParams params) returns json|ai:Error {
    json[] wire = [];
    foreach ResolvedMessage m in messages {
        wire.push(converseMessage(m));
    }

    map<json> inferenceConfig = {"maxTokens": params.maxTokens};
    setTemperature(inferenceConfig, params);
    // Per-call `stop` overrides configured stopSequences outright.
    string[]? stops = params.stopSequences;
    if stop is string {
        stops = [stop];
    }
    if stops is string[] && stops.length() > 0 {
        inferenceConfig["stopSequences"] = stops;
    }

    map<json> body = {"messages": wire, "inferenceConfig": inferenceConfig};

    if system is string {
        body["system"] = [{"text": system}];
    }
    if tools.length() > 0 {
        json[] toolSpecs = [];
        foreach ai:ChatCompletionFunctions t in tools {
            toolSpecs.push({
                "toolSpec": {"name": t.name, "description": t.description, "inputSchema": {"json": toolParameters(t)}}
            });
        }
        body["toolConfig"] = {"tools": toolSpecs};
    }

    // ---- passthrough — forwarded verbatim; mandatory for top_k/thinking/reasoning ----
    // Converse does not model `thinking`, so it rides the passthrough — merged in
    // rather than overwriting whatever the caller already put there.
    AdditionalRequestFields? additionalRequest = params?.additionalModelRequestFields;
    ThinkingConfig? thinking = params?.thinking;
    if thinking is ThinkingConfig {
        additionalRequest = foldRequestFields(additionalRequest, {"thinking": thinkingBody(thinking)});
    }
    // `effort` rides the passthrough too, not a native `outputConfig` member.
    //
    // Two first-party sources disagreed here: botocore models `outputConfig:
    // {textFormat, effort}` on ConverseRequest, while AWS's adaptive-thinking page
    // routes it through `additionalModelRequestFields: {"output_config": {"effort":
    // ...}}`. Live-verified 2026-08-11: the native member 400s on every model tried
    // (opus-4-8, sonnet-4-6, and — decisively — opus-4-7, which IS on Anthropic's
    // adaptive-only list, ruling out "wrong model") with "This model doesn't support
    // the effort field"; the identical value folded into
    // additionalModelRequestFields.output_config.effort is accepted (opus-4-7,
    // controlled pair against the same 400). AWS's docs were right; botocore's
    // modelled member is not honoured on the wire.
    Effort? effort = params?.effort;
    if effort is Effort {
        additionalRequest = foldRequestFields(additionalRequest, {"output_config": {"effort": effort}});
    }
    map<json>? additionalJson = additionalFieldsToJson(additionalRequest);
    if additionalJson != () {
        body["additionalModelRequestFields"] = additionalJson;
    }
    ServiceTier? tier = params.serviceTier;
    if tier is ServiceTier {
        // An OBJECT, not a bare string: the Converse request syntax is
        // `"serviceTier": { "type": "string" }`. Emitting the string 400s.
        // https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        body["serviceTier"] = {"type": tier};
    }
    boolean? latencyOptimized = params.latencyOptimized;
    if latencyOptimized == true {
        // `"performanceConfig": { "latency": "optimized" }` — an object, like
        // serviceTier. Only `optimized` is worth emitting; `standard` is the default,
        // so an unset/false flag sends nothing. Support is per model+region.
        // https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        body["performanceConfig"] = {"latency": "optimized"};
    }
    // Guardrail is a Converse BODY field.
    GuardrailConfig? guardrail = params.guardrail;
    if guardrail is GuardrailConfig {
        map<json> gc = {
            "guardrailIdentifier": guardrail.guardrailIdentifier,
            "guardrailVersion": guardrail.guardrailVersion
        };
        body["guardrailConfig"] = gc;
    }
    return body;
}

// Maps one resolved message to a Converse content block. Images ride the native
// `image` ContentBlock member — verified against the Converse API reference.
isolated function converseMessage(ResolvedMessage m) returns json {
    if m is ResolvedUserMessage {
        return {"role": "user", "content": converseContentBlocks(m.parts)};
    }
    if m is ai:ChatAssistantMessage {
        json[] blocks = [];
        string? c = m.content;
        if c is string && c != "" {
            blocks.push({"text": c});
        }
        ai:FunctionCall[]? toolCalls = m.toolCalls;
        if toolCalls is ai:FunctionCall[] {
            foreach ai:FunctionCall fc in toolCalls {
                blocks.push({"toolUse": {"toolUseId": fc.id ?: fc.name, "name": fc.name, "input": fc.arguments ?: {}}});
            }
        }
        return {"role": "assistant", "content": blocks};
    }
    // ai:FUNCTION result → Converse toolResult block.
    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": m.id ?: m.name, "content": [{"text": m.content ?: ""}]}}]
    };
}

// Decodes a Converse response. Always populates `usage` and
// `stopReason`; maps `guardrail_intervened` to `INTERVENED`.
isolated function decodeConverse(json response) returns DecodedResponse|ai:Error {
    map<json>|error rr = response.ensureType();
    if rr is error {
        return error ai:LlmInvalidResponseError("Converse response was not a JSON object", rr);
    }
    map<json> r = rr;
    string text = "";
    ai:FunctionCall[] toolCalls = [];

    map<json>? output = mapField(r, "output");
    if output is map<json> {
        map<json>? message = mapField(output, "message");
        if message is map<json> {
            json[]? content = arrField(message, "content");
            if content is json[] {
                foreach json blk in content {
                    if blk is map<json> {
                        string? txt = strField(blk, "text");
                        if txt is string {
                            text += txt;
                        }
                        map<json>? toolUse = mapField(blk, "toolUse");
                        if toolUse is map<json> {
                            toolCalls.push({
                                name: strField(toolUse, "name") ?: "",
                                arguments: mapField(toolUse, "input") ?: {},
                                id: strField(toolUse, "toolUseId")
                            });
                        }
                    }
                }
            }
        }
    }

    int inputTokens = 0;
    int outputTokens = 0;
    map<json>? usage = mapField(r, "usage");
    if usage is map<json> {
        inputTokens = intField(usage, "inputTokens") ?: 0;
        outputTokens = intField(usage, "outputTokens") ?: 0;
    }

    string stopReason = strField(r, "stopReason") ?: "end_turn";
    ai:ChatAssistantMessage message = {role: ai:ASSISTANT, content: text == "" ? () : text};
    if toolCalls.length() > 0 {
        message.toolCalls = toolCalls;
    }
    return {
        message,
        usage: {inputTokens, outputTokens},
        stopReason,
        responseId: (), // Converse returns the request id in a header, not the body
        // Converse reports it via stopReason; Nova-on-Invoke shares this decoder
        // but reports it as a body field instead, so check both.
        guardrailAction: stopReason == "guardrail_intervened" ? INTERVENED : invokeGuardrailAction(r)
    };
}
