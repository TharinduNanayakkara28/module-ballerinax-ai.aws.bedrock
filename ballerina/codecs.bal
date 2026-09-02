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

// Codec registry + selection (design §7, §7.2, §7.3). `selectCodec` runs once at
// construction (L1/L2 — §4). The Anthropic vertical slice ships three codecs;
// the other Invoke/Mantle dialects error clearly until their vendor phase lands.

// Converse — model-agnostic, so one codec serves every Converse model (§5, §9.3).
final readonly & ModelCodec CONVERSE_CODEC = {
    encode: encodeConverse,
    decode: decodeConverse,
    toolChoice: CONVERSE_TOOL_CHOICE,
    streamDialect: CONVERSE_STREAM
};

// Invoke-Anthropic — `anthropic_version: bedrock-2023-05-31` body field (§7.2).
final readonly & ModelCodec INVOKE_ANTHROPIC_CODEC = {
    encode: encodeInvokeAnthropic,
    decode: decodeAnthropicMessages,
    toolChoice: ANTHROPIC_TOOL_CHOICE,
    streamDialect: ANTHROPIC_STREAM
};

// Mantle Messages — `anthropic-version: 2023-06-01` header (added by transport, §7.3).
//
// Reuses ANTHROPIC_STREAM: Mantle serves Anthropic's Messages API verbatim, so it
// emits the SAME events as `InvokeModelWithResponseStream` — only the framing
// differs (SSE here, event-stream there), and framing is chosen from the route
// family, not the dialect.
final readonly & ModelCodec MANTLE_MESSAGES_CODEC = {
    encode: encodeMantleMessages,
    decode: decodeAnthropicMessages,
    toolChoice: ANTHROPIC_TOOL_CHOICE,
    streamDialect: ANTHROPIC_STREAM,
    streamFields: {"stream": true}
};

// Mantle Responses — OpenAI Responses API (GPT-5.x on `/openai/v1/responses`, §7.3).
final readonly & ModelCodec MANTLE_RESPONSES_CODEC = {
    encode: encodeResponses,
    decode: decodeResponses,
    // Responses forces tools with a FLAT `tool_choice`, unlike the Chat Completions
    // codecs below — same vendor, different dialect.
    toolChoice: RESPONSES_TOOL_CHOICE,
    streamDialect: RESPONSES_STREAM,
    streamFields: {"stream": true}
};

// Mantle Chat Completions — OpenAI chat shape (GLM on `/v1/chat/completions`, §7.3).
//
// `stream_options` is not optional decoration: on this dialect a streamed response
// reports NO usage at all unless the request asks for it, so omitting it would hand
// every Mantle chat caller a stream with no token counts — and leave the observe
// span reporting zero where the buffered path reports the real numbers.
final readonly & ModelCodec MANTLE_CHAT_CODEC = {
    encode: encodeOpenAIChat,
    decode: decodeOpenAIChat,
    toolChoice: OPENAI_CHAT_TOOL_CHOICE,
    streamDialect: OPENAI_CHAT_STREAM,
    streamFields: {"stream": true, "stream_options": {"include_usage": true}}
};

// Nova InvokeModel — `schemaVersion: messages-v1`; Converse-shaped response (§7.2).
// Nova's Invoke body is Converse-shaped, so it forces tools the CONVERSE way even
// though the route family is INVOKE — see `ToolChoiceStyle`.
final readonly & ModelCodec INVOKE_NOVA_CODEC = {
    encode: encodeNovaInvoke,
    decode: decodeConverse,
    toolChoice: CONVERSE_TOOL_CHOICE,
    streamDialect: CONVERSE_STREAM
};

// OpenAI-shaped InvokeModel — GPT-OSS, Qwen, DeepSeek (§7.2). NOT Mistral: that
// dialect differs on stop_reason, tool_choice, and usage — see `codec_mistral.bal`.
// No `streamFields`: on `bedrock-runtime` the streaming request body is identical
// to the buffered one — `InvokeModelWithResponseStream` is a different operation,
// not a flag — and a stray `"stream": true` would reach the vendor as an unknown
// inference parameter.
final readonly & ModelCodec INVOKE_OPENAI_CHAT_CODEC = {
    encode: encodeOpenAIChat,
    decode: decodeOpenAIChat,
    toolChoice: OPENAI_CHAT_TOOL_CHOICE,
    streamDialect: OPENAI_CHAT_STREAM
};

// Invoke-DeepSeek — text completion: `prompt` → `choices[].text`. NOT the OpenAI
// chat shape, despite the shared `choices` wrapper (§7.2).
final readonly & ModelCodec INVOKE_DEEPSEEK_CODEC = {
    encode: encodeDeepSeekInvoke,
    decode: decodeDeepSeekInvoke,
    toolChoice: NO_TOOL_CHOICE,
    streamDialect: TEXT_COMPLETION_STREAM
};

// Invoke-Mistral chat completion — `messages`/`choices`, `tool_choice: "any"` (§7.2).
// Streams through OPENAI_CHAT_STREAM despite having its own codec: the two dialects
// differ on the buffered path (`stop_reason`, `tool_choice: "any"`, no `usage`) but
// a streamed chunk differs only in that stop-reason spelling, which the shared
// decoder already absorbs — see stream_openai_chat.bal.
final readonly & ModelCodec INVOKE_MISTRAL_CHAT_CODEC = {
    encode: encodeMistralChat,
    decode: decodeMistralChat,
    toolChoice: MISTRAL_TOOL_CHOICE,
    streamDialect: OPENAI_CHAT_STREAM
};

// Invoke-Mistral text completion — `prompt`/`outputs`; no tools at all (§7.2).
final readonly & ModelCodec INVOKE_MISTRAL_TEXT_CODEC = {
    encode: encodeMistralText,
    decode: decodeMistralText,
    toolChoice: NO_TOOL_CHOICE,
    streamDialect: TEXT_COMPLETION_STREAM
};

// Selects the codec for a resolved route (design §7.2, §7.3). Runs at construction
// (§6). Fails cleanly for wire dialects / vendors not yet supported.
isolated function selectCodec(Route route, ModelSchema? schema = ()) returns readonly & ModelCodec|error {
    if route.family == CONVERSE {
        return CONVERSE_CODEC; // model-agnostic — serves every vendor (§5)
    }
    if route.family == MANTLE {
        MantleEntry entry = check route.mantleEntry.ensureType();
        match entry.codec {
            MESSAGES_CODEC => {
                return MANTLE_MESSAGES_CODEC;
            }
            RESPONSES_CODEC => {
                return MANTLE_RESPONSES_CODEC;
            }
            CHAT_CODEC => {
                return MANTLE_CHAT_CODEC;
            }
        }
        return error(string `unknown Mantle codec '${entry.codec}'`);
    }
    // INVOKE — keyed by modelSchema (imported ARNs) or vendor prefix (§7.2).
    return selectInvokeCodec(route.bareModelId, schema);
}

// Picks the InvokeModel codec by explicit `modelSchema` (imported ARNs) or the
// bare id's vendor prefix (design §5.4, §7.2).
isolated function selectInvokeCodec(string bareModelId, ModelSchema? schema) returns readonly & ModelCodec|error {
    if schema is ModelSchema {
        match schema {
            ANTHROPIC => {
                return INVOKE_ANTHROPIC_CODEC;
            }
            NOVA => {
                return INVOKE_NOVA_CODEC;
            }
            OPENAI => {
                return INVOKE_OPENAI_CHAT_CODEC;
            }
            DEEPSEEK => {
                return INVOKE_DEEPSEEK_CODEC;
            }
            MISTRAL => {
                return INVOKE_MISTRAL_CHAT_CODEC;
            }
            MISTRAL_TEXT => {
                return INVOKE_MISTRAL_TEXT_CODEC;
            }
            LLAMA => {
                return error("Llama Invoke codec is intentionally out of scope for this build (CLAUDE.md §3)");
            }
        }
    }
    if bareModelId.startsWith("anthropic.") {
        return INVOKE_ANTHROPIC_CODEC;
    }
    // `amazon.` is Amazon's whole first-party namespace, not a Nova-only one: the
    // Titan text models live there too and take a completely different Invoke body
    // (`inputText` + `textGenerationConfig`, not `schemaVersion: messages-v1`). Match
    // Nova exactly and let `amazon.titan-*` fall through to the trailing error, which
    // already names `modelSchema` and Converse as the remedies.
    // https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
    if bareModelId.startsWith("amazon.nova") {
        return INVOKE_NOVA_CODEC;
    }
    if bareModelId.startsWith("mistral.") {
        return usesMistralTextDialect(bareModelId) ? INVOKE_MISTRAL_TEXT_CODEC : INVOKE_MISTRAL_CHAT_CODEC;
    }
    if bareModelId.startsWith("deepseek.") {
        // R1 is text completion; V3.x is OpenAI-shaped chat — see codec_deepseek.bal.
        return usesDeepSeekTextDialect(bareModelId) ? INVOKE_DEEPSEEK_CODEC : INVOKE_OPENAI_CHAT_CODEC;
    }
    if bareModelId.startsWith("openai.") || bareModelId.startsWith("qwen.") ||
        bareModelId.startsWith("zai.") {
        return INVOKE_OPENAI_CHAT_CODEC;
    }
    return error(string `no InvokeModel codec for '${bareModelId}'; pass 'modelSchema' or use Converse`);
}

// Mistral ids that speak the `prompt`/`outputs` TEXT-completion dialect on
// InvokeModel. Everything else under `mistral.` speaks chat completion.
//
// This is an allowlist of the legacy dialect rather than the reverse because the
// two are told apart only by exact id — `mistral-large-2402` is text-completion
// while `mistral-large-2407` is chat-completion, same family, four months apart.
// New ids therefore default to chat, and a wrong guess surfaces as a Bedrock
// `ValidationException` the caller can act on; `modelSchema: MISTRAL_TEXT` is the
// escape hatch.
//
// text:  https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-text-completion.html
//        (supported models: Mistral 7B Instruct, Mixtral 8X7B; AWS's own InvokeModel
//        examples use this dialect for `mistral.mistral-large-2402-v1:0`)
// chat:  https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-large-2407.html
isolated function usesMistralTextDialect(string bareModelId) returns boolean =>
    bareModelId.startsWith("mistral.mistral-7b-instruct") ||
    bareModelId.startsWith("mistral.mixtral-") ||
    bareModelId.startsWith("mistral.mistral-large-2402");

// DeepSeek ids that speak the `prompt`/`choices[].text` TEXT-completion dialect on
// InvokeModel. Everything else under `deepseek.` speaks OpenAI-shaped chat
// completion (`messages`/`choices[].message`), same as the Mistral split above.
//
// R1 only. V3.1 (`deepseek.v3-v1:0`) and V3.2 (`deepseek.v3.2`) both take
// `{"messages": [...], "max_tokens": n}` on InvokeModel per their model cards, and a
// live V3.2 Invoke call against us-east-1 confirms it: sending `prompt` comes back
// `ValidationException ... missing field messages`.
//
// CONFLICTING FIRST-PARTY SOURCES on R1 — surfaced, not silently resolved:
// the DeepSeek parameters page documents the full text-completion request/response
// for R1 (and calls V3.1 text-completion too, which the V3.1 card contradicts),
// while R1's own model card now shows an Invoke sample using `messages`. We keep R1
// on the dialect that has a documented RESPONSE shape (`choices[].text` +
// `stop_reason`) — decoding is only defined for that pairing — and default every
// other DeepSeek id to chat.
//
// text:  https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html
// V3.2:  https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html
// V3.1:  https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-1.html
// R1:    https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-r1.html
isolated function usesDeepSeekTextDialect(string bareModelId) returns boolean =>
    bareModelId.startsWith("deepseek.r1");
