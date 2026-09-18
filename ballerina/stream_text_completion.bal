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

// The two Invoke TEXT-COMPLETION dialects -> `ai:ChatMessageChunk`.
//
//   Mistral text  {"outputs": [{"text": string, "stop_reason": string}]}
//   DeepSeek R1   {"choices": [{"text": string, "stop_reason": string}]}
//
// ONE decoder for both. They differ only in the name of the wrapping array — same
// members, same stop reasons, no tools (neither dialect models them), no `usage`.
// They get separate CODECS because their REQUESTS differ (a `<s>[INST]…[/INST]`
// template versus DeepSeek's `<|User|>` markers, and a different response field to
// read); on the stream there is nothing left to tell them apart, so the decoder
// accepts either key rather than being instantiated twice with a field name.
//
// SHAPE DERIVED, NOT QUOTED. AWS documents that both models stream
// (`InvokeModelWithResponseStream` is named on both pages) and documents the
// buffered body above, but neither page prints a streamed chunk. This follows the
// Bedrock convention every other Invoke vendor observes — each frame repeats the
// buffered shape with a PARTIAL `text`, and the last one carries
// `amazon-bedrock-invocationMetrics` — which is also what `INVOKE_NOVA_CONVERTER`'s
// metrics frame demonstrates live. Accepting both array keys and treating a missing
// one as "nothing to surface" keeps a wrong guess from erroring a good stream.
//
// text:     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-text-completion.html
// deepseek: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html

# Decodes one text-completion stream.
class TextCompletionStreamDecoder {
    *StreamChunkDecoder;

    isolated function decode(string eventType, json payload) returns StreamUpdate|ai:Error? {
        map<json> p = payload is map<json> ? payload : {};

        json[]? items = arrField(p, "outputs") ?: arrField(p, "choices");
        string? text = ();
        ai:FinishReason? finishReason = ();
        if items is json[] && items.length() > 0 {
            json first = items[0];
            if first is map<json> {
                text = strField(first, "text");
                finishReason = mapOpenAIFinishReason(strField(first, "stop_reason"));
            }
        }

        StreamUpdate update = {};
        ai:ChatMessageChunk chunk = {role: ai:ASSISTANT, finishReason};
        if text is string && text != "" {
            chunk.content = text;
        }
        if chunk.content is string || finishReason is ai:FinishReason {
            update.chunk = chunk;
        }
        // Bedrock staples the token counts onto the last frame. It is the ONLY
        // usage either dialect ever reports — their buffered bodies carry no
        // `usage` object at all — and that frame can arrive alongside the final
        // text or on its own.
        StreamUsage? usage = invocationMetricsUsage(p);
        if usage is StreamUsage {
            update.usage = usage;
        }
        return update.length() == 0 ? () : update;
    }
}
