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

// OpenAI Responses stream events -> `ai:ChatCompletionChunk`. Mantle only
// (`/openai/v1/responses` for GPT-5.x and Gemma 4, `/v1/responses` elsewhere).
//
// STRUCTURALLY UNLIKE THE OTHER DIALECTS. The rest stream DELTAS of one implicit
// message; Responses streams a LIFECYCLE over a list of output ITEMS, each with its
// own index, and names every step:
//
//     response.created              the response object: id, model
//     response.output_item.added    a new item — a `message` or a `function_call`
//     response.output_text.delta    a text fragment of a message item
//     response.function_call_arguments.delta   an argument fragment of a call item
//     response.reasoning_summary_text.delta    a fragment of the reasoning summary
//     response.output_item.done     the item is complete
//     response.completed            the whole response, with `usage`
//
// Two consequences the mapping has to absorb:
//
//   1. THERE IS NO `finish_reason`. The lifecycle terminal event carries a `status`
//      instead, so the reason is derived — see `responsesFinishReason`.
//   2. TOOL-CALL IDENTITY AND ARGUMENTS ARRIVE ON DIFFERENT EVENTS, correlated by
//      `output_index` — the item ordinal, which counts messages and reasoning items
//      too, so it needs the same remap onto tool-call ordinals every other dialect
//      needs (`ToolIndexMap`).
//
// The event name arrives BOTH as the SSE `event:` line and as the payload's `type`.
// The payload wins, with the header as the fallback: it is the one field guaranteed
// present whichever reader delivered the event.
//
// https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html

// Event `type` values.
const string RESPONSES_EVT_CREATED = "response.created";
const string RESPONSES_EVT_OUTPUT_ITEM_ADDED = "response.output_item.added";
const string RESPONSES_EVT_OUTPUT_TEXT_DELTA = "response.output_text.delta";
const string RESPONSES_EVT_FUNCTION_ARGS_DELTA = "response.function_call_arguments.delta";
const string RESPONSES_EVT_REASONING_SUMMARY_DELTA = "response.reasoning_summary_text.delta";
const string RESPONSES_EVT_REASONING_TEXT_DELTA = "response.reasoning_text.delta";
const string RESPONSES_EVT_COMPLETED = "response.completed";
const string RESPONSES_EVT_INCOMPLETE = "response.incomplete";
const string RESPONSES_EVT_FAILED = "response.failed";
const string RESPONSES_EVT_ERROR = "error";

# Decodes one Responses stream.
class ResponsesStreamDecoder {
    *StreamChunkDecoder;

    private final ToolIndexMap toolIndex = new;
    // Whether any tool call was streamed. The terminal event reports a `status`, not
    // a finish reason, so this is what separates a reply that ended because the
    // model wants a tool run from one that simply finished talking.
    private boolean sawToolCall = false;

    isolated function decode(string eventType, json payload) returns ai:ChatCompletionChunk|ai:Error? {
        map<json> p = payload is map<json> ? payload : {};
        string 'type = strField(p, "type") ?: eventType;

        match 'type {
            RESPONSES_EVT_CREATED => {
                return self.decodeCreated(p);
            }
            RESPONSES_EVT_OUTPUT_ITEM_ADDED => {
                return self.decodeItemAdded(p);
            }
            RESPONSES_EVT_OUTPUT_TEXT_DELTA => {
                string? text = strField(p, "delta");
                return text is string ? singleChoiceChunk({content: text}) : ();
            }
            RESPONSES_EVT_FUNCTION_ARGS_DELTA => {
                return self.decodeArgumentsDelta(p);
            }
            RESPONSES_EVT_REASONING_SUMMARY_DELTA|RESPONSES_EVT_REASONING_TEXT_DELTA => {
                // GPT-5.x returns a SUMMARY of its reasoning, not the raw trace —
                // readable text either way, so it populates `reasoning` exactly as
                // Claude's thinking does. It never joins `content`.
                string? reasoning = strField(p, "delta");
                return reasoning is string ? singleChoiceChunk({reasoning}) : ();
            }
            RESPONSES_EVT_COMPLETED|RESPONSES_EVT_INCOMPLETE => {
                return self.decodeTerminal(p);
            }
            RESPONSES_EVT_FAILED|RESPONSES_EVT_ERROR => {
                return self.decodeFailure(p);
            }
        }
        // Every other lifecycle event — `response.in_progress`,
        // `.output_item.done`, `.content_part.added`, `.output_text.done`, and
        // whatever OpenAI adds next. All of them restate something a delta already
        // delivered, so skipping keeps a duplicate out of the caller's text.
        return ();
    }

    // `response.created` opens the stream with the response object: the id a caller
    // should see on every chunk, and the model that actually served the request.
    private isolated function decodeCreated(map<json> p) returns ai:ChatCompletionChunk? {
        ai:ChatCompletionChunk chunk = singleChoiceChunk({role: ai:ASSISTANT});
        map<json>? response = mapField(p, "response");
        if response is () {
            return chunk;
        }
        string? id = strField(response, "id");
        if id is string {
            chunk.id = id;
        }
        string? model = strField(response, "model");
        if model is string {
            chunk.model = model;
        }
        return chunk;
    }

    // `response.output_item.added` is the ONLY event carrying a tool call's id and
    // name; the arguments arrive later, on their own events. A `message` or
    // `reasoning` item has nothing to announce — its content streams as deltas.
    private isolated function decodeItemAdded(map<json> p) returns ai:ChatCompletionChunk? {
        map<json>? item = mapField(p, "item");
        if item is () || strField(item, "type") != "function_call" {
            return ();
        }
        self.sawToolCall = true;
        ai:ToolCallChunk chunk = {
            index: self.toolIndex.indexFor(intField(p, "output_index") ?: 0),
            // `call_id` is what a tool RESULT must be addressed to; `id` is the
            // item's own handle. The contract wants the former.
            id: strField(item, "call_id") ?: strField(item, "id"),
            'function: {name: strField(item, "name")}
        };
        return singleChoiceChunk({toolCalls: [chunk]});
    }

    // Tool arguments as partial JSON, keyed by the same item ordinal the opening
    // `output_item.added` used, so a caller can join the two halves.
    private isolated function decodeArgumentsDelta(map<json> p) returns ai:ChatCompletionChunk? {
        string? partial = strField(p, "delta");
        if partial is () {
            return ();
        }
        self.sawToolCall = true;
        ai:ToolCallChunk chunk = {
            index: self.toolIndex.indexFor(intField(p, "output_index") ?: 0),
            'function: {arguments: partial}
        };
        return singleChoiceChunk({toolCalls: [chunk]});
    }

    // `response.completed` / `.incomplete` close the stream with the finished
    // response object, which is where `usage` lives.
    private isolated function decodeTerminal(map<json> p) returns ai:ChatCompletionChunk? {
        map<json> response = mapField(p, "response") ?: {};
        ai:ChatCompletionChunk chunk = singleChoiceChunk({},
                responsesFinishReason(response, self.sawToolCall));
        map<json>? usage = mapField(response, "usage");
        if usage is map<json> {
            ai:CompletionTokenUsage mapped = {};
            int? inputTokens = intField(usage, "input_tokens");
            int? outputTokens = intField(usage, "output_tokens");
            int? totalTokens = intField(usage, "total_tokens");
            if inputTokens is int {
                mapped.promptTokens = inputTokens;
            }
            if outputTokens is int {
                mapped.completionTokens = outputTokens;
            }
            // Responses reports a total; derive it only when it did not, so the
            // caller sees the service's own number wherever there is one.
            if totalTokens is int {
                mapped.totalTokens = totalTokens;
            } else if inputTokens is int && outputTokens is int {
                mapped.totalTokens = inputTokens + outputTokens;
            }
            chunk.usage = mapped;
        }
        return chunk;
    }

    // `response.failed` / `error`. Surfaced as an error rather than skipped: the
    // generation has failed, and ending the stream quietly would look to a caller
    // like a clean, complete answer.
    private isolated function decodeFailure(map<json> p) returns ai:Error {
        map<json> response = mapField(p, "response") ?: p;
        map<json> err = mapField(response, "error") ?: mapField(p, "error") ?: {};
        string detail = strField(err, "message") ?: strField(p, "message") ?: "unknown";
        string kind = strField(err, "code") ?: strField(err, "type") ?: "error";
        return error ai:LlmError(string `Bedrock stream error (${kind}): ${detail}`);
    }
}

// The finished response object -> `ai:FinishReason`.
//
// The Responses API has no `finish_reason` field: it reports a lifecycle `status`,
// and says WHY separately in `incomplete_details.reason`. So the reason is derived —
// and a completed response that produced tool calls is TOOL_CALLS, not STOP, because
// that is what an agent loop reads to decide whether to run a tool and continue.
isolated function responsesFinishReason(map<json> response, boolean sawToolCall) returns ai:FinishReason? {
    string? status = strField(response, "status");
    if status == "incomplete" {
        map<json>? details = mapField(response, "incomplete_details");
        string? reason = details is map<json> ? strField(details, "reason") : ();
        if reason == "content_filter" {
            return ai:CONTENT_FILTER;
        }
        // `max_output_tokens`, and anything else that truncates a reply.
        return ai:LENGTH;
    }
    if status == "completed" {
        return sawToolCall ? ai:TOOL_CALLS : ai:STOP;
    }
    // `failed`/`cancelled` reach `decodeFailure` instead, and an unknown status
    // returns `()` rather than a guess.
    return ();
}
