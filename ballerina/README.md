## Overview

This module provides native [Ballerina `ai`](https://central.ballerina.io/ballerina/ai/latest) model and
embedding providers for **AWS Bedrock**, implementing the standard `ai:ModelProvider` and
`ai:EmbeddingProvider` contracts.

Bedrock exposes LLMs through **two endpoints with incompatible wire contracts**, and this module hides
both:

| Endpoint | Inference APIs | Signing scope |
| --- | --- | --- |
| `bedrock-runtime.{region}.amazonaws.com` | InvokeModel, Converse | `bedrock` |
| `bedrock-mantle.{region}.api.aws` | Responses, Chat Completions, Messages | `bedrock-mantle` |

Claude Mythos Preview, GPT-5.5, and GPT-5.4 live **only** on `bedrock-mantle` — a Converse-only provider
cannot reach them at all. That is the reason this module exists.

### Key features

- Chat completion across seven vendors through one `ai:ModelProvider` contract
- Structured output (`generate()`) with native tool-forcing on Converse and InvokeModel
- Text embeddings through the `ai:EmbeddingProvider` contract, with order-preserving batching
- Automatic endpoint, dialect, and SigV4 signing-scope resolution per model id
- The full AWS credential chain (IMDSv2, ECS, EKS IRSA, SSO, profiles, `AssumeRole`) via
  `ballerinax/aws.auth` — zero credential configuration on AWS compute — plus Bedrock API keys (bearer)
- Cross-region inference (CRIS) profiles, provisioned and custom-deployment ARNs
- Guardrail support on both `bedrock-runtime` inference APIs
- Image input on Converse and Anthropic Messages; a named error, never a silent drop, elsewhere

### Providers

The public surface is split **by vendor**. Each class is a thin typed facade over one shared internal
spine (resolver → endpoint builder → converter → SigV4 transport).

**Chat** — `AnthropicModelProvider`, `OpenAIModelProvider`, `AmazonModelProvider` (Nova),
`MistralModelProvider`, `QwenModelProvider`, `GoogleModelProvider` (Gemma), `DeepSeekModelProvider`.

**Embeddings** — `TitanEmbeddingProvider`, `CohereEmbeddingProvider`.

**Knowledge base** — `BedrockManagedKnowledgeBase` (Bedrock owns the vector store) and
`BedrockVectorKnowledgeBase` (you own it), both implementing `ai:KnowledgeBase`. See
[Knowledge bases](#knowledge-bases) and [Self-managed knowledge bases](#self-managed-knowledge-bases).

A model AWS ships before this module updates an enum is still usable — pass its id as a `string`. Every
provider takes `<Vendor>Model|string`, so the enums are autocomplete and documentation, never a gate.
Passing a raw string skips the enum, not the routing: an id the resolver does not recognise resolves to
**Converse**, never to Mantle. A brand-new *Mantle-only* model is the one case that needs a module
release, because its request path cannot be derived from its id. See [Escape hatches](#escape-hatches).

> **A vendor is not an endpoint.** Which class you pick does not tell you which endpoint you reach:
> **Gemma 3** is dual-homed and defaults to `bedrock-mantle` (its Mantle path — `/v1/chat/completions` —
> even differs from **Gemma 4**'s `/openai/v1/responses`), while a runtime-only Claude like Sonnet 4.6
> stays on Converse. The module resolves this per model id, so you don't have to — but any model that
> resolves to Mantle needs the Mantle IAM permission below and cannot do structured output.

## Prerequisites

Before using this module in your Ballerina application, complete the following:

1. Create an [AWS account](https://portal.aws.amazon.com/billing/signup).
2. [Request access to the Bedrock foundation models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
   you intend to use, in the region you intend to call.
3. Arrange credentials. On EC2, ECS, EKS or Lambda there is **nothing to do** — the default
   credential chain picks up the instance profile, task role, or IRSA service account. Elsewhere,
   supply IAM access keys, an assumed role, a named profile, or a
   [Bedrock API key](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html).
4. Attach the IAM permissions for the endpoint you will reach: `bedrock:InvokeModel` for
   Converse/InvokeModel, and **additionally `bedrock-mantle:CreateInference`** for Mantle routes. See
   [Mantle needs a separate IAM permission](#mantle-needs-a-separate-iam-permission).

## Quickstart

To use the `ai.aws.bedrock` module in your Ballerina application, update the `.bal` file as follows:

### Step 1: Import the module

```ballerina
import ballerina/ai;
import ballerinax/ai.aws.bedrock;
```

### Step 2: Initialize the model provider

Only the model is required. Region falls back to `AWS_REGION`/`AWS_DEFAULT_REGION`, and credentials
to the AWS credential chain — so on AWS compute this is the whole thing:

```ballerina
final ai:ModelProvider claude = check new bedrock:AnthropicModelProvider(bedrock:CLAUDE_SONNET_4_6);
```

Every vendor follows the same shape — `(model, credentials?, region?, serviceUrl?, maxTokens?,
temperature?, *Config)`. Only `model` is positional-required, so pass `region` by name when
you are not also passing credentials:

```ballerina
final ai:ModelProvider nova = check new bedrock:AmazonModelProvider(bedrock:NOVA_PRO, region = "us-east-1");
final ai:ModelProvider gpt = check new bedrock:OpenAIModelProvider(bedrock:GPT_5_4, region = "us-east-2");
final ai:ModelProvider gemma = check new bedrock:GoogleModelProvider(bedrock:GEMMA_3_27B_IT, region = "us-east-1");
```

### Credentials

`credentials` defaults to `auth:DEFAULT_CREDENTIALS`, which walks the standard AWS chain —
environment variables, EKS IRSA web identity, IAM Identity Center (SSO), the shared config file,
`credential_process`, ECS container credentials, then EC2 IMDSv2 — with expiry and refresh handled
for you. **On EC2, ECS, EKS and Lambda you do not configure credentials at all.**

To be explicit, pass any [`ballerinax/aws.auth`](https://central.ballerina.io/ballerinax/aws/latest)
config, or a Bedrock API key:

```ballerina
import ballerinax/aws.auth;

// Long-lived keys (add `sessionToken` for temporary STS credentials).
bedrock:BedrockCredentials keys = {accessKeyId: "...", secretAccessKey: "..."};

// Cross-account: assume a role in another account.
bedrock:BedrockCredentials role = {
    roleArn: "arn:aws:iam::222222222222:role/IntegratorRole",
    externalId: "optional-for-third-party-access"
};

// A named profile from ~/.aws/credentials.
bedrock:BedrockCredentials profile = {profileName: "prod"};

// A Bedrock API key (bearer) — bypasses SigV4 entirely.
bedrock:BedrockCredentials apiKey = {apiKey: "..."};

final ai:ModelProvider claude =
    check new bedrock:AnthropicModelProvider(bedrock:CLAUDE_SONNET_4_6, role, "us-east-1");
```

> **Knowledge bases do NOT accept Bedrock API keys.** AWS states API keys "are limited to Amazon
> Bedrock and Amazon Bedrock Runtime actions" and cannot be used with *"Agents for Amazon Bedrock or
> Agents for Amazon Bedrock Runtime API operations"* — and both knowledge base planes
> (`bedrock-agent`, `bedrock-agent-runtime`) are exactly those. `BedrockManagedKnowledgeBase`
> therefore takes `KnowledgeBaseCredentials` (SigV4 only), so a bearer token is rejected at
> **compile time** rather than becoming an opaque runtime 403.

### Step 3: Invoke chat completion

```ballerina
ai:ChatAssistantMessage response = check claude->chat([
    {role: ai:SYSTEM, content: "Be brief."},
    {role: ai:USER, content: "Why is the sky blue?"}
]);
```

### Step 3b: Stream the reply

```ballerina
stream<ai:ChatCompletionChunk, ai:Error?> chunks = check claude->chatStream([
    {role: ai:USER, content: "Why is the sky blue?"}
]);
check from ai:ChatCompletionChunk chunk in chunks
    do {
        io:print(chunk.choices[0].delta.content ?: "");
    };
```

`generateStream()` is the text-only shortcut, projecting the same stream onto its content fragments and
skipping the role opener, tool-call fragments and the usage closer:

```ballerina
stream<string, ai:Error?> text = check claude->generateStream(`Why is the sky blue?`);
```

**Every route streams** — Converse, Invoke and Mantle, for every vendor. The wire differs, the contract
does not: `bedrock-runtime` answers AWS's binary event-stream from a sibling operation
(`converse-stream`, `invoke-with-response-stream`), while Mantle answers SSE from the *same* path,
switched on by `"stream": true` in the body.

Beyond `delta.content` a chunk may carry `delta.reasoning` (extended thinking — Claude and Nova stream
it as readable text), `delta.toolCalls` (partial-JSON argument fragments, correlated by `index`),
`finishReason` on the closing chunk, and `usage` on the final one.

**Only `string` streams.** `generateStream()` with any other target type is an `ai:Error`: a typed value
comes out of a forced tool call, and the arguments cannot be bound until the whole JSON has arrived —
there is no partial record to hand back. Use `generate()` for typed results.

**Close a stream you stop reading early.** Breaking out of the loop leaves the HTTP response — and its
pooled connection — open; `chunks.close()` releases it and the observability span. Draining to the end
closes itself.

**A mid-stream failure arrives on an HTTP 200.** Throttling, a model error or a guardrail trip is
delivered *inside* the stream (an exception frame on the event-stream wire, an `error` event on SSE) and
surfaces as an `ai:Error` rather than a short answer that looks complete. Retries cover only the
handshake: once chunks are delivered, re-sending would duplicate the answer rather than resume it.

### Step 4: Generate structured output

```ballerina
type Review record {| string sentiment; int score; |};

Review review = check claude->generate(`Rate this review: ${text}`);
```

> **Under `AUTO`, `generate()` falls back to Converse by itself.** A Mantle-capable model routes `chat()`
> to `bedrock-mantle`, which has no structured output — so when the same model is **also** served on
> `bedrock-runtime` (`CLAUDE_OPUS_5`, `CLAUDE_OPUS_4_8`, `CLAUDE_SONNET_5`, `CLAUDE_HAIKU_4_5`,
> `DEEPSEEK_V3_2`, Gemma 3, GLM 5, gpt-oss, Qwen3, Mistral Large 3), a typed `generate()` quietly
> resolves a second Converse spine and uses that. You do not have to set `apiFamily` yourself.
>
> **This means one provider can talk to two endpoints, which need two different IAM permissions:**
> `bedrock-mantle:CreateInference` for `chat()` and `bedrock:InvokeModel` for a typed `generate()`.
> Credentials holding only one will see the other path return 403.
>
> Two cases still return an `ai:Error` for a non-`string` target:
> - **Mantle-only models** (`GPT_5_4`, `GPT_5_5`, `CLAUDE_MYTHOS_5`, `CLAUDE_MYTHOS_PREVIEW`, Gemma 4).
>   There is no `bedrock-runtime` route to fall back to.
> - **An explicit `apiFamily = bedrock:MANTLE`.** The fallback is an `AUTO` convenience; naming a
>   destination explicitly is respected rather than silently overridden.
>
> A `string` target always returns text normally, and `chat()` is unaffected in every case.
>
> It is also unavailable on Mistral's **text-completion** dialect (see below), which has no tool-calling
> at all. This only bites when you force `apiFamily = INVOKE` on those ids — the default Converse route
> supports typed generation for every Mistral model.

### Step 5: Generate embeddings

```ballerina
final ai:EmbeddingProvider titan = check new bedrock:TitanEmbeddingProvider(
    bedrock:TITAN_EMBED_TEXT_V2, creds, "us-east-1", dimensions = 1024);

ai:Embedding vector = check titan->embed({content: "hello", 'type: "text-chunk"});
```

`batchEmbed` preserves input order. Note the wire asymmetry: Titan's `inputText` is a single string, so
n chunks is n sequential round trips; Cohere batches up to 96 per call.

## Three callouts that will silently cost you

### Mantle needs a separate IAM permission

**Mantle is a different IAM namespace.** Working `bedrock:InvokeModel` permissions are **not** enough —
Mantle requires **`bedrock-mantle:CreateInference`**, with its own managed policies. Without it you get
`AccessDenied` from a service you never meant to call, with no clue why. (This module's 403 error message
names the action for you.)

See [IAM for Bedrock powered by AWS Mantle](https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonbedrockpoweredbyawsmantle.html).

### Streaming needs its own IAM action on `bedrock-runtime`

**`bedrock:InvokeModelWithResponseStream` is a separate action** — `ConverseStream` included, despite
being authorized separately from `Converse`. So a role that calls `chat()` fine can be denied on
`chatStream()` alone. Mantle is the exception: `bedrock-mantle:CreateInference` authorizes streaming and
non-streaming alike.

### Cohere `inputType` decides your retrieval quality

**Cohere requires `input_type` on every request, and getting it wrong degrades retrieval silently** — no
error, no exception, just worse results. Use **`SEARCH_DOCUMENT` for your corpus** and **`SEARCH_QUERY`
for your queries**, constructing one provider per role:

```ballerina
// Ingest side
final ai:EmbeddingProvider ingest = check new bedrock:CohereEmbeddingProvider(
    bedrock:COHERE_EMBED_ENGLISH_V3, creds, "us-east-1", inputType = bedrock:SEARCH_DOCUMENT);

// Query side
final ai:EmbeddingProvider query = check new bedrock:CohereEmbeddingProvider(
    bedrock:COHERE_EMBED_ENGLISH_V3, creds, "us-east-1", inputType = bedrock:SEARCH_QUERY);
```

The `ai:EmbeddingProvider` contract carries no query-vs-document signal, which is exactly why this is
config rather than a method argument. It defaults to `SEARCH_DOCUMENT`.

## Routing

You pick a model; the module picks the wire dialect. The resolver runs once, at construction.

Under `AUTO` (the default), the preference order is **Mantle → Converse → Invoke**: any model with a
verified Mantle entry defaults to `bedrock-mantle`; Converse is chosen when the model has no Mantle
entry; Invoke is reached only by asking for it (`apiFamily = bedrock:INVOKE`).

| You pass | Resolves to |
| --- | --- |
| a Mantle-capable bare id (`anthropic.claude-opus-4-8`, `anthropic.claude-sonnet-5`, `openai.gpt-5.4`, `google.gemma-4-31b`) | Mantle, on that model's own path |
| a CRIS id (`us.anthropic.claude-opus-4-8`) | Converse, prefix re-applied on the wire |
| a bare id with no Mantle entry (`amazon.nova-pro-v1:0`, `anthropic.claude-sonnet-4-6`) | Converse |
| `provisioned-model/` · `custom-model-deployment/` · `inference-profile/` ARN | Converse |
| an unknown id | Converse — **never** Mantle |
| `imported-model/` ARN | **not supported** — construction error |

> **`generate()` with a typed (non-`string`) return errors on any model that `AUTO` sends to Mantle**,
> because Mantle has no structured-output path. To get typed generation on a dual-homed model such as
> Claude Sonnet 5 or Opus 4.8, force the runtime surface: `apiFamily = bedrock:CONVERSE` (or a
> `converse/` prefix). **`chat()` is unaffected** — it works the same on either route. A CRIS-prefixed id
> already resolves to Converse, so it is unaffected too.

A CRIS geo prefix (`us.`, `eu.`, …) is a `bedrock-runtime` concept — Mantle has no geo prefixes — so a
geo-prefixed id always stays on Converse regardless of the Mantle preference.

Mantle is matched by the `MANTLE_CAPABLE` table only. A model's *absence* from
that table is never taken as evidence it is a Mantle model: an unknown id sinks to Converse, so a typo
can never become a cryptic 403 from a different service with a different IAM namespace.

### Escape hatches

```ballerina
// 1. Force a family (default is AUTO, which runs the resolver)
check new bedrock:AnthropicModelProvider("anthropic.claude-haiku-4-5", creds, "us-east-1",
        apiFamily = bedrock:MANTLE);

// 2. Prefix override on the model string
check new bedrock:AnthropicModelProvider("mantle/anthropic.claude-haiku-4-5", creds, "us-east-1");

// 3. Any raw model id string is always accepted — the model enums are
//    conveniences, never a gate. A model AWS shipped after this release works today.
check new bedrock:AmazonModelProvider("amazon.nova-something-new-v1:0", creds, "us-east-1");
```

**A brand-new model needs no module release to reach Converse or Invoke** — pass its id as a string.
The one exception is a brand-new **Mantle** model: its request path is per-model data that cannot be
derived from the id, so it needs a table entry. Forcing `apiFamily = MANTLE` on a model absent from
`MANTLE_CAPABLE` returns a construction error rather than guessing a path.

### FIPS endpoints

Set `fips` and the host comes from AWS SDK endpoint metadata — no host-name guessing:

```ballerina
check new bedrock:AnthropicModelProvider(bedrock:CLAUDE_SONNET_4_6, "us-gov-west-1",
    config = {fips: true});
// → https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com
```

> FIPS applies to `bedrock-runtime` only. There is **no** `bedrock-mantle` FIPS host, so `fips` on a
> Mantle-resolved model is a **construction error** rather than a DNS failure at call time. Use
> `apiFamily = bedrock:CONVERSE` (or `INVOKE`) for a FIPS-compliant call.

It changes only the host dialled. The SigV4 signing scope, the request path, and the body are untouched.

### Custom endpoints (`serviceUrl`)

`serviceUrl` defaults to the template `https://bedrock-{endpoint}.{region}.{domain}`. Left at its
default, the origin is resolved entirely from AWS SDK endpoint metadata — every partition, the
FIPS variants, and per-service exceptions, with a standard-pattern fallback for regions newer than the
bundled metadata. That covers `amazonaws.com.cn` in China and `api.aws` for Mantle automatically.

```ballerina
// PrivateLink VPC endpoint, or any gateway / mock server — fully literal
serviceUrl = "https://vpce-0abc.bedrock-runtime.us-east-1.vpce.amazonaws.com"

// Partial override: pin the service segment, let region and domain resolve
serviceUrl = "https://bedrock-{endpoint}.{region}.{domain}"
```

It replaces the **origin only** — the route-derived request path is still appended — and it never
changes the SigV4 scope: a VPCE or gateway host still signs the route's own region and service. A
placeholder that survives substitution (a typo like `{regoin}`) is a construction error, not a DNS
failure.

### Inference parameters

`additionalModelRequestFields` forwards anything Converse does not model (Claude `top_k`/`thinking`, Nova
`reasoningConfig`, sampling knobs beyond `temperature`, …) verbatim.

The module deliberately exposes only `maxTokens` and `temperature` as first-class inference knobs — the
two an integration developer actually reaches for. Anything finer-grained (`top_p`, `top_k`, …) goes
through `additionalModelRequestFields` rather than cluttering the config record. Note that passthrough is
honoured on Converse, Nova, OpenAI-chat, Responses, Mistral, and Invoke-DeepSeek, but **not** on the
Invoke-Anthropic converter.

> **`temperature` has no default, and that is deliberate.** Leave it unset and the field is omitted from
> the request entirely, so the model applies its own default. This is not a style choice: Anthropic
> deprecated sampling parameters on Claude 4.7 and later (`CLAUDE_OPUS_4_8`, `CLAUDE_OPUS_5`,
> `CLAUDE_SONNET_5`, `CLAUDE_MYTHOS_5`) and OpenAI's GPT-5.x reasoning models (`GPT_5_4`, `GPT_5_5`,
> `GPT_5_6_*`) never accepted them. On those models **any** value returns
> `400 temperature is deprecated for this model`, so a module-level default would make them unusable out
> of the box. Set `temperature` only for models you know accept it — Nova, Mistral, Qwen, Gemma,
> DeepSeek, GPT-OSS, and Claude 4.6 and earlier.

`maxTokens` **does** default (to 4096). It is capped per model — Nova Pro/Lite/Micro top out at 5K output
tokens — and on adaptive-thinking models the thinking pass is billed against the same ceiling, so raise
it for long reasoning tasks.

## Vendor dialects

### Mistral speaks two InvokeModel dialects

Mistral is the one vendor whose `InvokeModel` wire shape cannot be derived from its vendor prefix:

| Dialect | Models | Wire |
| --- | --- | --- |
| [text completion](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-text-completion.html) | 7B Instruct, Mixtral 8x7B, **Large 24.02** | `prompt` (`<s>[INST]…[/INST]`) → `outputs[].text`; no tools |
| [chat completion](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-large-2407.html) | **Large 24.07**, newer ids | `messages`/`tools` → `choices[].message` |

Note that `mistral-large-**2402**` and `mistral-large-**2407**` are the same family four months apart and
speak *opposite* dialects. The module picks by id; an id it has never seen defaults to chat. If it guesses wrong, Bedrock returns a
`ValidationException` — switch to `apiFamily = bedrock:CONVERSE`, which is model-agnostic and sidesteps
the split entirely.

**Converse (the default) hides all of this** — the split only matters under `apiFamily = INVOKE`.

### DeepSeek does too

Same story, split by generation rather than by date:

| Dialect | Models | Wire |
| --- | --- | --- |
| [text completion](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html) | **R1** (`deepseek.r1-v1:0`) | `prompt` (DeepSeek's `<｜User｜>` template) → `choices[].text`; no tools |
| [chat completion](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-deepseek-deepseek-v3-2.html) | **V3.1**, **V3.2**, newer ids | `messages`/`tools` → `choices[].message` |

The module picks by id, and an id it has never seen defaults to chat. Again, only relevant under
`apiFamily = INVOKE` — Converse and Mantle are unaffected.

## Knowledge bases

`BedrockManagedKnowledgeBase` implements `ai:KnowledgeBase` against a Bedrock **managed** knowledge
base (`KnowledgeBaseConfiguration.type = MANAGED` — Bedrock owns the vector store; there is nothing
to provision). It spans two additional endpoints beyond the chat/embedding surface —
`bedrock-agent.{region}.amazonaws.com` (control: create/list/get/ingest/delete) and
`bedrock-agent-runtime.{region}.amazonaws.com` (data: retrieve) — both signing as SigV4 service
`bedrock`, same as Converse/InvokeModel.

`BedrockVectorKnowledgeBase` implements the same interface against a **self-managed** knowledge base
(`KnowledgeBaseConfiguration.type = VECTOR` — a vector store you provision and own), which is the
console's *Self-managed KB → Unstructured Vector Store KB*. Both classes use the same two endpoints
and the same signing scope; see [Self-managed knowledge bases](#self-managed-knowledge-bases) below
for what differs.

### Two ways to use it

**Attach to a knowledge base you configured in AWS** — pass its id. AWS owns ingestion through its
own native connectors (S3, SharePoint, Confluence, Google Drive, OneDrive, Web Crawler) on their own
sync schedule:

```ballerina
ai:KnowledgeBase kb = check new bedrock:BedrockManagedKnowledgeBase("GKICZMNWRG", creds, "us-east-1");
ai:QueryMatch[] matches = check kb->retrieve("What is our refund policy?", 5);
```

`retrieve()` searches across **every** data source on the knowledge base. `ingest()` and
`deleteByFilter()` need the knowledge base to also have a `CUSTOM` (direct-ingestion) data source —
construction fails, naming why, if it does not have one; add one in the console, or use the
find-or-create path below.

**Create and own it end to end** — pass a `KnowledgeBaseDefinition`. This class creates the knowledge
base and a `CUSTOM` data source, and every document flows through `ingest()`:

```ballerina
ai:KnowledgeBase kb = check new bedrock:BedrockManagedKnowledgeBase(
    {
        name: "support-docs",
        roleArn: "arn:aws:iam::123456789012:role/service-role/bedrock-kb-execution-role"
    },
    creds, "us-east-1");

check kb->ingest([{content: "Refunds are processed within 5 business days."}]);
```

**Find-or-create is by NAME.** `CreateKnowledgeBase` has no upsert and names are not unique per
account, so `init` searches for an exact name match first: exactly one match attaches (no writes);
no match creates one (~83s to become `ACTIVE`, bounded by `readyTimeout`); more than one match is a
construction error naming the candidate ids — pick the id and pass it as a `string` instead.

### Chunking

A `CUSTOM` data source's `chunkingStrategy` is fixed for its lifetime. **`FIXED_SIZE`** (the default
when this class creates one) means Bedrock chunks server-side — pass `chunkingStrategy: NONE` on
`KnowledgeBaseDefinition.dataSource` to chunk client-side with an `ai:Chunker` instead:

```ballerina
ai:KnowledgeBase kb = check new bedrock:BedrockManagedKnowledgeBase(
    {
        name: "support-docs",
        roleArn: "arn:...:role/service-role/bedrock-kb-execution-role",
        dataSource: {name: "custom-source", chunkingStrategy: bedrock:NONE}
    },
    creds, "us-east-1",
    chunker = new ai:MarkdownChunker());
```

`ManagedKnowledgeBaseConfig.chunker` is **detected**, not assumed, when left unset: `init` reads the
resolved data source's actual strategy and defaults to `ai:DISABLE` when Bedrock chunks server-side,
`ai:AUTO` when it is `NONE`. Passing an explicit `ai:Chunker` against a server-chunking data source
is a construction error — Bedrock would re-split whatever is submitted, silently overwriting the
chunker's own boundaries.

> **Ingestion needs a SECOND IAM action.** `bedrock:StartIngestionJob` **and**
> `bedrock:IngestKnowledgeBaseDocuments` are both required — working `bedrock:InvokeModel`/console
> permissions are not enough, and the failure mode is an `AccessDenied` that does not name the
> missing action on its own. This module's 403 error does name it.

> **`ingest()` is slow, by design.** `IngestKnowledgeBaseDocuments` returns 202 as soon as Bedrock has
> accepted the documents, not once they are indexed — ~14s measured for a single small document, ~47s
> for a 97KB one. `ingest()` therefore blocks until every document reaches a terminal status or
> `ingestTimeout` elapses, so a `retrieve()` immediately afterward sees them. There is deliberately no
> fire-and-forget mode: `ai:KnowledgeBase.ingest` returns a bare `Error?` with no job handle and no
> status method, so returning at the 202 would report success for a document that later lands `FAILED`
> and leave you no way to ever find out.

> **`retrieve()` cannot return more than 100 results.** `Retrieve` caps `numberOfResults` at 100 and
> returns **no `nextToken`** when results are truncated, so there is nothing to page with. `maxLimit`
> above 100 (including `-1`) is bounded by this.

### `deleteByFilter` is a reconstruction

**Bedrock has no metadata-based delete API and no way to read a document's metadata back**
(`ListKnowledgeBaseDocuments` carries status and identifier only; `GetDocumentContent` returns a
presigned content URL). `deleteByFilter` reconstructs one: it enumerates every document on every data
source, then asks `Retrieve` a single yes/no question per document — the caller's filter ANDed onto
Bedrock's own `_source_uri` system attribute **pinned to that one document**.

**The pin is what makes this sound.** Sending the caller's filter alone and collecting matches in bulk
under-deletes silently: measured, a filter matching 5 documents with `numberOfResults: 10` returned
only 3, with no `nextToken` to signal the loss. Narrowing to one document removes that failure — the
candidate set is that document's chunks, so nothing can crowd it out, and pinned probes come back at
0.88–1.0 against a relevance floor near 0.15 whatever the probe query says. A non-empty result means
"matches", an empty one means "does not match", and there is no third case.

The cost is **one `Retrieve` per document in the knowledge base**, so this is a maintenance operation,
not something to put on a request path. Documents on a non-`CUSTOM`/`S3` data source (a native
connector) cannot be deleted through this API at all, and are named in the returned error; deletes
that CAN be made still happen.

### `RetrieveAndGenerate` is unusable on managed knowledge bases

AWS documents this directly: *"This API cannot be used with managed knowledge bases."* Use
`retrieve()` plus your own model provider (`ai:augmentUserQuery` bridges the two), or AWS's
`AgenticRetrieveStream` outside this module.

## Self-managed knowledge bases

`BedrockVectorKnowledgeBase` is the sibling of `BedrockManagedKnowledgeBase` for
`KnowledgeBaseConfiguration.type = VECTOR`. Same three methods, same two endpoints, same SigV4 scope.
Use it when you want control over indexing and ranking; use the managed class when you do not want to
run a vector store.

```ballerina
import ballerinax/ai.aws.bedrock;

// Attach to a knowledge base that already exists.
final bedrock:BedrockVectorKnowledgeBase kb = check new ("KB1234ABCD");
```

```ballerina
// Or create the knowledge base and its CUSTOM data source from Ballerina.
// The VECTOR STORE ITSELF MUST ALREADY EXIST — see the callout below.
final bedrock:BedrockVectorKnowledgeBase kb = check new ({
    name: "support-articles",
    roleArn: "arn:aws:iam::123456789012:role/service-role/AmazonBedrockExecutionRoleForKnowledgeBase_1",
    embeddingModelArn: "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0",
    storageConfiguration: <bedrock:OpenSearchServerlessStorage>{
        collectionArn: "arn:aws:aoss:us-east-1:123456789012:collection/abcdefghij1234567890",
        vectorIndexName: "bedrock-index",
        fieldMapping: {
            vectorField: "embeddings",
            textField: "AMAZON_BEDROCK_TEXT_CHUNK",
            metadataField: "AMAZON_BEDROCK_METADATA"
        }
    }
});
```

> **The vector store must already exist.** This class never provisions one. `CreateKnowledgeBase`
> accepts only a `storageConfiguration` naming an existing collection, cluster, table, or bucket — the
> console's "Quick create a new vector store" has **no API equivalent**: *"If you prefer to let Amazon
> Bedrock create and manage a vector store for you, use the console."* Provision it with
> Terraform/CDK/the console first, then pass its ARNs. This is why the managed class is frictionless
> by comparison: there is nothing to provision.

Eight backends are supported, one record each: `OpenSearchServerlessStorage`,
`OpenSearchManagedClusterStorage`, `S3VectorsStorage`, `RdsStorage`, `NeptuneAnalyticsStorage`,
`PineconeStorage`, `RedisEnterpriseCloudStorage`, `MongoDbAtlasStorage`. Their field mappings are
**not** interchangeable — Pinecone and Neptune Analytics have no `vectorField` at all, and RDS adds
`primaryKeyField` plus an optional `customMetadataField`.

### What differs from the managed class

| | `BedrockManagedKnowledgeBase` | `BedrockVectorKnowledgeBase` |
|---|---|---|
| Vector store | Bedrock's, nothing to provision | Yours, must pre-exist |
| Embedding model | Optional (service-managed by default) | **Required** — `embeddingModelArn` |
| Chunking | Rejected on a service-managed model | Configurable, so `ai:Chunker` is usable |
| Data source body | `MANAGED_KNOWLEDGE_BASE_CONNECTOR` wrapper | Plain `{"type": "CUSTOM"}` |
| Search branch | `managedSearchConfiguration` | `vectorSearchConfiguration` |
| Reranking | `rerankingModelType` enum | `rerankingConfiguration` record |
| Search type override | Not available | `overrideSearchType`, opt-in |
| Reserved metadata prefix | `_` (`_source_uri`) | `x-amz-bedrock` |

`startsWith` and `stringContains` are supported by Bedrock on self-managed knowledge bases and not on
managed ones, but **neither class can emit them**: `ai:MetadataFilterOperator` has exactly eight members
(`==`, `!=`, `>`, `<`, `>=`, `<=`, `in`, `nin`) and none maps to either. See
[Not implemented](#not-implemented).

### `overrideSearchType` is backend-dependent, and two AWS sources disagree

Leave it unset unless you know your backend supports the value — unset means Bedrock picks a strategy
suited to the store, which is correct everywhere. The API reference says `HYBRID` works only on
**OpenSearch Serverless** with a filterable text field; the user guide says *"Amazon RDS, Amazon
OpenSearch Serverless, and MongoDB vector stores that contain a filterable text field."* Both agree it
is unavailable on S3 Vectors, Neptune Analytics, Pinecone, and Redis. Neither is treated as
authoritative here, so nothing is defaulted.

### `ingest()` needs two IAM permissions, and AWS names only one at a time

Same as the managed class: `bedrock:StartIngestionJob` **and**
`bedrock:IngestKnowledgeBaseDocuments`. Granting the one named in the first `AccessDenied` fails again
on the other. Separately, the knowledge base's own `roleArn` needs permissions on **your** vector store
(`aoss:APIAccessAll`, `es:ESHttp*`, `rds-data:*`, `neptune-graph:*`, `s3vectors:*`, or
`secretsmanager:GetSecretValue` depending on backend). Those belong to that role, not to this client's
credentials.

### Pitfalls this module cannot check for you

Each of these needs a query against your vector store to detect — credentials and network reach the
calling application does not have, since those permissions belong to the knowledge base's service role
and the store is often VPC-private. Construction validates everything Bedrock itself reports; the rest
is on you:

- **Index dimension must match the embedding model.** A mismatch fails ingestion with an opaque error.
- **OpenSearch must use the `faiss` engine.** With `nmslib`, metadata filtering does not work at all
  and the documented fix is to rebuild the index.
- **OpenSearch custom metadata fields must be `keyword`-typed** (or `text` with a `keyword` subfield).
  Without that, filtering on them fails with a *"Rewrite first"* error.
- **S3 Vectors caps metadata at 1 KB and 35 keys per vector.** Hierarchical chunking can exceed it,
  and *"the ingestion job will throw an exception."* S3 Vectors is also SEMANTIC-only, float32-only,
  and rejects `startsWith`/`stringContains`.
- **Aurora needs HNSW iterative index scans** (pgvector 0.8.0+) when you filter on metadata. Without
  them, selective filters **silently return fewer results than they should** — no error.
- **MongoDB Atlas metadata filtering does not work by default**; filters must be configured in the
  Atlas vector index first.

### `deleteByFilter` keeps a second probe the managed class dropped

The reconstruction is the same as [the managed one](#deletebyfilter-is-a-reconstruction), with one
difference. The managed class was able to drop its follow-up probe after measuring that a pinned probe
scores 0.88–1.0 against a relevance floor near 0.15, which makes a zero-hit result unambiguous.

**That measurement was taken against Bedrock's own vector store and does not transfer here** — on a
self-managed knowledge base the ranking engine is yours. So a zero-hit probe is re-run with the
document pin alone; if that also returns nothing, the document is reported as **indeterminate** rather
than silently skipped, which would under-delete. Cost is one to two `Retrieve` calls per document.
Once the equivalent measurement exists for a backend, the second probe can be dropped exactly as the
managed class dropped its own.

A probe counts as a match only when a returned result **is** the document it pinned — checked against
`x-amz-bedrock-kb-source-uri`, `location.customDocumentLocation.id`, or `location.s3Location.uri`.
Counting results instead would mean trusting your store to honour the pin, and AWS documents at least
one backend (MongoDB Atlas) where filtering silently does nothing by default; on such a store every
probe would "match" and the whole knowledge base would be deleted. The trade is that a backend
returning none of those identity fields makes every document indeterminate, so `deleteByFilter` becomes
a no-op that reports rather than a silent mass delete. `deleteByFilter` also **rejects a filter set
with no leaf predicates** — an empty or all-empty-groups `ai:MetadataFilters` would otherwise select
everything.

> **Nothing on this class has been verified against live AWS.** The reserved attribute name
> (`x-amz-bedrock-kb-source-uri`) is documented by AWS, but whether Bedrock populates it for a CUSTOM
> data source on a self-managed knowledge base, and how your store's relevance floor behaves, are
> unmeasured. Both are flagged in code comments.


## Guardrails

| Route | Mechanism |
| --- | --- |
| Converse | `guardrailConfig` body field |
| Invoke | `X-Amzn-Bedrock-Guardrail*` request headers; the fired signal returns in the response body |
| Mantle | not supported → construction error pointing at `ApplyGuardrail` |

A fired guardrail is never silently dropped on either supported route.

## Fails fast, before any network call

Construction errors are reserved for what AWS *cannot* diagnose for you:

- an `imported-model/` ARN (AWS applies no default chat template to imported weights)
- an unresolved `{placeholder}` left in `serviceUrl`
- `apiFamily = MANTLE` on a model with no known Mantle request path
- a guardrail on a Mantle route (the error names the standalone `ApplyGuardrail` API)
- Mantle on a partition with no `api.aws` host (`aws-cn`); GovCloud **is** supported
- a `custom-model/` ARN (an artifact, not a deployment)

Everything AWS *can* tell you — a model unavailable in a region, a bad id — is left to Bedrock's own
`ValidationException`, so this module never becomes a release dependency for AWS's catalogue.

## Images

Pass an `ai:ImageDocument` inside a prompt and it is sent as a real image, not as text:

```ballerina
byte[] png = check io:fileReadBytes("invoice.png");
ai:ImageDocument invoice = {content: png, metadata: {mimeType: "image/png"}};

ai:ChatAssistantMessage answer = check claude->chat({
    role: ai:USER,
    content: `Extract the total from this invoice: ${invoice}`
});
```

**Supported on the routes below.** Everywhere else an image is a **construction-time
`ai:Error` naming the dialect** — never silently dropped into the prompt text.

| Route | Images | Notes |
| --- | --- | --- |
| Converse (and Nova on InvokeModel) | ✅ | Native `image` content block |
| Anthropic Messages (InvokeModel **and** Mantle) | ✅ | base64 source |
| OpenAI chat completions (Mantle + GPT-OSS Invoke) | ❌ | unverified — see below |
| OpenAI Responses (Mantle, GPT-5.x) | ❌ | unverified — see below |
| Mistral chat (InvokeModel) | ❌ | sources disagree — see below |
| Mistral instruct / DeepSeek-R1 | ❌ | single prompt string; no content-part array |
| Any `ChatSystemMessage` | ❌ | `system` is text-only on every Bedrock route |

`mimeType` comes from `metadata.mimeType` when set, otherwise from the download's
`Content-Type`, otherwise from the file's magic bytes. If none of those identify it,
construction fails with a named error rather than guessing — both Converse's `format`
and Anthropic's `media_type` are required fields with no wildcard, so a guess is a
guaranteed 400. Only **png, jpeg, gif and webp** are accepted.

An `ai:Url` image is **downloaded by the connector** and sent as bytes, because
Bedrock never fetches on your behalf: Converse has no URL source at all, and
Anthropic-on-Bedrock accepts base64 only. Only `http(s)` URLs are fetched, redirects
are followed manually so every hop is re-checked, and the download is capped at 20 MiB.

> **Verifying a refused route.** The emitters for the OpenAI-shaped and Mistral chat
> dialects are written and unit-tested — only the refusal is in the way. Set
> `enableUnverifiedImageRoutes = true` in `Config.toml` and run `bal test --groups live`
> to push a real image through the module and see what AWS says. If the route accepts
> the body this module builds, the default flips permanently.
>
> **Why some routes refuse.** Image support is enabled only where a primary source
> confirms the wire shape. AWS's Mantle pages are JS-rendered and state nothing about
> image parts, and for Mistral's InvokeModel dialect AWS documents `content` as a
> string while Mistral's own API documents image chunks — two first-party sources
> disagreeing. Rather than guess and ship the silent-wrong-answer bug this feature
> exists to fix, those routes refuse. Each is a one-line change once a live call
> settles it.

Images are redacted in the observability span (`[image image/png, 12043 bytes]`), so
the payload never reaches your telemetry backend.

## Migrating from 0.9.x

Credentials moved to [`ballerinax/aws.auth`](https://central.ballerina.io/ballerinax/aws/latest), which
required reordering `init` — Ballerina requires required parameters before defaultable ones, and both
`region` and `credentials` are now defaultable.

**Argument order changed on all nine providers.** `model` is the only required parameter —
Ballerina requires required parameters before defaultable ones, so `credentials` had to move
after it in order to keep its default:

```ballerina
// 0.9.x
check new bedrock:AnthropicModelProvider(creds, bedrock:CLAUDE_SONNET_4_6, "us-east-1");

// now — model first, credentials optional
check new bedrock:AnthropicModelProvider(bedrock:CLAUDE_SONNET_4_6, creds, "us-east-1");
```

**`StaticCredentials` and `StsCredentials` were removed.** Both collapse into
`auth:StaticAuthConfig`, whose `sessionToken` is optional. Inline record literals are unchanged —
`{accessKeyId, secretAccessKey}` and `{accessKeyId, secretAccessKey, sessionToken}` both still work;
only code that named those types needs editing. `BearerToken` is unchanged.

**FIPS moved from a `serviceUrl` template to `config = {fips: true}`.** The old
`"https://bedrock-{endpoint}-fips.{region}.{domain}"` still works, but the flag takes the host from SDK
metadata and rejects FIPS-on-Mantle at construction.

Nothing else moved. `serviceUrl`, `DEFAULT_SERVICE_URL` and all three placeholders behave as before,
and SigV4 signing is unchanged — see [Not implemented](#not-implemented) for why signing stayed
in-module.

## Not implemented

Document/video/audio content blocks
(Converse models all three — see [Images](#images) for the image scope line), image/video/audio embeddings and
`StartAsyncInvoke` (the `ai:Chunk` contract carries text), provisioned-throughput embedding ARNs,
Meta/Llama, and Custom Model Import (`imported-model/` ARNs).

On knowledge bases specifically: **Kendra and SQL/Redshift** knowledge base types (only `MANAGED` and
`VECTOR` are implemented); provisioning the vector store itself, which the Bedrock API cannot do at all;
`implicitFilterConfiguration` on the self-managed search branch; and `startsWith`/`stringContains`
filters, which self-managed knowledge bases support but `ai:MetadataFilterOperator` has no operator for.

**SigV4 signing is not delegated to `aws.auth`,** though credential resolution and endpoint metadata
are. `auth:getSignedHeaders` builds its canonical URI by double-encoding while always treating `/` as a
path separator, so for a model-id ARN it produces `...inference-profile/us.anthropic...` where AWS
expects `...inference-profile%252Fus.anthropic...`. No input fixes this — a literal `/` survives both
passes, and a pre-encoded `%2F` becomes `%25252F` — so every provisioned-model, inference-profile and
custom-model-deployment ARN would fail with `SignatureDoesNotMatch`. Verified against `ballerinax/aws`
1.0.1 on 2026-08-09.

Deliberately **not** surfaced as config, because the `ai` contract has nowhere to return them:
`additionalModelResponseFieldPaths` (its result would be dropped — `ai:ChatAssistantMessage` is
`{role, content, toolCalls}`) and `requestMetadata` (write-only; it tags invocation logs).
`top_p`/`top_k` and other fine-grained sampling knobs go through `additionalModelRequestFields`.
