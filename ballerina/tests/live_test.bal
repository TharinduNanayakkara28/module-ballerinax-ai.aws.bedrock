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
import ballerina/lang.array;
import ballerina/test;

// Live integration tests. These call real AWS and cost real money,
// so they are inert by default: with no credentials configured every test returns
// immediately. `bal test` on a clean checkout runs the other suites unchanged.
//
// To run them, put credentials in `tests/Config.toml`:
//
//   [ballerinax.ai.aws.bedrock]
//   liveTestsEnabled = true          # REQUIRED — every live test carries
//                                    # `enable: liveTestsEnabled`, which defaults to
//                                    # false. Omit it and `bal test --groups live`
//                                    # runs zero tests and reports no failure.
//   liveAccessKeyId = "AKIA..."
//   liveSecretAccessKey = "..."
//   liveRegion = "us-east-1"
//   liveConverseModelArn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-4-6"
//
// then: bal test --groups live
//
// These are the checks that CANNOT be settled offline — every one of them exists
// because a golden-file test would happily pass against a body AWS rejects.

configurable boolean liveTestsEnabled = false;
configurable string liveAccessKeyId = "";
configurable string liveSecretAccessKey = "";
configurable string liveSessionToken = "";
configurable string liveRegion = "us-east-1";

// A US cross-region inference-profile ARN. Left empty by default
// because the account id makes it caller-specific.
configurable string liveConverseModelArn = "";

// Whether to exercise the FIPS endpoint. Separate from `liveTestsEnabled` because
// FIPS is a per-region deployment: `bedrock-runtime-fips.{region}.{domain}` resolves
// in the commercial and GovCloud partitions, but the MODEL must also be served there.
// A failure here is an availability fact about your region, not a module defect.
configurable boolean liveFipsEnabled = false;

// Whether the account has bedrock-mantle access. Mantle needs the separate
// `bedrock-mantle:CreateInference` IAM action, so an account with working
// `bedrock:InvokeModel` permissions may still 403 here — that is a real access
// gap, not a module defect, hence its own switch.
configurable boolean liveMantleEnabled = false;

// Skips the whole suite when nothing is configured.
function liveCredentials() returns BedrockCredentials? {
    if liveAccessKeyId == "" || liveSecretAccessKey == "" {
        return ();
    }
    if liveSessionToken != "" {
        return {accessKeyId: liveAccessKeyId, secretAccessKey: liveSecretAccessKey, sessionToken: liveSessionToken};
    }
    return {accessKeyId: liveAccessKeyId, secretAccessKey: liveSecretAccessKey};
}

// ---- Converse via a US CRIS inference-profile ARN ----

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveConverseViaCrisInferenceProfileArn() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () || liveConverseModelArn == "" {
        return;
    }
    // The ARN exercises the SigV4 path-encoding split: its `:` and `/`
    // characters are single-encoded on the wire and double-encoded in the
    // signature. Get that wrong and this is a 403 — no golden test can catch it.
    ai:ModelProvider provider = check new AnthropicModelProvider(liveConverseModelArn, creds, liveRegion);
    ai:ChatAssistantMessage response = check provider->chat([
        {role: ai:SYSTEM, content: "Answer with exactly one word."},
        {role: ai:USER, content: "What colour is the sky on a clear day?"}
    ]);
    string content = response.content ?: "";
    test:assertTrue(content.trim().length() > 0, "live Converse returned empty content");
}

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveConverseWithABareModelId() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () {
        return;
    }
    ai:ModelProvider provider = check new AnthropicModelProvider(CLAUDE_SONNET_4_6, creds, liveRegion);
    ai:ChatAssistantMessage response = check provider->chat({role: ai:USER, content: "Say OK."});
    test:assertTrue((response.content ?: "").trim().length() > 0);
}

// ---- generate() — native structured output on Converse ----

type LiveFruit record {|
    string name;
    string colour;
|};

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveGenerateOnConverseReturnsTheRecord() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () {
        return;
    }
    // Proves the forced-tool round trip end to end: the derived JSON schema is
    // accepted as a tool by AWS, and the model's tool-call arguments bind back
    // into the record.
    AnthropicModelProvider provider = check new (CLAUDE_SONNET_4_6, creds, liveRegion);
    LiveFruit fruit = check provider->generate(`Name one common fruit and its colour.`);
    test:assertTrue(fruit.name.trim().length() > 0, "generate() returned an empty name");
    test:assertTrue(fruit.colour.trim().length() > 0, "generate() returned an empty colour");
}

// ---- Mantle via openai.gpt-5.4 ----

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveMantleChat() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () || !liveMantleEnabled {
        return;
    }
    // Mantle is a different host, a different wire dialect, a different SigV4
    // signing scope, and a different IAM namespace. Nothing about this path is
    // shared with Converse except the credentials.
    ai:ModelProvider provider = check new OpenAIModelProvider(GPT_5_4, creds, liveRegion);
    ai:ChatAssistantMessage response = check provider->chat({role: ai:USER, content: "Say OK."});
    test:assertTrue((response.content ?: "").trim().length() > 0);
}

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveMantleRefusesStructuredOutputButReturnsText() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () || !liveMantleEnabled {
        return;
    }
    OpenAIModelProvider provider = check new (GPT_5_4, creds, liveRegion);

    // A typed target must be refused locally, without spending a call.
    LiveFruit|ai:Error typed = provider->generate(`Name one common fruit and its colour.`);
    test:assertTrue(typed is ai:Error, "Mantle must refuse a typed target");

    // ...but a string target still works.
    string text = check provider->generate(`Say OK.`);
    test:assertTrue(text.trim().length() > 0);
}

// ---- Embeddings: Titan V2 + Cohere Embed English v3 ----

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveTitanEmbedding() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () {
        return;
    }
    ai:EmbeddingProvider provider = check new TitanEmbeddingProvider(
        TITAN_EMBED_TEXT_V2, creds, liveRegion, dimensions = 1024);
    ai:Embedding embedding = check provider->embed({content: "hello world", 'type: "text-chunk"});
    test:assertTrue(embedding is float[], "Titan must return a dense vector");
    if embedding is float[] {
        test:assertEquals(embedding.length(), 1024, "the configured dimensions must reach the wire");
    }
}

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveCohereEmbeddingPreservesOrderAcrossWindows() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () {
        return;
    }
    // 100 chunks > Cohere's 96-per-call wire limit, so this crosses a window
    // boundary (96 + 4) and is the real test of index-based reassembly.
    ai:TextChunk[] chunks = [];
    foreach int i in 0 ..< 100 {
        chunks.push({content: string `item number ${i}`, 'type: "text-chunk"});
    }
    ai:EmbeddingProvider provider = check new CohereEmbeddingProvider(
        COHERE_EMBED_ENGLISH_V3, creds, liveRegion, inputType = SEARCH_DOCUMENT);
    ai:Embedding[] embeddings = check provider->batchEmbed(chunks);
    test:assertEquals(embeddings.length(), 100, "one embedding per input, in input order");

    // Re-embed one item from the far side of the window boundary on its own; it
    // must match the batched result at the same index. If reassembly dropped or
    // reordered a window, this is where it shows.
    ai:Embedding single = check provider->embed(chunks[97]);
    ai:Embedding batched = embeddings[97];
    if single is float[] && batched is float[] {
        test:assertEquals(single.length(), batched.length());
        test:assertTrue(single.length() > 0);
        // Compare the leading components rather than the whole vector: identical
        // input and settings, so these must agree.
        foreach int i in 0 ..< 8 {
            test:assertTrue((single[i] - batched[i]).abs() < 0.0001,
                    string `index 97 differs at component ${i} — batch reassembly is misaligned`);
        }
    } else {
        test:assertFail("Cohere must return dense vectors");
    }
}


// ---- serviceUrl: FIPS endpoint ----

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveFipsEndpointAcceptsASignedRequest() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () || !liveFipsEnabled {
        return;
    }
    // The ONE thing no offline test can settle: whether AWS ACCEPTS a request to the
    // FIPS host. DNS proves the name exists and the mock-server test proves we send
    // the right bytes to whatever origin we are given — but only a real call proves
    // the signature validates against a host we did not derive ourselves.
    //
    // The invariant under test: the host changes, the signing scope does NOT. If
    // `serviceUrl` leaked into the SigV4 credential scope this returns 403
    // SignatureDoesNotMatch, which is exactly the regression worth paying for.
    ai:ModelProvider provider = check new AnthropicModelProvider(
            "anthropic.claude-sonnet-4-6", creds, liveRegion,
            serviceUrl = "https://bedrock-{endpoint}-fips.{region}.{domain}");
    ai:ChatAssistantMessage response = check provider->chat([
        {role: ai:USER, content: "Reply with the single word: ok"}
    ]);
    string content = response.content ?: "";
    test:assertTrue(content.trim().length() > 0, "live FIPS Converse returned empty content");
}

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveDefaultAndFipsEndpointsAgree() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () || !liveFipsEnabled {
        return;
    }
    // Same prompt, same model, two origins. Both must succeed — this catches a FIPS
    // host that resolves and authenticates but does not actually serve the model in
    // this region, which would otherwise surface only to the first customer to try it.
    ai:ModelProvider dflt = check new AnthropicModelProvider(
            "anthropic.claude-sonnet-4-6", creds, liveRegion);
    // `fips` rather than a hand-written template: the host now comes from AWS SDK
    // endpoint metadata, so this also confirms the metadata's spelling is real.
    ai:ModelProvider fips = check new AnthropicModelProvider(
            "anthropic.claude-sonnet-4-6", creds, liveRegion, config = {fips: true});
    ai:ChatMessage[] prompt = [{role: ai:USER, content: "Reply with the single word: ok"}];
    ai:ChatAssistantMessage a = check dflt->chat(prompt);
    ai:ChatAssistantMessage b = check fips->chat(prompt);
    test:assertTrue((a.content ?: "").trim().length() > 0);
    test:assertTrue((b.content ?: "").trim().length() > 0);
}


// ---- effort: which Converse mechanism does Bedrock actually honour? ----

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveConverseEffortIsAccepted() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () {
        return;
    }
    // RESOLVED 2026-08-11 (live). Two first-party sources disagreed on how `effort`
    // reaches Converse:
    //
    //   botocore  -> a native `outputConfig: {effort}` member on ConverseRequest
    //   AWS docs  -> additionalModelRequestFields: {"output_config": {"effort": ...}}
    //
    // `converter_converse.bal` used to emit the NATIVE member and 400'd with "This
    // model doesn't support the effort field" on every model tried, including
    // opus-4-7 — which IS on Anthropic's adaptive-only list (effort is meant to be
    // ITS only depth control), ruling out "wrong model". The identical value folded
    // into additionalModelRequestFields.output_config.effort was accepted on the
    // same model/route. The encoder now emits the passthrough form; this test
    // guards the regression. Uses opus-4-7 rather than sonnet-4-6 (not on the
    // adaptive-only list, so it is not a safe model to assert `effort` support on).
    ai:ModelProvider provider = check new AnthropicModelProvider(
            "us.anthropic.claude-opus-4-7", creds, liveRegion,
            apiFamily = CONVERSE,
            thinking = {mode: ADAPTIVE},
            effort = EFFORT_LOW);
    ai:ChatAssistantMessage response = check provider->chat([
        {role: ai:USER, content: "Reply with the single word: ok"}
    ]);
    test:assertTrue((response.content ?: "").trim().length() > 0,
            "Converse rejected the passthrough output_config.effort form too — the finding needs revisiting");
}

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveAdaptiveThinkingOnTheMessagesDialect() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () || !liveMantleEnabled {
        return;
    }
    // The regression this whole change exists for: `thinking` used to be folded into
    // `additionalModelRequestFields`, which the Anthropic Messages encoder ignores —
    // so the knob was silently dropped on this exact route. It is now a top-level
    // body field, and `output_config.effort` rides beside it.
    ai:ModelProvider provider = check new AnthropicModelProvider(
            "anthropic.claude-haiku-4-5", creds, liveRegion,
            apiFamily = MANTLE,
            thinking = {mode: ADAPTIVE},
            effort = EFFORT_LOW);
    ai:ChatAssistantMessage response = check provider->chat([
        {role: ai:USER, content: "Reply with the single word: ok"}
    ]);
    test:assertTrue((response.content ?: "").trim().length() > 0);
}

// ---- Images: does each route actually accept what this module emits? ----
//
// These exist because no first-party source states whether the OpenAI-shaped Mantle
// and Invoke dialects accept image parts. Crucially they go THROUGH THE MODULE, so
// what is validated is the exact body it builds — hand-written JSON would only prove
// that the hand-written JSON works.
//
// Converse and Anthropic run by default. The other three need
// `enableUnverifiedImageRoutes = true` in Config.toml, since the module refuses them
// otherwise; a PASS there is the signal to flip that default permanently.

// A real 1x1 PNG. Must be a decodable image, not a signature stub: the model has to
// look at it, so AWS will reject anything malformed before the model ever sees it.
const string ONE_PX_PNG_B64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

function onePixelPng() returns ai:ImageDocument|error =>
    {content: check array:fromBase64(ONE_PX_PNG_B64), metadata: {mimeType: "image/png"}};

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveConverseAcceptsAnImage() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () {
        return;
    }
    // The CONTROL. Converse image support is documented, so a failure here means the
    // encoding is wrong — not that the route lacks support.
    ai:ImageDocument img = check onePixelPng();
    ai:ModelProvider provider = check new AnthropicModelProvider(
            CLAUDE_SONNET_4_6, creds, liveRegion, config = {apiFamily: CONVERSE});
    ai:ChatAssistantMessage response = check provider->chat({
        role: ai:USER,
        content: `Does this contain an image? Answer yes or no. ${img}`
    });
    test:assertTrue((response.content ?: "").trim().length() > 0, "Converse rejected the image block");
}

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveInvokeAnthropicAcceptsAnImage() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () {
        return;
    }
    // Confirms the base64 `source` shape on the Anthropic Messages dialect, and that
    // Bedrock really does refuse nothing about it.
    ai:ImageDocument img = check onePixelPng();
    ai:ModelProvider provider = check new AnthropicModelProvider(
            CLAUDE_SONNET_4_6, creds, liveRegion, config = {apiFamily: INVOKE});
    ai:ChatAssistantMessage response = check provider->chat({
        role: ai:USER,
        content: `Does this contain an image? Answer yes or no. ${img}`
    });
    test:assertTrue((response.content ?: "").trim().length() > 0, "Invoke-Anthropic rejected the image");
}

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveMantleResponsesImageSupportIsUnknown() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () || !liveMantleEnabled || !enableUnverifiedImageRoutes {
        return;
    }
    // UNVERIFIED. A 400 here is a RESULT, not a defect — it tells us the default
    // refusal is correct. A pass tells us to remove it.
    ai:ImageDocument img = check onePixelPng();
    ai:ModelProvider provider = check new OpenAIModelProvider(GPT_5_4, creds, liveRegion);
    ai:ChatAssistantMessage|ai:Error response = provider->chat({
        role: ai:USER,
        content: `Does this contain an image? Answer yes or no. ${img}`
    });
    if response is ai:Error {
        test:assertFail(string `Mantle Responses REJECTED the image — keep the default refusal. ` +
            string `Error: ${response.message()}`);
    }
    test:assertTrue((response.content ?: "").trim().length() > 0);
}

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveMantleChatCompletionsImageSupportIsUnknown() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () || !liveMantleEnabled || !enableUnverifiedImageRoutes {
        return;
    }
    ai:ImageDocument img = check onePixelPng();
    ai:ModelProvider provider = check new GoogleModelProvider(GEMMA_3_27B_IT, creds, liveRegion);
    ai:ChatAssistantMessage|ai:Error response = provider->chat({
        role: ai:USER,
        content: `Does this contain an image? Answer yes or no. ${img}`
    });
    if response is ai:Error {
        test:assertFail(string `Mantle chat-completions REJECTED the image — keep the default ` +
            string `refusal. Error: ${response.message()}`);
    }
    test:assertTrue((response.content ?: "").trim().length() > 0);
}

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveInvokeMistralChatImageSupportIsContested() returns error? {
    BedrockCredentials? creds = liveCredentials();
    if creds is () || !enableUnverifiedImageRoutes {
        return;
    }
    // AWS documents this dialect's `content` as a string; Mistral's own API documents
    // image chunks. This call settles which one describes Bedrock.
    ai:ImageDocument img = check onePixelPng();
    ai:ModelProvider provider = check new MistralModelProvider(
            MISTRAL_LARGE_3, creds, liveRegion, config = {apiFamily: INVOKE});
    ai:ChatAssistantMessage|ai:Error response = provider->chat({
        role: ai:USER,
        content: `Does this contain an image? Answer yes or no. ${img}`
    });
    if response is ai:Error {
        test:assertFail(string `Invoke-Mistral REJECTED the image — AWS's docs are right and the ` +
            string `default refusal stays. Error: ${response.message()}`);
    }
    test:assertTrue((response.content ?: "").trim().length() > 0);
}

// ---- Credential chain (ballerinax/aws.auth) ----
//
// The whole credential path was replaced: `auth:CredentialProvider` now resolves and
// refreshes, and `getCredentials()` is called per request rather than once at
// construction. Every other live test pins static keys, so NONE of them exercise the
// chain. These do.

// An IAM role to assume, e.g. "arn:aws:iam::222222222222:role/BedrockCaller".
// Empty by default: the account id makes it caller-specific.
configurable string liveAssumeRoleArn = "";

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveDefaultCredentialChainCanCallBedrock() returns error? {
    // No credentials argument at all — the whole point of the migration. Resolves
    // from whatever the environment offers: env vars, an EC2 instance profile, an ECS
    // task role, EKS IRSA, SSO, or ~/.aws/credentials.
    //
    // A failure here means the chain did not find usable credentials in THIS
    // environment; run it on the target compute (EC2/ECS/EKS) to prove the case that
    // matters. It is the only test that covers construction with no credentials.
    ai:ModelProvider provider = check new AnthropicModelProvider(CLAUDE_SONNET_4_6, region = liveRegion);
    ai:ChatAssistantMessage response = check provider->chat({role: ai:USER, content: "Say OK."});
    test:assertTrue((response.content ?: "").trim().length() > 0,
            "DEFAULT_CREDENTIALS resolved but the call failed");
}

@test:Config {groups: ["live"], enable: liveTestsEnabled}
function testLiveAssumeRoleCredentialsCanCallBedrock() returns error? {
    if liveAssumeRoleArn == "" {
        return;
    }
    // Cross-account access, which the module could not express at all before this
    // migration. Also the first path where credentials EXPIRE, so it exercises
    // refresh in a way static keys never can.
    BedrockCredentials assumed = {roleArn: liveAssumeRoleArn, stsRegion: liveRegion};
    ai:ModelProvider provider = check new AnthropicModelProvider(CLAUDE_SONNET_4_6, assumed, liveRegion);
    ai:ChatAssistantMessage response = check provider->chat({role: ai:USER, content: "Say OK."});
    test:assertTrue((response.content ?: "").trim().length() > 0);
}
