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
import ballerina/http;
import ballerina/lang.runtime;
import ballerina/time;

// The shared spine `BedrockManagedKnowledgeBase` is built over: two agent-plane
// transports, find-or-create, data-source resolution, chunking-strategy detection,
// and the document/knowledge-base wire calls both `ingest()` and `deleteByFilter()`
// need. `BedrockVectorKnowledgeBase` (knowledgebase_vector.bal) is built over the
// same spine, with its own request bodies where the self-managed API shape differs.

// Bedrock's `_source_uri` metadata attribute — injected on every retrieval result,
// holding the document's `customDocumentIdentifier.id` (CUSTOM sources) or S3 object
// URI (S3 sources). NOT in either service model: undocumented and non-contractual,
// confirmed only by calling the live API against a MANAGED knowledge base with a
// CUSTOM data source (2026-08-14). It is what makes `deleteByFilter`'s probe
// possible at all — Bedrock has no metadata-based delete and no way to read a
// document's metadata back any other way.
const string SOURCE_URI_METADATA_KEY = "_source_uri";

// `deleteByFilter`'s probe query text. Its CONTENT is irrelevant and this is not a
// tuning knob: every probe ANDs the caller's filter with a `_source_uri` leaf
// pinning ONE document, and pinning takes the result off the scoring path
// entirely. Measured 2026-08-14 with a deliberately nonsensical query
// ("zqxjkv quantum chromodynamics platypus 7823 borogoves"):
//
//   - single-chunk document, pinned  -> returned at score 1.0
//   - 69-chunk document, pinned      -> top chunk at 0.88 (floor sits near 0.15)
//   - a topically RELEVANT query on the same document scored LOWER (0.43),
//     confirming the pinned score is not query similarity
//
// It exists solely because `Retrieve` REJECTS an empty query: `{"text": ""}`,
// `{"text": " "}` and an omitted `text` all return 400 "Text input is required."
// (The service model's `KnowledgeBaseQueryTextString` declares `min: 0`, which the
// live API contradicts.)
const string FILTER_PROBE_QUERY = "PLACE HOLDER";

// `ListKnowledgeBases`/`ListDataSources`/`ListKnowledgeBaseDocuments` share one
// `MaxResults` shape declaring `max: 1000` — but the LIVE `ListKnowledgeBaseDocuments`
// API rejects anything over 100 ("maxResults must be less than or equal to 100"),
// contradicting the service model. Applied to all three list calls here
// defensively, since they share the shape.
const int KB_LIST_PAGE_SIZE = 100;

// `IngestKnowledgeBaseDocuments`/`DeleteKnowledgeBaseDocuments`/
// `GetKnowledgeBaseDocuments` all cap their array parameter at 10 elements
// (`KnowledgeBaseDocuments`/`DocumentIdentifiers`, both `max: 10`).
const int KB_DOCUMENT_BATCH_SIZE = 10;

// Poll interval for knowledge base / data source / document status.
const decimal KB_POLL_INTERVAL_SECONDS = 3;

// `CreateDataSource` is SYNCHRONOUS on the managed-KB path (measured against the
// live API: 200 AVAILABLE in the response, unlike `CreateKnowledgeBase`'s 202
// CREATING) — this bound is a defensive fallback only, used if a future data source
// is not already AVAILABLE in the create response.
const decimal DEFAULT_DATA_SOURCE_READY_TIMEOUT = 60;

// Document statuses that are retrievable (usable in `retrieve()` results and safe
// to enumerate for `deleteByFilter()`).
final readonly & string[] KB_DOC_USABLE_STATUSES = ["INDEXED", "PARTIALLY_INDEXED", "METADATA_PARTIALLY_INDEXED"];
// Terminal statuses that are NOT usable. `NOT_FOUND` is a tombstone for a deleted
// document — it arrives as HTTP 200 with this status, never a 404 (confirmed
// against the live API).
final readonly & string[] KB_DOC_FAILED_STATUSES = ["FAILED", "METADATA_UPDATE_FAILED", "IGNORED", "NOT_FOUND"];
// Transient statuses `ingest()` keeps polling through.
final readonly & string[] KB_DOC_IN_FLIGHT_STATUSES = ["PENDING", "STARTING", "IN_PROGRESS"];

// Which bedrock-agent-runtime search branch `retrieve()` uses. Fixed to managed —
// `vectorSearchConfiguration`'s knobs (`overrideSearchType`, `implicitFilterConfiguration`)
// are meaningless against a Bedrock-owned vector store.
const int KB_MAX_RESULTS_PER_CALL = 100;

// ============================================================================
// Spine resolution.
// ============================================================================

# Everything `BedrockManagedKnowledgeBase`'s methods read: two agent-plane
# transports (control on `bedrock-agent`, data on `bedrock-agent-runtime`), the
# resolved knowledge base / data source ids, and the detected chunking strategy.
# Module-private — the resolver's output, mirroring `Route`/`Endpoint`.
#
# + controlTransport - `bedrock-agent` (create/list/get KB & data source, ingest/list/get/delete documents)
# + dataTransport - `bedrock-agent-runtime` (retrieve)
# + knowledgeBaseId - The resolved knowledge base id
# + dataSourceId - The resolved `CUSTOM` data source id
# + chunkingStrategy - The resolved data source's actual chunking strategy
type KbSpine record {|
    BedrockTransport controlTransport;
    BedrockTransport dataTransport;
    string knowledgeBaseId;
    string dataSourceId;
    ChunkingStrategy chunkingStrategy;
|};

// The shared construction spine: transports -> find-or-create -> data-source
// resolution -> chunking detection. Every failure surfaces here, before any method
// is callable.
isolated function resolveKbSpine(string providerName, KnowledgeBaseCredentials credentials, string region,
        string serviceUrl, string|KnowledgeBaseDefinition knowledgeBase, string? dataSourceIdOverride,
        http:ClientConfiguration? httpConfig, RetryConfig? retryConfig, boolean fips,
        RerankingModelType? rerankingModelType = ())
        returns KbSpine|ai:Error {
    do {
        check guardRegion(region);
        check guardEmbeddingModelAgainstReranker(knowledgeBase, rerankingModelType);
        Endpoint controlEp = check buildAgentEndpoint(AGENT_CONTROL, region, serviceUrl, fips);
        Endpoint dataEp = check buildAgentEndpoint(AGENT_DATA, region, serviceUrl, fips);
        BedrockTransport controlTransport =
            check new (credentials, region, controlEp, httpConfig, retryConfig, true);
        BedrockTransport dataTransport =
            check new (credentials, region, dataEp, httpConfig, retryConfig, true);

        KbAttachResult attach = check resolveKnowledgeBase(controlTransport, knowledgeBase);
        string dataSourceId;
        if dataSourceIdOverride is string {
            dataSourceId = dataSourceIdOverride;
        } else if attach.createdDataSourceId is string {
            dataSourceId = <string>attach.createdDataSourceId;
        } else {
            dataSourceId = check resolveCustomDataSource(controlTransport, attach.knowledgeBaseId);
        }
        ChunkingStrategy strategy =
            check detectChunkingStrategy(controlTransport, attach.knowledgeBaseId, dataSourceId);
        return {
            controlTransport,
            dataTransport,
            knowledgeBaseId: attach.knowledgeBaseId,
            dataSourceId,
            chunkingStrategy: strategy
        };
    } on fail error e {
        if e is ai:Error {
            return e;
        }
        return error ai:Error(string `Failed to initialize ${providerName}: ${e.message()}`, e);
    }
}

// ============================================================================
// Find-or-create.
// ============================================================================

# Outcome of resolving `string|KnowledgeBaseDefinition` to a concrete knowledge
# base. `createdDataSourceId` is set ONLY when a new knowledge base (and its
# `CUSTOM` data source) was just created — in every other case (a bare id, or an
# existing knowledge base found by name) data-source resolution still has to run.
#
# + knowledgeBaseId - The attached or newly created knowledge base id
# + createdDataSourceId - The `CUSTOM` data source id, when this call just created it
type KbAttachResult record {|
    string knowledgeBaseId;
    string? createdDataSourceId;
|};

// `string` -> verify and attach (no writes). `KnowledgeBaseDefinition` -> find by
// name; exactly one match attaches, no match creates (knowledge base + its `CUSTOM`
// data source), more than one match is a construction error — `CreateKnowledgeBase`
// has no upsert and names are not unique per account, so guessing would risk
// creating a duplicate or attaching to the wrong one.
isolated function resolveKnowledgeBase(BedrockTransport controlTransport, string|KnowledgeBaseDefinition knowledgeBase)
        returns KbAttachResult|ai:Error {
    if knowledgeBase is string {
        check verifyKnowledgeBaseUsable(controlTransport, knowledgeBase);
        return {knowledgeBaseId: knowledgeBase, createdDataSourceId: ()};
    }
    string[] candidates = check listKnowledgeBaseIdsByName(controlTransport, knowledgeBase.name);
    if candidates.length() == 1 {
        check verifyKnowledgeBaseUsable(controlTransport, candidates[0]);
        return {knowledgeBaseId: candidates[0], createdDataSourceId: ()};
    }
    if candidates.length() > 1 {
        return error ai:Error(
            string `${candidates.length()} knowledge bases are named '${knowledgeBase.name}' ` +
            string `(${string:'join(", ", ...candidates)}) — names are not unique per account, so which one ` +
            "was meant is ambiguous. Pass the knowledge base id directly instead of a definition.");
    }
    // No match: create the knowledge base, wait for it to leave CREATING, then
    // create its CUSTOM data source and wait for that too — two asynchronously
    // provisioned resources, not one, is the real cost of this path.
    string kbId = check createKnowledgeBase(controlTransport, knowledgeBase);
    check pollKnowledgeBaseActive(controlTransport, kbId, knowledgeBase.readyTimeout);
    string dsId = check createCustomDataSource(controlTransport, kbId, knowledgeBase.dataSource);
    return {knowledgeBaseId: kbId, createdDataSourceId: dsId};
}

// Both that the knowledge base is ACTIVE and that it is actually a MANAGED one.
//
// The type check is not pedantry. EVERY behaviour this class depends on was measured
// against Bedrock's own vector store, on the `managedSearchConfiguration` branch:
// `retrieve()` sends that branch unconditionally, and `deleteByFilter`'s whole
// soundness argument rests on a pinned `_source_uri` filter taking the query off the
// scoring path (see `FILTER_PROBE_QUERY` for those measurements). A `VECTOR`
// knowledge base queries a CUSTOMER-owned store (OpenSearch Serverless, Pinecone,
// pgvector, ...) through a different branch and a different ranking engine, where
// none of that was measured and the pinned probe may score normally — which would
// put `deleteByFilter` back to silently under-deleting. Refuse at construction
// rather than half-work at runtime.
isolated function verifyKnowledgeBaseUsable(BedrockTransport controlTransport, string kbId) returns ai:Error? {
    map<json> kb = check getKnowledgeBase(controlTransport, kbId);
    string status = stringField(kb, "status") ?: "";
    if status != "ACTIVE" {
        return error ai:Error(
            string `Knowledge base '${kbId}' is not usable: status is '${status}' (expected 'ACTIVE'). ` +
            "Wait for it to finish provisioning, or check the AWS console for failure details.");
    }
    string kbType = stringField(asMap(kb["knowledgeBaseConfiguration"] ?: {}), "type") ?: "";
    // Absent type is tolerated: it is required in the service model, so a missing one
    // means an unexpected response shape rather than a non-managed knowledge base,
    // and failing construction over it would be a false positive.
    if kbType != "" && kbType != "MANAGED" {
        return error ai:Error(
            string `Knowledge base '${kbId}' is of type '${kbType}', but BedrockManagedKnowledgeBase ` +
            "supports only 'MANAGED' knowledge bases (the ones where Bedrock owns the vector store). " +
            "A 'VECTOR' knowledge base is backed by your own vector store and is served by a different " +
            "search branch, so retrieve() and deleteByFilter() are not valid against it. Use " +
            "BedrockVectorKnowledgeBase for a 'VECTOR' knowledge base.");
    }
}

isolated function listKnowledgeBaseIdsByName(BedrockTransport controlTransport, string name) returns string[]|ai:Error {
    string[] ids = [];
    string? nextToken = ();
    while true {
        map<json> body = {maxResults: KB_LIST_PAGE_SIZE};
        if nextToken is string {
            body["nextToken"] = nextToken;
        }
        TransportResponse response = check controlTransport.executeRequest("POST", "/knowledgebases/", body);
        map<json> respBody = asMap(response.body);
        json summariesJson = respBody["knowledgeBaseSummaries"] ?: [];
        if summariesJson is json[] {
            foreach json s in summariesJson {
                map<json> summary = asMap(s);
                if stringField(summary, "name") == name {
                    string? id = stringField(summary, "knowledgeBaseId");
                    if id is string {
                        ids.push(id);
                    }
                }
            }
        }
        nextToken = stringField(respBody, "nextToken");
        if nextToken is () {
            break;
        }
    }
    return ids;
}

isolated function getKnowledgeBase(BedrockTransport controlTransport, string kbId) returns map<json>|ai:Error {
    string path = string `/knowledgebases/${kbId}`;
    TransportResponse response = check controlTransport.executeRequest("GET", path, ());
    return asMap(asMap(response.body)["knowledgeBase"] ?: {});
}

// A caller-supplied embedding model and Bedrock's managed reranker are mutually
// exclusive: "If you create a knowledge base with a custom embedding model, the
// managed reranker is not available for that knowledge base."
// https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-create.html
//
// Both choices are permanent — `embeddingModelType` cannot be changed after creation
// — so this has to fail at construction. Discovering it from a failed `retrieve()`
// would mean the knowledge base was already built with the wrong combination and has
// to be rebuilt. Only checkable on the CREATE path: attaching by id says nothing
// about how the knowledge base was configured.
isolated function guardEmbeddingModelAgainstReranker(string|KnowledgeBaseDefinition knowledgeBase,
        RerankingModelType? rerankingModelType) returns ai:Error? {
    if knowledgeBase is string || rerankingModelType != RERANKING_MANAGED {
        return;
    }
    if knowledgeBase?.embeddingModel is ManagedEmbeddingModel {
        return error ai:Error(
            "'rerankingModelType' is RERANKING_MANAGED and 'knowledgeBase.embeddingModel' is set, but AWS " +
            "makes the managed reranker unavailable on a knowledge base created with a caller-supplied " +
            "embedding model. Both are permanent at creation time, so pick one: drop 'embeddingModel' to " +
            "keep managed reranking, or use RERANKING_NONE (or leave 'rerankingModelType' unset) to keep " +
            "your own embedding model.");
    }
}

// The `CreateKnowledgeBase` request body. Pure, so the embedding-model and KMS
// branches are testable without AWS.
//
// MANAGED needs no `storageConfiguration` at all — Bedrock owns the vector store.
// The embedding model is the caller's choice, and AWS is strict about which fields
// accompany which type: "When using MANAGED, you must not specify embeddingModelArn
// or embeddingModelConfiguration. When using CUSTOM, both fields are required."
// https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-create.html
// https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent_ManagedKnowledgeBaseConfiguration.html
isolated function createKnowledgeBaseRequestBody(KnowledgeBaseDefinition def) returns map<json> {
    ManagedEmbeddingModel? embeddingModel = def?.embeddingModel;
    map<json> managedConfig;
    if embeddingModel is ManagedEmbeddingModel {
        managedConfig = {
            embeddingModelType: "CUSTOM",
            embeddingModelArn: embeddingModel.embeddingModelArn,
            embeddingModelConfiguration: {
                bedrockEmbeddingModelConfiguration: {
                    dimensions: embeddingModel.dimensions,
                    embeddingDataType: embeddingModel.embeddingDataType
                }
            }
        };
    } else {
        managedConfig = {embeddingModelType: "MANAGED"};
    }
    string? kmsKeyArn = def?.kmsKeyArn;
    if kmsKeyArn is string {
        managedConfig["serverSideEncryptionConfiguration"] = {kmsKeyArn};
    }
    map<json> body = {
        name: def.name,
        roleArn: def.roleArn,
        knowledgeBaseConfiguration: {
            'type: "MANAGED",
            managedKnowledgeBaseConfiguration: managedConfig
        }
    };
    string? description = def.description;
    if description is string {
        body["description"] = description;
    }
    return body;
}

isolated function createKnowledgeBase(BedrockTransport controlTransport, KnowledgeBaseDefinition def)
        returns string|ai:Error {
    map<json> body = createKnowledgeBaseRequestBody(def);
    TransportResponse response = check controlTransport.executeRequest("PUT", "/knowledgebases/", body);
    map<json> kb = asMap(asMap(response.body)["knowledgeBase"] ?: {});
    string? id = stringField(kb, "knowledgeBaseId");
    if id is () {
        return error ai:Error("CreateKnowledgeBase response carried no 'knowledgeBaseId'");
    }
    return id;
}

// ~83s measured for a knowledge base to leave CREATING — far too long to block
// silently, hence the caller-controlled `readyTimeout`.
isolated function pollKnowledgeBaseActive(BedrockTransport controlTransport, string kbId, decimal timeoutSeconds)
        returns ai:Error? {
    time:Utc deadline = time:utcAddSeconds(time:utcNow(), timeoutSeconds);
    while true {
        map<json> kb = check getKnowledgeBase(controlTransport, kbId);
        string status = stringField(kb, "status") ?: "";
        if status == "ACTIVE" {
            return;
        }
        if status == "FAILED" || status == "DELETE_UNSUCCESSFUL" || status == "UPDATE_UNSUCCESSFUL" {
            return error ai:Error(
                string `Knowledge base '${kbId}' failed to become ACTIVE (status '${status}'): ` +
                failureReasonsOf(kb));
        }
        if time:utcDiffSeconds(deadline, time:utcNow()) <= 0d {
            return error ai:Error(
                string `Timed out after ${timeoutSeconds}s waiting for knowledge base '${kbId}' to become ` +
                string `ACTIVE (still '${status}'). Increase 'readyTimeout', or check the AWS console.`);
        }
        runtime:sleep(KB_POLL_INTERVAL_SECONDS);
    }
}

isolated function failureReasonsOf(map<json> details) returns string {
    json reasonsJson = details["failureReasons"] ?: [];
    if reasonsJson is json[] && reasonsJson.length() > 0 {
        return string:'join("; ", ...reasonsJson.map(r => r.toString()));
    }
    return "no failure reason reported";
}

// ============================================================================
// Data-source creation (when the module creates the knowledge base) and resolution
// (when it attaches to an existing one).
// ============================================================================

// A managed knowledge base REJECTS a bare `{"type": "CUSTOM"}` data source with
// "Unsupported data source type for MANAGED knowledge base type." — the console's
// "Custom" source is really `MANAGED_KNOWLEDGE_BASE_CONNECTOR` with the real type
// nested in `connectorParameters` (established by calling the live API; not
// documented). A self-managed (VECTOR) knowledge base takes the plain form instead
// — see `createVectorDataSourceRequestBody` in knowledgebase_vector_common.bal.
// The `CreateDataSource` request body. Pure, so the two things the live API demands
// — `connectorParameters.version`, and the ABSENCE of `vectorIngestionConfiguration`
// — are assertable without AWS.
isolated function createDataSourceRequestBody(DataSourceDefinition def) returns map<json>|ai:Error {
    map<json> body = {
        name: def.name,
        dataSourceConfiguration: {
            'type: "MANAGED_KNOWLEDGE_BASE_CONNECTOR",
            managedKnowledgeBaseConnectorConfiguration: {
                // `version` is REQUIRED — omitting it returns 400 "The 'version'
                // field is required in connector parameters." (measured), and AWS
                // documents it as "a required version field set to 1".
                // https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-connect-ds.html
                connectorParameters: {'type: "CUSTOM", version: "1"}
            }
        }
        // NO `vectorIngestionConfiguration`. Two separate reasons:
        //   - `parsingConfiguration`: managed knowledge bases only support
        //     SMART_PARSING, which is also the default, so sending it is noise.
        //   - `chunkingConfiguration`: REJECTED outright on a service-managed
        //     embedding model — "A chunking strategy cannot be specified with a
        //     managed embedding model. Omit chunkingConfiguration to use the
        //     default." (measured for NONE, FIXED_SIZE and SEMANTIC alike). AWS's
        //     own docs contradict this and even show a `DEFAULT` strategy value that
        //     the API rejects as invalid.
        //
        // Whether a CALLER-SUPPLIED embedding model (`ManagedEmbeddingModel`) lifts
        // the chunking restriction is UNVERIFIED — the test needs the knowledge
        // base's role to hold `bedrock:InvokeModel` on the embedding model. If it
        // does, this is where a `chunkingConfiguration` would be added, guarded on
        // `def?.embeddingModel is ManagedEmbeddingModel`, and `ChunkingStrategy`
        // becomes reachable through this path.
    };
    string? description = def.description;
    if description is string {
        body["description"] = description;
    }
    return body;
}

isolated function createCustomDataSource(BedrockTransport controlTransport, string kbId, DataSourceDefinition def)
        returns string|ai:Error {
    map<json> body = check createDataSourceRequestBody(def);
    string path = string `/knowledgebases/${kbId}/datasources/`;
    TransportResponse response = check controlTransport.executeRequest("PUT", path, body);
    map<json> dataSource = asMap(asMap(response.body)["dataSource"] ?: {});
    string? id = stringField(dataSource, "dataSourceId");
    if id is () {
        return error ai:Error("CreateDataSource response carried no 'dataSourceId'");
    }
    // Measured SYNCHRONOUS on the managed-KB path (200 AVAILABLE in the very response
    // above, across four probes) — but AWS documents it as ASYNCHRONOUS, "the data
    // source status transitions from CREATING to AVAILABLE". The poll below is
    // therefore not dead code: it is the documented behaviour, just not the observed
    // one.
    string status = stringField(dataSource, "status") ?: "";
    if status != "AVAILABLE" {
        check pollDataSourceAvailable(controlTransport, kbId, id, DEFAULT_DATA_SOURCE_READY_TIMEOUT);
    }
    return id;
}

isolated function pollDataSourceAvailable(BedrockTransport controlTransport, string kbId, string dsId,
        decimal timeoutSeconds) returns ai:Error? {
    time:Utc deadline = time:utcAddSeconds(time:utcNow(), timeoutSeconds);
    while true {
        map<json> dataSource = check getDataSource(controlTransport, kbId, dsId);
        string status = stringField(dataSource, "status") ?: "";
        if status == "AVAILABLE" {
            return;
        }
        if status == "FAILED" || status == "DELETE_UNSUCCESSFUL" {
            return error ai:Error(
                string `Data source '${dsId}' on knowledge base '${kbId}' failed to become AVAILABLE ` +
                string `(status '${status}')`);
        }
        if time:utcDiffSeconds(deadline, time:utcNow()) <= 0d {
            return error ai:Error(
                string `Timed out after ${timeoutSeconds}s waiting for data source '${dsId}' to become ` +
                "AVAILABLE. Increase 'readyTimeout', or check the AWS console.");
        }
        runtime:sleep(KB_POLL_INTERVAL_SECONDS);
    }
}

isolated function getDataSource(BedrockTransport controlTransport, string kbId, string dsId)
        returns map<json>|ai:Error {
    string path = string `/knowledgebases/${kbId}/datasources/${dsId}`;
    TransportResponse response = check controlTransport.executeRequest("GET", path, ());
    return asMap(asMap(response.body)["dataSource"] ?: {});
}

isolated function listDataSources(BedrockTransport controlTransport, string kbId) returns map<json>[]|ai:Error {
    map<json>[] summaries = [];
    string? nextToken = ();
    string path = string `/knowledgebases/${kbId}/datasources/`;
    while true {
        map<json> body = {maxResults: KB_LIST_PAGE_SIZE};
        if nextToken is string {
            body["nextToken"] = nextToken;
        }
        TransportResponse response = check controlTransport.executeRequest("POST", path, body);
        map<json> respBody = asMap(response.body);
        json items = respBody["dataSourceSummaries"] ?: [];
        if items is json[] {
            foreach json item in items {
                summaries.push(asMap(item));
            }
        }
        nextToken = stringField(respBody, "nextToken");
        if nextToken is () {
            break;
        }
    }
    return summaries;
}

// Resolves the single `CUSTOM` data source `ingest()`/`deleteByFilter()` write to.
// `DataSourceSummary` carries no type at all, so this costs one `GetDataSource` per
// data source on the knowledge base.
isolated function resolveCustomDataSource(BedrockTransport controlTransport, string kbId) returns string|ai:Error {
    map<json>[] summaries = check listDataSources(controlTransport, kbId);
    string[] candidates = [];
    foreach map<json> summary in summaries {
        string? dsId = stringField(summary, "dataSourceId");
        if dsId is () {
            continue;
        }
        map<json> dataSource = check getDataSource(controlTransport, kbId, dsId);
        if effectiveDataSourceType(dataSource) == "CUSTOM" {
            candidates.push(dsId);
        }
    }
    if candidates.length() == 1 {
        return candidates[0];
    }
    if candidates.length() == 0 {
        return error ai:Error(
            string `Knowledge base '${kbId}' has no 'CUSTOM' data source: 'ingest()'/'deleteByFilter()' have ` +
            "nowhere to write. Add a CUSTOM (direct-ingestion) data source in the AWS console, or pass a " +
            "'KnowledgeBaseDefinition' instead of a bare id so this class creates one.");
    }
    return error ai:Error(
        string `Knowledge base '${kbId}' has ${candidates.length()} 'CUSTOM' data sources ` +
        string `(${string:'join(", ", ...candidates)}) — ambiguous. Pass 'dataSourceId' explicitly.`);
}

// The data source's EFFECTIVE type: `dataSourceConfiguration.type` directly, except
// when it is `MANAGED_KNOWLEDGE_BASE_CONNECTOR` — the wrapper every managed-KB data
// source uses — in which case the real type is nested inside `connectorParameters`.
//
// OBSERVED ON THE LIVE API, NOT DOCUMENTED: the service model declares
// `connectorParameters` a free-form `Document` (arbitrary JSON) on BOTH the write and
// read paths, but the live `GetDataSource`/`ListDataSources` response returns it as a
// JSON-ENCODED STRING, not an object — asymmetric with what `CreateDataSource`
// accepts. Both shapes are handled here so a future AWS fix does not silently break
// this.
isolated function effectiveDataSourceType(map<json> dataSource) returns string {
    map<json> config = asMap(dataSource["dataSourceConfiguration"] ?: {});
    string wireType = stringField(config, "type") ?: "";
    if wireType != "MANAGED_KNOWLEDGE_BASE_CONNECTOR" {
        return wireType;
    }
    map<json> managed = asMap(config["managedKnowledgeBaseConnectorConfiguration"] ?: {});
    json connectorParams = managed["connectorParameters"] ?: {};
    map<json> parsedParams = {};
    if connectorParams is string {
        json|error parsed = connectorParams.fromJsonString();
        if parsed is map<json> {
            parsedParams = parsed;
        }
    } else if connectorParams is map<json> {
        parsedParams = connectorParams;
    }
    return stringField(parsedParams, "type") ?: wireType;
}

// ============================================================================
// Chunking-strategy detection.
// ============================================================================

// Reads the RESOLVED data source's actual `chunkingStrategy` back — never assumed —
// so `ManagedKnowledgeBaseConfig.chunker`'s default can be picked safely: `ai:DISABLE`
// when Bedrock chunks server-side, `ai:AUTO` when the strategy is `NONE`.
//
// UNRESOLVED (see the `ChunkingStrategy` doc comment for the full write-up): a
// managed knowledge base may never report `chunkingConfiguration` at all. Measured
// 2026-08-14, `GetDataSource` on a console-created managed data source returned
// `vectorIngestionConfiguration: {parsingConfiguration: {parsingStrategy:
// SMART_PARSING}}` and nothing else — despite the console exposing a chunking
// selector for that same data source. That data source was created with "Default
// chunking", so the omission may just mean "nothing explicit was set"; it has NOT
// been tested whether a data source created with an explicit strategy echoes one
// back. If it never does, the `NONE` branch below is unreachable on data sources
// this module did not create, and such callers must pass an `ai:Chunker`
// explicitly. The fallback below is chosen to fail in the safe direction either way.
isolated function detectChunkingStrategy(BedrockTransport controlTransport, string kbId, string dsId)
        returns ChunkingStrategy|ai:Error {
    map<json> dataSource = check getDataSource(controlTransport, kbId, dsId);
    map<json> vectorIngestion = asMap(dataSource["vectorIngestionConfiguration"] ?: {});
    map<json> chunking = asMap(vectorIngestion["chunkingConfiguration"] ?: {});
    // Absent 'chunkingConfiguration' means Bedrock applies its own default, which is
    // FIXED_SIZE — never silently treat a missing field as NONE, or an explicit
    // 'ai:Chunker' would double-chunk without any construction-time warning.
    string strategy = stringField(chunking, "chunkingStrategy") ?: "FIXED_SIZE";
    match strategy {
        "NONE" => {
            return NONE;
        }
        "HIERARCHICAL" => {
            return HIERARCHICAL;
        }
        "SEMANTIC" => {
            return SEMANTIC;
        }
        _ => {
            return FIXED_SIZE;
        }
    }
}

// ============================================================================
// Document operations — ingest / list / get / delete. Shared by `ingest()` and
// `deleteByFilter()` in knowledgebase_managed.bal.
// ============================================================================

isolated function ingestDocumentsBatch(BedrockTransport controlTransport, string kbId, string dsId,
        json[] documents) returns map<json>[]|ai:Error {
    string path = string `/knowledgebases/${kbId}/datasources/${dsId}/documents`;
    map<json> body = {documents};
    TransportResponse response = check controlTransport.executeRequest("PUT", path, body);
    return documentDetailsOf(response.body);
}

isolated function getDocumentsBatch(BedrockTransport controlTransport, string kbId, string dsId, string[] ids)
        returns map<json>[]|ai:Error {
    json[] identifiers = ids.map(id => <json>{dataSourceType: "CUSTOM", custom: {id}});
    string path = string `/knowledgebases/${kbId}/datasources/${dsId}/documents/getDocuments`;
    map<json> body = {documentIdentifiers: identifiers};
    TransportResponse response = check controlTransport.executeRequest("POST", path, body);
    return documentDetailsOf(response.body);
}

isolated function deleteDocumentsBatch(BedrockTransport controlTransport, string kbId, string dsId,
        json[] identifiers) returns map<json>[]|ai:Error {
    string path = string `/knowledgebases/${kbId}/datasources/${dsId}/documents/deleteDocuments`;
    map<json> body = {documentIdentifiers: identifiers};
    TransportResponse response = check controlTransport.executeRequest("POST", path, body);
    return documentDetailsOf(response.body);
}

isolated function listKnowledgeBaseDocuments(BedrockTransport controlTransport, string kbId, string dsId)
        returns map<json>[]|ai:Error {
    map<json>[] details = [];
    string? nextToken = ();
    string path = string `/knowledgebases/${kbId}/datasources/${dsId}/documents`;
    while true {
        map<json> body = {maxResults: KB_LIST_PAGE_SIZE};
        if nextToken is string {
            body["nextToken"] = nextToken;
        }
        TransportResponse response = check controlTransport.executeRequest("POST", path, body);
        map<json> respBody = asMap(response.body);
        foreach map<json> detail in documentDetailsOf(respBody) {
            details.push(detail);
        }
        nextToken = stringField(respBody, "nextToken");
        if nextToken is () {
            break;
        }
    }
    return details;
}

isolated function documentDetailsOf(json body) returns map<json>[] {
    json items = asMap(body)["documentDetails"] ?: [];
    map<json>[] details = [];
    if items is json[] {
        foreach json item in items {
            details.push(asMap(item));
        }
    }
    return details;
}

isolated function documentIdOf(map<json> detail) returns string? {
    map<json> identifier = asMap(detail["identifier"] ?: {});
    return stringField(asMap(identifier["custom"] ?: {}), "id");
}

# The final (terminal) outcome of one submitted document: its last-seen status and,
# on failure, the reason Bedrock reported.
#
# + status - The terminal `DocumentStatus` (see `KB_DOC_USABLE_STATUSES`/`KB_DOC_FAILED_STATUSES`)
# + statusReason - Bedrock's explanation, present mainly alongside `IGNORED`
type DocumentOutcome record {|
    string status;
    string? statusReason;
|};

// Polls `GetKnowledgeBaseDocuments` until every id in `ids` leaves
// `KB_DOC_IN_FLIGHT_STATUSES`, or `timeoutSeconds` elapses. ~14s was measured for a
// single small document — ingestion is inherently slow, hence the caller-controlled
// timeout rather than a fixed short one.
isolated function pollDocumentsTerminal(BedrockTransport controlTransport, string kbId, string dsId,
        string[] ids, decimal timeoutSeconds) returns map<DocumentOutcome>|ai:Error {
    map<DocumentOutcome> outcomes = {};
    string[] pending = ids.clone();
    time:Utc deadline = time:utcAddSeconds(time:utcNow(), timeoutSeconds);
    while pending.length() > 0 {
        string[] stillPending = [];
        foreach string[] batch in partitionStrings(pending, KB_DOCUMENT_BATCH_SIZE) {
            map<json>[] details = check getDocumentsBatch(controlTransport, kbId, dsId, batch);
            foreach map<json> detail in details {
                string? id = documentIdOf(detail);
                if id is () {
                    continue;
                }
                string status = stringField(detail, "status") ?: "";
                if KB_DOC_IN_FLIGHT_STATUSES.indexOf(status) is int {
                    stillPending.push(id);
                } else {
                    outcomes[id] = {status, statusReason: stringField(detail, "statusReason")};
                }
            }
        }
        pending = stillPending;
        if pending.length() == 0 {
            break;
        }
        if time:utcDiffSeconds(deadline, time:utcNow()) <= 0d {
            return error ai:Error(
                string `Timed out after ${timeoutSeconds}s waiting for ${pending.length()} document(s) to ` +
                string `finish indexing (still in progress: ${string:'join(", ", ...pending)}). Increase ` +
                "'ingestTimeout' — indexing took ~47s for a single 97KB document in testing.");
        }
        runtime:sleep(KB_POLL_INTERVAL_SECONDS);
    }
    return outcomes;
}

# One enumerated, retrievable document that `deleteByFilter` can potentially delete.
#
# + sourceValue - `customDocumentIdentifier.id` (CUSTOM) or the S3 object URI (S3) — also what `_source_uri` holds
# + identifier - The ready-to-send `DocumentIdentifier` for `DeleteKnowledgeBaseDocuments`
type DeletableDocument record {|
    string sourceValue;
    json identifier;
|};

// Enumerates the documents on one data source that are both retrievable
// (`KB_DOC_USABLE_STATUSES`) and deletable (`dataSourceType` is `CUSTOM` or `S3` —
// `DocumentIdentifier` has no other members). `ListKnowledgeBaseDocuments` is
// exhaustive but carries no metadata, which is why `deleteByFilter` still has to
// probe each one through `Retrieve` — see knowledgebase_managed.bal.
isolated function listDeletableDocuments(BedrockTransport controlTransport, string kbId, string dsId,
        string dataSourceType) returns DeletableDocument[]|ai:Error {
    DeletableDocument[] docs = [];
    foreach map<json> detail in check listKnowledgeBaseDocuments(controlTransport, kbId, dsId) {
        string status = stringField(detail, "status") ?: "";
        if KB_DOC_USABLE_STATUSES.indexOf(status) is () {
            // Skip NOT_FOUND tombstones, FAILED, and anything still in flight —
            // none of these are reachable through Retrieve to probe against.
            continue;
        }
        map<json> identifier = asMap(detail["identifier"] ?: {});
        if dataSourceType == "CUSTOM" {
            string? id = stringField(asMap(identifier["custom"] ?: {}), "id");
            if id is string {
                docs.push({sourceValue: id, identifier: {dataSourceType: "CUSTOM", custom: {id}}});
            }
        } else if dataSourceType == "S3" {
            string? uri = stringField(asMap(identifier["s3"] ?: {}), "uri");
            if uri is string {
                docs.push({sourceValue: uri, identifier: {dataSourceType: "S3", s3: {uri}}});
            }
        }
    }
    return docs;
}

// ============================================================================
// Retrieve (bedrock-agent-runtime).
// ============================================================================

// One `Retrieve` round trip on the MANAGED search branch. `filter` is an
// already-built `RetrievalFilter` JSON value (see knowledgebase_filter.bal) or `()`
// to search unfiltered. Returns the raw `retrievalResults[]` plus a `nextToken` for
// the caller to page with.
isolated function callRetrieve(BedrockTransport dataTransport, string kbId, string query, json? filter,
        int numberOfResults, RerankingModelType? reranking, string? nextToken)
        returns [json[], string?]|ai:Error {
    map<json> managedSearch = {numberOfResults};
    if filter is json {
        managedSearch["filter"] = filter;
    }
    if reranking is RerankingModelType {
        managedSearch["rerankingModelType"] = reranking;
    }
    map<json> body = {
        retrievalQuery: {text: query},
        retrievalConfiguration: {managedSearchConfiguration: managedSearch}
    };
    if nextToken is string {
        body["nextToken"] = nextToken;
    }
    string path = string `/knowledgebases/${kbId}/retrieve`;
    TransportResponse response = check dataTransport.executeRequest("POST", path, body);
    map<json> respBody = asMap(response.body);
    json resultsJson = respBody["retrievalResults"] ?: [];
    json[] results = resultsJson is json[] ? resultsJson : [];
    return [results, stringField(respBody, "nextToken")];
}

// The `deleteByFilter` probe: does at least one result come back for the pinned
// document under `filter`? `numberOfResults: 1` is enough, and is enough even for a
// document that split into dozens of chunks — every chunk of a document inherits
// that document's `metadata` (it is attached per-DOCUMENT on
// `IngestKnowledgeBaseDocuments`, and there is no per-chunk metadata input), so a
// filter matches ALL of a document's chunks or NONE of them. One surviving chunk is
// therefore complete evidence about the whole document, and nothing a second chunk
// could add would change the answer.
//
// Reranking is deliberately NOT applied: it imposes its own relevance cut on top of
// the search, which could drop the single result this existence check depends on.
isolated function retrieveHasMatch(BedrockTransport dataTransport, string kbId, json filter)
        returns boolean|ai:Error {
    [json[], string?] [results, _] = check callRetrieve(dataTransport, kbId, FILTER_PROBE_QUERY, filter, 1, (), ());
    return results.length() > 0;
}

// ============================================================================
// Small pure helpers.
// ============================================================================

isolated function asMap(json j) returns map<json> => j is map<json> ? j : {};

isolated function stringField(map<json> m, string key) returns string? {
    json v = m[key] ?: ();
    return v is string ? v : ();
}

isolated function partitionStrings(string[] items, int batchSize) returns string[][] {
    string[][] batches = [];
    int index = 0;
    while index < items.length() {
        int end = index + batchSize;
        if end > items.length() {
            end = items.length();
        }
        batches.push(items.slice(index, end));
        index = end;
    }
    return batches;
}

isolated function partitionJson(json[] items, int batchSize) returns json[][] {
    json[][] batches = [];
    int index = 0;
    while index < items.length() {
        int end = index + batchSize;
        if end > items.length() {
            end = items.length();
        }
        batches.push(items.slice(index, end));
        index = end;
    }
    return batches;
}
