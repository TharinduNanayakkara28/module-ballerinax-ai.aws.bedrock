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

# A Bedrock **self-managed** knowledge base — `KnowledgeBaseConfiguration.type` is
# `VECTOR`, meaning your own vector store rather than Bedrock's — exposed through
# `ai:KnowledgeBase`. This is the console's *Self-managed KB → Unstructured Vector
# Store KB*.
#
# Two ways to use it, mirroring `BedrockManagedKnowledgeBase`:
#
# - **Attach to an existing knowledge base** (pass its id): `retrieve()` searches
#   every data source on it; `ingest()`/`deleteByFilter()` need it to also have a
#   `CUSTOM` (direct-ingestion) data source — construction fails, naming why, if it
#   does not.
# - **Create and own it end to end** (pass a `VectorKnowledgeBaseDefinition`): this
#   class creates the knowledge base and a `CUSTOM` data source, and every document
#   flows through `ingest()`. Find-or-create by NAME.
#
# ## The vector store must already exist
#
# This class never provisions one. `CreateKnowledgeBase` accepts only a
# `storageConfiguration` naming an existing collection, cluster, table, or bucket;
# the console's "Quick create a new vector store" has no API equivalent — *"If you
# prefer to let Amazon Bedrock create and manage a vector store for you, use the
# console"*. Provision it with Terraform/CDK/the console, then pass its ARNs.
# https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html
#
# ## IAM
#
# `ingest()` needs BOTH `bedrock:StartIngestionJob` and
# `bedrock:IngestKnowledgeBaseDocuments` — AWS reports only one as missing per
# attempt, so granting the one named in the first `AccessDenied` fails again on the
# other. The knowledge base's own `roleArn`
# additionally needs permissions on your vector store; those belong to that role,
# not to this client's credentials.
# https://docs.aws.amazon.com/bedrock/latest/userguide/kb-permissions.html
#
# ## What is not measured
#
# Every behaviour `BedrockManagedKnowledgeBase` relies on was measured against the
# live API on Bedrock's own vector store. None of those checks ran against a
# self-managed one, so on this class the equivalents are documented-but-unmeasured — see
# `VECTOR_SOURCE_URI_METADATA_KEY` and `deleteByFilter` below. Where the managed
# class could simplify on the strength of a measurement, this one keeps the
# conservative form.
public distinct isolated client class BedrockVectorKnowledgeBase {
    *ai:KnowledgeBase;

    private final BedrockTransport controlTransport;
    private final BedrockTransport dataTransport;
    private final string knowledgeBaseId;
    private final string dataSourceId;
    private final ai:Chunker|ai:AUTO|ai:DISABLE chunker;
    private final decimal ingestTimeout;
    private final int? numberOfResults;
    private final SearchType? overrideSearchType;
    // `readonly &` because the class is `isolated`: a plain mutable record could not
    // be held in a `final` field, nor read outside a `lock`. `SearchType` needs no
    // such treatment — it is an enum, and so already immutable.
    private final readonly & VectorRerankingConfig? rerankingConfiguration;

    # + knowledgeBase - An existing knowledge base id/ARN, or a `VectorKnowledgeBaseDefinition` to
    #                   find-or-create by name. The vector store it names must already exist
    # + credentials - Defaults to the full AWS credential chain (env vars, EKS IRSA,
    #                 SSO, shared config, `credential_process`, ECS container credentials,
    #                 EC2 IMDSv2), so nothing needs configuring on AWS compute. Pass an
    #                 `auth:StaticAuthConfig`, `auth:AssumeRoleConfig`, ... for an explicit source.
    #                 SigV4 only — Bedrock API keys do not work on the agent planes, see
    #                 `KnowledgeBaseCredentials`
    # + region - Defaults to AWS_REGION/AWS_DEFAULT_REGION
    # + serviceUrl - Endpoint origin for BOTH agent planes (`bedrock-agent` and
    #                `bedrock-agent-runtime`). The default template resolves per plane from AWS
    #                SDK endpoint metadata. Override it only for a host AWS cannot derive
    #                (PrivateLink, an egress gateway, or a local mock)
    # + config - Data source override, chunking, ingest/retrieve tuning, HTTP/retry settings
    # + return - `nil` on success; otherwise an `ai:Error`
    public isolated function init(
            @display {label: "Knowledge Base"} string|VectorKnowledgeBaseDefinition knowledgeBase,
            @display {label: "AWS Credentials"} KnowledgeBaseCredentials credentials = auth:DEFAULT_CREDENTIALS,
            @display {label: "Region"} string region = defaultRegion(),
            @display {label: "Service URL"} string serviceUrl = DEFAULT_SERVICE_URL,
            @display {label: "Configuration"} *VectorKnowledgeBaseConfig config)
            returns ai:Error? {
        KbSpine spine = check resolveVectorKbSpine("BedrockVectorKnowledgeBase", credentials, region, serviceUrl,
            knowledgeBase, config);
        self.controlTransport = spine.controlTransport;
        self.dataTransport = spine.dataTransport;
        self.knowledgeBaseId = spine.knowledgeBaseId;
        self.dataSourceId = spine.dataSourceId;
        self.chunker = check resolveChunker(config?.chunker, spine.chunkingStrategy);
        self.ingestTimeout = config.ingestTimeout;
        self.numberOfResults = config?.numberOfResults;
        self.overrideSearchType = config?.overrideSearchType;
        VectorRerankingConfig? reranking = config?.rerankingConfiguration;
        self.rerankingConfiguration = reranking is VectorRerankingConfig ? reranking.cloneReadOnly() : ();
    }

    # Ingests documents into the `CUSTOM` data source, chunking client-side first
    # when the data source's `chunkingStrategy` is `NONE` (detected at construction —
    # see `VectorKnowledgeBaseConfig.chunker`).
    #
    # SLOW BY NATURE: `IngestKnowledgeBaseDocuments` returns 202 as soon as Bedrock
    # has accepted the documents, not once they are indexed. This call therefore
    # blocks until every document reaches a terminal status or `ingestTimeout`
    # elapses, so a `retrieve()` immediately afterward sees them.
    #
    # There is deliberately no fire-and-forget mode. `ai:KnowledgeBase.ingest`
    # returns a bare `Error?` with no job handle and no status method, so returning
    # at the 202 would report success for a document that later lands `FAILED` and
    # leave the caller no way to ever discover it.
    #
    # + documents - The documents or chunks to index; only text content is supported
    # + return - An `ai:Error` if any document fails to submit or to index; `nil` otherwise
    public isolated function ingest(ai:Chunk[]|ai:Document[]|ai:Document documents) returns ai:Error? {
        (ai:Chunk|ai:Document)[] items = documents is ai:Chunk[]|ai:Document[] ? documents : [documents];
        (ai:Chunk|ai:Document)[] chunked = check self.applyChunker(items);

        string[] documentIds = [];
        json[] wireDocuments = [];
        foreach ai:Chunk|ai:Document item in chunked {
            [json, string] [wireDoc, id] = check chunkToKnowledgeBaseDocument(item);
            wireDocuments.push(wireDoc);
            documentIds.push(id);
        }

        foreach json[] batch in partitionJson(wireDocuments, KB_DOCUMENT_BATCH_SIZE) {
            map<json>[] _ = check ingestDocumentsBatch(self.controlTransport, self.knowledgeBaseId,
                self.dataSourceId, batch);
        }

        map<DocumentOutcome> outcomes = check pollDocumentsTerminal(self.controlTransport, self.knowledgeBaseId,
            self.dataSourceId, documentIds, self.ingestTimeout);
        string[] failed = [];
        foreach [string, DocumentOutcome] [id, outcome] in outcomes.entries() {
            if KB_DOC_FAILED_STATUSES.indexOf(outcome.status) is int {
                failed.push(string `${id} (${outcome.status}: ${outcome.statusReason ?: "no reason reported"})`);
            }
        }
        if failed.length() > 0 {
            return error ai:Error(
                string `${failed.length()} of ${documentIds.length()} document(s) failed to index: ` +
                string:'join("; ", ...failed));
        }
    }

    # Retrieves relevant chunks. Searches across EVERY data source on the knowledge
    # base, not just the `CUSTOM` one `ingest()` writes to.
    #
    # Sends the `vectorSearchConfiguration` branch — never
    # `managedSearchConfiguration`, which is the managed knowledge base's branch and
    # carries different members.
    #
    # + query - The text query to search for
    # + maxLimit - The maximum number of items to return, or `-1` for no limit (subject to your vector store's own relevance cutoff)
    # + filters - Optional metadata filters. Operator support is BACKEND-DEPENDENT — see the module README
    # + return - Matching chunks with similarity scores, or an `ai:Error`
    public isolated function retrieve(string query, int maxLimit = 10, ai:MetadataFilters? filters = ())
            returns ai:QueryMatch[]|ai:Error {
        if maxLimit != -1 && maxLimit <= 0 {
            return error ai:Error("'maxLimit' must be a positive integer, or -1 for no limit");
        }
        json? userFilter = ();
        if filters is ai:MetadataFilters {
            userFilter = check metadataFiltersToRetrievalFilter(filters);
        }

        int configuredCap = self.numberOfResults ?: KB_MAX_RESULTS_PER_CALL;
        int cap = configuredCap < KB_MAX_RESULTS_PER_CALL ? configuredCap : KB_MAX_RESULTS_PER_CALL;
        int perCall = maxLimit == -1 ? cap : (maxLimit < cap ? maxLimit : cap);

        ai:QueryMatch[] matches = [];
        string? nextToken = ();
        while true {
            [json[], string?] [results, respNextToken] = check callVectorRetrieve(self.dataTransport,
                self.knowledgeBaseId, query, userFilter, perCall, self.overrideSearchType,
                self.rerankingConfiguration, nextToken);
            foreach json result in results {
                matches.push(check retrievalResultToQueryMatch(result));
                if maxLimit != -1 && matches.length() >= maxLimit {
                    return matches.slice(0, maxLimit);
                }
            }
            nextToken = respNextToken;
            if nextToken is () {
                break;
            }
        }
        return matches;
    }

    # Deletes documents matching `filters`.
    #
    # Bedrock has no metadata-based delete and no way to read a document's metadata
    # back — `KnowledgeBaseDocumentDetail` carries only `dataSourceId`, `identifier`,
    # `knowledgeBaseId`, `status`, `statusReason` and `updatedAt`, with no metadata
    # member on any data source type. So this is a reconstruction: enumerate every
    # document, then ask `Retrieve` one yes/no question per document — the caller's
    # filter ANDed with a leaf pinning that ONE document by
    # `x-amz-bedrock-kb-source-uri`.
    #
    # ## Why this keeps a second probe the managed class does not
    #
    # Pinning exists because sending the caller's filter alone under-deletes: a
    # relevance floor drops matching documents with no `nextToken` to signal
    # truncation. `BedrockManagedKnowledgeBase` was able to drop its follow-up probe
    # after measuring that pinned probes score 0.88-1.0 against a floor near 0.15,
    # making a zero-hit result unambiguous.
    #
    # THAT MEASUREMENT DOES NOT TRANSFER HERE. It was taken against Bedrock's own
    # vector store; on a self-managed knowledge base the ranking engine is YOUR
    # store's, and nothing guarantees a pinned probe clears its floor the same way.
    # So a zero-hit result is re-probed with the id leaf ALONE: if that also returns
    # nothing, the document was unreachable rather than genuinely excluded, and it is
    # reported as indeterminate instead of being silently skipped — which would
    # under-delete. Once the equivalent measurement exists for a given backend, this
    # second probe can be dropped exactly as the managed class dropped its own.
    #
    # Costs one to two `Retrieve` calls per document in the knowledge base — a
    # maintenance operation, not something to put on a request path.
    #
    # Runs over every data source, not only the `CUSTOM` one `ingest()` writes to.
    # `DocumentIdentifier.dataSourceType` has only `CUSTOM`/`S3` members, so documents
    # from any other data source cannot be deleted through this API at all — those are
    # named in the returned error rather than silently skipped.
    #
    # Deletes that CAN be made still happen even when some documents or data sources
    # cannot be reached.
    #
    # + filters - The metadata filters used to identify which documents to delete
    # + return - An `ai:Error` naming indeterminate documents or undeletable data sources; `nil` otherwise
    public isolated function deleteByFilter(ai:MetadataFilters filters) returns ai:Error? {
        json? userFilter = check metadataFiltersToRetrievalFilter(filters);
        // A filter set that constrains nothing would make every per-document probe
        // "does this document exist" — every one hits, and the whole knowledge base
        // is deleted. `ai:KnowledgeBase.deleteByFilter` takes filters as a required
        // argument, so a caller assembling them from a collection that happened to be
        // empty would get silent total deletion. Refuse instead: "delete everything"
        // must be explicit, never a degenerate case.
        //
        // Both conditions are needed. A wholly empty group yields `()`; a group of
        // nested EMPTY groups instead yields a non-nil `{"andAll": [null, null]}`
        // (see `vectorFilterLeafCount`), which a nil check alone would let through.
        if userFilter is () || vectorFilterLeafCount(filters) == 0 {
            return error ai:Error(
                "deleteByFilter requires at least one metadata filter — an 'ai:MetadataFilters' with no " +
                "leaf predicates matches every document, which would delete the entire knowledge base. " +
                "Pass a filter that selects the documents to remove.");
        }

        map<json>[] dataSourceSummaries = check listDataSources(self.controlTransport, self.knowledgeBaseId);
        string[] undeletableDataSources = [];
        string[] indeterminate = [];
        map<json[]> toDeleteByDataSource = {};

        foreach map<json> summary in dataSourceSummaries {
            string? dsId = stringField(summary, "dataSourceId");
            if dsId is () {
                continue;
            }
            map<json> dataSource = check getDataSource(self.controlTransport, self.knowledgeBaseId, dsId);
            string effectiveType = effectiveDataSourceType(dataSource);
            if effectiveType != "CUSTOM" && effectiveType != "S3" {
                undeletableDataSources.push(string `${dsId} (${effectiveType})`);
                continue;
            }

            DeletableDocument[] candidates =
                check listDeletableDocuments(self.controlTransport, self.knowledgeBaseId, dsId, effectiveType);
            json[] matchesForThisSource = [];
            foreach DeletableDocument candidate in candidates {
                json probeFilter = withVectorSourceUriFilter(userFilter, candidate.sourceValue);
                if check vectorRetrieveHasMatch(self.dataTransport, self.knowledgeBaseId, probeFilter,
                        candidate.sourceValue) {
                    matchesForThisSource.push(candidate.identifier);
                    continue;
                }
                // Zero hits: re-probe with the id leaf alone to tell "genuinely
                // excluded by the filter" from "the store's relevance floor hid it
                // from the first probe too" (or ignored the pin entirely).
                json reachabilityFilter = withVectorSourceUriFilter((), candidate.sourceValue);
                if !check vectorRetrieveHasMatch(self.dataTransport, self.knowledgeBaseId, reachabilityFilter,
                        candidate.sourceValue) {
                    indeterminate.push(string `${candidate.sourceValue} (data source ${dsId})`);
                }
                // else: reachable and genuinely excluded by the filter — skip, sound.
            }
            if matchesForThisSource.length() > 0 {
                toDeleteByDataSource[dsId] = matchesForThisSource;
            }
        }

        foreach [string, json[]] [dsId, identifiers] in toDeleteByDataSource.entries() {
            foreach json[] batch in partitionJson(identifiers, KB_DOCUMENT_BATCH_SIZE) {
                map<json>[] _ = check deleteDocumentsBatch(self.controlTransport, self.knowledgeBaseId, dsId, batch);
            }
        }

        if indeterminate.length() == 0 && undeletableDataSources.length() == 0 {
            return;
        }
        string[] problems = [];
        if indeterminate.length() > 0 {
            problems.push(string `${indeterminate.length()} document(s) could not be confirmed to match or not ` +
                string `match the filter (the vector store's relevance floor hid them from the probe): ` +
                string:'join(", ", ...indeterminate));
        }
        if undeletableDataSources.length() > 0 {
            problems.push(string `${undeletableDataSources.length()} data source(s) are not deletable through ` +
                string `this API — only CUSTOM/S3 support 'DeleteKnowledgeBaseDocuments': ` +
                string:'join(", ", ...undeletableDataSources));
        }
        return error ai:Error(
            string `deleteByFilter deleted every confirmed match, but: ${string:'join("; ", ...problems)}`);
    }

    private isolated function applyChunker((ai:Chunk|ai:Document)[] items) returns (ai:Chunk|ai:Document)[]|ai:Error {
        ai:Chunker|ai:AUTO|ai:DISABLE chunker = self.chunker;
        if chunker is ai:DISABLE {
            return items;
        }
        (ai:Chunk|ai:Document)[] chunked = [];
        foreach ai:Chunk|ai:Document item in items {
            ai:Chunker chunkerToUse = chunker is ai:Chunker ? chunker : guessChunkerForKb(item);
            chunked.push(...check chunkerToUse.chunk(item));
        }
        return chunked;
    }
}
