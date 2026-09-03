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

// resolveRoute — the resolution ladder. Pure: no I/O, no state, fully table-testable
// without AWS credentials. All construction-time routing
// decisions are made here.

// Resolves a model id (bare, CRIS-prefixed, ARN, or `mantle/|converse/|invoke/`
// prefixed) to a fully-specified `Route`. Returns an `error` for
// the cases AWS cannot diagnose for us: `custom-model/` and `imported-model/`
// ARNs, and `apiFamily = MANTLE` on a model absent from `MANTLE_CAPABLE`.
isolated function resolveRoute(string model, string region, RouteConfig config = {}) returns Route|error {
    // ---- step 1: explicit override (prefix and/or config.apiFamily) ----
    // AUTO (the config default) means "no forced family" — run the ladder.
    [RouteFamily?, string] [prefixFamily, work] = stripRoutePrefix(model);
    // Narrowing to `RouteFamily` IS the AUTO filter: `AUTO` is the one `ApiFamily`
    // member that is not a destination, so anything that survives the type test is
    // a forced family.
    ApiFamily? rawConfigFamily = config.apiFamily;
    RouteFamily? configFamily = rawConfigFamily is RouteFamily ? rawConfigFamily : ();
    RouteFamily? explicitFamily = configFamily ?: prefixFamily;

    // ---- step 2: ARN dispatch (region + partition are authoritative) ----
    if isArn(work) {
        return resolveArn(work, region, config, explicitFamily);
    }

    // ---- bare / CRIS-prefixed id: normalize then walk steps 3-6 ----
    string effRegion = region;
    string partition = partitionForRegion(region);
    return resolveBareId(work, effRegion, partition, config, explicitFamily);
}

// Splits an optional `mantle/|converse/|invoke/` route prefix off the model
// string. Returns the implied family (if any) and the
// remaining id.
isolated function stripRoutePrefix(string model) returns [RouteFamily?, string] {
    if model.startsWith("mantle/") {
        return [MANTLE, model.substring("mantle/".length())];
    }
    if model.startsWith("converse/") {
        return [CONVERSE, model.substring("converse/".length())];
    }
    if model.startsWith("invoke/") {
        return [INVOKE, model.substring("invoke/".length())];
    }
    return [(), model];
}

// ARN dispatch — the resource-type token gives the family before any call.
// The ARN's region/partition override `config.region`.
isolated function resolveArn(string arnStr, string region, RouteConfig config, RouteFamily? explicitFamily)
        returns Route|error {
    ParsedArn arn = check parseArn(arnStr);

    if arn.'service != "bedrock" {
        return error(string `not a Bedrock ARN: service segment is '${arn.'service}', expected 'bedrock'`);
    }

    // The ARN's region is authoritative — but it is legitimately EMPTY on
    // global ARNs such as `arn:aws:bedrock::123:foundation-model/anthropic.claude-v2`.
    // Copying "" through would build the host `bedrock-runtime..amazonaws.com` and
    // surface as an opaque DNS failure, so fall back to the caller's region.
    string arnRegion = arn.region == "" ? region : arn.region;

    // `foundation-model/` carries a bare, globally-addressable id — strip to it
    // and fall through to the allowlists.
    if arn.resourceType == "foundation-model" {
        return resolveBareId(arn.resourceId, arnRegion, arn.partition, config, explicitFamily);
    }

    // `custom-model/` is an artifact, not a deployment — AWS's prose directs
    // users to the deployment/Provisioned-Throughput ARN. Policy
    // choice, recorded so a reviewer can overrule it.
    if arn.resourceType == "custom-model" {
        return error(string `'custom-model/' ARN is a model artifact, not a deployment; ` +
            string `pass the 'custom-model-deployment/' (on-demand) or 'provisioned-model/' ARN instead`);
    }

    // `imported-model/` (Custom Model Import) is out of scope. AWS applies no default
    // chat template to imported weights, so the request body cannot be built without
    // the caller naming the wire dialect — which was the whole job of the removed
    // `modelSchema` config. Rather than keep a knob on every provider for a case
    // integration developers do not hit, refuse it by name.
    if arn.resourceType == "imported-model" {
        return error(string `'imported-model/' ARNs are not supported: AWS applies no default chat ` +
            string `template to imported weights, so this module cannot build a request body for them. ` +
            string `Use a foundation-model, inference-profile, or provisioned-model ARN instead`);
    }

    // Every remaining opaque ARN keeps its family through the sink.
    RouteFamily family = explicitFamily ?: CONVERSE;

    MantleEntry? mantleEntry = ();
    if family == MANTLE {
        // An opaque ARN is not a bare id, so it cannot be in MANTLE_CAPABLE.
        mantleEntry = check mantleEntryForBare(arnStr);
    }

    return {
        family,
        bareModelId: arnStr,
        geoPrefix: (),
        effectiveModelId: arnStr, // opaque ARNs go on the wire verbatim (URL-encoded in endpoint.bal)
        region: arnRegion,
        partition: arn.partition,
        mantleEntry
    };
}

// Bare/CRIS-prefixed id: normalize (strip geo prefix, keep it) then walk ladder
// steps 3-6.
isolated function resolveBareId(string id, string region, string partition, RouteConfig config,
        RouteFamily? explicitFamily) returns Route|error {
    [string, string?] [bareId, geoPrefix] = normalizeModelId(id);

    // step 1 (explicit): outranks the tables.
    if explicitFamily is RouteFamily {
        return buildBareRoute(explicitFamily, bareId, geoPrefix, region, partition);
    }

    // step 2: AUTO preference order MANTLE → CONVERSE → INVOKE. A
    // Mantle-capable model prefers Mantle; membership in MANTLE_CAPABLE means we hold
    // a verified path/auth/converter for it, so this is still table-driven — an unknown
    // model is absent from the table and sinks to Converse, never Mantle by
    // elimination (routing by elimination stays forbidden; only the preference for
    // the KNOWN case flips).
    //
    // GUARD: only a BARE id prefers Mantle. A CRIS geo prefix (`us.`, `eu.`, ...) is a
    // bedrock-RUNTIME concept — Mantle has no geo prefixes — so a geo-prefixed id
    // signals cross-region runtime intent and stays on Converse
    // (`us.anthropic.claude-opus-4-8` → Converse with the prefix re-applied).
    if geoPrefix is () {
        MantleEntry? entry = MANTLE_CAPABLE[bareId];
        if entry is MantleEntry {
            return mantleRoute(bareId, geoPrefix, region, partition, entry);
        }
    }

    // step 3: everything else (and every geo-prefixed id) → CONVERSE.
    return buildBareRoute(CONVERSE, bareId, geoPrefix, region, partition);
}

// Builds a CONVERSE/INVOKE/MANTLE route from a resolved family + bare id.
isolated function buildBareRoute(RouteFamily family, string bareId, string? geoPrefix, string region,
        string partition) returns Route|error {
    if family == MANTLE {
        return mantleRoute(bareId, geoPrefix, region, partition, check mantleEntryForBare(bareId));
    }
    // CONVERSE/INVOKE take the CRIS-prefixed id on the wire.
    return {
        family,
        bareModelId: bareId,
        geoPrefix,
        effectiveModelId: applyGeoPrefix(bareId, geoPrefix),
        region,
        partition,
        mantleEntry: ()
    };
}

// Builds a MANTLE route. Mantle takes the BARE id on the wire — never the CRIS
// geo prefix — unless the entry names a different Mantle-side id.
isolated function mantleRoute(string bareId, string? geoPrefix, string region, string partition,
        MantleEntry entry) returns Route {
    return {
        family: MANTLE,
        bareModelId: bareId,
        geoPrefix,
        // A model may be published under different ids per endpoint (see
        // `MantleEntry.modelId`); the entry wins when it says so.
        effectiveModelId: entry?.modelId ?: bareId,
        region,
        partition,
        mantleEntry: entry
    };
}

// MANTLE_CAPABLE lookup for a bare id. Capability ≠ membership: a
// model the user forced onto Mantle must have a path entry, because a Mantle path
// is not derivable from the model id. A model AWS has added since our last release
// therefore cannot be forced onto Mantle until the table ships it.
isolated function mantleEntryForBare(string bareId) returns MantleEntry|error {
    MantleEntry? entry = MANTLE_CAPABLE[bareId];
    if entry is () {
        return error(string `model '${bareId}' is not available on Mantle (no known request path). ` +
            string `Use 'apiFamily = CONVERSE' or 'INVOKE', or upgrade the module if AWS has since ` +
            string `added it to bedrock-mantle`);
    }
    return entry;
}

// Strips a CRIS geo prefix for lookup, keeping it for per-family re-application.
// Returns [bareId, geoPrefix?].
isolated function normalizeModelId(string id) returns [string, string?] {
    int? dot = id.indexOf(".");
    if dot is int {
        string maybePrefix = id.substring(0, dot);
        if CRIS_PREFIXES.indexOf(maybePrefix) is int {
            return [id.substring(dot + 1), maybePrefix];
        }
    }
    return [id, ()];
}

// Re-applies a CRIS geo prefix to a bare id (Converse/Invoke wire form).
isolated function applyGeoPrefix(string bareId, string? geoPrefix) returns string
    => geoPrefix is string ? string `${geoPrefix}.${bareId}` : bareId;

// Partition inferred from a region string. ARNs carry their own
// partition; bare-id routes derive it here.
isolated function partitionForRegion(string region) returns string {
    if region.startsWith("us-gov-") {
        return "aws-us-gov";
    }
    if region.startsWith("cn-") {
        return "aws-cn";
    }
    return "aws";
}
