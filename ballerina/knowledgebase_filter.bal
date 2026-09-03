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

// `ai:MetadataFilters` -> Bedrock `RetrievalFilter` JSON. Pure, no I/O — golden/
// table-testable directly. Two wire constraints that would otherwise surface as a
// runtime 400 from Bedrock rather than a construction-time or call-time mistake:
//
//   1. `RetrievalFilter` is a UNION — exactly one member (`equals`, `andAll`, ...)
//      may be set on any single JSON object.
//   2. `RetrievalFilterList` (the value of `andAll`/`orAll`) has `min: 2`. A group
//      that reduces to a single child must be FLATTENED to that child's own JSON,
//      not wrapped in a one-element array.
// https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_RetrievalFilter.html

// The eight `ai:MetadataFilterOperator` members, mapped 1:1 to their `RetrievalFilter`
// field name. `IN`/`NOT_IN` are the only operators whose `value` must be an array —
// enforced here rather than left for Bedrock to reject with an opaque 400.
isolated function metadataFilterToRetrievalFilter(ai:MetadataFilter filter) returns json|ai:Error {
    string key = filter.key;
    json value = filter.value;
    if (filter.operator == ai:IN || filter.operator == ai:NOT_IN) && value !is json[] {
        return error ai:Error(
            string `MetadataFilter operator '${filter.operator}' requires an array 'value' (got ` +
            string `${value.toJsonString()}) for key '${key}'`);
    }
    json attribute = {key, value};
    match filter.operator {
        ai:EQUAL => {
            return {'equals: attribute};
        }
        ai:NOT_EQUAL => {
            return {notEquals: attribute};
        }
        ai:GREATER_THAN => {
            return {greaterThan: attribute};
        }
        ai:GREATER_THAN_OR_EQUAL => {
            return {greaterThanOrEquals: attribute};
        }
        ai:LESS_THAN => {
            return {lessThan: attribute};
        }
        ai:LESS_THAN_OR_EQUAL => {
            return {lessThanOrEquals: attribute};
        }
        ai:IN => {
            return {'in: attribute};
        }
        ai:NOT_IN => {
            return {notIn: attribute};
        }
    }
    // Unreachable: `ai:MetadataFilterOperator` is a closed 8-member enum and every
    // member is matched above. Kept only because `match` on an open string type
    // requires an exhaustive-looking clause to compile.
    return error ai:Error(string `Unsupported metadata filter operator: '${filter.operator}'`);
}

// `ai:MetadataFilters` (a possibly-nested AND/OR group) -> a single `RetrievalFilter`
// JSON value, or `()` when the group is empty (the caller then omits Bedrock's
// `filter` field entirely rather than sending a meaningless empty union).
isolated function metadataFiltersToRetrievalFilter(ai:MetadataFilters filters) returns json?|ai:Error {
    json[] children = [];
    foreach ai:MetadataFilters|ai:MetadataFilter child in filters.filters {
        json? childJson = child is ai:MetadataFilter
            ? check metadataFilterToRetrievalFilter(child)
            : check metadataFiltersToRetrievalFilter(child);
        if childJson is json {
            children.push(childJson);
        }
    }
    if children.length() == 0 {
        return ();
    }
    // FLATTEN: a one-element group is not a group at all on the wire — `andAll`/
    // `orAll` reject fewer than 2 entries.
    if children.length() == 1 {
        return children[0];
    }
    string groupKey = filters.condition == ai:OR ? "orAll" : "andAll";
    return {[groupKey]: children};
}

// Combines an already-built `RetrievalFilter` (or none) with a leaf
// `_source_uri == id` filter — the probe `deleteByFilter` runs per candidate
// document. Always produces either a bare leaf (no user filter) or a 2-element
// `andAll` (user filter + the id leaf), so the `min: 2` rule is satisfied by
// construction; no flattening is needed here because the count is fixed at 1 or 2.
isolated function withSourceUriFilter(json? userFilter, string documentId) returns json {
    json idLeaf = {'equals: {key: SOURCE_URI_METADATA_KEY, value: documentId}};
    if userFilter is () {
        return idLeaf;
    }
    return {andAll: [userFilter, idLeaf]};
}
