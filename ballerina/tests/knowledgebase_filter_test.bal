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
import ballerina/test;

// Pure table tests for `ai:MetadataFilters` -> Bedrock `RetrievalFilter`. No I/O.

@test:Config {}
function testEveryOperatorMapsToItsOwnRetrievalFilterField() returns error? {
    map<ai:MetadataFilterOperator> operatorToWireField = {
        "equals": ai:EQUAL,
        "notEquals": ai:NOT_EQUAL,
        "greaterThan": ai:GREATER_THAN,
        "greaterThanOrEquals": ai:GREATER_THAN_OR_EQUAL,
        "lessThan": ai:LESS_THAN,
        "lessThanOrEquals": ai:LESS_THAN_OR_EQUAL
    };
    foreach [string, ai:MetadataFilterOperator] [wireField, operator] in operatorToWireField.entries() {
        ai:MetadataFilters filters = {filters: [{key: "tenant", operator, value: "acme"}]};
        json|ai:Error result = metadataFiltersToRetrievalFilter(filters);
        test:assertTrue(result is json, string `operator ${operator} failed: ${result is ai:Error ? result.message() : ""}`);
        if result is map<json> {
            test:assertTrue(result.hasKey(wireField), string `expected key '${wireField}' in ${result.toJsonString()}`);
            json attribute = result[wireField];
            test:assertEquals(attribute, {key: "tenant", value: "acme"});
        }
    }
}

@test:Config {}
function testInAndNotInRequireAnArrayValue() returns error? {
    ai:MetadataFilters inFilters = {filters: [{key: "genre", operator: ai:IN, value: ["cooking", "sports"]}]};
    json|ai:Error inResult = metadataFiltersToRetrievalFilter(inFilters);
    test:assertTrue(inResult is map<json> && (<map<json>>inResult).hasKey("in"));

    ai:MetadataFilters notInFilters = {filters: [{key: "genre", operator: ai:NOT_IN, value: ["a", "b"]}]};
    json|ai:Error notInResult = metadataFiltersToRetrievalFilter(notInFilters);
    test:assertTrue(notInResult is map<json> && (<map<json>>notInResult).hasKey("notIn"));

    // A non-array value on IN/NOT_IN is a clear error, not a malformed request sent
    // to Bedrock.
    ai:MetadataFilters badFilter = {filters: [{key: "genre", operator: ai:IN, value: "cooking"}]};
    json|ai:Error badResult = metadataFiltersToRetrievalFilter(badFilter);
    test:assertTrue(badResult is ai:Error);
}

@test:Config {}
function testNestedAndOrGroupsProduceAndAllOrAll() returns error? {
    ai:MetadataFilters filters = {
        condition: ai:AND,
        filters: [
            {key: "tenant", operator: ai:EQUAL, value: "acme"},
            {
                condition: ai:OR,
                filters: [
                    {key: "status", operator: ai:EQUAL, value: "draft"},
                    {key: "status", operator: ai:EQUAL, value: "review"}
                ]
            }
        ]
    };
    json? result = check metadataFiltersToRetrievalFilter(filters);
    test:assertTrue(result is map<json>);
    if result is map<json> {
        test:assertTrue(result.hasKey("andAll"));
        json[] andAll = <json[]>result["andAll"];
        test:assertEquals(andAll.length(), 2);
        map<json> orGroup = <map<json>>andAll[1];
        test:assertTrue(orGroup.hasKey("orAll"));
        test:assertEquals((<json[]>orGroup["orAll"]).length(), 2);
    }
}

@test:Config {}
function testSingleChildGroupFlattensRatherThanWrapping() returns error? {
    // RetrievalFilterList has min:2 — a group that reduces to one child must not be
    // sent as a one-element andAll/orAll.
    ai:MetadataFilters filters = {filters: [{key: "tenant", operator: ai:EQUAL, value: "acme"}]};
    json? result = check metadataFiltersToRetrievalFilter(filters);
    test:assertTrue(result is map<json>);
    if result is map<json> {
        test:assertFalse(result.hasKey("andAll"), "single-child group must flatten, not wrap in andAll");
        test:assertTrue(result.hasKey("equals"));
    }
}

@test:Config {}
function testNestedSingleChildGroupAlsoFlattens() returns error? {
    ai:MetadataFilters filters = {
        filters: [
            {key: "tenant", operator: ai:EQUAL, value: "acme"},
            {
                // A nested group with exactly one child.
                filters: [{key: "status", operator: ai:EQUAL, value: "draft"}]
            }
        ]
    };
    json? result = check metadataFiltersToRetrievalFilter(filters);
    test:assertTrue(result is map<json>);
    if result is map<json> {
        json[] andAll = <json[]>result["andAll"];
        test:assertEquals(andAll.length(), 2);
        // The second child flattened to a bare 'equals' leaf, not a one-element andAll.
        map<json> secondChild = <map<json>>andAll[1];
        test:assertTrue(secondChild.hasKey("equals"), secondChild.toJsonString());
    }
}

@test:Config {}
function testEmptyFilterGroupProducesNil() returns error? {
    ai:MetadataFilters filters = {filters: []};
    json?|ai:Error result = metadataFiltersToRetrievalFilter(filters);
    test:assertTrue(result is ());
}

@test:Config {}
function testWithSourceUriFilterWithNoUserFilterIsABareLeaf() {
    json result = withSourceUriFilter((), "doc-123");
    test:assertEquals(result, {'equals: {key: "_source_uri", value: "doc-123"}});
}

@test:Config {}
function testWithSourceUriFilterWithAUserFilterIsATwoElementAndAll() {
    json userFilter = {'equals: {key: "tenant", value: "acme"}};
    json result = withSourceUriFilter(userFilter, "doc-123");
    test:assertTrue(result is map<json>);
    map<json> resultMap = <map<json>>result;
    test:assertTrue(resultMap.hasKey("andAll"));
    json[] andAll = <json[]>resultMap["andAll"];
    // Exactly 2 — satisfies RetrievalFilterList's min:2 by construction, no
    // flattening needed here.
    test:assertEquals(andAll.length(), 2);
    test:assertEquals(andAll[0], userFilter);
    test:assertEquals(andAll[1], {'equals: {key: "_source_uri", value: "doc-123"}});
}
