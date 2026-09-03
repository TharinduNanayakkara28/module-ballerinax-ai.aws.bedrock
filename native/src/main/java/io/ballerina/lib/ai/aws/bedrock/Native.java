/*
 * Copyright (c) 2026, WSO2 LLC. (https://www.wso2.com).
 *
 * WSO2 LLC. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package io.ballerina.lib.ai.aws.bedrock;

import io.ballerina.runtime.api.Module;
import io.ballerina.runtime.api.creators.ErrorCreator;
import io.ballerina.runtime.api.creators.TypeCreator;
import io.ballerina.runtime.api.creators.ValueCreator;
import io.ballerina.runtime.api.types.AnnotatableType;
import io.ballerina.runtime.api.types.ArrayType;
import io.ballerina.runtime.api.types.JsonType;
import io.ballerina.runtime.api.types.PredefinedTypes;
import io.ballerina.runtime.api.types.ReferenceType;
import io.ballerina.runtime.api.types.Type;
import io.ballerina.runtime.api.types.TypeTags;
import io.ballerina.runtime.api.types.UnionType;
import io.ballerina.runtime.api.utils.StringUtils;
import io.ballerina.runtime.api.utils.TypeUtils;
import io.ballerina.runtime.api.values.BArray;
import io.ballerina.runtime.api.values.BError;
import io.ballerina.runtime.api.values.BMap;
import io.ballerina.runtime.api.values.BString;
import io.ballerina.runtime.api.values.BTypedesc;

import java.util.List;

import static io.ballerina.runtime.api.creators.ValueCreator.createMapValue;

/**
 * Runtime JSON-schema generation for {@code generate()}'s expected response type.
 *
 * <p>Ported from the reference provider modules (module-ballerinax-ai.openai's
 * {@code Native.java}) so this module owns its schema generation, as its siblings
 * do. It previously reflected into {@code ballerina/ai}'s internal
 * {@code io.ballerina.stdlib.ai.wso2.Native}: an undocumented class of another
 * module, reached by name, whose disappearance would have silently degraded
 * {@code generate()} to an empty schema rather than failing.
 *
 * <p>Record schemas are attached at the call site as a {@code ballerina/ai}
 * {@code JsonSchema} annotation by ballerina/ai's compiler plugin (verified to fire
 * for call sites typed with the concrete provider class, not only the
 * {@code ai:ModelProvider} interface). This class reads that annotation and derives
 * schemas structurally for arrays, unions and simple types.
 *
 * <p>This step is AUTHORITATIVE: it either returns a schema or raises an
 * {@code ai:Error}, and never defers to a Ballerina-side fallback. It covers a strict
 * superset of what a pure-Ballerina implementation can express (recursive arrays and
 * unions, not just simple types and simple-member arrays), so a "generated at compile
 * time" flag gating a fallback would never fire — the fallback and its flag were
 * removed rather than left as unreachable code that reads like a safety net.
 *
 * @since 0.1.0
 */
public final class Native {
    public static final String ANY_OF = "anyOf";
    public static final String BALLERINA_AI = "ballerina/ai";
    public static final String JSON_SCHEMA = "JsonSchema";

    private Native() {
    }

    public static Object generateJsonSchemaForTypedescNative(BTypedesc td) {
        try {
            return generateJsonSchemaForType(td.getDescribingType());
        } catch (BError e) {
            return createAIError(e.getErrorMessage());
        }
    }

    private static Object generateJsonSchemaForType(Type t) throws BError {
        Type impliedType = TypeUtils.getImpliedType(t);
        if (isSimpleType(impliedType)) {
            return createSimpleTypeSchema(impliedType);
        }

        return switch (impliedType) {
            case JsonType ignored -> generateJsonSchemaForJson();
            case ArrayType arrayType -> generateJsonSchemaForArrayType(arrayType);
            case UnionType unionType -> generateUnionTypeSchema(unionType);
            case ReferenceType referenceType -> getJsonSchemaFromAnnotatableType(referenceType);
            default -> throw ErrorCreator.createError(StringUtils.fromString(
                    "Runtime schema generation is not yet supported for type " + impliedType.getName()));
        };
    }

    private static BError createAIError(BString message) {
        return ErrorCreator.createError(new Module("ballerina", "ai", "1"),
                "Error", message, null, null);
    }

    private static BMap<BString, Object> createSimpleTypeSchema(Type type) {
        BMap<BString, Object> schemaMap = createMapValue(TypeCreator.createMapType(PredefinedTypes.TYPE_JSON));
        schemaMap.put(StringUtils.fromString("type"), StringUtils.fromString(getStringRepresentation(type)));
        return schemaMap;
    }

    private static Object generateUnionTypeSchema(UnionType unionType) {
        BMap<BString, Object> schemaMap = createMapValue(TypeCreator.createMapType(PredefinedTypes.TYPE_JSON));
        List<Type> memberTypes = unionType.getMemberTypes();
        BArray schemas = ValueCreator.createArrayValue(
                TypeCreator.createArrayType(PredefinedTypes.TYPE_JSON));
        for (Type memberType : memberTypes) {
            Object schema = generateJsonSchemaForType(memberType);
            schemas.append(schema);
        }
        if (schemas.size() == 1) {
            return schemas.get(0);
        }
        schemaMap.put(StringUtils.fromString(ANY_OF), schemas);
        return schemaMap;
    }

    private static Object getJsonSchemaFromAnnotatableType(ReferenceType referenceType) {
        Type referredType = referenceType.getReferredType();
        if (referredType instanceof AnnotatableType annotatableType) {
            BMap<BString, Object> annotations = annotatableType.getAnnotations();
            for (BString key : annotations.getKeys()) {
                if (key.getValue().startsWith(BALLERINA_AI) && key.getValue().endsWith(JSON_SCHEMA)) {
                    Object schema = annotations.get(key);
                    if (schema instanceof BMap) {
                        return schema;
                    }
                }
            }
        }
        throw ErrorCreator.createError(StringUtils.fromString(
                "Runtime schema generation is not yet supported for type: " + referenceType.getName()));
    }

    private static BMap<BString, Object> generateJsonSchemaForJson() {
        BString[] bStringValues = new BString[6];
        bStringValues[0] = StringUtils.fromString("object");
        bStringValues[1] = StringUtils.fromString("array");
        bStringValues[2] = StringUtils.fromString("string");
        bStringValues[3] = StringUtils.fromString("number");
        bStringValues[4] = StringUtils.fromString("boolean");
        bStringValues[5] = StringUtils.fromString("null");
        BMap<BString, Object> schemaMap = createMapValue(TypeCreator.createMapType(PredefinedTypes.TYPE_JSON));
        schemaMap.put(StringUtils.fromString("type"), ValueCreator.createArrayValue(bStringValues));
        return schemaMap;
    }

    private static boolean isSimpleType(Type type) {
        int mask = type.getBasicType().all();
        // A single basic-type bit (or the nil mask) is simple. A COMBINED mask such
        // as `int?` or `boolean|int` is a union and must fall through to
        // generateUnionTypeSchema: createSimpleTypeSchema calls
        // getStringRepresentation, which has no case for a combined mask and returns
        // null — emitting the invalid schema {"type": null}. Observed on `int?[]`,
        // whose member type reaches here as int|().
        return mask == 0 || ((mask & (mask - 1)) == 0 && mask <= 0b100000);
    }

    private static String getStringRepresentation(Type type) {
        if (type.getTag() == TypeTags.NULL_TAG) {
            return "null";
        }
        return switch (type.getBasicType().all()) {
            case 0b000000 -> "null";
            case 0b000010 -> "boolean";
            case 0b000100 -> "integer";
            case 0b001000, 0b010000 -> "number";
            case 0b100000 -> "string";
            default -> null;
        };
    }

    private static Object generateJsonSchemaForArrayType(ArrayType arrayType) {
        BMap<BString, Object> schemaMap = createMapValue(TypeCreator.createMapType(PredefinedTypes.TYPE_JSON));
        Type elementType = TypeUtils.getImpliedType(arrayType.getElementType());
        schemaMap.put(StringUtils.fromString("type"), StringUtils.fromString("array"));
        schemaMap.put(StringUtils.fromString("items"), generateJsonSchemaForType(elementType));
        return schemaMap;
    }
}
