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

import io.ballerina.runtime.api.Environment;
import io.ballerina.runtime.api.Module;
import io.ballerina.runtime.api.utils.StringUtils;
import io.ballerina.runtime.api.values.BObject;
import io.ballerina.runtime.api.values.BTypedesc;

/**
 * Native shim for {@code ModelProvider.generate()} (design §3, §8). A dependently
 * typed function must be external in Ballerina, so this delegates straight back to
 * the Ballerina {@code generateLlmResponse}, whose {@code anydata} result the
 * runtime coerces to the caller's expected type.
 *
 * <p>Schema derivation lives in {@code to_json_schema.bal}, as in the reference
 * provider modules — not here.
 *
 * @since 0.1.0
 */
public final class Generator {

    private Generator() {
    }

    public static Object generate(Environment env, BObject modelProvider,
                                  BObject prompt, BTypedesc expectedResponseTypedesc) {
        return env.getRuntime().callFunction(
                new Module("ballerinax", "ai.aws.bedrock", "0"), "generateLlmResponse", null,
                modelProvider.get(StringUtils.fromString("supportsStructuredOutput")),
                modelProvider.get(StringUtils.fromString("genFamily")),
                // NOTE: these are looked up by NAME at runtime, so renaming a provider
                // field here is a silent break — the compiler cannot see it. Any change
                // to a `private final` field on the seven provider classes must be
                // mirrored below.
                modelProvider.get(StringUtils.fromString("genConverter")),
                modelProvider.get(StringUtils.fromString("genTransport")),
                modelProvider.get(StringUtils.fromString("genModelId")),
                modelProvider.get(StringUtils.fromString("genHeaders")),
                modelProvider.get(StringUtils.fromString("params")),
                prompt, expectedResponseTypedesc);
    }
}
