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
import io.ballerina.runtime.api.values.BObject;
import io.ballerina.runtime.api.values.BTypedesc;

/**
 * Native shim for {@code ModelProvider.generateStream()}. A dependently typed
 * function must be external in Ballerina, so this delegates straight back to the
 * Ballerina {@code generateLlmResponseStream}.
 *
 * <p>Unlike its sibling {@link Generator}, this passes the provider OBJECT rather
 * than unpacking its resolved fields. {@code generateLlmResponseStream} obtains its
 * chunks by calling {@code chatStream} as a remote method on the provider, so it
 * needs the object itself; there is no field list to forward. That also makes this
 * shim immune to the field-renaming hazard documented in {@code Generator}.
 *
 * @since 0.9.0
 */
public final class StreamGenerator {

    private StreamGenerator() {
    }

    public static Object generateStream(Environment env, BObject modelProvider,
                                        BObject prompt, BTypedesc expectedResponseTypedesc) {
        // Module version is the MAJOR component only, and this package is 0.9.0 —
        // hence "0", matching Generator. A "1" here resolves no module and fails at
        // runtime, not compile time.
        return env.getRuntime().callFunction(
                new Module("ballerinax", "ai.aws.bedrock", "0"), "generateLlmResponseStream", null,
                modelProvider, prompt, expectedResponseTypedesc);
    }
}
