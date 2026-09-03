# Ballerina AWS Bedrock AI Provider Library

[![Build](https://github.com/ballerina-platform/module-ballerinax-ai.aws.bedrock/actions/workflows/ci.yml/badge.svg)](https://github.com/ballerina-platform/module-ballerinax-ai.aws.bedrock/actions/workflows/ci.yml)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/ballerina-platform/module-ballerinax-ai.aws.bedrock.svg)](https://github.com/ballerina-platform/module-ballerinax-ai.aws.bedrock/commits/main)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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

For usage details, the routing table, and the full provider list, see the
[module documentation](ballerina/README.md).

## Issues and projects

The **Issues** and **Projects** tabs are disabled for this repository as it is part of the Ballerina
library. To report bugs, request new features, start new discussions, view project boards, etc., visit
the Ballerina library [parent repository](https://github.com/ballerina-platform/ballerina-library).

This repository only contains the source code for the module.

## Build from the source

### Prerequisites

1. Download and install Java SE Development Kit (JDK) version 21 (from one of the following locations).

   - [Oracle](https://www.oracle.com/java/technologies/downloads/)
   - [OpenJDK](https://adoptium.net/)

     > **Note:** Set the JAVA_HOME environment variable to the path name of the directory into which you
     > installed JDK.

2. Download and install [Ballerina Swan Lake](https://ballerina.io/).

3. Generate a GitHub access token with read package permissions, then set the following `env` variables:

   ```shell
   export packageUser=<Your GitHub Username>
   export packagePAT=<GitHub Personal Access Token>
   ```

### Build options

Execute the commands below to build from the source.

1. To build the package:

   ```bash
   ./gradlew clean build
   ```

2. To run the tests:

   ```bash
   ./gradlew clean test
   ```

3. To run a group of tests:

   ```bash
   ./gradlew clean test -Pgroups=<test_group_names>
   ```

4. To build without the tests:

   ```bash
   ./gradlew clean build -x test
   ```

5. To debug the package with a remote debugger:

   ```bash
   ./gradlew clean build -Pdebug=<port>
   ```

6. To debug with the Ballerina language:

   ```bash
   ./gradlew clean build -PbalJavaDebug=<port>
   ```

7. Publish the generated artifacts to the local Ballerina Central repository:

   ```bash
   ./gradlew clean build -PpublishToLocalCentral=true
   ```

8. Publish the generated artifacts to the Ballerina Central repository:

   ```bash
   ./gradlew clean build -PpublishToCentral=true
   ```

### Project layout

This is a Gradle multi-project build. `./gradlew build` runs the following, in order:

| Project | Directory | Produces |
| --- | --- | --- |
| `:ai.aws.bedrock-native` | `native/` | the `generate()` runtime shim jar |
| `:ai.aws.bedrock-compiler-plugin` | `compiler-plugin/` | the code-modifier jar (+ its `ballerina-to-openapi` dependency) |
| `:ai.aws.bedrock-ballerina` | `ballerina/` | the Ballerina package (`bal build` + `bal test`) |

Both Java projects must be built **before** `bal build`: `ballerina/Ballerina.toml` and
`ballerina/CompilerPlugin.toml` reference their jars by path. To iterate on the Ballerina sources alone
once the jars exist:

```bash
cd ballerina && bal build && bal test
```

**Why the compiler plugin is required.** `generate()` is declared `external` and returns an inferred
`typedesc<anydata>`. The plugin (`AiAwsBedrockCodeModifier`) walks every `generate()` call site whose
receiver is one of this package's seven provider classes, derives the JSON schema of the expected return
type, and attaches it to that type as an `@ai:JsonSchema` annotation. The runtime shim reads that
annotation to bind the model's response back into the caller's type. Without the plugin, records have no
derivable schema and `generate()` fails at runtime. Adding a new provider class means adding its name to
`MODEL_PROVIDER_CLASS_NAMES` in `GenerateMethodModificationTask` — a class missing from that list
silently loses type binding, with no compile error at the call site.

Note that `./gradlew build` invokes the `io.ballerina.plugin` Gradle plugin's `commitTomlFiles` task,
which runs `git commit` on `Ballerina.toml`, `Dependencies.toml`, and `CompilerPlugin.toml`. This is the
standard behaviour for `ballerina-library` connector repositories and is used by the release pipeline.

## Contribute to Ballerina

As an open-source project, Ballerina welcomes contributions from the community.

For more information, go to the [contribution guidelines](https://github.com/ballerina-platform/ballerina-lang/blob/master/CONTRIBUTING.md).

## Code of conduct

All the contributors are encouraged to read the [Ballerina Code of Conduct](https://ballerina.io/code-of-conduct).

## Useful links

- Chat live with us via our [Discord server](https://discord.gg/ballerinalang).
- Post all technical questions on Stack Overflow with the [#ballerina](https://stackoverflow.com/questions/tagged/ballerina) tag.
- For more information go to the [`ai.aws.bedrock` module](ballerina/README.md).
- For example demonstrations of the usage, go to [Ballerina By Examples](https://ballerina.io/learn/by-example/).
