# Development Guide

## Status

The language, runtime, package manager, and build system have not yet been selected. The commands below are therefore requirements for the eventual guide, not executable commands for the current repository.

## 1. Supported environment

Document and pin:
- Operating systems and minimum versions.
- Language/runtime and exact supported version.
- Package manager and lockfile.
- Required system packages and native libraries.
- External accounts, API keys, and service configuration.
- Optional GPU/accelerator requirements.
- Disk, memory, and network requirements.
- Known platform-specific limitations.

## 2. Clean setup procedure

After the stack is selected, provide a tested, copy-pasteable sequence that:
1. Checks prerequisites and prints actionable errors.
2. Clones the repository and selects the supported revision.
3. Creates an isolated environment where appropriate.
4. Installs dependencies from the lockfile.
5. Validates required configuration without exposing secret values.
6. Acquires or prepares the smallest permitted sample dataset.
7. Runs format/lint/type checks and unit tests.
8. Starts the application or pipeline.
9. Runs a documented smoke test.
10. Cleans generated artifacts safely.

The sequence must be tested on a clean environment. Do not invent commands before knowing the stack.

## 3. Configuration

- Provide `.env.example` only if environment variables are used.
- Explain each variable, whether it is required, its safe default, and how to obtain its value.
- Fail early with a clear message when required configuration is missing.
- Never log tokens, passwords, or full sensitive payloads.
- Keep development, test, and deployment configuration separate.

## 4. Daily workflow

1. Select a scoped issue and confirm acceptance criteria.
2. Inspect relevant code, docs, and data contracts.
3. Implement the smallest coherent change.
4. Add or update tests for expected and failure behavior.
5. Run formatting, linting, type checks, tests, and smoke checks.
6. Update documentation, data provenance, and diagrams when affected.
7. Provide evidence in the issue or pull request.
8. Review for secrets, unintended data, dependency changes, and generated files.

## 5. Dependencies

Every dependency should have a clear purpose. Pin or lock dependencies, check license compatibility, remove unused packages, and document external services. Record why a heavy framework or model is needed rather than adopting it by default.

## 6. Troubleshooting

Maintain a table with symptom, likely cause, diagnostic command, resolution, and environment. Include setup failures, network/API limits, malformed data, missing configuration, startup failures, and demo-day fallback procedures once they are known.
