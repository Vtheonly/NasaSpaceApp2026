# Contributing

## Workflow

1. Read the README and relevant documents before changing implementation.
2. Work from an issue with a clear problem statement, owner, scope, and acceptance criteria.
3. Keep each change focused and avoid unrelated refactors.
4. Add tests for changed behavior and relevant failure cases.
5. Run the documented checks and report their actual results.
6. Update documentation, diagrams, source attribution, and examples when affected.
7. Review the final diff for secrets, generated artifacts, licenses, and unintended scope.
8. Link the issue and provide reproducible verification evidence.

## Definition of done

- Acceptance criteria are met.
- Automated checks relevant to the change pass, or failures are clearly documented.
- Important edge cases are covered.
- Documentation matches the behavior.
- External data and assets have provenance and appropriate terms.
- No secrets or restricted data are introduced.
- Remaining limitations are stated.

## Issue quality

Issues should describe the problem or outcome, why it matters, concrete deliverables, acceptance criteria, dependencies, and verification. Split large work into deliverable-sized issues with explicit dependencies. Avoid vague tasks such as “finish everything.”

## Coding conventions

The language and formatter are not selected yet. Once chosen, pin the relevant toolchain and document naming, formatting, linting, type checking, error handling, and test conventions here. Prefer clear names, small cohesive functions, explicit contracts, and tested domain logic.

## Documentation conventions

- Use Markdown headings in hierarchical order.
- Link related files and issues.
- Label unknowns and assumptions rather than presenting them as facts.
- Include runnable commands only after verifying them against the chosen stack.
- Keep the repository map and architecture diagrams synchronized with actual files.
