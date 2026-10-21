# Architecture Decision Records (ADRs)

Record decisions that materially affect architecture, dependencies, data handling, deployment, security, or project scope. Keep decisions short, evidence-based, and linked to relevant issues.

## Naming

Use sequential filenames such as `0001-select-application-stack.md`.

## Template

```markdown
# ADR-NNNN: Decision title

- Status: Proposed | Accepted | Superseded | Rejected
- Date: YYYY-MM-DD
- Owners: names or roles
- Related issues: #NNN

## Context
What problem, requirement, or constraint requires a decision?

## Decision drivers
List constraints and evaluation criteria.

## Options considered
Describe realistic alternatives and their trade-offs.

## Decision
State the chosen option and why evidence supports it.

## Consequences
Document benefits, costs, risks, migration implications, and limitations.

## Verification
How will the team verify that the decision works?

## Revisit conditions
What evidence or change would justify reconsidering it?
```

Do not create an ADR for every trivial code change. Do create one when reversing a decision later would be costly or when the reasoning needs to be auditable.
