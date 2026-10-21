# Challenge and Requirements

## Status

The actual challenge selection and official challenge text are not yet recorded. This document is a working template; it must be completed using official NASA Space Apps 2026 materials and team decisions.

## 1. Official challenge record

- Challenge title: TBD
- Challenge URL: TBD
- Official prompt captured on: TBD
- Judging criteria / evaluation rubric: TBD
- Relevant event rules and submission requirements: TBD
- Submission deadline and time zone: TBD
- Required deliverables: TBD
- Source links and archived copies: TBD

Quote or summarize only what the source supports. Record retrieval dates for time-sensitive event requirements.

## 2. Problem framing

Complete after reading the official prompt:

- Who experiences the problem?
- What decision, task, or understanding is difficult today?
- What evidence demonstrates that the problem matters?
- What is the current alternative or baseline?
- What outcome should improve?
- What is explicitly outside the challenge's scope?
- What would make the solution meaningfully different from existing approaches?

Avoid solution-first reasoning. A framework or model is not itself a problem statement.

## 3. User and use cases

For each primary user/persona, document context, goals, constraints, digital access, relevant accessibility needs, and what success looks like. Validate personas with evidence; label hypothetical personas as assumptions.

Use-case template:

| Field | Description |
|---|---|
| ID | UC-01, UC-02, ... |
| Actor | User or system |
| Trigger | What starts the workflow |
| Preconditions | Required data and system state |
| Main flow | Observable steps from start to result |
| Alternate/error flows | Missing, invalid, stale, or unavailable inputs |
| Output | What the user receives |
| Acceptance test | How a reviewer verifies success |

## 4. Requirements catalogue

Give each requirement a stable ID.

- **FR-xxx:** functional requirements.
- **NFR-xxx:** quality requirements such as performance, accessibility, reliability, and usability.
- **DATA-xxx:** data acquisition, transformation, provenance, and quality.
- **SEC-xxx:** security and privacy.
- **SUB-xxx:** event and submission requirements.
- **DOC-xxx:** documentation and reproducibility.

Each requirement must include: statement, rationale/source, priority, owner, dependencies, acceptance criteria, verification method, and status.

### Priority definitions

- **P0 — Essential:** solution cannot satisfy the challenge or be evaluated without it.
- **P1 — Important:** materially improves the core solution or rubric coverage.
- **P2 — Optional:** only attempt after P0 requirements are working and tested.

## 5. Traceability matrix

Create and maintain a table with the following columns:

| Requirement ID | Official criterion / evidence | User need | Design component | Implementation issue | Test ID | Demo evidence | Status |
|---|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | Not started |

Every judging criterion should map to evidence. Every major feature should map to a user need or explicit criterion.

## 6. Acceptance criteria

Write acceptance criteria as observable outcomes, not implementation wishes. Prefer Given/When/Then or clear input/output conditions. Include at least one negative or failure-path case for important workflows.

## 7. Scope control

Before adding a feature, answer:
1. Which user need or requirement does it satisfy?
2. Which criterion or project objective does it support?
3. How will it be verified?
4. What is the smallest useful implementation?
5. What cost, risk, or complexity does it add?

If there is no defensible answer, defer the feature.
