# Architecture Decision Records

These records document the selected direction for **Harmonization of MODIS and VIIRS Hot Spots**. Accepted means the project direction is chosen; it does not mean performance is proven.

| ADR | Decision | Status |
|---|---|---|
| ADR-001 | Spatio-temporal harmonization and sequence modeling rather than primary image super-resolution | Selected; implementation/evidence required |
| ADR-002 | Tweedie gradient-boosted regression for cross-sensor FRP harmonization | Primary candidate; compare with baselines |
| ADR-003 | PatchTST candidate for calendar/anomaly/forecast modeling | Selected candidate; evaluate against seasonal baselines |
| ADR-004 | Prithvi-EO 2.0 only for optional HLS optical burn-scar verification | Selected boundary; verify inputs/checkpoint |
| ADR-005 | Use processed FIRMS georeferenced records before considering raw swath reconstruction | Selected approach; verify source coverage |
| ADR-006 | Forecast probabilistic regional activity windows, not exact ignition points | Scientific boundary |
| ADR-007 | Separate GitHub source, offline wheels, model weights, and processed data | Selected artifact strategy |
| ADR-008 | Require internet-disabled quarantine before expensive compute | Operational gate |
| ADR-009 | Five-pass data validation before training | Quality gate to implement |
| ADR-010 | Run the user-facing app outside the isolated Kaggle compute environment | Selected deployment boundary |

## ADR template

Use one file per significant decision, e.g. `ADR-002-tweedie-harmonization.md`.

```markdown
# ADR-NNN: Short title
- Status: Proposed | Accepted | Superseded
- Date:
- Owners:

## Context
Constraints, problem, and evidence.
## Decision
What was chosen and explicitly excluded.
## Alternatives
Realistic candidates.
## Rationale
Why the choice fits the challenge and constraints.
## Consequences
Benefits, costs, risks, and required work.
## Validation plan
Tests, metrics, and evidence needed.
## Revisit when
Specific conditions that would justify changing the decision.
```

Record results and supersede a decision when evidence warrants it. Keep details in the relevant architecture/model documents and link issues rather than duplicating entire specifications.