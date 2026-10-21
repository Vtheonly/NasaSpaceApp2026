# Ownership, review, and evidence-recording practices (issue #3)

## Ownership model

The repository currently has a single accountable owner: **@Vtheonly**. Every
component below has that owner recorded in the enforced
`.github/CODEOWNERS` file, so GitHub automatically requests their review on
any pull request touching the component. Where work is deliberately not
owned - the optional Prithvi/HLS burn-scar audit - it is marked
**UNASSIGNED (optional, non-blocking)** rather than silently missing; anyone
proposing that work must propose its owner in the same PR. This satisfies the
issue #3 acceptance criterion "every P0 deliverable has an owner or is
explicitly unassigned" with a mechanism (CODEOWNERS), not a promise.

## Component ownership matrix

| Component (paths) | Owner | Review requirement | Recorded evidence |
|---|---|---|---|
| FIRMS ingestion + validation (`src/data/`, `scripts/validate_data.py`, `scripts/inventory_firms.py`) | @Vtheonly | Gate reports must be committed/persisted for every run; fail-closed behavior preserved by tests | `docs/evidence/firms_inventory/`, gate JSON artifacts, `tests/unit/test_gate.py`, `tests/integration/test_inventory_firms.py` |
| Source registry + licensing (`docs/RESEARCH_AND_DATA.md`) | @Vtheonly | Any schema/coverage claim must cite the official page + access date; regenerated via the inventory tool | `docs/RESEARCH_AND_DATA.md`, `docs/evidence/firms_inventory/` |
| Harmonization (`src/harmonization/`, `src/geo/`, `scripts/train_harmonizer.py`) | @Vtheonly | Chronological hold-out evaluation; baselines must not regress | `harmonizer_*/manifest.json`, `tests/scientific/` |
| Calendar + forecast (`src/calendar/`, `src/models/`, `scripts/train_forecaster.py`) | @Vtheonly | Rolling-origin backtests; seasonal-naive fallback stays available without torch | `forecaster_*/manifest.json`, `tests/unit/test_forecast.py` |
| End-to-end pipeline + demo (`scripts/run_pipeline.py`, `scripts/run_demo.py`) | @Vtheonly | `pipeline_run.json` stage ledger green; demo report links claims to artifacts | `pipeline_run.json`, `DEMO_REPORT.md` |
| Viewer app (`app/`, `scripts/build_app.py`, `docs/UI_VALIDATION.md`) | @Vtheonly | Usability/a11y contract tests green; explicit states for missing data/forecast | `app_build.json`, `tests/integration/test_app.py` |
| Offline notebooks (`notebooks/`) | @Vtheonly | Each notebook runs without network; staging/quarantine order preserved | `notebooks/01-06` |
| Governance + docs + CI (`.github/`, `docs/`) | @Vtheonly | Docs change in the same PR as the behavior they describe; CI green | `.github/workflows/ci.yml`, this file |
| Optional Prithvi/HLS burn-scar audit | **UNASSIGNED (optional)** | Cannot block any P0/P1 deliverable; owner must be proposed with the first PR | `docs/POST_HACKATHON_BACKLOG.md` |

## P0 ledger (status as of 2026-10-22)

Every P0 issue is closed except this ownership record itself, which this
document plus `.github/CODEOWNERS` completes:

| P0 issue | Deliverable | Status | Where the evidence lives |
|---|---|---|---|
| #1 | Challenge validation + locked requirements | closed | `docs/CHALLENGE_AND_REQUIREMENTS.md`, PR #26 verification |
| #2 | AOI workflow + measurable acceptance criteria | closed | `docs/CHALLENGE_AND_REQUIREMENTS.md`, pipeline stage contract |
| #3 | Ownership + review/evidence practices | **this PR** | `.github/CODEOWNERS`, this file |
| #4 | Requirement-to-module traceability | closed | `docs/CHALLENGE_AND_REQUIREMENTS.md` traceability section (PR #26) |
| #5 | Three-pillar solution without scope drift | closed | offline pipeline + demo + notebooks (merged PRs #21-#25) |
| #6 | FIRMS source inventory + validation | closed (PR #27) | `docs/RESEARCH_AND_DATA.md`, `docs/evidence/firms_inventory/` |
| #7 | Modular architecture + artifact contracts | closed (PR #21) | `docs/ARCHITECTURE.md`, `src/artifacts/` |
| #8 | End-to-end AOI -> calendar workflow | closed (PR #25) | `scripts/run_pipeline.py`, `tests/integration/test_run_pipeline.py` |
| #9 | Reproducible setup + six offline notebooks | closed (PR #25) | `notebooks/01-06` |
| #10 | Minimum viable harmonization + calendar | closed (PR #21) | ~360-test suite, demo artifacts |
| #11 | Five-pass data gate + scientific validation | closed (PR #21) | `src/data/gate.py`, `tests/scientific/` |
| #15 | Model contracts + assumptions documentation | closed | `docs/MODEL_CONTRACTS.md` |
| #16 | Rehearsed scientific demo | closed (PR #25) | `docs/DEMO_GUIDE.md`, `DEMO_REPORT.md` |
| #17 | Official submission requirements verified | closed (PR #26) | `docs/SUBMISSION_CHECKLIST.md`, `docs/CHALLENGE_AND_REQUIREMENTS.md` |

Blockers: none. No P0 deliverable is unowned; the only unassigned item is the
optional post-hackathon Prithvi audit, which is explicitly out of the
critical path.

## Review, documentation, and evidence-recording practices

These are the working practices this repository already follows; recording
them makes them the standard rather than a habit:

1. **Feature-branch -> pull request -> merge into main.** No direct commits to
   main; every change lands through a PR that references the issue it closes
   ("Closes #N") so the issue ledger stays truthful.
2. **CODEOWNERS-enforced review.** GitHub requests the component owner's
   review automatically via `.github/CODEOWNERS`; the matrix above defines
   what that review must check per component (hold-out evaluation, backtest
   evidence, explicit-state tests, etc.).
3. **Evidence over adjectives.** A claim is done only when a committed,
   machine-generated artifact backs it: `docs/evidence/firms_inventory/` for
   source facts, `pipeline_run.json` for stage outcomes, `manifest.json` +
   SHA-256 for models, `app_build.json` (with contrast ratios) for the UI,
   `DEMO_REPORT.md` for demo claims. Synthetic-fixture results are always
   labeled as such.
4. **Docs move with code.** The documentation describing a behavior changes
   in the same PR as the behavior (enforced in review; see CONTRIBUTING.md).
   Schemas/coverage claims additionally require the official-page access date.
5. **CI is the floor.** The full pytest suite plus quality checks run on
   every PR (`ci.yml`); a red CI blocks merge regardless of ownership.
6. **No fabricated data, ever.** Missing coverage, missing forecasts, and
   empty AOIs render as explicit states (MODEL_CONTRACTS.md section 8), and
   tests enforce those states - reviewers reject any change that silently
   zero-fills or extrapolates.
