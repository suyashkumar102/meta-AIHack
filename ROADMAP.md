# Hackstreet Boys Roadmap

## Team

- Team name: Hackstreet Boys
- Members:
  - Roopal Guha Neogi
  - Suyash Kumar
- Submission deadline: April 8, 2026, 11:59 PM IST
- Current planning checkpoint: April 3, 2026

## What We Are Optimizing For

These are the main wins for the final stretch, in order:

1. **RL improvement**
2. **Robustness**
3. **Real-world grounding**
4. **Submission safety**

In practice, that means:

- improve the reward and episode behavior only where changes are low-risk and test-backed
- add strong automated validation so the repo feels reliable, not hand-wavy
- ground our taxonomy and partial-credit choices against real external IT support data without trying to absorb that data into the runtime dataset this late
- avoid broad refactors that create new failure modes near submission

## Honest Scope Call

What is viable before the deadline:

- unit tests
- smoke tests
- focused integration tests
- deterministic regression checks
- lightweight RL-oriented scoring improvements
- grounding audits against public real-world support datasets

What is **not** viable before the deadline:

- replacing `data/dataset.json` with an external dataset
- redesigning the taxonomy from scratch
- large architecture rewrites
- open-ended benchmark expansion without validation

## Guardrails

To stay on track:

1. do not merge external datasets into the main runtime dataset before submission
2. do not broaden the action schema or rename fields
3. do not make reward changes unless tests prove exact, zero, and partial-credit cases clearly
4. every Codex-generated code change must end with tests or validation evidence
5. prefer small, bounded implementation passes over one large all-at-once rewrite

## Working Model With Codex

Using Codex to generate all implementation work is viable **if we keep each ask narrow and verifiable**.

Best pattern:

1. ask for one bounded change set
2. add or update tests in the same pass
3. run the relevant checks
4. only then move to the next improvement

Bad pattern:

- ask for tests, scoring changes, dataset expansion, CI, and docs all in one prompt

## Last-Mile Phase Plan

### Phase 1: Test Foundation
**Window:** April 3 to April 4

**Primary objective:** make the current env provably correct before we tune anything

Deliverables:

- add `pytest`-based test structure
- add unit tests for:
  - `server/grader.py`
  - `server/reward.py`
  - `server/tasks.py`
  - `models.py` where validation matters
- add smoke tests for:
  - environment `reset()`
  - environment `step()`
  - deterministic seeded behavior
  - score range `[0.0, 1.0]`
- add focused integration tests for:
  - FastAPI endpoints such as `/health`, `/tasks`, `/reset`, `/step`, `/state`
  - one full seeded episode through the app surface

Most important assertions in this phase:

- exact matches score `1.0`
- unrelated wrong labels score `0.0`
- only approved near-miss pairs receive partial credit
- assignment group and resolution action remain exact-match fields
- the environment is deterministic when seeded
- the baseline path still completes all tasks

Exit criteria:

- tests clearly prove the scorer is **not** "always fuzzy"
- core environment behavior is covered by automated checks
- we can change scoring logic later without guessing whether we broke it

### Phase 2: RL Improvement Without Big Risk
**Window:** April 4 to April 5

**Primary objective:** make the reward surface better for RL while preserving determinism and judge clarity

Allowed improvements in this phase:

- refine `ISSUE_TYPE_SIMILARITY` only where justified and test-backed
- tighten priority partial-credit coverage if tests show obvious gaps
- improve episode history if it helps multi-step learning and does not complicate grading
- add deterministic regression checks around expected baseline behavior
- optionally add a safe `queue_size` override in `reset()` only if it is clean and fully tested

Non-goals for this phase:

- no new fields in the public schema
- no major reward-architecture refactor
- no broad rubric redesign

Decision rule:

- if a proposed RL improvement makes scoring harder to explain, skip it
- if it improves learning signal and is easy to test, keep it

Exit criteria:

- reward logic is still simple to explain
- exactness is preserved where it should be exact
- any extra partial credit is intentional, narrow, and documented by tests

### Phase 3: Real-World Grounding Audit
**Window:** April 5 to April 6

**Primary objective:** show that our labels and ambiguity rules are grounded in real support data, without late-stage dataset merge risk

Grounding approach:

- audit our taxonomy against public real-world support datasets
- use those datasets as reference material, not as direct training/runtime data
- document what they validate about our domain, labels, and near-miss structure

Recommended external references:

- `Classification of IT Support Tickets` (Zenodo): manually classified IT support tickets
- `Semantic Similarity of IT Support Tickets` (Zenodo): manually labeled support-ticket similarity pairs
- `MSDialog`: Microsoft technical support conversations for realistic support-language patterns

Concrete work in this phase:

- compare our issue types to external category patterns
- review whether our ambiguous tickets reflect real support ambiguity
- justify or reject candidate partial-credit pairs using external examples
- note any obvious taxonomy blind spots for future work

Important constraint:

- do **not** import external rows into `data/dataset.json` at this stage
- do **not** claim full external-dataset benchmarking unless we actually run it

Exit criteria:

- we can honestly say our environment design is grounded against real support data
- any scoring adjustments introduced in Phase 2 have an external rationale, not just intuition

### Phase 4: Hardening And Regression Safety
**Window:** April 6 to April 7

**Primary objective:** make the repo reliable from the outside, not just locally understandable

Deliverables:

- run the full test suite on the merged repo state
- keep or improve Docker smoke coverage
- if feasible, add CI for `pytest` in addition to Docker smoke
- rerun heuristic baseline and confirm it remains stable after test/scoring changes
- verify docs still match the implemented behavior

Exit criteria:

- runtime behavior, tests, and docs all agree
- no unresolved ambiguity remains about the baseline numbers
- Docker and app-surface behavior have at least one real validation path

### Phase 5: Freeze And Submission Packaging
**Window:** April 7 to April 8

**Primary objective:** stop taking avoidable risk

Allowed work:

- bug fixes
- doc corrections
- metadata fixes
- smoke-test reruns
- submission packaging

Avoid in this phase:

- new dataset content
- scoring experiments
- structural refactors
- "nice-to-have" features

Exit criteria:

- the repo is stable
- the docs are accurate
- the submission story is clear

## Test Strategy

### Unit Tests

Goal:

- prove the scorer, reward helpers, and dataset/task loaders behave exactly as intended

Priority unit targets:

- `grade_action()` exact-match, zero-score, and partial-credit cases
- unsupported `task_id` behavior
- task weights summing to expected behavior
- reward helper bounds
- dataset loader behavior including Windows BOM handling

### Smoke Tests

Goal:

- prove the environment works end to end with minimal assumptions

Priority smoke targets:

- `reset()` returns a valid observation
- `step()` advances the queue
- final reward stays in `[0.0, 1.0]`
- same seed gives the same episode behavior
- heuristic baseline completes without crashing

### Integration Tests

Goal:

- prove the real app surface behaves correctly, not just the pure Python helpers

Priority integration targets:

- `/health`
- `/tasks`
- `/reset`
- `/step`
- `/state`
- one full seeded episode through the app or client layer

## RL Improvement Rules

We should improve RL usefulness in ways that keep the env judge-friendly.

Good RL improvements:

- clearer deterministic feedback
- better exact-vs-partial boundaries
- richer but still simple episode history
- deterministic controls that help reproducible rollouts

Bad RL improvements:

- vague similarity expansion without examples
- turning exact business-routing fields into fuzzy fields
- adding complexity that makes the README harder to explain

## Grounding Rules

Grounding matters, but it must stay lightweight this late.

Good grounding work:

- audit our taxonomy against public support-ticket datasets
- use real support phrasing to validate dataset realism
- use labeled similarity pairs to justify a few near-miss cases

Bad grounding work:

- rushed ingestion of external datasets
- category remapping that forces taxonomy churn
- unsupported claims that our scores are benchmarked externally when they are not

## Ownership Split For The Final Stretch

### Roopal ownership

- grounding audit
- ticket realism review
- documentation updates
- competitive-positioning clarity

### Suyash ownership

- tests
- runtime hardening
- scoring and reward implementation changes
- Docker and integration validation

### Shared review items

- any changes to partial-credit rules
- any benchmark number updates
- final submission claims

## Priority Order If Time Gets Tight

If the deadline compresses further, do this exact order:

1. unit tests proving non-fuzzy scoring behavior
2. smoke and integration tests for seeded deterministic runs
3. grounding audit against external real-world support datasets
4. low-risk RL reward improvements
5. CI and extra polish

## Definition Of Done For This Final Plan

We are done when:

1. the scorer is test-backed and clearly not "always fuzzy"
2. the environment has unit, smoke, and integration coverage
3. the main RL improvements are implemented without hurting clarity
4. grounding is supported by external real-world support datasets
5. Docker, baseline behavior, and docs are all in sync

## Simple Rule To Remember

Improve learning signal.
Prove correctness.
Ground the story in real support data.
Do not take late-stage dataset-merging risk.
