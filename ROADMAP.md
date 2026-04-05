# Hackstreet Boys Final Roadmap

## Team

- Team name: Hackstreet Boys
- Members:
  - Roopal Guha Neogi
  - Suyash Kumar
- Submission deadline: April 8, 2026, 11:59 PM IST

## How To Use This File

- `PROJECT_STATUS.md` is the canonical log of completed work.
- This roadmap is the remaining execution plan from the current repo state to final submission.
- `required.md` is now the combined official-requirements and project-compliance file.
- `KNOWLEDGE.md` defines the current repo truth and judge-facing explanation.
- `analysis/competition_notes.md` is the merged internal competitive note. Use it to prioritize work, but do not mention competitor repos in public-facing docs.

## What We Are Optimizing For

The highest-value wins from now to submission are:

1. **Robustness**
   - prove the env works through unit, smoke, and integration tests
   - make Docker and clean reruns boring and reliable

2. **RL improvement**
   - keep the reward deterministic
   - make sure scoring is not "always fuzzy"
   - add only small, safe improvements that strengthen reward quality or episode usefulness

3. **Real-world grounding**
   - ground our taxonomy and partial-credit choices against real public support-ticket datasets
   - do this as an audit / evidence layer, not as a late dataset merge

4. **Submission readiness**
   - satisfy every requirement from `required.md` and `KNOWLEDGE.md`
   - keep the repo easy for judges to understand and rerun

## Current Repo State

The repo already has:

- locked IT helpdesk routing domain
- locked vocabulary and task names
- 3-task difficulty ladder
- deterministic grading with limited partial credit
- working heuristic baseline
- merged local validation on `/health`, `/tasks`, and `inference.py`
- current local benchmark reference:
  - Task 1: `1.0000`
  - Task 2: `0.8800`
  - Task 3: `0.9400`
  - Overall: `0.9400`

The remaining work should be treated as targeted strengthening, not broad feature invention.

## Submission Gates That Must Still Hold

These come directly from `required.md` and `KNOWLEDGE.md`:

- the environment starts correctly
- `reset()`, `step()`, and `state()` behave correctly
- 3 tasks exist and remain meaningfully different
- grader scores stay in `[0.0, 1.0]`
- `inference.py` runs reproducibly without crashing
- `inference.py` uses the OpenAI client with `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- structured stdout logs follow the official `[START]`, `[STEP]`, and `[END]` format
- `openenv validate` passes
- Docker builds and starts cleanly
- HF deployment responds cleanly and reset works
- inference stays inside the official runtime / machine envelope
- docs and metadata are current
- the repo is easy for judges to understand and rerun

## Scope Decisions

### Do Now

- add tests:
  - unit
  - smoke
  - integration
- prove the scorer is crisp where it should be crisp
- add only safe RL-oriented improvements
- add external grounding evidence without changing the runtime dataset
- finish packaging / deployment readiness
- verify official validation constraints, not just local happy-path behavior

### Do Not Do Before Submission

- MCP migration
- transform-based reward refactor
- large dataset expansion
- external dataset merge into `data/dataset.json`
- major schema changes
- broad prompt / inference rewrites that could disturb the stable baseline
- dependency churn just for polish

## Codex-First Working Rules

Because we are using Codex to generate code, we should optimize for small, bounded tasks:

1. one prompt = one scoped change set
2. keep ownership by file group
3. require tests for any scorer or runtime change
4. review the diff before accepting generated code
5. rerun the relevant test slice after each meaningful change
6. do not ask Codex for a giant multi-file redesign this late

## Phased Plan

## Phase 1: Test And Robustness Foundation

**Window:** April 3 to April 4

**Goal:** eliminate the biggest competitive weakness identified in `analysis/competition_notes.md`: lack of checked-in tests.

### Must produce

- `tests/` with at least:
  - grader unit tests
  - task / dataset loader unit tests
  - reward / score-range unit tests
  - environment smoke tests
  - API integration tests

### Test plan

#### Unit tests

- exact match gives `1.0`
- unsupported task IDs fail clearly
- only intended near-miss issue-type pairs get partial credit
- unrelated wrong issue types get `0.0`
- priority proximity rules behave exactly as defined
- assignment group and resolution action remain exact-match only
- task weights sum and apply correctly
- dataset loads cleanly with `utf-8-sig`

#### Smoke tests

- `reset()` returns a valid observation
- `step()` advances queue progress
- `state()` reflects runtime state
- seeded resets are deterministic
- scores remain in `[0.0, 1.0]`
- one full episode per task completes without errors

#### Integration tests

- `/health`
- `/tasks`
- `/reset`
- `/step`
- `/state`
- one end-to-end seeded episode over HTTP or client path
- one heuristic `inference.py` regression check on expected overall behavior

### Why this phase matters

- addresses the biggest repo-quality gap vs stronger competitors
- improves robustness
- gives us safe rails for all later RL and grounding changes

## Phase 2: Scoring Calibration And Safe RL Improvements

**Window:** April 4 to April 5

**Goal:** improve RL usefulness without destabilizing the submission.

### Must produce

- scorer calibration evidence that the system is not "always fuzzy"
- only a few safe RL-oriented improvements if tests stay green

### Required calibration checks

- exact-match path is dominant and clearly tested
- fuzziness exists only in explicitly defined cases
- wrong labels outside the similarity map score `0.0`
- assignment group and resolution action remain exact
- final episode reward stays bounded and deterministic

### Safe improvement candidates from `analysis/competition_notes.md`

- expand `ISSUE_TYPE_SIMILARITY` with only a few defensible pairs, if backed by grounding review
- enrich `history` with:
  - ticket title
  - predicted fields
- optionally support `queue_size` as a reset kwarg only if the change is tiny and fully tested

### Hard stop

- if a change touches behavior and shifts baseline numbers unexpectedly, stop and stabilize rather than stacking more changes

## Phase 3: Real-World Grounding Audit

**Window:** April 5 to April 6

**Goal:** add defensible evidence that our taxonomy and partial-credit logic are grounded in real support data, without merging external data into runtime.

### Grounding strategy

- use real public support datasets as reference material
- compare their labels / examples against our taxonomy
- create an internal audit, not a runtime dependency

### Recommended grounding references

- `Classification of IT Support Tickets` (Zenodo, 2,229 manually classified tickets)
- `Semantic Similarity of IT Support Tickets` (Zenodo, 300 manually labeled ticket pairs)
- `MSDialog` for real technical-support conversation patterns and terminology

### Must produce

- an internal grounding note or checklist that captures:
  - which public datasets were reviewed
  - how our labels map to real-world ticket themes
  - which partial-credit pairs are defensible
  - which proposed similarity pairs were rejected as too fuzzy

### Useful output

- 10 to 20 grounding examples:
  - real ticket theme
  - closest label in our taxonomy
  - whether it should be exact-match only or partial-credit-adjacent

### Why this phase matters

- strengthens real-world credibility
- supports RL reward quality with evidence
- helps avoid arbitrary or over-fuzzy scorer changes

## Phase 4: Packaging, Deployment, And Judge-Facing Polish

**Window:** April 6 to April 7

**Goal:** close the submission-readiness gaps surfaced in `analysis/competition_notes.md`.

### Must produce

- Hugging Face Spaces README frontmatter
- `.openenvignore`
- `openenv validate` evidence
- Docker smoke evidence on the merged branch
- one clean-copy rerun if possible
- structured inference logging verified against the official format
- a practical check that inference remains inside the official runtime envelope

### Nice-to-have only if green

- short TRL / GRPO example in `README.md`
- concise note in docs that grading is deterministic, partially structured, and not purely fuzzy

### Do not do here

- no dataset expansion
- no major inference rewrite
- no architecture refactor

## Phase 5: Freeze And Submit

**Window:** April 8

**Goal:** submit from a calm, validated repo state.

### Final day rules

- only typo-level, doc-level, or packaging-only fixes
- no risky scorer changes
- no runtime refactors
- no dataset edits unless they fix a blocker
- stop risky edits several hours before submission
- if possible, run the official validator or the closest local equivalent before final push

## Ownership From Now Until Submission

### Roopal ownership

Primary files:

- `data/dataset.json`
- `server/tasks.py`
- `server/grader.py`
- `README.md`
- `KNOWLEDGE.md`

Primary responsibilities:

- scorer calibration and label quality
- unit tests around grader / task rules / dataset invariants
- real-world grounding audit
- judge-facing explanation of deterministic scoring and real-world realism
- safe reward-quality improvements only when grounded and tested

Concrete deliverables:

- grader unit tests
- grounding mapping note
- any similarity-matrix update, if justified
- doc updates if benchmark numbers or scoring explanation change
- README frontmatter and judge-facing clarity
- official requirement compliance review through `required.md`

### Suyash ownership

Primary files:

- `models.py`
- `server/environment.py`
- `server/app.py`
- `server/reward.py`
- `client.py`
- `inference.py`
- `openenv.yaml`
- `server/Dockerfile`
- `pyproject.toml`
- `requirements.txt`

Primary responsibilities:

- smoke and integration tests
- runtime stability
- Docker and deployment readiness
- inference reproducibility
- clean rerun evidence
- optional small RL-signal improvements on the runtime side

Concrete deliverables:

- env smoke tests
- API integration tests
- heuristic inference regression path
- `.openenvignore`
- Docker smoke confirmation
- clean-copy rerun if possible
- structured inference logging compliance

### Shared responsibilities

- do not rename schemas or vocabulary
- rerun the benchmark after any behavior-affecting change
- keep `PROJECT_STATUS.md` honest
- use the GitHub Actions Docker smoke workflow when local Docker is blocked
- review Codex-generated diffs before accepting them
- freeze feature work by the end of April 7
- do not casually change the `[START]`, `[STEP]`, `[END]` inference log format once implemented

## Date-By-Date Execution Plan

## April 3, 2026

Primary goal:

- lock the execution plan and begin test scaffolding immediately

Roopal:

- finalize the exact scorer behaviors that must be proven by tests
- list the exact-match-only cases and intended partial-credit cases
- begin grader and task-loader unit tests

Suyash:

- scaffold `tests/`
- begin smoke tests for `reset()`, `step()`, `state()`, and deterministic seeded behavior
- confirm how integration tests will hit the app cleanly
- review `required.md` and identify the exact official validation items still not reflected in runtime / inference behavior

Shared checkpoint:

- test strategy is agreed
- file ownership is clear
- no one is making unscoped runtime changes yet

## April 4, 2026

Primary goal:

- land the first complete test layer

Roopal:

- complete grader, task, and dataset unit tests
- add explicit tests showing where fuzziness is allowed and where it is not

Suyash:

- complete smoke tests
- add first-pass integration tests for `/health`, `/tasks`, `/reset`, and `/step`
- begin checking how current `inference.py` differs from the official structured logging requirement

Shared checkpoint:

- checked-in tests exist
- the repo can prove deterministic scoring and score bounds
- any failing behavior is triaged before adding improvements

## April 5, 2026

Primary goal:

- improve RL usefulness safely

Roopal:

- start the grounding audit using the selected public datasets
- decide whether any additional similarity pairs are truly defensible

Suyash:

- add integration coverage for full seeded episode flow and `state()`
- add a light heuristic regression path for `inference.py`
- optionally enrich observation history if tests are already green
- bring `inference.py` closer to official structured logging format if the change can be done safely

Shared checkpoint:

- tests are stable
- any RL-oriented change is small and justified
- no baseline drift goes unexplained

## April 6, 2026

Primary goal:

- finish grounding evidence and close packaging gaps

Roopal:

- finish grounding audit note
- land only the scorer adjustments supported by audit evidence, if any
- update docs to reflect deterministic, grounded scoring

Suyash:

- add `.openenvignore`
- verify Docker smoke workflow on the merged branch
- check deployment assumptions around `app_port`, `/docs`, `/health`, `/ws`, and `/web`
- run `openenv validate` or the closest available validation path
- verify structured inference logging and runtime-envelope expectations

Shared checkpoint:

- grounding evidence exists
- packaging gaps are closed or explicitly blocked
- benchmark references are still current

## April 7, 2026

Primary goal:

- freeze on a green, submission-ready repo

Roopal:

- final docs consistency pass across `README.md` and `KNOWLEDGE.md`
- add a short TRL / GRPO usage example only if everything else is already green

Suyash:

- do a clean-copy install-and-run pass if possible
- rerun heuristic baseline if any runtime-side change landed
- freeze runtime files by end of day

Shared checkpoint:

- tests are green
- Docker evidence exists
- docs, metadata, and runtime tell the same story
- feature work stops unless the gated competitive-hardening window below is explicitly activated after all required checks are already green

## After April 7 If Green: Competitive Hardening Window

**Window:** late April 7 to early April 8 only if all required gates are already green

**Goal:** improve the repo's competitive position against the strongest submissions by winning on reliability, validation quality, RL usefulness, and judge readability rather than by trying to match their architecture complexity.

### Activation rule

Activate this block only if all of the following are already true:

- smoke, unit, and integration tests are green
- Docker evidence exists or the blocker is clearly external
- `openenv validate` has passed or the closest available validator path is already recorded
- structured inference logging is already compliant or one tiny remaining fix is clearly isolated
- the benchmark is stable and any behavior-changing diff can still be rerun safely

If any of those are not true, skip this entire block and proceed directly to freeze / submission.

### Allowed competitive upgrades

- strengthen validation proof:
  - add or tighten environment smoke tests
  - add or tighten API integration tests
  - add one lightweight heuristic regression check for `inference.py`
- strengthen deployment proof:
  - record `openenv validate` evidence
  - record Docker smoke evidence
  - record deployment-assumption checks for `app_port`, `/health`, `/docs`, `/ws`, and `/web`
  - record one clean-copy rerun if practical
- add only tiny RL-signal improvements if fully tested and benchmark-stable:
  - enrich `history` with ticket title and predicted fields
  - add `queue_size` as a reset kwarg only if the change remains small, bounded, and fully tested
- add final judge-facing polish only after runtime proof is green:
  - short TRL / GRPO README example
  - concise README note on why our dense deterministic reward is more RL-friendly than binary-only grading

### Hard limits

- do not add MCP
- do not add a simulator layer
- do not add browser or multimodal features
- do not expand the runtime dataset
- do not make broad inference rewrites
- do not stack multiple behavior changes without rerunning the benchmark

### Decision rule

- if a competitive-hardening change is tiny, tested, and clearly improves trust or judge readability, it is allowed
- if it adds architectural ambition at the expense of stability, skip it
- if it causes unexplained baseline drift, revert to the last green state and submit

### Ownership

Roopal:

- final judge-facing README / KNOWLEDGE / `required.md` polish
- RL-justification wording around deterministic partial credit
- TRL / GRPO example only after all runtime proof is green

Suyash:

- validation evidence
- deployment proof
- tiny runtime-side RL-signal improvements only if fully tested

Shared checkpoint:

- the repo is already submission-safe before this block starts
- every change in this block is optional
- if time gets tight, cut this whole block first

## April 8, 2026

Primary goal:

- submit early from a calm repo state

Morning:

- if the repo is already fully green, optionally activate the competitive-hardening window above for one last small, tested improvement
- run final smoke / test slice on the submission branch
- verify required files are present
- verify README and metadata are current
- run the final validation checklist from `required.md`

Afternoon:

- only typo-level or packaging-only fixes
- no risky code changes

Final rule:

- stop risky edits several hours before 11:59 PM IST
- submit as soon as the repo is clearly green

## Cut Order If Time Gets Tight

Cut these first:

1. the entire competitive-hardening window after April 7
1. `queue_size` reset kwarg
2. richer `history`
3. TRL / GRPO README example
4. any optional similarity expansion beyond the most defensible cases

Do not cut these:

1. tests
2. scorer crispness checks
3. Docker / deployment validation
4. grounding audit evidence
5. final benchmark sanity rerun if behavior changed
6. official structured inference logging compliance

## Definition Of Done

The project is ready when:

1. unit, smoke, and integration tests exist and cover the critical paths
2. scoring is demonstrably deterministic and not fuzzy by default
3. a grounding audit against real public support datasets exists
4. the heuristic baseline still runs successfully
5. the inference path is compliant with the official log format
6. `openenv validate` and Docker checks are validated
7. docs and metadata are current and judge-friendly
8. the repo is frozen and submitted on time

## Simple Rule To Remember

Roopal owns the labels, scoring truth, grounding, and public clarity.
Suyash owns the runtime, tests beyond unit scope, packaging, and reproducibility rails.
Both of you should optimize for a clean, defensible, rerunnable submission rather than last-minute complexity.
