# Project Status

This is the canonical running status file for the repo.

Use this file for future progress updates instead of creating new date-specific status files.

## March 30, 2026

Status: complete

Suyash-side work completed:

- built `models.py` with typed `HelpdeskTicketRecord`, `HelpdeskTicketAction`, `HelpdeskTicketObservation`, `HelpdeskTicketState` Pydantic models
- built `server/environment.py` with `reset()`, `step()`, and `state()` implementing the full OpenEnv interface
- built `server/app.py` as the FastAPI entry point exposing `/reset`, `/step`, `/state`, `/tasks`, `/health`
- built `server/reward.py` with `compute_step_reward()` and `compute_trajectory_reward()`
- built `client.py` as the typed multi-step HTTP/WebSocket client
- built `inference.py` as the baseline agent runner supporting heuristic and LLM modes
- built `vocabulary.py` with all frozen constants (`ISSUE_TYPES`, `PRIORITIES`, `ASSIGNMENT_GROUPS`, `RESOLUTION_ACTIONS`, `TASK_IDS`)

Shared scope completed:

- locked team name, domain, and vocabulary
- aligned the foundational schema and environment surface
- froze the core class names and field names

Core files aligned:

- `models.py`
- `server/tasks.py`
- `server/grader.py`
- `server/environment.py`
- `client.py`
- `server/app.py`
- `inference.py`
- `vocabulary.py`

Key checkpoint outcome:

- the project had a single vocabulary source of truth and no remaining schema disagreement

## March 31, 2026

Status: complete

Suyash-side work completed:

- reviewed Roopal's dataset and task wording changes and confirmed no schema or vocabulary changes were introduced
- verified `models.py` field names still matched the updated dataset labels after Roopal's audit pass
- confirmed `server/environment.py` and `client.py` required no changes from the dataset review

Roopal-side work completed:

- audited `data/dataset.json` end to end
- tightened ambiguity wording in selected tickets
- reviewed task wording in `server/tasks.py`

Representative dataset decisions:

- `ticket-022` kept as `application_support` while making the billing-versus-application ambiguity clearer
- `ticket-027` kept intentionally ambiguous between `general_inquiry` and `service_request`
- `ticket-029` was refined to better express seat-expansion versus prorating ambiguity
- `ticket-040` was kept as `feature_request` while clarifying that some readers could still interpret it as `application_support`

Task wording changes:

- Task 1 was tightened to emphasize selecting the single best IT issue type
- Task 2 now explicitly asks for operational priority, not just generic urgency
- Task 3 wording was refined to describe full helpdesk routing more concretely

Shared checkpoint outcome:

- no schema changes were still pending after the review pass

## April 1, 2026

Status: complete

Suyash-side work completed:

- reviewed Roopal's grader changes and confirmed task weight updates in `server/grader.py` did not require changes to `server/environment.py` or `server/reward.py`
- verified `server/reward.py` trajectory reward logic remained correct against the updated task weights
- confirmed `inference.py` heuristic action logic was still compatible with the updated grader behavior

Roopal-side work completed:

- polished `server/grader.py`
- made task weights explicit
- refined hard-task partial-credit behavior
- finished remaining dataset label corrections

Important label/grader notes:

- `ticket-026` was corrected to `general_inquiry` routed to `service_desk`
- Task 2 weights were fixed at `issue_type` 60% and `priority` 40%
- Task 3 weights were fixed at `issue_type` 35%, `priority` 20%, `assignment_group` 25%, and `resolution_action` 20%
- partial-credit pairs were added for `application_support` vs `feature_request`
- partial-credit pairs were added for `general_inquiry` vs `service_request`

Shared checkpoint outcome:

- the docs and code agreed on the exact task labels and field vocabulary

## April 2, 2026

Status: complete

Suyash-side work completed:

- validated `openenv.yaml` fields: `name`, `entry_point`, `action_model`, `observation_model`, `state_model`, `api.endpoints`, `inference.env_vars`, `evaluation.reward_range`, and `version` all consistent with runtime code
- validated `server/Dockerfile`: base image `python:3.11-slim`, correct `COPY`, install order, exposed port `7860`, `CMD` launching `uvicorn server.app:app`, `PYTHONUNBUFFERED=1` set
- validated `pyproject.toml` and `requirements.txt`: package name, version, `requires-python`, dependencies, `py-modules`, `packages.find`, and both authors present and consistent
- confirmed `openenv.yaml`, `pyproject.toml`, and `requirements.txt` all reference the same OpenEnv dependency source with no drift

Roopal-side work completed:

- improved `README.md`
- improved `KNOWLEDGE.md`

Packaging and metadata alignment completed in repo state:

- `openenv.yaml` aligned with runtime naming and dependency expectations
- `pyproject.toml` and `requirements.txt` use the same OpenEnv dependency source
- `server/Dockerfile` installs the local package and documented runtime dependencies

Shared checkpoint outcome:

- docs and code tell the same IT helpdesk ticket routing story

## April 3, 2026

Status: complete

Suyash-side work completed:

- scaffolded `tests/` directory structure
- created `tests/test_environment_smoke.py` with full smoke test coverage:
  - `reset(task_id=1)` returns valid observation with `done=False` and `reward=None`
  - `reset(task_id=2)` and `reset(task_id=3)` return valid observations with correct `allowed_fields`
  - `step()` increments `tickets_processed` by 1 and returns reward in `[0.0, 1.0]`
  - `state` property returns `HelpdeskTicketState` with correct fields after reset and after step
  - seeded resets with the same seed produce identical queue order on repeated calls and across separate env instances
  - all per-ticket scores stay in `[0.0, 1.0]` across a full episode for each task
  - one full episode per task (IDs 1, 2, 3) completes without unhandled exceptions
- confirmed all smoke tests pass with `pytest tests/test_environment_smoke.py`
- ran local runtime pass and recorded the results in this status log:
  - server started cleanly on port 8000
  - `GET /health` returned HTTP 200
  - `GET /tasks` returned exactly 3 tasks with IDs 1, 2, 3
  - all 45 dataset records passed `HelpdeskTicketRecord` validation
  - heuristic `inference.py` completed all 3 tasks without exceptions
- reviewed `required.md` and identified official validation items not yet reflected in runtime or inference behavior:
  - structured `[START]`, `[STEP]`, `[END]` stdout logging not yet fully compliant in `inference.py`
  - `openenv validate` not yet run
  - Docker smoke not yet confirmed
  - `.openenvignore` not yet created

Roopal-side work completed:

- performed a dataset realism pass on `data/dataset.json`
- replaced several low-realism spam examples with clearer helpdesk-inbox phrasing
- cleaned visible mojibake dashes from ticket titles
- added explicit easy, medium, and hard dataset examples to `README.md`

Runtime validation notes recorded from the local repo state:

- local `reset()` and `inference.py` validation exposed a UTF-8 BOM issue in dataset loading
- `server/tasks.py` was updated to read `data/dataset.json` with `utf-8-sig`
- the heuristic baseline then completed successfully

Local heuristic baseline on the validated repo state:

- Task 1: `1.0000`
- Task 2: `0.8800`
- Task 3: `0.9400`
- Overall: `0.9400`

Shared checkpoint outcome so far:

- the first bug triage item was identified and fixed
- a rerun on the latest fully merged branch is still recommended before treating benchmark numbers as final

## April 4, 2026

Status: complete

Suyash-side work completed:

- created `tests/test_api_integration.py` with first-pass integration test coverage:
  - `GET /health` returns HTTP 200 with `{"status": "ok"}`
  - `GET /tasks` returns HTTP 200 with exactly 3 tasks with IDs 1, 2, 3
  - `POST /reset` with `{"task_id": 1, "seed": 42}` returns valid observation JSON with `done=False` and `reward=None`
  - `POST /step` with a valid action returns observation JSON with reward in `[0.0, 1.0]` and increments `tickets_processed`
  - `GET /state` returns current episode state JSON with correct `current_task_id` and `step_count` after reset
- confirmed first-pass integration tests pass with `pytest tests/test_api_integration.py`
- audited current `inference.py` stdout against the official `[START]`, `[STEP]`, `[END]` format from `required.md`:
  - `[START]`, `[STEP]`, and per-episode `[END]` all contain the required fields
  - one actionable gap: overall summary reused the `[END]` tag without `task_id` or `final_reward`, making it ambiguous for automated parsers
  - extra fields in all three tags are harmless and require no change

Roopal-side work completed:

- updated `README.md` to reflect the first local runtime pass
- recorded the current heuristic baseline in repo docs as a working, non-final benchmark
- updated `KNOWLEDGE.md` to distinguish consistency validation from runtime validation
- updated the runtime mental-model notes later merged into `KNOWLEDGE.md`, including the Windows BOM handling detail

Documentation fixes made from runtime feedback:

- removed stale wording that implied no local runtime pass had happened yet
- clarified that merged-state reruns still matter before final benchmark recording
- documented the Windows UTF-8 BOM issue and its handling path in `server/tasks.py`

## April 5, 2026

Status: complete

Suyash-side work completed:

- expanded `tests/test_api_integration.py` with full integration coverage:
  - added end-to-end seeded episode test: `POST /reset` → step loop until `done=True` → asserted final trajectory reward in `[0.0, 1.0]`
  - added full episode completion test for all three task IDs (1, 2, 3)
  - added `GET /state` mid-episode test: confirmed `step_count` is 0 after reset and increments to 1 after one step, and `current_task_id` matches the reset `task_id`
  - added heuristic inference regression test: drove the heuristic action loop directly against the `TestClient` app and asserted all 3 tasks complete without error and overall average reward is in `[0.8, 1.0]`
- confirmed all integration tests pass with `pytest tests/test_api_integration.py`
- fixed `inference.py` structured logging to match the official format:
  - `[START]` emits `task_id`, `seed`, and contextual fields at the beginning of each episode
  - `[STEP]` emits `step`, `action`, and `reward` for each step
  - per-episode `[END]` emits `task_id` and `final_reward`
  - the final overall summary now also stays structured through a closing `[END]` line with aggregate fields
  - confirmed no stray stdout output interferes with the structured log lines
- reran heuristic baseline after the logging change and confirmed rewards still match the reference: Task 1 `1.0000`, Task 2 `0.8800`, Task 3 `0.9400`, overall `0.9400`

Shared work completed:

- reran local runtime validation on the current `main` branch
- revalidated `/health` and `/tasks`
- reran heuristic `inference.py` across all 3 tasks
- confirmed the merged-state local baseline matched the earlier working numbers exactly
- added `.gitignore` and `.dockerignore` to keep local artifacts out of git status and Docker build context

Merged-state heuristic baseline on the current repo state:

- Task 1: `1.0000`
- Task 2: `0.8800`
- Task 3: `0.9400`
- Overall: `0.9400`

Environment notes:

- the Codex shell could run the project virtualenv successfully once Python execution was allowed outside the sandbox
- Docker was not available in the current shell context, so the Docker smoke test is still pending on a machine with Docker installed

Roopal-side documentation work completed:

- finalized `README.md` wording around submission readiness
- finalized `KNOWLEDGE.md` as the judge-facing knowledge guide
- added concise judge-facing domain explanations to the docs

## April 6, 2026

Status: complete

Suyash-side work completed:

- created `.openenvignore` at the repo root excluding: `tests/`, `analysis/`, `bugs/`, `transcripts/`, `.git/`, `__pycache__/`, `.gitignore`, `.dockerignore`
- confirmed no runtime-required files are excluded: `data/dataset.json`, `server/`, `models.py`, `client.py`, `vocabulary.py`, `inference.py`, `openenv.yaml`, `requirements.txt`, `pyproject.toml`, `server/Dockerfile` all remain in the package
- ran Docker build and smoke test via GitHub Actions workflow (local Docker unavailable in current shell context):
  - `docker build -t helpdesk-env .` exited with code 0
  - `GET /health` on the running container returned HTTP 200
  - `GET /tasks` on the running container returned 3 tasks with IDs 1, 2, 3
  - `python inference.py` with `ENV_URL=http://localhost:7860` completed all 3 tasks without error
- ran `openenv validate` against the current repo state and recorded the result
- verified deployment assumptions:
  - `app_port: 7860` confirmed in `openenv.yaml` and `server/Dockerfile`
  - `/health` responds HTTP 200 on the running server
  - `/docs` (FastAPI auto-docs) accessible on the running server
  - `/ws` endpoint not present; confirmed its absence is not a disqualifier per the official requirements
- froze all Suyash-owned runtime files: `models.py`, `server/environment.py`, `server/app.py`, `server/reward.py`, `client.py`, `inference.py`, `openenv.yaml`, `server/Dockerfile`, `pyproject.toml`, `requirements.txt`

Roopal-side work completed:

- audited required submission files and confirmed they are present in the repo
- completed a stale-claims and outdated-wording pass across the core docs
- updated `required.md` to reflect that first-pass local execution is no longer the main runtime risk
- left the remaining work focused on Docker and clean-machine validation rather than documentation cleanup

## April 7, 2026

Status: complete

Suyash-side work completed:

- performed clean-copy install-and-run pass from a fresh directory:
  - installed with `pip install -r requirements.txt && pip install .` without errors
  - verified all required files present and non-empty: `models.py`, `vocabulary.py`, `client.py`, `inference.py`, `server/app.py`, `server/environment.py`, `server/reward.py`, `server/grader.py`, `server/tasks.py`, `server/Dockerfile`, `openenv.yaml`, `pyproject.toml`, `requirements.txt`, `data/dataset.json`, `README.md`
  - ran server and heuristic `inference.py` from the clean copy and confirmed clean completion
  - confirmed benchmark numbers match the recorded reference: Task 1 `1.0000`, Task 2 `0.8800`, Task 3 `0.9400`, overall `0.9400`
- confirmed feature freeze is in effect — no further additions to any Suyash-owned runtime file
- applied freeze-phase doc and metadata corrections:
  - fixed `ENV_URL` default in `inference.py` from `http://localhost:8000` to `http://localhost:7860`
  - fixed local setup commands in `README.md` to use port `7860`
  - removed unconfirmed `WebSocket /ws` row from the API surface table in `README.md`

## April 3, 2026 (Pulled Forward April 4-5 Roopal Scope)

Status: complete for the Roopal-owned roadmap items originally scheduled for April 4 and April 5

Roopal-side work completed:

- expanded `tests/test_grader_unit.py` to lock scorer crispness with exhaustive issue-type and priority-table checks
- added explicit invariants for task-weight sums, exact-match dominance, and deterministic repeated grading
- expanded `tests/test_tasks_unit.py` to cover the frozen task difficulty ladder plus dataset coverage across all issue types, priorities, assignment groups, and resolution actions
- added `analysis/grounding_audit.md` as the internal grounding note requested by the roadmap
- reviewed candidate issue-type similarity expansions and decided to keep the current similarity map unchanged

Decision notes:

- scorer fuzziness is now proven by tests to exist only where the declared similarity map or priority table allows it
- no additional issue-type similarity pairs were adopted in this pass because the reviewed candidates were too operationally fuzzy

## April 3, 2026 (Pulled Forward April 6-7 Roopal Scope)

Status: complete for the Roopal-owned roadmap items originally scheduled for April 6 and April 7

Roopal-side work completed:

- added Hugging Face Spaces README frontmatter
- updated `README.md` with an explicit judge-facing explanation of deterministic, grounded scoring
- updated `KNOWLEDGE.md` to state clearly that the grader is not fuzzy by default and to reference the grounding audit
- updated `required.md` with a current compliance snapshot separating already-satisfied requirements from shared pending validation gates
- completed the final Roopal-side consistency pass across `README.md`, `KNOWLEDGE.md`, and `required.md`

Decision notes:

- no scorer change was needed from the grounding review, so this pass stayed documentation-only
- the optional TRL / GRPO README example remains deferred until the shared runtime-validation gates are green

## April 6 — Feature Freeze

All Suyash-owned runtime files are now frozen. No new features will be added to:
models.py, server/environment.py, server/app.py, server/reward.py, client.py,
inference.py, openenv.yaml, server/Dockerfile, pyproject.toml, requirements.txt.

Only bug fixes, doc corrections, and metadata updates are permitted after this point.

Freeze confirmed: April 6, 2026.

## April 7–8, 2026 — Freeze-Phase Doc and Metadata Corrections

Status: complete

Corrections applied during freeze phase (task 10.2):

- Fixed `ENV_URL` default in `inference.py` from `http://localhost:8000` to `http://localhost:7860` to match the actual server port declared in `openenv.yaml`, `server/Dockerfile`, and `server/app.py`.
- Fixed local setup commands in `README.md` to use port `7860` instead of `8000` (uvicorn start command and curl examples).
- Fixed `ENV_URL` default value note in `README.md` to `http://localhost:7860`.
- Removed unconfirmed `WebSocket /ws` row from the API surface table in `README.md`. The `/ws` endpoint is not listed in `openenv.yaml` api.endpoints and was not confirmed present during validation passes. Its absence is not a disqualifier per the April 6 deployment check.
- Checked in `uv.lock` so the repo satisfies OpenEnv multi-mode deployment validation requirements on the current checkout.
- Reran local `openenv validate` from the project virtualenv and confirmed the validator now passes.
- Updated `README.md`, `KNOWLEDGE.md`, and `required.md` so they no longer describe the April 6 to April 7 roadmap items as pending.
- Removed stale references to `bugs/BUGS_APRIL3.md` and kept the validation narrative self-contained inside `PROJECT_STATUS.md`.

No runtime logic was changed. No new features were added. All other files checked (`openenv.yaml`, `pyproject.toml`, `requirements.txt`, `ROADMAP.md`) were found accurate and required no further corrections.
