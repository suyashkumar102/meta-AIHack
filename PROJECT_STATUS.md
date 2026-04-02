# Project Status

This is the canonical running status file for the repo.

Use this file for future progress updates instead of creating new date-specific status files.

## March 30, 2026

Status: complete

Scope completed:

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

Status: Roopal work complete, shared validation underway

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

Status: Roopal work complete, shared rerun still pending

Roopal-side work completed:

- updated `README.md` to reflect the first local runtime pass
- recorded the current heuristic baseline in repo docs as a working, non-final benchmark
- updated `KNOWLEDGE.md` to distinguish consistency validation from runtime validation
- updated `MENTAL_MODEL.md` with runtime-validated notes and the Windows BOM handling detail

Documentation fixes made from runtime feedback:

- removed stale wording that implied no local runtime pass had happened yet
- clarified that merged-state reruns still matter before final benchmark recording
- documented the Windows UTF-8 BOM issue and its handling path in `server/tasks.py`

## April 5, 2026

Status: shared merged-state rerun complete, Docker smoke test still pending

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

Status: Roopal-side repo audit complete, shared execution checks still pending

Roopal-side work completed:

- audited required submission files and confirmed they are present in the repo
- completed a stale-claims and outdated-wording pass across the core docs
- updated `PLAN.md` to reflect that first-pass local execution is no longer the main runtime risk
- left the remaining work focused on Docker and clean-machine validation rather than documentation cleanup

## Open Items

Still pending after the current checkpoint:

- perform a Docker smoke test from the current merged repo state
- do a clean-machine dry run if possible before final submission freeze
