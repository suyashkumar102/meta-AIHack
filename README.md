---
title: IT Helpdesk Ticket Routing OpenEnv
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - helpdesk
  - ticket-routing
  - customer-support
---

# IT Helpdesk Ticket Routing OpenEnv

> Meta PyTorch OpenEnv Hackathon Round 1 submission  
> Team Hackstreet Boys: Roopal Guha Neogi, Suyash Kumar

This repository contains a deterministic OpenEnv environment for IT helpdesk ticket routing. An agent is shown one ticket at a time from a short queue and must predict the right issue type, operational priority, assignment group, and next action.

## Judge-Facing Summary

If a judge reads only one short explanation, it should be this:

- this environment models a real enterprise workflow, not a toy classification task
- each ticket requires typed routing decisions that are easy to score deterministically
- the task ladder moves cleanly from single-field classification to full operational routing
- the repo is small enough to rerun quickly and explicit enough to understand without hidden business logic

## What This Environment Simulates

The environment models a realistic helpdesk workflow:

1. a new ticket enters the queue
2. the agent reads the ticket title and description
3. the agent may investigate with lightweight tools, then submit structured routing fields
4. the grader assigns deterministic credit
5. the environment advances to the next ticket until the queue is complete

This domain is useful for OpenEnv because it is operationally realistic, easy to evaluate with typed outputs, and naturally supports a clean easy-to-hard task ladder.

## Why This Is A Good Hackathon Domain

- it reflects real enterprise support operations
- the action space is structured and judge-friendly, with a small investigate-versus-submit split
- correctness can be scored deterministically
- the hard task is meaningfully harder than the easy and medium tasks
- the environment is small enough to rerun quickly

## Environment Overview

The project uses a queue-based episode model.

- `reset()` samples a task and a queue of 3 to 5 tickets
- `step()` grades one ticket submission at a time
- `state()` exposes the internal episode snapshot
- final reward is based on average ticket quality across the queue

The environment classes and vocabulary are intentionally frozen to keep collaboration and judging simple.

## Task Ladder

| ID | Name | Difficulty | Required Fields | What The Agent Must Do |
|----|------|------------|-----------------|-------------------------|
| 1 | Issue Type Classification | Easy | `issue_type` | classify the ticket into the best issue category |
| 2 | Issue Type And Priority | Medium | `issue_type`, `priority` | classify the issue and estimate urgency |
| 3 | Full Ticket Routing | Hard | `issue_type`, `priority`, `assignment_group`, `resolution_action` | perform full routing and next-step selection |

## Locked Vocabulary

### Issue types

- `billing_license`
- `identity_access`
- `application_support`
- `service_request`
- `spam_phishing`
- `general_inquiry`
- `security_compliance`
- `onboarding`
- `feature_request`

### Priorities

- `critical`
- `high`
- `medium`
- `low`

### Assignment groups

- `license_ops`
- `service_desk`
- `application_team`
- `procurement`
- `security_team`
- `onboarding_ops`

### Resolution actions

- `fulfill`
- `escalate`
- `assign`
- `ignore`
- `acknowledge`

## Observation And State Model

The agent only sees routing inputs, not labels.

Visible ticket fields:

- `ticket_id`
- `title`
- `requester`
- `description`
- optional `ambiguity_note`
- optional `related_ticket_id`
- optional `related_ticket_preview`

Each observation also includes:

- `task_id`
- `task_name`
- `instructions`
- `allowed_fields`
- `available_tools`
- `investigation_budget_remaining`
- `last_tool_result`
- `queue_size`
- `tickets_remaining`
- `tickets_after_current`
- `tickets_processed`
- `queue_position`
- `history`
- standard OpenEnv fields such as `done` and `reward`

The internal `HelpdeskTicketState` tracks:

- `episode_id`
- `step_count`
- `current_task_id`
- `seed`
- `queue_ticket_ids`
- `current_ticket_index`
- `per_ticket_scores`
- `total_reward`
- `reward`
- `done`

## Grading And Reward

Scoring is deterministic and normalized to `[0.0, 1.0]`.

The action model now supports two paths:

- `action_type="submit"` for the final routing answer
- `action_type="investigate"` with a small built-in tool surface before submission

Available tools:

- `lookup_related_ticket`
- `lookup_requester_history`

Per-field behavior:

- `issue_type`: exact match, with a few near-miss partial-credit pairs
- `priority`: exact match or proximity credit
- `assignment_group`: exact match
- `resolution_action`: exact match

Task weights:

| Task | Issue Type | Priority | Assignment Group | Resolution Action |
|------|------------|----------|------------------|-------------------|
| 1 | 100% | - | - | - |
| 2 | 60% | 40% | - | - |
| 3 | 35% | 20% | 25% | 20% |

Final episode reward:

```text
average(per_ticket_scores)
```

The result is clamped to `[0.0, 1.0]`.

Step reward is lightly milestone-shaped: high per-ticket scores get a small bonus and very low scores get a small penalty before the final clamp.

Final reward also includes a tiny queue-economics penalty only when the agent exceeds the free investigation budget. One investigation per queued ticket is free; extra investigation steps reduce the final reward slightly.

## Grounded Scoring

The grader is intentionally not fuzzy by default.

- exact match is the dominant path for every field
- `assignment_group` and `resolution_action` are exact-match only
- `priority` only gets proximity credit from the declared table in `server/grader.py`
- `issue_type` only gets partial credit for a small declared similarity map
- wrong labels outside those explicit maps score `0.0`

That scoring policy is now backed by checked-in unit tests in `tests/test_grader_unit.py` and `tests/test_tasks_unit.py`.

The label set and partial-credit choices were also reviewed against public IT-support references captured in `analysis/grounding_audit.md`, including:

- `Classification of IT Support Tickets`
- `Semantic Similarity of IT Support Tickets`
- `MSDialog`

That grounding pass supported keeping the current similarity map small and explainable. No new issue-type similarity pairs were added from the review.

## Dataset Snapshot

The labeled dataset in `data/dataset.json` currently contains 45 tickets spanning straightforward and ambiguous helpdesk scenarios.

It includes:

- billing and license requests
- identity and access issues
- application support incidents
- service and procurement requests
- spam or phishing reports
- security and compliance work
- onboarding tickets
- feature requests
- follow-up cases linked through `related_ticket_id`

## Difficulty Coverage

The difficulty ladder is visible both in the task fields and in the dataset itself.

Easy-style examples:

- `ticket-020`: straightforward general inquiry with low urgency and a clean `general_inquiry` label
- `ticket-041`: clear onboarding request for a new contractor account
- `ticket-044`: obvious phishing-style lure that should map cleanly to `spam_phishing`

Medium-style examples:

- `ticket-001`: billing dispute that still requires the agent to judge urgency correctly
- `ticket-028`: application incident where the issue type is clear but priority still matters
- `ticket-036`: procurement-style proof-of-concept request that should route as a `service_request`

Hard-style examples:

- `ticket-022`: mixed billing and application signals in one ticket
- `ticket-029`: seat expansion combined with a prorating question
- `ticket-038`: follow-up billing thread with escalated urgency
- `ticket-045`: repeated account suspension thread with legal-escalation pressure

## Repository Layout

```text
server/
  app.py
  environment.py
  grader.py
  reward.py
  tasks.py
  Dockerfile
data/
  dataset.json
models.py
client.py
inference.py
vocabulary.py
openenv.yaml
pyproject.toml
requirements.txt
README.md
KNOWLEDGE.md
required.md
ROADMAP.md
```

## Core Files

- `models.py`: typed action, observation, state, and dataset record models
- `server/environment.py`: queue-based episode engine
- `server/tasks.py`: task definitions and dataset loader
- `server/grader.py`: deterministic scoring logic
- `server/reward.py`: reward helpers
- `client.py`: typed client for multi-step episodes
- `inference.py`: baseline agent runner
- `vocabulary.py`: frozen constants and routing defaults

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Start the environment locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Basic checks:

```bash
curl http://localhost:7860/health
curl http://localhost:7860/tasks
```

## Running The Baseline Inference Script

The baseline script supports single-task evaluator mode by default, plus an explicit local batch override.

### Heuristic mode

If no LLM credentials are set, it uses a keyword-based ticket router:

```bash
python inference.py
```

By default that runs exactly one task and emits exactly one `[START] ... [END]` block. To target a specific task:

```bash
TASK_ID=3 python inference.py
```

### LLM mode

Set these environment variables first:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Then run:

```bash
python inference.py
```

Optional target:

- `ENV_URL`
- default value: `http://localhost:7860`
- `TASK_ID`
- `RUN_ALL_TASKS`

To reproduce the multi-task local benchmark sweep:

```bash
RUN_ALL_TASKS=1 python inference.py
```

## Runtime Validation Snapshot

The repo has now completed both the first local heuristic validation pass and a merged-state rerun on the current `main` branch.

Validated locally:

- server startup
- `/health`
- `/tasks`
- `/reset`
- heuristic `inference.py` run across all 3 tasks with `RUN_ALL_TASKS=1`

Current local heuristic results:

| Task | Result |
|------|--------|
| Issue Type Classification | `1.0000` |
| Issue Type And Priority | `0.8800` |
| Full Ticket Routing | `0.9400` |
| Overall | `0.9400` |

The merged-state rerun matched these same numbers exactly, so they are the current benchmark reference for the repo. The April 6 to April 7 validation pass then closed the remaining roadmap gates with Docker smoke coverage via GitHub Actions, a clean-copy install-and-run rerun, structured inference-log verification, and a passing local `openenv validate` check after checking in `uv.lock`.

### Windows note

During the first runtime pass, the repo surfaced a Windows-specific JSON issue where `data/dataset.json` could include a UTF-8 BOM. The dataset loader in `server/tasks.py` now reads the file with `utf-8-sig`, so the environment resets cleanly even when the file was saved by a Windows editor.

## Docker

Build:

```bash
docker build -f server/Dockerfile -t helpdesk-ticket-routing .
```

Run locally:

```bash
docker run -p 7860:7860 helpdesk-ticket-routing
```

Then run inference against it (default `ENV_URL` points to `http://localhost:7860`):

```bash
RUN_ALL_TASKS=1 python inference.py
```

If you publish the container on a different host port, set `ENV_URL` accordingly before running `inference.py`.

If local Docker is blocked by machine setup, the repo also includes a GitHub Actions smoke test at `.github/workflows/docker-smoke-test.yml`. That workflow builds the image on a GitHub-hosted runner, starts the container, checks `/health` and `/tasks`, and runs heuristic `inference.py` against the container.

## API Surface

OpenEnv provides the core environment endpoints, and the repo adds a custom task listing route.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | health check |
| POST | `/reset` | start a new episode |
| POST | `/step` | submit an action |
| GET | `/state` | inspect internal state |
| GET | `/tasks` | list task metadata |
| GET | `/web` | lightweight HF Space UI |
| GET | `/docs` | interactive API docs |

## Submission Readiness

The repo is already aligned on:

- team name and members
- domain and vocabulary
- task ladder
- typed models
- grader and reward design
- packaging metadata and Docker entry point
- Hugging Face Spaces README frontmatter
- judge-facing documentation of deterministic, grounded scoring

An April 6 repo audit also confirmed that all required submission files are present:

- runtime: `models.py`, `client.py`, `inference.py`, `server/app.py`, `server/environment.py`, `server/grader.py`, `server/reward.py`, `server/tasks.py`
- data and metadata: `data/dataset.json`, `openenv.yaml`, `pyproject.toml`, `requirements.txt`, `server/Dockerfile`
- docs and planning: `README.md`, `KNOWLEDGE.md`, `required.md`, `PROJECT_STATUS.md`, `ROADMAP.md`

Roadmap status through April 7 is complete:

- unit, smoke, and integration tests are checked in and green
- Docker smoke coverage exists through `.github/workflows/docker-smoke-test.yml`
- `openenv validate` now passes on the current repo state
- structured `inference.py` logging is verified by tests and the merged-state rerun
- a clean-copy install-and-run pass has been completed

The remaining April 8 work is operational rather than implementation-heavy:

- run the final submission-branch sanity slice before pushing
- perform the live Hugging Face Space ping and reset check on the deployed submission artifact if a fresh deployment is created

The short TRL / GRPO README example from the roadmap remains intentionally deferred because it is optional and lower priority than freeze-phase stability.
