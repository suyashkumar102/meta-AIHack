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
- the task ladder now keeps full routing on every task and scales observability, queue pressure, and operational controls instead
- the repo is small enough to rerun quickly and explicit enough to understand without hidden business logic

## What This Environment Simulates

The environment models a realistic helpdesk workflow:

1. a new ticket enters the queue
2. the agent reads the ticket title and description
3. the agent may investigate, request more information, open an incident, defer the ticket, or submit a routing decision
4. the queue state mutates: capacity shrinks, incidents stay open, deferred tickets return later, poor handling can spawn follow-up tickets, and good or bad handling can reshape later tickets in the same request cluster
5. the grader assigns deterministic credit
6. the environment advances until the queue is complete

For hard-task tickets, the environment can now withhold decisive routing context until the agent uses the right investigation tool. That keeps the task from collapsing into one-shot classification and makes tool choice part of the policy.

This domain is useful for OpenEnv because it is operationally realistic, easy to evaluate with typed outputs, and naturally supports a clean easy-to-hard task ladder.

## Why This Is A Good Hackathon Domain

- it reflects real enterprise support operations
- the action space is structured and judge-friendly, but now includes meaningful operational controls beyond investigate-versus-submit
- correctness can be scored deterministically
- the hard task is meaningfully harder than the easy and medium tasks
- the environment is small enough to rerun quickly

## Environment Overview

The project uses a queue-based episode model.

- `reset()` samples a task and a queue of 3 to 5 tickets
- `step()` lets the agent investigate, request clarification, defer, open incidents, or submit one ticket at a time
- `state()` exposes the internal episode snapshot
- hard-task episodes also track queue-level capacity, incident slots, clustered follow-on tickets, alternate acceptable routes, planning penalties, SLA pressure, and dynamic follow-up tickets across the queue
- final evaluation is based on the queue outcome, not on isolated per-ticket classification alone

The environment classes and vocabulary are intentionally frozen to keep collaboration and judging simple.

## Lightweight Policy Improvement Loop

The repo includes a local policy runner in `policy_learning.py`. It still does not update model weights, but it now does more than cosmetic search: it evaluates repeated seeded rollouts, learns cue-conditioned tool preferences for investigation, uses the same planning-aware deterministic submit logic as `inference.py`, and ranks policies by terminal rubric reward first, then queue-management quality, with lower planning penalty as the next tie-breaker.

That gives the project a meaningful improvement loop for judge demos:

- compare `no_investigation`, `investigate_when_context_hidden`, and `adaptive_cue_bandit`
- log per-step rewards, feedback summaries, planning penalties, and reward components to JSONL
- learn when to use `lookup_queue_capacity_forecast` and `lookup_queue_cluster_summary` versus the other investigation tools
- select the best policy on train seeds, then re-evaluate it on holdout seeds

Example commands:

```bash
python policy_learning.py compare --seeds 42-51 --task-ids 1,2,3
python policy_learning.py search --train-seeds 40-49 --eval-seeds 50-59 --task-ids 1,2,3
```

Artifacts are written to `analysis/policy_learning_runs/` by default:

- `compare_summary.json`
- `compare_episodes.jsonl`
- `compare_trajectories.jsonl`
- `search_summary.json`
- `search_train_episodes.jsonl`
- `search_train_trajectories.jsonl`
- `search_eval_episodes.jsonl`
- `search_eval_trajectories.jsonl`

The default submit policy inside this runner stays deterministic and local. It reuses the repo's heuristic routing logic plus planning-aware routing overrides, and the policy loop can now also exercise operational actions such as `request_info`, `open_incident`, and `defer` without depending on external LLM latency or API cost.

## Task Ladder

| ID | Name | Difficulty | Required Fields | What The Agent Must Do |
|----|------|------------|-----------------|-------------------------|
| 1 | Guided Full Routing | Easy | `issue_type`, `priority`, `assignment_group`, `resolution_action` | route a mostly visible ticket correctly |
| 2 | Contextual Full Routing | Medium | `issue_type`, `priority`, `assignment_group`, `resolution_action` | route under partial observability with investigation, clarification, and moderate queue carry-over |
| 3 | Adaptive Queue Routing | Hard | `issue_type`, `priority`, `assignment_group`, `resolution_action` | route while managing queue pressure, incidents, clustered follow-ons, deferrals, and downstream follow-ups |

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
- optional `planning_note`
- optional `customer_update_note`
- optional `related_ticket_id`
- optional `related_ticket_preview`
- optional `routing_options`
- optional `capacity_state`
- optional `operational_context`
- optional `cluster_summary`
- optional `generated_from_ticket_id`

Each observation also includes:

- `task_id`
- `task_name`
- `instructions`
- `allowed_fields`
- `available_action_types`
- `available_tools`
- `investigation_budget_remaining`
- `last_tool_result`
- `queue_size`
- `tickets_remaining`
- `tickets_after_current`
- `tickets_processed`
- `queue_position`
- `average_score_so_far`
- `progress_fraction`
- `history`
- `last_reward_components`
- `rubric_reward` on terminal observations
- `metadata.last_feedback_summary` for compact reward / penalty feedback
- `metadata.capacity_state` on hard-task episodes
- `metadata.planning_penalty_total` and `metadata.planning_penalty_applied`
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
- `team_capacity_remaining`
- `high_priority_slots_remaining`
- `escalation_slots_remaining`
- `incident_slots_remaining`
- `planning_penalty_total`
- `incident_gap_total`
- `sla_breach_count`
- `queue_management_score`
- `queue_management_breakdown`
- `dynamic_queue_events`

## Grading And Reward

Scoring is deterministic and normalized to `[0.0, 1.0]`.

The action model now supports five paths:

- `action_type="submit"` for the final routing answer
- `action_type="investigate"` with a small built-in tool surface before submission
- `action_type="request_info"` to ask for customer / operator clarification on the current ticket
- `action_type="open_incident"` to reserve incident handling capacity before routing risky tickets
- `action_type="defer"` to push a ticket later in the queue and accept the downstream queue consequences

Available tools:

- `lookup_related_ticket`
- `lookup_requester_history`
- `lookup_internal_routing_note`
- `lookup_queue_capacity_forecast`
- `lookup_queue_cluster_summary`

Hard-task investigation behavior:

- some ambiguous and non-default-routing tickets start with both redacted titles and redacted descriptions
- linked-ticket previews and internal routing notes stay hidden until the matching tool is used
- capacity-sensitive tickets can expose queue pressure, future demand, and alternate routing options through `lookup_queue_capacity_forecast`
- cluster-sensitive tickets can expose future related tickets, shared-requester load, and active incident coverage through `lookup_queue_cluster_summary`
- detailed cluster counts and future queue-demand breakdowns stay hidden until the matching queue tool is used
- only useful investigation steps return a small positive shaping reward
- blind or repeated probing does not pay by default
- premature hard-task submission can incur a shaping penalty even when the visible text looks plausible
- resource-greedy routing can add planning penalties later in the queue even when a single ticket looks correct in isolation
- incident-sensitive tickets can require an explicit `open_incident` step to avoid future follow-up debt
- strong handling on an earlier clustered ticket can make later tickets cheaper to acknowledge, while weak handling can escalate those later tickets
- bad or incomplete hard-task handling can append a deterministic follow-up ticket later in the same episode
- terminal `rubric_reward` remains the objective evaluation signal, while per-step `reward` is the denser training signal

Per-field behavior:

- `issue_type`: exact match, with a few near-miss partial-credit pairs
- `priority`: exact match or proximity credit
- `assignment_group`: exact match, with a small declared partial-credit map for nearby ownership mistakes
- `resolution_action`: exact match, with a small declared partial-credit map for nearby next-step mistakes
- hard task only: some tickets also declare an alternate acceptable route with a reduced score multiplier, so the grader can reward capacity-aware fallback choices without collapsing into full fuzziness

Task weights:

| Task | Issue Type | Priority | Assignment Group | Resolution Action |
|------|------------|----------|------------------|-------------------|
| 1 | 40% | 20% | 20% | 20% |
| 2 | 32% | 20% | 24% | 24% |
| 3 | 30% | 20% | 25% | 25% |

Final episode rubric reward is queue-based:

```text
clamp(route_trajectory_reward * route_weight + queue_management_score * queue_weight - extra investigation penalties)
```

Both `reward` and `rubric_reward` now use the closed interval `[0.0, 1.0]`.

Step reward is lightly milestone-shaped: high per-ticket scores get a small bonus and very low scores get a small penalty before the final clamp.

Final reward also includes a queue-economics penalty when the agent exceeds the free investigation budget. One investigation-style step per queued ticket is free, but extra investigation or clarification steps reduce the final reward more noticeably than before. On hard-task queues, assignment-group capacity, high-priority slots, escalation slots, incident slots, and deferred-ticket SLA pressure all create cross-ticket trade-offs.

To make the environment more RL-friendly, each observation now also surfaces structured reward telemetry:

- `last_reward_components` exposes ticket score, shaped step reward, milestone adjustment, trajectory reward when applicable, and any investigation penalty applied
- `average_score_so_far` and `progress_fraction` expose trajectory progress without leaking future labels
- medium and hard telemetry now also exposes terminal `queue_management_score` plus a queue-management breakdown
- hard-task telemetry includes planning penalties, capacity usage, and the post-action capacity snapshot
- `history` retains the same reward components plus a compact `feedback_summary` string for downstream agents

## Grounded Scoring

The grader is intentionally narrow and declared, not fully fuzzy.

- exact match is the dominant path for every field
- `assignment_group` and `resolution_action` now expose only a small declared partial-credit map for nearby mistakes
- `priority` only gets proximity credit from the declared table in `server/grader.py`
- `issue_type` only gets partial credit for a small declared similarity map
- hard-task alternate routes must be explicitly declared in the dataset and carry an explicit score multiplier
- wrong labels outside those explicit maps score `0.0`

That scoring policy is now backed by checked-in unit tests in `tests/test_grader_unit.py` and `tests/test_tasks_unit.py`.

The label set and partial-credit choices were also reviewed against public IT-support references captured in `analysis/grounding_audit.md`, including:

- `Classification of IT Support Tickets`
- `Semantic Similarity of IT Support Tickets`
- `MSDialog`

That grounding pass supported keeping the current similarity map small and explainable. No new issue-type similarity pairs were added from the review.

## Dataset Snapshot

The effective labeled dataset now contains 70 tickets spanning straightforward, ambiguous, and planning-sensitive helpdesk scenarios.

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
- 16 tickets with explicit ambiguity notes
- 7 linked follow-up cases
- 22 tickets with declared alternate routes for queue-level planning

## Difficulty Coverage

The difficulty ladder is now visible in observability and control, not just in the submitted field count.

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
- generated `*-followup` tickets: deterministic reopened cases that only appear when the earlier handling was incomplete or operationally risky

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

The baseline script defaults to all declared tasks when `TASK_ID` is not set, which keeps local runs aligned with validator-style sweeps.

### Heuristic mode

If no LLM credentials are set, it uses a keyword-based ticket router:

```bash
python inference.py
```

By default that runs all declared tasks and emits a structured `[START] ... [STEP] ... [END]` block for each task. To target a specific task:

```bash
TASK_ID=3 python inference.py
```

### LLM mode

Set these environment variables first:

  - `API_BASE_URL`
  - `MODEL_NAME`
  - `API_KEY`
  - `HF_TOKEN`

Then run:

```bash
python inference.py
```

Optional target:

- `ENV_URL`
- default value: `http://localhost:7860`
- `SEED`
- `TASK_ID`
- `RUN_ALL_TASKS`
  compatibility alias for local tooling; all tasks already run by default when `TASK_ID` is unset

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

Current local smoke expectations:

- the baseline completes all 3 tasks successfully
- rewards remain in range for every task
- the hard task now depends much more heavily on investigation behavior, so exact seed-level baseline numbers are no longer treated as the benchmark reference for the repo

The April 6 to April 7 validation pass then closed the remaining roadmap gates with Docker smoke coverage via GitHub Actions, a clean-copy install-and-run rerun, structured inference-log verification, and a passing local `openenv validate` check after checking in `uv.lock`.

### Windows note

During the first runtime pass, the repo surfaced a Windows-specific JSON issue where `data/dataset.json` could include a UTF-8 BOM. The dataset loader in `server/tasks.py` now reads the file with `utf-8-sig`, so the environment resets cleanly even when the file was saved by a Windows editor.

## Docker

Build:

```bash
docker build -t helpdesk-ticket-routing .
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
