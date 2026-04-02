# IT Helpdesk Ticket Routing Mental Model

This file is the practical mental model of the repo in its current form.

## What The Project Is

This repository is an OpenEnv environment for IT helpdesk ticket routing.

The environment presents a small queue of tickets. For each ticket, the agent must decide:

- issue type
- priority
- assignment group
- resolution action

## Main Runtime Flow

```text
inference.py
    |
    v
client.py  <---->  server/app.py
                         |
                         v
                server/environment.py
                  |       |        |
                  v       v        v
            grader.py  reward.py  tasks.py
                                  |
                                  v
                           data/dataset.json
```

## Main Files

- `models.py`
  Typed models for tickets, actions, observations, and state.

- `server/environment.py`
  Main environment engine.

- `server/grader.py`
  Deterministic partial-credit scorer.

- `server/reward.py`
  Step and trajectory reward helpers.

- `server/tasks.py`
  Task definitions and dataset loading.

- `client.py`
  Typed client used for multi-step interaction.

- `inference.py`
  Baseline runner with LLM mode and heuristic mode.

## Task Ladder

### Task 1

- predict `issue_type`

### Task 2

- predict `issue_type`
- predict `priority`

### Task 3

- predict `issue_type`
- predict `priority`
- predict `assignment_group`
- predict `resolution_action`

## Label Vocabulary

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

## Observation And State

The observation exposes:

- task metadata
- the current ticket
- queue progress counters
- history
- reward and done status

The state tracks:

- current task
- seed
- queue ticket IDs
- current ticket index
- per-ticket scores
- total reward

## Reward Logic

- each step returns the current ticket score
- the final reward is the average of per-ticket scores
- a small overshoot penalty exists as a safeguard

## Runtime Notes

The repo has now passed both the initial local heuristic run and a merged-state rerun on the current `main` branch.

Current local baseline:

- Task 1: `1.0000`
- Task 2: `0.8800`
- Task 3: `0.9400`
- Overall: `0.9400`

The merged-state rerun matched the same baseline numbers exactly.

One practical implementation note from runtime validation:

- `data/dataset.json` may be saved with a UTF-8 BOM on Windows, so `server/tasks.py` intentionally loads it with `utf-8-sig`

## Dataset Shape

Each record includes:

- `ticket_id`
- `title`
- `requester`
- `description`
- `issue_type`
- `priority`
- `assignment_group`
- `resolution_action`
- optional `ambiguity_note`
- optional `related_ticket_id`

## Short Version

If coming back later, remember this:

- the repo is a helpdesk ticket router
- the architecture is a small OpenEnv stack
- one ticket is shown at a time
- the agent predicts structured routing fields
- the grader gives deterministic partial credit
- `inference.py` is the baseline agent runner
- the local heuristic path now works end to end on the current merged repo state
