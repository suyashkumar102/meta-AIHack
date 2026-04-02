# IT Helpdesk Ticket Routing OpenEnv - Knowledge Guide

## What The Hackathon Is Looking For

The judges want a real-world environment that follows the OpenEnv pattern and can be understood quickly.

That means this repo needs:

1. typed action, observation, and state models
2. working `reset()`, `step()`, and `state()`
3. at least three difficulty levels
4. deterministic grading
5. meaningful reward shaping
6. a baseline `inference.py`
7. Docker and metadata that are easy to rerun

## Why IT Helpdesk Ticket Routing Fits Well

This domain is a strong fit because it is:

- realistic
- structured
- judge-friendly
- deterministic to grade
- naturally multi-step

A helpdesk agent has to decide what the ticket is about, how urgent it is, who should own it, and what should happen next. That maps cleanly to a typed action object.

## The Repo In One Sentence

This environment simulates a short helpdesk queue where an agent routes one ticket at a time and is graded on structured routing quality.

## Judge-Facing Explanation

If a judge asks why this environment is a strong submission, the concise answer is:

1. IT helpdesk routing is a real operational workflow with clear business value.
2. The input is realistic free-form ticket text, but the output is typed and easy to grade deterministically.
3. The three-task ladder creates a clean progression from basic classification to full queue routing.
4. The repo stays judge-friendly because the vocabulary, task labels, and scoring rules are all explicit and frozen.

## Frozen Project Identity

- Team name: `Hackstreet Boys`
- Members: `Roopal Guha Neogi`, `Suyash Kumar`
- Domain: `IT Helpdesk Ticket Routing`
- OpenEnv name: `it_helpdesk_ticket_routing_openenv`
- App environment name: `it_helpdesk_ticket_routing`

## Frozen Runtime Vocabulary

### Fields

- `issue_type`
- `priority`
- `assignment_group`
- `resolution_action`

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

## Main Models

### `HelpdeskTicketRecord`

Represents the labeled dataset row used for grading.

Important fields:

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

### `HelpdeskTicketAction`

Represents the agent submission. Fields are optional because different tasks score different subsets.

### `HelpdeskTicketObservation`

Represents what the agent sees for each step:

- task metadata
- visible ticket fields
- queue progress
- score history

### `HelpdeskTicketState`

Represents the internal episode state used by the environment.

## Episode Flow

### `reset()`

On reset, the environment:

1. chooses the task definition
2. samples a queue of 3 to 5 tickets
3. initializes a new episode id and state
4. returns the first observation

### `step(action)`

On each step, the environment:

1. grades the action against the current ticket
2. stores the per-ticket score
3. increments queue progress
4. returns the next observation or final result

### `state()`

Returns the internal state snapshot for debugging or inspection.

## Task Design

### Task 1: Issue Type Classification

The agent only predicts:

- `issue_type`

Purpose:

- establish the simplest classification baseline

### Task 2: Issue Type And Priority

The agent predicts:

- `issue_type`
- `priority`

Purpose:

- force the agent to understand both topic and urgency

### Task 3: Full Ticket Routing

The agent predicts:

- `issue_type`
- `priority`
- `assignment_group`
- `resolution_action`

Purpose:

- evaluate complete operational routing behavior

## Grading Mental Model

The grader is deterministic and intentionally simple to explain.

- `issue_type` gets exact or partial credit for selected near-miss pairs
- `priority` gets exact or proximity credit
- `assignment_group` gets exact credit
- `resolution_action` gets exact credit

Task weighting:

- Task 1: only `issue_type`
- Task 2: `issue_type` 60%, `priority` 40%
- Task 3: `issue_type` 35%, `priority` 20%, `assignment_group` 25%, `resolution_action` 20%

## Reward Mental Model

Step reward:

- current ticket score clamped to `[0.0, 1.0]`

Final reward:

- average of ticket scores
- minus a small overshoot penalty for taking more steps than the queue length

This gives dense feedback while still rewarding efficient episode completion.

## Dataset Mental Model

The dataset is small enough to audit manually but varied enough to support a meaningful benchmark.

Current structure:

- 45 tickets
- clear easy examples
- medium cases where urgency matters
- harder ambiguous cases
- follow-up tickets connected through `related_ticket_id`

The dataset is meant to test routing judgment, not just keyword spotting.

## Inference Script In Simple Terms

`inference.py` is the baseline agent runner.

It:

1. connects to the environment
2. loads the available tasks
3. runs one episode per task
4. picks an action for each ticket
5. sends the action back through the client
6. records rewards
7. prints a task-by-task summary

It supports:

- heuristic mode with no external model
- LLM mode through an OpenAI-compatible API

## Files That Matter Most

- `vocabulary.py`: locked constants and default routing maps
- `models.py`: typed schema and validation
- `server/environment.py`: episode engine
- `server/tasks.py`: task ladder and dataset loader
- `server/grader.py`: deterministic scoring
- `server/reward.py`: reward helpers
- `server/app.py`: OpenEnv app entry point
- `client.py`: typed multi-step client
- `openenv.yaml`: environment metadata
- `server/Dockerfile`: container entry point

## Validation Notes

The repo has now gone through two useful validation phases.

### April 2 consistency pass

This was the documentation and packaging alignment pass.

What needed to agree:

- docs say ticket routing, not email processing
- docs use the same vocabulary as the code
- `openenv.yaml`, `pyproject.toml`, and `requirements.txt` describe the same runtime surface
- Docker startup matches the documented server entry point
- local setup instructions match the current repo layout

### April 3 and April 4 runtime-feedback pass

The first local runtime pass was then completed and surfaced a practical issue:

- `data/dataset.json` was saved with a UTF-8 BOM, which caused `json.load()` to fail during environment creation on Windows

That issue is now handled in `server/tasks.py` by loading the dataset with `utf-8-sig`.

The local heuristic baseline completed successfully after that fix with:

- Task 1: `1.0000`
- Task 2: `0.8800`
- Task 3: `0.9400`
- Overall: `0.9400`

A merged-state rerun on the current `main` branch matched those same numbers exactly.

## April 6 Repo Audit

An April 6 documentation and repo audit confirmed:

- all required runtime, data, metadata, and documentation files are present in the workspace
- the docs consistently describe IT helpdesk ticket routing rather than the old email-triage domain
- the current local benchmark reference is `1.0000`, `0.8800`, `0.9400`, overall `0.9400`
- the remaining work is execution validation, not documentation cleanup

## What Still Needs Hands-On Verification

The biggest remaining checks are packaging and clean-machine checks, not merge-state local execution.

Still pending:

1. confirm Docker starts cleanly
2. do a clean-machine dry run if possible

## One-Minute Summary

If you come back to this repo later, remember:

- the domain is IT helpdesk ticket routing
- the environment is a short queue, not a single-shot classifier
- the agent predicts structured routing fields
- grading is deterministic with limited partial credit
- the inference script is the baseline player
- merged-state local validation is complete, and Docker is the main remaining hands-on check
