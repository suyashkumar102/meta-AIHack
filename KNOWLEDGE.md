# IT Helpdesk Ticket Routing OpenEnv - Knowledge Guide

## What This Repo Needs To Prove

The judges want a real-world environment that follows the OpenEnv pattern and can be understood quickly.

That means this repo needs:

1. typed action, observation, and state models
2. working `reset()`, `step()`, and `state()`
3. at least three difficulty levels
4. deterministic grading
5. meaningful reward shaping
6. a baseline `inference.py`
7. Docker and metadata that are easy to rerun

## Why This Domain Fits

IT helpdesk routing is a strong hackathon fit because it is:

- realistic
- structured
- judge-friendly
- deterministic to grade
- naturally multi-step

A helpdesk agent has to decide what the ticket is about, how urgent it is, who should own it, and what should happen next. The current runtime now supports a small two-mode action object: investigate first when needed, then submit the final routing answer.

## The Repo In One Sentence

This environment simulates a short helpdesk queue where an agent routes one ticket at a time and is graded on structured routing quality.

## Judge-Facing Explanation

If a judge asks why this environment is strong, the concise answer is:

1. IT helpdesk routing is a real operational workflow with clear business value.
2. The input is realistic free-form ticket text, but the output is typed and easy to grade deterministically.
3. The three-task ladder creates a clean progression from basic classification to full queue routing.
4. The repo stays judge-friendly because the vocabulary, task labels, and scoring rules are explicit and frozen.

## Frozen Project Identity

- Team name: `Hackstreet Boys`
- Members: `Roopal Guha Neogi`, `Suyash Kumar`
- Domain: `IT Helpdesk Ticket Routing`
- OpenEnv name: `it_helpdesk_ticket_routing_openenv`
- App environment name: `it_helpdesk_ticket_routing`

## Practical Mental Model

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

The repo is a small OpenEnv stack:

- `inference.py` drives episodes
- `client.py` talks to the app
- `server/environment.py` manages queue state and episode flow
- `server/grader.py` scores actions
- `server/reward.py` computes step and final reward behavior
- `server/tasks.py` defines the task ladder and loads the dataset
- `data/dataset.json` stores the labeled helpdesk tickets

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

Represents the agent step. `action_type="submit"` carries routing fields, while `action_type="investigate"` uses a small built-in tool surface before the final submission.

### `HelpdeskTicketObservation`

Represents what the agent sees for each step:

- task metadata
- visible ticket fields
- optional ambiguity or follow-up context
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

## Observation And State At A Glance

The observation exposes:

- task metadata
- the current ticket
- available investigation tools
- remaining free investigation budget
- the latest tool result, when one was requested
- queue progress counters
- history
- reward and done status

Useful queue counters now include:

- `tickets_remaining`: not-yet-processed tickets, including the current ticket when one is active
- `tickets_after_current`: how many tickets remain after the current one
- `queue_position`: 1-based position of the current ticket in the queue

The state tracks:

- current task
- seed
- queue ticket IDs
- current ticket index
- per-ticket scores
- total reward
- investigation step count

## Task Design

### Task 1: Issue Type Classification

The agent ultimately predicts:

- `issue_type`

Purpose:

- establish the simplest classification baseline

### Task 2: Issue Type And Priority

The agent ultimately predicts:

- `issue_type`
- `priority`

Purpose:

- force the agent to understand both topic and urgency

### Task 3: Full Ticket Routing

The agent ultimately predicts:

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

Just as important, the grader is not fuzzy by default:

- exact matches stay dominant
- wrong issue types outside the declared similarity map score `0.0`
- wrong priorities outside the declared proximity table score `0.0`
- assignment group and resolution action never receive partial credit

Task weighting:

- Task 1: only `issue_type`
- Task 2: `issue_type` 60%, `priority` 40%
- Task 3: `issue_type` 35%, `priority` 20%, `assignment_group` 25%, `resolution_action` 20%

This is now proven in checked-in unit tests rather than left as a docs claim.

## Reward Mental Model

Step reward:

- current ticket score with a small milestone bonus for strong steps and a small penalty for very weak steps

Final reward:

- average of ticket scores
- minus a tiny penalty only if the agent exceeds the free investigation budget for the queue

This keeps the reward dense and deterministic, removes the dead overshoot logic, and adds a small queue-level economics signal without disturbing the no-tool baseline path.

## Dataset Mental Model

The dataset is small enough to audit manually but varied enough to support a meaningful benchmark.

Current structure:

- 45 tickets
- clear easy examples
- medium cases where urgency matters
- harder ambiguous cases
- follow-up tickets connected through `related_ticket_id`

When a follow-up link exists, the observation can now surface a lightweight `related_ticket_preview`, and the tool layer can fetch richer related-ticket or requester-history context so the agent does not have to route every ticket from isolated text alone.

The dataset is meant to test routing judgment, not just keyword spotting.

## Grounding Note

The taxonomy and limited partial-credit policy were reviewed against public IT-support references recorded in `analysis/grounding_audit.md`.

The grounding inputs used for that review were:

- `Classification of IT Support Tickets`
- `Semantic Similarity of IT Support Tickets`
- `MSDialog`

The key conclusion was to keep the similarity map narrow. The current issue-type near misses are defensible, but broader additions would blur operationally distinct routing actions too much this late in the submission cycle.

## Inference Script In Simple Terms

`inference.py` is the baseline agent runner.

It:

1. connects to the environment
2. loads the available tasks
3. runs one episode for the requested task
4. picks an action for each ticket
5. sends the action back through the client
6. records rewards
7. prints structured logs for that run

It supports:

- heuristic mode with no external model
- LLM mode through an OpenAI-compatible API
- lightweight investigation-tool calls before the final submit action
- an explicit local `RUN_ALL_TASKS=1` override when you want the old multi-task sweep

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

The repo has already gone through two useful validation phases.

### April 2 consistency pass

This was the documentation and packaging alignment pass.

What needed to agree:

- docs say ticket routing, not email processing
- docs use the same vocabulary as the code
- `openenv.yaml`, `pyproject.toml`, and `requirements.txt` describe the same runtime surface
- Docker startup matches the documented server entry point
- local setup instructions match the current repo layout

### April 3 and April 4 runtime-feedback pass

The first local runtime pass surfaced one practical issue:

- `data/dataset.json` was saved with a UTF-8 BOM, which caused `json.load()` to fail during environment creation on Windows

That issue is now handled in `server/tasks.py` by loading the dataset with `utf-8-sig`.

The local heuristic baseline completed successfully after that fix with:

- Task 1: `1.0000`
- Task 2: `0.8800`
- Task 3: `0.9400`
- Overall: `0.9400`

A merged-state rerun on the current `main` branch matched those same numbers exactly.

### April 6 repo audit

An April 6 audit confirmed:

- all required runtime, data, metadata, and documentation files are present
- the docs consistently describe IT helpdesk ticket routing rather than the old email-triage domain
- the current local benchmark reference is still `1.0000`, `0.8800`, `0.9400`, overall `0.9400`
- the remaining work is execution validation, not documentation cleanup

### April 6 and April 7 Roopal-side doc pass

That follow-up pass added the remaining Roopal-owned public-clarity items:

- Hugging Face Spaces README frontmatter
- explicit judge-facing explanation that scoring is deterministic and only partially fuzzy in declared places
- an internal grounding note tying the label space to public IT-support datasets
- a refreshed compliance snapshot in `required.md`

The optional TRL / GRPO README example remains intentionally deferred because it is optional and lower priority than freeze-phase stability.

## April 3-7 Status

The roadmap through April 7 is now closed in the current repo state.

That means the repo now has:

1. checked-in unit, smoke, and integration tests
2. Docker smoke coverage through the GitHub Actions workflow
3. a clean-copy install-and-run pass
4. structured `inference.py` logging verification
5. a passing local `openenv validate` result after checking in `uv.lock`

## Submission-Day Reminders

The remaining work belongs to the April 8 submission window rather than the April 3 to April 7 implementation window:

1. rerun the final sanity slice on the submission branch
2. verify the live Hugging Face Space ping and reset path after the final push if a fresh deployment is created

## One-Minute Summary

If you come back to this repo later, remember:

- the domain is IT helpdesk ticket routing
- the environment is a short queue, not a single-shot classifier
- the architecture is a compact OpenEnv stack
- one ticket is shown at a time
- the agent predicts structured routing fields
- the grader gives deterministic partial credit
- `inference.py` is the baseline agent runner
- merged-state validation, Docker smoke coverage, clean-copy rerun, and local validator readiness are all now in place
