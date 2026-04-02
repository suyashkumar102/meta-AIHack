# IT Helpdesk Ticket Routing OpenEnv - Project Plan

## Project Goal

Build a polished OpenEnv environment for IT helpdesk ticket routing that satisfies:

- real-world utility
- strong task and grader quality
- clean environment design
- OpenEnv spec compliance
- reproducible baseline inference
- Docker and Hugging Face deployment readiness

## Current Product Definition

The environment simulates a helpdesk queue. An agent receives one ticket at a time and predicts:

- `issue_type`
- `priority`
- `assignment_group`
- `resolution_action`

The project keeps three tasks:

1. Issue Type Classification
2. Issue Type And Priority
3. Full Ticket Routing

## What Must Be True At Submission

### Pass or fail requirements

- the environment responds correctly
- OpenEnv metadata is valid
- `reset()`, `step()`, and `state()` work
- there are at least 3 tasks
- graders return scores in `[0.0, 1.0]`
- `inference.py` runs and prints reproducible results
- Docker builds and starts cleanly

### Scored requirements

- the task should clearly feel like real helpdesk work
- the hard task should require meaningful reasoning
- partial credit should be useful and deterministic
- docs should be clear enough for judges to understand quickly

## Core Files

### Runtime

- `models.py`
- `server/environment.py`
- `server/grader.py`
- `server/reward.py`
- `server/tasks.py`
- `server/app.py`
- `client.py`
- `inference.py`

### Data and metadata

- `data/dataset.json`
- `openenv.yaml`
- `server/Dockerfile`
- `pyproject.toml`
- `requirements.txt`

### Docs

- `README.md`
- `KNOWLEDGE.md`
- `MENTAL_MODEL.md`

## Technical Priorities

### P0

1. keep the environment behavior correct
2. verify the task definitions and graders
3. make the baseline script reliable
4. confirm dataset coverage and label consistency

### P1

1. validate Docker
2. validate deployment assumptions
3. record baseline scores
4. polish docs

### P2

1. strengthen ticket wording for realism
2. expand hard-case examples if needed
3. remove low-signal artifacts from the repo

## Quality Checks To Perform

### Environment

- reset starts a clean episode
- each step advances the queue correctly
- the final step returns trajectory reward
- state reflects the real internal status

### Grader

- exact matches score `1.0`
- near misses get partial credit where intended
- unsupported task IDs fail clearly
- scores vary across examples

### Inference

- heuristic mode works without model credentials
- LLM mode reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- output is reproducible when the seed is fixed

### Docs

- no outdated domain references remain
- team and project metadata are correct
- setup and run instructions are accurate

## Risks

### Runtime risk

The first local execution pass and a merged-state rerun have already completed successfully. The remaining runtime risk is Docker and clean-machine behavior, not first-pass local execution.

### Benchmark risk

The current merged-state local benchmark has already been recorded. The remaining benchmark risk is making sure Docker or clean-machine validation does not surface a late behavioral mismatch.

### Deployment risk

Docker and Hugging Face behavior should be validated before the final submission window.

## Definition Of Done

The project is ready when:

1. the environment runs locally end to end
2. the heuristic baseline runs successfully
3. Docker build and run both succeed
4. the docs are clean, current, and submission-ready
5. the repo clearly presents Hackstreet Boys as the team
