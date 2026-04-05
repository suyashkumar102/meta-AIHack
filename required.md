# Round 1 Requirements And Project Compliance Plan

## Official Problem Statement

Round 1 requires building a complete, real-world OpenEnv environment that an AI agent can learn from through the standard `step()` / `reset()` / `state()` API.

### Key requirements at a glance

- must simulate a real-world task, not a game or toy
- must implement the full OpenEnv spec with typed models and `openenv.yaml`
- must include at least 3 tasks with agent graders spanning easy -> medium -> hard
- graders must return scores in `[0.0, 1.0]`
- reward must provide meaningful partial-progress signal
- must include a reproducible baseline `inference.py`
- must deploy to Hugging Face Spaces with a working Dockerfile
- README must include environment description, action / observation spaces, setup, usage, and baseline scores

## Official Functional Requirements

### Real-world task simulation

The environment must simulate a task humans actually do. The official examples include:

- email triage
- code review
- data cleaning
- scheduling
- customer support
- content moderation

### OpenEnv spec compliance

The environment must implement the OpenEnv interface with:

- typed Observation model
- typed Action model
- typed state model
- `step(action)`
- `reset()`
- `state()`
- `openenv.yaml`

This is expected to be checked through `openenv validate`.

### Minimum 3 tasks with agent graders

Each task must have:

- a concrete objective
- a programmatic grader
- score output in `[0.0, 1.0]`
- deterministic success / failure criteria
- clear difficulty progression from easy to hard

### Meaningful reward function

The reward should:

- provide signal across the full trajectory
- reward partial progress
- penalize clearly undesirable behavior

### Baseline inference script

The baseline must:

- use the OpenAI client for LLM calls
- live at the project root as `inference.py`
- produce reproducible scores
- complete successfully across all 3 tasks

## Official Non-Functional Requirements

### Hugging Face Spaces

- must deploy as a containerized HF Space
- should be tagged with `openenv`
- should respond successfully when pinged

### Containerized execution

- must include a working Dockerfile
- should start cleanly with `docker build` + `docker run`

### Documentation

README must include:

- environment description and motivation
- action space definition
- observation space definition
- task descriptions with difficulty expectations
- setup and usage instructions
- baseline scores

## Official Evaluation Criteria

### Weights

| Parameter | Weight | What judges look for |
|-----------|--------|----------------------|
| Real-world utility | 30% | Genuine practical task and value |
| Task & grader quality | 25% | Clear objectives, fair graders, real progression |
| Environment design | 20% | Clean state, sensible API, good reward shaping |
| Code quality & spec compliance | 15% | OpenEnv compliance, structure, typing, tests, Docker |
| Creativity & novelty | 10% | Original domain, mechanics, reward ideas |

### Phase 1: Automated validation

Pass / fail gate:

- HF Space deploys
- OpenEnv spec compliance
- Dockerfile builds
- baseline reproduces
- 3+ tasks with graders

### Phase 2: Agentic evaluation

Scored:

- baseline agent rerun
- standard Open LLM agent run against the environment
- score variance check

### Phase 3: Human review

Top submissions are reviewed by Meta and Hugging Face engineers for:

- real-world utility
- creativity
- exploit resistance

## Official Disqualification Criteria

- environment does not deploy or respond
- plagiarized or trivially modified existing environment
- graders always return the same score
- no baseline inference script

## Official Pre-Submission Checklist

All of these must pass:

- HF Space deploys and responds
- automated ping to the Space URL returns `200`
- reset path works on the deployed environment
- `openenv validate` passes
- Dockerfile builds
- baseline inference completes and produces scores
- 3+ tasks with graders are present and score in `[0.0, 1.0]`

## Mandatory Additional Instructions

### Required inference environment variables

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

The official text also mentions `OPENAI_API_KEY` in one place, but the more specific submission instructions above consistently emphasize `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`. We should follow the later, more specific instruction while continuing to use the OpenAI client.

### Inference script constraints

- script must be named `inference.py`
- it must live in the project root
- all LLM calls must use the OpenAI client
- stdout logs must strictly follow the `[START]`, `[STEP]`, and `[END]` format from the official sample

### Infra restrictions

- inference runtime should stay under 20 minutes
- env and inference should run on a machine with `vcpu=2` and `memory=8gb`

### Validator

- run the official pre-submission validation script before final submission if possible

---

## Project Compliance Plan

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

### Pass / fail requirements

- the environment responds correctly
- OpenEnv metadata is valid
- `reset()`, `step()`, and `state()` work
- there are at least 3 tasks
- graders return scores in `[0.0, 1.0]`
- `inference.py` runs and prints reproducible results
- `inference.py` uses the OpenAI client and required env vars
- structured stdout logging matches the official format
- `openenv validate` passes
- Docker builds and starts cleanly
- HF Space responds and reset works

### Scored requirements

- the task clearly feels like real helpdesk work
- the hard task requires meaningful reasoning
- partial credit is useful and deterministic
- docs are clear enough for judges to understand quickly
- reward is informative over the trajectory, not only at the end

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
- `required.md`

## Technical Priorities

### P0

1. keep environment behavior correct
2. verify task definitions and graders
3. make the baseline script reliable and compliant with official logging format
4. confirm dataset coverage and label consistency
5. validate the official submission gates, not just local behavior

### P1

1. validate Docker
2. validate deployment assumptions
3. record baseline scores
4. polish docs
5. verify the runtime envelope and structured inference logs

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
- episode boundaries are sensible

### Grader

- exact matches score `1.0`
- near misses get partial credit where intended
- unsupported task IDs fail clearly
- scores vary across examples
- graders do not collapse to constant scores

### Inference

- heuristic mode works without model credentials
- LLM mode reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- uses the OpenAI client
- stdout follows `[START]`, `[STEP]`, and `[END]`
- output is reproducible when the seed is fixed
- runtime stays below the official time budget

### Deployment and validation

- `openenv validate` passes
- Docker build succeeds
- Docker run succeeds
- HF ping / reset behavior works
- official validator script is run if practical

### Docs

- no outdated domain references remain
- team and project metadata are correct
- setup and run instructions are accurate
- README reflects the current inference and deployment path

## Risks

### Runtime risk

The first local execution pass, merged-state rerun, clean-copy rerun, and local validator pass have already succeeded. The remaining runtime risk is submission-day deployment execution, not first-pass local behavior.

### Benchmark risk

The current local benchmark is already recorded. Remaining benchmark risk is whether deployment / validation changes expose a mismatch late.

### Deployment risk

Docker smoke coverage, `openenv validate`, and structured inference logging are now verified in the repo state. The remaining deployment risk is the live Hugging Face Space ping and reset check after the final push if a fresh deployment is created.

## Definition Of Done

The project is ready when:

1. the environment runs locally end to end
2. unit, smoke, and integration tests cover the critical paths
3. the heuristic baseline runs successfully
4. the inference script is compliant with the official logging format
5. `openenv validate` passes
6. Docker build and run both succeed
7. HF deployment checks succeed or are as close to verified as possible before submission
8. the docs are clean, current, and submission-ready
9. the repo clearly presents Hackstreet Boys as the team

## Current Compliance Snapshot

As of April 7, 2026, the roadmap gates through the end of the freeze window are in place:

- real-world task definition is clear and stable
- typed models, `reset()`, `step()`, `state()`, and `openenv.yaml` are present in the repo
- 3-task easy -> medium -> hard ladder is present
- graders are deterministic and bounded to `[0.0, 1.0]`
- unit tests now prove scorer crispness, task invariants, and dataset coverage
- smoke tests now prove environment behavior, seeded determinism, score bounds, and full-episode completion
- integration tests now cover `/health`, `/tasks`, `/reset`, `/step`, `/state`, full seeded episodes, and heuristic regression
- baseline heuristic results are recorded in the docs
- the README now includes Hugging Face Spaces frontmatter and a judge-facing grounded-scoring explanation
- an internal grounding audit exists in `analysis/grounding_audit.md`
- `.openenvignore` is present
- Docker smoke coverage exists through the checked-in GitHub Actions workflow and recorded April 6 run
- `inference.py` structured `[START]`, `[STEP]`, and `[END]` logging is verified
- `uv.lock` is checked in and `openenv validate` now passes on the current repo state
- a clean-copy install-and-run pass has been completed

The remaining April 8 work is operational rather than implementation-heavy:

- Hugging Face deployment ping and reset verification
- the final submission-branch sanity rerun before push if any last-minute packaging-only change lands

The roadmap's short TRL / GRPO README example remains optional and is still deferred because it is not required for submission readiness.
