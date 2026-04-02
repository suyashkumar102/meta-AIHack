# Competition Knowledge Base — OpenEnv Hackathon

> Source: github.com/meta-pytorch/OpenEnv/tree/main/envs  
> Gathered: April 4, 2026  
> Purpose: Internal competitive intelligence — NOT for commit/push

---

## Full Environment Inventory (27 envs)

| Env | Domain | Complexity | Reward Type | Multi-step? | MCP? |
|-----|--------|------------|-------------|-------------|------|
| `atari_env` | Classic games | Medium | Dense | Yes | No |
| `browsergym_env` | Web browser automation | Very High | Task-based | Yes | No |
| `calendar_env` | Calendar/scheduling agent | High | SQL verifier | Yes | Yes (MCP) |
| `carla_env` | Autonomous driving sim | Very High | Dense | Yes | No |
| `chat_env` | Conversation/tokenization | Low | Custom transform | Yes | No |
| `chess_env` | Chess game | Medium | Win/loss | Yes | No |
| `coding_env` | Python code execution | Medium | Exit code / transform | Yes | No |
| `connect4_env` | Connect 4 game | Low | Win/loss | Yes | No |
| `dipg_safety_env` | Safety/policy | Medium | Unknown | Yes | No |
| `dm_control_env` | DeepMind Control Suite | High | Dense | Yes | No |
| `echo_env` | Reference/minimal | Minimal | Echo | No | No |
| `finqa_env` | Financial QA (SEC 10-K) | High | Fuzzy numerical | Yes | Yes (MCP) |
| `finrl_env` | Financial RL trading | High | Portfolio return | Yes | No |
| `git_env` | Git operations | Medium | Task-based | Yes | No |
| `grid_world_env` | Grid navigation | Low | Sparse | Yes | No |
| `julia_env` | Julia code execution | Medium | Exit code | Yes | No |
| `kernrl` | Kernel/OS operations | High | Unknown | Yes | No |
| `maze_env` | Maze navigation | Low | Sparse | Yes | No |
| `openapp_env` | Web app UI (BrowserGym) | Extreme | Task-based | Yes | No |
| `openspiel_env` | Multi-agent games | High | Game outcome | Yes | No |
| `reasoning_gym_env` | Reasoning tasks (100+ datasets) | Medium | Exact/partial | Single-step | No |
| `repl_env` | REPL execution | Medium | Exit code | Yes | No |
| `snake_env` | Snake game | Low | Score | Yes | No |
| `sumo_rl_env` | Traffic simulation | High | Traffic flow | Yes | No |
| `tbench2_env` | Terminal Bench 2 (shell tasks) | High | pytest pass/fail | Yes | No |
| `textarena_env` | Text-based games | Medium | Game outcome | Yes | No |
| `unity_env` | Unity 3D simulation | Very High | Task-based | Yes | No |

---

## Deep Dives: Most Relevant Envs

### 1. `finqa_env` — Financial QA

**What it does**: Agents answer complex financial questions from SEC 10-K filings using SQL tool calls.

**Architecture**:
- Subclasses `MCPEnvironment` (not plain `Environment`) — uses FastMCP with `@mcp.tool` decorators
- Tools: `get_descriptions`, `get_table_info`, `sql_query`, `submit_answer`
- Dataset: 290 questions from HuggingFace (`snorkelai/finqa-data`)
- Max steps: 50 per episode
- Reward: Binary (1.0 / 0.0) with fuzzy numerical matching (1% relative tolerance + 1.0 absolute tolerance)
- Handles `\boxed{}` LaTeX format, percentages, fractions, thousands separators, negative parens

**Reward sophistication**: Very high. The `rewards.py` is ~300 lines handling multi-value answers, year-labeled pairs, percentage normalization, and both relative + absolute tolerance checks simultaneously.

**Key differentiator**: MCP protocol for tool discovery. Client uses `await env.list_tools()` to discover tools at runtime. This is the most "agentic" env in the repo.

**Integration**: Explicitly shows TRL/GRPO integration pattern in README.

---

### 2. `coding_env` — Python Code Execution

**What it does**: Executes arbitrary Python code in a sandboxed environment.

**Architecture**:
- `PythonCodeActEnv` wraps a `PyExecutor` (sandboxed subprocess)
- `create_safe_coding_transform()` — transform pipeline for reward computation
- Action: `CodeAction(code: str)`
- Observation: `CodeObservation(stdout, stderr, exit_code)`
- State: `CodeState(episode_id, step_count, last_exit_code)`
- Reward: computed by transform (not in step directly) — extensible pattern

**Key differentiator**: Transform-based reward. The environment itself doesn't compute reward — a pluggable `Transform` object does. This is the cleanest separation of concerns in the repo.

**Testing**: Has both unit tests (`test_python_codeact_reset`, `test_python_codeact_rewards`) and integration tests (`test_coding_env_integration`). Most tested env in the repo.

---

### 3. `reasoning_gym_env` — Reasoning Tasks

**What it does**: Wraps the `reasoning-gym` library (100+ reasoning datasets) as a single-step OpenEnv.

**Architecture**:
- Single-step episodes: `reset()` gives question, `step()` gives score + done=True
- Composite datasets: mix multiple datasets with weights
- Dataset persistence: same dataset reused across resets until config changes
- Supports `dataset_name`, `seed`, `size`, `dataset_specs` in `reset()` kwargs
- Reward: 0.0–1.0 (dataset-dependent, may use partial credit)

**Key differentiator**: Massive breadth (100+ task types in one env). The `reset()` kwargs pattern for dataset configuration is very clean. Also has `openenv push` CLI for HuggingFace Spaces deployment.

**Scale**: uv.lock is 551KB — large dependency tree from reasoning-gym.

---

### 4. `tbench2_env` — Terminal Bench 2

**What it does**: Wraps Terminal-Bench-2 shell tasks. Agent executes shell commands and is evaluated by pytest.

**Architecture**:
- Two modes: `local` (direct process) and `docker` (per-task container)
- Rich action type: `exec`, `write`, `view`, `wait`, `kill`, `write_file`, `evaluate`, `close`
- Session IDs for streaming/non-blocking processes
- Reward: Binary (pytest pass/fail) on `evaluate` action
- Intermediate steps: `reward=None`

**Key differentiator**: Most realistic "agentic" shell environment. The session ID pattern for streaming processes is unique. Docker-in-Docker mode for full fidelity.

---

### 5. `openapp_env` — Web App UI

**What it does**: Wraps OpenApps (calendar, todo, messenger, maps) + BrowserGym for browser-based UI agent training.

**Architecture**:
- Runs TWO services in Docker: OpenApps server (port 5001) + FastAPI (port 8000)
- `start.sh` orchestrates both
- BrowserGym for browser automation (Playwright/Chromium)
- Docker image: ~5.7GB (includes Chromium)
- Multimodal: screenshots + DOM observations

**Key differentiator**: Most complex env in the repo. Multimodal (visual + text). Real browser interaction. Closest to real-world agent deployment.

---

### 6. `calendar_env` — Calendar Scheduling

**What it does**: Calendar management tasks with SQL database verification.

**Architecture**:
- MCP-based (like finqa_env)
- Has `client_notebooks/` — Jupyter notebook for interactive evaluation
- Has `mcp_databases/` — SQLite databases for state
- Scenario-based: `scenario_config.json` drives task + verifiers
- Verifiers: SQL queries that check task completion
- Supports OpenAI, Anthropic, Google providers

**Key differentiator**: Scenario config pattern. Verifier-based reward (SQL queries check if the agent actually completed the task). Most "enterprise workflow" env.

---

### 7. `chat_env` — Chat/Tokenization

**What it does**: Manages conversation history + tokenization for LLM RL training.

**Architecture**:
- Action: `ChatAction(tokens: torch.Tensor)` — takes raw model tokens
- Observation: `ChatObservation(messages, tokens)` — both human-readable + model-ready
- Transform-based reward (pluggable)
- Dual representation: messages (human) + tokens (model)
- No HTTP overhead option: can use directly without server

**Key differentiator**: Designed for direct LLM RL training loop. The only env that takes raw PyTorch tensors as actions. Pairs with GRPO/PPO training loops directly.

---

## Structural Patterns Observed Across All Envs

### File Structure (canonical)
```
env_name/
├── __init__.py          # exports
├── models.py            # Action, Observation, State
├── client.py            # EnvClient subclass
├── openenv.yaml         # metadata
├── pyproject.toml       # packaging
├── README.md            # HuggingFace Space frontmatter + docs
└── server/
    ├── __init__.py
    ├── app.py           # FastAPI
    ├── environment.py   # core logic
    └── Dockerfile
```

### README Frontmatter (HuggingFace Spaces)
Every env README has YAML frontmatter:
```yaml
---
title: ...
emoji: ...
colorFrom: ...
colorTo: ...
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---
```
This is required for HuggingFace Spaces deployment. Our README does NOT have this.

### openenv.yaml — Minimal Pattern
Most envs have very minimal `openenv.yaml` (just name + entry_point). Our yaml is the most detailed in the repo.

### Dockerfile Patterns
- Most use `openenv-base:latest` as base image (not `python:3.11-slim`)
- Our Dockerfile uses `python:3.11-slim` directly — this is the standalone/HF Spaces pattern
- The `openenv-base` pattern is for the monorepo CI/CD workflow

### Testing
- `coding_env`: most tested (unit + integration)
- Most envs: no tests at all
- Our env: no tests (matches majority)

### MCP vs HTTP
- Most envs: plain HTTP (`Environment` base class)
- `finqa_env`, `calendar_env`: MCP (`MCPEnvironment` base class, FastMCP tools)
- MCP envs are more "agentic" — tools are discoverable at runtime

### Reward Patterns
| Pattern | Envs | Description |
|---------|------|-------------|
| Binary (0/1) | finqa, tbench2, reasoning_gym | Pass/fail |
| Dense partial | ours, chess, atari | Continuous [0,1] |
| Transform-based | coding, chat | Pluggable reward function |
| SQL verifier | calendar | DB state check |
| Game outcome | chess, connect4, openspiel | Win/loss/draw |

---

## Deployment Patterns

### HuggingFace Spaces
- `openenv push` CLI command (seen in reasoning_gym README)
- Spaces get: `/web` (UI), `/docs` (Swagger), `/health`, `/ws` (WebSocket)
- `base_path: /web` in README frontmatter
- Our env: missing HF Spaces frontmatter in README

### Docker
- Most envs: `openenv-base:latest` (monorepo CI)
- Standalone envs (ours, openapp): `python:3.11-slim`
- openapp: 5.7GB image (Chromium)
- Our image: minimal (python:3.11-slim + pip deps)

---

## Dataset Sizes

| Env | Dataset Size | Source |
|-----|-------------|--------|
| finqa | 290 questions | HuggingFace (snorkelai/finqa-data) |
| reasoning_gym | 100+ datasets, configurable size | reasoning-gym library |
| calendar | SQLite DBs | Custom |
| ours | 45 tickets | Custom (data/dataset.json) |
| coding | N/A (generates tasks) | N/A |
| tbench2 | Terminal-Bench-2 repo | GitHub auto-download |

---

## Key Technical Observations

1. **MCP is the emerging pattern** for tool-using agents. finqa and calendar both use it. Our env uses plain HTTP — simpler but less "agentic."

2. **Transform-based rewards** (coding_env, chat_env) are the cleanest architecture for extensible reward shaping. Our reward is hardcoded in `reward.py`.

3. **`openenv push` CLI** exists for HuggingFace Spaces deployment. We should use it.

4. **README frontmatter** is required for HF Spaces. Our README is missing it.

5. **Composite/configurable datasets** (reasoning_gym) are a strong differentiator. Our dataset is fixed at 45 tickets.

6. **WebSocket endpoint** (`/ws`) is mentioned in reasoning_gym README as a HF Spaces feature. Our env already has `/ws` via the OpenEnv base.

7. **`uv.lock`** files appear in chat_env and reasoning_gym — reproducible dependency locking. We use `requirements.txt` only.

8. **`.openenvignore`** file in finqa_env — analogous to `.dockerignore` for the OpenEnv push CLI.

9. **`base_path: /web`** in HF Spaces frontmatter — the web UI is at `/web`, not `/`. Our env would need this.

10. **Episode length**: Most envs are either single-step (reasoning_gym) or unbounded (coding, tbench2). Our env is bounded (3–5 steps) — a clean middle ground.
