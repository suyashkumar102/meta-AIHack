# Competition Knowledge Base And Action Plan

> Source: github.com/meta-pytorch/OpenEnv/tree/main/envs
> Gathered: April 4, 2026
> Purpose: Internal competitive intelligence plus action planning - NOT for commit/push

---

## Full Environment Inventory

| Env | Domain | Complexity | Reward Type | Multi-step? | MCP? |
|-----|--------|------------|-------------|-------------|------|
| `atari_env` | Classic games | Medium | Dense | Yes | No |
| `browsergym_env` | Web browser automation | Very High | Task-based | Yes | No |
| `calendar_env` | Calendar / scheduling agent | High | SQL verifier | Yes | Yes |
| `carla_env` | Autonomous driving sim | Very High | Dense | Yes | No |
| `chat_env` | Conversation / tokenization | Low | Custom transform | Yes | No |
| `coding_env` | Python code execution | Medium | Exit code / transform | Yes | No |
| `echo_env` | Reference / minimal | Minimal | Echo | No | No |
| `finqa_env` | Financial QA | High | Fuzzy numerical | Yes | Yes |
| `openapp_env` | Web app UI | Extreme | Task-based | Yes | No |
| `reasoning_gym_env` | Reasoning tasks | Medium | Exact / partial | Single-step | No |
| `tbench2_env` | Terminal tasks | High | Pytest pass/fail | Yes | No |

This is not the full raw repo dump anymore. It is the subset that matters most for competitive positioning and late-stage prioritization.

---

## Most Relevant Competitor Patterns

### `finqa_env`

- strong MCP / tool-using architecture
- larger dataset than ours
- binary-style reward with fuzzy numerical matching
- explicit TRL / GRPO integration story

### `coding_env`

- strongest test story
- clean transform-based reward separation
- reference example of strong code quality and architecture hygiene

### `reasoning_gym_env`

- broadest dataset coverage
- configurable dataset / size pattern
- useful deployment references for `openenv push`

### `tbench2_env`

- strong agentic shell-task realism
- binary evaluation via pytest
- little intermediate reward signal

### `openapp_env`

- highest complexity
- multimodal / browser-based
- difficult to beat on ambition, easier to beat on simplicity and reproducibility

### `calendar_env`

- enterprise workflow flavor
- scenario + verifier pattern
- stronger on MCP sophistication than on reward density

---

## Structural Patterns Across The Field

### Packaging

- every serious repo has `models.py`, `client.py`, `openenv.yaml`, `pyproject.toml`, `README.md`, and a `server/` package
- Hugging Face Spaces frontmatter is standard in competitor `README.md` files
- `.openenvignore` appears in some stronger submissions

### Reward patterns

| Pattern | Examples | Notes |
|---------|----------|-------|
| Binary | `finqa_env`, `tbench2_env` | easy to verify, weaker RL signal |
| Dense partial | ours, games | stronger RL learning signal |
| Transform-based | `coding_env`, `chat_env` | architecturally clean |
| SQL / verifier based | `calendar_env` | strong task verification |

### Testing patterns

- many repos have little or no tests
- `coding_env` is still the strongest example of checked-in testing
- this makes tests a high-value differentiator for us

### Deployment patterns

- Spaces usually expose `/web`, `/docs`, `/health`, and `/ws`
- `openenv push` is the expected deployment workflow
- `README` frontmatter and Docker correctness matter more than polish extras

---

## Key Technical Observations

1. MCP is useful, but too big to add late.
2. Transform-based reward is elegant, but not a deadline-critical refactor.
3. HF Spaces frontmatter is expected and missing in our repo.
4. `.openenvignore` is a cheap packaging win.
5. Configurable datasets are nice, but external dataset merge is too risky late.
6. Strong tests improve trust more than minor architectural polish.
7. Dense, deterministic, partial-credit reward is one of our real advantages.

---

## Actionable Inferences

## Critical Missing Items

### 1. README frontmatter for HF Spaces

This is still the cleanest obvious gap. Add it before submission.

Recommended fields:

```yaml
---
title: IT Helpdesk Ticket Routing OpenEnv
emoji: "ticket"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - helpdesk
  - ticket-routing
  - nlp
---
```

### 2. `.openenvignore`

Cheap packaging improvement. Worth adding.

### 3. Verified deployment assumptions

We should explicitly verify:

- `app_port: 7860`
- `/health`
- `/docs`
- `/ws`
- `/web`

---

## High-Value Improvements That Still Make Sense

### 4. Strengthen the scorer only in grounded, tested ways

Possible additions to `ISSUE_TYPE_SIMILARITY`:

- `onboarding` vs `service_request`
- `feature_request` vs `service_request`
- `security_compliance` vs `identity_access`
- `billing_license` vs `identity_access`

Only do this if:

- the ambiguity is real
- the change is backed by tests
- it does not blur operationally distinct actions too much

### 5. Add richer `history` if low-risk

Candidate additions:

- ticket title
- predicted fields

This can help multi-step reasoning without changing the core task.

### 6. Add `queue_size` as an optional `reset()` kwarg

Nice RL/training flexibility, but lower priority than tests, scorer crispness, Docker, and deployment readiness.

### 7. Add a short TRL / GRPO example to README

Good judge-facing signal once the repo is already green.

---

## Improvements To Defer

- MCP migration
- transform-based reward refactor
- major dataset expansion
- external dataset merge into runtime
- broad inference rewrite
- dependency churn just for polish

---

## Competitive Positioning

### Our strengths

1. strong real-world enterprise domain
2. dense deterministic reward
3. partial-credit grading that is still explainable
4. clean 3-task difficulty ladder
5. strong heuristic baseline
6. compact, rerunnable environment design

### Our weaknesses

1. weaker checked-in test story unless we fix it
2. missing HF Spaces frontmatter unless we fix it
3. smaller dataset than some top competitors
4. less ambitious architecture than the strongest simulator-style or MCP-heavy entries

---

## Priority Action List

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| P0 | Add tests and prove scorer crispness | 1-2 hrs | High |
| P0 | Add HF Spaces frontmatter to README | 5 min | High |
| P0 | Add `.openenvignore` | 5 min | Medium |
| P1 | Add grounding audit against public support datasets | 1-2 hrs | High |
| P1 | Expand similarity pairs only if grounded and tested | 20-40 min | Medium |
| P1 | Add richer `history` if low-risk | 20 min | Medium |
| P1 | Add TRL / GRPO README example | 30 min | High |
| P2 | Add `queue_size` kwarg | 15 min | Low |
| P3 | Expand dataset substantially | 2+ hrs | Medium but risky |
| P3 | Transform-based reward refactor | 1 hr | Low |
