# Gap Analysis — IT Helpdesk Ticket Routing OpenEnv

Deep cross-reference of the codebase against every concrete mentor statement from the bootcamp transcript and Discord Q&A.

---

## GAP 1 — CRITICAL: `inference.py` runs all 3 tasks in one invocation

**Mentor (4/1/26, 9:48 PM, confirmed twice):**
> "inference.py should execute a single task per run and emit exactly one [START] … [END] block. The evaluation system handles running across multiple tasks, so batching all tasks in one invocation is not expected."

**Your code in `inference.py`:**
```python
TASKS = list(TASK_IDS)  # [1, 2, 3]
for task_id in TASKS:   # loops all 3
    emit_log("START", ...)
    ...
    emit_log("END", ...)
emit_log("END", overall_avg=...)  # second END
```

The evaluator calls `inference.py` once per task. Your script ignores that and runs all 3 itself, emitting 3 `[START]`/`[END]` pairs. The evaluator expects exactly one. There is no `TASK_ID` env var read anywhere.

---

## GAP 2 — CRITICAL: `state()` response is missing `reward` and `done` fields

**Mentor (4/1/26, 9:33 PM):**
> "state() must return minimum: `{ 'observation': ..., 'reward': last_step_reward, 'done': True/False }`"

**Your `HelpdeskTicketState` model:**
```python
class HelpdeskTicketState(State):
    current_task_id: Optional[int] = None
    seed: Optional[int] = None
    queue_ticket_ids: list[str]
    current_ticket_index: int = 0
    per_ticket_scores: list[float]
    total_reward: float = 0.0
    # NO reward field (last step reward)
    # NO done field
```

`GET /state` returns this model directly. The evaluator checking `state()` for `reward` and `done` will find neither. `total_reward` is the accumulated reward, not the last step reward — which the mentor explicitly said NOT to return.

---

## GAP 3 — MEDIUM: `history` in observation is too sparse for RL usefulness

**Ben (YouTube bootcamp, ~00:31:07):**
> "process supervision... give these more detailed rewards... enrich history with ticket title, predicted fields"

**Your `_build_observation` history:**
```python
history.append({"step": i + 1, "score": s})
# final entry gets: {"step": N, "ticket_id": ..., "score": ..., "breakdown": ...}
```

Non-final history entries only have `step` and `score`. No ticket title, no predicted action fields. The agent cannot learn from history because it cannot see what it predicted or what the ticket was. This directly weakens RL signal quality.

---

## GAP 4 — MEDIUM: No milestone/delta reward shaping — flat score passthrough

**Mentor (4/1/26, 9:34 PM):**
> "A deterministic terminal grader with partial credit is valid, but it's better to include some intermediate (non-terminal) reward signals as well so the environment provides step-wise feedback. Milestone-based shaping is preferred over dense per-action rewards."

**Your `step()` in `environment.py`:**
```python
if is_done:
    final_reward = traj_reward   # trajectory reward only at end
else:
    final_reward = step_reward   # per-ticket score for non-final steps
```

You do return `step_reward` on non-final steps, which is correct. But `step_reward` is just `compute_step_reward(score)` which is `max(0.0, min(1.0, score))` — identical to the raw score. There is no shaping, no milestone signal, no delta-based signal. This is a quality gap, not a blocker.

---

## GAP 5 — MEDIUM: `observation.history` doesn't include the predicted action

**Your `_build_observation`:**
```python
history_entry = {
    "ticket_id": current_ticket.ticket_id,
    "score": score,
    "breakdown": breakdown,
}
```

The agent's own predicted action is never stored in history. When the agent looks at history to decide its next action, it cannot see what it previously predicted. This is a real RL signal gap — the agent has no memory of its own decisions.

---

## GAP 6 — LOW: `tickets_remaining` semantics slightly ambiguous

**Your `_build_observation`:**
```python
tickets_remaining=max(0, queue_size - idx),
```

`idx` is `current_ticket_index` which has already been incremented by `step()` before `_build_observation` is called. During the episode, `tickets_remaining` counts the current ticket as "remaining" even though it is being processed. Minor but could confuse an LLM agent reading the observation.

---

## GAP 7 — LOW: `openenv.yaml` `entry_point` vs `pyproject.toml` `server` script mismatch

**Mentor (3/31/26, 11:27 PM):**
> "The validator is checking for a specific callable entrypoint. In some setups, it expects a main() function instead of an app object."

**Your `pyproject.toml`:**
```toml
[project.scripts]
server = "server.app:main"
```

**Your `openenv.yaml`:**
```yaml
entry_point: server.environment:HelpdeskTicketRoutingEnvironment
```

These point to different things. The validator may check `entry_point` in `openenv.yaml` and expect it to match `[project.scripts] server`. This inconsistency could cause validation confusion.

---

## GAP 8 — LOW: No `/web` UI endpoint — blank HF Space page

**Ben (YouTube, ~00:45:08):**
> "They're small apps and they're based as spaces. So they're deployed with a UI and an API."

The echo env example had `/web` for the UI. Your app has no `/web` route. The mentor said UI is optional and not scored, but the HF Space will show a blank page with no UI, which looks unpolished to judges doing Phase 3 human review.

---

## Summary

| # | Gap | Severity | File(s) |
|---|-----|----------|---------|
| 1 | `inference.py` runs all 3 tasks, evaluator expects 1 per run | CRITICAL | `inference.py` |
| 2 | `GET /state` missing `reward` (last step) and `done` fields | CRITICAL | `models.py`, `environment.py` |
| 3 | `history` missing predicted action — agent has no memory of decisions | MEDIUM | `environment.py` |
| 4 | No milestone/delta reward shaping — flat score passthrough | MEDIUM | `reward.py` |
| 5 | `history` non-final entries missing ticket title | MEDIUM | `environment.py` |
| 6 | `tickets_remaining` semantics slightly ambiguous | LOW | `environment.py` |
| 7 | `openenv.yaml` `entry_point` vs `pyproject.toml` `server` script mismatch | LOW | `openenv.yaml`, `pyproject.toml` |
| 8 | No `/web` UI — blank HF Space page | LOW | `server/app.py` |
