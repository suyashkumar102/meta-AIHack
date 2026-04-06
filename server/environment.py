from __future__ import annotations

import random
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from models import (
    HelpdeskTicketAction,
    HelpdeskTicketObservation,
    HelpdeskTicketRecord,
    HelpdeskTicketState,
)
from server.grader import grade_action
from server.reward import compute_step_reward, compute_trajectory_reward
from server.tasks import get_task_definition, load_dataset


QUEUE_SIZE_RANGE = (3, 5)
AVAILABLE_TOOLS = ("lookup_related_ticket", "lookup_requester_history")
FREE_INVESTIGATIONS_PER_TICKET = 1
EXTRA_INVESTIGATION_COST = 0.02
MAX_EXTRA_INVESTIGATION_PENALTY = 0.15


def _coerce_optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


class HelpdeskTicketRoutingEnvironment(
    Environment[HelpdeskTicketAction, HelpdeskTicketObservation, HelpdeskTicketState]
):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._dataset = load_dataset()
        self._tickets_by_id = {ticket.ticket_id: ticket for ticket in self._dataset}
        self._rng = random.Random()
        self._queue: list[HelpdeskTicketRecord] = []
        self._state = HelpdeskTicketState()

    # ------------------------------------------------------------------
    # OpenEnv required interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> HelpdeskTicketObservation:
        normalized_seed = _coerce_optional_int(seed, "seed")
        task_id_value = _coerce_optional_int(kwargs.get("task_id", 1), "task_id")
        queue_size_value = _coerce_optional_int(kwargs.get("queue_size"), "queue_size")
        task_id = 1 if task_id_value is None else task_id_value
        task = get_task_definition(task_id)
        if queue_size_value is not None and queue_size_value < 1:
            raise ValueError("queue_size must be >= 1")

        if normalized_seed is not None:
            self._rng.seed(normalized_seed)

        if queue_size_value is None:
            queue_size = self._rng.randint(*QUEUE_SIZE_RANGE)
        else:
            queue_size = min(queue_size_value, len(self._dataset))
        self._queue = self._rng.sample(self._dataset, min(queue_size, len(self._dataset)))

        self._state = HelpdeskTicketState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_task_id=task_id,
            seed=normalized_seed,
            queue_ticket_ids=[t.ticket_id for t in self._queue],
            current_ticket_index=0,
            per_ticket_scores=[],
            total_reward=0.0,
            investigation_budget_remaining=queue_size * FREE_INVESTIGATIONS_PER_TICKET,
        )

        return self._build_observation(task)

    def step(
        self,
        action: HelpdeskTicketAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HelpdeskTicketObservation:
        if not self._queue or self._state.current_task_id is None:
            raise RuntimeError("Environment has not been reset.")

        idx = self._state.current_ticket_index
        if idx >= len(self._queue):
            raise RuntimeError("Episode already done — call reset().")

        current_ticket = self._queue[idx]
        task_id = self._state.current_task_id
        task = get_task_definition(task_id)

        if action.action_type == "investigate":
            return self._handle_investigation_action(task, current_ticket, action, idx)

        submitted_fields = {
            f
            for f, v in action.model_dump(exclude_none=True).items()
            if v is not None
            and f not in {"action_type", "tool_name", "tool_target_ticket_id"}
        }
        allowed = set(task["allowed_fields"])
        extra_fields = submitted_fields - allowed
        if extra_fields:
            # Penalty: record score 0.0, advance index, return penalty observation
            self._state.per_ticket_scores.append(0.0)
            self._state.history_entries.append(
                self._build_history_entry(
                    current_ticket,
                    predicted=action.model_dump(exclude_none=True),
                    score=0.0,
                    breakdown={},
                    queue_position=idx + 1,
                    penalty_reason=f"extra_fields: {sorted(extra_fields)}",
                )
            )
            self._state.step_count += 1
            self._state.current_ticket_index += 1
            is_done = self._state.current_ticket_index >= len(self._queue)
            self._state.done = is_done
            if is_done:
                traj_reward = compute_trajectory_reward(
                    self._state.per_ticket_scores, len(self._queue), self._state.step_count
                )
                final_reward = self._apply_episode_economics(traj_reward)
                self._state.total_reward = final_reward
            else:
                final_reward = 0.0
            self._state.last_step_reward = final_reward
            self._state.reward = final_reward
            self._state.last_tool_result = None
            return self._build_observation(task, done=is_done, reward=final_reward)

        score, breakdown = grade_action(action, current_ticket, task_id)
        step_reward = compute_step_reward(score)

        is_done = (self._state.current_ticket_index + 1) >= len(self._queue)

        if is_done:
            self._state.per_ticket_scores.append(score)
            self._state.step_count += 1
            self._state.current_ticket_index += 1
            traj_reward = compute_trajectory_reward(
                self._state.per_ticket_scores,
                len(self._queue),
                self._state.step_count,
            )
            final_reward = self._apply_episode_economics(traj_reward)
            self._state.total_reward = final_reward
        else:
            self._state.per_ticket_scores.append(score)
            self._state.step_count += 1
            self._state.current_ticket_index += 1
            final_reward = step_reward

        history_entry = self._build_history_entry(
            current_ticket,
            predicted=action.model_dump(exclude_none=True),
            score=score,
            breakdown=breakdown,
            queue_position=idx + 1,
        )
        self._state.history_entries.append(history_entry)

        self._state.last_step_reward = final_reward
        self._state.reward = final_reward
        self._state.done = is_done
        self._state.last_tool_result = None

        return self._build_observation(task, done=is_done, reward=final_reward)

    @property
    def state(self) -> HelpdeskTicketState:
        return self._state.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_episode_economics(self, base_reward: float) -> float:
        free_investigations = len(self._queue) * FREE_INVESTIGATIONS_PER_TICKET
        extra_investigations = max(0, self._state.investigation_steps - free_investigations)
        penalty = min(
            MAX_EXTRA_INVESTIGATION_PENALTY,
            extra_investigations * EXTRA_INVESTIGATION_COST,
        )
        return max(0.0, min(1.0, base_reward - penalty))

    def _lookup_related_ticket(
        self,
        current_ticket: HelpdeskTicketRecord,
        target_ticket_id: str | None,
    ) -> dict[str, Any]:
        target_id = target_ticket_id or current_ticket.related_ticket_id
        if target_id is None:
            return {
                "tool_name": "lookup_related_ticket",
                "found": False,
                "message": "Current ticket has no linked related_ticket_id.",
            }
        related_ticket = self._tickets_by_id.get(target_id)
        if related_ticket is None:
            return {
                "tool_name": "lookup_related_ticket",
                "found": False,
                "message": f"Ticket {target_id!r} was not found in the dataset.",
            }
        return {
            "tool_name": "lookup_related_ticket",
            "found": True,
            "ticket": {
                "ticket_id": related_ticket.ticket_id,
                "title": related_ticket.title,
                "requester": related_ticket.requester,
                "description": related_ticket.description,
                "issue_type": related_ticket.issue_type,
                "priority": related_ticket.priority,
                "assignment_group": related_ticket.assignment_group,
                "resolution_action": related_ticket.resolution_action,
            },
        }

    def _lookup_requester_history(self, current_ticket: HelpdeskTicketRecord) -> dict[str, Any]:
        matches = [
            {
                "ticket_id": ticket.ticket_id,
                "title": ticket.title,
                "issue_type": ticket.issue_type,
                "priority": ticket.priority,
                "assignment_group": ticket.assignment_group,
                "resolution_action": ticket.resolution_action,
            }
            for ticket in self._dataset
            if ticket.requester == current_ticket.requester
            and ticket.ticket_id != current_ticket.ticket_id
        ]
        return {
            "tool_name": "lookup_requester_history",
            "found": bool(matches),
            "requester": current_ticket.requester,
            "matches": matches,
        }

    def _run_investigation_tool(
        self,
        current_ticket: HelpdeskTicketRecord,
        tool_name: str,
        target_ticket_id: str | None,
    ) -> dict[str, Any]:
        if tool_name == "lookup_related_ticket":
            return self._lookup_related_ticket(current_ticket, target_ticket_id)
        if tool_name == "lookup_requester_history":
            return self._lookup_requester_history(current_ticket)
        raise ValueError(f"Unsupported tool_name: {tool_name}")

    def _handle_investigation_action(
        self,
        task: dict,
        current_ticket: HelpdeskTicketRecord,
        action: HelpdeskTicketAction,
        idx: int,
    ) -> HelpdeskTicketObservation:
        if action.tool_name is None:
            raise ValueError("Investigate actions require tool_name")
        submitted_fields = {
            field
            for field in ("issue_type", "priority", "assignment_group", "resolution_action")
            if getattr(action, field) is not None
        }
        if submitted_fields:
            raise ValueError(
                "Investigate actions cannot include submit fields: "
                f"{sorted(submitted_fields)}"
            )

        tool_result = self._run_investigation_tool(
            current_ticket,
            action.tool_name,
            action.tool_target_ticket_id,
        )
        self._state.step_count += 1
        self._state.investigation_steps += 1
        self._state.investigation_budget_remaining = max(
            0,
            self._state.investigation_budget_remaining - 1,
        )
        self._state.last_tool_result = tool_result
        self._state.last_step_reward = 0.0
        self._state.reward = 0.0
        self._state.done = False
        self._state.history_entries.append(
            self._build_history_entry(
                current_ticket,
                predicted=action.model_dump(exclude_none=True),
                score=0.0,
                breakdown={},
                queue_position=idx + 1,
                tool_result=tool_result,
            )
        )
        return self._build_observation(task, done=False, reward=0.0)

    def _build_ticket_view(self, ticket: HelpdeskTicketRecord) -> dict[str, Any]:
        ticket_view: dict[str, Any] = {
            "ticket_id": ticket.ticket_id,
            "title": ticket.title,
            "requester": ticket.requester,
            "description": ticket.description,
        }
        if ticket.ambiguity_note is not None:
            ticket_view["ambiguity_note"] = ticket.ambiguity_note
        if ticket.related_ticket_id is not None:
            ticket_view["related_ticket_id"] = ticket.related_ticket_id
            related_ticket = self._tickets_by_id.get(ticket.related_ticket_id)
            if related_ticket is not None:
                ticket_view["related_ticket_preview"] = {
                    "ticket_id": related_ticket.ticket_id,
                    "title": related_ticket.title,
                    "requester": related_ticket.requester,
                    "description": related_ticket.description,
                }
        return ticket_view

    def _build_history_entry(
        self,
        ticket: HelpdeskTicketRecord,
        *,
        predicted: dict[str, Any],
        score: float,
        breakdown: dict[str, float],
        queue_position: int,
        penalty_reason: str | None = None,
        tool_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        history_entry: dict[str, Any] = {
            "ticket_id": ticket.ticket_id,
            "title": ticket.title,
            "requester": ticket.requester,
            "predicted": predicted,
            "score": score,
            "breakdown": breakdown,
            "queue_position": queue_position,
        }
        if ticket.ambiguity_note is not None:
            history_entry["ambiguity_note"] = ticket.ambiguity_note
        if ticket.related_ticket_id is not None:
            history_entry["related_ticket_id"] = ticket.related_ticket_id
            related_ticket = self._tickets_by_id.get(ticket.related_ticket_id)
            if related_ticket is not None:
                history_entry["related_ticket_preview"] = {
                    "ticket_id": related_ticket.ticket_id,
                    "title": related_ticket.title,
                    "requester": related_ticket.requester,
                    "description": related_ticket.description,
                }
        if penalty_reason is not None:
            history_entry["penalty_reason"] = penalty_reason
        if tool_result is not None:
            history_entry["tool_result"] = tool_result
        return history_entry

    def _build_observation(
        self,
        task: dict,
        done: bool = False,
        reward: float | None = None,
    ) -> HelpdeskTicketObservation:
        idx = self._state.current_ticket_index
        queue_size = len(self._queue)

        if idx < queue_size:
            ticket = self._queue[idx]
            ticket_view = self._build_ticket_view(ticket)
            queue_position = idx + 1
        else:
            ticket_view = None
            queue_position = 0

        history = list(self._state.history_entries)
        tickets_remaining = max(0, queue_size - idx)
        tickets_after_current = max(
            0,
            tickets_remaining - (1 if ticket_view is not None else 0),
        )

        return HelpdeskTicketObservation(
            done=done,
            reward=reward,
            metadata={
                "queue_position": queue_position,
                "tickets_remaining_includes_current": ticket_view is not None,
                "has_ambiguity_note": bool(ticket_view and ticket_view.get("ambiguity_note")),
                "has_related_ticket_context": bool(
                    ticket_view and ticket_view.get("related_ticket_preview")
                ),
                "action_mode": "investigate_or_submit",
            },
            task_id=task["id"],
            task_name=task["name"],
            instructions=task["instructions"],
            allowed_fields=list(task["allowed_fields"]),
            available_tools=list(AVAILABLE_TOOLS),
            investigation_budget_remaining=self._state.investigation_budget_remaining,
            last_tool_result=self._state.last_tool_result,
            current_ticket=ticket_view,
            queue_size=queue_size,
            tickets_remaining=tickets_remaining,
            tickets_after_current=tickets_after_current,
            tickets_processed=idx,
            queue_position=queue_position,
            history=history,
        )
