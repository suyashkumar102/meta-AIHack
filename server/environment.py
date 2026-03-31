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
    def __init__(self) -> None:
        super().__init__()
        self._dataset = load_dataset()
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
        task_id = 1 if task_id_value is None else task_id_value
        task = get_task_definition(task_id)

        if normalized_seed is not None:
            self._rng.seed(normalized_seed)

        queue_size = self._rng.randint(*QUEUE_SIZE_RANGE)
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

        score, breakdown = grade_action(action, current_ticket, task_id)
        step_reward = compute_step_reward(score)

        self._state.per_ticket_scores.append(score)
        self._state.step_count += 1
        self._state.current_ticket_index += 1

        is_done = self._state.current_ticket_index >= len(self._queue)

        if is_done:
            traj_reward = compute_trajectory_reward(
                self._state.per_ticket_scores,
                len(self._queue),
                self._state.step_count,
            )
            self._state.total_reward = traj_reward
            final_reward = traj_reward
        else:
            final_reward = step_reward

        history_entry = {
            "ticket_id": current_ticket.ticket_id,
            "score": score,
            "breakdown": breakdown,
        }

        return self._build_observation(
            task,
            done=is_done,
            reward=final_reward,
            extra_history=history_entry,
        )

    @property
    def state(self) -> HelpdeskTicketState:
        return self._state.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        task: dict,
        done: bool = False,
        reward: float | None = None,
        extra_history: dict | None = None,
    ) -> HelpdeskTicketObservation:
        idx = self._state.current_ticket_index
        queue_size = len(self._queue)

        if idx < queue_size:
            ticket = self._queue[idx]
            ticket_view = {
                "ticket_id": ticket.ticket_id,
                "title": ticket.title,
                "requester": ticket.requester,
                "description": ticket.description,
            }
        else:
            ticket_view = None

        history: list[dict] = []
        for i, s in enumerate(self._state.per_ticket_scores):
            history.append({"step": i + 1, "score": s})
        if extra_history and history:
            history[-1] = {"step": len(history), **extra_history}

        return HelpdeskTicketObservation(
            done=done,
            reward=reward,
            metadata={},
            task_id=task["id"],
            task_name=task["name"],
            instructions=task["instructions"],
            allowed_fields=list(task["allowed_fields"]),
            current_ticket=ticket_view,
            queue_size=queue_size,
            tickets_remaining=max(0, queue_size - idx),
            tickets_processed=idx,
            history=history,
        )
