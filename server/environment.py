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
from server.reward import (
    clamp_open_unit_interval,
    compute_step_adjustments,
    compute_trajectory_adjustments,
)
from server.tasks import get_task_definition, load_dataset
from vocabulary import (
    ISSUE_TYPE_TO_ASSIGNMENT_GROUP,
    ISSUE_TYPE_TO_RESOLUTION_ACTION,
)


QUEUE_SIZE_RANGE = (3, 5)
AVAILABLE_ACTION_TYPES = ("submit", "investigate")
AVAILABLE_TOOLS = (
    "lookup_related_ticket",
    "lookup_requester_history",
    "lookup_internal_routing_note",
)
FREE_INVESTIGATIONS_PER_TICKET = 1
EXTRA_INVESTIGATION_COST = 0.04
MAX_EXTRA_INVESTIGATION_PENALTY = 0.25
USEFUL_INVESTIGATION_REWARD = 0.03
PREMATURE_SUBMIT_PENALTY = 0.22
NONDEFAULT_HIDDEN_CONTEXT_PENALTY = 0.08
CONTEXT_COMPLETION_BONUS = 0.06
TRAJECTORY_CONTEXT_COMPLETION_BONUS = 0.04
PRIORITY_UNDERSHOOT_PENALTY = 0.03
SEVERE_PRIORITY_UNDERSHOOT_PENALTY = 0.07
DANGEROUS_RESOLUTION_PENALTY = 0.05
NONDEFAULT_ROUTING_FOLLOWTHROUGH_BONUS = 0.02

TASK3_INVESTIGATION_TOOL_PLAN: dict[str, tuple[str, ...]] = {
    "ticket-021": ("lookup_related_ticket", "lookup_requester_history"),
    "ticket-022": ("lookup_internal_routing_note",),
    "ticket-027": ("lookup_internal_routing_note",),
    "ticket-029": ("lookup_internal_routing_note",),
    "ticket-038": ("lookup_related_ticket", "lookup_requester_history"),
    "ticket-045": ("lookup_related_ticket", "lookup_requester_history"),
    "TKT-NONDEFAULT-001": ("lookup_internal_routing_note",),
    "TKT-NONDEFAULT-002": ("lookup_internal_routing_note",),
    "TKT-NONDEFAULT-003": ("lookup_internal_routing_note",),
}

HARD_TASK_DESCRIPTION_REDACTIONS: dict[str, str] = {
    "ticket-021": (
        "Production checkout is still unstable after a recent fix. "
        "Additional routing context is available via investigation."
    ),
    "ticket-022": (
        "Usage charges increased while the integration was failing. "
        "Additional routing context is available via investigation."
    ),
    "ticket-027": (
        "A vendor offer arrived with a near-term deadline. "
        "Additional routing context is available via investigation."
    ),
    "ticket-029": (
        "A team needs a large seat expansion right away. "
        "Additional routing context is available via investigation."
    ),
    "ticket-038": (
        "A prior invoice discrepancy is still unresolved and now time-sensitive. "
        "Additional routing context is available via investigation."
    ),
    "ticket-045": (
        "A company-wide suspension remains unresolved after repeated follow-ups. "
        "Additional routing context is available via investigation."
    ),
    "TKT-NONDEFAULT-001": (
        "A user needs help with a billing-style question. "
        "Additional routing context is available via investigation."
    ),
    "TKT-NONDEFAULT-002": (
        "A client compliance scan surfaced a product-specific issue. "
        "Additional routing context is available via investigation."
    ),
    "TKT-NONDEFAULT-003": (
        "A contractor onboarding workflow is blocked by an account problem. "
        "Additional routing context is available via investigation."
    ),
}

HARD_TASK_TITLE_REDACTIONS: dict[str, str] = {
    "ticket-021": "Production workflow regression",
    "ticket-022": "Time-sensitive account review",
    "ticket-027": "Commercial workflow request",
    "ticket-029": "Urgent expansion request",
    "ticket-038": "Repeated invoice follow-up",
    "ticket-045": "Company-wide account issue",
    "TKT-NONDEFAULT-001": "Billing-style routing question",
    "TKT-NONDEFAULT-002": "Compliance ownership question",
    "TKT-NONDEFAULT-003": "Workflow blocker with hidden owner",
}


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
            average_score_so_far=0.0,
            investigation_budget_remaining=queue_size * FREE_INVESTIGATIONS_PER_TICKET,
            investigation_penalty_applied=0.0,
            last_reward_components={},
            ticket_tool_usage={},
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
            and f not in {"action_type", "tool_name", "tool_target_ticket_id", "metadata"}
        }
        allowed = set(task["allowed_fields"])
        extra_fields = submitted_fields - allowed
        if extra_fields:
            # Penalty: record an open-interval score, advance index, return penalty observation
            invalid_score = clamp_open_unit_interval(0.0)
            self._state.per_ticket_scores.append(invalid_score)
            self._state.average_score_so_far = self._current_average_score()
            self._state.step_count += 1
            self._state.current_ticket_index += 1
            is_done = self._state.current_ticket_index >= len(self._queue)
            self._state.done = is_done
            trajectory_reward = None
            trajectory_components = None
            investigation_penalty = self._compute_episode_penalty() if is_done else 0.0
            if is_done:
                trajectory_components = compute_trajectory_adjustments(
                    self._state.per_ticket_scores,
                    len(self._queue),
                    self._state.step_count,
                    completion_bonus=self._trajectory_consistency_bonus(),
                )
                trajectory_reward = trajectory_components["final_reward"]
                final_reward = self._apply_episode_economics(trajectory_reward)
                self._state.total_reward = final_reward
            else:
                final_reward = clamp_open_unit_interval(0.0)
            reward_components = self._build_reward_components(
                ticket_score=invalid_score,
                field_breakdown={},
                shaped_step_reward=0.0,
                reward_kind="trajectory" if is_done else "step_penalty",
                final_reward=final_reward,
                trajectory_reward=trajectory_reward,
                investigation_penalty=investigation_penalty,
                penalty_reason=f"extra_fields: {sorted(extra_fields)}",
                extra_details={
                    "trajectory_average_reward": (
                        trajectory_components["average_reward"]
                        if trajectory_components is not None
                        else None
                    ),
                    "trajectory_completion_bonus": (
                        trajectory_components["completion_bonus"]
                        if trajectory_components is not None
                        else None
                    ),
                    "trajectory_consistency_bonus": (
                        trajectory_components["consistency_bonus"]
                        if trajectory_components is not None
                        else None
                    ),
                },
            )
            self._state.history_entries.append(
                self._build_history_entry(
                    current_ticket,
                    predicted=action.model_dump(exclude_none=True),
                    score=invalid_score,
                    breakdown={},
                    queue_position=idx + 1,
                    reward=final_reward,
                    rubric_reward=final_reward if is_done else None,
                    reward_kind="trajectory" if is_done else "step_penalty",
                    penalty_reason=f"extra_fields: {sorted(extra_fields)}",
                    reward_components=reward_components,
                )
            )
            self._state.last_step_reward = final_reward
            self._state.reward = final_reward
            self._state.investigation_penalty_applied = self._compute_episode_penalty()
            self._state.last_tool_result = None
            self._state.last_reward_components = reward_components
            return self._build_observation(
                task,
                done=is_done,
                reward=final_reward,
                rubric_reward=final_reward if is_done else None,
            )

        previous_average = self._current_average_score()
        score, breakdown = grade_action(action, current_ticket, task_id)
        context_penalty, missing_required_count = self._submit_context_penalty(current_ticket)
        process_bonus = self._context_completion_bonus(
            current_ticket,
            missing_required_count=missing_required_count,
            score=score,
        )
        risk_penalty = self._operational_risk_penalty(
            current_ticket,
            action,
            task_id=task_id,
        )
        step_adjustments = compute_step_adjustments(
            score,
            previous_average=previous_average,
            process_bonus=process_bonus,
            risk_penalty=risk_penalty,
        )
        step_reward = step_adjustments["final_reward"]

        is_done = (self._state.current_ticket_index + 1) >= len(self._queue)
        trajectory_reward = None
        trajectory_components = None
        investigation_penalty = 0.0
        rubric_reward = None

        if is_done:
            self._state.per_ticket_scores.append(score)
            self._state.average_score_so_far = self._current_average_score()
            self._state.step_count += 1
            self._state.current_ticket_index += 1
            trajectory_components = compute_trajectory_adjustments(
                self._state.per_ticket_scores,
                len(self._queue),
                self._state.step_count,
                completion_bonus=self._trajectory_consistency_bonus(),
            )
            trajectory_reward = trajectory_components["final_reward"]
            rubric_reward = self._apply_episode_economics(trajectory_reward)
            final_reward = clamp_open_unit_interval(rubric_reward - context_penalty)
            self._state.total_reward = rubric_reward
            investigation_penalty = self._compute_episode_penalty()
        else:
            self._state.per_ticket_scores.append(score)
            self._state.average_score_so_far = self._current_average_score()
            self._state.step_count += 1
            self._state.current_ticket_index += 1
            final_reward = clamp_open_unit_interval(step_reward - context_penalty)

        reward_components = self._build_reward_components(
            ticket_score=score,
            field_breakdown=breakdown,
            shaped_step_reward=step_reward,
            reward_kind="trajectory" if is_done else "step",
            final_reward=final_reward,
            milestone_adjustment=step_adjustments["milestone_adjustment"],
            trajectory_reward=trajectory_reward,
            investigation_penalty=investigation_penalty,
            extra_details={
                "context_gap_penalty": context_penalty,
                "context_completion_bonus": process_bonus,
                "risk_penalty": risk_penalty,
                "delta_adjustment": step_adjustments["delta_adjustment"],
                "required_investigation_count": len(self._required_tools_for_ticket(current_ticket)),
                "hidden_context_remaining_count": missing_required_count,
                "hidden_context_revealed_count": len(
                    self._used_tools_for_ticket(current_ticket.ticket_id)
                ),
                "rubric_reward": rubric_reward,
                "trajectory_average_reward": (
                    trajectory_components["average_reward"]
                    if trajectory_components is not None
                    else None
                ),
                "trajectory_completion_bonus": (
                    trajectory_components["completion_bonus"]
                    if trajectory_components is not None
                    else None
                ),
                "trajectory_consistency_bonus": (
                    trajectory_components["consistency_bonus"]
                    if trajectory_components is not None
                    else None
                ),
            },
        )

        history_entry = self._build_history_entry(
            current_ticket,
            predicted=action.model_dump(exclude_none=True),
            score=score,
            breakdown=breakdown,
            queue_position=idx + 1,
            reward=final_reward,
            rubric_reward=rubric_reward if is_done else None,
            reward_kind="trajectory" if is_done else "step",
            reward_components=reward_components,
        )
        self._state.history_entries.append(history_entry)

        self._state.last_step_reward = final_reward
        self._state.reward = final_reward
        self._state.done = is_done
        self._state.investigation_penalty_applied = self._compute_episode_penalty()
        self._state.last_tool_result = None
        self._state.last_reward_components = reward_components

        return self._build_observation(
            task,
            done=is_done,
            reward=final_reward,
            rubric_reward=rubric_reward if is_done else None,
        )

    @property
    def state(self) -> HelpdeskTicketState:
        return self._state.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_episode_penalty(self) -> float:
        free_investigations = len(self._queue) * FREE_INVESTIGATIONS_PER_TICKET
        extra_investigations = max(0, self._state.investigation_steps - free_investigations)
        return min(
            MAX_EXTRA_INVESTIGATION_PENALTY,
            extra_investigations * EXTRA_INVESTIGATION_COST,
        )

    def _apply_episode_economics(self, base_reward: float) -> float:
        penalty = self._compute_episode_penalty()
        return clamp_open_unit_interval(base_reward - penalty)

    def _current_average_score(self) -> float:
        if not self._state.per_ticket_scores:
            return 0.0
        return sum(self._state.per_ticket_scores) / len(self._state.per_ticket_scores)

    def _internal_routing_note_for_ticket(
        self,
        ticket: HelpdeskTicketRecord,
    ) -> str | None:
        if ticket.ambiguity_note is not None:
            return ticket.ambiguity_note
        if self._state.current_task_id != 3:
            return None

        default_group = ISSUE_TYPE_TO_ASSIGNMENT_GROUP.get(
            ticket.issue_type,
            ticket.assignment_group,
        )
        default_action = ISSUE_TYPE_TO_RESOLUTION_ACTION.get(
            ticket.issue_type,
            ticket.resolution_action,
        )
        note_parts: list[str] = []

        if ticket.assignment_group != default_group:
            note_parts.append(
                "Routing override: send this to "
                f"{ticket.assignment_group} rather than the default {default_group} queue."
            )
        if ticket.resolution_action != default_action:
            note_parts.append(
                "Action override: use "
                f"{ticket.resolution_action} instead of the default {default_action} next step."
            )
        if ticket.issue_type == "onboarding" and ticket.assignment_group == "service_desk":
            note_parts.append(
                "The onboarding workflow is blocked by an access dependency, so the unblocker owns the next move."
            )
        if (
            ticket.issue_type == "security_compliance"
            and ticket.assignment_group == "application_team"
        ):
            note_parts.append(
                "This compliance issue needs a product-team fix rather than a central security handoff."
            )
        if ticket.issue_type == "billing_license" and ticket.assignment_group == "procurement":
            note_parts.append(
                "Treat this as commercial procurement work instead of routine license fulfillment."
            )

        if not note_parts:
            return None
        return " ".join(note_parts)

    def _ticket_has_nondefault_routing(self, ticket: HelpdeskTicketRecord) -> bool:
        return (
            ticket.assignment_group
            != ISSUE_TYPE_TO_ASSIGNMENT_GROUP.get(ticket.issue_type, ticket.assignment_group)
            or ticket.resolution_action
            != ISSUE_TYPE_TO_RESOLUTION_ACTION.get(
                ticket.issue_type, ticket.resolution_action
            )
        )

    def _ticket_mentions_follow_up(self, ticket: HelpdeskTicketRecord) -> bool:
        text = f"{ticket.title} {ticket.description}".lower()
        return any(
            phrase in text
            for phrase in (
                "re:",
                "follow-up",
                "following up",
                "still",
                "third update",
                "reference ticket",
                "regression",
                "unresolved",
            )
        )

    def _ticket_repeated_requester_count(self, ticket: HelpdeskTicketRecord) -> int:
        return sum(1 for candidate in self._dataset if candidate.requester == ticket.requester)

    def _tool_has_available_context(
        self,
        ticket: HelpdeskTicketRecord,
        tool_name: str,
    ) -> bool:
        if tool_name == "lookup_related_ticket":
            return (
                ticket.related_ticket_id is not None
                and ticket.related_ticket_id in self._tickets_by_id
            )
        if tool_name == "lookup_requester_history":
            return self._ticket_repeated_requester_count(ticket) >= 2
        if tool_name == "lookup_internal_routing_note":
            return self._internal_routing_note_for_ticket(ticket) is not None
        return False

    def _required_tools_for_ticket(
        self,
        ticket: HelpdeskTicketRecord,
        task_id: int | None = None,
    ) -> list[str]:
        resolved_task_id = self._state.current_task_id if task_id is None else task_id
        if resolved_task_id != 3:
            return []
        required_tools: list[str] = list(TASK3_INVESTIGATION_TOOL_PLAN.get(ticket.ticket_id, ()))
        if ticket.related_ticket_id is not None and "lookup_related_ticket" not in required_tools:
            required_tools.append("lookup_related_ticket")
        if (
            self._internal_routing_note_for_ticket(ticket) is not None
            and "lookup_internal_routing_note" not in required_tools
        ):
            required_tools.append("lookup_internal_routing_note")
        if (
            self._ticket_repeated_requester_count(ticket) >= 2
            and (
                ticket.related_ticket_id is not None
                or self._ticket_mentions_follow_up(ticket)
                or self._ticket_has_nondefault_routing(ticket)
                or ticket.priority in {"high", "critical"}
            )
            and "lookup_requester_history" not in required_tools
        ):
            required_tools.append("lookup_requester_history")
        filtered_required_tools: list[str] = []
        for tool_name in required_tools:
            if tool_name in filtered_required_tools:
                continue
            if self._tool_has_available_context(ticket, tool_name):
                filtered_required_tools.append(tool_name)
        return filtered_required_tools

    def _used_tools_for_ticket(self, ticket_id: str) -> list[str]:
        return list(self._state.ticket_tool_usage.get(ticket_id, []))

    def _remaining_tools_for_ticket(
        self,
        ticket: HelpdeskTicketRecord,
        task_id: int | None = None,
    ) -> list[str]:
        required_tools = self._required_tools_for_ticket(ticket, task_id)
        used_tools = set(self._used_tools_for_ticket(ticket.ticket_id))
        return [tool for tool in required_tools if tool not in used_tools]

    def _record_tool_usage(self, ticket_id: str, tool_name: str) -> None:
        used = self._state.ticket_tool_usage.setdefault(ticket_id, [])
        if tool_name not in used:
            used.append(tool_name)

    def _tool_progress_for_ticket(self, ticket: HelpdeskTicketRecord) -> dict[str, Any]:
        required_tools = self._required_tools_for_ticket(ticket)
        revealed_tools = self._used_tools_for_ticket(ticket.ticket_id)
        remaining_tools = self._remaining_tools_for_ticket(ticket)
        total_required = max(1, len(required_tools))
        return {
            "required_tools": required_tools,
            "revealed_tools": revealed_tools,
            "remaining_tools": remaining_tools,
            "revealed_count": len(revealed_tools),
            "remaining_count": len(remaining_tools),
            "completeness": round(len(revealed_tools) / total_required, 2),
        }

    def _default_redacted_description(self, ticket: HelpdeskTicketRecord) -> str:
        if ticket.related_ticket_id is not None:
            return (
                "This is a follow-up operational issue. "
                "Additional routing context is available via investigation."
            )
        if self._internal_routing_note_for_ticket(ticket) is not None:
            return (
                "The visible request is not enough to choose the final owner and next step. "
                "Additional routing context is available via investigation."
            )
        if self._ticket_has_nondefault_routing(ticket):
            return (
                "The visible request looks straightforward, but the decisive routing detail is hidden until investigation."
            )
        return (
            "Additional routing context is available via investigation before final submission."
        )

    def _default_redacted_title(self, ticket: HelpdeskTicketRecord) -> str:
        if ticket.related_ticket_id is not None:
            return "Follow-up request with hidden routing context"
        if self._internal_routing_note_for_ticket(ticket) is not None:
            return "Routing clarification required"
        if self._ticket_mentions_follow_up(ticket):
            return "Priority support follow-up"
        return "Helpdesk routing decision"

    def _visible_title(self, ticket: HelpdeskTicketRecord) -> str:
        if self._state.current_task_id == 3 and self._remaining_tools_for_ticket(ticket):
            return HARD_TASK_TITLE_REDACTIONS.get(
                ticket.ticket_id,
                self._default_redacted_title(ticket),
            )
        return ticket.title

    def _visible_description(self, ticket: HelpdeskTicketRecord) -> str:
        if self._state.current_task_id == 3 and self._remaining_tools_for_ticket(ticket):
            return HARD_TASK_DESCRIPTION_REDACTIONS.get(
                ticket.ticket_id,
                self._default_redacted_description(ticket),
            )
        return ticket.description

    def _submit_context_penalty(self, ticket: HelpdeskTicketRecord) -> tuple[float, int]:
        progress = self._tool_progress_for_ticket(ticket)
        required_tools = progress["required_tools"]
        remaining_tools = progress["remaining_tools"]
        if not required_tools or not remaining_tools:
            return 0.0, 0
        penalty = PREMATURE_SUBMIT_PENALTY * (
            len(remaining_tools) / max(1, len(required_tools))
        )
        if self._ticket_has_nondefault_routing(ticket):
            penalty += NONDEFAULT_HIDDEN_CONTEXT_PENALTY * (
                len(remaining_tools) / max(1, len(required_tools))
            )
        return round(min(0.45, penalty), 4), len(remaining_tools)

    def _context_completion_bonus(
        self,
        ticket: HelpdeskTicketRecord,
        *,
        missing_required_count: int,
        score: float,
    ) -> float:
        if not self._required_tools_for_ticket(ticket):
            return 0.0
        if missing_required_count != 0 or score < 0.75:
            return 0.0
        bonus = CONTEXT_COMPLETION_BONUS
        if self._ticket_has_nondefault_routing(ticket):
            bonus += NONDEFAULT_ROUTING_FOLLOWTHROUGH_BONUS
        return bonus

    def _trajectory_consistency_bonus(self) -> float:
        if not self._queue:
            return 0.0
        hidden_context_tickets = [
            ticket for ticket in self._queue if self._required_tools_for_ticket(ticket)
        ]
        if not hidden_context_tickets:
            return 0.0
        resolved = sum(
            1 for ticket in hidden_context_tickets if not self._remaining_tools_for_ticket(ticket)
        )
        resolution_rate = resolved / len(hidden_context_tickets)
        return round(TRAJECTORY_CONTEXT_COMPLETION_BONUS * resolution_rate, 4)

    def _operational_risk_penalty(
        self,
        ticket: HelpdeskTicketRecord,
        action: HelpdeskTicketAction,
        *,
        task_id: int,
    ) -> float:
        if task_id < 2 or action.priority is None:
            priority_penalty = 0.0
        else:
            priority_rank = {"critical": 3, "high": 2, "medium": 1, "low": 0}
            expected_rank = priority_rank.get(ticket.priority, 0)
            predicted_rank = priority_rank.get(action.priority, 0)
            gap = expected_rank - predicted_rank
            if gap >= 2:
                priority_penalty = SEVERE_PRIORITY_UNDERSHOOT_PENALTY
            elif gap == 1 and ticket.priority in {"high", "critical"}:
                priority_penalty = PRIORITY_UNDERSHOOT_PENALTY
            else:
                priority_penalty = 0.0

        resolution_penalty = 0.0
        if task_id == 3 and action.resolution_action is not None:
            if (
                ticket.issue_type in {"identity_access", "application_support", "security_compliance"}
                and ticket.priority in {"high", "critical"}
                and action.resolution_action == "acknowledge"
            ):
                resolution_penalty += DANGEROUS_RESOLUTION_PENALTY
            if ticket.issue_type == "spam_phishing" and action.resolution_action == "fulfill":
                resolution_penalty += PRIORITY_UNDERSHOOT_PENALTY

        return round(priority_penalty + resolution_penalty, 4)

    def _build_reward_components(
        self,
        *,
        ticket_score: float,
        field_breakdown: dict[str, float],
        shaped_step_reward: float,
        reward_kind: str,
        final_reward: float,
        milestone_adjustment: float = 0.0,
        trajectory_reward: float | None = None,
        investigation_penalty: float = 0.0,
        penalty_reason: str | None = None,
        extra_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        components: dict[str, Any] = {
            "reward_kind": reward_kind,
            "ticket_score": ticket_score,
            "field_breakdown": field_breakdown,
            "shaped_step_reward": shaped_step_reward,
            "milestone_adjustment": milestone_adjustment,
            "final_reward": final_reward,
            "average_score_so_far": self._current_average_score(),
            "investigation_penalty_applied": investigation_penalty,
        }
        if trajectory_reward is not None:
            components["trajectory_reward"] = trajectory_reward
        if penalty_reason is not None:
            components["penalty_reason"] = penalty_reason
        if extra_details:
            components.update(extra_details)
        return components

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

    def _lookup_internal_routing_note(self, current_ticket: HelpdeskTicketRecord) -> dict[str, Any]:
        routing_note = self._internal_routing_note_for_ticket(current_ticket)
        found = routing_note is not None
        return {
            "tool_name": "lookup_internal_routing_note",
            "found": found,
            "ticket_id": current_ticket.ticket_id,
            "routing_note": routing_note if found else "",
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
        if tool_name == "lookup_internal_routing_note":
            return self._lookup_internal_routing_note(current_ticket)
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
        required_tools = self._required_tools_for_ticket(current_ticket)
        already_used = action.tool_name in self._used_tools_for_ticket(current_ticket.ticket_id)
        useful_investigation = (
            action.tool_name in required_tools
            and not already_used
            and bool(tool_result.get("found", True))
        )
        self._record_tool_usage(current_ticket.ticket_id, action.tool_name)
        self._state.step_count += 1
        self._state.investigation_steps += 1
        self._state.investigation_budget_remaining = max(
            0,
            self._state.investigation_budget_remaining - 1,
        )
        self._state.last_tool_result = tool_result
        investigation_reward = USEFUL_INVESTIGATION_REWARD if useful_investigation else 0.0
        investigation_score = 0.0
        self._state.last_step_reward = investigation_reward
        self._state.reward = investigation_reward
        self._state.done = False
        self._state.investigation_penalty_applied = self._compute_episode_penalty()
        progress = self._tool_progress_for_ticket(current_ticket)
        reward_components = self._build_reward_components(
            ticket_score=investigation_score,
            field_breakdown={},
            shaped_step_reward=investigation_reward,
            reward_kind="investigation",
            final_reward=investigation_reward,
            investigation_penalty=self._state.investigation_penalty_applied,
            extra_details={
                "new_context_revealed": useful_investigation,
                "required_investigation_count": len(required_tools),
                "hidden_context_remaining_count": progress["remaining_count"],
                "hidden_context_revealed_count": progress["revealed_count"],
                "context_completeness": progress["completeness"],
                "tool_name": action.tool_name,
            },
        )
        self._state.history_entries.append(
            self._build_history_entry(
                current_ticket,
                predicted=action.model_dump(exclude_none=True),
                score=investigation_score,
                breakdown={},
                queue_position=idx + 1,
                reward=investigation_reward,
                reward_kind="investigation",
                tool_result=tool_result,
                reward_components=reward_components,
            )
        )
        self._state.last_reward_components = reward_components
        return self._build_observation(task, done=False, reward=investigation_reward)

    def _build_ticket_view(self, ticket: HelpdeskTicketRecord) -> dict[str, Any]:
        progress = self._tool_progress_for_ticket(ticket)
        remaining_tools = progress["remaining_tools"]
        ticket_view: dict[str, Any] = {
            "ticket_id": ticket.ticket_id,
            "title": self._visible_title(ticket),
            "requester": ticket.requester,
            "description": self._visible_description(ticket),
        }
        if progress["required_tools"]:
            ticket_view["context_status"] = {
                "investigation_required": True,
                "hidden_context_remaining": bool(progress["remaining_count"]),
                "context_gap_count": progress["remaining_count"],
                "revealed_context_count": progress["revealed_count"],
                "context_completeness": progress["completeness"],
                "investigations_used_for_ticket": progress["revealed_count"],
                "recommended_tools": list(remaining_tools),
            }
        if ticket.ambiguity_note is not None and "lookup_internal_routing_note" not in remaining_tools:
            ticket_view["ambiguity_note"] = ticket.ambiguity_note
        if ticket.related_ticket_id is not None and "lookup_related_ticket" not in remaining_tools:
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

    def _build_feedback_summary(
        self,
        *,
        predicted: dict[str, Any],
        score: float,
        breakdown: dict[str, float],
        reward: float | None = None,
        rubric_reward: float | None = None,
        reward_kind: str | None = None,
        penalty_reason: str | None = None,
        tool_result: dict[str, Any] | None = None,
        reward_components: dict[str, Any] | None = None,
    ) -> str:
        parts: list[str] = []

        if reward_kind == "investigation":
            tool_name = predicted.get("tool_name") or (tool_result or {}).get("tool_name")
            parts.append(f"Investigation step used {tool_name or 'a tool'}")
            if reward_components and reward_components.get("new_context_revealed"):
                parts.append("new context was revealed")
        elif penalty_reason is not None:
            parts.append(f"Penalty applied: {penalty_reason}")
        else:
            parts.append(f"Ticket score={score:.2f}")

        if breakdown:
            field_scores = ", ".join(
                f"{field}={value:.2f}" for field, value in sorted(breakdown.items())
            )
            parts.append(f"field_scores[{field_scores}]")
        if reward is not None:
            parts.append(f"reward={reward:.2f}")
        if rubric_reward is not None:
            parts.append(f"rubric_reward={rubric_reward:.2f}")
        if reward_components:
            context_gap_penalty = reward_components.get("context_gap_penalty")
            if context_gap_penalty:
                parts.append(f"context_gap_penalty={context_gap_penalty:.2f}")
            hidden_context_remaining_count = reward_components.get(
                "hidden_context_remaining_count"
            )
            if hidden_context_remaining_count:
                parts.append(
                    f"hidden_context_remaining={hidden_context_remaining_count}"
                )
            context_completion_bonus = reward_components.get("context_completion_bonus")
            if context_completion_bonus:
                parts.append(f"context_bonus={context_completion_bonus:.2f}")
            risk_penalty = reward_components.get("risk_penalty")
            if risk_penalty:
                parts.append(f"risk_penalty={risk_penalty:.2f}")

        return "; ".join(parts)

    def _build_history_entry(
        self,
        ticket: HelpdeskTicketRecord,
        *,
        predicted: dict[str, Any],
        score: float,
        breakdown: dict[str, float],
        queue_position: int,
        reward: float | None = None,
        rubric_reward: float | None = None,
        reward_kind: str | None = None,
        penalty_reason: str | None = None,
        tool_result: dict[str, Any] | None = None,
        reward_components: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        progress = self._tool_progress_for_ticket(ticket)
        remaining_tools = progress["remaining_tools"]
        history_entry: dict[str, Any] = {
            "ticket_id": ticket.ticket_id,
            "title": ticket.title,
            "requester": ticket.requester,
            "predicted": predicted,
            "score": score,
            "breakdown": breakdown,
            "queue_position": queue_position,
        }
        if reward is not None:
            history_entry["reward"] = reward
        if rubric_reward is not None:
            history_entry["rubric_reward"] = rubric_reward
        if reward_kind is not None:
            history_entry["reward_kind"] = reward_kind
        if ticket.ambiguity_note is not None and "lookup_internal_routing_note" not in remaining_tools:
            history_entry["ambiguity_note"] = ticket.ambiguity_note
        if ticket.related_ticket_id is not None and "lookup_related_ticket" not in remaining_tools:
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
        if reward_components is not None:
            history_entry["reward_components"] = reward_components
        if progress["required_tools"]:
            history_entry["context_progress"] = {
                "hidden_context_remaining": bool(progress["remaining_count"]),
                "context_gap_count": progress["remaining_count"],
                "revealed_context_count": progress["revealed_count"],
                "context_completeness": progress["completeness"],
            }
        history_entry["feedback_summary"] = self._build_feedback_summary(
            predicted=predicted,
            score=score,
            breakdown=breakdown,
            reward=reward,
            rubric_reward=rubric_reward,
            reward_kind=reward_kind,
            penalty_reason=penalty_reason,
            tool_result=tool_result,
            reward_components=reward_components,
        )
        return history_entry

    def _build_observation(
        self,
        task: dict,
        done: bool = False,
        reward: float | None = None,
        rubric_reward: float | None = None,
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
        last_history_entry = history[-1] if history else None
        tickets_remaining = max(0, queue_size - idx)
        tickets_after_current = max(
            0,
            tickets_remaining - (1 if ticket_view is not None else 0),
        )
        progress_fraction = (idx / queue_size) if queue_size else 0.0

        metadata = {
            "queue_position": queue_position,
            "tickets_remaining_includes_current": ticket_view is not None,
            "has_ambiguity_note": bool(ticket_view and ticket_view.get("ambiguity_note")),
            "has_related_ticket_context": bool(
                ticket_view and ticket_view.get("related_ticket_preview")
            ),
            "has_hidden_context": bool(
                ticket_view
                and (ticket_view.get("context_status") or {}).get("hidden_context_remaining")
            ),
            "action_mode": "investigate_or_submit",
            "available_action_types": list(AVAILABLE_ACTION_TYPES),
            "average_score_so_far": self._state.average_score_so_far,
            "progress_fraction": progress_fraction,
            "investigation_penalty_applied": self._state.investigation_penalty_applied,
        }
        if last_history_entry is not None:
            metadata["last_score"] = last_history_entry.get("score")
            metadata["last_reward"] = last_history_entry.get("reward")
            metadata["last_reward_kind"] = last_history_entry.get("reward_kind")
            metadata["last_breakdown"] = last_history_entry.get("breakdown")
            metadata["last_feedback_summary"] = last_history_entry.get("feedback_summary")
            metadata["last_reward_components"] = last_history_entry.get("reward_components", {})
            if "penalty_reason" in last_history_entry:
                metadata["last_penalty_reason"] = last_history_entry["penalty_reason"]

        return HelpdeskTicketObservation(
            done=done,
            reward=reward,
            rubric_reward=rubric_reward,
            metadata=metadata,
            task_id=task["id"],
            task_name=task["name"],
            instructions=task["instructions"],
            allowed_fields=list(task["allowed_fields"]),
            available_action_types=list(AVAILABLE_ACTION_TYPES),
            available_tools=list(AVAILABLE_TOOLS),
            investigation_budget_remaining=self._state.investigation_budget_remaining,
            last_tool_result=self._state.last_tool_result,
            current_ticket=ticket_view,
            queue_size=queue_size,
            tickets_remaining=tickets_remaining,
            tickets_after_current=tickets_after_current,
            tickets_processed=idx,
            queue_position=queue_position,
            average_score_so_far=self._state.average_score_so_far,
            progress_fraction=progress_fraction,
            history=history,
            last_reward_components=dict(self._state.last_reward_components),
        )
