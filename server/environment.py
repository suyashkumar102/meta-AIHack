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
BASE_AVAILABLE_TOOLS = (
    "lookup_related_ticket",
    "lookup_requester_history",
    "lookup_internal_routing_note",
    "lookup_queue_capacity_forecast",
    "lookup_queue_cluster_summary",
)
TASK_AVAILABLE_ACTION_TYPES: dict[int, tuple[str, ...]] = {
    1: ("submit", "investigate"),
    2: ("submit", "investigate", "request_info", "defer"),
    3: ("submit", "investigate", "request_info", "defer", "open_incident"),
}
TASK_AVAILABLE_TOOLS: dict[int, tuple[str, ...]] = {
    1: (
        "lookup_related_ticket",
        "lookup_requester_history",
        "lookup_internal_routing_note",
    ),
    2: (
        "lookup_related_ticket",
        "lookup_requester_history",
        "lookup_internal_routing_note",
        "lookup_queue_cluster_summary",
    ),
    3: BASE_AVAILABLE_TOOLS,
}
FREE_INVESTIGATIONS_PER_TICKET = 1
EXTRA_INVESTIGATION_COST = 0.04
MAX_EXTRA_INVESTIGATION_PENALTY = 0.25
USEFUL_INVESTIGATION_REWARD = 0.03
USEFUL_REQUEST_INFO_REWARD = 0.025
INCIDENT_OPEN_REWARD = 0.03
REQUEST_INFO_CONTEXT_COMPLETION_BONUS = 0.02
PREMATURE_SUBMIT_PENALTY = 0.22
NONDEFAULT_HIDDEN_CONTEXT_PENALTY = 0.08
CONTEXT_COMPLETION_BONUS = 0.06
TRAJECTORY_CONTEXT_COMPLETION_BONUS = 0.04
PRIORITY_UNDERSHOOT_PENALTY = 0.03
SEVERE_PRIORITY_UNDERSHOOT_PENALTY = 0.07
DANGEROUS_RESOLUTION_PENALTY = 0.05
NONDEFAULT_ROUTING_FOLLOWTHROUGH_BONUS = 0.02
TEAM_CAPACITY_OVERFLOW_PENALTY = 0.08
HIGH_PRIORITY_SLOT_OVERFLOW_PENALTY = 0.06
ESCALATION_SLOT_OVERFLOW_PENALTY = 0.05
PLANNING_SUCCESS_BONUS = 0.05
INCIDENT_SLOT_OVERFLOW_PENALTY = 0.05
INCIDENT_GAP_PENALTY = 0.07
SLA_BREACH_PENALTY = 0.04
FOLLOW_UP_SPAWN_THRESHOLD = 0.72
MAX_DEFERS_PER_TICKET = 1
CLUSTER_STABILIZE_SCORE_THRESHOLD = 0.84
CLUSTER_DESTABILIZE_SCORE_THRESHOLD = 0.72
CLUSTER_INCIDENT_RELIEF_MULTIPLIER = 0.94
CLUSTER_OWNER_RELIEF_MULTIPLIER = 0.86
TASK_QUEUE_MANAGEMENT_WEIGHT: dict[int, float] = {
    1: 0.0,
    2: 0.2,
    3: 0.32,
}

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
        self._queue = self._sample_queue(task_id, min(queue_size, len(self._dataset)))
        (
            team_capacity_initial,
            high_priority_slots_initial,
            escalation_slots_initial,
            incident_slots_initial,
        ) = self._initial_capacity_state_for_queue(task_id)

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
            planning_penalty_applied=0.0,
            last_reward_components={},
            ticket_tool_usage={},
            team_capacity_initial=team_capacity_initial,
            team_capacity_remaining=dict(team_capacity_initial),
            high_priority_slots_initial=high_priority_slots_initial,
            high_priority_slots_remaining=high_priority_slots_initial,
            escalation_slots_initial=escalation_slots_initial,
            escalation_slots_remaining=escalation_slots_initial,
            incident_slots_initial=incident_slots_initial,
            incident_slots_remaining=incident_slots_initial,
            planning_penalty_total=0.0,
            capacity_pressure_tickets_resolved=0,
            cluster_stabilizations_total=0,
            cluster_destabilizations_total=0,
            ticket_request_info_usage={},
            ticket_defer_counts={},
            open_incident_ticket_ids=[],
            incident_actions_used=0,
            incident_gap_total=0.0,
            deferred_ticket_count=0,
            sla_breach_count=0,
            spawned_follow_up_ticket_ids=[],
            spawned_follow_up_source_ids=[],
            dynamic_queue_events=[],
            queue_management_score=0.0,
            queue_management_breakdown={},
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
        if action.action_type not in self._available_action_types_for_task(task_id):
            raise ValueError(
                f"Unsupported action_type {action.action_type!r} for task {task_id}"
            )

        if action.action_type == "investigate":
            return self._handle_investigation_action(task, current_ticket, action, idx)
        if action.action_type == "request_info":
            return self._handle_request_info_action(task, current_ticket, action, idx)
        if action.action_type == "defer":
            return self._handle_defer_action(task, current_ticket, action, idx)
        if action.action_type == "open_incident":
            return self._handle_open_incident_action(task, current_ticket, action, idx)

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
            rubric_details: dict[str, Any] = {}
            if is_done:
                trajectory_components = compute_trajectory_adjustments(
                    self._state.per_ticket_scores,
                    len(self._queue),
                    self._state.step_count,
                    completion_bonus=self._trajectory_consistency_bonus(),
                )
                trajectory_reward = trajectory_components["final_reward"]
                final_reward, rubric_details = self._finalize_terminal_rubric(
                    trajectory_reward
                )
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
                    **rubric_details,
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
        incident_gap_penalty = self._incident_gap_penalty(current_ticket, action)
        capacity_penalty, capacity_details = self._apply_capacity_usage(
            current_ticket,
            action,
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
        rubric_details: dict[str, Any] = {}

        if is_done:
            self._state.per_ticket_scores.append(score)
            self._state.average_score_so_far = self._current_average_score()
            self._state.step_count += 1
            self._state.current_ticket_index += 1
            trajectory_components = compute_trajectory_adjustments(
                self._state.per_ticket_scores,
                len(self._queue),
                self._state.step_count,
                completion_bonus=(
                    self._trajectory_consistency_bonus() + self._planning_success_bonus()
                ),
            )
            trajectory_reward = trajectory_components["final_reward"]
            rubric_reward, rubric_details = self._finalize_terminal_rubric(
                trajectory_reward
            )
            final_reward = clamp_open_unit_interval(
                rubric_reward - context_penalty - capacity_penalty - incident_gap_penalty
            )
            self._state.total_reward = rubric_reward
            investigation_penalty = self._compute_episode_penalty()
        else:
            self._state.per_ticket_scores.append(score)
            self._state.average_score_so_far = self._current_average_score()
            self._state.step_count += 1
            self._state.current_ticket_index += 1
            final_reward = clamp_open_unit_interval(
                step_reward - context_penalty - capacity_penalty - incident_gap_penalty
            )

        spawned_follow_up_ticket_id = None
        if self._should_spawn_follow_up(
            current_ticket,
            score=score,
            context_penalty=context_penalty,
            incident_gap_penalty=incident_gap_penalty,
        ):
            spawned_follow_up = self._spawn_follow_up_ticket(current_ticket)
            spawned_follow_up_ticket_id = spawned_follow_up.ticket_id
            if is_done:
                is_done = False
                trajectory_reward = None
                trajectory_components = None
                rubric_reward = None
                rubric_details = {}
                final_reward = clamp_open_unit_interval(
                    step_reward - context_penalty - capacity_penalty - incident_gap_penalty
                )
                self._state.total_reward = 0.0
                self._state.queue_management_score = 0.0
                self._state.queue_management_breakdown = {}
        if incident_gap_penalty > 0.0:
            self._state.incident_gap_total = round(
                self._state.incident_gap_total + incident_gap_penalty,
                4,
            )
        cluster_stabilized_ticket_ids = self._stabilize_future_cluster_tickets(
            current_ticket,
            score=score,
            context_penalty=context_penalty,
            incident_gap_penalty=incident_gap_penalty,
        )
        cluster_destabilized_ticket_ids: list[str] = []
        if not cluster_stabilized_ticket_ids:
            cluster_destabilized_ticket_ids = self._destabilize_future_cluster_tickets(
                current_ticket,
                score=score,
                context_penalty=context_penalty,
                incident_gap_penalty=incident_gap_penalty,
            )

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
                "incident_gap_penalty": incident_gap_penalty,
                "capacity_penalty": capacity_penalty,
                "delta_adjustment": step_adjustments["delta_adjustment"],
                "required_investigation_count": len(self._required_tools_for_ticket(current_ticket)),
                "hidden_context_remaining_count": missing_required_count,
                "hidden_context_revealed_count": len(
                    self._used_tools_for_ticket(current_ticket.ticket_id)
                ),
                "planning_penalty_total": self._state.planning_penalty_total,
                "planning_penalty_applied": self._state.planning_penalty_applied,
                "planning_success_bonus": self._planning_success_bonus()
                if is_done
                else 0.0,
                "spawned_follow_up_ticket_id": spawned_follow_up_ticket_id,
                "cluster_stabilized_ticket_ids": cluster_stabilized_ticket_ids,
                "cluster_destabilized_ticket_ids": cluster_destabilized_ticket_ids,
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
                **rubric_details,
            },
        )
        reward_components.update(capacity_details)

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
        self._state.planning_penalty_applied = capacity_penalty
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

    def _queue_management_blend_weight(self, task_id: int | None = None) -> float:
        resolved_task_id = self._state.current_task_id if task_id is None else task_id
        return TASK_QUEUE_MANAGEMENT_WEIGHT.get(int(resolved_task_id or 1), 0.0)

    def _context_resolution_score(self) -> float:
        hidden_context_tickets = [
            ticket
            for ticket in self._queue
            if self._required_tools_for_ticket(ticket, self._state.current_task_id)
        ]
        if not hidden_context_tickets:
            return 1.0
        total_required = 0
        total_resolved = 0
        for ticket in hidden_context_tickets:
            progress = self._tool_progress_for_ticket(ticket)
            total_required += max(1, len(progress["required_tools"]))
            total_resolved += max(
                0,
                len(progress["required_tools"]) - len(progress["remaining_tools"]),
            )
        return round(
            max(0.0, min(1.0, total_resolved / max(1, total_required))),
            4,
        )

    def _follow_up_containment_score(self) -> float:
        follow_up_risk_tickets = [
            ticket
            for ticket in self._queue
            if ticket.generated_from_ticket_id is None
            and (
                self._requires_incident(ticket)
                or self._ticket_mentions_follow_up(ticket)
                or ticket.related_ticket_id is not None
                or ticket.priority in {"high", "critical"}
            )
        ]
        if not follow_up_risk_tickets:
            return 1.0
        spawn_rate = len(self._state.spawned_follow_up_ticket_ids) / max(
            1,
            len(follow_up_risk_tickets),
        )
        generated_follow_up_scores = [
            float(entry.get("score", 0.0))
            for entry in self._state.history_entries
            if entry.get("generated_from_ticket_id") is not None
        ]
        recovery_credit = (
            sum(generated_follow_up_scores) / len(generated_follow_up_scores)
            if generated_follow_up_scores
            else 0.0
        )
        score = (1.0 - min(1.0, 0.7 * spawn_rate)) + (
            min(1.0, spawn_rate) * 0.3 * recovery_credit
        )
        return round(max(0.0, min(1.0, score)), 4)

    def _incident_management_score(self) -> float:
        if (self._state.current_task_id or 1) < 3:
            return 1.0
        incident_sensitive_tickets = [
            ticket
            for ticket in self._queue
            if ticket.generated_from_ticket_id is None and self._requires_incident(ticket)
        ]
        if not incident_sensitive_tickets:
            return 1.0
        coverage_ratio = sum(
            1 for ticket in incident_sensitive_tickets if self._incident_open_for_ticket(ticket)
        ) / max(1, len(incident_sensitive_tickets))
        gap_ratio = min(
            1.0,
            self._state.incident_gap_total
            / max(
                INCIDENT_GAP_PENALTY,
                len(incident_sensitive_tickets) * INCIDENT_GAP_PENALTY,
            ),
        )
        score = (0.65 * (1.0 - gap_ratio)) + (0.35 * coverage_ratio)
        return round(max(0.0, min(1.0, score)), 4)

    def _sla_quality_score(self) -> float:
        breach_denominator = max(1, self._state.deferred_ticket_count or len(self._queue))
        breach_ratio = min(1.0, self._state.sla_breach_count / breach_denominator)
        score = 1.0 - breach_ratio
        return round(max(0.0, min(1.0, score)), 4)

    def _planning_quality_score(self) -> float:
        if (self._state.current_task_id or 1) < 3:
            return 1.0
        capacity_sensitive_count = sum(
            1 for ticket in self._queue if self._ticket_has_alternate_route(ticket)
        )
        route_coverage = (
            min(
                1.0,
                self._state.capacity_pressure_tickets_resolved / capacity_sensitive_count,
            )
            if capacity_sensitive_count
            else 1.0
        )
        max_expected_penalty = max(
            0.12,
            len(self._queue)
            * (
                TEAM_CAPACITY_OVERFLOW_PENALTY
                + HIGH_PRIORITY_SLOT_OVERFLOW_PENALTY
                + ESCALATION_SLOT_OVERFLOW_PENALTY
            ),
        )
        penalty_score = 1.0 - min(
            1.0,
            self._state.planning_penalty_total / max_expected_penalty,
        )
        score = (0.6 * penalty_score) + (0.4 * route_coverage)
        return round(max(0.0, min(1.0, score)), 4)

    def _cluster_coordination_score(self) -> float:
        if (self._state.current_task_id or 1) < 2:
            return 1.0
        clustered_tickets = [
            ticket
            for ticket in self._queue
            if ticket.service_cluster_id
            or ticket.related_ticket_id is not None
            or ticket.generated_from_ticket_id is not None
            or self._ticket_repeated_requester_count(ticket) >= 2
        ]
        if not clustered_tickets:
            return 1.0
        cluster_count = max(1, len(clustered_tickets))
        destabilization_ratio = min(
            1.0,
            self._state.cluster_destabilizations_total / cluster_count,
        )
        stabilization_ratio = min(
            1.0,
            self._state.cluster_stabilizations_total / cluster_count,
        )
        score = 1.0 - (0.75 * destabilization_ratio) + (0.25 * stabilization_ratio)
        return round(max(0.0, min(1.0, score)), 4)

    def _queue_management_breakdown(self, trajectory_reward: float) -> tuple[float, dict[str, Any]]:
        task_id = int(self._state.current_task_id or 1)
        if task_id < 2:
            proxy_score = round(clamp_open_unit_interval(trajectory_reward), 4)
            return proxy_score, {"routing_trajectory_proxy": proxy_score}

        component_scores: dict[str, float] = {
            "context_resolution": self._context_resolution_score(),
            "cluster_coordination": self._cluster_coordination_score(),
            "follow_up_containment": self._follow_up_containment_score(),
            "sla_management": self._sla_quality_score(),
        }
        if task_id >= 3:
            component_scores["planning_quality"] = self._planning_quality_score()
            component_scores["incident_management"] = self._incident_management_score()
            component_weights = {
                "context_resolution": 0.2,
                "planning_quality": 0.24,
                "incident_management": 0.2,
                "cluster_coordination": 0.16,
                "follow_up_containment": 0.12,
                "sla_management": 0.08,
            }
        else:
            component_weights = {
                "context_resolution": 0.38,
                "cluster_coordination": 0.26,
                "follow_up_containment": 0.2,
                "sla_management": 0.16,
            }

        aggregate_score = round(
            sum(
                component_scores[name] * weight
                for name, weight in component_weights.items()
            ),
            4,
        )
        breakdown: dict[str, Any] = {
            name: round(score, 4) for name, score in component_scores.items()
        }
        breakdown["weights"] = {
            name: round(weight, 4) for name, weight in component_weights.items()
        }
        breakdown["cluster_stabilizations_total"] = self._state.cluster_stabilizations_total
        breakdown["cluster_destabilizations_total"] = self._state.cluster_destabilizations_total
        breakdown["spawned_follow_up_count"] = len(self._state.spawned_follow_up_ticket_ids)
        breakdown["sla_breach_count"] = self._state.sla_breach_count
        breakdown["planning_penalty_total"] = round(self._state.planning_penalty_total, 4)
        breakdown["incident_gap_total"] = round(self._state.incident_gap_total, 4)
        breakdown["aggregate"] = aggregate_score
        return aggregate_score, breakdown

    def _finalize_terminal_rubric(
        self,
        trajectory_reward: float,
    ) -> tuple[float, dict[str, Any]]:
        task_id = int(self._state.current_task_id or 1)
        queue_management_score, queue_management_breakdown = self._queue_management_breakdown(
            trajectory_reward
        )
        route_weight = round(1.0 - self._queue_management_blend_weight(task_id), 4)
        queue_weight = round(self._queue_management_blend_weight(task_id), 4)
        blended_reward = clamp_open_unit_interval(
            (route_weight * trajectory_reward) + (queue_weight * queue_management_score)
        )
        episode_economics_penalty = round(self._compute_episode_penalty(), 4)
        rubric_reward = self._apply_episode_economics(blended_reward)
        self._state.queue_management_score = queue_management_score
        self._state.queue_management_breakdown = dict(queue_management_breakdown)
        return rubric_reward, {
            "trajectory_routing_reward": trajectory_reward,
            "queue_management_score": queue_management_score,
            "queue_management_breakdown": dict(queue_management_breakdown),
            "route_objective_weight": route_weight,
            "queue_management_weight": queue_weight,
            "blended_objective_before_economics": blended_reward,
            "episode_economics_penalty": episode_economics_penalty,
        }

    def _available_action_types_for_task(self, task_id: int | None = None) -> list[str]:
        resolved_task_id = self._state.current_task_id if task_id is None else task_id
        return list(TASK_AVAILABLE_ACTION_TYPES.get(int(resolved_task_id or 1), ("submit",)))

    def _available_tools_for_task(self, task_id: int | None = None) -> list[str]:
        resolved_task_id = self._state.current_task_id if task_id is None else task_id
        return list(TASK_AVAILABLE_TOOLS.get(int(resolved_task_id or 1), ()))

    def _sync_queue_ticket_ids(self) -> None:
        self._state.queue_ticket_ids = [ticket.ticket_id for ticket in self._queue]

    def _cluster_sample_groups(self) -> list[list[HelpdeskTicketRecord]]:
        groups: dict[str, list[HelpdeskTicketRecord]] = {}
        for ticket in self._dataset:
            if not ticket.service_cluster_id:
                continue
            groups.setdefault(ticket.service_cluster_id, []).append(ticket)
        return [tickets for tickets in groups.values() if len(tickets) >= 2]

    def _cluster_ticket_order_key(self, ticket: HelpdeskTicketRecord) -> tuple[int, int, str]:
        priority_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        follow_up_depth = 1 if ticket.related_ticket_id or ticket.generated_from_ticket_id else 0
        return (
            follow_up_depth,
            priority_rank.get(ticket.priority, 4),
            ticket.ticket_id,
        )

    def _sample_queue(self, task_id: int, queue_size: int) -> list[HelpdeskTicketRecord]:
        if queue_size <= 0:
            return []
        if task_id not in {2, 3} or queue_size < 3:
            return self._rng.sample(self._dataset, queue_size)

        cluster_groups = self._cluster_sample_groups()
        if not cluster_groups:
            return self._rng.sample(self._dataset, queue_size)

        chosen_group = self._rng.choice(cluster_groups)
        max_cluster_take = min(len(chosen_group), 3 if queue_size >= 4 else 2)
        cluster_take = max(2, min(max_cluster_take, queue_size - 1))
        cluster_subset = self._rng.sample(chosen_group, cluster_take)
        cluster_subset_ids = {ticket.ticket_id for ticket in cluster_subset}

        filler_count = max(0, queue_size - len(cluster_subset))
        remaining_pool = [
            ticket for ticket in self._dataset if ticket.ticket_id not in cluster_subset_ids
        ]
        filler_subset = (
            self._rng.sample(remaining_pool, filler_count) if filler_count > 0 else []
        )

        ordered_cluster = sorted(cluster_subset, key=self._cluster_ticket_order_key)
        remaining_cluster = ordered_cluster[1:]
        ordered_queue: list[HelpdeskTicketRecord] = []
        if ordered_cluster:
            ordered_queue.append(ordered_cluster[0])
        while filler_subset or remaining_cluster:
            if filler_subset:
                ordered_queue.append(filler_subset.pop(0))
            if remaining_cluster:
                ordered_queue.append(remaining_cluster.pop(0))
        return ordered_queue[:queue_size]

    def _cluster_keys_for_ticket(self, ticket: HelpdeskTicketRecord) -> set[str]:
        keys: set[str] = set()
        if ticket.service_cluster_id:
            keys.add(f"cluster:{ticket.service_cluster_id}")
        if ticket.related_ticket_id:
            keys.add(f"ticket:{ticket.related_ticket_id}")
        if ticket.generated_from_ticket_id:
            keys.add(f"ticket:{ticket.generated_from_ticket_id}")
        if any(
            candidate.ticket_id != ticket.ticket_id
            and (
                candidate.related_ticket_id == ticket.ticket_id
                or candidate.generated_from_ticket_id == ticket.ticket_id
            )
            for candidate in self._tickets_by_id.values()
        ):
            keys.add(f"ticket:{ticket.ticket_id}")
        if self._ticket_repeated_requester_count(ticket) >= 2:
            keys.add(f"requester:{ticket.requester}")
        return keys

    def _tickets_share_cluster(
        self,
        first: HelpdeskTicketRecord,
        second: HelpdeskTicketRecord,
    ) -> bool:
        if first.ticket_id == second.ticket_id:
            return False
        return bool(self._cluster_keys_for_ticket(first) & self._cluster_keys_for_ticket(second))

    def _future_cluster_ticket_indexes(
        self,
        ticket: HelpdeskTicketRecord,
        *,
        start_index: int,
    ) -> list[int]:
        indexes: list[int] = []
        for index in range(start_index, len(self._queue)):
            future_ticket = self._queue[index]
            if self._tickets_share_cluster(ticket, future_ticket):
                indexes.append(index)
        return indexes

    def _ticket_queue_index(self, ticket: HelpdeskTicketRecord) -> int | None:
        for index, candidate in enumerate(self._queue):
            if candidate.ticket_id == ticket.ticket_id:
                return index
        return None

    def _cluster_summary(
        self,
        ticket: HelpdeskTicketRecord,
        *,
        start_index: int | None = None,
    ) -> dict[str, Any]:
        if start_index is None:
            ticket_index = self._ticket_queue_index(ticket)
            effective_start = (
                ticket_index + 1
                if ticket_index is not None
                else self._state.current_ticket_index + 1
            )
        else:
            effective_start = start_index
        future_indexes = self._future_cluster_ticket_indexes(
            ticket,
            start_index=effective_start,
        )
        future_tickets = [self._queue[index] for index in future_indexes]
        return {
            "service_cluster_id": ticket.service_cluster_id,
            "cluster_keys": sorted(self._cluster_keys_for_ticket(ticket)),
            "future_cluster_ticket_count": len(future_tickets),
            "future_cluster_ticket_ids": [candidate.ticket_id for candidate in future_tickets],
            "future_high_priority_count": sum(
                1 for candidate in future_tickets if candidate.priority in {"high", "critical"}
            ),
            "shared_requester_count": self._ticket_repeated_requester_count(ticket),
            "active_incident_cover": self._incident_open_for_ticket(ticket),
        }

    def _append_note(self, existing_note: str | None, addition: str | None) -> str | None:
        if not addition:
            return existing_note
        if not existing_note:
            return addition
        if addition in existing_note:
            return existing_note
        return f"{existing_note} {addition}"

    def _replace_queue_ticket(
        self,
        index: int,
        updated_ticket: HelpdeskTicketRecord,
    ) -> None:
        self._queue[index] = updated_ticket
        self._tickets_by_id[updated_ticket.ticket_id] = updated_ticket

    def _stabilize_future_cluster_tickets(
        self,
        current_ticket: HelpdeskTicketRecord,
        *,
        score: float,
        context_penalty: float,
        incident_gap_penalty: float,
    ) -> list[str]:
        if (self._state.current_task_id or 1) < 2:
            return []
        if score < CLUSTER_STABILIZE_SCORE_THRESHOLD:
            return []
        if context_penalty > 0.0 or incident_gap_penalty > 0.0:
            return []

        future_indexes = self._future_cluster_ticket_indexes(
            current_ticket,
            start_index=self._state.current_ticket_index,
        )
        if not future_indexes:
            return []

        incident_cover = self._incident_open_for_ticket(current_ticket)
        relief_multiplier = (
            CLUSTER_INCIDENT_RELIEF_MULTIPLIER
            if incident_cover
            else CLUSTER_OWNER_RELIEF_MULTIPLIER
        )
        planning_note = (
            "An earlier incident bridge is already active for this request cluster, so later "
            "updates can be acknowledged and coordinated instead of being re-triaged from scratch."
            if incident_cover
            else "An earlier ticket in this request cluster already has an accountable owner, "
            "so later updates can be coordinated rather than fully re-triaged."
        )
        customer_note = (
            "The requester said a single coordinated owner is acceptable as long as the update is linked to the existing workstream."
        )
        updated_ticket_ids: list[str] = []
        for index in future_indexes:
            future_ticket = self._queue[index]
            updates: dict[str, Any] = {
                "planning_note": self._append_note(future_ticket.planning_note, planning_note),
                "customer_update_note": self._append_note(
                    future_ticket.customer_update_note,
                    customer_note,
                ),
            }
            if (
                not self._ticket_has_alternate_route(future_ticket)
                or future_ticket.alternate_route_score_multiplier < relief_multiplier
            ):
                alternate_priority = (
                    "high"
                    if incident_cover and future_ticket.priority == "critical"
                    else "medium"
                    if incident_cover and future_ticket.priority == "high"
                    else future_ticket.alternate_priority or future_ticket.priority
                )
                updates.update(
                    {
                        "alternate_issue_type": (
                            future_ticket.alternate_issue_type or future_ticket.issue_type
                        ),
                        "alternate_priority": alternate_priority,
                        "alternate_assignment_group": "service_desk",
                        "alternate_resolution_action": (
                            "acknowledge" if incident_cover else "assign"
                        ),
                        "alternate_route_score_multiplier": relief_multiplier,
                    }
                )
            updated_ticket = future_ticket.model_copy(update=updates)
            self._replace_queue_ticket(index, updated_ticket)
            updated_ticket_ids.append(updated_ticket.ticket_id)

        if updated_ticket_ids:
            self._state.cluster_stabilizations_total += len(updated_ticket_ids)
            self._record_dynamic_queue_event(
                "stabilize_cluster",
                source_ticket_id=current_ticket.ticket_id,
                affected_ticket_ids=updated_ticket_ids,
                incident_cover=incident_cover,
            )
        return updated_ticket_ids

    def _destabilize_future_cluster_tickets(
        self,
        current_ticket: HelpdeskTicketRecord,
        *,
        score: float,
        context_penalty: float,
        incident_gap_penalty: float,
    ) -> list[str]:
        if (self._state.current_task_id or 1) < 2:
            return []
        if score >= CLUSTER_DESTABILIZE_SCORE_THRESHOLD:
            if context_penalty <= 0.0 and incident_gap_penalty <= 0.0:
                return []

        future_indexes = self._future_cluster_ticket_indexes(
            current_ticket,
            start_index=self._state.current_ticket_index,
        )
        if not future_indexes:
            return []

        planning_note = (
            "Earlier handling in this request cluster did not settle ownership, so this follow-on "
            "arrives with more urgency and may need firmer coordination."
        )
        customer_note = (
            "The requester is escalating because the earlier response did not fully resolve the blocker."
        )
        updated_ticket_ids: list[str] = []
        for index in future_indexes:
            future_ticket = self._queue[index]
            updates: dict[str, Any] = {
                "priority": self._escalate_priority_level(future_ticket.priority),
                "planning_note": self._append_note(future_ticket.planning_note, planning_note),
                "customer_update_note": self._append_note(
                    future_ticket.customer_update_note,
                    customer_note,
                ),
                "incident_recommended": (
                    future_ticket.incident_recommended
                    or current_ticket.priority in {"high", "critical"}
                    or self._requires_incident(current_ticket)
                ),
            }
            if future_ticket.related_ticket_id is None:
                updates["related_ticket_id"] = current_ticket.ticket_id
            updated_ticket = future_ticket.model_copy(update=updates)
            self._replace_queue_ticket(index, updated_ticket)
            updated_ticket_ids.append(updated_ticket.ticket_id)

        if updated_ticket_ids:
            self._state.cluster_destabilizations_total += len(updated_ticket_ids)
            self._record_dynamic_queue_event(
                "destabilize_cluster",
                source_ticket_id=current_ticket.ticket_id,
                affected_ticket_ids=updated_ticket_ids,
            )
        return updated_ticket_ids

    def _ticket_has_alternate_route(self, ticket: HelpdeskTicketRecord) -> bool:
        return any(
            value is not None
            for value in (
                ticket.alternate_issue_type,
                ticket.alternate_priority,
                ticket.alternate_assignment_group,
                ticket.alternate_resolution_action,
            )
        ) and ticket.alternate_route_score_multiplier > 0.0

    def _route_for_ticket(
        self,
        ticket: HelpdeskTicketRecord,
        *,
        use_alternate: bool = False,
    ) -> dict[str, str]:
        if use_alternate and self._ticket_has_alternate_route(ticket):
            return {
                "issue_type": ticket.alternate_issue_type or ticket.issue_type,
                "priority": ticket.alternate_priority or ticket.priority,
                "assignment_group": (
                    ticket.alternate_assignment_group or ticket.assignment_group
                ),
                "resolution_action": (
                    ticket.alternate_resolution_action or ticket.resolution_action
                ),
            }
        return {
            "issue_type": ticket.issue_type,
            "priority": ticket.priority,
            "assignment_group": ticket.assignment_group,
            "resolution_action": ticket.resolution_action,
        }

    def _route_for_action(
        self,
        ticket: HelpdeskTicketRecord,
        action: HelpdeskTicketAction,
    ) -> dict[str, str]:
        primary_route = self._route_for_ticket(ticket)
        return {
            "issue_type": action.issue_type or primary_route["issue_type"],
            "priority": action.priority or primary_route["priority"],
            "assignment_group": (
                action.assignment_group or primary_route["assignment_group"]
            ),
            "resolution_action": (
                action.resolution_action or primary_route["resolution_action"]
            ),
        }

    def _route_capacity_cost(self, route: dict[str, str]) -> dict[str, Any]:
        return {
            "assignment_group": route["assignment_group"],
            "team_slots": 1,
            "high_priority_slots": 1
            if route["priority"] in {"high", "critical"}
            else 0,
            "escalation_slots": 1
            if route["resolution_action"] in {"assign", "escalate"}
            else 0,
        }

    def _routing_options_for_ticket(self, ticket: HelpdeskTicketRecord) -> list[dict[str, Any]]:
        options = [
            {
                "label": "primary",
                "score_multiplier": 1.0,
                **self._route_for_ticket(ticket),
                "capacity_cost": self._route_capacity_cost(self._route_for_ticket(ticket)),
            }
        ]
        if self._ticket_has_alternate_route(ticket):
            alternate_route = self._route_for_ticket(ticket, use_alternate=True)
            options.append(
                {
                    "label": "alternate",
                    "score_multiplier": ticket.alternate_route_score_multiplier,
                    **alternate_route,
                    "capacity_cost": self._route_capacity_cost(alternate_route),
                }
            )
        return options

    def _initial_capacity_state_for_queue(
        self,
        task_id: int,
    ) -> tuple[dict[str, int], int, int, int]:
        if task_id != 3:
            return {}, 0, 0, 0

        primary_group_demand: dict[str, int] = {}
        alternate_relief_by_group: dict[str, int] = {}
        all_groups: set[str] = set()
        high_priority_demand = 0
        high_priority_relief = 0
        escalation_demand = 0
        escalation_relief = 0
        incident_demand = 0

        for ticket in self._queue:
            primary_route = self._route_for_ticket(ticket)
            all_groups.add(primary_route["assignment_group"])
            primary_group_demand[primary_route["assignment_group"]] = (
                primary_group_demand.get(primary_route["assignment_group"], 0) + 1
            )
            if primary_route["priority"] in {"high", "critical"}:
                high_priority_demand += 1
            if primary_route["resolution_action"] in {"assign", "escalate"}:
                escalation_demand += 1
            if self._requires_incident(ticket):
                incident_demand += 1

            if self._ticket_has_alternate_route(ticket):
                alternate_route = self._route_for_ticket(ticket, use_alternate=True)
                all_groups.add(alternate_route["assignment_group"])
                if alternate_route["assignment_group"] != primary_route["assignment_group"]:
                    alternate_relief_by_group[primary_route["assignment_group"]] = (
                        alternate_relief_by_group.get(
                            primary_route["assignment_group"],
                            0,
                        )
                        + 1
                    )
                if (
                    primary_route["priority"] in {"high", "critical"}
                    and alternate_route["priority"] not in {"high", "critical"}
                ):
                    high_priority_relief += 1
                if (
                    primary_route["resolution_action"] in {"assign", "escalate"}
                    and alternate_route["resolution_action"] not in {"assign", "escalate"}
                ):
                    escalation_relief += 1

        team_capacity_initial: dict[str, int] = {}
        for group in sorted(all_groups):
            demand = primary_group_demand.get(group, 0)
            relief = alternate_relief_by_group.get(group, 0)
            if demand <= 1:
                team_capacity_initial[group] = 1 if group in all_groups else 0
            elif relief > 0:
                team_capacity_initial[group] = max(1, demand - 1)
            else:
                team_capacity_initial[group] = demand

        if high_priority_demand <= 1:
            high_priority_slots_initial = high_priority_demand
        elif high_priority_relief > 0:
            high_priority_slots_initial = max(1, high_priority_demand - 1)
        else:
            high_priority_slots_initial = high_priority_demand

        if escalation_demand <= 1:
            escalation_slots_initial = escalation_demand
        elif escalation_relief > 0:
            escalation_slots_initial = max(1, escalation_demand - 1)
        else:
            escalation_slots_initial = escalation_demand

        if incident_demand <= 1:
            incident_slots_initial = incident_demand
        else:
            incident_slots_initial = max(1, incident_demand - 1)

        return (
            team_capacity_initial,
            high_priority_slots_initial,
            escalation_slots_initial,
            incident_slots_initial,
        )

    def _future_queue_demand(self) -> dict[str, Any]:
        future_tickets = self._queue[self._state.current_ticket_index + 1 :]
        team_demand: dict[str, int] = {}
        high_priority_needed = 0
        escalation_needed = 0
        capacity_sensitive_tickets = 0
        incident_needed = 0
        clustered_follow_ons = 0

        for ticket in future_tickets:
            route = self._route_for_ticket(ticket)
            team_demand[route["assignment_group"]] = (
                team_demand.get(route["assignment_group"], 0) + 1
            )
            if route["priority"] in {"high", "critical"}:
                high_priority_needed += 1
            if route["resolution_action"] in {"assign", "escalate"}:
                escalation_needed += 1
            if self._ticket_has_alternate_route(ticket):
                capacity_sensitive_tickets += 1
            if self._requires_incident(ticket):
                incident_needed += 1
            if self._cluster_keys_for_ticket(ticket):
                clustered_follow_ons += 1

        return {
            "remaining_ticket_count": len(future_tickets),
            "team_demand": team_demand,
            "high_priority_needed": high_priority_needed,
            "escalation_needed": escalation_needed,
            "capacity_sensitive_tickets": capacity_sensitive_tickets,
            "incident_needed": incident_needed,
            "clustered_follow_ons": clustered_follow_ons,
        }

    def _capacity_state_snapshot(self) -> dict[str, Any]:
        return {
            "team_capacity_remaining": dict(self._state.team_capacity_remaining),
            "team_capacity_initial": dict(self._state.team_capacity_initial),
            "high_priority_slots_remaining": self._state.high_priority_slots_remaining,
            "high_priority_slots_initial": self._state.high_priority_slots_initial,
            "escalation_slots_remaining": self._state.escalation_slots_remaining,
            "escalation_slots_initial": self._state.escalation_slots_initial,
            "incident_slots_remaining": self._state.incident_slots_remaining,
            "incident_slots_initial": self._state.incident_slots_initial,
        }

    def _planning_route_recommendation(self, ticket: HelpdeskTicketRecord) -> dict[str, Any]:
        primary_route = self._route_for_ticket(ticket)
        alternate_route = (
            self._route_for_ticket(ticket, use_alternate=True)
            if self._ticket_has_alternate_route(ticket)
            else None
        )
        future_demand = self._future_queue_demand()
        capacity_state = self._capacity_state_snapshot()

        def pressure_score(route: dict[str, str]) -> int:
            cost = self._route_capacity_cost(route)
            group_remaining = capacity_state["team_capacity_remaining"].get(
                route["assignment_group"],
                1,
            )
            group_pressure = max(
                0,
                future_demand["team_demand"].get(route["assignment_group"], 0)
                + cost["team_slots"]
                - group_remaining,
            )
            priority_pressure = max(
                0,
                future_demand["high_priority_needed"] + cost["high_priority_slots"]
                - capacity_state["high_priority_slots_remaining"],
            )
            escalation_pressure = max(
                0,
                future_demand["escalation_needed"] + cost["escalation_slots"]
                - capacity_state["escalation_slots_remaining"],
            )
            return group_pressure + priority_pressure + escalation_pressure

        primary_pressure = pressure_score(primary_route)
        alternate_pressure = (
            pressure_score(alternate_route) if alternate_route is not None else primary_pressure
        )
        preferred_label = (
            "alternate"
            if alternate_route is not None and alternate_pressure < primary_pressure
            else "primary"
        )
        return {
            "preferred_label": preferred_label,
            "primary_pressure": primary_pressure,
            "alternate_pressure": alternate_pressure,
            "capacity_state": capacity_state,
            "future_demand": future_demand,
        }

    def _ticket_is_capacity_sensitive(self, ticket: HelpdeskTicketRecord) -> bool:
        if self._state.current_task_id != 3 or not self._ticket_has_alternate_route(ticket):
            return False
        recommendation = self._planning_route_recommendation(ticket)
        return recommendation["preferred_label"] == "alternate" or any(
            value > 0
            for value in (
                recommendation["primary_pressure"],
                recommendation["alternate_pressure"],
            )
        )

    def _route_matches_alternate(
        self,
        ticket: HelpdeskTicketRecord,
        route: dict[str, str],
    ) -> bool:
        if not self._ticket_has_alternate_route(ticket):
            return False
        return route == self._route_for_ticket(ticket, use_alternate=True)

    def _apply_capacity_usage(
        self,
        ticket: HelpdeskTicketRecord,
        action: HelpdeskTicketAction,
    ) -> tuple[float, dict[str, Any]]:
        if self._state.current_task_id != 3:
            return 0.0, {}

        route = self._route_for_action(ticket, action)
        capacity_cost = self._route_capacity_cost(route)
        group = str(capacity_cost["assignment_group"])

        if group not in self._state.team_capacity_remaining:
            self._state.team_capacity_remaining[group] = 1
            self._state.team_capacity_initial.setdefault(group, 1)

        group_remaining = self._state.team_capacity_remaining[group]
        group_overflow = max(0, int(capacity_cost["team_slots"]) - group_remaining)
        self._state.team_capacity_remaining[group] = max(
            0,
            group_remaining - int(capacity_cost["team_slots"]),
        )

        high_priority_cost = int(capacity_cost["high_priority_slots"])
        high_priority_overflow = max(
            0,
            high_priority_cost - self._state.high_priority_slots_remaining,
        )
        self._state.high_priority_slots_remaining = max(
            0,
            self._state.high_priority_slots_remaining - high_priority_cost,
        )

        escalation_cost = int(capacity_cost["escalation_slots"])
        escalation_overflow = max(
            0,
            escalation_cost - self._state.escalation_slots_remaining,
        )
        self._state.escalation_slots_remaining = max(
            0,
            self._state.escalation_slots_remaining - escalation_cost,
        )

        capacity_penalty = round(
            group_overflow * TEAM_CAPACITY_OVERFLOW_PENALTY
            + high_priority_overflow * HIGH_PRIORITY_SLOT_OVERFLOW_PENALTY
            + escalation_overflow * ESCALATION_SLOT_OVERFLOW_PENALTY,
            4,
        )
        self._state.planning_penalty_total = round(
            self._state.planning_penalty_total + capacity_penalty,
            4,
        )
        self._state.planning_penalty_applied = capacity_penalty

        used_alternate_route = self._route_matches_alternate(ticket, route)
        if used_alternate_route:
            self._state.capacity_pressure_tickets_resolved += 1

        return capacity_penalty, {
            "capacity_cost": capacity_cost,
            "group_overflow": group_overflow,
            "high_priority_overflow": high_priority_overflow,
            "escalation_overflow": escalation_overflow,
            "used_alternate_route": used_alternate_route,
            "capacity_state_after_action": self._capacity_state_snapshot(),
        }

    def _planning_success_bonus(self) -> float:
        if self._state.current_task_id != 3 or self._state.planning_penalty_total > 0.0:
            return 0.0
        capacity_sensitive_count = sum(
            1 for ticket in self._queue if self._ticket_has_alternate_route(ticket)
        )
        if capacity_sensitive_count == 0:
            return 0.0
        coverage = min(
            1.0,
            self._state.capacity_pressure_tickets_resolved / capacity_sensitive_count,
        )
        return round(PLANNING_SUCCESS_BONUS * coverage, 4)

    def _internal_routing_note_for_ticket(
        self,
        ticket: HelpdeskTicketRecord,
    ) -> str | None:
        if self._state.current_task_id != 3:
            return ticket.ambiguity_note or ticket.planning_note

        note_parts: list[str] = []
        if ticket.ambiguity_note is not None:
            note_parts.append(ticket.ambiguity_note)
        if ticket.planning_note is not None:
            note_parts.append(ticket.planning_note)

        default_group = ISSUE_TYPE_TO_ASSIGNMENT_GROUP.get(
            ticket.issue_type,
            ticket.assignment_group,
        )
        default_action = ISSUE_TYPE_TO_RESOLUTION_ACTION.get(
            ticket.issue_type,
            ticket.resolution_action,
        )

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

    def _ticket_text(self, ticket: HelpdeskTicketRecord) -> str:
        return f"{ticket.title} {ticket.description}".lower()

    def _requires_incident(self, ticket: HelpdeskTicketRecord) -> bool:
        if ticket.incident_recommended:
            return True
        text = self._ticket_text(ticket)
        return (
            ticket.priority in {"high", "critical"}
            and ticket.issue_type
            in {"application_support", "identity_access", "security_compliance"}
            and any(
                phrase in text
                for phrase in (
                    "outage",
                    "cannot log in",
                    "login",
                    "regression",
                    "unstable",
                    "blocked",
                    "lockout",
                    "company-wide",
                    "production",
                    "unresolved",
                )
            )
        )

    def _incident_open_for_ticket(self, ticket: HelpdeskTicketRecord) -> bool:
        related_ids = {ticket.ticket_id}
        if ticket.related_ticket_id:
            related_ids.add(ticket.related_ticket_id)
        if ticket.generated_from_ticket_id:
            related_ids.add(ticket.generated_from_ticket_id)
        if any(ticket_id in self._state.open_incident_ticket_ids for ticket_id in related_ids):
            return True
        ticket_cluster_keys = self._cluster_keys_for_ticket(ticket)
        if not ticket_cluster_keys:
            return False
        for open_ticket_id in self._state.open_incident_ticket_ids:
            open_ticket = self._tickets_by_id.get(open_ticket_id)
            if open_ticket is None:
                continue
            if ticket_cluster_keys & self._cluster_keys_for_ticket(open_ticket):
                return True
        return False

    def _request_info_note_for_ticket(self, ticket: HelpdeskTicketRecord) -> str | None:
        note_parts: list[str] = []
        if ticket.customer_update_note:
            note_parts.append(ticket.customer_update_note)
        if ticket.related_ticket_id is not None:
            note_parts.append(
                "The requester confirmed this is connected to the earlier case and wants a single accountable owner."
            )
        if self._ticket_has_nondefault_routing(ticket):
            note_parts.append(
                "The requester clarified that the blocker owner matters more than the superficial request label."
            )
        if self._ticket_has_alternate_route(ticket):
            note_parts.append(
                "Operations said an acknowledged fallback path is acceptable if the preferred queue is saturated."
            )
        if self._requires_incident(ticket):
            note_parts.append(
                "Stakeholders asked for incident-style coordination because the issue is still operationally active."
            )
        if not note_parts:
            return None
        return " ".join(note_parts)

    def _request_info_used(self, ticket_id: str) -> bool:
        return self._state.ticket_request_info_usage.get(ticket_id, 0) > 0

    def _defer_count(self, ticket_id: str) -> int:
        return self._state.ticket_defer_counts.get(ticket_id, 0)

    def _record_dynamic_queue_event(self, event_type: str, **details: Any) -> None:
        self._state.dynamic_queue_events.append({"event_type": event_type, **details})

    def _escalate_priority_level(self, priority: str) -> str:
        if priority == "low":
            return "medium"
        if priority == "medium":
            return "high"
        return "critical"

    def _escalate_ticket_after_delay(
        self,
        ticket: HelpdeskTicketRecord,
        *,
        defer_count: int,
    ) -> HelpdeskTicketRecord:
        escalated_priority = self._escalate_priority_level(ticket.priority)
        description_suffix = (
            " The ticket was deferred earlier in the queue and now needs firmer ownership."
        )
        customer_update = (
            ticket.customer_update_note
            or "The requester followed up after the delay and wants a committed owner."
        )
        return ticket.model_copy(
            update={
                "priority": escalated_priority,
                "title": (
                    ticket.title
                    if ticket.title.lower().startswith("re:")
                    else f"Re: {ticket.title}"
                ),
                "description": f"{ticket.description}{description_suffix}",
                "customer_update_note": customer_update,
            }
        )

    def _should_spawn_follow_up(
        self,
        ticket: HelpdeskTicketRecord,
        *,
        score: float,
        context_penalty: float,
        incident_gap_penalty: float,
    ) -> bool:
        task_id = int(self._state.current_task_id or 1)
        if task_id < 2:
            return False
        if ticket.generated_from_ticket_id is not None:
            return False
        if ticket.ticket_id in self._state.spawned_follow_up_source_ids:
            return False
        follow_up_risk = (
            self._requires_incident(ticket)
            or self._ticket_mentions_follow_up(ticket)
            or ticket.related_ticket_id is not None
            or ticket.priority in {"high", "critical"}
            or self._cluster_summary(ticket)["future_cluster_ticket_count"] > 0
        )
        if not follow_up_risk:
            return False
        if task_id == 2 and not (
            ticket.related_ticket_id is not None
            or self._ticket_mentions_follow_up(ticket)
            or self._cluster_summary(ticket)["future_cluster_ticket_count"] > 0
            or self._ticket_repeated_requester_count(ticket) >= 2
        ):
            return False
        return (
            score < FOLLOW_UP_SPAWN_THRESHOLD
            or (context_penalty >= 0.15 and score < 0.9)
            or incident_gap_penalty > 0.0
        )

    def _spawn_follow_up_ticket(self, ticket: HelpdeskTicketRecord) -> HelpdeskTicketRecord:
        follow_up_ticket = HelpdeskTicketRecord(
            ticket_id=f"{ticket.ticket_id}-followup",
            title=(
                ticket.title
                if ticket.title.lower().startswith("re:")
                else f"Re: {ticket.title}"
            ),
            requester=ticket.requester,
            description=(
                "The earlier handling did not fully resolve the issue. The requester is "
                f"following up on {ticket.ticket_id} and needs a single accountable owner now."
            ),
            issue_type=ticket.issue_type,
            priority=(
                "critical"
                if ticket.priority in {"high", "critical"}
                else self._escalate_priority_level(ticket.priority)
            ),
            assignment_group=ticket.assignment_group,
            resolution_action=(
                "escalate"
                if ticket.priority in {"high", "critical"} or self._requires_incident(ticket)
                else ticket.resolution_action
            ),
            ambiguity_note=(
                ticket.ambiguity_note
                or "Prior routing did not settle ownership; route to the team that can actually unblock the issue."
            ),
            related_ticket_id=ticket.ticket_id,
            planning_note=ticket.planning_note,
            customer_update_note=(
                "The requester said the last response did not resolve the blocker and wants an accountable next owner."
            ),
            incident_recommended=self._requires_incident(ticket),
            generated_from_ticket_id=ticket.ticket_id,
            service_cluster_id=ticket.service_cluster_id or ticket.ticket_id,
        )
        self._queue.append(follow_up_ticket)
        self._tickets_by_id[follow_up_ticket.ticket_id] = follow_up_ticket
        self._sync_queue_ticket_ids()
        self._state.spawned_follow_up_ticket_ids.append(follow_up_ticket.ticket_id)
        self._state.spawned_follow_up_source_ids.append(ticket.ticket_id)
        self._record_dynamic_queue_event(
            "spawn_follow_up",
            source_ticket_id=ticket.ticket_id,
            follow_up_ticket_id=follow_up_ticket.ticket_id,
        )
        return follow_up_ticket

    def _ticket_repeated_requester_count(self, ticket: HelpdeskTicketRecord) -> int:
        return sum(
            1
            for candidate in self._tickets_by_id.values()
            if candidate.requester == ticket.requester
        )

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
        if tool_name == "lookup_queue_capacity_forecast":
            return self._state.current_task_id == 3 and (
                self._ticket_has_alternate_route(ticket)
                or self._future_queue_demand()["remaining_ticket_count"] > 0
            )
        if tool_name == "lookup_queue_cluster_summary":
            if (self._state.current_task_id or 1) < 2:
                return False
            cluster_summary = self._cluster_summary(ticket)
            return (
                cluster_summary["future_cluster_ticket_count"] > 0
                or cluster_summary["shared_requester_count"] > 1
            )
        return False

    def _required_tools_for_ticket(
        self,
        ticket: HelpdeskTicketRecord,
        task_id: int | None = None,
    ) -> list[str]:
        resolved_task_id = self._state.current_task_id if task_id is None else task_id
        if resolved_task_id is None or resolved_task_id < 2:
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
        if (
            resolved_task_id == 3
            and self._ticket_is_capacity_sensitive(ticket)
            and "lookup_queue_capacity_forecast" not in required_tools
        ):
            required_tools.append("lookup_queue_capacity_forecast")
        ticket_index = self._ticket_queue_index(ticket)
        cluster_start_index = (
            ticket_index + 1
            if ticket_index is not None
            else self._state.current_ticket_index + 1
        )
        if resolved_task_id == 3:
            cluster_summary = self._cluster_summary(
                ticket,
                start_index=cluster_start_index,
            )
            if (
                cluster_summary["future_cluster_ticket_count"] > 0
                and "lookup_queue_cluster_summary" not in required_tools
                and (
                    self._requires_incident(ticket)
                    or cluster_summary["future_high_priority_count"] > 0
                    or cluster_summary["shared_requester_count"] > 1
                )
            ):
                required_tools.append("lookup_queue_cluster_summary")
        if resolved_task_id == 2:
            cluster_summary = self._cluster_summary(
                ticket,
                start_index=cluster_start_index,
            )
            if (
                cluster_summary["future_cluster_ticket_count"] > 0
                and "lookup_queue_cluster_summary" not in required_tools
                and (
                    ticket.related_ticket_id is not None
                    or cluster_summary["shared_requester_count"] > 1
                    or self._ticket_mentions_follow_up(ticket)
                )
            ):
                required_tools.append("lookup_queue_cluster_summary")
        filtered_required_tools: list[str] = []
        allowed_tool_set = set(self._available_tools_for_task(resolved_task_id))
        for tool_name in required_tools:
            if tool_name in filtered_required_tools:
                continue
            if tool_name not in allowed_tool_set:
                continue
            if self._tool_has_available_context(ticket, tool_name):
                filtered_required_tools.append(tool_name)
        return filtered_required_tools

    def _recommended_operational_actions(self, ticket: HelpdeskTicketRecord) -> list[str]:
        recommended_actions: list[str] = []
        available_action_types = set(self._available_action_types_for_task())
        cluster_summary = self._cluster_summary(ticket)
        if (
            "request_info" in available_action_types
            and self._request_info_note_for_ticket(ticket) is not None
            and not self._request_info_used(ticket.ticket_id)
        ):
            recommended_actions.append("request_info")
        if (
            "open_incident" in available_action_types
            and self._requires_incident(ticket)
            and not self._incident_open_for_ticket(ticket)
        ):
            recommended_actions.append("open_incident")
        if (
            "defer" in available_action_types
            and self._defer_count(ticket.ticket_id) < MAX_DEFERS_PER_TICKET
            and self._state.current_ticket_index < len(self._queue) - 1
            and ticket.priority not in {"high", "critical"}
            and (
                bool(self._remaining_tools_for_ticket(ticket))
                or self._ticket_is_capacity_sensitive(ticket)
                or self._request_info_note_for_ticket(ticket) is not None
                or cluster_summary["future_cluster_ticket_count"] > 0
            )
        ):
            recommended_actions.append("defer")
        return recommended_actions

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
        request_info_used = self._request_info_used(ticket.ticket_id)
        operational_actions = self._recommended_operational_actions(ticket)
        return {
            "required_tools": required_tools,
            "revealed_tools": revealed_tools,
            "remaining_tools": remaining_tools,
            "revealed_count": len(revealed_tools),
            "remaining_count": len(remaining_tools),
            "completeness": round(len(revealed_tools) / total_required, 2),
            "request_info_used": request_info_used,
            "recommended_operational_actions": operational_actions,
        }

    def _default_redacted_description(self, ticket: HelpdeskTicketRecord) -> str:
        cluster_summary = self._cluster_summary(ticket)
        if cluster_summary["future_cluster_ticket_count"] > 0:
            return (
                "This ticket is part of a broader queue cluster and the best next step depends "
                "on downstream consequences. Additional routing context is available via investigation."
            )
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
        if self._ticket_has_alternate_route(ticket):
            return (
                "The queue is under resource pressure and this ticket may support more than "
                "one acceptable routing path. Additional planning context is available via investigation."
            )
        if self._ticket_has_nondefault_routing(ticket):
            return (
                "The visible request looks straightforward, but the decisive routing detail is hidden until investigation."
            )
        return (
            "Additional routing context is available via investigation before final submission."
        )

    def _default_redacted_title(self, ticket: HelpdeskTicketRecord) -> str:
        if self._cluster_summary(ticket)["future_cluster_ticket_count"] > 0:
            return "Clustered queue decision with hidden downstream impact"
        if ticket.related_ticket_id is not None:
            return "Follow-up request with hidden routing context"
        if self._internal_routing_note_for_ticket(ticket) is not None:
            return "Routing clarification required"
        if self._ticket_has_alternate_route(ticket):
            return "Capacity-sensitive routing decision"
        if self._ticket_mentions_follow_up(ticket):
            return "Priority support follow-up"
        return "Helpdesk routing decision"

    def _visible_title(self, ticket: HelpdeskTicketRecord) -> str:
        if self._state.current_task_id in {2, 3} and self._remaining_tools_for_ticket(ticket):
            return HARD_TASK_TITLE_REDACTIONS.get(
                ticket.ticket_id,
                self._default_redacted_title(ticket),
            )
        return ticket.title

    def _visible_description(self, ticket: HelpdeskTicketRecord) -> str:
        if self._state.current_task_id in {2, 3} and self._remaining_tools_for_ticket(ticket):
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

    def _incident_gap_penalty(
        self,
        ticket: HelpdeskTicketRecord,
        action: HelpdeskTicketAction,
    ) -> float:
        if self._state.current_task_id != 3:
            return 0.0
        if not self._requires_incident(ticket):
            return 0.0
        if self._incident_open_for_ticket(ticket):
            return 0.0
        if action.resolution_action in {"escalate", "assign"}:
            return round(INCIDENT_GAP_PENALTY / 2, 4)
        return INCIDENT_GAP_PENALTY

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
            for ticket in self._tickets_by_id.values()
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

    def _lookup_queue_capacity_forecast(
        self,
        current_ticket: HelpdeskTicketRecord,
    ) -> dict[str, Any]:
        recommendation = self._planning_route_recommendation(current_ticket)
        routing_options = self._routing_options_for_ticket(current_ticket)
        return {
            "tool_name": "lookup_queue_capacity_forecast",
            "found": True,
            "ticket_id": current_ticket.ticket_id,
            "preferred_route_label": recommendation["preferred_label"],
            "primary_pressure": recommendation["primary_pressure"],
            "alternate_pressure": recommendation["alternate_pressure"],
            "capacity_state": recommendation["capacity_state"],
            "future_queue_demand": recommendation["future_demand"],
            "routing_options": routing_options,
            "incident_recommended": self._requires_incident(current_ticket),
        }

    def _lookup_queue_cluster_summary(
        self,
        current_ticket: HelpdeskTicketRecord,
    ) -> dict[str, Any]:
        cluster_summary = self._cluster_summary(current_ticket)
        return {
            "tool_name": "lookup_queue_cluster_summary",
            "found": cluster_summary["future_cluster_ticket_count"] > 0
            or cluster_summary["shared_requester_count"] > 1,
            "ticket_id": current_ticket.ticket_id,
            **cluster_summary,
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
        if tool_name == "lookup_queue_capacity_forecast":
            return self._lookup_queue_capacity_forecast(current_ticket)
        if tool_name == "lookup_queue_cluster_summary":
            return self._lookup_queue_cluster_summary(current_ticket)
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
        if action.tool_name not in self._available_tools_for_task():
            raise ValueError(f"Unsupported tool_name for current task: {action.tool_name}")
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

    def _handle_request_info_action(
        self,
        task: dict,
        current_ticket: HelpdeskTicketRecord,
        action: HelpdeskTicketAction,
        idx: int,
    ) -> HelpdeskTicketObservation:
        submitted_fields = {
            field
            for field in ("issue_type", "priority", "assignment_group", "resolution_action")
            if getattr(action, field) is not None
        }
        if submitted_fields:
            raise ValueError(
                "request_info actions cannot include submit fields: "
                f"{sorted(submitted_fields)}"
            )

        ticket_id = current_ticket.ticket_id
        note = self._request_info_note_for_ticket(current_ticket)
        already_used = self._request_info_used(ticket_id)
        useful_request = note is not None and not already_used
        self._state.ticket_request_info_usage[ticket_id] = (
            self._state.ticket_request_info_usage.get(ticket_id, 0) + 1
        )
        self._state.step_count += 1
        self._state.investigation_steps += 1
        self._state.investigation_budget_remaining = max(
            0,
            self._state.investigation_budget_remaining - 1,
        )
        request_reward = USEFUL_REQUEST_INFO_REWARD if useful_request else 0.0
        tool_result = {
            "action_type": "request_info",
            "found": useful_request,
            "ticket_id": ticket_id,
            "customer_update_note": note if useful_request else "",
        }
        self._state.last_tool_result = tool_result
        self._state.last_step_reward = request_reward
        self._state.reward = request_reward
        self._state.done = False
        self._state.investigation_penalty_applied = self._compute_episode_penalty()
        progress = self._tool_progress_for_ticket(current_ticket)
        reward_components = self._build_reward_components(
            ticket_score=0.0,
            field_breakdown={},
            shaped_step_reward=request_reward,
            reward_kind="operational",
            final_reward=request_reward,
            investigation_penalty=self._state.investigation_penalty_applied,
            extra_details={
                "operational_action": "request_info",
                "new_context_revealed": useful_request,
                "customer_update_visible": useful_request,
                "hidden_context_remaining_count": progress["remaining_count"],
                "context_completeness": progress["completeness"],
            },
        )
        self._state.history_entries.append(
            self._build_history_entry(
                current_ticket,
                predicted=action.model_dump(exclude_none=True),
                score=0.0,
                breakdown={},
                queue_position=idx + 1,
                reward=request_reward,
                reward_kind="operational",
                tool_result=tool_result,
                reward_components=reward_components,
            )
        )
        self._state.last_reward_components = reward_components
        return self._build_observation(task, done=False, reward=request_reward)

    def _handle_defer_action(
        self,
        task: dict,
        current_ticket: HelpdeskTicketRecord,
        action: HelpdeskTicketAction,
        idx: int,
    ) -> HelpdeskTicketObservation:
        submitted_fields = {
            field
            for field in ("issue_type", "priority", "assignment_group", "resolution_action")
            if getattr(action, field) is not None
        }
        if submitted_fields:
            raise ValueError(
                "defer actions cannot include submit fields: "
                f"{sorted(submitted_fields)}"
            )

        ticket_id = current_ticket.ticket_id
        existing_count = self._defer_count(ticket_id)
        defer_allowed = (
            existing_count < MAX_DEFERS_PER_TICKET
            and idx < len(self._queue) - 1
            and self._state.current_task_id in {2, 3}
        )
        defer_count = existing_count + 1
        reward = 0.0
        sla_risk = current_ticket.priority in {"high", "critical"} or self._ticket_mentions_follow_up(
            current_ticket
        )
        moved_ticket = current_ticket

        if defer_allowed:
            self._state.ticket_defer_counts[ticket_id] = defer_count
            self._state.deferred_ticket_count += 1
            if sla_risk:
                self._state.sla_breach_count += 1
                moved_ticket = self._escalate_ticket_after_delay(
                    current_ticket,
                    defer_count=defer_count,
                )
            elif (
                self._remaining_tools_for_ticket(current_ticket)
                or self._request_info_note_for_ticket(current_ticket) is not None
                or self._ticket_is_capacity_sensitive(current_ticket)
            ):
                reward = REQUEST_INFO_CONTEXT_COMPLETION_BONUS
            self._queue.pop(idx)
            self._queue.append(moved_ticket)
            self._tickets_by_id[moved_ticket.ticket_id] = moved_ticket
            self._sync_queue_ticket_ids()
            self._record_dynamic_queue_event(
                "defer",
                ticket_id=ticket_id,
                defer_count=defer_count,
                sla_risk=sla_risk,
            )
        else:
            self._state.sla_breach_count += 1
            self._record_dynamic_queue_event(
                "defer_denied",
                ticket_id=ticket_id,
                defer_count=defer_count,
            )

        self._state.step_count += 1
        self._state.last_tool_result = {
            "action_type": "defer",
            "ticket_id": ticket_id,
            "defer_allowed": defer_allowed,
            "defer_count": defer_count,
            "sla_risk": sla_risk,
        }
        self._state.last_step_reward = reward
        self._state.reward = reward
        self._state.done = False
        reward_components = self._build_reward_components(
            ticket_score=0.0,
            field_breakdown={},
            shaped_step_reward=reward,
            reward_kind="operational",
            final_reward=reward,
            extra_details={
                "operational_action": "defer",
                "defer_allowed": defer_allowed,
                "defer_count": defer_count,
                "sla_breach_count": self._state.sla_breach_count,
            },
        )
        self._state.history_entries.append(
            self._build_history_entry(
                current_ticket,
                predicted=action.model_dump(exclude_none=True),
                score=0.0,
                breakdown={},
                queue_position=idx + 1,
                reward=reward,
                reward_kind="operational",
                tool_result=self._state.last_tool_result,
                reward_components=reward_components,
            )
        )
        self._state.last_reward_components = reward_components
        return self._build_observation(task, done=False, reward=reward)

    def _handle_open_incident_action(
        self,
        task: dict,
        current_ticket: HelpdeskTicketRecord,
        action: HelpdeskTicketAction,
        idx: int,
    ) -> HelpdeskTicketObservation:
        submitted_fields = {
            field
            for field in ("issue_type", "priority", "assignment_group", "resolution_action")
            if getattr(action, field) is not None
        }
        if submitted_fields:
            raise ValueError(
                "open_incident actions cannot include submit fields: "
                f"{sorted(submitted_fields)}"
            )

        useful_incident = (
            self._state.current_task_id == 3
            and self._requires_incident(current_ticket)
            and not self._incident_open_for_ticket(current_ticket)
        )
        overflow = 0
        incident_reward = 0.0
        if useful_incident:
            self._state.open_incident_ticket_ids.append(current_ticket.ticket_id)
            self._state.incident_actions_used += 1
            overflow = max(0, 1 - self._state.incident_slots_remaining)
            self._state.incident_slots_remaining = max(
                0,
                self._state.incident_slots_remaining - 1,
            )
            overflow_penalty = round(overflow * INCIDENT_SLOT_OVERFLOW_PENALTY, 4)
            if overflow_penalty > 0.0:
                self._state.planning_penalty_total = round(
                    self._state.planning_penalty_total + overflow_penalty,
                    4,
                )
                self._state.planning_penalty_applied = overflow_penalty
            incident_reward = clamp_open_unit_interval(
                INCIDENT_OPEN_REWARD - overflow_penalty
            )
            self._record_dynamic_queue_event(
                "open_incident",
                ticket_id=current_ticket.ticket_id,
                overflow=overflow,
            )

        self._state.step_count += 1
        self._state.last_tool_result = {
            "action_type": "open_incident",
            "ticket_id": current_ticket.ticket_id,
            "incident_open": useful_incident,
            "incident_slots_remaining": self._state.incident_slots_remaining,
            "overflow": overflow,
        }
        self._state.last_step_reward = incident_reward
        self._state.reward = incident_reward
        self._state.done = False
        reward_components = self._build_reward_components(
            ticket_score=0.0,
            field_breakdown={},
            shaped_step_reward=incident_reward,
            reward_kind="operational",
            final_reward=incident_reward,
            extra_details={
                "operational_action": "open_incident",
                "incident_open": useful_incident,
                "incident_slots_remaining": self._state.incident_slots_remaining,
            },
        )
        self._state.history_entries.append(
            self._build_history_entry(
                current_ticket,
                predicted=action.model_dump(exclude_none=True),
                score=0.0,
                breakdown={},
                queue_position=idx + 1,
                reward=incident_reward,
                reward_kind="operational",
                tool_result=self._state.last_tool_result,
                reward_components=reward_components,
            )
        )
        self._state.last_reward_components = reward_components
        return self._build_observation(task, done=False, reward=incident_reward)

    def _build_ticket_view(self, ticket: HelpdeskTicketRecord) -> dict[str, Any]:
        progress = self._tool_progress_for_ticket(ticket)
        remaining_tools = progress["remaining_tools"]
        used_tools = set(self._used_tools_for_ticket(ticket.ticket_id))
        operational_actions = progress["recommended_operational_actions"]
        cluster_summary = self._cluster_summary(ticket)
        cluster_hint = (
            cluster_summary["future_cluster_ticket_count"] > 0
            or cluster_summary["shared_requester_count"] > 1
        )
        ticket_view: dict[str, Any] = {
            "ticket_id": ticket.ticket_id,
            "title": self._visible_title(ticket),
            "requester": ticket.requester,
            "description": self._visible_description(ticket),
        }
        if self._state.current_task_id == 3:
            ticket_view["capacity_state"] = self._capacity_state_snapshot()
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
        ticket_view["operational_context"] = {
            "request_info_available": self._request_info_note_for_ticket(ticket) is not None,
            "request_info_used": progress["request_info_used"],
            "defer_count": self._defer_count(ticket.ticket_id),
            "incident_recommended": self._requires_incident(ticket),
            "incident_open": self._incident_open_for_ticket(ticket),
            "recommended_actions": operational_actions,
            "cluster_coordination_hint": cluster_hint,
            "shared_requester_pressure": cluster_summary["shared_requester_count"] > 1,
        }
        if "lookup_queue_cluster_summary" in used_tools:
            ticket_view["operational_context"].update(
                {
                    "service_cluster_id": ticket.service_cluster_id,
                    "future_cluster_ticket_count": cluster_summary["future_cluster_ticket_count"],
                    "future_cluster_ticket_ids": cluster_summary["future_cluster_ticket_ids"],
                    "shared_requester_count": cluster_summary["shared_requester_count"],
                    "active_incident_cover": cluster_summary["active_incident_cover"],
                }
            )
        if ticket.ambiguity_note is not None and "lookup_internal_routing_note" not in remaining_tools:
            ticket_view["ambiguity_note"] = ticket.ambiguity_note
        if (
            ticket.planning_note is not None
            and "lookup_internal_routing_note" not in remaining_tools
        ):
            ticket_view["planning_note"] = ticket.planning_note
        if self._request_info_used(ticket.ticket_id):
            ticket_view["customer_update_note"] = self._request_info_note_for_ticket(ticket)
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
        if self._ticket_has_alternate_route(ticket) and (
            "lookup_internal_routing_note" in used_tools
            or "lookup_queue_capacity_forecast" in used_tools
        ):
            ticket_view["routing_options"] = self._routing_options_for_ticket(ticket)
        if "lookup_queue_cluster_summary" in used_tools:
            ticket_view["cluster_summary"] = cluster_summary
        if ticket.generated_from_ticket_id is not None:
            ticket_view["generated_from_ticket_id"] = ticket.generated_from_ticket_id
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
        elif reward_kind == "operational":
            operational_action = (
                reward_components.get("operational_action")
                if reward_components
                else predicted.get("action_type")
            )
            parts.append(f"Operational step used {operational_action or 'an action'}")
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
            capacity_penalty = reward_components.get("capacity_penalty")
            if capacity_penalty:
                parts.append(f"capacity_penalty={capacity_penalty:.2f}")
            planning_penalty_total = reward_components.get("planning_penalty_total")
            if planning_penalty_total:
                parts.append(f"planning_penalty_total={planning_penalty_total:.2f}")
            incident_gap_penalty = reward_components.get("incident_gap_penalty")
            if incident_gap_penalty:
                parts.append(f"incident_gap_penalty={incident_gap_penalty:.2f}")
            queue_management_score = reward_components.get("queue_management_score")
            if queue_management_score is not None:
                parts.append(f"queue_management_score={queue_management_score:.2f}")
            spawned_follow_up_ticket_id = reward_components.get("spawned_follow_up_ticket_id")
            if spawned_follow_up_ticket_id:
                parts.append(f"spawned_follow_up={spawned_follow_up_ticket_id}")
            cluster_stabilized_ticket_ids = reward_components.get("cluster_stabilized_ticket_ids")
            if cluster_stabilized_ticket_ids:
                parts.append(
                    "cluster_stabilized=" + ",".join(cluster_stabilized_ticket_ids)
                )
            cluster_destabilized_ticket_ids = reward_components.get(
                "cluster_destabilized_ticket_ids"
            )
            if cluster_destabilized_ticket_ids:
                parts.append(
                    "cluster_destabilized=" + ",".join(cluster_destabilized_ticket_ids)
                )

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
        cluster_summary = self._cluster_summary(ticket)
        history_entry: dict[str, Any] = {
            "ticket_id": ticket.ticket_id,
            "title": ticket.title,
            "requester": ticket.requester,
            "predicted": predicted,
            "score": score,
            "breakdown": breakdown,
            "queue_position": queue_position,
            "operational_context": {
                "request_info_used": progress["request_info_used"],
                "defer_count": self._defer_count(ticket.ticket_id),
                "incident_open": self._incident_open_for_ticket(ticket),
                "recommended_actions": progress["recommended_operational_actions"],
                "cluster_coordination_hint": (
                    cluster_summary["future_cluster_ticket_count"] > 0
                    or cluster_summary["shared_requester_count"] > 1
                ),
            },
        }
        if "lookup_queue_cluster_summary" in self._used_tools_for_ticket(ticket.ticket_id):
            history_entry["operational_context"].update(
                {
                    "service_cluster_id": ticket.service_cluster_id,
                    "future_cluster_ticket_count": cluster_summary["future_cluster_ticket_count"],
                    "active_incident_cover": cluster_summary["active_incident_cover"],
                    "shared_requester_count": cluster_summary["shared_requester_count"],
                }
            )
        if self._state.current_task_id == 3:
            history_entry["capacity_state"] = self._capacity_state_snapshot()
        if reward is not None:
            history_entry["reward"] = reward
        if rubric_reward is not None:
            history_entry["rubric_reward"] = rubric_reward
        if reward_kind is not None:
            history_entry["reward_kind"] = reward_kind
        if ticket.ambiguity_note is not None and "lookup_internal_routing_note" not in remaining_tools:
            history_entry["ambiguity_note"] = ticket.ambiguity_note
        if (
            ticket.planning_note is not None
            and "lookup_internal_routing_note" not in remaining_tools
        ):
            history_entry["planning_note"] = ticket.planning_note
        if self._request_info_used(ticket.ticket_id):
            history_entry["customer_update_note"] = self._request_info_note_for_ticket(ticket)
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
        if (
            self._ticket_has_alternate_route(ticket)
            and (
                "lookup_internal_routing_note" not in remaining_tools
                or "lookup_queue_capacity_forecast" in self._used_tools_for_ticket(ticket.ticket_id)
            )
        ):
            history_entry["routing_options"] = self._routing_options_for_ticket(ticket)
        if "lookup_queue_cluster_summary" in self._used_tools_for_ticket(ticket.ticket_id):
            history_entry["cluster_summary"] = cluster_summary
        if penalty_reason is not None:
            history_entry["penalty_reason"] = penalty_reason
        if tool_result is not None:
            history_entry["tool_result"] = tool_result
        if reward_components is not None:
            history_entry["reward_components"] = reward_components
        if ticket.generated_from_ticket_id is not None:
            history_entry["generated_from_ticket_id"] = ticket.generated_from_ticket_id
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
            "available_action_types": self._available_action_types_for_task(),
            "average_score_so_far": self._state.average_score_so_far,
            "progress_fraction": progress_fraction,
            "investigation_penalty_applied": self._state.investigation_penalty_applied,
            "planning_penalty_total": self._state.planning_penalty_total,
            "planning_penalty_applied": self._state.planning_penalty_applied,
            "sla_breach_count": self._state.sla_breach_count,
            "incident_gap_total": self._state.incident_gap_total,
            "queue_management_score": self._state.queue_management_score,
            "queue_management_breakdown": dict(self._state.queue_management_breakdown),
            "dynamic_queue_events": list(self._state.dynamic_queue_events[-5:]),
            "clustered_follow_ons": self._future_queue_demand().get("clustered_follow_ons", 0),
        }
        if self._state.current_task_id == 3:
            metadata["capacity_state"] = self._capacity_state_snapshot()
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
            available_action_types=self._available_action_types_for_task(),
            available_tools=self._available_tools_for_task(),
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
