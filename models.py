from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator
from openenv.core.env_server.types import Action, Observation, State
from vocabulary import (
    ASSIGNMENT_GROUPS,
    ISSUE_TYPES,
    PRIORITIES,
    RESOLUTION_ACTIONS,
)


ISSUE_TYPE_SET = set(ISSUE_TYPES)
PRIORITY_SET = set(PRIORITIES)
ASSIGNMENT_GROUP_SET = set(ASSIGNMENT_GROUPS)
RESOLUTION_ACTION_SET = set(RESOLUTION_ACTIONS)
ACTION_TYPE_SET = {
    "submit",
    "investigate",
    "request_info",
    "defer",
    "open_incident",
}
TOOL_NAME_SET = {"lookup_related_ticket", "lookup_requester_history"}
TOOL_NAME_SET.add("lookup_internal_routing_note")
TOOL_NAME_SET.add("lookup_queue_capacity_forecast")
TOOL_NAME_SET.add("lookup_queue_cluster_summary")


def _validate_choice(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"{field_name} must be one of: {allowed_values}")
    return value


def _validate_optional_choice(
    value: Optional[str], allowed: set[str], field_name: str
) -> Optional[str]:
    if value is None:
        return None
    return _validate_choice(value, allowed, field_name)


class HelpdeskTicketRecord(BaseModel):
    ticket_id: str
    title: str
    requester: str
    description: str
    issue_type: str
    priority: str
    assignment_group: str
    resolution_action: str
    ambiguity_note: Optional[str] = None
    related_ticket_id: Optional[str] = None
    planning_note: Optional[str] = None
    alternate_issue_type: Optional[str] = None
    alternate_priority: Optional[str] = None
    alternate_assignment_group: Optional[str] = None
    alternate_resolution_action: Optional[str] = None
    alternate_route_score_multiplier: float = 0.0
    customer_update_note: Optional[str] = None
    incident_recommended: bool = False
    generated_from_ticket_id: Optional[str] = None
    service_cluster_id: Optional[str] = None

    @field_validator("issue_type")
    @classmethod
    def validate_issue_type(cls, value: str) -> str:
        return _validate_choice(value, ISSUE_TYPE_SET, "issue_type")

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, value: str) -> str:
        return _validate_choice(value, PRIORITY_SET, "priority")

    @field_validator("assignment_group")
    @classmethod
    def validate_assignment_group(cls, value: str) -> str:
        return _validate_choice(value, ASSIGNMENT_GROUP_SET, "assignment_group")

    @field_validator("resolution_action")
    @classmethod
    def validate_resolution_action(cls, value: str) -> str:
        return _validate_choice(value, RESOLUTION_ACTION_SET, "resolution_action")

    @field_validator("alternate_issue_type")
    @classmethod
    def validate_alternate_issue_type(cls, value: Optional[str]) -> Optional[str]:
        return _validate_optional_choice(value, ISSUE_TYPE_SET, "alternate_issue_type")

    @field_validator("alternate_priority")
    @classmethod
    def validate_alternate_priority(cls, value: Optional[str]) -> Optional[str]:
        return _validate_optional_choice(value, PRIORITY_SET, "alternate_priority")

    @field_validator("alternate_assignment_group")
    @classmethod
    def validate_alternate_assignment_group(cls, value: Optional[str]) -> Optional[str]:
        return _validate_optional_choice(
            value,
            ASSIGNMENT_GROUP_SET,
            "alternate_assignment_group",
        )

    @field_validator("alternate_resolution_action")
    @classmethod
    def validate_alternate_resolution_action(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        return _validate_optional_choice(
            value,
            RESOLUTION_ACTION_SET,
            "alternate_resolution_action",
        )

    @field_validator("alternate_route_score_multiplier")
    @classmethod
    def validate_alternate_route_score_multiplier(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("alternate_route_score_multiplier must be in [0.0, 1.0]")
        return value


class HelpdeskTicketAction(Action):
    action_type: str = "submit"
    tool_name: Optional[str] = None
    tool_target_ticket_id: Optional[str] = None
    issue_type: Optional[str] = None
    priority: Optional[str] = None
    assignment_group: Optional[str] = None
    resolution_action: Optional[str] = None

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, value: str) -> str:
        return _validate_choice(value, ACTION_TYPE_SET, "action_type")

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: Optional[str]) -> Optional[str]:
        return _validate_optional_choice(value, TOOL_NAME_SET, "tool_name")

    @field_validator("issue_type")
    @classmethod
    def validate_issue_type(cls, value: Optional[str]) -> Optional[str]:
        return _validate_optional_choice(value, ISSUE_TYPE_SET, "issue_type")

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, value: Optional[str]) -> Optional[str]:
        return _validate_optional_choice(value, PRIORITY_SET, "priority")

    @field_validator("assignment_group")
    @classmethod
    def validate_assignment_group(cls, value: Optional[str]) -> Optional[str]:
        return _validate_optional_choice(value, ASSIGNMENT_GROUP_SET, "assignment_group")

    @field_validator("resolution_action")
    @classmethod
    def validate_resolution_action(cls, value: Optional[str]) -> Optional[str]:
        return _validate_optional_choice(value, RESOLUTION_ACTION_SET, "resolution_action")


class HelpdeskTicketObservation(Observation):
    task_id: int = 0
    task_name: str = ""
    instructions: str = ""
    allowed_fields: list[str] = Field(default_factory=list)
    available_action_types: list[str] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
    investigation_budget_remaining: int = 0
    last_tool_result: Optional[dict[str, Any]] = None
    current_ticket: Optional[dict[str, Any]] = None
    queue_size: int = 0
    tickets_remaining: int = 0
    tickets_after_current: int = 0
    tickets_processed: int = 0
    queue_position: int = 0
    average_score_so_far: float = 0.0
    progress_fraction: float = 0.0
    history: list[dict[str, Any]] = Field(default_factory=list)
    last_reward_components: dict[str, Any] = Field(default_factory=dict)
    rubric_reward: Optional[float] = None


class HelpdeskTicketState(State):
    current_task_id: Optional[int] = None
    seed: Optional[int] = None
    queue_ticket_ids: list[str] = Field(default_factory=list)
    current_ticket_index: int = 0
    per_ticket_scores: list[float] = Field(default_factory=list)
    total_reward: float = 0.0
    last_step_reward: Optional[float] = None
    # `reward` is the field the evaluator checks on GET /state (mentor spec)
    reward: Optional[float] = None
    done: bool = False
    average_score_so_far: float = 0.0
    investigation_steps: int = 0
    investigation_budget_remaining: int = 0
    investigation_penalty_applied: float = 0.0
    planning_penalty_applied: float = 0.0
    last_tool_result: Optional[dict[str, Any]] = None
    last_reward_components: dict[str, Any] = Field(default_factory=dict)
    ticket_tool_usage: dict[str, list[str]] = Field(default_factory=dict)
    team_capacity_initial: dict[str, int] = Field(default_factory=dict)
    team_capacity_remaining: dict[str, int] = Field(default_factory=dict)
    high_priority_slots_initial: int = 0
    high_priority_slots_remaining: int = 0
    escalation_slots_initial: int = 0
    escalation_slots_remaining: int = 0
    planning_penalty_total: float = 0.0
    capacity_pressure_tickets_resolved: int = 0
    cluster_stabilizations_total: int = 0
    cluster_destabilizations_total: int = 0
    ticket_request_info_usage: dict[str, int] = Field(default_factory=dict)
    ticket_defer_counts: dict[str, int] = Field(default_factory=dict)
    open_incident_ticket_ids: list[str] = Field(default_factory=list)
    incident_slots_initial: int = 0
    incident_slots_remaining: int = 0
    incident_actions_used: int = 0
    incident_gap_total: float = 0.0
    deferred_ticket_count: int = 0
    sla_breach_count: int = 0
    spawned_follow_up_ticket_ids: list[str] = Field(default_factory=list)
    spawned_follow_up_source_ids: list[str] = Field(default_factory=list)
    dynamic_queue_events: list[dict[str, Any]] = Field(default_factory=list)
    queue_management_score: float = 0.0
    queue_management_breakdown: dict[str, Any] = Field(default_factory=dict)
    history_entries: list[dict] = Field(default_factory=list)
