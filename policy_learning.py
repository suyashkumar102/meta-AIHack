#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from models import HelpdeskTicketAction, HelpdeskTicketObservation
from server.environment import HelpdeskTicketRoutingEnvironment
from server.tasks import get_task_definition
from vocabulary import TASK_IDS


DEFAULT_COMPARE_POLICIES = (
    "no_investigation",
    "investigate_when_context_hidden",
    "adaptive_cue_bandit",
)
DEFAULT_SEARCH_POLICIES = (
    "no_investigation",
    "legacy_single_probe",
    "investigate_when_context_hidden",
    "adaptive_cue_bandit",
)
DEFAULT_OUTPUT_DIR = "analysis/policy_learning_runs"

SubmitBuilder = Callable[[dict[str, Any], list[str]], HelpdeskTicketAction]
EnvFactory = Callable[[], HelpdeskTicketRoutingEnvironment]


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    investigate_hidden_context: bool
    investigate_related_ticket_hint: bool
    investigate_ambiguity_history: bool
    max_investigations_per_ticket: int
    description: str
    strategy: str = "static"


POLICY_LIBRARY: dict[str, PolicyConfig] = {
    "no_investigation": PolicyConfig(
        name="no_investigation",
        strategy="static",
        investigate_hidden_context=False,
        investigate_related_ticket_hint=False,
        investigate_ambiguity_history=False,
        max_investigations_per_ticket=0,
        description="Always submit immediately and never investigate.",
    ),
    "legacy_single_probe": PolicyConfig(
        name="legacy_single_probe",
        strategy="static",
        investigate_hidden_context=False,
        investigate_related_ticket_hint=True,
        investigate_ambiguity_history=True,
        max_investigations_per_ticket=1,
        description="Mimics the earlier single-tool hint policy.",
    ),
    "investigate_when_context_hidden": PolicyConfig(
        name="investigate_when_context_hidden",
        strategy="static",
        investigate_hidden_context=True,
        investigate_related_ticket_hint=False,
        investigate_ambiguity_history=False,
        max_investigations_per_ticket=1,
        description="Investigate once when the environment shows hidden-context pressure.",
    ),
    "adaptive_cue_bandit": PolicyConfig(
        name="adaptive_cue_bandit",
        strategy="adaptive",
        investigate_hidden_context=True,
        investigate_related_ticket_hint=True,
        investigate_ambiguity_history=True,
        max_investigations_per_ticket=3,
        description=(
            "Learn cue-conditioned tool preferences from investigation rewards on train seeds."
        ),
    ),
}

AVAILABLE_TOOLS = (
    "lookup_related_ticket",
    "lookup_requester_history",
    "lookup_internal_routing_note",
    "lookup_queue_capacity_forecast",
    "lookup_queue_cluster_summary",
)


@dataclass
class AdaptiveToolBandit:
    exploration_rounds: int = 1
    cue_tool_totals: dict[str, dict[str, float]] = field(default_factory=dict)
    cue_tool_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    global_tool_totals: dict[str, float] = field(default_factory=dict)
    global_tool_counts: dict[str, int] = field(default_factory=dict)

    def choose_tool(self, cue: str, candidate_tools: list[str]) -> str:
        for tool_name in candidate_tools:
            if self.cue_tool_counts.get(cue, {}).get(tool_name, 0) < self.exploration_rounds:
                return tool_name
        return max(
            candidate_tools,
            key=lambda tool_name: (
                self._cue_average(cue, tool_name),
                self._global_average(tool_name),
                -candidate_tools.index(tool_name),
            ),
        )

    def record_reward(self, cue: str, tool_name: str, reward: float) -> None:
        cue_totals = self.cue_tool_totals.setdefault(cue, {})
        cue_counts = self.cue_tool_counts.setdefault(cue, {})
        cue_totals[tool_name] = cue_totals.get(tool_name, 0.0) + reward
        cue_counts[tool_name] = cue_counts.get(tool_name, 0) + 1
        self.global_tool_totals[tool_name] = self.global_tool_totals.get(tool_name, 0.0) + reward
        self.global_tool_counts[tool_name] = self.global_tool_counts.get(tool_name, 0) + 1

    def export(self) -> dict[str, Any]:
        return {
            "exploration_rounds": self.exploration_rounds,
            "cue_tool_averages": {
                cue: {
                    tool_name: round(self._cue_average(cue, tool_name), 6)
                    for tool_name in sorted(tool_totals)
                }
                for cue, tool_totals in sorted(self.cue_tool_totals.items())
            },
            "global_tool_averages": {
                tool_name: round(self._global_average(tool_name), 6)
                for tool_name in sorted(self.global_tool_totals)
            },
        }

    def frozen_copy(self) -> "AdaptiveToolBandit":
        return AdaptiveToolBandit(
            exploration_rounds=self.exploration_rounds,
            cue_tool_totals={
                cue: dict(tool_totals) for cue, tool_totals in self.cue_tool_totals.items()
            },
            cue_tool_counts={
                cue: dict(tool_counts) for cue, tool_counts in self.cue_tool_counts.items()
            },
            global_tool_totals=dict(self.global_tool_totals),
            global_tool_counts=dict(self.global_tool_counts),
        )

    def _cue_average(self, cue: str, tool_name: str) -> float:
        total = self.cue_tool_totals.get(cue, {}).get(tool_name, 0.0)
        count = self.cue_tool_counts.get(cue, {}).get(tool_name, 0)
        if count == 0:
            return self._global_average(tool_name)
        return total / count

    def _global_average(self, tool_name: str) -> float:
        total = self.global_tool_totals.get(tool_name, 0.0)
        count = self.global_tool_counts.get(tool_name, 0)
        if count == 0:
            return 0.0
        return total / count


def _dedupe_preserving_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def parse_int_spec(spec: str, *, field_name: str) -> list[int]:
    values: list[int] = []
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            try:
                start = int(start_raw)
                end = int(end_raw)
            except ValueError as exc:
                raise ValueError(f"{field_name} contains an invalid range: {part!r}") from exc
            if end < start:
                raise ValueError(f"{field_name} range must be ascending: {part!r}")
            values.extend(range(start, end + 1))
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise ValueError(f"{field_name} contains an invalid integer: {part!r}") from exc
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return _dedupe_preserving_order(values)


def parse_task_ids(spec: str) -> list[int]:
    task_ids = parse_int_spec(spec, field_name="task_ids")
    unsupported = [task_id for task_id in task_ids if task_id not in TASK_IDS]
    if unsupported:
        raise ValueError(f"Unsupported task_ids: {unsupported}")
    return task_ids


def resolve_policies(spec: str) -> list[PolicyConfig]:
    names = [name.strip() for name in spec.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one policy must be specified")
    policies: list[PolicyConfig] = []
    for name in names:
        if name not in POLICY_LIBRARY:
            raise ValueError(
                f"Unknown policy {name!r}. Available policies: {sorted(POLICY_LIBRARY)}"
            )
        policies.append(POLICY_LIBRARY[name])
    return policies


def default_submit_builder(
    ticket: dict[str, Any], allowed_fields: list[str]
) -> HelpdeskTicketAction:
    inference = importlib.import_module("inference")
    candidate = inference.heuristic_action(ticket, allowed_fields)
    candidate, _ = inference.apply_domain_overrides(ticket, candidate, allowed_fields)
    candidate, _ = inference.apply_capacity_planning_overrides(
        ticket,
        candidate,
        allowed_fields,
    )
    return HelpdeskTicketAction(**candidate)


def _routing_text(ticket: dict[str, Any]) -> str:
    parts = [
        str(ticket.get("title", "")),
        str(ticket.get("description", "")),
        str(ticket.get("ambiguity_note", "")),
        str(ticket.get("planning_note", "")),
        str(ticket.get("customer_update_note", "")),
        json.dumps(ticket.get("last_tool_result") or {}, sort_keys=True),
        json.dumps(ticket.get("routing_options") or [], sort_keys=True),
        json.dumps(ticket.get("operational_context") or {}, sort_keys=True),
        json.dumps(ticket.get("cluster_summary") or {}, sort_keys=True),
        json.dumps(ticket.get("capacity_state") or {}, sort_keys=True),
        json.dumps(ticket.get("future_queue_demand") or {}, sort_keys=True),
    ]
    related_preview = ticket.get("related_ticket_preview") or {}
    parts.extend(
        [
            str(related_preview.get("title", "")),
            str(related_preview.get("description", "")),
        ]
    )
    return " ".join(parts).lower()


def infer_ticket_cue(ticket: dict[str, Any]) -> str:
    text = _routing_text(ticket)
    context_status = ticket.get("context_status") or {}
    if (
        ticket.get("planning_note")
        or ticket.get("routing_options")
        or (ticket.get("operational_context") or {}).get("incident_recommended")
        or "lookup_queue_capacity_forecast"
        in (context_status.get("recommended_tools") or [])
        or any(
            phrase in text
            for phrase in (
                "capacity",
                "saturated",
                "backlog",
                "resource pressure",
                "alternate route",
            )
        )
    ):
        return "capacity_planning"
    if (
        bool((ticket.get("operational_context") or {}).get("cluster_coordination_hint"))
        or int((ticket.get("cluster_summary") or {}).get("future_cluster_ticket_count", 0) or 0)
        > 0
        or int((ticket.get("cluster_summary") or {}).get("shared_requester_count", 0) or 0)
        > 1
        or any(
            phrase in text
            for phrase in (
                "single coordinated owner",
                "existing workstream",
                "request cluster",
                "parallel workstream",
            )
        )
    ):
        return "cluster_coordination"
    if any(
        phrase in text
        for phrase in ("re:", "follow-up", "following up", "regression", "reference ticket", "third update")
    ):
        return "follow_up"
    if any(
        phrase in text
        for phrase in (
            "pricing",
            "quote",
            "vendor offer",
            "prorating",
            "seat expansion",
            "commercial",
        )
    ):
        return "commercial_ambiguity"
    if any(
        phrase in text
        for phrase in (
            "onboarding",
            "contractor",
            "permissions error",
            "blocked by an account problem",
        )
    ):
        return "workflow_blocker"
    if any(
        phrase in text
        for phrase in ("compliance scan", "vulnerability", "policy issue", "routing note")
    ):
        return "routing_note"
    if any(
        phrase in text
        for phrase in ("still", "again", "overdue", "legal", "priority")
    ):
        return "history_pressure"
    if any(phrase in text for phrase in ("incident", "outage", "lockout", "company-wide")):
        return "incident_pressure"
    return "generic_hidden_context"


def preferred_tool_order(
    ticket: dict[str, Any],
    *,
    hidden_context_remaining: bool,
) -> list[str]:
    text = _routing_text(ticket)
    context_status = ticket.get("context_status") or {}
    last_tool_result = ticket.get("last_tool_result") or {}
    last_tool_name = str(last_tool_result.get("tool_name", "") or "")
    recommended_tools = list(context_status.get("recommended_tools") or [])

    preferred_tools: list[str] = []
    if "lookup_queue_capacity_forecast" in recommended_tools:
        preferred_tools.append("lookup_queue_capacity_forecast")
    if last_tool_name == "lookup_related_ticket":
        preferred_tools.append("lookup_requester_history")
    if last_tool_name == "lookup_requester_history":
        preferred_tools.append("lookup_internal_routing_note")
    if last_tool_name == "lookup_internal_routing_note":
        preferred_tools.append("lookup_queue_cluster_summary")
    if last_tool_name == "lookup_queue_cluster_summary":
        preferred_tools.append("lookup_queue_capacity_forecast")

    if any(
        phrase in text
        for phrase in ("re:", "follow-up", "following up", "regression", "reference ticket")
    ) or ticket.get("related_ticket_id"):
        preferred_tools.append("lookup_related_ticket")

    if any(
        phrase in text
        for phrase in (
            "pricing",
            "quote",
            "vendor offer",
            "prorating",
            "seat expansion",
            "billing-style",
            "compliance scan",
            "vulnerability",
            "onboarding workflow",
            "permissions error",
            "blocked by an account problem",
        )
    ):
        preferred_tools.append("lookup_internal_routing_note")

    if any(
        phrase in text
        for phrase in ("still", "again", "overdue", "legal", "third update", "priority")
    ):
        preferred_tools.append("lookup_requester_history")

    if infer_ticket_cue(ticket) == "cluster_coordination":
        preferred_tools.append("lookup_queue_cluster_summary")

    if infer_ticket_cue(ticket) == "capacity_planning":
        preferred_tools.append("lookup_queue_capacity_forecast")

    preferred_tools.extend(recommended_tools)

    if hidden_context_remaining:
        preferred_tools.extend(
            [
                "lookup_queue_cluster_summary",
                "lookup_queue_capacity_forecast",
                "lookup_internal_routing_note",
                "lookup_related_ticket",
                "lookup_requester_history",
            ]
        )

    deduped_tools: list[str] = []
    for tool_name in preferred_tools:
        if tool_name not in deduped_tools:
            deduped_tools.append(tool_name)
    return deduped_tools


def select_cue_based_tool(
    ticket: dict[str, Any],
    *,
    hidden_context_remaining: bool,
    used_tools: set[str],
    available_tools: set[str] | None = None,
) -> str | None:
    preferred_tools = preferred_tool_order(
        ticket,
        hidden_context_remaining=hidden_context_remaining,
    )
    available_tool_set = set(available_tools or [])
    for tool_name in preferred_tools:
        if available_tool_set and tool_name not in available_tool_set:
            continue
        if tool_name not in used_tools:
            return tool_name
    return None


def choose_operational_action(
    ticket: dict[str, Any],
    history: list[dict[str, Any]],
    available_action_types: list[str] | None = None,
) -> tuple[HelpdeskTicketAction | None, str | None]:
    if not ticket:
        return None, None
    operational_context = ticket.get("operational_context") or {}
    recommended_actions = list(operational_context.get("recommended_actions") or [])
    available_action_set = set(available_action_types or [])
    current_ticket_id = str(ticket.get("ticket_id", ""))
    prior_ticket_history = [
        entry for entry in history if entry.get("ticket_id") == current_ticket_id
    ]
    used_action_types = {
        entry.get("predicted", {}).get("action_type")
        for entry in prior_ticket_history
        if entry.get("predicted")
    }

    for action_name in ("open_incident", "request_info", "defer"):
        if action_name not in recommended_actions:
            continue
        if available_action_set and action_name not in available_action_set:
            continue
        if action_name in used_action_types:
            continue
        if action_name == "defer" and not ticket.get("tickets_after_current", 0):
            continue
        return HelpdeskTicketAction(action_type=action_name), action_name
    return None, None


def merge_ticket_context(
    ticket: dict[str, Any],
    observation: HelpdeskTicketObservation,
) -> dict[str, Any]:
    merged_ticket = dict(ticket)
    if getattr(observation, "last_tool_result", None) is not None:
        merged_ticket["last_tool_result"] = observation.last_tool_result
        if observation.last_tool_result.get("tool_name") == "lookup_queue_capacity_forecast":
            if observation.last_tool_result.get("future_queue_demand") is not None:
                merged_ticket["future_queue_demand"] = observation.last_tool_result[
                    "future_queue_demand"
                ]
            if observation.last_tool_result.get("capacity_state") is not None:
                merged_ticket["capacity_state"] = observation.last_tool_result[
                    "capacity_state"
                ]
    merged_ticket["recent_history"] = list(getattr(observation, "history", []) or [])
    merged_ticket["queue_position"] = getattr(observation, "queue_position", None)
    merged_ticket["tickets_remaining"] = getattr(observation, "tickets_remaining", None)
    merged_ticket["tickets_after_current"] = getattr(observation, "tickets_after_current", None)
    merged_ticket["available_tools"] = list(getattr(observation, "available_tools", []) or [])
    merged_ticket["available_action_types"] = list(
        getattr(observation, "available_action_types", []) or []
    )
    merged_ticket["last_reward_components"] = dict(
        getattr(observation, "last_reward_components", {}) or {}
    )
    observation_metadata = getattr(observation, "metadata", {}) or {}
    if observation_metadata.get("last_feedback_summary"):
        merged_ticket["feedback_summary"] = observation_metadata["last_feedback_summary"]
    if observation_metadata.get("capacity_state") is not None:
        merged_ticket["capacity_state"] = observation_metadata["capacity_state"]
    if observation_metadata.get("future_queue_demand") is not None:
        merged_ticket["future_queue_demand"] = observation_metadata["future_queue_demand"]
    return merged_ticket


def choose_policy_action(
    policy: PolicyConfig,
    observation: HelpdeskTicketObservation,
    investigations_by_ticket: dict[str, int],
    submit_builder: SubmitBuilder,
    *,
    used_tools_by_ticket: dict[str, set[str]] | None = None,
    adaptive_bandit: AdaptiveToolBandit | None = None,
) -> tuple[HelpdeskTicketAction, str, str | None]:
    ticket = merge_ticket_context(observation.current_ticket or {}, observation)
    ticket_id = str(ticket.get("ticket_id", ""))
    ticket_investigations = investigations_by_ticket.get(ticket_id, 0)
    used_tools = set()
    if used_tools_by_ticket is not None:
        used_tools = set(used_tools_by_ticket.get(ticket_id, set()))
    context_status = ticket.get("context_status") or {}
    hidden_context_remaining = bool(context_status.get("hidden_context_remaining"))
    available_tools = set(getattr(observation, "available_tools", []) or [])

    if ticket_investigations < policy.max_investigations_per_ticket:
        if policy.strategy == "adaptive" and adaptive_bandit is not None and hidden_context_remaining:
            candidate_tools = [
                tool_name
                for tool_name in preferred_tool_order(
                    ticket,
                    hidden_context_remaining=hidden_context_remaining,
                )
                if tool_name not in used_tools and tool_name in available_tools
            ]
            if not candidate_tools:
                candidate_tools = [
                    tool_name
                    for tool_name in AVAILABLE_TOOLS
                    if tool_name not in used_tools and tool_name in available_tools
                ]
            if candidate_tools:
                cue = infer_ticket_cue(ticket)
                tool_name = adaptive_bandit.choose_tool(cue, candidate_tools)
                return (
                    HelpdeskTicketAction(action_type="investigate", tool_name=tool_name),
                    "adaptive_bandit_investigate",
                    cue,
                )

        if policy.investigate_hidden_context and hidden_context_remaining:
            tool_name = select_cue_based_tool(
                ticket,
                hidden_context_remaining=hidden_context_remaining,
                used_tools=used_tools,
                available_tools=available_tools,
            )
            if tool_name is not None:
                return (
                    HelpdeskTicketAction(action_type="investigate", tool_name=tool_name),
                    "investigate_hidden_context",
                    infer_ticket_cue(ticket),
                )
        if (
            policy.investigate_related_ticket_hint
            and ticket.get("related_ticket_id")
            and "lookup_related_ticket" not in used_tools
        ):
            return (
                HelpdeskTicketAction(
                    action_type="investigate",
                    tool_name="lookup_related_ticket",
                ),
                "investigate_related_ticket_hint",
                infer_ticket_cue(ticket),
            )
        if (
            policy.investigate_ambiguity_history
            and (
                ticket.get("ambiguity_note")
                or ticket.get("feedback_summary")
                or hidden_context_remaining
            )
            and "lookup_requester_history" not in used_tools
        ):
            return (
                HelpdeskTicketAction(
                    action_type="investigate",
                    tool_name="lookup_requester_history",
                ),
                "investigate_ambiguity_history",
                infer_ticket_cue(ticket),
            )

    operational_action, operational_source = choose_operational_action(
        ticket,
        list(getattr(observation, "history", []) or []),
        list(getattr(observation, "available_action_types", []) or []),
    )
    if operational_action is not None and operational_source is not None:
        return operational_action, operational_source, infer_ticket_cue(ticket)

    return submit_builder(ticket, list(observation.allowed_fields)), "submit", None


def rollout_episode(
    *,
    env: HelpdeskTicketRoutingEnvironment,
    policy: PolicyConfig,
    seed: int,
    task_id: int,
    submit_builder: SubmitBuilder,
    adaptive_bandit: AdaptiveToolBandit | None = None,
    update_adaptive: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task = get_task_definition(task_id)
    observation = env.reset(seed=seed, task_id=task_id)
    investigations_by_ticket: dict[str, int] = {}
    used_tools_by_ticket: dict[str, set[str]] = {}
    episode_return = 0.0
    trajectories: list[dict[str, Any]] = []

    while not observation.done:
        ticket = merge_ticket_context(observation.current_ticket or {}, observation)
        ticket_id = str(ticket.get("ticket_id", ""))
        action, action_source, action_cue = choose_policy_action(
            policy,
            observation,
            investigations_by_ticket,
            submit_builder,
            used_tools_by_ticket=used_tools_by_ticket,
            adaptive_bandit=adaptive_bandit,
        )
        next_observation = env.step(action)
        reward_value = float(next_observation.reward or 0.0)
        episode_return += reward_value
        if action.action_type == "investigate" and ticket_id:
            investigations_by_ticket[ticket_id] = investigations_by_ticket.get(ticket_id, 0) + 1
            used_tools_by_ticket.setdefault(ticket_id, set()).add(str(action.tool_name))
            if policy.strategy == "adaptive" and adaptive_bandit is not None and update_adaptive:
                adaptive_bandit.record_reward(
                    action_cue or infer_ticket_cue(ticket),
                    str(action.tool_name),
                    reward_value,
                )

        history_entry = env.state.history_entries[-1] if env.state.history_entries else {}
        trajectories.append(
            {
                "policy": policy.name,
                "seed": seed,
                "task_id": task_id,
                "task_name": task["name"],
                "episode_id": env.state.episode_id,
                "step_index": len(trajectories) + 1,
                "ticket_id": history_entry.get("ticket_id", ticket_id),
                "action_source": action_source,
                "action_cue": action_cue,
                "action": action.model_dump(exclude_none=True),
                "step_reward": reward_value,
                "rubric_reward": next_observation.rubric_reward,
                "done": next_observation.done,
                "feedback_summary": history_entry.get("feedback_summary"),
                "reward_kind": history_entry.get("reward_kind"),
                "score": history_entry.get("score"),
                "breakdown": history_entry.get("breakdown", {}),
                "reward_components": history_entry.get("reward_components", {}),
                "context_status_before_action": ticket.get("context_status"),
            }
        )
        observation = next_observation

    queue_size = max(1, len(env.state.queue_ticket_ids))
    terminal_reward = float(observation.reward or 0.0)
    terminal_rubric_reward = (
        float(observation.rubric_reward)
        if observation.rubric_reward is not None
        else terminal_reward
    )
    summary = {
        "policy": policy.name,
        "policy_config": asdict(policy),
        "seed": seed,
        "task_id": task_id,
        "task_name": task["name"],
        "episode_id": env.state.episode_id,
        "queue_size": queue_size,
        "step_count": env.state.step_count,
        "tickets_processed": len(env.state.per_ticket_scores),
        "investigation_steps": env.state.investigation_steps,
        "episode_return": episode_return,
        "normalized_return": episode_return / queue_size,
        "terminal_reward": terminal_reward,
        "terminal_rubric_reward": terminal_rubric_reward,
        "average_ticket_score": env.state.average_score_so_far,
        "queue_management_score": env.state.queue_management_score,
        "planning_penalty_total": env.state.planning_penalty_total,
        "capacity_pressure_tickets_resolved": env.state.capacity_pressure_tickets_resolved,
        "per_ticket_scores": list(env.state.per_ticket_scores),
    }
    if adaptive_bandit is not None and policy.strategy == "adaptive":
        summary["learned_tool_values"] = adaptive_bandit.export()
    return summary, trajectories


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(mean(values), 6)


def summarize_policy_episodes(
    policy: PolicyConfig,
    episode_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    for task_id in TASK_IDS:
        task_episodes = [
            episode for episode in episode_summaries if episode["task_id"] == task_id
        ]
        if not task_episodes:
            continue
        per_task[str(task_id)] = {
            "episodes": len(task_episodes),
            "avg_episode_return": _safe_mean(
                [float(episode["episode_return"]) for episode in task_episodes]
            ),
            "avg_normalized_return": _safe_mean(
                [float(episode["normalized_return"]) for episode in task_episodes]
            ),
            "avg_terminal_reward": _safe_mean(
                [float(episode["terminal_reward"]) for episode in task_episodes]
            ),
            "avg_terminal_rubric_reward": _safe_mean(
                [float(episode["terminal_rubric_reward"]) for episode in task_episodes]
            ),
            "avg_queue_management_score": _safe_mean(
                [float(episode["queue_management_score"]) for episode in task_episodes]
            ),
            "avg_planning_penalty_total": _safe_mean(
                [float(episode["planning_penalty_total"]) for episode in task_episodes]
            ),
            "avg_capacity_pressure_tickets_resolved": _safe_mean(
                [
                    float(episode["capacity_pressure_tickets_resolved"])
                    for episode in task_episodes
                ]
            ),
            "avg_investigation_steps": _safe_mean(
                [float(episode["investigation_steps"]) for episode in task_episodes]
            ),
        }

    return {
        "policy": policy.name,
        "config": asdict(policy),
        "episodes": len(episode_summaries),
        "avg_episode_return": _safe_mean(
            [float(episode["episode_return"]) for episode in episode_summaries]
        ),
        "avg_normalized_return": _safe_mean(
            [float(episode["normalized_return"]) for episode in episode_summaries]
        ),
        "avg_terminal_reward": _safe_mean(
            [float(episode["terminal_reward"]) for episode in episode_summaries]
        ),
        "avg_terminal_rubric_reward": _safe_mean(
            [float(episode["terminal_rubric_reward"]) for episode in episode_summaries]
        ),
        "avg_queue_management_score": _safe_mean(
            [float(episode["queue_management_score"]) for episode in episode_summaries]
        ),
        "avg_planning_penalty_total": _safe_mean(
            [float(episode["planning_penalty_total"]) for episode in episode_summaries]
        ),
        "avg_capacity_pressure_tickets_resolved": _safe_mean(
            [
                float(episode["capacity_pressure_tickets_resolved"])
                for episode in episode_summaries
            ]
        ),
        "avg_investigation_steps": _safe_mean(
            [float(episode["investigation_steps"]) for episode in episode_summaries]
        ),
        "avg_ticket_score": _safe_mean(
            [float(episode["average_ticket_score"]) for episode in episode_summaries]
        ),
        "per_task": per_task,
    }


def evaluate_policy(
    policy: PolicyConfig,
    seeds: Iterable[int],
    task_ids: Iterable[int],
    *,
    env_factory: EnvFactory = HelpdeskTicketRoutingEnvironment,
    submit_builder: SubmitBuilder = default_submit_builder,
    adaptive_bandit: AdaptiveToolBandit | None = None,
    update_adaptive: bool = False,
) -> dict[str, Any]:
    episode_summaries: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []

    for seed in seeds:
        for task_id in task_ids:
            env = env_factory()
            summary, episode_trajectories = rollout_episode(
                env=env,
                policy=policy,
                seed=seed,
                task_id=task_id,
                submit_builder=submit_builder,
                adaptive_bandit=adaptive_bandit,
                update_adaptive=update_adaptive,
            )
            episode_summaries.append(summary)
            trajectories.extend(episode_trajectories)

    result = {
        "policy": policy.name,
        "summary": summarize_policy_episodes(policy, episode_summaries),
        "episodes": episode_summaries,
        "trajectories": trajectories,
    }
    if adaptive_bandit is not None and policy.strategy == "adaptive":
        result["adaptive_bandit"] = adaptive_bandit.export()
    return result


def _selection_tuple(summary: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        float(summary["avg_terminal_rubric_reward"]),
        float(summary["avg_queue_management_score"]),
        -float(summary["avg_planning_penalty_total"]),
        float(summary["avg_episode_return"]),
        float(summary["avg_normalized_return"]),
        -float(summary["avg_investigation_steps"]),
    )


def select_best_policy(policy_runs: list[dict[str, Any]]) -> dict[str, Any]:
    return max(policy_runs, key=lambda run: _selection_tuple(run["summary"]))


def _delta(best: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return round(float(best[key]) - float(baseline[key]), 6)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def compare_policies(
    policies: list[PolicyConfig],
    seeds: list[int],
    task_ids: list[int],
    *,
    output_dir: Path,
    env_factory: EnvFactory = HelpdeskTicketRoutingEnvironment,
    submit_builder: SubmitBuilder = default_submit_builder,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    policy_runs = []
    for policy in policies:
        adaptive_bandit = AdaptiveToolBandit() if policy.strategy == "adaptive" else None
        policy_runs.append(
            evaluate_policy(
                policy,
                seeds,
                task_ids,
                env_factory=env_factory,
                submit_builder=submit_builder,
                adaptive_bandit=adaptive_bandit,
                update_adaptive=policy.strategy == "adaptive",
            )
        )
    best_run = select_best_policy(policy_runs)
    baseline_run = policy_runs[0]

    report = {
        "mode": "compare",
        "task_ids": task_ids,
        "seeds": seeds,
        "selection_metric": (
            "avg_terminal_rubric_reward_then_queue_management_then_lower_planning_penalty"
        ),
        "baseline_policy": baseline_run["policy"],
        "best_policy": best_run["policy"],
        "improvement_vs_baseline": {
            "avg_episode_return": _delta(
                best_run["summary"], baseline_run["summary"], "avg_episode_return"
            ),
            "avg_normalized_return": _delta(
                best_run["summary"], baseline_run["summary"], "avg_normalized_return"
            ),
            "avg_terminal_reward": _delta(
                best_run["summary"], baseline_run["summary"], "avg_terminal_reward"
            ),
            "avg_terminal_rubric_reward": _delta(
                best_run["summary"],
                baseline_run["summary"],
                "avg_terminal_rubric_reward",
            ),
            "avg_queue_management_score": _delta(
                best_run["summary"],
                baseline_run["summary"],
                "avg_queue_management_score",
            ),
            "avg_planning_penalty_total": _delta(
                best_run["summary"],
                baseline_run["summary"],
                "avg_planning_penalty_total",
            ),
        },
        "policy_summaries": [run["summary"] for run in policy_runs],
        "ranking": [
            run["policy"]
            for run in sorted(
                policy_runs,
                key=lambda run: _selection_tuple(run["summary"]),
                reverse=True,
            )
        ],
        "adaptive_bandits": {
            run["policy"]: run["adaptive_bandit"]
            for run in policy_runs
            if "adaptive_bandit" in run
        },
        "artifacts": {
            "summary": str(output_dir / "compare_summary.json"),
            "episodes": str(output_dir / "compare_episodes.jsonl"),
            "trajectories": str(output_dir / "compare_trajectories.jsonl"),
        },
    }

    _write_json(output_dir / "compare_summary.json", report)
    _write_jsonl(
        output_dir / "compare_episodes.jsonl",
        (
            {"policy": run["policy"], **episode}
            for run in policy_runs
            for episode in run["episodes"]
        ),
    )
    _write_jsonl(
        output_dir / "compare_trajectories.jsonl",
        (trajectory for run in policy_runs for trajectory in run["trajectories"]),
    )
    return report


def search_policies(
    candidate_policies: list[PolicyConfig],
    train_seeds: list[int],
    eval_seeds: list[int],
    task_ids: list[int],
    *,
    output_dir: Path,
    env_factory: EnvFactory = HelpdeskTicketRoutingEnvironment,
    submit_builder: SubmitBuilder = default_submit_builder,
    baseline_policy_name: str = "no_investigation",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    train_runs = []
    trained_bandits: dict[str, AdaptiveToolBandit] = {}
    for policy in candidate_policies:
        adaptive_bandit = AdaptiveToolBandit() if policy.strategy == "adaptive" else None
        train_run = evaluate_policy(
            policy,
            train_seeds,
            task_ids,
            env_factory=env_factory,
            submit_builder=submit_builder,
            adaptive_bandit=adaptive_bandit,
            update_adaptive=policy.strategy == "adaptive",
        )
        train_runs.append(train_run)
        if adaptive_bandit is not None:
            trained_bandits[policy.name] = adaptive_bandit.frozen_copy()
    selected_run = select_best_policy(train_runs)
    selected_policy = POLICY_LIBRARY[selected_run["policy"]]
    eval_selected = evaluate_policy(
        selected_policy,
        eval_seeds,
        task_ids,
        env_factory=env_factory,
        submit_builder=submit_builder,
        adaptive_bandit=trained_bandits.get(selected_policy.name),
        update_adaptive=False,
    )

    baseline_policy = POLICY_LIBRARY.get(baseline_policy_name, candidate_policies[0])
    eval_baseline = evaluate_policy(
        baseline_policy,
        eval_seeds,
        task_ids,
        env_factory=env_factory,
        submit_builder=submit_builder,
        adaptive_bandit=trained_bandits.get(baseline_policy.name),
        update_adaptive=False,
    )

    report = {
        "mode": "search",
        "task_ids": task_ids,
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "selection_metric": (
            "avg_terminal_rubric_reward_then_queue_management_then_lower_planning_penalty"
        ),
        "candidate_policies": [policy.name for policy in candidate_policies],
        "selected_policy": selected_policy.name,
        "baseline_policy": baseline_policy.name,
        "train_policy_summaries": [run["summary"] for run in train_runs],
        "trained_adaptive_bandits": {
            name: bandit.export() for name, bandit in trained_bandits.items()
        },
        "eval_selected_summary": eval_selected["summary"],
        "eval_baseline_summary": eval_baseline["summary"],
        "eval_improvement_vs_baseline": {
            "avg_episode_return": _delta(
                eval_selected["summary"],
                eval_baseline["summary"],
                "avg_episode_return",
            ),
            "avg_normalized_return": _delta(
                eval_selected["summary"],
                eval_baseline["summary"],
                "avg_normalized_return",
            ),
            "avg_terminal_reward": _delta(
                eval_selected["summary"],
                eval_baseline["summary"],
                "avg_terminal_reward",
            ),
            "avg_terminal_rubric_reward": _delta(
                eval_selected["summary"],
                eval_baseline["summary"],
                "avg_terminal_rubric_reward",
            ),
            "avg_queue_management_score": _delta(
                eval_selected["summary"],
                eval_baseline["summary"],
                "avg_queue_management_score",
            ),
            "avg_planning_penalty_total": _delta(
                eval_selected["summary"],
                eval_baseline["summary"],
                "avg_planning_penalty_total",
            ),
        },
        "artifacts": {
            "summary": str(output_dir / "search_summary.json"),
            "train_episodes": str(output_dir / "search_train_episodes.jsonl"),
            "train_trajectories": str(output_dir / "search_train_trajectories.jsonl"),
            "eval_episodes": str(output_dir / "search_eval_episodes.jsonl"),
            "eval_trajectories": str(output_dir / "search_eval_trajectories.jsonl"),
        },
    }

    _write_json(output_dir / "search_summary.json", report)
    _write_jsonl(
        output_dir / "search_train_episodes.jsonl",
        (
            {"policy": run["policy"], **episode}
            for run in train_runs
            for episode in run["episodes"]
        ),
    )
    _write_jsonl(
        output_dir / "search_train_trajectories.jsonl",
        (trajectory for run in train_runs for trajectory in run["trajectories"]),
    )
    _write_jsonl(
        output_dir / "search_eval_episodes.jsonl",
        (
            {"policy": eval_selected["policy"], **episode}
            for episode in eval_selected["episodes"]
        ),
    )
    _write_jsonl(
        output_dir / "search_eval_trajectories.jsonl",
        (trajectory for trajectory in eval_selected["trajectories"]),
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run seeded local rollouts and a small policy-improvement loop for the "
            "IT helpdesk OpenEnv environment."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare fixed policy choices across repeated seeded rollouts.",
    )
    compare_parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_COMPARE_POLICIES),
        help=f"Comma-separated policy names. Available: {', '.join(POLICY_LIBRARY)}",
    )
    compare_parser.add_argument(
        "--seeds",
        default="42-51",
        help="Comma-separated seeds or ranges, for example 42-51 or 42,50,60.",
    )
    compare_parser.add_argument(
        "--task-ids",
        default="1,2,3",
        help="Comma-separated task IDs or ranges, for example 1,2,3 or 1-3.",
    )
    compare_parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON and JSONL artifacts.",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Select the best policy on train seeds, then re-evaluate on holdout seeds.",
    )
    search_parser.add_argument(
        "--candidate-policies",
        default=",".join(DEFAULT_SEARCH_POLICIES),
        help=f"Comma-separated candidate policy names. Available: {', '.join(POLICY_LIBRARY)}",
    )
    search_parser.add_argument(
        "--train-seeds",
        default="40-49",
        help="Train seeds used for reward-based policy selection.",
    )
    search_parser.add_argument(
        "--eval-seeds",
        default="50-59",
        help="Holdout seeds used for the selected policy evaluation.",
    )
    search_parser.add_argument(
        "--task-ids",
        default="1,2,3",
        help="Comma-separated task IDs or ranges, for example 1,2,3 or 1-3.",
    )
    search_parser.add_argument(
        "--baseline-policy",
        default="no_investigation",
        help="Baseline policy used for the final improvement delta.",
    )
    search_parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON and JSONL artifacts.",
    )

    return parser


def _print_summary(label: str, summary: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                label: {
                    "policy": summary["policy"],
                    "avg_episode_return": summary["avg_episode_return"],
                    "avg_normalized_return": summary["avg_normalized_return"],
                    "avg_terminal_reward": summary["avg_terminal_reward"],
                    "avg_terminal_rubric_reward": summary["avg_terminal_rubric_reward"],
                    "avg_planning_penalty_total": summary["avg_planning_penalty_total"],
                    "avg_investigation_steps": summary["avg_investigation_steps"],
                }
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.command == "compare":
        policies = resolve_policies(args.policies)
        seeds = parse_int_spec(args.seeds, field_name="seeds")
        task_ids = parse_task_ids(args.task_ids)
        report = compare_policies(
            policies,
            seeds,
            task_ids,
            output_dir=output_dir,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    candidate_policies = resolve_policies(args.candidate_policies)
    train_seeds = parse_int_spec(args.train_seeds, field_name="train_seeds")
    eval_seeds = parse_int_spec(args.eval_seeds, field_name="eval_seeds")
    task_ids = parse_task_ids(args.task_ids)
    report = search_policies(
        candidate_policies,
        train_seeds,
        eval_seeds,
        task_ids,
        output_dir=output_dir,
        baseline_policy_name=args.baseline_policy,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
