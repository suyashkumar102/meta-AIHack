#!/usr/bin/env python3
"""
Inference script for the IT Helpdesk Ticket Routing OpenEnv environment.

Environment variables
---------------------
ENV_URL
    Base URL of the running OpenEnv server.
    Default: ``http://localhost:7860``

API_BASE_URL
    LLM provider base URL (OpenAI-compatible endpoint).
    Default: ``https://router.huggingface.co/v1``

MODEL_NAME
    Model identifier to use for LLM inference.
    Default: ``gpt-4o-mini``

API_KEY
    Proxy/API authentication token injected by the evaluator.
    No default is set.

HF_TOKEN
    Backward-compatible local fallback alias for API_KEY.
    No default is set.

TASK_ID
    Optional OpenEnv task ID to run. When unset, the script defaults to the
    full declared task set so evaluator-style runs exercise every grader.

RUN_ALL_TASKS
    Optional backwards-compatible local-development alias. The script already
    runs every available task when TASK_ID is unset.

LOCAL_IMAGE_NAME
    Optional compatibility variable from the sample inference pattern.
    This script does not use ``from_docker_image()``, so the value is unused here.

When MODEL_NAME and API_KEY are set explicitly, the script calls the LLM via the
OpenAI-compatible API at API_BASE_URL. For local compatibility, HF_TOKEN is accepted
as a fallback alias for API_KEY. Otherwise it falls back to the deterministic
heuristic baseline automatically.

All stdout logs use the required structured tags: ``[START]``, ``[STEP]``, and ``[END]``.
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
from openai import OpenAI

from client import HelpdeskTicketEnvClient
from models import HelpdeskTicketAction
from vocabulary import (
    ASSIGNMENT_GROUPS,
    APP_ENV_NAME,
    ISSUE_TYPES,
    ISSUE_TYPE_TO_ASSIGNMENT_GROUP,
    ISSUE_TYPE_TO_RESOLUTION_ACTION,
    PRIORITIES,
    RESOLUTION_ACTIONS,
    TASK_IDS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "gpt-4o-mini"


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError:
        print(
            f"[WARN] {name}={raw_value!r} is not a valid integer; using {default}.",
            flush=True,
        )
        return default

API_BASE_URL = (os.getenv("API_BASE_URL") or DEFAULT_API_BASE_URL).strip()
MODEL_NAME = (os.getenv("MODEL_NAME") or DEFAULT_MODEL_NAME).strip()
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = (os.getenv("API_KEY") or HF_TOKEN or "").strip() or None
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = (os.getenv("ENV_URL") or "http://localhost:7860").strip()

SEED = _get_int_env("SEED", 42)
TASK_ID_ENV = os.getenv("TASK_ID")
RUN_ALL_TASKS_ENV = os.getenv("RUN_ALL_TASKS", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


def llm_mode_enabled() -> bool:
    return bool(API_KEY) and bool(MODEL_NAME)


llm_client: OpenAI | None = None
if llm_mode_enabled():
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


RECENT_HISTORY_LIMIT = 2
ROUTING_PRIORS = "\n".join(
    f"- {issue_type}: assignment_group={ISSUE_TYPE_TO_ASSIGNMENT_GROUP[issue_type]}, "
    f"resolution_action={ISSUE_TYPE_TO_RESOLUTION_ACTION[issue_type]}"
    for issue_type in ISSUE_TYPES
)


SYSTEM_PROMPT = """\
You are an expert IT helpdesk ticket routing agent. Given a helpdesk ticket, you must produce a JSON object with the requested fields.

Valid values:
- issue_type: {issue_types}
- priority: {priorities}
- assignment_group: {assignment_groups}
- resolution_action: {resolution_actions}

Decision rules:
- Follow this environment's label ontology exactly; do not invent categories.
- Prefer the primary operational workflow label over a secondary technical symptom.
- Keep assignment_group and resolution_action consistent with the chosen issue_type unless the ticket explicitly justifies a different choice.
- Use investigation results and recent evaluation feedback when provided.

Domain conventions:
- Enterprise pricing, quotes, plan comparisons, and commercial procurement requests map to service_request, usually with medium priority.
- Onboarding work that is blocked by an access problem still maps to onboarding when the primary workflow is onboarding; the assignment_group may still be service_desk if the ticket says onboarding cannot resolve the access issue.
- Single-user sign-in, login, MFA, or 2FA lockouts map to identity_access and are usually high priority, not critical.
- Reserve critical priority for outages, widespread business blockers, or explicit urgent critical incidents.

Routing priors:
{routing_priors}

Return ONLY valid JSON with the requested fields. No markdown, no explanation.""".format(
    issue_types=", ".join(ISSUE_TYPES),
    priorities=", ".join(PRIORITIES),
    assignment_groups=", ".join(ASSIGNMENT_GROUPS),
    resolution_actions=", ".join(RESOLUTION_ACTIONS),
    routing_priors=ROUTING_PRIORS,
)


def format_recent_history_entries(
    history: list[dict[str, Any]], limit: int = RECENT_HISTORY_LIMIT
) -> str:
    if not history:
        return ""

    lines = ["Recent evaluation feedback (latest last):"]
    for entry in history[-limit:]:
        predicted = json.dumps(entry.get("predicted", {}), sort_keys=True)
        line = (
            f"- Ticket {entry.get('ticket_id', '?')}: predicted={predicted}, "
            f"score={entry.get('score', 0.0)}"
        )
        feedback_summary = entry.get("feedback_summary")
        if feedback_summary:
            line += f", feedback={feedback_summary}"
        reward = entry.get("reward")
        if reward is not None:
            line += f", reward={reward}"
        rubric_reward = entry.get("rubric_reward")
        if rubric_reward is not None:
            line += f", rubric_reward={rubric_reward}"
        breakdown = entry.get("breakdown") or {}
        if breakdown:
            line += f", breakdown={json.dumps(breakdown, sort_keys=True)}"
        penalty_reason = entry.get("penalty_reason")
        if penalty_reason:
            line += f", penalty_reason={penalty_reason}"
        tool_result = entry.get("tool_result")
        if tool_result is not None:
            line += f", tool_result={json.dumps(tool_result, sort_keys=True)}"
        reward_components = entry.get("reward_components")
        if reward_components:
            line += f", reward_components={json.dumps(reward_components, sort_keys=True)}"
        lines.append(line)
    return "\n".join(lines)


def build_llm_user_message(ticket: dict, allowed_fields: list[str], instructions: str) -> str:
    ambiguity_note = ticket.get("ambiguity_note")
    planning_note = ticket.get("planning_note")
    customer_update_note = ticket.get("customer_update_note")
    related_preview = ticket.get("related_ticket_preview") or {}
    last_tool_result = ticket.get("last_tool_result")
    context_status = ticket.get("context_status") or {}
    operational_context = ticket.get("operational_context") or {}
    recent_history = ticket.get("recent_history") or []
    feedback_summary = ticket.get("feedback_summary")
    last_reward_components = ticket.get("last_reward_components") or {}
    investigation_budget_remaining = ticket.get("investigation_budget_remaining")
    average_score_so_far = ticket.get("average_score_so_far")
    progress_fraction = ticket.get("progress_fraction")
    capacity_state = ticket.get("capacity_state")
    future_queue_demand = ticket.get("future_queue_demand")
    routing_options = ticket.get("routing_options") or []
    extra_context_lines: list[str] = []
    if ambiguity_note:
        extra_context_lines.append(f"Ambiguity note: {ambiguity_note}")
    if planning_note:
        extra_context_lines.append(f"Planning note: {planning_note}")
    if customer_update_note:
        extra_context_lines.append(f"Customer update: {customer_update_note}")
    if related_preview:
        extra_context_lines.extend(
            [
                "Related ticket preview:",
                f"- Title: {related_preview.get('title', '')}",
                f"- Requester: {related_preview.get('requester', '')}",
                f"- Description: {related_preview.get('description', '')}",
            ]
        )
    if last_tool_result is not None:
        extra_context_lines.append(
            "Investigation result: " + json.dumps(last_tool_result, sort_keys=True)
        )
    if context_status:
        extra_context_lines.append(
            "Context status: " + json.dumps(context_status, sort_keys=True)
        )
    if operational_context:
        extra_context_lines.append(
            "Operational context: " + json.dumps(operational_context, sort_keys=True)
        )
    if capacity_state:
        extra_context_lines.append(
            "Queue capacity state: " + json.dumps(capacity_state, sort_keys=True)
        )
    if future_queue_demand:
        extra_context_lines.append(
            "Future queue demand: " + json.dumps(future_queue_demand, sort_keys=True)
        )
    if routing_options:
        extra_context_lines.append(
            "Routing options: " + json.dumps(routing_options, sort_keys=True)
        )
    if feedback_summary:
        extra_context_lines.append(f"Latest environment feedback: {feedback_summary}")
    if last_reward_components:
        extra_context_lines.append(
            "Latest reward components: "
            + json.dumps(last_reward_components, sort_keys=True)
        )
    recent_history_block = format_recent_history_entries(recent_history)
    if recent_history_block:
        extra_context_lines.append(recent_history_block)
    queue_position = ticket.get("queue_position")
    tickets_remaining = ticket.get("tickets_remaining")
    if queue_position is not None and tickets_remaining is not None:
        extra_context_lines.append(
            f"Queue context: queue_position={queue_position}, tickets_remaining={tickets_remaining}"
        )
    if average_score_so_far is not None:
        extra_context_lines.append(f"Average score so far: {average_score_so_far}")
    if progress_fraction is not None:
        extra_context_lines.append(f"Episode progress: {progress_fraction}")
    if investigation_budget_remaining is not None:
        extra_context_lines.append(
            f"Investigation budget remaining: {investigation_budget_remaining}"
        )
    extra_context_block = ""
    if extra_context_lines:
        extra_context_block = "\n" + "\n".join(extra_context_lines)

    return (
        f"Instructions: {instructions}\n\n"
        f"Allowed fields: {', '.join(allowed_fields)}\n\n"
        f"Title: {ticket.get('title', '')}\n"
        f"Requester: {ticket.get('requester', '')}\n"
        f"Description: {ticket.get('description', '')}"
        f"{extra_context_block}\n\n"
        f"Respond with JSON containing ONLY these fields: {', '.join(allowed_fields)}"
    )


def call_llm(ticket: dict, allowed_fields: list[str], instructions: str) -> dict:
    assert llm_client is not None, "LLM client not configured"
    user_msg = build_llm_user_message(ticket, allowed_fields, instructions)

    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=256,
    )

    text = response.choices[0].message.content or "{}"
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _format_bool(value: bool) -> str:
    return str(bool(value)).lower()


def clamp_reported_score(score: float) -> float:
    return max(0.0, min(1.0, score))


def _format_action_for_log(action: HelpdeskTicketAction) -> str:
    return json.dumps(
        action.model_dump(exclude_none=True),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _format_error_for_log(error: str | None) -> str:
    if not error:
        return "null"
    return error.replace("\r", " ").replace("\n", " ")


def _format_reward_for_log(reward: float | None) -> str:
    return f"{clamp_reported_score(float(reward or 0.0)):.2f}"


def log_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={APP_ENV_NAME} model={MODEL_NAME}", flush=True)


def log_step(
    *,
    step: int,
    action: HelpdeskTicketAction,
    reward: float | None,
    done: bool,
    error: str | None,
) -> None:
    print(
        "[STEP] "
        f"step={step} "
        f"action={_format_action_for_log(action)} "
        f"reward={_format_reward_for_log(reward)} "
        f"done={_format_bool(done)} "
        f"error={_format_error_for_log(error)}",
        flush=True,
    )


def log_end(*, success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(_format_reward_for_log(reward) for reward in rewards)
    print(
        "[END] "
        f"success={_format_bool(success)} "
        f"steps={steps} "
        f"score={clamp_reported_score(score):.2f} "
        f"rewards={rewards_str}",
        flush=True,
    )


def get_tasks_to_run(available_tasks: dict) -> list[int]:
    available_task_ids = sorted(int(task_id) for task_id in available_tasks)
    if TASK_ID_ENV:
        try:
            task_id = int(TASK_ID_ENV)
        except ValueError:
            print(f"[ERROR] TASK_ID={TASK_ID_ENV!r} is not a valid integer", flush=True)
            raise SystemExit(1)
        if task_id not in available_task_ids:
            print(
                f"[ERROR] TASK_ID={task_id} not in available tasks {available_task_ids}",
                flush=True,
            )
            raise SystemExit(1)
        return [task_id]
    if not available_task_ids:
        return []
    # Default to all declared tasks so validator-style runs exercise all graders.
    return available_task_ids


# ---------------------------------------------------------------------------
# Heuristic fallback (no LLM needed)
# ---------------------------------------------------------------------------

KEYWORD_ISSUE_TYPES = {
    "invoice": "billing_license",
    "charge": "billing_license",
    "refund": "billing_license",
    "payment": "billing_license",
    "billing": "billing_license",
    "license": "billing_license",
    "sign in": "identity_access",
    "login": "identity_access",
    "password": "identity_access",
    "locked": "identity_access",
    "2fa": "identity_access",
    "sso": "identity_access",
    "bug": "application_support",
    "error": "application_support",
    "exception": "application_support",
    "crash": "application_support",
    "production": "application_support",
    "latency": "application_support",
    "timeout": "application_support",
    "webhook": "application_support",
    "migration": "application_support",
    "pricing": "service_request",
    "quote": "service_request",
    "demo": "service_request",
    "enterprise": "service_request",
    "rollout": "service_request",
    "sandbox": "service_request",
    "trial": "service_request",
    "seat": "service_request",
    "seats": "service_request",
    "spam": "spam_phishing",
    "click now": "spam_phishing",
    "guaranteed": "spam_phishing",
    "unsubscribe": "spam_phishing",
    "phishing": "spam_phishing",
    "compromised": "spam_phishing",
    "compliance": "security_compliance",
    "regulation": "security_compliance",
    "gdpr": "security_compliance",
    "audit": "security_compliance",
    "pentest": "security_compliance",
    "vulnerabilities": "security_compliance",
    "security policy": "security_compliance",
    "onboarding": "onboarding",
    "welcome": "onboarding",
    "getting started": "onboarding",
    "new hire": "onboarding",
    "contractor": "onboarding",
    "feedback": "feature_request",
    "suggestion": "feature_request",
    "improve": "feature_request",
    "roadmap": "feature_request",
    "export": "feature_request",
}


CRITICAL_PRIORITY_KEYWORDS = (
    "urgent",
    "critical",
    "blocking",
    "asap",
    "immediately",
    "locked out",
    "outage",
)

HIGH_PRIORITY_KEYWORDS = (
    "important",
    "high priority",
    "revenue",
    "today",
    "eod",
)

LOW_PRIORITY_KEYWORDS = ("low", "whenever", "no rush")

ESCALATE_KEYWORDS = (
    "refund",
    "charged twice",
    "still haven't",
    "following up",
    "needs immediate resolution",
    "locked out",
    "suspended",
    "legal",
)

FULFILL_KEYWORDS = (
    "please provide",
    "confirmation",
    "data processing addendum",
    "guidance",
    "fix",
    "reproducible",
    "outage",
    "policy",
    "mfa enabled",
)

PRICING_REQUEST_KEYWORDS = (
    "pricing breakdown",
    "enterprise tier pricing",
    "enterprise plan",
    "compare your enterprise plan",
    "comparing your enterprise plan",
    "quote",
    "pricing quote",
    "commercial proposal",
    "vendor comparison",
)

ONBOARDING_WORKFLOW_KEYWORDS = (
    "onboarding",
    "new hire",
    "contractor",
    "provisioned",
    "kickoff onboarding",
)

ACCESS_BLOCKER_KEYWORDS = (
    "access issue",
    "permissions error",
    "permission error",
    "account access is blocked",
    "cannot sign in",
    "can't sign in",
    "locked",
    "2fa",
    "mfa",
)

SERVICE_DESK_ONBOARDING_ESCALATION_KEYWORDS = (
    "onboarding team cannot resolve access issues",
    "routing to service desk",
    "route to service desk",
    "service desk",
)

CRITICAL_INCIDENT_KEYWORDS = (
    "outage",
    "company-wide",
    "all users",
    "widespread",
    "production down",
    "critical incident",
    "sev1",
)

HIGH_PRIORITY_SIGNAL_KEYWORDS = (
    "locked",
    "blocked",
    "cannot sign in",
    "can't sign in",
    "2fa",
    "mfa",
    "expedite",
    "start monday",
    "asap",
    "today",
    "eod",
    "urgent",
)

TIME_SENSITIVE_PRIORITY_KEYWORDS = (
    "expedite",
    "start monday",
    "today",
    "asap",
    "eod",
    "urgent",
    "immediately",
)


def build_routing_text(ticket: dict) -> str:
    related_preview = ticket.get("related_ticket_preview") or {}
    last_tool_result = ticket.get("last_tool_result") or {}
    routing_options = ticket.get("routing_options") or []
    operational_context = ticket.get("operational_context") or {}
    cluster_summary = ticket.get("cluster_summary") or {}
    return " ".join(
        [
            ticket.get("title", ""),
            ticket.get("description", ""),
            ticket.get("ambiguity_note", ""),
            ticket.get("planning_note", ""),
            ticket.get("customer_update_note", ""),
            related_preview.get("title", ""),
            related_preview.get("description", ""),
            json.dumps(last_tool_result, sort_keys=True),
            json.dumps(routing_options, sort_keys=True),
            json.dumps(operational_context, sort_keys=True),
            json.dumps(cluster_summary, sort_keys=True),
            json.dumps(ticket.get("capacity_state") or {}, sort_keys=True),
            json.dumps(ticket.get("future_queue_demand") or {}, sort_keys=True),
        ]
    ).lower()


def heuristic_priority(text: str) -> str:
    if any(word in text for word in CRITICAL_PRIORITY_KEYWORDS):
        return "critical"
    if any(word in text for word in HIGH_PRIORITY_KEYWORDS):
        return "high"
    if any(word in text for word in LOW_PRIORITY_KEYWORDS):
        return "low"
    return "medium"


def heuristic_resolution_action(text: str, issue_type: str) -> str:
    if issue_type == "spam_phishing":
        return "ignore"
    if issue_type == "service_request":
        return "assign"
    if issue_type in {"general_inquiry", "feature_request"}:
        return "acknowledge"
    if any(keyword in text for keyword in ESCALATE_KEYWORDS):
        return "escalate"
    if any(keyword in text for keyword in FULFILL_KEYWORDS):
        return "fulfill"
    return ISSUE_TYPE_TO_RESOLUTION_ACTION.get(issue_type, "acknowledge")


def heuristic_assignment_group(text: str, issue_type: str) -> str:
    if issue_type == "onboarding":
        if any(keyword in text for keyword in SERVICE_DESK_ONBOARDING_ESCALATION_KEYWORDS):
            return "service_desk"
        if any(keyword in text for keyword in ACCESS_BLOCKER_KEYWORDS) and any(
            keyword in text for keyword in ONBOARDING_WORKFLOW_KEYWORDS
        ):
            return "service_desk"
    return ISSUE_TYPE_TO_ASSIGNMENT_GROUP.get(issue_type, "service_desk")


def infer_issue_type(text: str) -> str:
    issue_type = "general_inquiry"
    for kw, mapped_issue_type in KEYWORD_ISSUE_TYPES.items():
        if kw in text:
            issue_type = mapped_issue_type
            break
    return issue_type


def heuristic_action(
    ticket: dict, allowed_fields: list[str], issue_type_override: str | None = None
) -> dict:
    text = build_routing_text(ticket)

    issue_type = issue_type_override or infer_issue_type(text)
    priority = heuristic_priority(text)
    resolution_action = heuristic_resolution_action(text, issue_type)

    result: dict[str, str] = {}
    if "issue_type" in allowed_fields:
        result["issue_type"] = issue_type
    if "priority" in allowed_fields:
        result["priority"] = priority
    if "assignment_group" in allowed_fields:
        result["assignment_group"] = heuristic_assignment_group(text, issue_type)
    if "resolution_action" in allowed_fields:
        result["resolution_action"] = resolution_action
    return result


def _get_routing_options(ticket: dict[str, Any]) -> list[dict[str, Any]]:
    options = ticket.get("routing_options") or []
    return [option for option in options if isinstance(option, dict)]


def _get_routing_option_by_label(
    ticket: dict[str, Any],
    label: str | None,
) -> dict[str, Any] | None:
    if label is None:
        return None
    for option in _get_routing_options(ticket):
        if option.get("label") == label:
            return option
    return None


def _route_option_fields_match(
    option: dict[str, Any],
    candidate: dict[str, Any],
    allowed_fields: list[str],
) -> bool:
    for field in ("issue_type", "priority", "assignment_group", "resolution_action"):
        if field not in allowed_fields:
            continue
        option_value = option.get(field)
        candidate_value = candidate.get(field)
        if option_value is None or candidate_value is None:
            continue
        if str(option_value) != str(candidate_value):
            return False
    return True


def _preferred_routing_label(ticket: dict[str, Any]) -> str | None:
    last_tool_result = ticket.get("last_tool_result") or {}
    tool_name = str(last_tool_result.get("tool_name", "") or "")
    preferred_label = str(last_tool_result.get("preferred_route_label", "") or "")
    if tool_name == "lookup_queue_capacity_forecast" and preferred_label in {
        "primary",
        "alternate",
    }:
        return preferred_label
    return None


def apply_capacity_planning_overrides(
    ticket: dict[str, Any],
    candidate: dict[str, Any],
    allowed_fields: list[str],
) -> tuple[dict[str, Any], list[str]]:
    updated = dict(candidate)
    reasons: list[str] = []
    preferred_label = _preferred_routing_label(ticket)
    preferred_option = _get_routing_option_by_label(ticket, preferred_label)
    if preferred_option is None:
        return updated, reasons

    current_matching_label = None
    for option in _get_routing_options(ticket):
        if _route_option_fields_match(option, updated, allowed_fields):
            current_matching_label = option.get("label")
            break

    if current_matching_label == preferred_label:
        return updated, reasons

    for field in ("issue_type", "priority", "assignment_group", "resolution_action"):
        if field not in allowed_fields:
            continue
        option_value = preferred_option.get(field)
        if option_value is None:
            continue
        updated[field] = option_value

    last_tool_result = ticket.get("last_tool_result") or {}
    reasons.append(
        "planning_override="
        f"{preferred_label}(primary_pressure={last_tool_result.get('primary_pressure')},"
        f"alternate_pressure={last_tool_result.get('alternate_pressure')})"
    )
    return updated, reasons


def apply_domain_overrides(
    ticket: dict, candidate: dict[str, Any], allowed_fields: list[str]
) -> tuple[dict[str, Any], list[str]]:
    updated = dict(candidate)
    reasons: list[str] = []
    text = build_routing_text(ticket)

    issue_type = updated.get("issue_type")
    if "issue_type" in allowed_fields and issue_type is not None:
        if (
            issue_type in {"billing_license", "general_inquiry"}
            and any(keyword in text for keyword in PRICING_REQUEST_KEYWORDS)
        ):
            updated["issue_type"] = "service_request"
            issue_type = "service_request"
            reasons.append("override_issue_type=service_request(pricing_request)")
        elif (
            issue_type == "identity_access"
            and any(keyword in text for keyword in ONBOARDING_WORKFLOW_KEYWORDS)
            and any(keyword in text for keyword in ACCESS_BLOCKER_KEYWORDS)
        ):
            updated["issue_type"] = "onboarding"
            issue_type = "onboarding"
            reasons.append("override_issue_type=onboarding(onboarding_access_blocker)")

    if issue_type is not None:
        if "assignment_group" in allowed_fields:
            desired_group = heuristic_assignment_group(text, issue_type)
            if updated.get("assignment_group") != desired_group:
                updated["assignment_group"] = desired_group
                reasons.append(f"override_assignment_group={desired_group}")
        if "resolution_action" in allowed_fields:
            desired_resolution = heuristic_resolution_action(text, issue_type)
            if updated.get("resolution_action") != desired_resolution:
                updated["resolution_action"] = desired_resolution
                reasons.append(f"override_resolution_action={desired_resolution}")

    if "priority" in allowed_fields and updated.get("priority") is not None:
        priority = updated["priority"]
        has_critical_signal = any(keyword in text for keyword in CRITICAL_INCIDENT_KEYWORDS)
        has_high_signal = any(keyword in text for keyword in HIGH_PRIORITY_SIGNAL_KEYWORDS)
        if priority == "critical" and not has_critical_signal:
            updated["priority"] = "high" if has_high_signal else "medium"
            reasons.append(f"override_priority={updated['priority']}(deescalated_from_critical)")
        elif (
            priority == "high"
            and issue_type in {"service_request", "onboarding"}
            and not any(keyword in text for keyword in TIME_SENSITIVE_PRIORITY_KEYWORDS)
        ):
            updated["priority"] = "medium"
            reasons.append("override_priority=medium(nonurgent_workflow_request)")
        elif (
            priority == "medium"
            and issue_type == "identity_access"
            and any(keyword in text for keyword in ("cannot sign in", "can't sign in", "2fa", "mfa", "locked"))
            and not has_critical_signal
        ):
            updated["priority"] = "high"
            reasons.append("override_priority=high(identity_lockout)")

    return updated, reasons


def build_action(
    ticket: dict, allowed_fields: list[str], instructions: str
) -> tuple[HelpdeskTicketAction, str, str | None]:
    heuristic_dict = heuristic_action(ticket, allowed_fields)
    heuristic_dict, heuristic_override_reasons = apply_domain_overrides(
        ticket,
        heuristic_dict,
        allowed_fields,
    )
    heuristic_dict, heuristic_planning_reasons = apply_capacity_planning_overrides(
        ticket,
        heuristic_dict,
        allowed_fields,
    )

    if llm_client is None:
        fallback_reason = None
        reason_parts = []
        if heuristic_override_reasons:
            reason_parts.append(f"domain_overrides={heuristic_override_reasons}")
        if heuristic_planning_reasons:
            reason_parts.append(f"planning_overrides={heuristic_planning_reasons}")
        if reason_parts:
            fallback_reason = "; ".join(reason_parts)
        return HelpdeskTicketAction(**heuristic_dict), "heuristic", fallback_reason

    try:
        llm_dict = call_llm(ticket, allowed_fields, instructions)
        validated_llm_fields: dict[str, Any] = {}
        rejected_fields: list[str] = []
        for field in allowed_fields:
            value = llm_dict.get(field)
            if value is None:
                continue
            try:
                HelpdeskTicketAction(**{field: value})
            except Exception:
                rejected_fields.append(field)
                continue
            validated_llm_fields[field] = value

        if not validated_llm_fields:
            raise ValueError("LLM returned no allowed fields")

        candidate = heuristic_action(
            ticket,
            allowed_fields,
            issue_type_override=validated_llm_fields.get("issue_type"),
        )
        candidate.update(validated_llm_fields)
        accepted_fields = list(validated_llm_fields)
        candidate, override_reasons = apply_domain_overrides(
            ticket,
            candidate,
            allowed_fields,
        )
        candidate, planning_override_reasons = apply_capacity_planning_overrides(
            ticket,
            candidate,
            allowed_fields,
        )

        backfilled_fields = [field for field in allowed_fields if field not in accepted_fields]
        if (
            backfilled_fields
            or rejected_fields
            or override_reasons
            or planning_override_reasons
        ):
            reason_parts = []
            if backfilled_fields:
                reason_parts.append(f"heuristic_backfill={backfilled_fields}")
            if rejected_fields:
                reason_parts.append(f"invalid_llm_fields={rejected_fields}")
            if override_reasons:
                reason_parts.append(f"domain_overrides={override_reasons}")
            if planning_override_reasons:
                reason_parts.append(f"planning_overrides={planning_override_reasons}")
            return (
                HelpdeskTicketAction(**candidate),
                "llm_backfilled",
                "; ".join(reason_parts),
            )

        return HelpdeskTicketAction(**candidate), "llm", None
    except Exception as exc:
        return (
            HelpdeskTicketAction(**heuristic_dict),
            "heuristic_fallback",
            "; ".join(
                part
                for part in (
                    str(exc),
                    (
                        f"domain_overrides={heuristic_override_reasons}"
                        if heuristic_override_reasons
                        else None
                    ),
                    (
                        f"planning_overrides={heuristic_planning_reasons}"
                        if heuristic_planning_reasons
                        else None
                    ),
                )
                if part
            ),
        )


def should_investigate(
    ticket: dict,
    history: list[dict[str, Any]],
    available_tools: list[str] | None = None,
) -> tuple[bool, str | None]:
    if not ticket:
        return False, None
    available_tool_set = set(available_tools or [])
    context_status = ticket.get("context_status") or {}
    hidden_context_remaining = bool(context_status.get("hidden_context_remaining"))
    investigation_required = bool(context_status.get("investigation_required"))
    if not investigation_required and not hidden_context_remaining:
        return False, None
    current_ticket_id = ticket.get("ticket_id")
    prior_ticket_history = [
        entry
        for entry in history
        if entry.get("ticket_id") == current_ticket_id
    ]
    already_investigated = any(
        entry.get("ticket_id") == current_ticket_id
        and entry.get("predicted", {}).get("action_type") == "investigate"
        for entry in history
    )
    investigations_used = sum(
        1
        for entry in prior_ticket_history
        if entry.get("predicted", {}).get("action_type") == "investigate"
    )
    if investigations_used >= 3:
        return False, None

    used_tools = {
        entry.get("predicted", {}).get("tool_name")
        for entry in prior_ticket_history
        if entry.get("predicted", {}).get("action_type") == "investigate"
    }
    recommended_tools = [
        tool_name
        for tool_name in context_status.get("recommended_tools", [])
        if tool_name not in used_tools
        and (not available_tool_set or tool_name in available_tool_set)
    ]
    if hidden_context_remaining and recommended_tools:
        return True, recommended_tools[0]

    routing_text = build_routing_text(ticket)
    last_tool_result = ticket.get("last_tool_result") or {}
    last_tool_name = str(last_tool_result.get("tool_name", "") or "")

    follow_up_signal = any(
        phrase in routing_text
        for phrase in (
            "re:",
            "follow-up",
            "following up",
            "regression",
            "reference ticket",
            "third update",
            "still",
            "unresolved",
        )
    )
    routing_ambiguity_signal = any(
        phrase in routing_text
        for phrase in (
            "billing-style",
            "prorating",
            "seat expansion",
            "vendor offer",
            "pricing",
            "compliance scan",
            "vulnerability",
            "onboarding workflow",
            "blocked by an account problem",
            "permissions error",
            "mixed workflow",
        )
    )
    requester_history_signal = any(
        phrase in routing_text
        for phrase in (
            "still haven't",
            "third update",
            "again",
            "follow-up",
            "priority",
            "legal",
            "overdue",
            "escalating",
        )
    )
    operational_context = ticket.get("operational_context") or {}
    cluster_summary = ticket.get("cluster_summary") or {}
    cluster_signal = (
        bool(operational_context.get("cluster_coordination_hint"))
        or int(cluster_summary.get("future_cluster_ticket_count", 0) or 0) > 0
        or int(cluster_summary.get("shared_requester_count", 0) or 0) > 1
        or any(
            phrase in routing_text
            for phrase in (
                "single coordinated owner",
                "existing workstream",
                "request cluster",
                "parallel workstream",
            )
        )
    )

    preferred_tools: list[str] = []
    if last_tool_name == "lookup_related_ticket":
        preferred_tools.append("lookup_requester_history")
    if last_tool_name == "lookup_requester_history":
        preferred_tools.append("lookup_internal_routing_note")
    if last_tool_name == "lookup_internal_routing_note":
        preferred_tools.append("lookup_queue_cluster_summary")
    if last_tool_name == "lookup_queue_cluster_summary":
        preferred_tools.append("lookup_queue_capacity_forecast")
    if follow_up_signal or ticket.get("related_ticket_id"):
        preferred_tools.append("lookup_related_ticket")
    if routing_ambiguity_signal or hidden_context_remaining:
        preferred_tools.append("lookup_internal_routing_note")
    if requester_history_signal:
        preferred_tools.append("lookup_requester_history")
    if cluster_signal:
        preferred_tools.append("lookup_queue_cluster_summary")
    if hidden_context_remaining:
        preferred_tools.extend(
            [
                "lookup_queue_cluster_summary",
                "lookup_queue_capacity_forecast",
                "lookup_related_ticket",
                "lookup_internal_routing_note",
                "lookup_requester_history",
            ]
        )

    for tool_name in preferred_tools:
        if available_tool_set and tool_name not in available_tool_set:
            continue
        if tool_name not in used_tools:
            return True, tool_name

    if already_investigated and not hidden_context_remaining:
        return False, None
    return False, None


def choose_operational_action(
    ticket: dict,
    history: list[dict[str, Any]],
    available_action_types: list[str] | None = None,
) -> tuple[HelpdeskTicketAction | None, str | None]:
    if not ticket:
        return None, None
    operational_context = ticket.get("operational_context") or {}
    recommended_actions = list(operational_context.get("recommended_actions") or [])
    available_action_set = set(available_action_types or [])
    current_ticket_id = ticket.get("ticket_id")
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
        if action_name == "defer" and ticket.get("tickets_after_current", 0) <= 0:
            continue
        return HelpdeskTicketAction(action_type=action_name), action_name
    return None, None


def merge_ticket_context(ticket: dict, observation: Any) -> dict:
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
    merged_ticket["recent_history"] = list(getattr(observation, "history", []))
    merged_ticket["queue_position"] = getattr(observation, "queue_position", None)
    merged_ticket["tickets_remaining"] = getattr(observation, "tickets_remaining", None)
    merged_ticket["tickets_after_current"] = getattr(observation, "tickets_after_current", None)
    merged_ticket["investigation_budget_remaining"] = getattr(
        observation,
        "investigation_budget_remaining",
        None,
    )
    merged_ticket["average_score_so_far"] = getattr(observation, "average_score_so_far", None)
    merged_ticket["progress_fraction"] = getattr(observation, "progress_fraction", None)
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
    if observation_metadata.get("planning_penalty_total") is not None:
        merged_ticket["planning_penalty_total"] = observation_metadata["planning_penalty_total"]
    if observation_metadata.get("planning_penalty_applied") is not None:
        merged_ticket["planning_penalty_applied"] = observation_metadata["planning_penalty_applied"]
    return merged_ticket


# ---------------------------------------------------------------------------
# Main loop using the HTTP-based sync EnvClient for multi-step episodes
# ---------------------------------------------------------------------------


def run() -> None:
    http = httpx.Client(base_url=ENV_URL, timeout=30.0)
    health = http.get("/health")
    health.raise_for_status()

    tasks_resp = http.get("/tasks")
    tasks_resp.raise_for_status()
    available_tasks = {t["id"]: t for t in tasks_resp.json()["tasks"]}
    http.close()

    tasks_to_run = get_tasks_to_run(available_tasks)
    if not tasks_to_run:
        return
    for task_id in tasks_to_run:
        if task_id not in available_tasks:
            continue

        task = available_tasks[task_id]
        log_start(task["name"])

        sync_client = HelpdeskTicketEnvClient(base_url=ENV_URL).sync()
        with sync_client:
            result = sync_client.reset(seed=SEED, task_id=task_id)
            obs = result.observation

            task_step_rewards: list[float] = []
            step_num = 0

            while not result.done:
                ticket = obs.current_ticket
                if ticket is None:
                    break

                while getattr(obs, "investigation_budget_remaining", 0) > 0:
                    investigate, tool_name = should_investigate(
                        ticket,
                        obs.history,
                        list(getattr(obs, "available_tools", []) or []),
                    )
                    if not investigate or tool_name is None:
                        break
                    tool_action = HelpdeskTicketAction(
                        action_type="investigate",
                        tool_name=tool_name,
                        tool_target_ticket_id=ticket.get("related_ticket_id"),
                    )
                    result = sync_client.step(tool_action)
                    obs = result.observation
                    step_num += 1
                    reward = float(result.reward or 0.0)
                    if result.reward is not None:
                        task_step_rewards.append(reward)
                    log_step(
                        step=step_num,
                        action=tool_action,
                        reward=reward,
                        done=bool(result.done),
                        error=None,
                    )
                    if result.done:
                        break
                    ticket = obs.current_ticket
                    if ticket is None:
                        break
                if result.done:
                    break
                ticket = obs.current_ticket
                if ticket is None:
                    break

                ticket_with_context = merge_ticket_context(ticket, obs)
                operational_action, operational_source = choose_operational_action(
                    ticket_with_context,
                    obs.history,
                    list(getattr(obs, "available_action_types", []) or []),
                )
                if operational_action is not None and operational_source is not None:
                    result = sync_client.step(operational_action)
                    obs = result.observation
                    step_num += 1
                    reward = float(result.reward or 0.0)
                    if result.reward is not None:
                        task_step_rewards.append(reward)
                    log_step(
                        step=step_num,
                        action=operational_action,
                        reward=reward,
                        done=bool(result.done),
                        error=operational_source,
                    )
                    continue
                action, action_source, fallback_reason = build_action(
                    ticket_with_context,
                    obs.allowed_fields,
                    obs.instructions,
                )
                result = sync_client.step(action)
                obs = result.observation

                step_num += 1
                reward = float(result.reward or 0.0)
                if result.reward is not None:
                    task_step_rewards.append(reward)

                log_step(
                    step=step_num,
                    action=action,
                    reward=reward,
                    done=bool(result.done),
                    error=fallback_reason,
                )

        final_rubric_reward = getattr(obs, "rubric_reward", None)
        final_reward = (
            float(final_rubric_reward)
            if final_rubric_reward is not None
            else (task_step_rewards[-1] if task_step_rewards else 0.0)
        )
        reported_score = clamp_reported_score(final_reward)
        log_end(
            success=bool(obs.done),
            steps=step_num,
            score=reported_score,
            rewards=task_step_rewards,
        )


if __name__ == "__main__":
    run()
