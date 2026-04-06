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
    Default: ``<your-active-model>``

HF_TOKEN
    HuggingFace authentication token for the LLM provider.
    No default is set.

TASK_ID
    Optional OpenEnv task ID to run. When unset, the script defaults to the
    first available task so it still emits exactly one ``[START]`` ... ``[END]``
    block for evaluator-style runs.

RUN_ALL_TASKS
    Optional local-development override. Set to ``1`` to run every available
    task in sequence and print the aggregate closing ``[END]`` summary.

LOCAL_IMAGE_NAME
    Optional compatibility variable from the sample inference pattern.
    This script does not use ``from_docker_image()``, so the value is unused here.

When both MODEL_NAME and HF_TOKEN are set explicitly, the script calls the LLM via the
OpenAI-compatible API at API_BASE_URL. Otherwise it falls back to the deterministic
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
DEFAULT_MODEL_NAME = "<your-active-model>"

API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

SEED = 42
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
    return bool(HF_TOKEN) and MODEL_NAME != DEFAULT_MODEL_NAME


llm_client: OpenAI | None = None
if llm_mode_enabled():
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


SYSTEM_PROMPT = """\
You are an expert IT helpdesk ticket routing agent. Given a helpdesk ticket, you must produce a JSON object with the requested fields.

Valid values:
- issue_type: {issue_types}
- priority: {priorities}
- assignment_group: {assignment_groups}
- resolution_action: {resolution_actions}

Return ONLY valid JSON with the requested fields. No markdown, no explanation.""".format(
    issue_types=", ".join(ISSUE_TYPES),
    priorities=", ".join(PRIORITIES),
    assignment_groups=", ".join(ASSIGNMENT_GROUPS),
    resolution_actions=", ".join(RESOLUTION_ACTIONS),
)


def call_llm(ticket: dict, allowed_fields: list[str], instructions: str) -> dict:
    assert llm_client is not None, "LLM client not configured"
    ambiguity_note = ticket.get("ambiguity_note")
    related_preview = ticket.get("related_ticket_preview") or {}
    last_tool_result = ticket.get("last_tool_result")
    extra_context_lines: list[str] = []
    if ambiguity_note:
        extra_context_lines.append(f"Ambiguity note: {ambiguity_note}")
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
    extra_context_block = ""
    if extra_context_lines:
        extra_context_block = "\n" + "\n".join(extra_context_lines)

    user_msg = (
        f"Instructions: {instructions}\n\n"
        f"Allowed fields: {', '.join(allowed_fields)}\n\n"
        f"Title: {ticket['title']}\n"
        f"Requester: {ticket['requester']}\n"
        f"Description: {ticket['description']}"
        f"{extra_context_block}\n\n"
        f"Respond with JSON containing ONLY these fields: {', '.join(allowed_fields)}"
    )

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


def emit_log(tag: str, **payload: Any) -> None:
    print(f"[{tag}] {json.dumps(payload, sort_keys=True, ensure_ascii=True)}")


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
    if RUN_ALL_TASKS_ENV:
        return available_task_ids
    if not available_task_ids:
        return []
    # Default to a single task so evaluation emits exactly one START/END block.
    return [available_task_ids[0]]


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


def heuristic_action(ticket: dict, allowed_fields: list[str]) -> dict:
    related_preview = ticket.get("related_ticket_preview") or {}
    last_tool_result = ticket.get("last_tool_result") or {}
    text = " ".join(
        [
            ticket.get("title", ""),
            ticket.get("description", ""),
            ticket.get("ambiguity_note", ""),
            related_preview.get("title", ""),
            related_preview.get("description", ""),
            json.dumps(last_tool_result, sort_keys=True),
        ]
    ).lower()

    issue_type = "general_inquiry"
    for kw, mapped_issue_type in KEYWORD_ISSUE_TYPES.items():
        if kw in text:
            issue_type = mapped_issue_type
            break

    priority = heuristic_priority(text)
    resolution_action = heuristic_resolution_action(text, issue_type)

    result: dict[str, str] = {}
    if "issue_type" in allowed_fields:
        result["issue_type"] = issue_type
    if "priority" in allowed_fields:
        result["priority"] = priority
    if "assignment_group" in allowed_fields:
        result["assignment_group"] = ISSUE_TYPE_TO_ASSIGNMENT_GROUP.get(
            issue_type, "service_desk"
        )
    if "resolution_action" in allowed_fields:
        result["resolution_action"] = resolution_action
    return result


def build_action(
    ticket: dict, allowed_fields: list[str], instructions: str
) -> tuple[HelpdeskTicketAction, str, str | None]:
    heuristic_dict = heuristic_action(ticket, allowed_fields)

    if llm_client is None:
        return HelpdeskTicketAction(**heuristic_dict), "heuristic", None

    try:
        llm_dict = call_llm(ticket, allowed_fields, instructions)
        candidate = {
            field: llm_dict[field]
            for field in allowed_fields
            if llm_dict.get(field) is not None
        }
        if not candidate:
            raise ValueError("LLM returned no allowed fields")
        return HelpdeskTicketAction(**candidate), "llm", None
    except Exception as exc:
        return (
            HelpdeskTicketAction(**heuristic_dict),
            "heuristic_fallback",
            str(exc),
        )


def should_investigate(ticket: dict, history: list[dict[str, Any]]) -> tuple[bool, str | None]:
    if not ticket:
        return False, None
    current_ticket_id = ticket.get("ticket_id")
    already_investigated = any(
        entry.get("ticket_id") == current_ticket_id
        and entry.get("predicted", {}).get("action_type") == "investigate"
        for entry in history
    )
    if already_investigated:
        return False, None
    if ticket.get("related_ticket_id"):
        return True, "lookup_related_ticket"
    if ticket.get("ambiguity_note"):
        return True, "lookup_requester_history"
    return False, None


def merge_ticket_context(ticket: dict, observation: Any) -> dict:
    merged_ticket = dict(ticket)
    if getattr(observation, "last_tool_result", None) is not None:
        merged_ticket["last_tool_result"] = observation.last_tool_result
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

    all_results: dict[int, dict[str, float | int]] = {}

    tasks_to_run = get_tasks_to_run(available_tasks)
    if not tasks_to_run:
        return
    single_task_mode = len(tasks_to_run) == 1

    for task_id in tasks_to_run:
        if task_id not in available_tasks:
            continue

        task = available_tasks[task_id]
        emit_log(
            "START",
            env_url=ENV_URL,
            mode="llm" if llm_client is not None else "heuristic",
            seed=SEED,
            task_difficulty=task["difficulty"],
            task_id=task_id,
            task_name=task["name"],
        )

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

                investigate, tool_name = should_investigate(ticket, obs.history)
                if (
                    investigate
                    and tool_name is not None
                    and getattr(obs, "investigation_budget_remaining", 0) > 0
                ):
                    tool_action = HelpdeskTicketAction(
                        action_type="investigate",
                        tool_name=tool_name,
                        tool_target_ticket_id=ticket.get("related_ticket_id"),
                    )
                    result = sync_client.step(tool_action)
                    obs = result.observation
                    step_num += 1
                    emit_log(
                        "STEP",
                        action=tool_action.model_dump(exclude_none=True),
                        action_source="investigation_tool",
                        done=bool(result.done),
                        fallback_reason=None,
                        reward=float(result.reward or 0.0),
                        step=step_num,
                        task_id=task_id,
                        ticket_id=ticket["ticket_id"],
                    )
                    if result.done:
                        break
                    ticket = obs.current_ticket
                    if ticket is None:
                        break

                ticket_with_context = merge_ticket_context(ticket, obs)
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

                emit_log(
                    "STEP",
                    action=action.model_dump(exclude_none=True),
                    action_source=action_source,
                    done=bool(result.done),
                    fallback_reason=fallback_reason,
                    reward=reward,
                    step=step_num,
                    task_id=task_id,
                    ticket_id=ticket["ticket_id"],
                )

        final_reward = task_step_rewards[-1] if task_step_rewards else 0.0
        all_results[task_id] = {
            "final_reward": final_reward,
            "step_count": step_num,
        }
        emit_log(
            "END",
            final_reward=round(final_reward, 4),
            step_count=step_num,
            task_id=task_id,
            task_name=task["name"],
        )

    overall = [
        float(all_results[task_id]["final_reward"])
        for task_id in tasks_to_run
        if task_id in all_results
    ]
    if not single_task_mode:
        overall_avg = round(sum(overall) / len(overall), 4) if overall else 0.0
        emit_log("END", overall_avg=overall_avg, tasks_completed=len(overall))


if __name__ == "__main__":
    run()
