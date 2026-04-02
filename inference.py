#!/usr/bin/env python3
"""
Inference script for the IT Helpdesk Ticket Routing OpenEnv environment.

Environment variables
---------------------
ENV_URL
    Base URL of the running OpenEnv server.
    Default: ``http://localhost:8000``
    Optional — when unset the script connects to the local server on port 8000.

API_BASE_URL
    LLM provider base URL (OpenAI-compatible endpoint).
    Default: ``https://router.huggingface.co/v1``
    Optional — only used when both MODEL_NAME and HF_TOKEN are set.

MODEL_NAME
    Model identifier to use for LLM inference (e.g. ``meta-llama/Llama-3.3-70B-Instruct``).
    Default: ``""`` (empty string)
    Optional — when unset (or empty) the script runs in heuristic mode without an LLM.

HF_TOKEN
    HuggingFace authentication token for the LLM provider.
    Default: ``""`` (empty string)
    Optional — when unset (or empty) the script runs in heuristic mode without an LLM.

When both MODEL_NAME and HF_TOKEN are set, the script calls the LLM via the OpenAI-compatible
API at API_BASE_URL. When either is unset, ``llm_client`` is ``None`` and ``build_action()``
falls back to ``heuristic_action()`` automatically.

Uses the HTTP-based sync EnvClient for multi-step episodes.
"""
from __future__ import annotations

import json
import os

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
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

SEED = 42
TASKS = list(TASK_IDS)

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

llm_client: OpenAI | None = None

if MODEL_NAME and HF_TOKEN:
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

    user_msg = (
        f"Instructions: {instructions}\n\n"
        f"Allowed fields: {', '.join(allowed_fields)}\n\n"
        f"Title: {ticket['title']}\n"
        f"Requester: {ticket['requester']}\n"
        f"Description: {ticket['description']}\n\n"
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
    text = (ticket.get("title", "") + " " + ticket.get("description", "")).lower()

    issue_type = "general_inquiry"
    for kw, mapped_issue_type in KEYWORD_ISSUE_TYPES.items():
        if kw in text:
            issue_type = mapped_issue_type
            break

    priority = heuristic_priority(text)
    resolution_action = heuristic_resolution_action(text, issue_type)

    result: dict = {}
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
) -> HelpdeskTicketAction:
    heuristic_dict = heuristic_action(ticket, allowed_fields)

    if llm_client is None:
        return HelpdeskTicketAction(**heuristic_dict)

    llm_dict = call_llm(ticket, allowed_fields, instructions)
    try:
        return HelpdeskTicketAction(**llm_dict)
    except Exception as exc:
        print(f"  Falling back to heuristic action due to invalid LLM output: {exc}")
        return HelpdeskTicketAction(**heuristic_dict)


# ---------------------------------------------------------------------------
# Main loop using WebSocket client for multi-step episodes
# ---------------------------------------------------------------------------

def run():
    # Quick HTTP health check
    http = httpx.Client(base_url=ENV_URL, timeout=30.0)
    health = http.get("/health")
    health.raise_for_status()
    print(f"Connected to {ENV_URL}: {health.json()}")

    tasks_resp = http.get("/tasks")
    tasks_resp.raise_for_status()
    available_tasks = {t["id"]: t for t in tasks_resp.json()["tasks"]}
    print(f"Available tasks: {[t['name'] for t in available_tasks.values()]}")
    http.close()

    all_results: dict[int, dict[str, float | int]] = {}

    for task_id in TASKS:
        if task_id not in available_tasks:
            print(f"Task {task_id} not available, skipping")
            continue

        task = available_tasks[task_id]
        print(f"\n--- Task {task_id}: {task['name']} ({task['difficulty']}) ---")

        # Use sync HTTP client for multi-step episode
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

                allowed = obs.allowed_fields
                instructions = obs.instructions

                action = build_action(ticket, allowed, instructions)
                result = sync_client.step(action)
                obs = result.observation

                step_num += 1
                print(f"  Step {step_num}: reward={result.reward} done={result.done}")

                if result.reward is not None:
                    task_step_rewards.append(float(result.reward))

        final_reward = task_step_rewards[-1] if task_step_rewards else 0.0
        all_results[task_id] = {
            "final_reward": final_reward,
            "step_count": step_num,
        }
        print(f"  Task {task_id} final reward: {final_reward:.4f}")

    # Summary
    print("\n=== RESULTS ===")
    overall: list[float] = []
    for tid in TASKS:
        if tid in all_results:
            final_reward = float(all_results[tid]["final_reward"])
            step_count = int(all_results[tid]["step_count"])
            overall.append(final_reward)
            print(f"Task {tid}: final_reward={final_reward:.4f} ({step_count} steps)")
    if overall:
        print(f"Overall: {sum(overall) / len(overall):.4f}")


if __name__ == "__main__":
    run()
