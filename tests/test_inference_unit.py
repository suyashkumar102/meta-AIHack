from __future__ import annotations

import contextlib
import importlib
import io
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import openenv_test_stubs  # noqa: F401

from models import HelpdeskTicketObservation


def _load_inference_module(env: dict[str, str] | None = None):
    env = env or {}

    client_stub = types.ModuleType("client")

    class PlaceholderEnvClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def sync(self):
            raise NotImplementedError

    client_stub.HelpdeskTicketEnvClient = PlaceholderEnvClient

    with mock.patch.dict(os.environ, env, clear=True):
        with mock.patch.dict(sys.modules, {"client": client_stub}):
            sys.modules.pop("inference", None)
            return importlib.import_module("inference")


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def get(self, path: str) -> FakeResponse:
        if path == "/health":
            return FakeResponse({"status": "ok"})
        if path == "/tasks":
            return FakeResponse(
                {
                    "tasks": [
                        {
                            "id": 1,
                            "name": "Issue Type Classification",
                            "difficulty": "easy",
                            "instructions": "Classify the issue type.",
                            "allowed_fields": ["issue_type"],
                        }
                    ]
                }
            )
        raise AssertionError(f"Unexpected path: {path}")

    def close(self) -> None:
        return None


class FakeSyncClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def reset(self, seed: int, task_id: int):
        observation = HelpdeskTicketObservation(
            task_id=task_id,
            task_name="Issue Type Classification",
            instructions="Classify the issue type.",
            allowed_fields=["issue_type"],
            current_ticket={
                "ticket_id": "ticket-001",
                "title": "Invoice issue",
                "requester": "user@example.com",
                "description": "Customer was charged twice.",
            },
            queue_size=1,
            tickets_remaining=1,
            tickets_processed=0,
            history=[],
            done=False,
            reward=None,
            metadata={},
        )
        return SimpleNamespace(observation=observation, done=False, reward=None)

    def step(self, action):
        observation = HelpdeskTicketObservation(
            task_id=1,
            task_name="Issue Type Classification",
            instructions="Classify the issue type.",
            allowed_fields=["issue_type"],
            current_ticket=None,
            queue_size=1,
            tickets_remaining=0,
            tickets_processed=1,
            history=[],
            done=True,
            reward=1.0,
            metadata={},
        )
        return SimpleNamespace(observation=observation, done=True, reward=1.0)


class FakeEnvClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def sync(self) -> FakeSyncClient:
        return FakeSyncClient()


class InferenceUnitTests(unittest.TestCase):
    def test_hf_token_has_no_default_and_model_name_keeps_allowed_default(self) -> None:
        inference = _load_inference_module()

        self.assertEqual(
            inference.API_BASE_URL,
            "https://router.huggingface.co/v1",
        )
        self.assertEqual(inference.MODEL_NAME, "<your-active-model>")
        self.assertIsNone(inference.HF_TOKEN)
        self.assertFalse(inference.llm_mode_enabled())

    def test_seed_env_override_is_respected(self) -> None:
        inference = _load_inference_module({"SEED": "7"})

        self.assertEqual(inference.SEED, 7)

    def test_invalid_seed_env_falls_back_to_default(self) -> None:
        inference = _load_inference_module({"SEED": "not-an-int"})

        self.assertEqual(inference.SEED, 42)

    def test_run_uses_only_structured_start_step_end_logs(self) -> None:
        inference = _load_inference_module()

        with mock.patch.object(inference.httpx, "Client", FakeHttpClient):
            with mock.patch.object(inference, "HelpdeskTicketEnvClient", FakeEnvClient):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    inference.run()

        lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("[START] "))
        self.assertTrue(any(line.startswith("[STEP] ") for line in lines))
        self.assertTrue(lines[-1].startswith("[END] "))
        self.assertTrue(
            all(
                line.startswith("[START] ")
                or line.startswith("[STEP] ")
                or line.startswith("[END] ")
                for line in lines
            )
        )

    def test_default_task_selection_runs_single_first_task(self) -> None:
        inference = _load_inference_module()

        self.assertEqual(
            inference.get_tasks_to_run({1: {}, 2: {}, 3: {}}),
            [1],
        )

    def test_run_all_tasks_override_keeps_local_batch_mode_available(self) -> None:
        inference = _load_inference_module({"RUN_ALL_TASKS": "1"})

        self.assertEqual(
            inference.get_tasks_to_run({1: {}, 2: {}, 3: {}}),
            [1, 2, 3],
        )

    def test_build_llm_user_message_includes_recent_history_feedback(self) -> None:
        inference = _load_inference_module()

        ticket = {
            "ticket_id": "ticket-xyz",
            "title": "Contractor onboarding blocked by access issue",
            "requester": "pm@contractorco.com",
            "description": "Access permissions are blocking contractor setup.",
            "context_status": {
                "investigation_required": True,
                "revealed_tools": [],
                "remaining_tools": ["lookup_internal_routing_note"],
                "hints": ["An internal routing note may disambiguate the correct workflow."],
            },
            "last_tool_result": {"tool_name": "lookup_requester_history", "found": False},
            "feedback_summary": "Ticket score=0.40; field_scores[issue_type=0.40]; reward=0.40",
            "last_reward_components": {"ticket_score": 0.4, "final_reward": 0.4},
            "investigation_budget_remaining": 2,
            "average_score_so_far": 0.7,
            "progress_fraction": 0.5,
            "recent_history": [
                {
                    "ticket_id": "ticket-prev",
                    "predicted": {"issue_type": "identity_access"},
                    "score": 0.4,
                    "breakdown": {"issue_type": 0.4},
                    "penalty_reason": "extra_fields: ['assignment_group']",
                    "feedback_summary": "Penalty applied: extra_fields: ['assignment_group']; reward=0.00",
                    "reward_components": {"reward_kind": "step_penalty", "final_reward": 0.0},
                }
            ],
            "queue_position": 2,
            "tickets_remaining": 4,
        }

        message = inference.build_llm_user_message(
            ticket,
            ["issue_type"],
            "Read the ticket and select the single best IT issue type.",
        )

        self.assertIn("Recent evaluation feedback", message)
        self.assertIn("score=0.4", message)
        self.assertIn("penalty_reason=extra_fields", message)
        self.assertIn("Latest environment feedback", message)
        self.assertIn("Context status", message)
        self.assertIn("Latest reward components", message)
        self.assertIn("Average score so far: 0.7", message)
        self.assertIn("Episode progress: 0.5", message)
        self.assertIn("Investigation budget remaining: 2", message)
        self.assertIn("Investigation result", message)
        self.assertIn("queue_position=2", message)

    def test_build_action_backfills_missing_fields_from_heuristic(self) -> None:
        inference = _load_inference_module()
        inference.llm_client = object()

        ticket = {
            "ticket_id": "ticket-018",
            "title": "Question about enterprise tier pricing",
            "requester": "finance@urbanstack.io",
            "description": (
                "We're comparing your enterprise plan against two competitors. "
                "Can you send over a detailed pricing breakdown?"
            ),
        }

        with mock.patch.object(
            inference,
            "call_llm",
            return_value={"issue_type": "service_request"},
        ):
            action, action_source, fallback_reason = inference.build_action(
                ticket,
                ["issue_type", "priority", "assignment_group", "resolution_action"],
                "Perform full helpdesk routing.",
            )

        self.assertEqual(action.issue_type, "service_request")
        self.assertEqual(action.priority, "medium")
        self.assertEqual(action.assignment_group, "procurement")
        self.assertEqual(action.resolution_action, "assign")
        self.assertEqual(action_source, "llm_backfilled")
        self.assertIn("heuristic_backfill", fallback_reason or "")

    def test_build_action_ignores_invalid_llm_fields_and_keeps_valid_ones(self) -> None:
        inference = _load_inference_module()
        inference.llm_client = object()

        ticket = {
            "ticket_id": "ticket-018",
            "title": "Question about enterprise tier pricing",
            "requester": "finance@urbanstack.io",
            "description": (
                "We're comparing your enterprise plan against two competitors. "
                "Can you send over a detailed pricing breakdown?"
            ),
        }

        with mock.patch.object(
            inference,
            "call_llm",
            return_value={
                "issue_type": "service_request",
                "priority": "urgent",
            },
        ):
            action, action_source, fallback_reason = inference.build_action(
                ticket,
                ["issue_type", "priority"],
                "Read the ticket, select the best IT issue type, and estimate the priority.",
            )

        self.assertEqual(action.issue_type, "service_request")
        self.assertEqual(action.priority, "medium")
        self.assertEqual(action_source, "llm_backfilled")
        self.assertIn("invalid_llm_fields=['priority']", fallback_reason or "")

    def test_build_action_backfills_dependent_fields_from_llm_issue_type(self) -> None:
        inference = _load_inference_module()
        inference.llm_client = object()

        ticket = {
            "ticket_id": "ticket-002",
            "title": "Can not sign in after 2FA reset",
            "requester": "ops@laneeight.io",
            "description": (
                "I was forced to reset 2FA and now the account stays locked even "
                "with the backup code."
            ),
        }

        with mock.patch.object(
            inference,
            "call_llm",
            return_value={"issue_type": "identity_access"},
        ):
            action, action_source, fallback_reason = inference.build_action(
                ticket,
                ["issue_type", "assignment_group", "resolution_action"],
                "Perform full helpdesk routing.",
            )

        self.assertEqual(action.issue_type, "identity_access")
        self.assertEqual(action.assignment_group, "service_desk")
        self.assertEqual(action.resolution_action, "fulfill")
        self.assertEqual(action_source, "llm_backfilled")
        self.assertIn("heuristic_backfill", fallback_reason or "")

    def test_build_action_normalizes_pricing_request_issue_type(self) -> None:
        inference = _load_inference_module()
        inference.llm_client = object()

        ticket = {
            "ticket_id": "ticket-018",
            "title": "Question about enterprise tier pricing",
            "requester": "finance@urbanstack.io",
            "description": (
                "We're comparing your enterprise plan against two competitors. "
                "Can you send over a detailed pricing breakdown?"
            ),
        }

        with mock.patch.object(
            inference,
            "call_llm",
            return_value={
                "issue_type": "billing_license",
                "priority": "medium",
            },
        ):
            action, action_source, fallback_reason = inference.build_action(
                ticket,
                ["issue_type", "priority", "assignment_group", "resolution_action"],
                "Perform full helpdesk routing.",
            )

        self.assertEqual(action.issue_type, "service_request")
        self.assertEqual(action.assignment_group, "procurement")
        self.assertEqual(action.resolution_action, "assign")
        self.assertEqual(action.priority, "medium")
        self.assertEqual(action_source, "llm_backfilled")
        self.assertIn("domain_overrides", fallback_reason or "")

    def test_build_action_normalizes_onboarding_access_blocker(self) -> None:
        inference = _load_inference_module()
        inference.llm_client = object()

        ticket = {
            "ticket_id": "TKT-NONDEFAULT-003",
            "title": "Contractor onboarding blocked by access issue",
            "requester": "pm@contractorco.com",
            "description": (
                "A new contractor cannot complete onboarding because their account "
                "access is blocked by a permissions error. The onboarding team "
                "cannot resolve access issues; routing to service desk."
            ),
            "ambiguity_note": "Contractor onboarding blocked by access issue, routed to service desk",
        }

        with mock.patch.object(
            inference,
            "call_llm",
            return_value={
                "issue_type": "identity_access",
                "priority": "high",
            },
        ):
            action, action_source, fallback_reason = inference.build_action(
                ticket,
                ["issue_type", "priority", "assignment_group", "resolution_action"],
                "Perform full helpdesk routing.",
            )

        self.assertEqual(action.issue_type, "onboarding")
        self.assertEqual(action.priority, "medium")
        self.assertEqual(action.assignment_group, "service_desk")
        self.assertEqual(action.resolution_action, "fulfill")
        self.assertEqual(action_source, "llm_backfilled")
        self.assertIn("domain_overrides", fallback_reason or "")

    def test_build_action_deescalates_nonurgent_onboarding_priority(self) -> None:
        inference = _load_inference_module()
        inference.llm_client = object()

        ticket = {
            "ticket_id": "ticket-008",
            "title": "Kickoff onboarding session for newly activated account",
            "requester": "admin@brightpath.io",
            "description": (
                "We activated our account this week and need an onboarding call plus "
                "admin setup guidance for six internal users."
            ),
        }

        with mock.patch.object(
            inference,
            "call_llm",
            return_value={
                "issue_type": "onboarding",
                "priority": "high",
            },
        ):
            action, action_source, fallback_reason = inference.build_action(
                ticket,
                ["issue_type", "priority"],
                "Read the ticket, select the best IT issue type, and estimate the priority.",
            )

        self.assertEqual(action.issue_type, "onboarding")
        self.assertEqual(action.priority, "medium")
        self.assertEqual(action_source, "llm_backfilled")
        self.assertIn("domain_overrides", fallback_reason or "")

    def test_merge_ticket_context_carries_feedback_summary_from_observation(self) -> None:
        inference = _load_inference_module()

        observation = SimpleNamespace(
            last_tool_result={"tool_name": "lookup_requester_history", "found": True},
            history=[{"ticket_id": "ticket-prev", "score": 0.4}],
            queue_position=2,
            tickets_remaining=4,
            investigation_budget_remaining=1,
            average_score_so_far=0.55,
            progress_fraction=0.4,
            last_reward_components={"ticket_score": 0.4, "final_reward": 0.4},
            metadata={"last_feedback_summary": "Ticket score=0.40; reward=0.40"},
        )

        merged = inference.merge_ticket_context(
            {
                "ticket_id": "ticket-xyz",
                "title": "Contractor onboarding blocked by access issue",
            },
            observation,
        )

        self.assertEqual(merged["feedback_summary"], "Ticket score=0.40; reward=0.40")
        self.assertEqual(merged["investigation_budget_remaining"], 1)
        self.assertEqual(merged["average_score_so_far"], 0.55)
        self.assertEqual(merged["progress_fraction"], 0.4)
        self.assertEqual(merged["last_reward_components"]["final_reward"], 0.4)
        self.assertEqual(merged["queue_position"], 2)
        self.assertEqual(merged["tickets_remaining"], 4)
        self.assertEqual(merged["last_tool_result"]["tool_name"], "lookup_requester_history")

    def test_should_investigate_uses_remaining_tools_from_context_status(self) -> None:
        inference = _load_inference_module()

        investigate, tool_name = inference.should_investigate(
            {
                "ticket_id": "ticket-021",
                "context_status": {
                    "remaining_tools": [
                        "lookup_related_ticket",
                        "lookup_requester_history",
                    ]
                },
            },
            [],
        )

        self.assertTrue(investigate)
        self.assertEqual(tool_name, "lookup_related_ticket")


if __name__ == "__main__":
    unittest.main()
