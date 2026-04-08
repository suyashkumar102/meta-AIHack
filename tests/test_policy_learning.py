from __future__ import annotations

import os
import sys
import types as _types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import openenv_test_stubs  # noqa: F401

if "openenv.core.env_server.interfaces" not in sys.modules:
    _interfaces_mod = _types.ModuleType("openenv.core.env_server.interfaces")

    class _Environment:
        def __init__(self) -> None:
            pass

        def __init_subclass__(cls, **kwargs: object) -> None:
            super().__init_subclass__(**kwargs)

        @classmethod
        def __class_getitem__(cls, item: object) -> type:
            return cls

    _interfaces_mod.Environment = _Environment  # type: ignore[attr-defined]
    sys.modules["openenv.core.env_server.interfaces"] = _interfaces_mod


from models import HelpdeskTicketAction, HelpdeskTicketObservation
from policy_learning import (
    POLICY_LIBRARY,
    choose_policy_action,
    compare_policies,
    infer_ticket_cue,
    parse_int_spec,
    rollout_episode,
    search_policies,
)
from server.environment import HelpdeskTicketRoutingEnvironment
from server.tasks import get_task_definition


class SingleTicketEnvironment(HelpdeskTicketRoutingEnvironment):
    def __init__(self, ticket_id: str) -> None:
        super().__init__()
        self._forced_ticket_id = ticket_id

    def reset(self, seed=None, episode_id=None, **kwargs):
        observation = super().reset(seed=seed, episode_id=episode_id, **kwargs)
        ticket = self._tickets_by_id[self._forced_ticket_id]
        self._queue = [ticket]
        self._state.current_task_id = int(kwargs.get("task_id", 3))
        self._state.queue_ticket_ids = [ticket.ticket_id]
        self._state.current_ticket_index = 0
        self._state.per_ticket_scores = []
        self._state.total_reward = 0.0
        self._state.last_step_reward = None
        self._state.reward = None
        self._state.done = False
        self._state.average_score_so_far = 0.0
        self._state.investigation_steps = 0
        self._state.investigation_budget_remaining = len(self._queue)
        self._state.investigation_penalty_applied = 0.0
        self._state.last_tool_result = None
        self._state.last_reward_components = {}
        self._state.ticket_tool_usage = {}
        self._state.history_entries = []
        return self._build_observation(get_task_definition(self._state.current_task_id))


def _context_sensitive_submit_builder(
    ticket: dict[str, object], allowed_fields: list[str]
) -> HelpdeskTicketAction:
    if ticket.get("ambiguity_note"):
        values = {
            "issue_type": "onboarding",
            "priority": "medium",
            "assignment_group": "service_desk",
            "resolution_action": "fulfill",
        }
    else:
        values = {
            "issue_type": "identity_access",
            "priority": "high",
            "assignment_group": "service_desk",
            "resolution_action": "fulfill",
        }
    return HelpdeskTicketAction(
        **{field: value for field, value in values.items() if field in allowed_fields}
    )


class PolicyLearningTests(unittest.TestCase):
    def test_parse_int_spec_expands_ranges(self) -> None:
        self.assertEqual(parse_int_spec("42-44,44,46", field_name="seeds"), [42, 43, 44, 46])

    def test_choose_policy_action_prefers_hidden_context_tools(self) -> None:
        policy = POLICY_LIBRARY["investigate_when_context_hidden"]
        observation = HelpdeskTicketObservation(
            current_ticket={
                "ticket_id": "ticket-021",
                "title": "Re: Production checkout throwing null reference exception",
                "description": "Additional routing context is available via investigation.",
                "context_status": {
                    "hidden_context_remaining": True,
                    "context_gap_count": 2,
                    "revealed_context_count": 0,
                    "context_completeness": 0.0,
                }
            },
            allowed_fields=["issue_type"],
        )

        action, source, cue = choose_policy_action(
            policy,
            observation,
            {},
            _context_sensitive_submit_builder,
            used_tools_by_ticket={},
        )

        self.assertEqual(action.action_type, "investigate")
        self.assertEqual(action.tool_name, "lookup_related_ticket")
        self.assertEqual(source, "investigate_hidden_context")
        self.assertEqual(cue, "follow_up")

    def test_choose_policy_action_submits_when_investigation_disabled(self) -> None:
        policy = POLICY_LIBRARY["no_investigation"]
        observation = HelpdeskTicketObservation(
            current_ticket={
                "ticket_id": "ticket-021",
                "title": "Re: Production checkout throwing null reference exception",
                "description": "Additional routing context is available via investigation.",
                "context_status": {"hidden_context_remaining": True, "context_gap_count": 1},
            },
            allowed_fields=["issue_type", "priority"],
        )

        action, source, cue = choose_policy_action(
            policy,
            observation,
            {},
            _context_sensitive_submit_builder,
            used_tools_by_ticket={},
        )

        self.assertEqual(action.action_type, "submit")
        self.assertEqual(action.issue_type, "identity_access")
        self.assertEqual(source, "submit")
        self.assertIsNone(cue)

    def test_rollout_episode_rewards_context_aware_policy(self) -> None:
        no_investigation = POLICY_LIBRARY["no_investigation"]
        context_aware = POLICY_LIBRARY["investigate_when_context_hidden"]

        no_summary, _ = rollout_episode(
            env=SingleTicketEnvironment("TKT-NONDEFAULT-003"),
            policy=no_investigation,
            seed=42,
            task_id=3,
            submit_builder=_context_sensitive_submit_builder,
        )
        context_summary, _ = rollout_episode(
            env=SingleTicketEnvironment("TKT-NONDEFAULT-003"),
            policy=context_aware,
            seed=42,
            task_id=3,
            submit_builder=_context_sensitive_submit_builder,
        )

        self.assertLess(no_summary["terminal_reward"], context_summary["terminal_reward"])
        self.assertLess(no_summary["normalized_return"], context_summary["normalized_return"])
        self.assertGreaterEqual(context_summary["investigation_steps"], 1)

    def test_search_policies_selects_adaptive_policy(self) -> None:
        report = search_policies(
            [
                POLICY_LIBRARY["no_investigation"],
                POLICY_LIBRARY["adaptive_cue_bandit"],
            ],
            train_seeds=[41, 42],
            eval_seeds=[43],
            task_ids=[3],
            output_dir=os.path.join(os.getcwd(), "analysis", "policy_learning_test"),
            env_factory=lambda: SingleTicketEnvironment("TKT-NONDEFAULT-003"),
            submit_builder=_context_sensitive_submit_builder,
        )

        self.assertEqual(report["selected_policy"], "adaptive_cue_bandit")
        self.assertGreater(
            report["eval_improvement_vs_baseline"]["avg_normalized_return"],
            0.0,
        )
        self.assertIn("adaptive_cue_bandit", report["trained_adaptive_bandits"])

    def test_compare_policies_reports_improvement(self) -> None:
        report = compare_policies(
            [
                POLICY_LIBRARY["no_investigation"],
                POLICY_LIBRARY["adaptive_cue_bandit"],
            ],
            seeds=[42],
            task_ids=[3],
            output_dir=os.path.join(os.getcwd(), "analysis", "policy_learning_compare_test"),
            env_factory=lambda: SingleTicketEnvironment("TKT-NONDEFAULT-003"),
            submit_builder=_context_sensitive_submit_builder,
        )

        self.assertEqual(report["best_policy"], "adaptive_cue_bandit")
        self.assertGreater(report["improvement_vs_baseline"]["avg_terminal_reward"], 0.0)
        self.assertIn("avg_queue_management_score", report["improvement_vs_baseline"])
        self.assertIn("avg_queue_management_score", report["policy_summaries"][0])

    def test_infer_ticket_cue_distinguishes_workflow_blocker(self) -> None:
        cue = infer_ticket_cue(
            {
                "title": "Contractor onboarding blocked by access issue",
                "description": "A contractor onboarding workflow is blocked by a permissions error.",
            }
        )
        self.assertEqual(cue, "workflow_blocker")


if __name__ == "__main__":
    unittest.main()
