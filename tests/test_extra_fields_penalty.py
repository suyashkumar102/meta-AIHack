"""
Tests for action field validation (Task 4) in HelpdeskTicketRoutingEnvironment.step().

Validates Requirement 7: Step Validates Action Fields Against Task Contract.
"""
from __future__ import annotations

import sys
import os
import unittest
import types as _types

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
from server.environment import HelpdeskTicketRoutingEnvironment
from server.tasks import TASKS
from vocabulary import ISSUE_TYPES, PRIORITIES, ASSIGNMENT_GROUPS, RESOLUTION_ACTIONS


def _make_env() -> HelpdeskTicketRoutingEnvironment:
    return HelpdeskTicketRoutingEnvironment()


class TestExtraFieldsPenalty(unittest.TestCase):
    """Requirement 7: step() rejects actions with fields outside the task's allowed_fields."""

    def test_extra_fields_returns_reward_zero(self) -> None:
        """Task 1 only allows issue_type and priority; submitting assignment_group triggers penalty."""
        env = _make_env()
        obs = env.reset(seed=42, task_id=1)

        # Task 1 allowed_fields should NOT include assignment_group
        self.assertNotIn("assignment_group", obs.allowed_fields)

        # Submit an action with an extra field (assignment_group) not in task 1's allowed_fields
        action = HelpdeskTicketAction(
            issue_type=ISSUE_TYPES[0],
            priority=PRIORITIES[0],
            assignment_group=ASSIGNMENT_GROUPS[0],  # extra field
        )
        penalty_obs = env.step(action)

        self.assertIsInstance(penalty_obs, HelpdeskTicketObservation)
        self.assertEqual(penalty_obs.reward, 0.0)

    def test_extra_fields_advances_ticket_index(self) -> None:
        """Penalty step must advance tickets_processed by 1."""
        env = _make_env()
        obs = env.reset(seed=42, task_id=1)
        self.assertEqual(obs.tickets_processed, 0)

        action = HelpdeskTicketAction(
            issue_type=ISSUE_TYPES[0],
            assignment_group=ASSIGNMENT_GROUPS[0],  # extra field for task 1
        )
        penalty_obs = env.step(action)

        self.assertEqual(penalty_obs.tickets_processed, 1)

    def test_extra_fields_records_score_zero(self) -> None:
        """per_ticket_scores must contain 0.0 after a penalty step."""
        env = _make_env()
        env.reset(seed=42, task_id=1)

        action = HelpdeskTicketAction(
            issue_type=ISSUE_TYPES[0],
            assignment_group=ASSIGNMENT_GROUPS[0],  # extra field
        )
        env.step(action)

        state = env.state
        self.assertEqual(len(state.per_ticket_scores), 1)
        self.assertEqual(state.per_ticket_scores[0], 0.0)

    def test_extra_fields_history_entry_has_penalty_reason(self) -> None:
        """History entry for a penalty step must include penalty_reason."""
        env = _make_env()
        env.reset(seed=42, task_id=1)

        action = HelpdeskTicketAction(
            issue_type=ISSUE_TYPES[0],
            assignment_group=ASSIGNMENT_GROUPS[0],  # extra field
        )
        penalty_obs = env.step(action)

        self.assertEqual(len(penalty_obs.history), 1)
        entry = penalty_obs.history[0]
        self.assertIn("penalty_reason", entry)
        self.assertIn("assignment_group", entry["penalty_reason"])
        self.assertEqual(entry["score"], 0.0)

    def test_no_extra_fields_grades_normally(self) -> None:
        """When action fields are within allowed_fields, grading proceeds normally (reward != forced 0.0)."""
        env = _make_env()
        obs = env.reset(seed=42, task_id=1)

        # Build action using only allowed fields
        allowed = obs.allowed_fields
        action_kwargs = {}
        if "issue_type" in allowed:
            action_kwargs["issue_type"] = ISSUE_TYPES[0]
        if "priority" in allowed:
            action_kwargs["priority"] = PRIORITIES[0]

        action = HelpdeskTicketAction(**action_kwargs)
        result_obs = env.step(action)

        # Should be a valid observation; reward may be any value in [0.0, 1.0]
        self.assertIsInstance(result_obs, HelpdeskTicketObservation)
        self.assertIsNotNone(result_obs.reward)
        # No penalty_reason in history
        self.assertEqual(len(result_obs.history), 1)
        self.assertNotIn("penalty_reason", result_obs.history[0])

    def test_extra_fields_no_exception_raised(self) -> None:
        """Requirement 7.4: extra fields must not raise an unhandled exception."""
        env = _make_env()
        env.reset(seed=42, task_id=1)

        action = HelpdeskTicketAction(
            issue_type=ISSUE_TYPES[0],
            priority=PRIORITIES[0],
            assignment_group=ASSIGNMENT_GROUPS[0],
            resolution_action=RESOLUTION_ACTIONS[0],  # multiple extra fields
        )
        try:
            obs = env.step(action)
        except Exception as exc:  # noqa: BLE001
            self.fail(f"step() raised an unexpected exception: {exc}")

        self.assertIsInstance(obs, HelpdeskTicketObservation)

    def test_extra_fields_done_flag_set_correctly_on_last_ticket(self) -> None:
        """When the penalty step is on the last ticket, done stays True and reward stays episode-level."""
        env = _make_env()
        obs = env.reset(seed=42, task_id=1)
        queue_size = obs.queue_size
        tickets_by_id = env._tickets_by_id  # noqa: SLF001 - test-only inspection

        # Process all tickets except the last one normally
        for _ in range(queue_size - 1):
            current_ticket_id = obs.current_ticket["ticket_id"]
            current_ticket = tickets_by_id[current_ticket_id]
            obs = env.step(HelpdeskTicketAction(issue_type=current_ticket.issue_type))

        # Now trigger penalty on the last ticket
        current_ticket_id = obs.current_ticket["ticket_id"]
        current_ticket = tickets_by_id[current_ticket_id]
        action = HelpdeskTicketAction(
            issue_type=current_ticket.issue_type,
            assignment_group=ASSIGNMENT_GROUPS[0],  # extra field
        )
        final_obs = env.step(action)

        self.assertTrue(final_obs.done)
        expected_reward = (queue_size - 1) / queue_size
        self.assertAlmostEqual(final_obs.reward, expected_reward, places=9)
        self.assertAlmostEqual(env.state.total_reward, expected_reward, places=9)


if __name__ == "__main__":
    unittest.main()
