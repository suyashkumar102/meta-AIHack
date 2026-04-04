"""
Smoke tests for HelpdeskTicketRoutingEnvironment.

Covers: reset(), step(), state property, seeded determinism,
per-ticket score bounds, and full episode completion for all task IDs.

Run with:
    pytest tests/test_environment_smoke.py
"""
from __future__ import annotations

import sys
import os
import unittest

# Ensure the repo root is on sys.path so imports resolve without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import openenv_test_stubs  # noqa: F401  — must come before any openenv imports

# The shared stub covers openenv.core.env_server.types but not .interfaces.
# Patch in the interfaces module so environment.py can import Environment.
import sys
import types as _types

if "openenv.core.env_server.interfaces" not in sys.modules:
    _interfaces_mod = _types.ModuleType("openenv.core.env_server.interfaces")

    class _Environment:
        """Minimal stub matching the openenv-core Environment base class."""
        def __init__(self) -> None:
            pass

        def __init_subclass__(cls, **kwargs: object) -> None:
            super().__init_subclass__(**kwargs)

        @classmethod
        def __class_getitem__(cls, item: object) -> type:
            return cls

    _interfaces_mod.Environment = _Environment  # type: ignore[attr-defined]
    sys.modules["openenv.core.env_server.interfaces"] = _interfaces_mod

from models import HelpdeskTicketObservation, HelpdeskTicketState
from server.environment import HelpdeskTicketRoutingEnvironment
from server.tasks import TASKS
from vocabulary import ISSUE_TYPES, PRIORITIES, ASSIGNMENT_GROUPS, RESOLUTION_ACTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env() -> HelpdeskTicketRoutingEnvironment:
    return HelpdeskTicketRoutingEnvironment()


def _heuristic_action_dict(obs: HelpdeskTicketObservation) -> dict:
    """Return a minimal valid action dict for the given observation."""
    allowed = obs.allowed_fields
    action: dict = {}
    if "issue_type" in allowed:
        action["issue_type"] = ISSUE_TYPES[0]
    if "priority" in allowed:
        action["priority"] = PRIORITIES[0]
    if "assignment_group" in allowed:
        action["assignment_group"] = ASSIGNMENT_GROUPS[0]
    if "resolution_action" in allowed:
        action["resolution_action"] = RESOLUTION_ACTIONS[0]
    return action


def _run_full_episode(env: HelpdeskTicketRoutingEnvironment, task_id: int, seed: int = 42):
    """Reset and step through an entire episode; return list of (obs, reward) tuples."""
    from models import HelpdeskTicketAction

    obs = env.reset(seed=seed, task_id=task_id)
    results = []
    while not obs.done:
        action = HelpdeskTicketAction(**_heuristic_action_dict(obs))
        obs = env.step(action)
        results.append((obs, obs.reward))
    return results


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestResetReturnsValidObservation(unittest.TestCase):
    """1.1.1 — reset(task_id=1) returns a valid observation."""

    def test_reset_task1_done_false_reward_none(self) -> None:
        env = _make_env()
        obs = env.reset(seed=42, task_id=1)

        self.assertIsInstance(obs, HelpdeskTicketObservation)
        self.assertFalse(obs.done)
        self.assertIsNone(obs.reward)
        self.assertEqual(obs.task_id, 1)
        self.assertIsNotNone(obs.current_ticket)
        self.assertGreater(obs.queue_size, 0)
        self.assertEqual(obs.tickets_processed, 0)


class TestResetAllTaskIds(unittest.TestCase):
    """1.1.2 — reset(task_id=2) and reset(task_id=3) return valid observations."""

    def _assert_valid_reset_obs(self, obs: HelpdeskTicketObservation, task_id: int) -> None:
        self.assertIsInstance(obs, HelpdeskTicketObservation)
        self.assertFalse(obs.done)
        self.assertIsNone(obs.reward)
        self.assertEqual(obs.task_id, task_id)
        self.assertIsNotNone(obs.current_ticket)
        self.assertGreater(obs.queue_size, 0)
        self.assertEqual(obs.tickets_processed, 0)
        # allowed_fields must match the task definition
        self.assertEqual(obs.allowed_fields, TASKS[task_id]["allowed_fields"])

    def test_reset_task2(self) -> None:
        env = _make_env()
        obs = env.reset(seed=42, task_id=2)
        self._assert_valid_reset_obs(obs, 2)

    def test_reset_task3(self) -> None:
        env = _make_env()
        obs = env.reset(seed=42, task_id=3)
        self._assert_valid_reset_obs(obs, 3)


class TestStepAdvancesTicketsProcessed(unittest.TestCase):
    """1.1.3 — step() increments tickets_processed by 1 and reward is in [0.0, 1.0]."""

    def test_step_increments_tickets_processed(self) -> None:
        from models import HelpdeskTicketAction

        env = _make_env()
        obs = env.reset(seed=42, task_id=1)
        self.assertEqual(obs.tickets_processed, 0)

        action = HelpdeskTicketAction(**_heuristic_action_dict(obs))
        obs2 = env.step(action)

        self.assertEqual(obs2.tickets_processed, 1)

    def test_step_reward_in_unit_interval(self) -> None:
        from models import HelpdeskTicketAction

        env = _make_env()
        obs = env.reset(seed=42, task_id=1)
        action = HelpdeskTicketAction(**_heuristic_action_dict(obs))
        obs2 = env.step(action)

        self.assertIsNotNone(obs2.reward)
        self.assertGreaterEqual(obs2.reward, 0.0)
        self.assertLessEqual(obs2.reward, 1.0)


class TestStateProperty(unittest.TestCase):
    """1.1.4 — state property returns HelpdeskTicketState with correct fields."""

    def test_state_after_reset(self) -> None:
        env = _make_env()
        env.reset(seed=42, task_id=2)
        state = env.state

        self.assertIsInstance(state, HelpdeskTicketState)
        self.assertEqual(state.current_task_id, 2)
        self.assertEqual(state.seed, 42)
        self.assertEqual(state.current_ticket_index, 0)
        self.assertEqual(state.step_count, 0)
        self.assertEqual(state.per_ticket_scores, [])
        self.assertGreater(len(state.queue_ticket_ids), 0)

    def test_state_after_step(self) -> None:
        from models import HelpdeskTicketAction

        env = _make_env()
        obs = env.reset(seed=42, task_id=1)
        action = HelpdeskTicketAction(**_heuristic_action_dict(obs))
        env.step(action)
        state = env.state

        self.assertIsInstance(state, HelpdeskTicketState)
        self.assertEqual(state.step_count, 1)
        self.assertEqual(state.current_ticket_index, 1)
        self.assertEqual(len(state.per_ticket_scores), 1)
        self.assertGreaterEqual(state.per_ticket_scores[0], 0.0)
        self.assertLessEqual(state.per_ticket_scores[0], 1.0)

    def test_state_is_deep_copy(self) -> None:
        """Mutating the returned state must not affect the environment's internal state."""
        env = _make_env()
        env.reset(seed=42, task_id=1)
        state = env.state
        state.step_count = 999

        self.assertEqual(env.state.step_count, 0)


class TestSeededDeterminism(unittest.TestCase):
    """1.1.5 — seeded resets with the same seed produce the same queue order."""

    def test_same_seed_same_queue(self) -> None:
        env = _make_env()

        env.reset(seed=42, task_id=1)
        queue_a = list(env.state.queue_ticket_ids)

        env.reset(seed=42, task_id=1)
        queue_b = list(env.state.queue_ticket_ids)

        self.assertEqual(queue_a, queue_b)

    def test_different_seeds_likely_different_queues(self) -> None:
        """Different seeds should (with very high probability) produce different queues."""
        env = _make_env()

        env.reset(seed=0, task_id=1)
        queue_0 = list(env.state.queue_ticket_ids)

        env.reset(seed=99999, task_id=1)
        queue_99999 = list(env.state.queue_ticket_ids)

        # Not guaranteed, but the probability of collision is negligible.
        self.assertNotEqual(queue_0, queue_99999)

    def test_seeded_reset_on_separate_env_instances(self) -> None:
        """Two independent env instances with the same seed must produce the same queue."""
        env1 = _make_env()
        env2 = _make_env()

        env1.reset(seed=7, task_id=3)
        env2.reset(seed=7, task_id=3)

        self.assertEqual(env1.state.queue_ticket_ids, env2.state.queue_ticket_ids)


class TestPerTicketScoreBounds(unittest.TestCase):
    """1.1.6 — all per-ticket scores stay in [0.0, 1.0] across a full episode."""

    def _assert_scores_in_bounds(self, task_id: int) -> None:
        env = _make_env()
        _run_full_episode(env, task_id=task_id, seed=42)
        state = env.state
        for score in state.per_ticket_scores:
            self.assertGreaterEqual(score, 0.0, f"task {task_id}: score {score} < 0")
            self.assertLessEqual(score, 1.0, f"task {task_id}: score {score} > 1")

    def test_scores_in_bounds_task1(self) -> None:
        self._assert_scores_in_bounds(1)

    def test_scores_in_bounds_task2(self) -> None:
        self._assert_scores_in_bounds(2)

    def test_scores_in_bounds_task3(self) -> None:
        self._assert_scores_in_bounds(3)


class TestFullEpisodeCompletion(unittest.TestCase):
    """1.1.7 — one full episode per task completes without unhandled exceptions."""

    def _run_and_assert_episode(self, task_id: int) -> None:
        env = _make_env()
        results = _run_full_episode(env, task_id=task_id, seed=42)

        # At least one step was taken
        self.assertGreater(len(results), 0)

        # Final observation must be done
        final_obs, final_reward = results[-1]
        self.assertTrue(final_obs.done)

        # Final reward must be in [0.0, 1.0]
        self.assertIsNotNone(final_reward)
        self.assertGreaterEqual(final_reward, 0.0)
        self.assertLessEqual(final_reward, 1.0)

        # tickets_processed must equal queue_size at end
        self.assertEqual(final_obs.tickets_processed, final_obs.queue_size)

    def test_full_episode_task1(self) -> None:
        self._run_and_assert_episode(1)

    def test_full_episode_task2(self) -> None:
        self._run_and_assert_episode(2)

    def test_full_episode_task3(self) -> None:
        self._run_and_assert_episode(3)


if __name__ == "__main__":
    unittest.main()
