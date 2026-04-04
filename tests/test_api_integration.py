"""
API integration tests for the Helpdesk Ticket Routing OpenEnv server.

Uses FastAPI's TestClient (via starlette) to test the live app without
needing a running server.

Run with:
    pytest meta-AIHack/tests/test_api_integration.py -v
"""
from __future__ import annotations

import sys
import os
import types
import unittest
from typing import Any, Optional

# Ensure the repo root (parent of tests/) is on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# -----------------------------------------------------------------------
# Step 1: Install openenv type stubs BEFORE any openenv imports.
# -----------------------------------------------------------------------
import openenv_test_stubs  # noqa: F401

# -----------------------------------------------------------------------
# Step 2: Install the interfaces stub (Environment base class).
# -----------------------------------------------------------------------
if "openenv.core.env_server.interfaces" not in sys.modules:
    _interfaces_mod = types.ModuleType("openenv.core.env_server.interfaces")

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

# -----------------------------------------------------------------------
# Step 3: Install a create_app stub into openenv.core.env_server.
#
# The stub creates a real FastAPI app with the standard OpenEnv routes:
#   GET  /health  → {"status": "ok"}
#   POST /reset   → calls env.reset(seed=..., task_id=...) → observation JSON
#   POST /step    → calls env.step(action) → observation JSON
#   GET  /state   → calls env.state → state JSON
# -----------------------------------------------------------------------
_env_server_mod = sys.modules["openenv.core.env_server"]

if not hasattr(_env_server_mod, "create_app"):
    from fastapi import FastAPI, Request
    from pydantic import BaseModel

    # Define request models at module level so FastAPI/Pydantic can resolve them.
    class _ResetRequest(BaseModel):
        task_id: Optional[int] = 1
        seed: Optional[int] = None

    def _create_app_stub(env_class, action_model, observation_model, env_name: str = ""):
        """
        Stub for openenv.core.env_server.create_app.

        Returns a real FastAPI app with the standard OpenEnv routes wired up.
        The environment instance is shared across all requests within a session.
        """
        _app = FastAPI(title=env_name)
        _env_instance = env_class()

        @_app.get("/health")
        def health():
            return {"status": "ok"}

        @_app.post("/reset")
        def reset(body: _ResetRequest):
            obs = _env_instance.reset(seed=body.seed, task_id=body.task_id)
            return obs.model_dump()

        @_app.post("/step")
        async def step(request: Request):
            payload = await request.json()
            action = action_model.model_validate(payload)
            obs = _env_instance.step(action)
            return obs.model_dump()

        @_app.get("/state")
        def state():
            return _env_instance.state.model_dump()

        return _app

    _env_server_mod.create_app = _create_app_stub

# -----------------------------------------------------------------------
# Now it is safe to import the app (which calls create_app internally).
# -----------------------------------------------------------------------
from starlette.testclient import TestClient
from server.app import app

client = TestClient(app)


# -----------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------

def _reset(task_id: int = 1, seed: int = 42):
    return client.post("/reset", json={"task_id": task_id, "seed": seed})


# -----------------------------------------------------------------------
# Test classes
# -----------------------------------------------------------------------

class TestHealthEndpoint(unittest.TestCase):
    """2.1.1 — GET /health returns HTTP 200 with {"status": "ok"}."""

    def test_health_returns_200(self):
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_returns_ok_body(self):
        resp = client.get("/health")
        self.assertEqual(resp.json(), {"status": "ok"})


class TestTasksEndpoint(unittest.TestCase):
    """2.1.2 — GET /tasks returns HTTP 200 with exactly 3 tasks with IDs 1, 2, 3."""

    def test_tasks_returns_200(self):
        resp = client.get("/tasks")
        self.assertEqual(resp.status_code, 200)

    def test_tasks_returns_exactly_3_tasks(self):
        resp = client.get("/tasks")
        data = resp.json()
        self.assertIn("tasks", data)
        self.assertEqual(len(data["tasks"]), 3)

    def test_tasks_have_ids_1_2_3(self):
        resp = client.get("/tasks")
        ids = {t["id"] for t in resp.json()["tasks"]}
        self.assertEqual(ids, {1, 2, 3})


class TestResetEndpoint(unittest.TestCase):
    """2.1.3 — POST /reset returns a valid observation JSON."""

    def setUp(self):
        self.resp = _reset(task_id=1, seed=42)
        self.data = self.resp.json()

    def test_reset_returns_200(self):
        self.assertEqual(self.resp.status_code, 200)

    def test_reset_done_is_false(self):
        self.assertFalse(self.data["done"])

    def test_reset_reward_is_null(self):
        self.assertIsNone(self.data["reward"])

    def test_reset_task_id_is_1(self):
        self.assertEqual(self.data["task_id"], 1)

    def test_reset_tickets_processed_is_0(self):
        self.assertEqual(self.data["tickets_processed"], 0)

    def test_reset_allowed_fields_non_empty(self):
        self.assertIsInstance(self.data["allowed_fields"], list)
        self.assertGreater(len(self.data["allowed_fields"]), 0)


class TestStepEndpoint(unittest.TestCase):
    """2.1.4 — POST /step returns observation JSON with reward in [0.0, 1.0]."""

    def setUp(self):
        # Reset first so the environment is in a known state.
        _reset(task_id=1, seed=42)
        self.resp = client.post("/step", json={"issue_type": "billing_license"})
        self.data = self.resp.json()

    def test_step_returns_200(self):
        self.assertEqual(self.resp.status_code, 200)

    def test_step_reward_is_float_in_unit_interval(self):
        reward = self.data["reward"]
        self.assertIsNotNone(reward)
        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, 0.0)
        self.assertLessEqual(reward, 1.0)

    def test_step_tickets_processed_is_1(self):
        self.assertEqual(self.data["tickets_processed"], 1)


class TestStateEndpoint(unittest.TestCase):
    """2.1.5 — GET /state returns current episode state JSON after a reset."""

    def setUp(self):
        _reset(task_id=2, seed=7)
        self.resp = client.get("/state")
        self.data = self.resp.json()

    def test_state_returns_200(self):
        self.assertEqual(self.resp.status_code, 200)

    def test_state_current_task_id_is_2(self):
        self.assertEqual(self.data["current_task_id"], 2)

    def test_state_step_count_is_0(self):
        self.assertEqual(self.data["step_count"], 0)

    def test_state_queue_ticket_ids_non_empty(self):
        self.assertIsInstance(self.data["queue_ticket_ids"], list)
        self.assertGreater(len(self.data["queue_ticket_ids"]), 0)


# -----------------------------------------------------------------------
# Task 4.1 — Full seeded episode and mid-episode state tests
# -----------------------------------------------------------------------

class TestFullSeededEpisode(unittest.TestCase):
    """2.1.6 — One end-to-end seeded episode over HTTP completes all steps
    and returns a final trajectory reward in [0.0, 1.0].

    Validates: Requirements 2.1.6
    """

    def test_full_episode_final_reward_in_unit_interval(self):
        """4.1.1 — reset → step loop until done → final trajectory reward in [0.0, 1.0]."""
        # Reset with a fixed seed for determinism.
        reset_resp = _reset(task_id=1, seed=42)
        self.assertEqual(reset_resp.status_code, 200)
        obs = reset_resp.json()
        self.assertFalse(obs["done"])

        # Retrieve allowed_fields from the observation so we can build a valid action.
        allowed_fields = obs["allowed_fields"]
        self.assertGreater(len(allowed_fields), 0)

        final_reward = None
        max_steps = 20  # safety cap — queue is at most 5 tickets
        for _ in range(max_steps):
            # Build a minimal valid action using the first allowed field.
            action_payload: dict = {}
            if "issue_type" in allowed_fields:
                action_payload["issue_type"] = "general_inquiry"
            if "priority" in allowed_fields:
                action_payload["priority"] = "medium"
            if "assignment_group" in allowed_fields:
                action_payload["assignment_group"] = "service_desk"
            if "resolution_action" in allowed_fields:
                action_payload["resolution_action"] = "acknowledge"

            step_resp = client.post("/step", json=action_payload)
            self.assertEqual(step_resp.status_code, 200)
            obs = step_resp.json()

            reward = obs.get("reward")
            self.assertIsNotNone(reward)
            self.assertIsInstance(reward, float)
            self.assertGreaterEqual(reward, 0.0)
            self.assertLessEqual(reward, 1.0)

            if obs["done"]:
                final_reward = reward
                break

        self.assertIsNotNone(final_reward, "Episode did not complete within max_steps")
        self.assertGreaterEqual(final_reward, 0.0)
        self.assertLessEqual(final_reward, 1.0)

    def test_full_episode_all_tasks_complete(self):
        """4.1.1 — Full seeded episode completes for each task ID (1, 2, 3)."""
        for task_id in (1, 2, 3):
            with self.subTest(task_id=task_id):
                reset_resp = _reset(task_id=task_id, seed=42)
                self.assertEqual(reset_resp.status_code, 200)
                obs = reset_resp.json()
                allowed_fields = obs["allowed_fields"]

                action_payload: dict = {}
                if "issue_type" in allowed_fields:
                    action_payload["issue_type"] = "general_inquiry"
                if "priority" in allowed_fields:
                    action_payload["priority"] = "medium"
                if "assignment_group" in allowed_fields:
                    action_payload["assignment_group"] = "service_desk"
                if "resolution_action" in allowed_fields:
                    action_payload["resolution_action"] = "acknowledge"

                completed = False
                for _ in range(20):
                    step_resp = client.post("/step", json=action_payload)
                    self.assertEqual(step_resp.status_code, 200)
                    obs = step_resp.json()
                    if obs["done"]:
                        completed = True
                        break

                self.assertTrue(completed, f"Task {task_id} episode did not complete")


class TestStateMidEpisode(unittest.TestCase):
    """4.1.2 — GET /state reflects correct state mid-episode.

    After reset, step_count is 0. After one step, step_count increments to 1.

    Validates: Requirements 2.1.5
    """

    def test_state_step_count_is_0_after_reset(self):
        """step_count is 0 immediately after reset."""
        _reset(task_id=1, seed=99)
        state_resp = client.get("/state")
        self.assertEqual(state_resp.status_code, 200)
        state = state_resp.json()
        self.assertEqual(state["step_count"], 0)

    def test_state_step_count_increments_after_step(self):
        """step_count increments from 0 to 1 after one step."""
        _reset(task_id=1, seed=99)

        # Confirm step_count is 0 before stepping.
        state_before = client.get("/state").json()
        self.assertEqual(state_before["step_count"], 0)

        # Take one step.
        client.post("/step", json={"issue_type": "general_inquiry"})

        # Confirm step_count is now 1.
        state_after = client.get("/state").json()
        self.assertEqual(state_after["step_count"], 1)

    def test_state_task_id_matches_reset(self):
        """current_task_id in state matches the task_id used in reset."""
        for task_id in (1, 2, 3):
            with self.subTest(task_id=task_id):
                _reset(task_id=task_id, seed=42)
                state = client.get("/state").json()
                self.assertEqual(state["current_task_id"], task_id)


# -----------------------------------------------------------------------
# Task 4.2 — Heuristic inference regression check
# -----------------------------------------------------------------------

class TestHeuristicInferenceRegression(unittest.TestCase):
    """2.2 — Heuristic inference regression: all 3 tasks complete without error
    and overall average reward is in [0.8, 1.0].

    This test drives the inference loop directly against the TestClient app,
    using the same heuristic_action logic as inference.py but routing HTTP
    calls through the in-process TestClient instead of a live server.

    Validates: Requirements 2.2.1, 2.2.2
    """

    # Import heuristic helpers from inference.py at class level so they are
    # available without a live server.
    @classmethod
    def setUpClass(cls):
        import sys
        import os
        import types as _types

        # Ensure the repo root is on sys.path so inference.py is importable.
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # The test stubs only cover openenv.core.env_server.  inference.py
        # imports client.py which needs openenv.core.env_client.  Install a
        # minimal stub so the import succeeds without a live openenv install.
        if "openenv.core.env_client" not in sys.modules:
            _ec_mod = _types.ModuleType("openenv.core.env_client")

            class _StepResult:
                def __init__(self, observation=None, reward=None, done=False):
                    self.observation = observation
                    self.reward = reward
                    self.done = done

            class _EnvClient:
                def __class_getitem__(cls, item):
                    return cls

            _ec_mod.EnvClient = _EnvClient  # type: ignore[attr-defined]
            _ec_mod.StepResult = _StepResult  # type: ignore[attr-defined]
            sys.modules["openenv.core.env_client"] = _ec_mod

        import inference as _inf
        cls._heuristic_action = staticmethod(_inf.heuristic_action)
        cls._SEED = _inf.SEED
        cls._TASKS = list(_inf.TASKS)

    def _run_heuristic_episode(self, task_id: int) -> float:
        """Run one full heuristic episode for the given task_id via TestClient.

        Returns the final trajectory reward.
        """
        reset_resp = client.post("/reset", json={"task_id": task_id, "seed": self._SEED})
        self.assertEqual(reset_resp.status_code, 200, f"reset failed for task {task_id}")
        obs = reset_resp.json()
        self.assertFalse(obs["done"])

        allowed_fields: list = obs["allowed_fields"]
        final_reward = 0.0

        for _ in range(20):  # safety cap
            ticket = obs.get("current_ticket")
            if ticket is None:
                break

            action_dict = self._heuristic_action(ticket, allowed_fields)
            step_resp = client.post("/step", json=action_dict)
            self.assertEqual(step_resp.status_code, 200, f"step failed for task {task_id}")
            obs = step_resp.json()

            reward = obs.get("reward")
            self.assertIsNotNone(reward)
            self.assertIsInstance(reward, float)
            self.assertGreaterEqual(reward, 0.0)
            self.assertLessEqual(reward, 1.0)

            if obs["done"]:
                final_reward = float(reward)
                break

        return final_reward

    def test_all_tasks_complete_without_error(self):
        """4.2.1 — All 3 tasks complete without raising an exception."""
        for task_id in self._TASKS:
            with self.subTest(task_id=task_id):
                # Should not raise.
                reward = self._run_heuristic_episode(task_id)
                self.assertIsInstance(reward, float)

    def test_overall_average_reward_in_expected_range(self):
        """4.2.2 — Overall average reward across all 3 tasks is in [0.8, 1.0],
        consistent with the recorded heuristic baseline of 0.9400.
        """
        rewards = []
        for task_id in self._TASKS:
            reward = self._run_heuristic_episode(task_id)
            rewards.append(reward)

        self.assertEqual(len(rewards), 3, "Expected rewards for all 3 tasks")
        overall_avg = sum(rewards) / len(rewards)
        self.assertGreaterEqual(
            overall_avg,
            0.8,
            f"Overall average reward {overall_avg:.4f} is below 0.8 (baseline: 0.9400)",
        )
        self.assertLessEqual(
            overall_avg,
            1.0,
            f"Overall average reward {overall_avg:.4f} exceeds 1.0",
        )


if __name__ == "__main__":
    unittest.main()
