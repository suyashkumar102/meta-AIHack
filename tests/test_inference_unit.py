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


if __name__ == "__main__":
    unittest.main()
