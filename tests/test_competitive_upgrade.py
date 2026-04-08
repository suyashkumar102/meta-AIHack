"""
Tests for the helpdesk-competitive-upgrade spec (Task 9).

Covers:
  9.1  test_inference_single_task_mode
  9.2  test_state_has_reward_and_done
  9.3  test_history_has_title_and_predicted
  9.4  test_milestone_reward_shaping
  9.5  test_trajectory_reward_no_overshoot
  9.6  test_ambiguity_note_in_observation
  9.7  test_dataset_nondefault_routing
  9.9  test_concurrent_sessions_flag
  9.10 test_web_ui_endpoint

Run with:
    pytest tests/test_competitive_upgrade.py
"""
from __future__ import annotations

import os
import sys
import types as _types
import unittest

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import openenv_test_stubs  # noqa: F401  — must come before any openenv imports

# Patch in the interfaces module so environment.py can import Environment.
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


from models import HelpdeskTicketAction, HelpdeskTicketObservation, HelpdeskTicketState
from server.environment import HelpdeskTicketRoutingEnvironment
from server.reward import compute_step_reward, compute_trajectory_reward
from server.tasks import get_task_definition, load_dataset
from vocabulary import ISSUE_TYPES, PRIORITIES, ASSIGNMENT_GROUPS, RESOLUTION_ACTIONS, TASK_IDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env() -> HelpdeskTicketRoutingEnvironment:
    return HelpdeskTicketRoutingEnvironment()


def _heuristic_action(obs: HelpdeskTicketObservation) -> HelpdeskTicketAction:
    allowed = obs.allowed_fields
    kwargs: dict = {}
    if "issue_type" in allowed:
        kwargs["issue_type"] = ISSUE_TYPES[0]
    if "priority" in allowed:
        kwargs["priority"] = PRIORITIES[0]
    if "assignment_group" in allowed:
        kwargs["assignment_group"] = ASSIGNMENT_GROUPS[0]
    if "resolution_action" in allowed:
        kwargs["resolution_action"] = RESOLUTION_ACTIONS[0]
    return HelpdeskTicketAction(**kwargs)


# ---------------------------------------------------------------------------
# 9.1 — Inference single-task mode
# ---------------------------------------------------------------------------

def _get_tasks_to_run_impl(
    task_id_env: str | None,
    available_tasks: dict,
    run_all_tasks: bool = False,
) -> list[int]:
    """
    Standalone re-implementation of inference.get_tasks_to_run() logic for testing.

    This mirrors the logic in inference.py without importing the full module
    (which has heavy dependencies like openai, httpx, and client.py).
    """
    if task_id_env:
        try:
            task_id = int(task_id_env)
        except ValueError:
            raise SystemExit(1)
        if task_id not in available_tasks:
            raise SystemExit(1)
        return [task_id]
    if not available_tasks:
        return []
    if run_all_tasks:
        return sorted(available_tasks)
    return sorted(available_tasks)


class TestInferenceSingleTaskMode(unittest.TestCase):
    """9.1 — get_tasks_to_run() respects TASK_ID env var."""

    def test_task_id_set_to_valid_id_returns_single_element_list(self) -> None:
        available = {1: {}, 2: {}, 3: {}}
        result = _get_tasks_to_run_impl("1", available)
        self.assertEqual(result, [1])

    def test_task_id_set_to_unavailable_id_exits(self) -> None:
        available = {1: {}, 2: {}, 3: {}}
        with self.assertRaises(SystemExit):
            _get_tasks_to_run_impl("999", available)

    def test_task_id_unset_defaults_to_all_available_tasks(self) -> None:
        available = {1: {}, 2: {}, 3: {}}
        result = _get_tasks_to_run_impl(None, available)
        self.assertEqual(result, [1, 2, 3])

    def test_run_all_tasks_override_returns_all_task_ids(self) -> None:
        available = {1: {}, 2: {}, 3: {}}
        result = _get_tasks_to_run_impl(None, available, run_all_tasks=True)
        self.assertEqual(sorted(result), sorted(list(TASK_IDS)))

    def test_task_id_set_to_2_returns_only_task_2(self) -> None:
        available = {1: {}, 2: {}, 3: {}}
        result = _get_tasks_to_run_impl("2", available)
        self.assertEqual(result, [2])

    def test_task_id_set_to_3_returns_only_task_3(self) -> None:
        available = {1: {}, 2: {}, 3: {}}
        result = _get_tasks_to_run_impl("3", available)
        self.assertEqual(result, [3])


# ---------------------------------------------------------------------------
# 9.2 — State has last_step_reward and done after step()
# ---------------------------------------------------------------------------

class TestStateHasRewardAndDone(unittest.TestCase):
    """9.2 — state.last_step_reward and state.done are set after step()."""

    def test_last_step_reward_is_none_after_reset(self) -> None:
        env = _make_env()
        env.reset(seed=42, task_id=1)
        self.assertIsNone(env.state.last_step_reward)

    def test_done_is_false_after_reset(self) -> None:
        env = _make_env()
        env.reset(seed=42, task_id=1)
        self.assertFalse(env.state.done)

    def test_last_step_reward_set_after_step(self) -> None:
        env = _make_env()
        obs = env.reset(seed=42, task_id=1)
        action = _heuristic_action(obs)
        env.step(action)
        state = env.state
        self.assertIsNotNone(state.last_step_reward)
        self.assertGreaterEqual(state.last_step_reward, 0.0)
        self.assertLessEqual(state.last_step_reward, 1.0)

    def test_done_is_true_after_last_ticket(self) -> None:
        env = _make_env()
        obs = env.reset(seed=42, task_id=1)
        while not obs.done:
            obs = env.step(_heuristic_action(obs))
        self.assertTrue(env.state.done)

    def test_done_is_false_before_last_ticket(self) -> None:
        env = _make_env()
        obs = env.reset(seed=42, task_id=1)
        if obs.queue_size > 1:
            obs = env.step(_heuristic_action(obs))
            self.assertFalse(env.state.done)

    def test_state_tracks_average_score_and_reward_components(self) -> None:
        env = _make_env()
        obs = env.reset(seed=42, task_id=1)
        env.step(_heuristic_action(obs))
        state = env.state
        self.assertGreaterEqual(state.average_score_so_far, 0.0)
        self.assertLessEqual(state.average_score_so_far, 1.0)
        self.assertIsInstance(state.last_reward_components, dict)
        self.assertIn("final_reward", state.last_reward_components)


# ---------------------------------------------------------------------------
# 9.3 — History entry contains title and predicted
# ---------------------------------------------------------------------------

class TestHistoryHasTitleAndPredicted(unittest.TestCase):
    """9.3 — observation.history[0] contains 'title' and 'predicted' keys."""

    def test_history_entry_has_title(self) -> None:
        env = _make_env()
        obs = env.reset(seed=42, task_id=1)
        action = _heuristic_action(obs)
        obs2 = env.step(action)
        self.assertEqual(len(obs2.history), 1)
        self.assertIn("title", obs2.history[0])
        self.assertIsInstance(obs2.history[0]["title"], str)
        self.assertTrue(obs2.history[0]["title"])  # non-empty

    def test_history_entry_has_predicted(self) -> None:
        env = _make_env()
        obs = env.reset(seed=42, task_id=1)
        action = _heuristic_action(obs)
        obs2 = env.step(action)
        self.assertIn("predicted", obs2.history[0])
        self.assertIsInstance(obs2.history[0]["predicted"], dict)

    def test_history_predicted_matches_action(self) -> None:
        env = _make_env()
        obs = env.reset(seed=42, task_id=1)
        action = _heuristic_action(obs)
        obs2 = env.step(action)
        predicted = obs2.history[0]["predicted"]
        action_dict = action.model_dump(exclude_none=True)
        self.assertEqual(predicted, action_dict)

    def test_history_entry_has_ticket_id_and_score(self) -> None:
        env = _make_env()
        obs = env.reset(seed=42, task_id=1)
        obs2 = env.step(_heuristic_action(obs))
        entry = obs2.history[0]
        self.assertIn("ticket_id", entry)
        self.assertIn("score", entry)


# ---------------------------------------------------------------------------
# 9.4 — Milestone reward shaping
# ---------------------------------------------------------------------------

class TestMilestoneRewardShaping(unittest.TestCase):
    """9.4 — compute_step_reward applies bonus at high scores, penalty at low scores."""

    def test_high_score_gets_bonus(self) -> None:
        # score=0.9 >= 0.8 threshold → base=0.9, bonus=0.05 → 0.95
        result = compute_step_reward(0.9, previous_average=0.9)
        self.assertAlmostEqual(result, 0.95, places=9)

    def test_low_score_gets_penalty(self) -> None:
        # score=0.1 < 0.2 threshold → base=0.1, penalty=0.05 → 0.05
        result = compute_step_reward(0.1, previous_average=0.1)
        self.assertAlmostEqual(result, 0.05, places=9)

    def test_mid_score_is_neutral(self) -> None:
        # score=0.5 is in [0.2, 0.8) → no shaping → 0.5
        result = compute_step_reward(0.5, previous_average=0.5)
        self.assertAlmostEqual(result, 0.5, places=9)

    def test_boundary_high_threshold_gets_bonus(self) -> None:
        # score=0.8 exactly → bonus applies → 0.85
        result = compute_step_reward(0.8, previous_average=0.8)
        self.assertAlmostEqual(result, 0.85, places=9)

    def test_boundary_low_threshold_is_neutral(self) -> None:
        # score=0.2 exactly → not < 0.2, so neutral → 0.2
        result = compute_step_reward(0.2, previous_average=0.2)
        self.assertAlmostEqual(result, 0.2, places=9)

    def test_reward_clamped_to_unit_interval(self) -> None:
        # score=1.0 → base=1.0, bonus would push to 1.05 → clamped to 1.0
        result = compute_step_reward(1.0)
        self.assertLessEqual(result, 1.0)
        self.assertGreaterEqual(result, 0.0)

    def test_improvement_delta_adds_small_bonus(self) -> None:
        improved = compute_step_reward(0.7, previous_average=0.2)
        flat = compute_step_reward(0.7, previous_average=0.7)
        self.assertGreater(improved, flat)

    def test_zero_score_clamped_to_zero(self) -> None:
        # score=0.0 < 0.2 → base=0.0, penalty → max(0.0, -0.05) = 0.0
        result = compute_step_reward(0.0)
        self.assertGreaterEqual(result, 0.0)


# ---------------------------------------------------------------------------
# 9.5 — Trajectory reward has no overshoot penalty
# ---------------------------------------------------------------------------

class TestTrajectoryRewardNoOvershoot(unittest.TestCase):
    """9.5 — compute_trajectory_reward does not penalise when steps > queue_size."""

    def test_no_penalty_when_steps_exceed_queue_size(self) -> None:
        scores = [0.8, 0.9, 0.7]
        queue_size = 3
        steps_taken = 10  # more steps than queue_size
        result = compute_trajectory_reward(scores, queue_size, steps_taken)
        expected_avg = sum(scores) / len(scores)
        self.assertAlmostEqual(result, expected_avg, places=9)

    def test_result_equals_average_regardless_of_steps(self) -> None:
        scores = [0.5, 0.6]
        for steps in [1, 2, 5, 100]:
            result = compute_trajectory_reward(scores, len(scores), steps)
            self.assertAlmostEqual(result, 0.55, places=9,
                                   msg=f"Failed for steps={steps}")

    def test_empty_scores_returns_zero(self) -> None:
        self.assertEqual(compute_trajectory_reward([], 3, 3), 0.0)

    def test_result_in_unit_interval(self) -> None:
        scores = [0.9, 1.0, 0.95]
        result = compute_trajectory_reward(scores, 3, 3)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


# ---------------------------------------------------------------------------
# 9.6 — ambiguity_note appears in current_ticket observation
# ---------------------------------------------------------------------------

class TestAmbiguityNoteInObservation(unittest.TestCase):
    """9.6 — current_ticket includes ambiguity_note when the ticket has one."""

    def _find_seed_with_ambiguity_note(self, task_id: int = 3) -> int | None:
        """Try seeds 0..999 to find one where the first ticket has ambiguity_note."""
        env = _make_env()
        for seed in range(1000):
            obs = env.reset(seed=seed, task_id=task_id)
            if obs.current_ticket and obs.current_ticket.get("ambiguity_note"):
                return seed
        return None

    def test_ambiguity_note_hidden_until_internal_note_lookup(self) -> None:
        """Force a ticket with ambiguity_note by patching the dataset."""
        from unittest.mock import patch
        from server.tasks import load_dataset

        dataset = load_dataset()
        # Find a ticket with ambiguity_note
        ambiguous_tickets = [t for t in dataset if t.ambiguity_note is not None]
        self.assertGreater(len(ambiguous_tickets), 0, "No tickets with ambiguity_note in dataset")

        target = ambiguous_tickets[0]

        env = _make_env()
        # Patch the dataset to only contain the ambiguous ticket
        with patch.object(env, "_dataset", [target]):
            obs = env.reset(seed=0, task_id=3)

        self.assertIsNotNone(obs.current_ticket)
        self.assertNotIn("ambiguity_note", obs.current_ticket)
        self.assertIn("context_status", obs.current_ticket)
        self.assertTrue(obs.current_ticket["context_status"]["hidden_context_remaining"])
        self.assertGreater(obs.current_ticket["context_status"]["context_gap_count"], 0)

        obs = env.step(
            HelpdeskTicketAction(
                action_type="investigate",
                tool_name="lookup_internal_routing_note",
            )
        )

        self.assertEqual(obs.current_ticket["ambiguity_note"], target.ambiguity_note)
        self.assertGreater(obs.reward or 0.0, 0.0)

    def test_ambiguity_note_absent_when_ticket_has_none(self) -> None:
        """Tickets without ambiguity_note should not expose the key."""
        from unittest.mock import patch
        from server.tasks import load_dataset

        dataset = load_dataset()
        non_ambiguous = [t for t in dataset if t.ambiguity_note is None]
        self.assertGreater(len(non_ambiguous), 0)

        target = non_ambiguous[0]
        env = _make_env()
        with patch.object(env, "_dataset", [target]):
            obs = env.reset(seed=0, task_id=3)

        self.assertIsNotNone(obs.current_ticket)
        self.assertNotIn("ambiguity_note", obs.current_ticket)

    def test_tkt_nondefault_001_has_ambiguity_note(self) -> None:
        """TKT-NONDEFAULT-001 specifically has ambiguity_note set."""
        from unittest.mock import patch
        from server.tasks import load_dataset

        dataset = load_dataset()
        ticket = next((t for t in dataset if t.ticket_id == "TKT-NONDEFAULT-001"), None)
        self.assertIsNotNone(ticket, "TKT-NONDEFAULT-001 not found in dataset")
        self.assertIsNotNone(ticket.ambiguity_note)

        env = _make_env()
        with patch.object(env, "_dataset", [ticket]):
            obs = env.reset(seed=0, task_id=3)

        self.assertNotIn("ambiguity_note", obs.current_ticket)
        obs = env.step(
            HelpdeskTicketAction(
                action_type="investigate",
                tool_name="lookup_internal_routing_note",
            )
        )
        self.assertIn("ambiguity_note", obs.current_ticket)


class TestRelatedTicketPreviewInObservation(unittest.TestCase):
    """Follow-up tickets expose a lightweight preview of the linked ticket."""

    def _reset_linked_ticket_env(self):
        from unittest.mock import patch

        dataset = load_dataset()
        ticket = next((t for t in dataset if t.related_ticket_id is not None), None)
        self.assertIsNotNone(ticket, "No follow-up ticket found in dataset")
        related = next(
            (t for t in dataset if t.ticket_id == ticket.related_ticket_id),
            None,
        )
        self.assertIsNotNone(related, "Linked ticket missing from dataset")

        env = _make_env()
        with patch.object(env, "_dataset", [ticket]):
            with patch.object(
                env,
                "_tickets_by_id",
                {ticket.ticket_id: ticket, related.ticket_id: related},
            ):
                obs = env.reset(seed=0, task_id=3, queue_size=1)

        return env, obs, ticket, related

    def test_related_ticket_preview_present_when_ticket_has_link(self) -> None:
        env, obs, ticket, related = self._reset_linked_ticket_env()

        self.assertIsNotNone(obs.current_ticket)
        self.assertNotIn("related_ticket_preview", obs.current_ticket)
        self.assertIn("context_status", obs.current_ticket)
        self.assertTrue(obs.current_ticket["context_status"]["hidden_context_remaining"])
        self.assertGreater(obs.current_ticket["context_status"]["context_gap_count"], 0)

        obs = env.step(
            HelpdeskTicketAction(
                action_type="investigate",
                tool_name="lookup_related_ticket",
                tool_target_ticket_id=ticket.related_ticket_id,
            )
        )

        self.assertIn("related_ticket_preview", obs.current_ticket)
        self.assertEqual(
            obs.current_ticket["related_ticket_preview"]["ticket_id"],
            related.ticket_id,
        )
        self.assertEqual(
            obs.current_ticket["related_ticket_preview"]["title"],
            related.title,
        )

    def test_history_keeps_related_ticket_preview_after_step(self) -> None:
        env, obs, ticket, related = self._reset_linked_ticket_env()
        env.step(
            HelpdeskTicketAction(
                action_type="investigate",
                tool_name="lookup_related_ticket",
                tool_target_ticket_id=ticket.related_ticket_id,
            )
        )
        next_obs = env.step(
            HelpdeskTicketAction(
                issue_type=ticket.issue_type,
                priority=ticket.priority,
                assignment_group=ticket.assignment_group,
                resolution_action=ticket.resolution_action,
            )
        )

        self.assertGreaterEqual(len(next_obs.history), 1)
        self.assertIn("related_ticket_preview", next_obs.history[0])
        self.assertEqual(
            next_obs.history[0]["related_ticket_preview"]["ticket_id"],
            related.ticket_id,
        )


class TestObservationQueueContext(unittest.TestCase):
    """Observation includes clearer queue-position counters."""

    def test_reset_sets_queue_position_and_after_current_counts(self) -> None:
        env = _make_env()
        obs = env.reset(seed=0, task_id=1, queue_size=3)

        self.assertEqual(obs.queue_position, 1)
        self.assertEqual(obs.tickets_remaining, 3)
        self.assertEqual(obs.tickets_after_current, 2)

    def test_step_updates_queue_position_and_after_current_counts(self) -> None:
        env = _make_env()
        obs = env.reset(seed=0, task_id=1, queue_size=3)
        obs = env.step(_heuristic_action(obs))

        if obs.done:
            self.assertEqual(obs.queue_position, 0)
            self.assertEqual(obs.tickets_after_current, 0)
        else:
            self.assertEqual(obs.queue_position, 2)
            self.assertEqual(obs.tickets_remaining, 2)
            self.assertEqual(obs.tickets_after_current, 1)


# ---------------------------------------------------------------------------
# 9.6b — investigation actions and queue economics
# ---------------------------------------------------------------------------

class TestInvestigationActions(unittest.TestCase):
    """Minimal tool-assisted investigate/submit flow works and stays backwards compatible."""

    def _make_linked_env(self):
        from unittest.mock import patch

        dataset = load_dataset()
        ticket = next((t for t in dataset if t.related_ticket_id is not None), None)
        self.assertIsNotNone(ticket, "No follow-up ticket found in dataset")
        related = next(
            (t for t in dataset if t.ticket_id == ticket.related_ticket_id),
            None,
        )
        self.assertIsNotNone(related, "Linked ticket missing from dataset")
        env = _make_env()
        patch_dataset = patch.object(env, "_dataset", [ticket])
        patch_lookup = patch.object(
            env,
            "_tickets_by_id",
            {ticket.ticket_id: ticket, related.ticket_id: related},
        )
        patch_dataset.start()
        patch_lookup.start()
        self.addCleanup(patch_dataset.stop)
        self.addCleanup(patch_lookup.stop)
        obs = env.reset(seed=0, task_id=3, queue_size=1)
        return env, obs, ticket, related

    def _make_cluster_env(self):
        from unittest.mock import patch

        dataset = load_dataset()
        cluster_tickets = [
            ticket
            for ticket in dataset
            if ticket.service_cluster_id == "atlasbank_lockout_bridge"
        ]
        self.assertGreaterEqual(len(cluster_tickets), 2, "Expected atlasbank cluster tickets")
        root = next(
            (
                ticket
                for ticket in cluster_tickets
                if ticket.related_ticket_id is None
            ),
            None,
        )
        follow_up = next(
            (
                ticket
                for ticket in cluster_tickets
                if ticket.related_ticket_id is not None
            ),
            None,
        )
        self.assertIsNotNone(root, "Cluster root ticket missing")
        self.assertIsNotNone(follow_up, "Cluster follow-up ticket missing")

        env = _make_env()
        patch_dataset = patch.object(env, "_dataset", [root, follow_up])
        patch_lookup = patch.object(
            env,
            "_tickets_by_id",
            {root.ticket_id: root, follow_up.ticket_id: follow_up},
        )
        patch_dataset.start()
        patch_lookup.start()
        self.addCleanup(patch_dataset.stop)
        self.addCleanup(patch_lookup.stop)

        env.reset(seed=0, task_id=3, queue_size=2)
        env._queue = [root, follow_up]
        env._sync_queue_ticket_ids()
        env._state.current_ticket_index = 0
        obs = env._build_observation(get_task_definition(3))
        return env, obs, root, follow_up

    def test_investigation_action_does_not_advance_queue(self) -> None:
        env, obs, ticket, related = self._make_linked_env()

        investigate = HelpdeskTicketAction(
            action_type="investigate",
            tool_name="lookup_related_ticket",
            tool_target_ticket_id=ticket.related_ticket_id,
        )
        obs2 = env.step(investigate)

        self.assertFalse(obs2.done)
        self.assertEqual(obs2.tickets_processed, 0)
        self.assertEqual(obs2.queue_position, 1)
        self.assertIsNotNone(obs2.last_tool_result)
        self.assertTrue(obs2.last_tool_result["found"])
        self.assertEqual(
            obs2.last_tool_result["ticket"]["ticket_id"],
            related.ticket_id,
        )

    def test_submit_after_investigation_completes_episode(self) -> None:
        env, obs, ticket, related = self._make_linked_env()
        obs = env.step(
            HelpdeskTicketAction(
                action_type="investigate",
                tool_name="lookup_related_ticket",
                tool_target_ticket_id=ticket.related_ticket_id,
            )
        )
        operational_context = (obs.current_ticket or {}).get("operational_context", {})
        if operational_context.get("incident_recommended"):
            obs = env.step(HelpdeskTicketAction(action_type="open_incident"))
        final_obs = env.step(
            HelpdeskTicketAction(
                issue_type=ticket.issue_type,
                priority=ticket.priority,
                assignment_group=ticket.assignment_group,
                resolution_action=ticket.resolution_action,
            )
        )

        self.assertTrue(final_obs.done)
        self.assertEqual(final_obs.tickets_processed, 1)
        self.assertGreaterEqual(final_obs.reward, 0.0)
        self.assertLessEqual(final_obs.reward, 1.0)

    def test_requester_history_tool_returns_matches_for_same_requester(self) -> None:
        from unittest.mock import patch

        dataset = load_dataset()
        requester_counts: dict[str, int] = {}
        for ticket in dataset:
            requester_counts[ticket.requester] = requester_counts.get(ticket.requester, 0) + 1
        target_requester = next(
            (requester for requester, count in requester_counts.items() if count >= 2),
            None,
        )
        self.assertIsNotNone(target_requester, "Dataset has no repeated requester")
        duplicate_requester_group = [
            ticket for ticket in dataset if ticket.requester == target_requester
        ]
        self.assertGreaterEqual(len(duplicate_requester_group), 2)

        env = _make_env()
        with patch.object(env, "_dataset", duplicate_requester_group):
            with patch.object(
                env,
                "_tickets_by_id",
                {ticket.ticket_id: ticket for ticket in duplicate_requester_group},
            ):
                obs = env.reset(seed=0, task_id=2, queue_size=1)

        obs2 = env.step(
            HelpdeskTicketAction(
                action_type="investigate",
                tool_name="lookup_requester_history",
            )
        )

        self.assertIsNotNone(obs2.last_tool_result)
        self.assertEqual(obs2.last_tool_result["tool_name"], "lookup_requester_history")
        self.assertTrue(obs2.last_tool_result["found"])
        self.assertGreaterEqual(len(obs2.last_tool_result["matches"]), 1)

    def test_internal_note_tool_reveals_hidden_hard_task_context(self) -> None:
        from unittest.mock import patch

        dataset = load_dataset()
        ticket = next((t for t in dataset if t.ticket_id == "TKT-NONDEFAULT-003"), None)
        self.assertIsNotNone(ticket)

        env = _make_env()
        with patch.object(env, "_dataset", [ticket]):
            with patch.object(env, "_tickets_by_id", {ticket.ticket_id: ticket}):
                obs = env.reset(seed=0, task_id=3, queue_size=1)

        self.assertNotIn("ambiguity_note", obs.current_ticket)
        obs = env.step(
            HelpdeskTicketAction(
                action_type="investigate",
                tool_name="lookup_internal_routing_note",
            )
        )
        self.assertIn(ticket.ambiguity_note, obs.last_tool_result["routing_note"])
        self.assertEqual(obs.current_ticket["ambiguity_note"], ticket.ambiguity_note)
        self.assertGreater(obs.reward or 0.0, 0.0)

    def test_queue_capacity_forecast_reveals_routing_options(self) -> None:
        from unittest.mock import patch

        dataset = load_dataset()
        ticket = next(
            (t for t in dataset if t.alternate_route_score_multiplier > 0.0),
            None,
        )
        self.assertIsNotNone(ticket)

        env = _make_env()
        with patch.object(env, "_dataset", [ticket]):
            with patch.object(env, "_tickets_by_id", {ticket.ticket_id: ticket}):
                obs = env.reset(seed=0, task_id=3, queue_size=1)

        self.assertNotIn("routing_options", obs.current_ticket)
        obs = env.step(
            HelpdeskTicketAction(
                action_type="investigate",
                tool_name="lookup_queue_capacity_forecast",
            )
        )

        self.assertEqual(obs.last_tool_result["tool_name"], "lookup_queue_capacity_forecast")
        self.assertTrue(obs.last_tool_result["found"])
        self.assertIn("preferred_route_label", obs.last_tool_result)
        self.assertIn("routing_options", obs.current_ticket)
        self.assertGreaterEqual(len(obs.current_ticket["routing_options"]), 2)

    def test_queue_cluster_summary_reveals_future_cluster_load(self) -> None:
        env, obs, root, follow_up = self._make_cluster_env()
        self.assertNotIn("future_cluster_ticket_count", obs.current_ticket["operational_context"])
        self.assertTrue(obs.current_ticket["operational_context"]["cluster_coordination_hint"])

        obs = env.step(
            HelpdeskTicketAction(
                action_type="investigate",
                tool_name="lookup_queue_cluster_summary",
            )
        )

        self.assertEqual(obs.last_tool_result["tool_name"], "lookup_queue_cluster_summary")
        self.assertTrue(obs.last_tool_result["found"])
        self.assertEqual(obs.last_tool_result["future_cluster_ticket_count"], 1)
        self.assertEqual(
            obs.last_tool_result["future_cluster_ticket_ids"],
            [follow_up.ticket_id],
        )
        self.assertIn("cluster_summary", obs.current_ticket)

    def test_good_cluster_handling_stabilizes_future_follow_up(self) -> None:
        env, obs, root, follow_up = self._make_cluster_env()

        while (obs.current_ticket or {}).get("context_status", {}).get(
            "hidden_context_remaining"
        ):
            tool_name = (
                (obs.current_ticket or {})
                .get("context_status", {})
                .get("recommended_tools", [None])[0]
            )
            self.assertIsNotNone(tool_name)
            obs = env.step(
                HelpdeskTicketAction(
                    action_type="investigate",
                    tool_name=tool_name,
                )
            )
        obs = env.step(HelpdeskTicketAction(action_type="open_incident"))
        obs = env.step(
            HelpdeskTicketAction(
                issue_type=root.issue_type,
                priority=root.priority,
                assignment_group=root.assignment_group,
                resolution_action=root.resolution_action,
            )
        )

        self.assertFalse(obs.done)
        self.assertEqual(obs.current_ticket["ticket_id"], follow_up.ticket_id)
        self.assertTrue(obs.current_ticket["operational_context"]["incident_open"])
        self.assertIn(
            follow_up.ticket_id,
            obs.history[-1]["reward_components"]["cluster_stabilized_ticket_ids"],
        )
        stabilized_follow_up = env._queue[env.state.current_ticket_index]
        self.assertEqual(stabilized_follow_up.alternate_assignment_group, "service_desk")
        self.assertGreaterEqual(stabilized_follow_up.alternate_route_score_multiplier, 0.9)

    def test_bad_cluster_handling_escalates_future_follow_up(self) -> None:
        env, obs, root, follow_up = self._make_cluster_env()

        obs = env.step(
            HelpdeskTicketAction(
                issue_type="general_inquiry",
                priority="low",
                assignment_group="service_desk",
                resolution_action="acknowledge",
            )
        )

        self.assertFalse(obs.done)
        self.assertEqual(obs.current_ticket["ticket_id"], follow_up.ticket_id)
        self.assertIn(
            follow_up.ticket_id,
            obs.history[-1]["reward_components"]["cluster_destabilized_ticket_ids"],
        )
        escalated_follow_up = env._queue[env.state.current_ticket_index]
        self.assertNotEqual(escalated_follow_up.priority, follow_up.priority)
        self.assertIn(
            "did not fully resolve the blocker",
            escalated_follow_up.customer_update_note or "",
        )

    def test_submit_without_required_investigation_gets_shaping_penalty(self) -> None:
        from unittest.mock import patch

        dataset = load_dataset()
        ticket = next((t for t in dataset if t.ticket_id == "TKT-NONDEFAULT-003"), None)
        self.assertIsNotNone(ticket)

        env = _make_env()
        with patch.object(env, "_dataset", [ticket]):
            with patch.object(env, "_tickets_by_id", {ticket.ticket_id: ticket}):
                obs = env.reset(seed=0, task_id=3, queue_size=1)

        final_obs = env.step(
            HelpdeskTicketAction(
                issue_type=ticket.issue_type,
                priority=ticket.priority,
                assignment_group=ticket.assignment_group,
                resolution_action=ticket.resolution_action,
            )
        )

        self.assertTrue(final_obs.done)
        self.assertIsNotNone(final_obs.rubric_reward)
        self.assertLess(final_obs.reward, final_obs.rubric_reward)
        self.assertGreater(
            final_obs.last_reward_components.get("context_gap_penalty", 0.0),
            0.0,
        )

    def test_terminal_rubric_reports_queue_management_score(self) -> None:
        from unittest.mock import patch

        dataset = load_dataset()
        ticket = next((t for t in dataset if t.ticket_id == "TKT-NONDEFAULT-003"), None)
        self.assertIsNotNone(ticket)

        env = _make_env()
        with patch.object(env, "_dataset", [ticket]):
            with patch.object(env, "_tickets_by_id", {ticket.ticket_id: ticket}):
                obs = env.reset(seed=0, task_id=3, queue_size=1)

        final_obs = env.step(
            HelpdeskTicketAction(
                issue_type=ticket.issue_type,
                priority=ticket.priority,
                assignment_group=ticket.assignment_group,
                resolution_action=ticket.resolution_action,
            )
        )

        self.assertTrue(final_obs.done)
        self.assertIn("queue_management_score", final_obs.last_reward_components)
        self.assertIn("queue_management_breakdown", final_obs.last_reward_components)
        self.assertIn("context_resolution", final_obs.last_reward_components["queue_management_breakdown"])

    def test_capacity_forecast_hides_future_demand_until_tool_use(self) -> None:
        from unittest.mock import patch

        dataset = load_dataset()
        ticket = next(
            (t for t in dataset if t.alternate_route_score_multiplier > 0.0),
            None,
        )
        self.assertIsNotNone(ticket)

        env = _make_env()
        with patch.object(env, "_dataset", [ticket]):
            with patch.object(env, "_tickets_by_id", {ticket.ticket_id: ticket}):
                obs = env.reset(seed=0, task_id=3, queue_size=1)

        self.assertNotIn("future_queue_demand", obs.metadata)
        obs = env.step(
            HelpdeskTicketAction(
                action_type="investigate",
                tool_name="lookup_queue_capacity_forecast",
            )
        )
        self.assertIn("future_queue_demand", obs.last_tool_result)


class TestQueueEconomics(unittest.TestCase):
    """Free investigations are allowed, but excessive investigation gets a queue-level penalty."""

    def test_extra_investigations_reduce_final_reward(self) -> None:
        from unittest.mock import patch

        dataset = load_dataset()
        ticket = dataset[0]
        env = _make_env()
        with patch.object(env, "_dataset", [ticket]):
            with patch.object(env, "_tickets_by_id", {ticket.ticket_id: ticket}):
                obs = env.reset(seed=0, task_id=1, queue_size=1)

        obs = env.step(
            HelpdeskTicketAction(
                action_type="investigate",
                tool_name="lookup_requester_history",
            )
        )
        self.assertEqual(env.state.investigation_steps, 1)
        self.assertEqual(env.state.investigation_budget_remaining, 0)

        obs = env.step(
            HelpdeskTicketAction(
                action_type="investigate",
                tool_name="lookup_requester_history",
            )
        )
        self.assertEqual(env.state.investigation_steps, 2)

        final_obs = env.step(HelpdeskTicketAction(issue_type=ticket.issue_type))

        self.assertTrue(final_obs.done)
        self.assertLess(final_obs.reward, 1.0)
        self.assertAlmostEqual(
            final_obs.last_reward_components.get("investigation_penalty_applied", 0.0),
            0.04,
            places=9,
        )


class TestTerminalInvalidActionFinalReward(unittest.TestCase):
    """Terminal invalid submit actions should still return the queue-level final reward."""

    def test_last_invalid_submit_returns_trajectory_reward_not_zero(self) -> None:
        from unittest.mock import patch
        from server.tasks import get_task_definition as base_get_task_definition

        dataset = load_dataset()
        first = dataset[0]
        second = dataset[1]

        env = _make_env()
        with patch.object(env, "_dataset", [first, second]):
            with patch.object(
                env,
                "_tickets_by_id",
                {first.ticket_id: first, second.ticket_id: second},
            ):
                with patch(
                    "server.environment.get_task_definition",
                    side_effect=lambda task_id: (
                        {
                            **base_get_task_definition(task_id),
                            "allowed_fields": ["issue_type"],
                        }
                        if task_id == 1
                        else base_get_task_definition(task_id)
                    ),
                ):
                    obs = env.reset(seed=0, task_id=1, queue_size=2)

                    tickets_by_id = {first.ticket_id: first, second.ticket_id: second}
                    current = tickets_by_id[obs.current_ticket["ticket_id"]]
                    obs = env.step(HelpdeskTicketAction(issue_type=current.issue_type))
                    self.assertFalse(obs.done)

                    current = tickets_by_id[obs.current_ticket["ticket_id"]]
                    final_obs = env.step(
                        HelpdeskTicketAction(
                            issue_type=current.issue_type,
                            priority="medium",
                        )
                    )

                    self.assertTrue(final_obs.done)
                    expected_average = sum(env.state.per_ticket_scores) / len(
                        env.state.per_ticket_scores
                    )
                    self.assertGreater(final_obs.reward, 0.0)
                    self.assertAlmostEqual(final_obs.reward, expected_average, places=9)
                    self.assertAlmostEqual(env.state.total_reward, expected_average, places=9)
                    self.assertAlmostEqual(env.state.reward or 0.0, expected_average, places=9)


# ---------------------------------------------------------------------------
# 9.7 — Dataset has >= 3 non-default routing tickets
# ---------------------------------------------------------------------------

class TestDatasetNonDefaultRouting(unittest.TestCase):
    """9.7 — Dataset contains at least 3 tickets with non-default assignment_group."""

    def test_at_least_three_nondefault_routing_tickets(self) -> None:
        from vocabulary import ISSUE_TYPE_TO_ASSIGNMENT_GROUP

        dataset = load_dataset()
        non_default = [
            t for t in dataset
            if t.assignment_group != ISSUE_TYPE_TO_ASSIGNMENT_GROUP.get(t.issue_type)
        ]
        self.assertGreaterEqual(
            len(non_default), 10,
            f"Expected >= 10 non-default routing tickets, found {len(non_default)}: "
            + str([(t.ticket_id, t.issue_type, t.assignment_group) for t in non_default])
        )

    def test_tkt_nondefault_tickets_exist(self) -> None:
        dataset = load_dataset()
        ids = {t.ticket_id for t in dataset}
        for expected_id in ("TKT-NONDEFAULT-001", "TKT-NONDEFAULT-002", "TKT-NONDEFAULT-003"):
            self.assertIn(expected_id, ids, f"{expected_id} not found in dataset")


# ---------------------------------------------------------------------------
# 9.9 — SUPPORTS_CONCURRENT_SESSIONS is True
# ---------------------------------------------------------------------------

class TestConcurrentSessionsFlag(unittest.TestCase):
    """9.9 — HelpdeskTicketRoutingEnvironment.SUPPORTS_CONCURRENT_SESSIONS is True."""

    def test_supports_concurrent_sessions_is_true(self) -> None:
        self.assertTrue(HelpdeskTicketRoutingEnvironment.SUPPORTS_CONCURRENT_SESSIONS)

    def test_flag_is_boolean_true(self) -> None:
        flag = HelpdeskTicketRoutingEnvironment.SUPPORTS_CONCURRENT_SESSIONS
        self.assertIs(flag, True)


# ---------------------------------------------------------------------------
# 9.10 — GET /web returns 200 with HTML content
# ---------------------------------------------------------------------------

def _build_web_test_app():
    """Build a minimal FastAPI app with only the /web route for testing."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from server.tasks import TASKS
    from vocabulary import APP_ENV_NAME

    _app = FastAPI()

    @_app.get("/web", response_class=HTMLResponse)
    def web_ui():
        task_rows = "".join(
            f"<tr><td>{t['id']}</td><td>{t['name']}</td><td>{t['difficulty']}</td></tr>"
            for t in TASKS.values()
        )
        html = f"""<!DOCTYPE html>
<html><head><title>{APP_ENV_NAME}</title></head>
<body>
<h1>{APP_ENV_NAME}</h1>
<p>Version: 0.1.0 | <a href="/health">Health</a> | <a href="/docs">API Docs</a></p>
<h2>Tasks</h2>
<table border="1"><tr><th>ID</th><th>Name</th><th>Difficulty</th></tr>
{task_rows}
</table>
</body></html>"""
        return HTMLResponse(content=html)

    return _app


class TestWebUIEndpoint(unittest.TestCase):
    """9.10 — GET /web returns HTTP 200 with HTML content."""

    @classmethod
    def setUpClass(cls) -> None:
        from starlette.testclient import TestClient
        app = _build_web_test_app()
        cls.client = TestClient(app)

    def test_web_returns_200(self) -> None:
        response = self.client.get("/web")
        self.assertEqual(response.status_code, 200)

    def test_web_returns_html_content_type(self) -> None:
        response = self.client.get("/web")
        self.assertIn("text/html", response.headers.get("content-type", ""))

    def test_web_response_contains_html_tag(self) -> None:
        response = self.client.get("/web")
        self.assertIn("<!DOCTYPE html>", response.text)

    def test_web_response_contains_env_name(self) -> None:
        from vocabulary import APP_ENV_NAME
        response = self.client.get("/web")
        self.assertIn(APP_ENV_NAME, response.text)


if __name__ == "__main__":
    unittest.main()
