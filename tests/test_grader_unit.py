from __future__ import annotations

import unittest

import openenv_test_stubs  # noqa: F401

from models import HelpdeskTicketAction, HelpdeskTicketRecord
from server.grader import (
    ISSUE_TYPE_SIMILARITY,
    PRIORITY_SCORES,
    TASK_WEIGHTS,
    grade_action,
)
from vocabulary import ISSUE_TYPES, PRIORITIES


def _ticket(
    *,
    issue_type: str = "billing_license",
    priority: str = "high",
    assignment_group: str = "license_ops",
    resolution_action: str = "fulfill",
) -> HelpdeskTicketRecord:
    return HelpdeskTicketRecord(
        ticket_id="ticket-test",
        title="Test ticket",
        requester="user@example.com",
        description="Synthetic ticket used for deterministic grader tests.",
        issue_type=issue_type,
        priority=priority,
        assignment_group=assignment_group,
        resolution_action=resolution_action,
    )


class GraderUnitTests(unittest.TestCase):
    def test_task_3_exact_match_scores_one(self) -> None:
        ticket = _ticket()
        action = HelpdeskTicketAction(
            issue_type="billing_license",
            priority="high",
            assignment_group="license_ops",
            resolution_action="fulfill",
        )

        score, breakdown = grade_action(action, ticket, task_id=3)

        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(
            breakdown,
            {
                "issue_type": 1.0,
                "priority": 1.0,
                "assignment_group": 1.0,
                "resolution_action": 1.0,
            },
        )

    def test_unknown_task_id_raises(self) -> None:
        ticket = _ticket()
        action = HelpdeskTicketAction(issue_type="billing_license")

        with self.assertRaisesRegex(ValueError, "Unsupported task_id"):
            grade_action(action, ticket, task_id=99)

    def test_issue_type_partial_credit_only_for_known_similarity_pair(self) -> None:
        ticket = _ticket(issue_type="billing_license")
        action = HelpdeskTicketAction(issue_type="service_request")

        score, breakdown = grade_action(action, ticket, task_id=1)

        self.assertAlmostEqual(score, 0.4)
        self.assertEqual(breakdown, {"issue_type": 0.4})

    def test_issue_type_scoring_matches_declared_similarity_table_exhaustively(self) -> None:
        for expected in ISSUE_TYPES:
            for predicted in ISSUE_TYPES:
                with self.subTest(expected=expected, predicted=predicted):
                    ticket = _ticket(issue_type=expected)
                    action = HelpdeskTicketAction(issue_type=predicted)

                    score, breakdown = grade_action(action, ticket, task_id=1)

                    expected_score = (
                        1.0
                        if predicted == expected
                        else ISSUE_TYPE_SIMILARITY.get((predicted, expected), 0.0)
                    )
                    self.assertAlmostEqual(score, expected_score)
                    self.assertEqual(breakdown, {"issue_type": expected_score})

    def test_unrelated_issue_type_gets_zero_not_fuzzy_credit(self) -> None:
        ticket = _ticket(issue_type="onboarding")
        action = HelpdeskTicketAction(issue_type="spam_phishing")

        score, breakdown = grade_action(action, ticket, task_id=1)

        self.assertAlmostEqual(score, 0.0)
        self.assertEqual(breakdown, {"issue_type": 0.0})

    def test_priority_scoring_uses_defined_proximity_table(self) -> None:
        ticket = _ticket(priority="critical")
        action = HelpdeskTicketAction(issue_type="billing_license", priority="high")

        score, breakdown = grade_action(action, ticket, task_id=2)

        self.assertAlmostEqual(breakdown["issue_type"], 1.0)
        self.assertAlmostEqual(breakdown["priority"], 0.6)
        self.assertAlmostEqual(score, 0.84)

    def test_priority_scoring_matches_declared_table_exhaustively(self) -> None:
        for expected in PRIORITIES:
            for predicted in PRIORITIES:
                with self.subTest(expected=expected, predicted=predicted):
                    ticket = _ticket(priority=expected)
                    action = HelpdeskTicketAction(
                        issue_type="billing_license",
                        priority=predicted,
                    )

                    score, breakdown = grade_action(action, ticket, task_id=2)

                    priority_score = (
                        1.0
                        if predicted == expected
                        else PRIORITY_SCORES.get((predicted, expected), 0.0)
                    )
                    self.assertEqual(
                        breakdown,
                        {"issue_type": 1.0, "priority": priority_score},
                    )
                    self.assertAlmostEqual(score, 0.6 + 0.4 * priority_score)

    def test_task_2_weights_apply_as_documented(self) -> None:
        ticket = _ticket(priority="high")
        action = HelpdeskTicketAction(issue_type="billing_license", priority="medium")

        score, breakdown = grade_action(action, ticket, task_id=2)

        self.assertEqual(breakdown, {"issue_type": 1.0, "priority": 0.5})
        self.assertAlmostEqual(score, 0.8)

    def test_assignment_group_is_exact_match_only(self) -> None:
        ticket = _ticket()
        action = HelpdeskTicketAction(
            issue_type="billing_license",
            priority="high",
            assignment_group="service_desk",
            resolution_action="fulfill",
        )

        score, breakdown = grade_action(action, ticket, task_id=3)

        self.assertEqual(breakdown["assignment_group"], 0.0)
        self.assertAlmostEqual(score, 0.75)

    def test_task_3_weights_apply_as_documented(self) -> None:
        ticket = _ticket(priority="high")
        action = HelpdeskTicketAction(
            issue_type="billing_license",
            priority="medium",
            assignment_group="service_desk",
            resolution_action="fulfill",
        )

        score, breakdown = grade_action(action, ticket, task_id=3)

        self.assertEqual(
            breakdown,
            {
                "issue_type": 1.0,
                "priority": 0.5,
                "assignment_group": 0.0,
                "resolution_action": 1.0,
            },
        )
        self.assertAlmostEqual(score, 0.65)

    def test_resolution_action_is_exact_match_only(self) -> None:
        ticket = _ticket()
        action = HelpdeskTicketAction(
            issue_type="billing_license",
            priority="high",
            assignment_group="license_ops",
            resolution_action="assign",
        )

        score, breakdown = grade_action(action, ticket, task_id=3)

        self.assertEqual(breakdown["resolution_action"], 0.0)
        self.assertAlmostEqual(score, 0.8)

    def test_partial_credit_tables_never_override_exact_match(self) -> None:
        for pair, value in ISSUE_TYPE_SIMILARITY.items():
            with self.subTest(table="issue_type", pair=pair):
                self.assertGreater(value, 0.0)
                self.assertLess(value, 1.0)

        for pair, value in PRIORITY_SCORES.items():
            with self.subTest(table="priority", pair=pair):
                self.assertGreater(value, 0.0)
                self.assertLess(value, 1.0)

    def test_task_weights_sum_to_one_for_each_task(self) -> None:
        for task_id, weights in TASK_WEIGHTS.items():
            with self.subTest(task_id=task_id):
                self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_grade_action_is_deterministic_for_same_inputs(self) -> None:
        ticket = _ticket(issue_type="service_request", priority="medium")
        action = HelpdeskTicketAction(
            issue_type="general_inquiry",
            priority="low",
            assignment_group="license_ops",
            resolution_action="assign",
        )

        first_score, first_breakdown = grade_action(action, ticket, task_id=3)
        second_score, second_breakdown = grade_action(action, ticket, task_id=3)

        self.assertEqual(first_score, second_score)
        self.assertEqual(first_breakdown, second_breakdown)


if __name__ == "__main__":
    unittest.main()
