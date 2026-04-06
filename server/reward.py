from __future__ import annotations

MILESTONE_HIGH_THRESHOLD = 0.8
MILESTONE_LOW_THRESHOLD = 0.2
MILESTONE_BONUS = 0.05
MILESTONE_PENALTY = 0.05


def compute_step_reward(score: float) -> float:
    base = max(0.0, min(1.0, score))
    if score >= MILESTONE_HIGH_THRESHOLD:
        return min(1.0, base + MILESTONE_BONUS)
    if score < MILESTONE_LOW_THRESHOLD:
        return max(0.0, base - MILESTONE_PENALTY)
    return base


def compute_trajectory_reward(
    per_ticket_scores: list[float], queue_size: int, steps_taken: int
) -> float:
    if not per_ticket_scores:
        return 0.0
    avg = sum(per_ticket_scores) / len(per_ticket_scores)
    return max(0.0, min(1.0, avg))
