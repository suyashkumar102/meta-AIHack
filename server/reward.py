from __future__ import annotations

MILESTONE_HIGH_THRESHOLD = 0.8
MILESTONE_LOW_THRESHOLD = 0.2
MILESTONE_BONUS = 0.05
MILESTONE_PENALTY = 0.05
DELTA_REWARD_WEIGHT = 0.08
DELTA_REWARD_CAP = 0.04
PROCESS_BONUS_CAP = 0.08
RISK_PENALTY_CAP = 0.12
OPEN_INTERVAL_EPSILON = 0.001


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))


def clamp_open_unit_interval(value: float, epsilon: float = OPEN_INTERVAL_EPSILON) -> float:
    return max(epsilon, min(1.0 - epsilon, value))


def compute_step_adjustments(
    score: float,
    *,
    previous_average: float = 0.0,
    process_bonus: float = 0.0,
    risk_penalty: float = 0.0,
) -> dict[str, float]:
    base = _clamp_unit_interval(score)

    if score >= MILESTONE_HIGH_THRESHOLD:
        milestone_adjustment = MILESTONE_BONUS
    elif score < MILESTONE_LOW_THRESHOLD:
        milestone_adjustment = -MILESTONE_PENALTY
    else:
        milestone_adjustment = 0.0

    delta_adjustment = _clamp_delta((base - previous_average) * DELTA_REWARD_WEIGHT)
    bounded_process_bonus = max(0.0, min(PROCESS_BONUS_CAP, process_bonus))
    bounded_risk_penalty = max(0.0, min(RISK_PENALTY_CAP, risk_penalty))
    final_reward = _clamp_unit_interval(
        base
        + milestone_adjustment
        + delta_adjustment
        + bounded_process_bonus
        - bounded_risk_penalty
    )

    return {
        "base_reward": base,
        "milestone_adjustment": milestone_adjustment,
        "delta_adjustment": delta_adjustment,
        "process_bonus": bounded_process_bonus,
        "risk_penalty": bounded_risk_penalty,
        "final_reward": final_reward,
    }


def _clamp_delta(value: float) -> float:
    return max(-DELTA_REWARD_CAP, min(DELTA_REWARD_CAP, value))


def compute_step_reward(
    score: float,
    *,
    previous_average: float = 0.0,
    process_bonus: float = 0.0,
    risk_penalty: float = 0.0,
) -> float:
    return compute_step_adjustments(
        score,
        previous_average=previous_average,
        process_bonus=process_bonus,
        risk_penalty=risk_penalty,
    )["final_reward"]


def compute_trajectory_adjustments(
    per_ticket_scores: list[float],
    queue_size: int,
    steps_taken: int,
    *,
    completion_bonus: float = 0.0,
    consistency_bonus: float = 0.0,
) -> dict[str, float]:
    if not per_ticket_scores:
        return {
            "average_reward": 0.0,
            "completion_bonus": 0.0,
            "consistency_bonus": 0.0,
            "final_reward": 0.0,
        }
    avg = sum(per_ticket_scores) / len(per_ticket_scores)
    bounded_completion_bonus = max(0.0, min(0.08, completion_bonus))
    bounded_consistency_bonus = max(0.0, min(0.05, consistency_bonus))
    final_reward = clamp_open_unit_interval(
        avg + bounded_completion_bonus + bounded_consistency_bonus
    )
    return {
        "average_reward": avg,
        "completion_bonus": bounded_completion_bonus,
        "consistency_bonus": bounded_consistency_bonus,
        "final_reward": final_reward,
    }


def compute_trajectory_reward(
    per_ticket_scores: list[float],
    queue_size: int,
    steps_taken: int,
    *,
    completion_bonus: float = 0.0,
    consistency_bonus: float = 0.0,
) -> float:
    return compute_trajectory_adjustments(
        per_ticket_scores,
        queue_size,
        steps_taken,
        completion_bonus=completion_bonus,
        consistency_bonus=consistency_bonus,
    )["final_reward"]
