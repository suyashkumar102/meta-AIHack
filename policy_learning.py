#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

from models import HelpdeskTicketAction, HelpdeskTicketObservation
from server.environment import HelpdeskTicketRoutingEnvironment
from server.tasks import get_task_definition
from vocabulary import TASK_IDS


DEFAULT_COMPARE_POLICIES = (
    "no_investigation",
    "investigate_when_context_hidden",
)
DEFAULT_SEARCH_POLICIES = (
    "no_investigation",
    "legacy_single_probe",
    "investigate_when_context_hidden",
    "context_chain",
    "hybrid_context",
)
DEFAULT_OUTPUT_DIR = "analysis/policy_learning_runs"

SubmitBuilder = Callable[[dict[str, Any], list[str]], HelpdeskTicketAction]
EnvFactory = Callable[[], HelpdeskTicketRoutingEnvironment]


@dataclass(frozen=True)
class PolicyConfig:
    name: str
    investigate_hidden_context: bool
    investigate_related_ticket_hint: bool
    investigate_ambiguity_history: bool
    max_investigations_per_ticket: int
    description: str


POLICY_LIBRARY: dict[str, PolicyConfig] = {
    "no_investigation": PolicyConfig(
        name="no_investigation",
        investigate_hidden_context=False,
        investigate_related_ticket_hint=False,
        investigate_ambiguity_history=False,
        max_investigations_per_ticket=0,
        description="Always submit immediately and never investigate.",
    ),
    "legacy_single_probe": PolicyConfig(
        name="legacy_single_probe",
        investigate_hidden_context=False,
        investigate_related_ticket_hint=True,
        investigate_ambiguity_history=True,
        max_investigations_per_ticket=1,
        description="Mimics the earlier single-tool hint policy.",
    ),
    "investigate_when_context_hidden": PolicyConfig(
        name="investigate_when_context_hidden",
        investigate_hidden_context=True,
        investigate_related_ticket_hint=False,
        investigate_ambiguity_history=False,
        max_investigations_per_ticket=1,
        description="Investigate once when the environment says context is hidden.",
    ),
    "context_chain": PolicyConfig(
        name="context_chain",
        investigate_hidden_context=True,
        investigate_related_ticket_hint=False,
        investigate_ambiguity_history=False,
        max_investigations_per_ticket=3,
        description="Follow the environment's required-tool chain until context is revealed.",
    ),
    "hybrid_context": PolicyConfig(
        name="hybrid_context",
        investigate_hidden_context=True,
        investigate_related_ticket_hint=True,
        investigate_ambiguity_history=True,
        max_investigations_per_ticket=3,
        description="Use hidden-context signals first, then legacy ambiguity hints.",
    ),
}


def _dedupe_preserving_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def parse_int_spec(spec: str, *, field_name: str) -> list[int]:
    values: list[int] = []
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            try:
                start = int(start_raw)
                end = int(end_raw)
            except ValueError as exc:
                raise ValueError(f"{field_name} contains an invalid range: {part!r}") from exc
            if end < start:
                raise ValueError(f"{field_name} range must be ascending: {part!r}")
            values.extend(range(start, end + 1))
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise ValueError(f"{field_name} contains an invalid integer: {part!r}") from exc
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return _dedupe_preserving_order(values)


def parse_task_ids(spec: str) -> list[int]:
    task_ids = parse_int_spec(spec, field_name="task_ids")
    unsupported = [task_id for task_id in task_ids if task_id not in TASK_IDS]
    if unsupported:
        raise ValueError(f"Unsupported task_ids: {unsupported}")
    return task_ids


def resolve_policies(spec: str) -> list[PolicyConfig]:
    names = [name.strip() for name in spec.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one policy must be specified")
    policies: list[PolicyConfig] = []
    for name in names:
        if name not in POLICY_LIBRARY:
            raise ValueError(
                f"Unknown policy {name!r}. Available policies: {sorted(POLICY_LIBRARY)}"
            )
        policies.append(POLICY_LIBRARY[name])
    return policies


def default_submit_builder(
    ticket: dict[str, Any], allowed_fields: list[str]
) -> HelpdeskTicketAction:
    inference = importlib.import_module("inference")
    candidate = inference.heuristic_action(ticket, allowed_fields)
    candidate, _ = inference.apply_domain_overrides(ticket, candidate, allowed_fields)
    return HelpdeskTicketAction(**candidate)


def choose_policy_action(
    policy: PolicyConfig,
    observation: HelpdeskTicketObservation,
    investigations_by_ticket: dict[str, int],
    submit_builder: SubmitBuilder,
) -> tuple[HelpdeskTicketAction, str]:
    ticket = observation.current_ticket or {}
    ticket_id = str(ticket.get("ticket_id", ""))
    ticket_investigations = investigations_by_ticket.get(ticket_id, 0)
    revealed_tools = set(((ticket.get("context_status") or {}).get("revealed_tools") or []))
    remaining_tools = list(((ticket.get("context_status") or {}).get("remaining_tools") or []))

    if ticket_investigations < policy.max_investigations_per_ticket:
        if policy.investigate_hidden_context and remaining_tools:
            tool_name = str(remaining_tools[0])
            return (
                HelpdeskTicketAction(action_type="investigate", tool_name=tool_name),
                "investigate_hidden_context",
            )
        if (
            policy.investigate_related_ticket_hint
            and ticket.get("related_ticket_id")
            and "lookup_related_ticket" not in revealed_tools
        ):
            return (
                HelpdeskTicketAction(
                    action_type="investigate",
                    tool_name="lookup_related_ticket",
                ),
                "investigate_related_ticket_hint",
            )
        if (
            policy.investigate_ambiguity_history
            and ticket.get("ambiguity_note")
            and "lookup_requester_history" not in revealed_tools
        ):
            return (
                HelpdeskTicketAction(
                    action_type="investigate",
                    tool_name="lookup_requester_history",
                ),
                "investigate_ambiguity_history",
            )

    return submit_builder(ticket, list(observation.allowed_fields)), "submit"


def rollout_episode(
    *,
    env: HelpdeskTicketRoutingEnvironment,
    policy: PolicyConfig,
    seed: int,
    task_id: int,
    submit_builder: SubmitBuilder,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task = get_task_definition(task_id)
    observation = env.reset(seed=seed, task_id=task_id)
    investigations_by_ticket: dict[str, int] = {}
    episode_return = 0.0
    trajectories: list[dict[str, Any]] = []

    while not observation.done:
        ticket = observation.current_ticket or {}
        ticket_id = str(ticket.get("ticket_id", ""))
        action, action_source = choose_policy_action(
            policy,
            observation,
            investigations_by_ticket,
            submit_builder,
        )
        next_observation = env.step(action)
        reward_value = float(next_observation.reward or 0.0)
        episode_return += reward_value
        if action.action_type == "investigate" and ticket_id:
            investigations_by_ticket[ticket_id] = investigations_by_ticket.get(ticket_id, 0) + 1

        history_entry = env.state.history_entries[-1] if env.state.history_entries else {}
        trajectories.append(
            {
                "policy": policy.name,
                "seed": seed,
                "task_id": task_id,
                "task_name": task["name"],
                "episode_id": env.state.episode_id,
                "step_index": len(trajectories) + 1,
                "ticket_id": history_entry.get("ticket_id", ticket_id),
                "action_source": action_source,
                "action": action.model_dump(exclude_none=True),
                "step_reward": reward_value,
                "rubric_reward": next_observation.rubric_reward,
                "done": next_observation.done,
                "feedback_summary": history_entry.get("feedback_summary"),
                "reward_kind": history_entry.get("reward_kind"),
                "score": history_entry.get("score"),
                "breakdown": history_entry.get("breakdown", {}),
                "reward_components": history_entry.get("reward_components", {}),
                "context_status_before_action": ticket.get("context_status"),
            }
        )
        observation = next_observation

    queue_size = max(1, len(env.state.queue_ticket_ids))
    terminal_reward = float(observation.reward or 0.0)
    terminal_rubric_reward = (
        float(observation.rubric_reward)
        if observation.rubric_reward is not None
        else terminal_reward
    )
    summary = {
        "policy": policy.name,
        "policy_config": asdict(policy),
        "seed": seed,
        "task_id": task_id,
        "task_name": task["name"],
        "episode_id": env.state.episode_id,
        "queue_size": queue_size,
        "step_count": env.state.step_count,
        "tickets_processed": len(env.state.per_ticket_scores),
        "investigation_steps": env.state.investigation_steps,
        "episode_return": episode_return,
        "normalized_return": episode_return / queue_size,
        "terminal_reward": terminal_reward,
        "terminal_rubric_reward": terminal_rubric_reward,
        "average_ticket_score": env.state.average_score_so_far,
        "per_ticket_scores": list(env.state.per_ticket_scores),
    }
    return summary, trajectories


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(mean(values), 6)


def summarize_policy_episodes(
    policy: PolicyConfig,
    episode_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    for task_id in TASK_IDS:
        task_episodes = [
            episode for episode in episode_summaries if episode["task_id"] == task_id
        ]
        if not task_episodes:
            continue
        per_task[str(task_id)] = {
            "episodes": len(task_episodes),
            "avg_episode_return": _safe_mean(
                [float(episode["episode_return"]) for episode in task_episodes]
            ),
            "avg_normalized_return": _safe_mean(
                [float(episode["normalized_return"]) for episode in task_episodes]
            ),
            "avg_terminal_reward": _safe_mean(
                [float(episode["terminal_reward"]) for episode in task_episodes]
            ),
            "avg_terminal_rubric_reward": _safe_mean(
                [float(episode["terminal_rubric_reward"]) for episode in task_episodes]
            ),
            "avg_investigation_steps": _safe_mean(
                [float(episode["investigation_steps"]) for episode in task_episodes]
            ),
        }

    return {
        "policy": policy.name,
        "config": asdict(policy),
        "episodes": len(episode_summaries),
        "avg_episode_return": _safe_mean(
            [float(episode["episode_return"]) for episode in episode_summaries]
        ),
        "avg_normalized_return": _safe_mean(
            [float(episode["normalized_return"]) for episode in episode_summaries]
        ),
        "avg_terminal_reward": _safe_mean(
            [float(episode["terminal_reward"]) for episode in episode_summaries]
        ),
        "avg_terminal_rubric_reward": _safe_mean(
            [float(episode["terminal_rubric_reward"]) for episode in episode_summaries]
        ),
        "avg_investigation_steps": _safe_mean(
            [float(episode["investigation_steps"]) for episode in episode_summaries]
        ),
        "avg_ticket_score": _safe_mean(
            [float(episode["average_ticket_score"]) for episode in episode_summaries]
        ),
        "per_task": per_task,
    }


def evaluate_policy(
    policy: PolicyConfig,
    seeds: Iterable[int],
    task_ids: Iterable[int],
    *,
    env_factory: EnvFactory = HelpdeskTicketRoutingEnvironment,
    submit_builder: SubmitBuilder = default_submit_builder,
) -> dict[str, Any]:
    episode_summaries: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []

    for seed in seeds:
        for task_id in task_ids:
            env = env_factory()
            summary, episode_trajectories = rollout_episode(
                env=env,
                policy=policy,
                seed=seed,
                task_id=task_id,
                submit_builder=submit_builder,
            )
            episode_summaries.append(summary)
            trajectories.extend(episode_trajectories)

    return {
        "policy": policy.name,
        "summary": summarize_policy_episodes(policy, episode_summaries),
        "episodes": episode_summaries,
        "trajectories": trajectories,
    }


def _selection_tuple(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(summary["avg_normalized_return"]),
        float(summary["avg_terminal_reward"]),
        float(summary["avg_terminal_rubric_reward"]),
        -float(summary["avg_investigation_steps"]),
    )


def select_best_policy(policy_runs: list[dict[str, Any]]) -> dict[str, Any]:
    return max(policy_runs, key=lambda run: _selection_tuple(run["summary"]))


def _delta(best: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return round(float(best[key]) - float(baseline[key]), 6)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def compare_policies(
    policies: list[PolicyConfig],
    seeds: list[int],
    task_ids: list[int],
    *,
    output_dir: Path,
    env_factory: EnvFactory = HelpdeskTicketRoutingEnvironment,
    submit_builder: SubmitBuilder = default_submit_builder,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    policy_runs = [
        evaluate_policy(
            policy,
            seeds,
            task_ids,
            env_factory=env_factory,
            submit_builder=submit_builder,
        )
        for policy in policies
    ]
    best_run = select_best_policy(policy_runs)
    baseline_run = policy_runs[0]

    report = {
        "mode": "compare",
        "task_ids": task_ids,
        "seeds": seeds,
        "selection_metric": "avg_normalized_return",
        "baseline_policy": baseline_run["policy"],
        "best_policy": best_run["policy"],
        "improvement_vs_baseline": {
            "avg_episode_return": _delta(
                best_run["summary"], baseline_run["summary"], "avg_episode_return"
            ),
            "avg_normalized_return": _delta(
                best_run["summary"], baseline_run["summary"], "avg_normalized_return"
            ),
            "avg_terminal_reward": _delta(
                best_run["summary"], baseline_run["summary"], "avg_terminal_reward"
            ),
            "avg_terminal_rubric_reward": _delta(
                best_run["summary"],
                baseline_run["summary"],
                "avg_terminal_rubric_reward",
            ),
        },
        "policy_summaries": [run["summary"] for run in policy_runs],
        "ranking": [
            run["policy"]
            for run in sorted(
                policy_runs,
                key=lambda run: _selection_tuple(run["summary"]),
                reverse=True,
            )
        ],
        "artifacts": {
            "summary": str(output_dir / "compare_summary.json"),
            "episodes": str(output_dir / "compare_episodes.jsonl"),
            "trajectories": str(output_dir / "compare_trajectories.jsonl"),
        },
    }

    _write_json(output_dir / "compare_summary.json", report)
    _write_jsonl(
        output_dir / "compare_episodes.jsonl",
        (
            {"policy": run["policy"], **episode}
            for run in policy_runs
            for episode in run["episodes"]
        ),
    )
    _write_jsonl(
        output_dir / "compare_trajectories.jsonl",
        (trajectory for run in policy_runs for trajectory in run["trajectories"]),
    )
    return report


def search_policies(
    candidate_policies: list[PolicyConfig],
    train_seeds: list[int],
    eval_seeds: list[int],
    task_ids: list[int],
    *,
    output_dir: Path,
    env_factory: EnvFactory = HelpdeskTicketRoutingEnvironment,
    submit_builder: SubmitBuilder = default_submit_builder,
    baseline_policy_name: str = "no_investigation",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    train_runs = [
        evaluate_policy(
            policy,
            train_seeds,
            task_ids,
            env_factory=env_factory,
            submit_builder=submit_builder,
        )
        for policy in candidate_policies
    ]
    selected_run = select_best_policy(train_runs)
    selected_policy = POLICY_LIBRARY[selected_run["policy"]]
    eval_selected = evaluate_policy(
        selected_policy,
        eval_seeds,
        task_ids,
        env_factory=env_factory,
        submit_builder=submit_builder,
    )

    baseline_policy = POLICY_LIBRARY.get(baseline_policy_name, candidate_policies[0])
    eval_baseline = evaluate_policy(
        baseline_policy,
        eval_seeds,
        task_ids,
        env_factory=env_factory,
        submit_builder=submit_builder,
    )

    report = {
        "mode": "search",
        "task_ids": task_ids,
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "selection_metric": "avg_normalized_return",
        "candidate_policies": [policy.name for policy in candidate_policies],
        "selected_policy": selected_policy.name,
        "baseline_policy": baseline_policy.name,
        "train_policy_summaries": [run["summary"] for run in train_runs],
        "eval_selected_summary": eval_selected["summary"],
        "eval_baseline_summary": eval_baseline["summary"],
        "eval_improvement_vs_baseline": {
            "avg_episode_return": _delta(
                eval_selected["summary"],
                eval_baseline["summary"],
                "avg_episode_return",
            ),
            "avg_normalized_return": _delta(
                eval_selected["summary"],
                eval_baseline["summary"],
                "avg_normalized_return",
            ),
            "avg_terminal_reward": _delta(
                eval_selected["summary"],
                eval_baseline["summary"],
                "avg_terminal_reward",
            ),
            "avg_terminal_rubric_reward": _delta(
                eval_selected["summary"],
                eval_baseline["summary"],
                "avg_terminal_rubric_reward",
            ),
        },
        "artifacts": {
            "summary": str(output_dir / "search_summary.json"),
            "train_episodes": str(output_dir / "search_train_episodes.jsonl"),
            "train_trajectories": str(output_dir / "search_train_trajectories.jsonl"),
            "eval_episodes": str(output_dir / "search_eval_episodes.jsonl"),
            "eval_trajectories": str(output_dir / "search_eval_trajectories.jsonl"),
        },
    }

    _write_json(output_dir / "search_summary.json", report)
    _write_jsonl(
        output_dir / "search_train_episodes.jsonl",
        (
            {"policy": run["policy"], **episode}
            for run in train_runs
            for episode in run["episodes"]
        ),
    )
    _write_jsonl(
        output_dir / "search_train_trajectories.jsonl",
        (trajectory for run in train_runs for trajectory in run["trajectories"]),
    )
    _write_jsonl(
        output_dir / "search_eval_episodes.jsonl",
        (
            {"policy": eval_selected["policy"], **episode}
            for episode in eval_selected["episodes"]
        ),
    )
    _write_jsonl(
        output_dir / "search_eval_trajectories.jsonl",
        (trajectory for trajectory in eval_selected["trajectories"]),
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run seeded local rollouts and a small policy-improvement loop for the "
            "IT helpdesk OpenEnv environment."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare fixed policy choices across repeated seeded rollouts.",
    )
    compare_parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_COMPARE_POLICIES),
        help=f"Comma-separated policy names. Available: {', '.join(POLICY_LIBRARY)}",
    )
    compare_parser.add_argument(
        "--seeds",
        default="42-51",
        help="Comma-separated seeds or ranges, for example 42-51 or 42,50,60.",
    )
    compare_parser.add_argument(
        "--task-ids",
        default="1,2,3",
        help="Comma-separated task IDs or ranges, for example 1,2,3 or 1-3.",
    )
    compare_parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON and JSONL artifacts.",
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Select the best policy on train seeds, then re-evaluate on holdout seeds.",
    )
    search_parser.add_argument(
        "--candidate-policies",
        default=",".join(DEFAULT_SEARCH_POLICIES),
        help=f"Comma-separated candidate policy names. Available: {', '.join(POLICY_LIBRARY)}",
    )
    search_parser.add_argument(
        "--train-seeds",
        default="40-49",
        help="Train seeds used for reward-based policy selection.",
    )
    search_parser.add_argument(
        "--eval-seeds",
        default="50-59",
        help="Holdout seeds used for the selected policy evaluation.",
    )
    search_parser.add_argument(
        "--task-ids",
        default="1,2,3",
        help="Comma-separated task IDs or ranges, for example 1,2,3 or 1-3.",
    )
    search_parser.add_argument(
        "--baseline-policy",
        default="no_investigation",
        help="Baseline policy used for the final improvement delta.",
    )
    search_parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON and JSONL artifacts.",
    )

    return parser


def _print_summary(label: str, summary: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                label: {
                    "policy": summary["policy"],
                    "avg_episode_return": summary["avg_episode_return"],
                    "avg_normalized_return": summary["avg_normalized_return"],
                    "avg_terminal_reward": summary["avg_terminal_reward"],
                    "avg_terminal_rubric_reward": summary["avg_terminal_rubric_reward"],
                    "avg_investigation_steps": summary["avg_investigation_steps"],
                }
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.command == "compare":
        policies = resolve_policies(args.policies)
        seeds = parse_int_spec(args.seeds, field_name="seeds")
        task_ids = parse_task_ids(args.task_ids)
        report = compare_policies(
            policies,
            seeds,
            task_ids,
            output_dir=output_dir,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    candidate_policies = resolve_policies(args.candidate_policies)
    train_seeds = parse_int_spec(args.train_seeds, field_name="train_seeds")
    eval_seeds = parse_int_spec(args.eval_seeds, field_name="eval_seeds")
    task_ids = parse_task_ids(args.task_ids)
    report = search_policies(
        candidate_policies,
        train_seeds,
        eval_seeds,
        task_ids,
        output_dir=output_dir,
        baseline_policy_name=args.baseline_policy,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
