"""

This script orchestrates the following:
  - heuristic baseline
  - rollout ablation
  - improvement ablation
  - 40-seed small study
  - hyperparameter sweeps
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

from plot_results import generate_all_plots
from sweep_analysis import plot_sweep_summary, summarize_sweep
from tournament import AgentSpec, named_agent_specs, print_summary, run_matchup_sequential, run_tournament, save_metadata, save_results


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sanitize_value(value) -> str:
    text = str(value)
    text = text.replace(".", "p")
    text = text.replace("-", "m")
    return text


def write_matchup_summary(results_dir: str) -> None:
    records_path = os.path.join(results_dir, "game_records.csv")
    if not os.path.exists(records_path):
        return

    with open(records_path, newline="") as f:
        records = list(csv.DictReader(f))

    summary = defaultdict(lambda: {
        "games": 0,
        "wins_a": 0.0,
        "turns": [],
        "survival_a": [],
        "survival_b": [],
        "length_a": [],
        "length_b": [],
    })

    for rec in records:
        key = (rec["agent_a"], rec["agent_b"])
        row = summary[key]
        row["games"] += 1
        row["wins_a"] += float(rec["score_a"])
        row["turns"].append(float(rec["turns"]))
        row["survival_a"].append(float(rec["turns_survived_a"]))
        row["survival_b"].append(float(rec["turns_survived_b"]))
        row["length_a"].append(float(rec["final_length_a"]))
        row["length_b"].append(float(rec["final_length_b"]))

    out_path = os.path.join(results_dir, "matchup_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "agent_a",
                "agent_b",
                "games",
                "wins_a",
                "wins_b",
                "win_rate_a",
                "win_rate_b",
                "avg_turns",
                "avg_survival_a",
                "avg_survival_b",
                "avg_final_length_a",
                "avg_final_length_b",
            ],
        )
        writer.writeheader()
        for (agent_a, agent_b), row in sorted(summary.items()):
            games = row["games"]
            wins_a = row["wins_a"]
            writer.writerow({
                "agent_a": agent_a,
                "agent_b": agent_b,
                "games": games,
                "wins_a": round(wins_a, 3),
                "wins_b": round(games - wins_a, 3),
                "win_rate_a": round(wins_a / games, 4),
                "win_rate_b": round((games - wins_a) / games, 4),
                "avg_turns": round(sum(row["turns"]) / games, 3),
                "avg_survival_a": round(sum(row["survival_a"]) / games, 3),
                "avg_survival_b": round(sum(row["survival_b"]) / games, 3),
                "avg_final_length_a": round(sum(row["length_a"]) / games, 3),
                "avg_final_length_b": round(sum(row["length_b"]) / games, 3),
            })
    print(f"[saved] {out_path}")


def write_game_records(results_dir: str, records: List[dict]) -> None:
    if not records:
        return
    out_path = os.path.join(results_dir, "game_records.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"[saved] {out_path}")


def run_experiment(
    results_dir: str,
    agent_specs: List[AgentSpec],
    seeds: int,
    mcts_iterations: int,
    max_turns: int,
    workers: int,
    metadata_extra: Dict | None = None,
    plot: bool = True,
) -> None:
    _ensure_dir(results_dir)
    seed_list = list(range(seeds))
    results = run_tournament(
        agent_specs,
        seed_list,
        mcts_iterations,
        num_workers=workers,
        max_turns=max_turns,
    )
    print_summary(results)
    save_results(results, results_dir)
    save_metadata(
        results_dir,
        agent_specs,
        seed_list,
        mcts_iterations,
        max_turns,
        extra=metadata_extra or {},
    )
    write_matchup_summary(results_dir)
    if plot:
        generate_all_plots(results_dir)


def preset_heuristic_baseline(base_dir: str, seeds: int, iterations: int, max_turns: int, workers: int) -> None:
    run_experiment(
        os.path.join(base_dir, "heuristic_baseline"),
        named_agent_specs(["Random", "Heuristic"]),
        seeds=seeds,
        mcts_iterations=iterations,
        max_turns=max_turns,
        workers=workers,
        metadata_extra={"preset": "heuristic_baseline"},
    )


def preset_rollout_ablation(base_dir: str, seeds: int, iterations: int, max_turns: int, workers: int) -> None:
    run_experiment(
        os.path.join(base_dir, "rollout_ablation"),
        named_agent_specs(["Random", "Heuristic", "MCTS-Vanilla", "MCTS-HeurRollout"]),
        seeds=seeds,
        mcts_iterations=iterations,
        max_turns=max_turns,
        workers=workers,
        metadata_extra={"preset": "rollout_ablation"},
    )


def preset_vs_heuristic(base_dir: str, seeds: int, iterations: int, max_turns: int, workers: int) -> None:
    results_dir = os.path.join(base_dir, "vs_heuristic")
    _ensure_dir(results_dir)
    seed_list = list(range(seeds))
    heuristic = named_agent_specs(["Heuristic"])[0]
    challengers = named_agent_specs([
        "Random",
        "MCTS-Vanilla",
        "MCTS-HeurRollout",
        "MCTS-Prior",
        "MCTS-Opponent",
        "MCTS-RAVE",
        "MCTS-UCB1Tuned",
        "MCTS-All",
    ])

    all_records = []
    for challenger in challengers:
        print(f"\n=== {heuristic.name} vs {challenger.name} ===")
        all_records.extend(
            run_matchup_sequential(
                heuristic,
                challenger,
                seed_list,
                iterations,
                max_turns=max_turns,
            )
        )

    write_game_records(results_dir, all_records)
    save_metadata(
        results_dir,
        [heuristic] + challengers,
        seed_list,
        iterations,
        max_turns,
        extra={"preset": "vs_heuristic", "anchor_agent": heuristic.name},
    )
    write_matchup_summary(results_dir)
    generate_all_plots(results_dir)


def preset_improvement_ablation(base_dir: str, seeds: int, iterations: int, max_turns: int, workers: int) -> None:
    run_experiment(
        os.path.join(base_dir, "improvement_ablation"),
        named_agent_specs([
            "Random",
            "Heuristic",
            "MCTS-Vanilla",
            "MCTS-HeurRollout",
            "MCTS-Prior",
            "MCTS-Opponent",
            "MCTS-RAVE",
            "MCTS-UCB1Tuned",
            "MCTS-All",
        ]),
        seeds=seeds,
        mcts_iterations=iterations,
        max_turns=max_turns,
        workers=workers,
        metadata_extra={"preset": "improvement_ablation"},
    )


def preset_small_study_40(base_dir: str, iterations: int, max_turns: int, workers: int) -> None:
    run_experiment(
        os.path.join(base_dir, "small_study_40_seeds"),
        named_agent_specs(["Random", "Heuristic", "MCTS-Vanilla", "MCTS-All"]),
        seeds=40,
        mcts_iterations=iterations,
        max_turns=max_turns,
        workers=workers,
        metadata_extra={"preset": "small_study_40_seeds"},
    )


def run_hyperparameter_sweep(
    sweep_dir: str,
    parameter_name: str,
    parameter_values: List,
    iterations: int,
    max_turns: int,
    workers: int,
    seeds: int,
    config_overrides_builder,
) -> None:
    _ensure_dir(sweep_dir)
    for value in parameter_values:
        agent_name = f"MCTS-All-{parameter_name}-{_sanitize_value(value)}"
        spec = AgentSpec(
            name=agent_name,
            agent_type="mcts",
            config_name="ALL_IMPROVEMENTS_CONFIG",
            config_overrides=config_overrides_builder(value),
        )
        run_dir = os.path.join(sweep_dir, f"{parameter_name}_{_sanitize_value(value)}")
        run_experiment(
            run_dir,
            named_agent_specs(["Random", "Heuristic", "MCTS-Vanilla"]) + [spec],
            seeds=seeds,
            mcts_iterations=int(value) if parameter_name == "iterations" else iterations,
            max_turns=max_turns,
            workers=workers,
            metadata_extra={
                "preset": f"{parameter_name}_sweep",
                "parameter_name": parameter_name,
                "parameter_value": value,
                "focus_agent": agent_name,
                "base_config_name": "ALL_IMPROVEMENTS_CONFIG",
            },
        )

    summary_csv = summarize_sweep(sweep_dir)
    plot_sweep_summary(summary_csv)


def preset_hyperparameter_sweeps(base_dir: str, iterations: int, max_turns: int, workers: int, seeds: int) -> None:
    run_hyperparameter_sweep(
        os.path.join(base_dir, "hyper_iterations"),
        parameter_name="iterations",
        parameter_values=[25, 50, 100],
        iterations=iterations,
        max_turns=max_turns,
        workers=workers,
        seeds=seeds,
        config_overrides_builder=lambda _value: {},
    )
    run_hyperparameter_sweep(
        os.path.join(base_dir, "hyper_rollout_depth"),
        parameter_name="rollout_depth",
        parameter_values=[4, 8, 12],
        iterations=iterations,
        max_turns=max_turns,
        workers=workers,
        seeds=seeds,
        config_overrides_builder=lambda value: {"rollout_depth": int(value)},
    )
    run_hyperparameter_sweep(
        os.path.join(base_dir, "hyper_exploration"),
        parameter_name="exploration",
        parameter_values=[0.6, 1.05, 1.4],
        iterations=iterations,
        max_turns=max_turns,
        workers=workers,
        seeds=seeds,
        config_overrides_builder=lambda value: {"exploration": float(value)},
    )
    run_hyperparameter_sweep(
        os.path.join(base_dir, "hyper_prior_bonus"),
        parameter_name="prior_bonus_scale",
        parameter_values=[1.5, 4.0, 8.0],
        iterations=iterations,
        max_turns=max_turns,
        workers=workers,
        seeds=seeds,
        config_overrides_builder=lambda value: {"prior_bonus_scale": float(value)},
    )


PRESET_HELP = {
    "heuristic_baseline": "Heuristic vs Random baseline",
    "rollout_ablation": "Random vs heuristic rollouts study",
    "vs_heuristic": "All key agents matched against Heuristic for report ladder plots",
    "improvement_ablation": "Full MCTS improvement ablation",
    "small_study_40": "40-seed four-agent tournament",
    "hyper_sweeps": "All four hyperparameter sweep families",
    "all_required": "Run all report experiments",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible report experiment suite")
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(PRESET_HELP.keys()),
        help="Preset(s) to run. Repeatable. Default: all_required.",
    )
    parser.add_argument("--base-dir", default="experiment_runs", help="Base output directory")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker count")
    parser.add_argument("--max-turns", type=int, default=200, help="Maximum turns per game")
    parser.add_argument("--iterations", type=int, default=50, help="Default MCTS iterations per move")
    parser.add_argument("--seeds", type=int, default=20, help="Default seeds for non-40-seed presets")
    args = parser.parse_args()

    presets = args.preset or ["all_required"]
    base_dir = args.base_dir
    _ensure_dir(base_dir)

    expanded = []
    for preset in presets:
        if preset == "all_required":
            expanded.extend([
                "heuristic_baseline",
                "rollout_ablation",
                "vs_heuristic",
                "improvement_ablation",
                "small_study_40",
                "hyper_sweeps",
            ])
        else:
            expanded.append(preset)

    seen = set()
    ordered = []
    for preset in expanded:
        if preset not in seen:
            ordered.append(preset)
            seen.add(preset)

    for preset in ordered:
        print(f"\n{'=' * 70}\nRunning preset: {preset}\n{'=' * 70}")
        if preset == "heuristic_baseline":
            preset_heuristic_baseline(base_dir, args.seeds, args.iterations, args.max_turns, args.workers)
        elif preset == "rollout_ablation":
            preset_rollout_ablation(base_dir, args.seeds, args.iterations, args.max_turns, args.workers)
        elif preset == "vs_heuristic":
            preset_vs_heuristic(base_dir, args.seeds, args.iterations, args.max_turns, args.workers)
        elif preset == "improvement_ablation":
            preset_improvement_ablation(base_dir, args.seeds, args.iterations, args.max_turns, args.workers)
        elif preset == "small_study_40":
            preset_small_study_40(base_dir, args.iterations, args.max_turns, args.workers)
        elif preset == "hyper_sweeps":
            preset_hyperparameter_sweeps(base_dir, args.iterations, args.max_turns, args.workers, args.seeds)


if __name__ == "__main__":
    main()
