"""
Aggregate and plot hyperparameter sweep results.

Each child directory of a sweep folder must contain:
  - metadata.json
  - game_records.csv
  - ratings.csv

metadata.json should include:
  - parameter_name
  - parameter_value
  - focus_agent
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_csv(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(text: str) -> float:
    return float(text)


def _compute_agent_metrics(records: List[dict], focus_agent: str, max_turns: int) -> Dict[str, float]:
    wins = 0.0
    games = 0
    survival = []
    final_length = []
    placements = []
    capped = 0

    pair_wins = defaultdict(float)
    pair_games = defaultdict(int)

    for rec in records:
        turns = int(rec["turns"])
        if turns >= max_turns:
            capped += 1

        a = rec["agent_a"]
        b = rec["agent_b"]
        score_a = float(rec["score_a"])
        score_b = 1.0 - score_a

        pair_games[(a, b)] += 1
        pair_games[(b, a)] += 1
        pair_wins[(a, b)] += score_a
        pair_wins[(b, a)] += score_b

        if a == focus_agent:
            wins += score_a
            games += 1
            survival.append(float(rec["turns_survived_a"]))
            final_length.append(float(rec["final_length_a"]))
            placements.append(float(rec["placement_a"]))
        elif b == focus_agent:
            wins += score_b
            games += 1
            survival.append(float(rec["turns_survived_b"]))
            final_length.append(float(rec["final_length_b"]))
            placements.append(float(rec["placement_b"]))

    metrics = {
        "games": float(games),
        "wins": wins,
        "win_rate": wins / games if games else 0.0,
        "avg_survival": sum(survival) / len(survival) if survival else 0.0,
        "avg_final_length": sum(final_length) / len(final_length) if final_length else 0.0,
        "avg_placement": sum(placements) / len(placements) if placements else 0.0,
        "game_cap_rate": capped / len(records) if records else 0.0,
    }

    for opponent in ("Heuristic", "MCTS-Vanilla", "Random"):
        games_vs = pair_games.get((focus_agent, opponent), 0)
        wins_vs = pair_wins.get((focus_agent, opponent), 0.0)
        if games_vs:
            metrics[f"vs_{opponent.lower().replace('-', '_')}_win_rate"] = wins_vs / games_vs

    return metrics


def summarize_sweep(sweep_dir: str) -> str:
    rows = []

    for entry in sorted(os.listdir(sweep_dir)):
        child = os.path.join(sweep_dir, entry)
        if not os.path.isdir(child):
            continue

        metadata_path = os.path.join(child, "metadata.json")
        records_path = os.path.join(child, "game_records.csv")
        ratings_path = os.path.join(child, "ratings.csv")
        if not (os.path.exists(metadata_path) and os.path.exists(records_path) and os.path.exists(ratings_path)):
            continue

        with open(metadata_path) as f:
            metadata = json.load(f)

        focus_agent = metadata["focus_agent"]
        parameter_name = metadata["parameter_name"]
        parameter_value = metadata["parameter_value"]
        records = _load_csv(records_path)
        ratings = _load_csv(ratings_path)
        rating_row = next((row for row in ratings if row["agent"] == focus_agent), None)
        if rating_row is None:
            continue

        metrics = _compute_agent_metrics(records, focus_agent, int(metadata.get("max_turns", 200)))
        row = {
            "run_dir": entry,
            "parameter_name": parameter_name,
            "parameter_value": parameter_value,
            "focus_agent": focus_agent,
            "elo": _safe_float(rating_row["elo"]),
            "ts_conservative": _safe_float(rating_row["ts_conservative"]),
        }
        row.update(metrics)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No sweep runs found under {sweep_dir}")

    rows.sort(key=lambda row: float(row["parameter_value"]))

    summary_path = os.path.join(sweep_dir, "summary.csv")
    fieldnames = list(rows[0].keys())
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {summary_path}")
    return summary_path


def plot_sweep_summary(summary_csv: str) -> None:
    rows = _load_csv(summary_csv)
    sweep_dir = os.path.dirname(summary_csv)
    parameter_name = rows[0]["parameter_name"]
    x = [float(row["parameter_value"]) for row in rows]

    panels = [
        ("win_rate", "Overall Win Rate"),
        ("elo", "ELO"),
        ("ts_conservative", "TrueSkill Conservative"),
        ("avg_survival", "Average Survival"),
        ("avg_final_length", "Average Final Length"),
        ("vs_heuristic_win_rate", "Win Rate vs Heuristic"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for ax, (key, title) in zip(axes_flat, panels):
        y = [float(row.get(key, 0.0)) for row in rows]
        ax.plot(x, y, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel(parameter_name)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Hyperparameter Sweep: {parameter_name}", fontsize=14)
    fig.tight_layout()
    out = os.path.join(sweep_dir, f"{parameter_name}_sweep.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[saved] {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and plot a hyperparameter sweep folder")
    parser.add_argument("sweep_dir", help="Directory containing one subdirectory per sweep setting")
    args = parser.parse_args()

    summary_csv = summarize_sweep(args.sweep_dir)
    plot_sweep_summary(summary_csv)


if __name__ == "__main__":
    main()
