"""
Generate the report-specific figure set.

Inputs:
  - tournament folder (default: small_study_40_seeds)
  - vs_heuristic folder (default: experiment_runs/vs_heuristic)
  - hyper sweep folders under experiment_runs/

Outputs:
  - 11 report-oriented plots in the chosen output directory
  - report_matchup_database.csv summarising the focused Heuristic matchups
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AGENT_COLORS: Dict[str, str] = {
    "Random": "#888888",
    "Heuristic": "#D97706",
    "MCTS-Vanilla": "#3B82F6",
    "MCTS-HeurRollout": "#8B5CF6",
    "MCTS-Prior": "#EC4899",
    "MCTS-Opponent": "#F59E0B",
    "MCTS-RAVE": "#10B981",
    "MCTS-UCB1Tuned": "#EF4444",
    "MCTS-All": "#00CC00",
}

matplotlib.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})


def _color(name: str) -> str:
    return AGENT_COLORS.get(name, "#555555")


def _load_csv(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _pair_records(records: List[dict], agent_x: str, agent_y: str) -> List[dict]:
    result = []
    for rec in records:
        a, b = rec["agent_a"], rec["agent_b"]
        if {a, b} == {agent_x, agent_y}:
            result.append(rec)
    return result


def _win_rate_for(agent: str, opponent: str, records: List[dict]) -> float:
    wins = 0.0
    games = 0
    for rec in _pair_records(records, agent, opponent):
        a, b = rec["agent_a"], rec["agent_b"]
        score_a = float(rec["score_a"])
        if a == agent:
            wins += score_a
        else:
            wins += 1.0 - score_a
        games += 1
    return wins / games if games else 0.0


def _survival_for(agent: str, opponent: str, records: List[dict]) -> List[float]:
    values = []
    for rec in _pair_records(records, agent, opponent):
        if rec["agent_a"] == agent:
            values.append(float(rec["turns_survived_a"]))
        else:
            values.append(float(rec["turns_survived_b"]))
    return values


def _final_lengths_for(agent: str, opponent: str, records: List[dict]) -> List[float]:
    values = []
    for rec in _pair_records(records, agent, opponent):
        if rec["agent_a"] == agent:
            values.append(float(rec["final_length_a"]))
        else:
            values.append(float(rec["final_length_b"]))
    return values


def _savefig(fig, out_path: str) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {out_path}")


def write_report_matchup_database(records: List[dict], out_dir: str) -> str:
    out_path = os.path.join(out_dir, "report_matchup_database.csv")
    by_pair = defaultdict(lambda: {"games": 0, "wins_heuristic": 0.0, "turns": [], "heur_survival": [], "opp_survival": [], "heur_length": [], "opp_length": []})
    for rec in records:
        a, b = rec["agent_a"], rec["agent_b"]
        if "Heuristic" not in {a, b}:
            continue
        opponent = b if a == "Heuristic" else a
        row = by_pair[("Heuristic", opponent)]
        row["games"] += 1
        if a == "Heuristic":
            row["wins_heuristic"] += float(rec["score_a"])
            row["heur_survival"].append(float(rec["turns_survived_a"]))
            row["opp_survival"].append(float(rec["turns_survived_b"]))
            row["heur_length"].append(float(rec["final_length_a"]))
            row["opp_length"].append(float(rec["final_length_b"]))
        else:
            row["wins_heuristic"] += 1.0 - float(rec["score_a"])
            row["heur_survival"].append(float(rec["turns_survived_b"]))
            row["opp_survival"].append(float(rec["turns_survived_a"]))
            row["heur_length"].append(float(rec["final_length_b"]))
            row["opp_length"].append(float(rec["final_length_a"]))
        row["turns"].append(float(rec["turns"]))

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "anchor_agent",
                "opponent",
                "games",
                "heuristic_win_rate",
                "opponent_win_rate",
                "avg_turns",
                "avg_heuristic_survival",
                "avg_opponent_survival",
                "avg_heuristic_final_length",
                "avg_opponent_final_length",
            ],
        )
        writer.writeheader()
        for (_anchor, opponent), row in sorted(by_pair.items()):
            games = row["games"]
            heur_wr = row["wins_heuristic"] / games if games else 0.0
            writer.writerow({
                "anchor_agent": "Heuristic",
                "opponent": opponent,
                "games": games,
                "heuristic_win_rate": round(heur_wr, 4),
                "opponent_win_rate": round(1.0 - heur_wr, 4),
                "avg_turns": round(sum(row["turns"]) / games, 3),
                "avg_heuristic_survival": round(sum(row["heur_survival"]) / games, 3),
                "avg_opponent_survival": round(sum(row["opp_survival"]) / games, 3),
                "avg_heuristic_final_length": round(sum(row["heur_length"]) / games, 3),
                "avg_opponent_final_length": round(sum(row["opp_length"]) / games, 3),
            })
    print(f"[saved] {out_path}")
    return out_path


def plot_heuristic_vs_random_winrate(records: List[dict], out_dir: str) -> None:
    wr_h = _win_rate_for("Heuristic", "Random", records)
    wr_r = 1.0 - wr_h
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Heuristic", "Random"], [wr_h, wr_r], color=[_color("Heuristic"), _color("Random")], alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Rate")
    ax.set_title("Heuristic vs Random")
    for idx, value in enumerate([wr_h, wr_r]):
        ax.text(idx, value + 0.02, f"{value:.1%}", ha="center")
    _savefig(fig, os.path.join(out_dir, "01_heuristic_vs_random_winrate.png"))


def plot_heuristic_vs_random_survival(records: List[dict], out_dir: str) -> None:
    data = [
        _survival_for("Heuristic", "Random", records),
        _survival_for("Random", "Heuristic", records),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, patch_artist=True, labels=["Heuristic", "Random"])
    for patch, label in zip(bp["boxes"], ["Heuristic", "Random"]):
        patch.set_facecolor(_color(label))
        patch.set_alpha(0.75)
    ax.set_ylabel("Turns Survived")
    ax.set_title("Heuristic vs Random Survival")
    _savefig(fig, os.path.join(out_dir, "02_heuristic_vs_random_survival.png"))


def plot_rollout_vs_vanilla_vs_heuristic(records: List[dict], out_dir: str) -> None:
    agents = ["MCTS-Vanilla", "MCTS-HeurRollout"]
    win_rates = [_win_rate_for(agent, "Heuristic", records) for agent in agents]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(agents, win_rates, color=[_color(a) for a in agents], alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Rate vs Heuristic")
    ax.set_title("Effect of Heuristic Rollouts")
    ax.tick_params(axis="x", rotation=15)
    for idx, value in enumerate(win_rates):
        ax.text(idx, value + 0.02, f"{value:.1%}", ha="center")
    _savefig(fig, os.path.join(out_dir, "03_rollout_vs_vanilla_vs_heuristic.png"))


def plot_mcts_improvement_ladder(records: List[dict], out_dir: str) -> None:
    labels = ["Vanilla", "+ Heuristic Rollouts", "+ Priors", "+ Opponent Model", "+ UCB1-Tuned", "All"]
    agents = ["MCTS-Vanilla", "MCTS-HeurRollout", "MCTS-Prior", "MCTS-Opponent", "MCTS-UCB1Tuned", "MCTS-All"]
    values = [_win_rate_for(agent, "Heuristic", records) for agent in agents]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(range(len(labels)), values, marker="o", linewidth=2.5, color="#111111")
    for idx, (label, agent, value) in enumerate(zip(labels, agents, values)):
        ax.scatter(idx, value, s=110, color=_color(agent), zorder=3)
        ax.text(idx, value + 0.03, f"{value:.1%}", ha="center", fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Rate vs Heuristic")
    ax.set_title("MCTS Improvement Ladder Against Heuristic")
    _savefig(fig, os.path.join(out_dir, "04_mcts_improvement_ladder_vs_heuristic.png"))


def plot_tournament_win_matrix(records: List[dict], out_dir: str) -> None:
    agents = sorted({rec["agent_a"] for rec in records} | {rec["agent_b"] for rec in records})
    idx = {a: i for i, a in enumerate(agents)}
    wins = defaultdict(float)
    games = defaultdict(int)
    for rec in records:
        a, b = rec["agent_a"], rec["agent_b"]
        score_a = float(rec["score_a"])
        wins[(a, b)] += score_a
        wins[(b, a)] += 1.0 - score_a
        games[(a, b)] += 1
        games[(b, a)] += 1

    mat = np.full((len(agents), len(agents)), np.nan)
    for a in agents:
        for b in agents:
            if a == b:
                continue
            if games[(a, b)]:
                mat[idx[a], idx[b]] = wins[(a, b)] / games[(a, b)]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    masked = np.ma.masked_invalid(mat)
    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad("#dddddd")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(agents)))
    ax.set_yticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=25, ha="right")
    ax.set_yticklabels(agents)
    ax.set_title("Tournament Pairwise Win Matrix")
    for i in range(len(agents)):
        for j in range(len(agents)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Win Rate")
    _savefig(fig, os.path.join(out_dir, "05_tournament_win_matrix.png"))


def plot_tournament_ratings(ratings: List[dict], out_dir: str) -> None:
    ratings = sorted(ratings, key=lambda row: float(row["elo"]), reverse=True)
    agents = [row["agent"] for row in ratings]
    elos = [float(row["elo"]) for row in ratings]
    ts = [float(row["ts_conservative"]) for row in ratings]
    colors = [_color(a) for a in agents]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.barh(agents, elos, color=colors, alpha=0.85)
    ax1.axvline(1500, color="gray", linestyle="--", linewidth=1)
    ax1.invert_yaxis()
    ax1.set_title("Tournament ELO")
    ax1.set_xlabel("ELO")
    ax2.barh(agents, ts, color=colors, alpha=0.85)
    ax2.invert_yaxis()
    ax2.set_title("Tournament TrueSkill Conservative")
    ax2.set_xlabel("mu - 3 sigma")
    _savefig(fig, os.path.join(out_dir, "06_tournament_ratings.png"))


def plot_tournament_survival(records: List[dict], out_dir: str) -> None:
    survival = defaultdict(list)
    for rec in records:
        survival[rec["agent_a"]].append(float(rec["turns_survived_a"]))
        survival[rec["agent_b"]].append(float(rec["turns_survived_b"]))
    agents = sorted(survival.keys(), key=lambda a: -np.mean(survival[a]))
    data = [survival[a] for a in agents]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bp = ax.boxplot(data, patch_artist=True, labels=agents)
    for patch, agent in zip(bp["boxes"], agents):
        patch.set_facecolor(_color(agent))
        patch.set_alpha(0.75)
    ax.set_ylabel("Turns Survived")
    ax.set_title("Tournament Survival Distribution")
    ax.tick_params(axis="x", rotation=20)
    _savefig(fig, os.path.join(out_dir, "07_tournament_survival_boxplot.png"))


def plot_hyper_sweep(summary_csv: str, title: str, out_path: str) -> None:
    if not os.path.exists(summary_csv):
        print(f"[WARN] Missing sweep summary: {summary_csv}")
        return
    rows = _load_csv(summary_csv)
    rows.sort(key=lambda row: float(row["parameter_value"]))
    x = [float(row["parameter_value"]) for row in rows]
    y1 = [float(row["win_rate"]) for row in rows]
    y2 = [float(row.get("vs_heuristic_win_rate", 0.0)) for row in rows]
    y3 = [float(row["ts_conservative"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(x, y1, marker="o", linewidth=2.3, label="Overall Win Rate")
    axes[0].plot(x, y2, marker="s", linewidth=2.0, label="Win Rate vs Heuristic")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(title)
    axes[0].set_xlabel(rows[0]["parameter_name"])
    axes[0].set_ylabel("Win Rate")
    axes[0].legend()
    axes[1].plot(x, y3, marker="o", linewidth=2.3, color="#222222")
    axes[1].set_title(f"{title} (TrueSkill)")
    axes[1].set_xlabel(rows[0]["parameter_name"])
    axes[1].set_ylabel("TrueSkill Conservative")
    _savefig(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 11 report plots from tournament, vs-heuristic, and hyper-sweep results")
    parser.add_argument("--tournament-dir", default="small_study_40_seeds")
    parser.add_argument("--vs-heuristic-dir", default="experiment_runs/vs_heuristic")
    parser.add_argument("--experiment-base-dir", default="experiment_runs")
    parser.add_argument("--out-dir", default="report_plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tournament_records = _load_csv(os.path.join(args.tournament_dir, "game_records.csv"))
    tournament_ratings = _load_csv(os.path.join(args.tournament_dir, "ratings.csv"))
    vs_heuristic_records = _load_csv(os.path.join(args.vs_heuristic_dir, "game_records.csv"))

    write_report_matchup_database(vs_heuristic_records, args.out_dir)

    plot_heuristic_vs_random_winrate(vs_heuristic_records, args.out_dir)
    plot_heuristic_vs_random_survival(vs_heuristic_records, args.out_dir)
    plot_rollout_vs_vanilla_vs_heuristic(vs_heuristic_records, args.out_dir)
    plot_mcts_improvement_ladder(vs_heuristic_records, args.out_dir)
    plot_tournament_win_matrix(tournament_records, args.out_dir)
    plot_tournament_ratings(tournament_ratings, args.out_dir)
    plot_tournament_survival(tournament_records, args.out_dir)

    plot_hyper_sweep(
        os.path.join(args.experiment_base_dir, "hyper_iterations", "summary.csv"),
        "Iterations Sweep",
        os.path.join(args.out_dir, "08_hyper_iterations_effect.png"),
    )
    plot_hyper_sweep(
        os.path.join(args.experiment_base_dir, "hyper_rollout_depth", "summary.csv"),
        "Rollout Depth Sweep",
        os.path.join(args.out_dir, "09_hyper_rollout_depth_effect.png"),
    )
    plot_hyper_sweep(
        os.path.join(args.experiment_base_dir, "hyper_exploration", "summary.csv"),
        "Exploration Sweep",
        os.path.join(args.out_dir, "10_hyper_exploration_effect.png"),
    )
    plot_hyper_sweep(
        os.path.join(args.experiment_base_dir, "hyper_prior_bonus", "summary.csv"),
        "Prior Bonus Sweep",
        os.path.join(args.out_dir, "11_hyper_prior_bonus_effect.png"),
    )


if __name__ == "__main__":
    main()
