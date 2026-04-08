"""
Ablation study plot generator.

Reads CSVs from RESULTS_DIR and saves plots to RESULTS_DIR/plots/.
Does NOT import any game code — only stdlib + numpy + matplotlib.

Usage:
    python plot_results.py
"""

import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

RESULTS_DIR = os.environ.get("RESULTS_DIR", "tournament_results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

AGENT_COLORS: Dict[str, str] = {
    "Random":           "#888888",
    "Heuristic":        "#D97706",
    "MCTS-Vanilla":     "#3B82F6",
    "MCTS-HeurRollout": "#8B5CF6",
    "MCTS-Prior":       "#EC4899",
    "MCTS-Opponent":    "#F59E0B",
    "MCTS-RAVE":        "#10B981",
    "MCTS-UCB1Tuned":   "#EF4444",
    "MCTS-All":         "#00CC00",
}

# Clean matplotlib style
matplotlib.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def _color(agent: str) -> str:
    return AGENT_COLORS.get(agent, "#555555")


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def _load_records(path: str) -> List[dict]:
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_ratings(path: str) -> List[dict]:
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Helper: win rate per agent
# ---------------------------------------------------------------------------

def _compute_win_rates(records: List[dict]):
    wins: Dict[str, int] = defaultdict(int)
    games: Dict[str, int] = defaultdict(int)
    for rec in records:
        a = rec["agent_a"]
        b = rec["agent_b"]
        score_a = float(rec["score_a"])
        games[a] += 1
        games[b] += 1
        if score_a == 1.0:
            wins[a] += 1
        elif score_a == 0.0:
            wins[b] += 1
        else:
            wins[a] += 0  # draw — 0 win credit (but counted in games)
    all_agents = sorted(games.keys())
    win_rates = {}
    win_se = {}
    for ag in all_agents:
        n = games[ag]
        w = wins[ag]
        p = w / n if n > 0 else 0.0
        win_rates[ag] = p
        win_se[ag] = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
    return win_rates, win_se, games


# ---------------------------------------------------------------------------
# Plot 1: Win rates bar chart
# ---------------------------------------------------------------------------

def plot_win_rates(records: List[dict]) -> None:
    win_rates, win_se, games = _compute_win_rates(records)
    agents = sorted(win_rates.keys(), key=lambda a: -win_rates[a])
    rates = [win_rates[a] for a in agents]
    ses = [win_se[a] for a in agents]
    colors = [_color(a) for a in agents]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(agents, rates, color=colors, yerr=ses, capsize=4, alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="50%")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Rate")
    ax.set_title("Agent Win Rates (±1 SE)")
    ax.tick_params(axis="x", rotation=35)
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{rate:.1%}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.legend()
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "win_rates.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------------------------------------------------------------------
# Plot 2: ELO ratings + TrueSkill conservative score
# ---------------------------------------------------------------------------

def plot_elo_ratings(ratings: List[dict]) -> None:
    if not ratings:
        return
    agents = [r["agent"] for r in ratings]
    elos = [float(r["elo"]) for r in ratings]
    ts_cons = [float(r.get("ts_conservative", 0)) for r in ratings]

    # Sort descending by ELO
    order = sorted(range(len(agents)), key=lambda i: -elos[i])
    agents_s = [agents[i] for i in order]
    elos_s = [elos[i] for i in order]
    ts_s = [ts_cons[i] for i in order]
    colors = [_color(a) for a in agents_s]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ELO bar
    ax1.barh(agents_s, elos_s, color=colors, alpha=0.85)
    ax1.axvline(1500, color="gray", linestyle="--", linewidth=1.2, label="1500")
    ax1.set_xlabel("ELO Rating")
    ax1.set_title("ELO Ratings")
    ax1.legend()
    ax1.invert_yaxis()

    # TrueSkill conservative (μ − 3σ)
    ax2.barh(agents_s, ts_s, color=colors, alpha=0.85)
    ax2.axvline(0, color="gray", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("TrueSkill μ − 3σ")
    ax2.set_title("TrueSkill Conservative Score")
    ax2.invert_yaxis()

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "elo_ratings.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------------------------------------------------------------------
# Plot 3: Turns survived violin plot
# ---------------------------------------------------------------------------

def plot_turns_survived(records: List[dict]) -> None:
    survival: Dict[str, List[float]] = defaultdict(list)
    for rec in records:
        survival[rec["agent_a"]].append(float(rec["turns_survived_a"]))
        survival[rec["agent_b"]].append(float(rec["turns_survived_b"]))

    if not survival:
        return

    agents = sorted(survival.keys(), key=lambda a: -np.median(survival[a]))
    data = [survival[a] for a in agents]

    fig, ax = plt.subplots(figsize=(11, 5))
    parts = ax.violinplot(data, positions=range(len(agents)), showmedians=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(_color(agents[i]))
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels(agents, rotation=35, ha="right")
    ax.set_ylabel("Turns Survived")
    ax.set_title("Distribution of Turns Survived per Agent")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "turns_survived.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------------------------------------------------------------------
# Plot 4: Ablation ladder
# ---------------------------------------------------------------------------

_ABLATION_LADDER = [
    ("Heuristic Rollouts",   "MCTS-HeurRollout", "MCTS-Vanilla"),
    ("+ Prior Guidance",      "MCTS-Prior",        "MCTS-HeurRollout"),
    ("+ Opponent Modeling",   "MCTS-Opponent",     "MCTS-Prior"),
    ("+ RAVE",                "MCTS-RAVE",         "MCTS-Opponent"),
    ("+ UCB1-Tuned",          "MCTS-UCB1Tuned",    "MCTS-Opponent"),
    ("+ All",                 "MCTS-All",          "MCTS-Opponent"),
]


def _matchup_win_rate(records: List[dict], a: str, b: str) -> Optional[float]:
    """Win rate of a vs b (a as agent_a or agent_b)."""
    wins = 0
    total = 0
    for rec in records:
        if rec["agent_a"] == a and rec["agent_b"] == b:
            total += 1
            if float(rec["score_a"]) == 1.0:
                wins += 1
        elif rec["agent_a"] == b and rec["agent_b"] == a:
            total += 1
            if float(rec["score_a"]) == 0.0:
                wins += 1
    return wins / total if total > 0 else None


def plot_ablation_ladder(records: List[dict]) -> None:
    labels = []
    rates = []
    colors_list = []

    for label, agent, baseline in _ABLATION_LADDER:
        wr = _matchup_win_rate(records, agent, baseline)
        if wr is None:
            continue
        labels.append(label)
        rates.append(wr)
        colors_list.append("#10B981" if wr > 0.5 else "#EF4444")

    if not labels:
        print("[WARN] No ablation ladder data found.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, rates, color=colors_list, alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="50%")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Rate vs Baseline")
    ax.set_title("Ablation: Win Rate of Each Improvement over Baseline")
    ax.tick_params(axis="x", rotation=30)
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{rate:.1%}",
            ha="center", va="bottom", fontsize=10,
        )
    ax.legend()
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "ablation_ladder.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------------------------------------------------------------------
# Plot 5: Win matrix heatmap
# ---------------------------------------------------------------------------

def plot_win_matrix(records: List[dict]) -> None:
    win_rates, _, games = _compute_win_rates(records)
    agents = sorted(games.keys(), key=lambda a: -win_rates.get(a, 0))
    n = len(agents)
    idx = {a: i for i, a in enumerate(agents)}

    matrix = np.full((n, n), np.nan)

    for rec in records:
        a = rec["agent_a"]
        b = rec["agent_b"]
        score_a = float(rec["score_a"])
        ia, ib = idx[a], idx[b]
        if np.isnan(matrix[ia, ib]):
            matrix[ia, ib] = 0.0
        if np.isnan(matrix[ib, ia]):
            matrix[ib, ia] = 0.0
        matrix[ia, ib] += score_a
        matrix[ib, ia] += 1.0 - score_a

    # Normalize by number of records per pair
    pair_counts = defaultdict(int)
    for rec in records:
        pair_counts[(rec["agent_a"], rec["agent_b"])] += 1

    norm_matrix = np.full((n, n), np.nan)
    for (a, b), count in pair_counts.items():
        if count > 0:
            ia, ib = idx[a], idx[b]
            total_ab = pair_counts.get((a, b), 0)
            total_ba = pair_counts.get((b, a), 0)
            # Collect raw win counts then normalize
            pass

    # Simpler: accumulate wins then divide
    win_count = defaultdict(float)
    game_count = defaultdict(int)
    for rec in records:
        a = rec["agent_a"]
        b = rec["agent_b"]
        score_a = float(rec["score_a"])
        win_count[(a, b)] += score_a
        win_count[(b, a)] += 1.0 - score_a
        game_count[(a, b)] += 1
        game_count[(b, a)] += 1

    norm_matrix = np.full((n, n), np.nan)
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i == j:
                continue
            cnt = game_count.get((a, b), 0)
            if cnt > 0:
                norm_matrix[i, j] = win_count[(a, b)] / cnt

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), max(6, n * 0.9)))
    masked = np.ma.masked_invalid(norm_matrix)
    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agents, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(agents, fontsize=9)
    ax.set_title("Pairwise Win Rate (row beats column)")

    for i in range(n):
        for j in range(n):
            val = norm_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if 0.3 < val < 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Win Rate")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "win_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------------------------------------------------------------------
# Plot 6: ELO progression
# ---------------------------------------------------------------------------

def plot_elo_progression(records: List[dict]) -> None:
    # Prefer non-random matchups for the 4 interesting ones
    interesting_matchups_preference = [
        ("MCTS-All", "Heuristic"),
        ("MCTS-All", "MCTS-Vanilla"),
        ("MCTS-RAVE", "MCTS-Opponent"),
        ("MCTS-Opponent", "MCTS-Prior"),
        ("MCTS-Prior", "MCTS-HeurRollout"),
        ("MCTS-HeurRollout", "MCTS-Vanilla"),
        ("MCTS-All", "Random"),
        ("Heuristic", "Random"),
    ]

    # Find which matchups actually exist in records
    existing = set()
    for rec in records:
        existing.add((rec["agent_a"], rec["agent_b"]))

    selected = []
    for pair in interesting_matchups_preference:
        if pair in existing:
            selected.append(pair)
        if len(selected) >= 4:
            break

    if not selected:
        print("[WARN] Not enough matchups for ELO progression plot.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()

    for ax_idx, (a, b) in enumerate(selected):
        ax = axes_flat[ax_idx]
        elo_a, elo_b = 1500.0, 1500.0
        elo_hist_a = [elo_a]
        elo_hist_b = [elo_b]

        for rec in records:
            if rec["agent_a"] == a and rec["agent_b"] == b:
                score_a = float(rec["score_a"])
                elo_a, elo_b = _update_elo(elo_a, elo_b, score_a)
                elo_hist_a.append(elo_a)
                elo_hist_b.append(elo_b)

        ax.plot(elo_hist_a, color=_color(a), linewidth=2, label=a)
        ax.plot(elo_hist_b, color=_color(b), linewidth=2, label=b, linestyle="--")
        ax.axhline(1500, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{a} vs {b}")
        ax.set_xlabel("Game #")
        ax.set_ylabel("ELO")
        ax.legend(fontsize=8)

    # Hide unused subplots
    for i in range(len(selected), 4):
        axes_flat[i].set_visible(False)

    fig.suptitle("ELO Progression per Matchup", fontsize=13)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "elo_progression.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


def _update_elo(elo_a: float, elo_b: float, score_a: float, k: float = 32.0):
    expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    new_a = elo_a + k * (score_a - expected_a)
    new_b = elo_b + k * ((1 - score_a) - (1 - expected_a))
    return new_a, new_b


# ---------------------------------------------------------------------------
# Plot 7: Survival box plot
# ---------------------------------------------------------------------------

def plot_survival_boxplot(records: List[dict]) -> None:
    survival: Dict[str, List[float]] = defaultdict(list)
    for rec in records:
        survival[rec["agent_a"]].append(float(rec["turns_survived_a"]))
        survival[rec["agent_b"]].append(float(rec["turns_survived_b"]))

    if not survival:
        return

    agents = sorted(survival.keys(), key=lambda a: -np.mean(survival[a]))
    data = [survival[a] for a in agents]
    means = [np.mean(d) for d in data]

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black", "linewidth": 2})

    for patch, agent in zip(bp["boxes"], agents):
        patch.set_facecolor(_color(agent))
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(agents) + 1))
    ax.set_xticklabels(agents, rotation=35, ha="right")
    ax.set_ylabel("Turns Survived")
    ax.set_title("Turns Survived per Agent (sorted by mean)")

    for i, (mean, agent) in enumerate(zip(means, agents), start=1):
        ax.text(i, mean + 1, f"{mean:.1f}", ha="center", va="bottom", fontsize=8, color="black")

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "survival_boxplot.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[saved] {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    records_path = os.path.join(RESULTS_DIR, "game_records.csv")
    ratings_path = os.path.join(RESULTS_DIR, "ratings.csv")

    records = _load_records(records_path)
    ratings = _load_ratings(ratings_path)

    if not records:
        print("[ERROR] No game records found. Run tournament.py first.")
        sys.exit(1)

    print(f"Loaded {len(records)} game records, {len(ratings)} agent ratings.")
    print(f"Saving plots to {PLOTS_DIR}/")

    plot_win_rates(records)
    plot_elo_ratings(ratings)
    plot_turns_survived(records)
    plot_ablation_ladder(records)
    plot_win_matrix(records)
    plot_elo_progression(records)
    plot_survival_boxplot(records)

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
