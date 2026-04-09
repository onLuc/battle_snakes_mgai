#!/usr/bin/env python3
"""
plot_all.py  --  Generate all report plots from BattleSnake experiment results.

Reads CSVs written by run_experiments.py and produces publication-quality figures:

  Tournament plots (results/tournament/plots/):
    - win_rates.png         : bar chart of overall win rates with ±1 SE
    - elo_ratings.png       : ELO + TrueSkill conservative side-by-side
    - survival_boxplot.png  : turns-survived distribution per agent
    - win_matrix.png        : pairwise win-rate heatmap
    - mcts_vs_heuristic.png : each MCTS variant vs Heuristic (key comparison)
    - elo_progression.png   : how ELO evolves game-by-game

  Per-sweep plots (results/hyper_<param>/):
    - <param>_sweep.png     : multi-panel sensitivity plot

  Combined overview (results/):
    - hyperparameter_overview.png : all 4 sweeps side-by-side (ELO + win rate)

Usage:
    python plot_all.py                            # default: reads from ./results/
    python plot_all.py --results-dir my_results   # custom directory
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive, works without a display
import matplotlib.pyplot as plt
import numpy as np

try:
    import trueskill
    _TRUESKILL_AVAILABLE = True
except ImportError:
    _TRUESKILL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Colour palette — consistent with agent identity across all plots
# ---------------------------------------------------------------------------
AGENT_COLORS: Dict[str, str] = {
    "Random":           "#888888",
    "Heuristic":        "#D97706",
    "MCTS-Vanilla":     "#3B82F6",
    "MCTS-HeurRollout": "#8B5CF6",
    "MCTS-Prior":       "#EC4899",
    "MCTS-Opponent":    "#F59E0B",
    "MCTS-RAVE":        "#10B981",
    "MCTS-UCB1Tuned":   "#EF4444",
    "MCTS-All":         "#22C55E",
}

TOURNAMENT_AGENTS = ["Random", "Heuristic", "MCTS-Vanilla", "MCTS-All"]
HEURISTIC_COMPARISON_AGENTS = [
    "MCTS-All",
    "MCTS-Opponent",
    "MCTS-Vanilla",
    "MCTS-HeurRollout",
]

matplotlib.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
})


def _color(agent: str) -> str:
    """Map an agent name to a colour, handling sweep-agent variants gracefully."""
    if agent.startswith("MCTS-All-"):
        return AGENT_COLORS["MCTS-All"]
    return AGENT_COLORS.get(agent, "#555555")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _filter_records_to_agents(records: List[dict], allowed_agents: List[str]) -> List[dict]:
    allowed = set(allowed_agents)
    return [
        rec for rec in records
        if rec["agent_a"] in allowed and rec["agent_b"] in allowed
    ]


def _filter_heuristic_matchups(records: List[dict], variants: List[str]) -> List[dict]:
    allowed = set(variants)
    filtered = []
    for rec in records:
        a, b = rec["agent_a"], rec["agent_b"]
        if a == "Heuristic" and b in allowed:
            filtered.append(rec)
        elif b == "Heuristic" and a in allowed:
            filtered.append(rec)
    return filtered


def _ensure_plots_dir(results_dir: str) -> str:
    d = os.path.join(results_dir, "plots")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Shared metric computations
# ---------------------------------------------------------------------------

def _compute_win_rates(records: List[dict]):
    """Return (win_rates, win_se, games) dicts keyed by agent name."""
    wins:  Dict[str, float] = defaultdict(float)
    games: Dict[str, int]   = defaultdict(int)
    for rec in records:
        a, b = rec["agent_a"], rec["agent_b"]
        score_a = float(rec["score_a"])
        games[a] += 1
        games[b] += 1
        if score_a == 1.0:
            wins[a] += 1.0
        elif score_a == 0.0:
            wins[b] += 1.0

    win_rates = {ag: wins[ag] / games[ag] for ag in games}
    win_se    = {
        ag: np.sqrt(win_rates[ag] * (1.0 - win_rates[ag]) / games[ag])
        for ag in games
    }
    return win_rates, win_se, dict(games)


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _elo_update(ea: float, eb: float, score_a: float, k: float = 32.0):
    exp_a = 1.0 / (1.0 + 10.0 ** ((eb - ea) / 400.0))
    return ea + k * (score_a - exp_a), eb + k * ((1 - score_a) - (1 - exp_a))


def _compute_ratings_rows(records: List[dict], agent_names: List[str]) -> List[dict]:
    elo: Dict[str, float] = {name: 1500.0 for name in agent_names}
    ts_env = trueskill.TrueSkill() if _TRUESKILL_AVAILABLE else None
    ts = {name: ts_env.create_rating() for name in agent_names} if ts_env else {}

    sorted_records = sorted(records, key=lambda r: (r["agent_a"], r["agent_b"], int(r["seed"])))
    for rec in sorted_records:
        a, b = rec["agent_a"], rec["agent_b"]
        if a not in elo or b not in elo:
            continue
        score_a = float(rec["score_a"])
        elo[a], elo[b] = _elo_update(elo[a], elo[b], score_a)

        if ts_env:
            if score_a == 1.0:
                ts[a], ts[b] = trueskill.rate_1vs1(ts[a], ts[b])
            elif score_a == 0.0:
                ts[b], ts[a] = trueskill.rate_1vs1(ts[b], ts[a])
            else:
                ts[a], ts[b] = trueskill.rate_1vs1(ts[a], ts[b], drawn=True)

    rows = []
    for name in sorted(agent_names, key=lambda n: -elo[n]):
        if ts_env:
            mu = ts[name].mu
            sigma = ts[name].sigma
            conservative = mu - 3.0 * sigma
        else:
            mu = sigma = conservative = 0.0
        rows.append({
            "agent": name,
            "elo": round(elo[name], 2),
            "ts_mu": round(mu, 4),
            "ts_sigma": round(sigma, 4),
            "ts_conservative": round(conservative, 4),
        })
    return rows


def _write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_matchup_summary(path: str, records: List[dict]) -> None:
    summary: Dict[Tuple[str, str], dict] = defaultdict(lambda: {
        "games": 0, "wins_a": 0.0,
        "turns": [], "survival_a": [], "survival_b": [],
        "length_a": [], "length_b": [],
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

    rows = []
    for (agent_a, agent_b), row in sorted(summary.items()):
        games = row["games"]
        wins_a = row["wins_a"]
        rows.append({
            "agent_a": agent_a,
            "agent_b": agent_b,
            "games": games,
            "wins_a": round(wins_a, 3),
            "wins_b": round(games - wins_a, 3),
            "win_rate_a": round(wins_a / games, 4) if games else 0.0,
            "win_rate_b": round((games - wins_a) / games, 4) if games else 0.0,
            "avg_turns": round(sum(row["turns"]) / games, 3) if games else 0.0,
            "avg_survival_a": round(sum(row["survival_a"]) / games, 3) if games else 0.0,
            "avg_survival_b": round(sum(row["survival_b"]) / games, 3) if games else 0.0,
            "avg_final_length_a": round(sum(row["length_a"]) / games, 3) if games else 0.0,
            "avg_final_length_b": round(sum(row["length_b"]) / games, 3) if games else 0.0,
        })
    _write_csv(path, rows, [
        "agent_a", "agent_b", "games",
        "wins_a", "wins_b", "win_rate_a", "win_rate_b",
        "avg_turns", "avg_survival_a", "avg_survival_b",
        "avg_final_length_a", "avg_final_length_b",
    ])


# ---------------------------------------------------------------------------
# Plot 1: Win rate bar chart
# ---------------------------------------------------------------------------

def plot_win_rates(records: List[dict], plots_dir: str, title_suffix: str = "") -> None:
    wr, se, games = _compute_win_rates(records)
    agents = sorted(wr, key=lambda a: -wr[a])
    rates  = [wr[a]     for a in agents]
    ses    = [se[a]     for a in agents]
    colors = [_color(a) for a in agents]

    fig, ax = plt.subplots(figsize=(max(8, len(agents) * 1.4), 5))
    bars = ax.bar(agents, rates, color=colors, yerr=ses, capsize=4, alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="50% chance line")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Win Rate")
    ax.set_title(f"Agent Win Rates (±1 SE){title_suffix}")
    ax.tick_params(axis="x", rotation=35)
    for bar, rate, n in zip(bars, rates, [games[a] for a in agents]):
        lo, hi = _wilson_ci(int(rate * n), n)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ses[agents.index(bar.get_x() + bar.get_width() / 2)] + 0.02
            if False else bar.get_height() + 0.02,
            f"{rate:.1%}\n(n={n})",
            ha="center", va="bottom", fontsize=8,
        )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(plots_dir, "win_rates.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# Plot 2: ELO + TrueSkill ratings
# ---------------------------------------------------------------------------

def plot_elo_ratings(ratings_rows: List[dict], plots_dir: str) -> None:
    if not ratings_rows:
        return

    agents  = [r["agent"]                        for r in ratings_rows]
    elos    = [float(r["elo"])                   for r in ratings_rows]
    ts_cons = [float(r.get("ts_conservative", 0)) for r in ratings_rows]

    order    = sorted(range(len(agents)), key=lambda i: -elos[i])
    agents_s = [agents[i] for i in order]
    elos_s   = [elos[i]   for i in order]
    ts_s     = [ts_cons[i] for i in order]
    colors   = [_color(a) for a in agents_s]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(agents_s) * 0.7 + 1.5)))

    ax1.barh(agents_s, elos_s, color=colors, alpha=0.85)
    ax1.axvline(1500, color="gray", linestyle="--", linewidth=1.2, label="ELO=1500")
    ax1.set_xlabel("ELO Rating")
    ax1.set_title("ELO Ratings (sorted)")
    ax1.legend(fontsize=9)
    ax1.invert_yaxis()
    for i, (name, val) in enumerate(zip(agents_s, elos_s)):
        ax1.text(val + 5, i, f"{val:.0f}", va="center", fontsize=9)

    ax2.barh(agents_s, ts_s, color=colors, alpha=0.85)
    ax2.axvline(0, color="gray", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("TrueSkill  μ − 3σ  (conservative score)")
    ax2.set_title("TrueSkill Conservative Score")
    ax2.invert_yaxis()
    for i, (name, val) in enumerate(zip(agents_s, ts_s)):
        ax2.text(val + 0.1, i, f"{val:.2f}", va="center", fontsize=9)

    fig.suptitle("Agent Skill Ratings", fontsize=13, y=1.01)
    fig.tight_layout()
    out = os.path.join(plots_dir, "elo_ratings.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# Plot 3: Turns-survived box plot
# ---------------------------------------------------------------------------

def plot_survival_boxplot(records: List[dict], plots_dir: str) -> None:
    survival: Dict[str, List[float]] = defaultdict(list)
    for rec in records:
        survival[rec["agent_a"]].append(float(rec["turns_survived_a"]))
        survival[rec["agent_b"]].append(float(rec["turns_survived_b"]))
    if not survival:
        return

    agents = sorted(survival, key=lambda a: -np.mean(survival[a]))
    data   = [survival[a] for a in agents]
    means  = [np.mean(d) for d in data]
    colors = [_color(a) for a in agents]

    fig, ax = plt.subplots(figsize=(max(8, len(agents) * 1.4), 5))
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2},
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    for patch, agent in zip(bp["boxes"], agents):
        patch.set_facecolor(_color(agent))
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(agents) + 1))
    ax.set_xticklabels(agents, rotation=35, ha="right")
    ax.set_ylabel("Turns Survived")
    ax.set_title("Turns Survived Distribution per Agent (sorted by mean)")

    for i, mean in enumerate(means, start=1):
        ax.text(i, mean + 1.5, f"μ={mean:.0f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = os.path.join(plots_dir, "survival_boxplot.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# Plot 4: Pairwise win-rate heatmap
# ---------------------------------------------------------------------------

def plot_win_matrix(records: List[dict], plots_dir: str) -> None:
    wr, _, games = _compute_win_rates(records)
    agents = sorted(games, key=lambda a: -wr.get(a, 0))
    n      = len(agents)
    idx    = {a: i for i, a in enumerate(agents)}

    win_count:  Dict[Tuple, float] = defaultdict(float)
    game_count: Dict[Tuple, int]   = defaultdict(int)
    for rec in records:
        a, b  = rec["agent_a"], rec["agent_b"]
        score = float(rec["score_a"])
        win_count[(a, b)]  += score
        win_count[(b, a)]  += 1.0 - score
        game_count[(a, b)] += 1
        game_count[(b, a)] += 1

    matrix = np.full((n, n), np.nan)
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i == j:
                continue
            cnt = game_count.get((a, b), 0)
            if cnt > 0:
                matrix[i, j] = win_count[(a, b)] / cnt

    fig, ax = plt.subplots(figsize=(max(7, n * 1.15), max(5, n * 0.95)))
    masked = np.ma.masked_invalid(matrix)
    cmap   = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#e8e8e8")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agents, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(agents, fontsize=9)
    ax.set_title("Pairwise Win Rate  (row agent vs column agent)")
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Agent")

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val):
                txt_color = "black" if 0.3 < val < 0.7 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=txt_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Win Rate", shrink=0.8)
    fig.tight_layout()
    out = os.path.join(plots_dir, "win_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# Plot 5: MCTS variants vs Heuristic (key assignment comparison)
# ---------------------------------------------------------------------------

def plot_mcts_vs_heuristic(records: List[dict], plots_dir: str) -> None:
    """
    For each MCTS agent, show win rate vs Heuristic with 95% Wilson CI.
    Answers: 'What do you notice compared to the Heuristic baseline?'
    """
    records = _filter_heuristic_matchups(records, HEURISTIC_COMPARISON_AGENTS)
    all_agents = {rec["agent_a"] for rec in records} | {rec["agent_b"] for rec in records}
    mcts_agents = [a for a in HEURISTIC_COMPARISON_AGENTS if a in all_agents]
    if not mcts_agents:
        print("  [SKIP] mcts_vs_heuristic — no MCTS agents found")
        return

    results = {}
    for agent in mcts_agents:
        wins, total = 0, 0
        for rec in records:
            if rec["agent_a"] == agent and rec["agent_b"] == "Heuristic":
                total += 1
                if float(rec["score_a"]) == 1.0:
                    wins += 1
            elif rec["agent_b"] == agent and rec["agent_a"] == "Heuristic":
                total += 1
                if float(rec["score_a"]) == 0.0:
                    wins += 1
        if total > 0:
            results[agent] = (wins, total)

    if not results:
        print("  [SKIP] mcts_vs_heuristic — no Heuristic matchups found")
        return

    agents_sorted = [a for a in HEURISTIC_COMPARISON_AGENTS if a in results]
    rates  = [results[a][0] / results[a][1]  for a in agents_sorted]
    counts = [results[a][1]                  for a in agents_sorted]
    lo_ci  = [_wilson_ci(results[a][0], results[a][1])[0] for a in agents_sorted]
    hi_ci  = [_wilson_ci(results[a][0], results[a][1])[1] for a in agents_sorted]
    yerr_lo = [r - lo for r, lo in zip(rates, lo_ci)]
    yerr_hi = [hi - r for r, hi in zip(rates, hi_ci)]
    colors  = [_color(a) for a in agents_sorted]

    fig, ax = plt.subplots(figsize=(max(8, len(agents_sorted) * 1.5), 5))
    bars = ax.bar(agents_sorted, rates, color=colors,
                  yerr=[yerr_lo, yerr_hi], capsize=5, alpha=0.85,
                  error_kw={"elinewidth": 1.5})
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, label="50% (even matchup)")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Win Rate vs Heuristic")
    ax.set_title("Selected MCTS Variants vs Heuristic  (95% Wilson CI)")
    ax.tick_params(axis="x", rotation=35)
    for bar, rate, n in zip(bars, rates, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            f"{rate:.1%}\n(n={n})",
            ha="center", va="bottom", fontsize=8,
        )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(plots_dir, "mcts_vs_heuristic.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# Plot 6: ELO progression over game sequence
# ---------------------------------------------------------------------------

def plot_elo_progression(records: List[dict], plots_dir: str) -> None:
    """Show how ELO evolves as games are played, for key matchups."""
    preference = [
        ("MCTS-All",  "Heuristic"),
        ("MCTS-All",  "MCTS-Vanilla"),
        ("Heuristic", "MCTS-Vanilla"),
        ("Heuristic", "Random"),
        ("MCTS-All",  "Random"),
        ("MCTS-Vanilla", "Random"),
    ]
    existing = {(rec["agent_a"], rec["agent_b"]) for rec in records}
    selected = [p for p in preference if p in existing][:4]

    if not selected:
        return

    rows = max(1, (len(selected) + 1) // 2)
    cols = min(2, len(selected))
    fig, axes = plt.subplots(rows, cols, figsize=(13, rows * 4.5))
    axes_flat = np.array(axes).flatten()

    for ax_idx, (a, b) in enumerate(selected):
        ax = axes_flat[ax_idx]
        ea, eb = 1500.0, 1500.0
        hist_a = [ea]
        hist_b = [eb]
        for rec in records:
            if rec["agent_a"] == a and rec["agent_b"] == b:
                ea, eb = _elo_update(ea, eb, float(rec["score_a"]))
                hist_a.append(ea)
                hist_b.append(eb)

        x = list(range(len(hist_a)))
        ax.plot(x, hist_a, color=_color(a), linewidth=2.0, label=a)
        ax.plot(x, hist_b, color=_color(b), linewidth=2.0, linestyle="--", label=b)
        ax.axhline(1500, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(f"{a}  vs  {b}")
        ax.set_xlabel("Game #")
        ax.set_ylabel("ELO")
        ax.legend(fontsize=8)

    for i in range(len(selected), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("ELO Progression by Matchup", fontsize=13)
    fig.tight_layout()
    out = os.path.join(plots_dir, "elo_progression.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# Hyperparameter sweep: per-family multi-panel plot
# ---------------------------------------------------------------------------

def plot_sweep_family(family_dir: str, param_name: str) -> None:
    """
    Generate a multi-panel sensitivity plot for one hyperparameter family.
    Panels: overall win rate, ELO, TrueSkill, avg survival,
            win rate vs Heuristic, win rate vs MCTS-Vanilla.
    """
    summary_path = os.path.join(family_dir, "summary.csv")
    if not os.path.exists(summary_path):
        print(f"  [SKIP] No summary.csv in {family_dir}")
        return

    rows = _load_csv(summary_path)
    if not rows:
        return

    x = [float(r["parameter_value"]) for r in rows]
    labels = [r.get("run_dir", str(v)) for r, v in zip(rows, x)]

    panels = [
        ("win_rate",                    "Overall Win Rate",          True),
        ("elo",                         "ELO Rating",                False),
        ("ts_conservative",             "TrueSkill Conservative",    False),
        ("avg_survival",                "Avg Survival (turns)",      True),
        ("vs_heuristic_win_rate",       "Win Rate vs Heuristic",     True),
        ("vs_mcts_vanilla_win_rate",    "Win Rate vs MCTS-Vanilla",  True),
    ]
    available = [(k, t, is_rate) for k, t, is_rate in panels if k in rows[0]]
    ncols = 3
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes_flat = np.array(axes).flatten()

    for ax, (key, title, is_rate) in zip(axes_flat, available):
        y = [float(r.get(key, 0.0)) for r in rows]
        best_idx = int(np.argmax(y))

        ax.plot(x, y, marker="o", linewidth=2.0, color="#22C55E", zorder=3)
        ax.scatter([x[best_idx]], [y[best_idx]], s=120, color="#DC2626",
                   zorder=4, label=f"best={x[best_idx]}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(param_name)
        if is_rate:
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Annotate each point with its value
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.3f}", xy=(xi, yi),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7)

    for i in range(len(available), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(f"Hyperparameter Sensitivity: {param_name}", fontsize=14)
    fig.tight_layout()
    out = os.path.join(family_dir, f"{param_name}_sweep.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# Combined overview: all 4 sweeps in one figure
# ---------------------------------------------------------------------------

def plot_hyperparameter_overview(base_dir: str, sweeps: List[Tuple[str, str]]) -> None:
    """
    2×2 grid: one panel per parameter family.
    Each panel shows ELO (left axis) and Win Rate (right axis) vs the parameter.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for ax_idx, (family, param) in enumerate(sweeps):
        ax = axes_flat[ax_idx]
        summary_path = os.path.join(base_dir, family, "summary.csv")
        if not os.path.exists(summary_path):
            ax.set_title(f"{param}  (no data)")
            ax.set_visible(True)
            continue

        rows = _load_csv(summary_path)
        if not rows:
            continue

        x      = [float(r["parameter_value"]) for r in rows]
        y_elo  = [float(r.get("elo", 1500))   for r in rows]
        y_wr   = [float(r.get("win_rate", 0)) for r in rows]

        ax2 = ax.twinx()
        l1, = ax.plot(x, y_elo, marker="o", color="#3B82F6",
                      linewidth=2, label="ELO")
        l2, = ax2.plot(x, y_wr, marker="s", color="#22C55E",
                       linewidth=2, linestyle="--", label="Win Rate")
        ax.set_xlabel(param)
        ax.set_ylabel("ELO", color="#3B82F6")
        ax2.set_ylabel("Win Rate", color="#22C55E")
        ax2.set_ylim(0, 1)
        ax.set_title(f"{param}")
        ax.legend(handles=[l1, l2], fontsize=8, loc="best")
        ax.grid(alpha=0.3)

        # Mark best by ELO
        best = x[int(np.argmax(y_elo))]
        ax.axvline(best, color="#3B82F6", linewidth=0.8, linestyle=":",
                   alpha=0.6)
        ax.text(best, min(y_elo) * 0.99, f"best\n{best}", ha="center",
                va="top", fontsize=7, color="#3B82F6")

    fig.suptitle("Hyperparameter Sensitivity Overview\n"
                 "(ELO = blue solid, Win Rate = green dashed)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(base_dir, "hyperparameter_overview.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  [saved] {out}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

SWEEPS_CONFIG = [
    ("hyper_iterations",   "iterations"),
    ("hyper_rollout_depth","rollout_depth"),
    ("hyper_exploration",  "exploration"),
    ("hyper_prior_bonus",  "prior_bonus_scale"),
]


def plot_tournament_section(tournament_dir: str) -> None:
    records_path = os.path.join(tournament_dir, "game_records.csv")
    records = _load_csv(records_path)

    if not records:
        print(f"  [WARN] No tournament game records at {records_path}")
        print("         Run:  python run_experiments.py --mode tournament")
        return

    core_records = _filter_records_to_agents(records, TOURNAMENT_AGENTS)
    if not core_records:
        print(f"  [WARN] No core tournament records found in {records_path}")
        return

    ratings = _compute_ratings_rows(core_records, TOURNAMENT_AGENTS)

    plots_dir = _ensure_plots_dir(tournament_dir)
    n_records = len(core_records)
    n_agents  = len({rec["agent_a"] for rec in core_records} | {rec["agent_b"] for rec in core_records})
    print(f"  Loaded {n_records} core tournament records, {n_agents} agents")
    print(f"  Plots → {plots_dir}/")

    _write_csv(
        os.path.join(tournament_dir, "ratings.csv"),
        ratings,
        ["agent", "elo", "ts_mu", "ts_sigma", "ts_conservative"],
    )
    _write_matchup_summary(os.path.join(tournament_dir, "matchup_summary.csv"), core_records)

    plot_win_rates(core_records,       plots_dir, " — Tournament")
    plot_elo_ratings(ratings,     plots_dir)
    plot_survival_boxplot(core_records, plots_dir)
    plot_win_matrix(core_records,      plots_dir)
    plot_mcts_vs_heuristic(records, plots_dir)
    plot_elo_progression(core_records,  plots_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all BattleSnake experiment plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Run run_experiments.py first to generate the CSVs.",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Base results directory (same value used for run_experiments.py).",
    )
    args = parser.parse_args()

    base_dir = os.path.join(os.getcwd(), args.results_dir)

    if not os.path.exists(base_dir):
        print(f"\n[ERROR] Results directory not found:  {base_dir}")
        print("  Run run_experiments.py first to generate CSVs.")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  BattleSnake Plot Generator")
    print(f"  results dir: {base_dir}")
    print("=" * 65)

    # ── Tournament plots ─────────────────────────────────────────────────────
    tournament_dir = os.path.join(base_dir, "tournament")
    if os.path.isdir(tournament_dir):
        print("\n=== Tournament Plots ===")
        plot_tournament_section(tournament_dir)
    else:
        print(f"\n[WARN] Tournament directory not found: {tournament_dir}")

    # ── Per-sweep plots ───────────────────────────────────────────────────────
    any_sweep = False
    for family, param in SWEEPS_CONFIG:
        family_dir = os.path.join(base_dir, family)
        if os.path.isdir(family_dir):
            print(f"\n=== Sweep: {param} ===")
            plot_sweep_family(family_dir, param)
            any_sweep = True
        else:
            print(f"  [SKIP] {family_dir} not found")

    # ── Combined hyperparameter overview ─────────────────────────────────────
    if any_sweep:
        print("\n=== Hyperparameter Overview ===")
        plot_hyperparameter_overview(base_dir, SWEEPS_CONFIG)

    print("\n" + "=" * 65)
    print("  All plots generated.")
    print(f"  Tournament plots : {os.path.join(base_dir, 'tournament', 'plots')}/")
    print(f"  Sweep plots      : {base_dir}/hyper_*/")
    print(f"  Overview         : {os.path.join(base_dir, 'hyperparameter_overview.png')}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
