"""
Ablation tournament: runs matchups, computes ELO + TrueSkill, saves CSV results.

Usage:
    python tournament.py [--quick]

Environment variables:
    NUM_SEEDS   — number of seeds per matchup (default 30)
    MCTS_BUDGET — time budget per MCTS move in seconds (default 0.10)
    MAX_TURNS   — max turns per game (default 200)
    RESULTS_DIR — output directory (default "tournament_results")
"""

import argparse
import csv
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

# MCTS imports (configs + move function)
from MCTS import (
    ALL_IMPROVEMENTS_CONFIG,
    HEURISTIC_ROLLOUT_CONFIG,
    OPPONENT_AWARE_CONFIG,
    PRIOR_GUIDED_CONFIG,
    RAVE_CONFIG,
    UCB1_TUNED_CONFIG,
    VANILLA_CONFIG,
    mcts_move,
)
from agent_core import (
    fallback_move,
    get_legal_moves,
    heuristic_move,
    make_state_from_game,
)
from simulate_game import run_game

# Optional TrueSkill
try:
    import trueskill

    _TRUESKILL_AVAILABLE = True
except ImportError:
    _TRUESKILL_AVAILABLE = False
    print("[WARNING] trueskill not installed — TrueSkill ratings will be skipped.")

# ---------------------------------------------------------------------------
# Settings from environment
# ---------------------------------------------------------------------------

NUM_SEEDS = int(os.environ.get("NUM_SEEDS", "30"))
MCTS_BUDGET = float(os.environ.get("MCTS_BUDGET", "0.10"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "200"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", "tournament_results")

# ---------------------------------------------------------------------------
# Default ablation matchups
# ---------------------------------------------------------------------------

DEFAULT_MATCHUPS = [
    ("MCTS-All", "Random"),
    ("MCTS-All", "Heuristic"),
    ("MCTS-All", "MCTS-Vanilla"),
    ("MCTS-HeurRollout", "MCTS-Vanilla"),    # isolate rollout improvement
    ("MCTS-Prior", "MCTS-HeurRollout"),       # isolate prior guidance
    ("MCTS-Opponent", "MCTS-Prior"),          # isolate opponent modeling
    ("MCTS-RAVE", "MCTS-Opponent"),           # isolate RAVE
    ("MCTS-UCB1Tuned", "MCTS-Opponent"),      # isolate UCB1-Tuned
    ("Heuristic", "Random"),
]

# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _random_agent() -> Callable:
    import random as _random
    from agent_core import get_legal_moves as _get_legal_moves, make_state_from_game as _make_state

    def agent(game_state: dict) -> str:
        state = _make_state(game_state)
        legal = _get_legal_moves(state, state["you_id"])
        return _random.choice(legal) if legal else "up"

    return agent


def _heuristic_agent() -> Callable:
    from agent_core import heuristic_move as _hm

    def agent(game_state: dict) -> str:
        move, _ = _hm(game_state)
        return move

    return agent


def _mcts_agent(config) -> Callable:
    from MCTS import mcts_move as _mm

    def agent(game_state: dict) -> str:
        move, _ = _mm(game_state, config=config, time_budget=MCTS_BUDGET)
        return move

    return agent


AGENT_REGISTRY: Dict[str, Callable[[], Callable]] = {
    "Random":           lambda: _random_agent(),
    "Heuristic":        lambda: _heuristic_agent(),
    "MCTS-Vanilla":     lambda: _mcts_agent(VANILLA_CONFIG),
    "MCTS-HeurRollout": lambda: _mcts_agent(HEURISTIC_ROLLOUT_CONFIG),
    "MCTS-Prior":       lambda: _mcts_agent(PRIOR_GUIDED_CONFIG),
    "MCTS-Opponent":    lambda: _mcts_agent(OPPONENT_AWARE_CONFIG),
    "MCTS-RAVE":        lambda: _mcts_agent(RAVE_CONFIG),
    "MCTS-UCB1Tuned":   lambda: _mcts_agent(UCB1_TUNED_CONFIG),
    "MCTS-All":         lambda: _mcts_agent(ALL_IMPROVEMENTS_CONFIG),
}


def make_agent(name: str) -> Callable:
    factory = AGENT_REGISTRY.get(name)
    if factory is None:
        raise ValueError(f"Unknown agent: {name!r}. Known: {list(AGENT_REGISTRY)}")
    return factory()


# ---------------------------------------------------------------------------
# ELO
# ---------------------------------------------------------------------------

def update_elo(
    elo_a: float, elo_b: float, score_a: float, k: float = 32.0
) -> Tuple[float, float]:
    """
    score_a = 1.0 (win), 0.5 (draw), 0.0 (loss).
    Returns (new_elo_a, new_elo_b).
    """
    expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a
    new_a = elo_a + k * (score_a - expected_a)
    new_b = elo_b + k * (score_b - expected_b)
    return new_a, new_b


# ---------------------------------------------------------------------------
# Matchup runner
# ---------------------------------------------------------------------------

def run_matchup(
    name_a: str,
    name_b: str,
    seeds: List[int],
) -> List[dict]:
    """Run one game per seed between name_a and name_b."""
    records = []
    agent_a = make_agent(name_a)
    agent_b = make_agent(name_b)

    for seed in seeds:
        print(f"  [{name_a} vs {name_b}] seed={seed} ...", end=" ", flush=True)

        result = run_game(
            agents=[(name_a, agent_a), (name_b, agent_b)],
            seed=seed,
            max_turns=MAX_TURNS,
        )

        res_map = {r["name"]: r for r in result["results"]}
        r_a = res_map.get(name_a, {})
        r_b = res_map.get(name_b, {})

        placement_a = r_a.get("placement", 2)
        placement_b = r_b.get("placement", 1)

        if placement_a < placement_b:
            score_a = 1.0
        elif placement_a == placement_b:
            score_a = 0.5
        else:
            score_a = 0.0

        record = {
            "seed": seed,
            "agent_a": name_a,
            "agent_b": name_b,
            "turns": result["turns"],
            "placement_a": placement_a,
            "placement_b": placement_b,
            "score_a": score_a,
            "turns_survived_a": r_a.get("turns_survived", 0),
            "turns_survived_b": r_b.get("turns_survived", 0),
            "final_length_a": r_a.get("final_length", 3),
            "final_length_b": r_b.get("final_length", 3),
        }
        records.append(record)
        outcome = "WIN" if score_a == 1.0 else ("DRAW" if score_a == 0.5 else "LOSS")
        print(f"{outcome} (turns={result['turns']})")

    return records


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------

def run_tournament(
    matchups: List[Tuple[str, str]],
    seeds: List[int],
) -> dict:
    """
    Run all matchups.  Returns dict with:
        records, elo, trueskill_mu, trueskill_sigma, trueskill_score
    """
    all_records = []

    # Collect all agent names
    all_names = set()
    for a, b in matchups:
        all_names.add(a)
        all_names.add(b)
    all_names = sorted(all_names)

    # ELO ratings
    elo: Dict[str, float] = {name: 1500.0 for name in all_names}

    # TrueSkill ratings
    ts_ratings: Dict[str, object] = {}
    if _TRUESKILL_AVAILABLE:
        ts_env = trueskill.TrueSkill()
        ts_ratings = {name: ts_env.create_rating() for name in all_names}

    for name_a, name_b in matchups:
        print(f"\n=== {name_a} vs {name_b} ===")
        records = run_matchup(name_a, name_b, seeds)
        all_records.extend(records)

        # Update ELO and TrueSkill after every game
        for rec in records:
            score_a = rec["score_a"]
            elo[name_a], elo[name_b] = update_elo(elo[name_a], elo[name_b], score_a)

            if _TRUESKILL_AVAILABLE:
                if score_a == 1.0:
                    ts_ratings[name_a], ts_ratings[name_b] = trueskill.rate_1vs1(
                        ts_ratings[name_a], ts_ratings[name_b]
                    )
                elif score_a == 0.0:
                    ts_ratings[name_b], ts_ratings[name_a] = trueskill.rate_1vs1(
                        ts_ratings[name_b], ts_ratings[name_a]
                    )
                else:
                    ts_ratings[name_a], ts_ratings[name_b] = trueskill.rate_1vs1(
                        ts_ratings[name_a], ts_ratings[name_b], drawn=True
                    )

    # Build TrueSkill summary
    ts_mu: Dict[str, float] = {}
    ts_sigma: Dict[str, float] = {}
    ts_score: Dict[str, float] = {}
    if _TRUESKILL_AVAILABLE:
        for name in all_names:
            r = ts_ratings[name]
            ts_mu[name] = r.mu
            ts_sigma[name] = r.sigma
            ts_score[name] = r.mu - 3.0 * r.sigma
    else:
        for name in all_names:
            ts_mu[name] = 0.0
            ts_sigma[name] = 0.0
            ts_score[name] = 0.0

    return {
        "records": all_records,
        "elo": elo,
        "trueskill_mu": ts_mu,
        "trueskill_sigma": ts_sigma,
        "trueskill_score": ts_score,
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # game_records.csv
    records = results["records"]
    if records:
        record_path = os.path.join(RESULTS_DIR, "game_records.csv")
        fieldnames = list(records[0].keys())
        with open(record_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"[saved] {record_path}")

    # ratings.csv
    elo = results["elo"]
    ts_mu = results["trueskill_mu"]
    ts_sigma = results["trueskill_sigma"]
    ts_score = results["trueskill_score"]

    all_names = sorted(elo.keys())
    ratings_path = os.path.join(RESULTS_DIR, "ratings.csv")
    with open(ratings_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["agent", "elo", "ts_mu", "ts_sigma", "ts_conservative"],
        )
        writer.writeheader()
        for name in all_names:
            writer.writerow({
                "agent": name,
                "elo": round(elo[name], 2),
                "ts_mu": round(ts_mu.get(name, 0.0), 4),
                "ts_sigma": round(ts_sigma.get(name, 0.0), 4),
                "ts_conservative": round(ts_score.get(name, 0.0), 4),
            })
    print(f"[saved] {ratings_path}")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    elo = results["elo"]
    ts_score = results["trueskill_score"]

    all_names = sorted(elo.keys(), key=lambda n: -elo[n])

    print("\n" + "=" * 70)
    print(f"{'Agent':<25} {'ELO':>8} {'TS-Conservative':>16}")
    print("-" * 70)
    for name in all_names:
        print(
            f"{name:<25} {elo[name]:>8.1f} {ts_score.get(name, 0.0):>16.3f}"
        )
    print("=" * 70)

    # Win rates
    records = results["records"]
    wins: Dict[str, int] = {}
    games: Dict[str, int] = {}
    for rec in records:
        a, b = rec["agent_a"], rec["agent_b"]
        for n in (a, b):
            if n not in wins:
                wins[n] = 0
                games[n] = 0
        games[a] += 1
        games[b] += 1
        if rec["score_a"] == 1.0:
            wins[a] += 1
        elif rec["score_a"] == 0.0:
            wins[b] += 1

    print("\nWin rates:")
    print(f"{'Agent':<25} {'Wins':>6} {'Games':>6} {'WinRate':>9}")
    print("-" * 50)
    for name in sorted(games.keys(), key=lambda n: -(wins.get(n, 0) / max(1, games[n]))):
        w = wins.get(name, 0)
        g = games.get(name, 1)
        print(f"{name:<25} {w:>6} {g:>6} {w/g:>9.1%}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation tournament")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use only 10 seeds (for quick testing)",
    )
    args = parser.parse_args()

    n_seeds = 10 if args.quick else NUM_SEEDS
    seeds = list(range(n_seeds))

    print(f"Tournament settings: seeds={n_seeds}, budget={MCTS_BUDGET}s, max_turns={MAX_TURNS}")
    print(f"Results directory: {RESULTS_DIR}")

    results = run_tournament(DEFAULT_MATCHUPS, seeds)
    print_summary(results)
    save_results(results)
