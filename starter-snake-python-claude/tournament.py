"""
Full round-robin tournament: every agent plays every other agent.
Uses iteration-based MCTS budget so all variants get the same search depth.
Supports multiprocessing to parallelise games across CPU cores.

Two modes:
  --study   Focused study: MCTS-All vs Heuristic/MCTS-Vanilla/Random only.
            Default 200 iterations, 10 seeds. Good for validating MCTS-All.
  (default) Full round-robin ablation with all 7 agents.
            Default 50 iterations, 10 seeds.

Usage:
    python tournament.py                                    # full round-robin
    python tournament.py --study                            # focused study
    python tournament.py --seeds 30 --iterations 100        # custom
    python tournament.py --study --workers 4                # parallel on 4 cores

Environment variables:
    NUM_SEEDS       — number of seeds per matchup (default 10)
    MCTS_ITERATIONS — iterations per MCTS move (default depends on mode)
    MAX_TURNS       — max turns per game (default 200)
    RESULTS_DIR     — output directory (default "tournament_results")
"""

import argparse
import csv
import os
import sys
import time
from itertools import combinations
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple

# Optional TrueSkill
try:
    import trueskill

    _TRUESKILL_AVAILABLE = True
except ImportError:
    _TRUESKILL_AVAILABLE = False
    print("[WARNING] trueskill not installed — TrueSkill ratings will be skipped.")

# ---------------------------------------------------------------------------
# Settings from environment (defaults — can be overridden by CLI args)
# ---------------------------------------------------------------------------

NUM_SEEDS = int(os.environ.get("NUM_SEEDS", "10"))
MAX_TURNS = int(os.environ.get("MAX_TURNS", "200"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", "tournament_results")

# ---------------------------------------------------------------------------
# Agent configs — serialisable names mapped to their MCTS configs.
# We avoid closures/lambdas so that multiprocessing can pickle the work items.
# ---------------------------------------------------------------------------

# Import configs here so they're available at module level
from MCTS import (
    ALL_IMPROVEMENTS_CONFIG,
    HEURISTIC_ROLLOUT_CONFIG,
    OPPONENT_AWARE_CONFIG,
    UCB1_TUNED_CONFIG,
    VANILLA_CONFIG,
)

# Maps agent name → (agent_type, config_or_None)
# agent_type is one of: "random", "heuristic", "mcts"
AGENT_CONFIGS: Dict[str, Tuple[str, object]] = {
    "Random":           ("random", None),
    "Heuristic":        ("heuristic", None),
    "MCTS-Vanilla":     ("mcts", VANILLA_CONFIG),
    "MCTS-HeurRollout": ("mcts", HEURISTIC_ROLLOUT_CONFIG),
    "MCTS-Opponent":    ("mcts", OPPONENT_AWARE_CONFIG),
    "MCTS-UCB1Tuned":   ("mcts", UCB1_TUNED_CONFIG),
    "MCTS-All":         ("mcts", ALL_IMPROVEMENTS_CONFIG),
}


def make_agent(name: str, mcts_iterations: int) -> Callable:
    """Create an agent callable.  mcts_iterations is passed explicitly
    so that the function doesn't rely on mutable module-level state."""
    agent_type, config = AGENT_CONFIGS[name]

    if agent_type == "random":
        import random as _random
        from agent_core import get_legal_moves as _glm, make_state_from_game as _ms

        def agent(game_state: dict) -> str:
            state = _ms(game_state)
            legal = _glm(state, state["you_id"])
            return _random.choice(legal) if legal else "up"
        return agent

    elif agent_type == "heuristic":
        from agent_core import heuristic_move as _hm

        def agent(game_state: dict) -> str:
            move, _ = _hm(game_state)
            return move
        return agent

    else:  # mcts
        from MCTS import mcts_move as _mm
        _cfg = config
        _iters = mcts_iterations

        def agent(game_state: dict) -> str:
            move, _ = _mm(game_state, config=_cfg, max_iterations=_iters)
            return move
        return agent


# ---------------------------------------------------------------------------
# ELO
# ---------------------------------------------------------------------------

def update_elo(
    elo_a: float, elo_b: float, score_a: float, k: float = 32.0
) -> Tuple[float, float]:
    expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    expected_b = 1.0 - expected_a
    score_b = 1.0 - score_a
    new_a = elo_a + k * (score_a - expected_a)
    new_b = elo_b + k * (score_b - expected_b)
    return new_a, new_b


# ---------------------------------------------------------------------------
# Single-game runner (top-level function so it can be pickled)
# ---------------------------------------------------------------------------

def _run_single_game(args_tuple):
    """Run one game.  Designed to be called via multiprocessing.Pool.map().

    Parameters are packed in a tuple for Pool compatibility:
        (name_a, name_b, seed, mcts_iterations, max_turns)

    Returns a record dict.
    """
    name_a, name_b, seed, mcts_iterations, max_turns = args_tuple

    from simulate_game import run_game as _run_game

    agent_a = make_agent(name_a, mcts_iterations)
    agent_b = make_agent(name_b, mcts_iterations)

    t0 = time.time()
    result = _run_game(
        agents=[(name_a, agent_a), (name_b, agent_b)],
        seed=seed,
        max_turns=max_turns,
    )
    elapsed = time.time() - t0

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

    outcome = "WIN" if score_a == 1.0 else ("DRAW" if score_a == 0.5 else "LOSS")
    print(f"  [{name_a} vs {name_b}] seed={seed} {outcome} "
          f"(turns={result['turns']}, {elapsed:.1f}s)", flush=True)

    return {
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


# ---------------------------------------------------------------------------
# Matchup runner (sequential, used when workers=1)
# ---------------------------------------------------------------------------

def run_matchup_sequential(
    name_a: str,
    name_b: str,
    seeds: List[int],
    mcts_iterations: int,
) -> List[dict]:
    """Run one game per seed between name_a and name_b (single process)."""
    work_items = [
        (name_a, name_b, seed, mcts_iterations, MAX_TURNS)
        for seed in seeds
    ]
    return [_run_single_game(item) for item in work_items]


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------

def run_tournament(
    agent_names: List[str],
    seeds: List[int],
    mcts_iterations: int,
    num_workers: int = 1,
) -> dict:
    """
    Full round-robin: every agent plays every other agent.
    If num_workers > 1, games are parallelised across CPU cores.
    """
    # Generate all pairwise matchups
    matchups = list(combinations(agent_names, 2))
    total_games = len(matchups) * len(seeds)
    print(f"\nRound-robin: {len(agent_names)} agents, "
          f"{len(matchups)} matchups, {len(seeds)} seeds = {total_games} games"
          f" (workers={num_workers})")

    tournament_start = time.time()

    if num_workers > 1:
        # --- PARALLEL: build all work items, dispatch to pool ---
        all_work = []
        for name_a, name_b in matchups:
            for seed in seeds:
                all_work.append((name_a, name_b, seed, mcts_iterations, MAX_TURNS))

        print(f"Dispatching {len(all_work)} games across {num_workers} workers...")
        with Pool(processes=num_workers) as pool:
            all_records = pool.map(_run_single_game, all_work)
    else:
        # --- SEQUENTIAL: run matchup by matchup for cleaner output ---
        all_records = []
        games_played = 0
        for name_a, name_b in matchups:
            print(f"\n=== {name_a} vs {name_b} ({games_played}/{total_games} done) ===")
            records = run_matchup_sequential(name_a, name_b, seeds, mcts_iterations)
            all_records.extend(records)
            games_played += len(records)

    elapsed_total = time.time() - tournament_start
    print(f"\nTournament completed: {total_games} games in {elapsed_total:.0f}s "
          f"({elapsed_total/60:.1f} min)")

    # --- Compute ratings ---
    elo: Dict[str, float] = {name: 1500.0 for name in agent_names}

    ts_ratings: Dict[str, object] = {}
    if _TRUESKILL_AVAILABLE:
        ts_env = trueskill.TrueSkill()
        ts_ratings = {name: ts_env.create_rating() for name in agent_names}

    # Sort records by (agent_a, agent_b, seed) for deterministic rating updates
    all_records.sort(key=lambda r: (r["agent_a"], r["agent_b"], r["seed"]))

    for rec in all_records:
        na, nb = rec["agent_a"], rec["agent_b"]
        score_a = rec["score_a"]
        elo[na], elo[nb] = update_elo(elo[na], elo[nb], score_a)

        if _TRUESKILL_AVAILABLE:
            if score_a == 1.0:
                ts_ratings[na], ts_ratings[nb] = trueskill.rate_1vs1(
                    ts_ratings[na], ts_ratings[nb]
                )
            elif score_a == 0.0:
                ts_ratings[nb], ts_ratings[na] = trueskill.rate_1vs1(
                    ts_ratings[nb], ts_ratings[na]
                )
            else:
                ts_ratings[na], ts_ratings[nb] = trueskill.rate_1vs1(
                    ts_ratings[na], ts_ratings[nb], drawn=True
                )

    # Build TrueSkill summary
    ts_mu: Dict[str, float] = {}
    ts_sigma: Dict[str, float] = {}
    ts_score: Dict[str, float] = {}
    if _TRUESKILL_AVAILABLE:
        for name in agent_names:
            r = ts_ratings[name]
            ts_mu[name] = r.mu
            ts_sigma[name] = r.sigma
            ts_score[name] = r.mu - 3.0 * r.sigma
    else:
        for name in agent_names:
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

def save_results(results: dict, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    # game_records.csv
    records = results["records"]
    if records:
        record_path = os.path.join(results_dir, "game_records.csv")
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
    ratings_path = os.path.join(results_dir, "ratings.csv")
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
# Summary printer (with pairwise win matrix)
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

    # Overall win rates
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

    print("\nOverall win rates:")
    print(f"{'Agent':<25} {'Wins':>6} {'Games':>6} {'WinRate':>9}")
    print("-" * 50)
    for name in sorted(games.keys(), key=lambda n: -(wins.get(n, 0) / max(1, games[n]))):
        w = wins.get(name, 0)
        g = games.get(name, 1)
        print(f"{name:<25} {w:>6} {g:>6} {w/g:>9.1%}")

    # Pairwise win matrix
    print("\nPairwise win rates (row beat column):")
    pair_wins: Dict[Tuple[str, str], int] = {}
    pair_games: Dict[Tuple[str, str], int] = {}
    for rec in records:
        a, b = rec["agent_a"], rec["agent_b"]
        pair_games[(a, b)] = pair_games.get((a, b), 0) + 1
        pair_games[(b, a)] = pair_games.get((b, a), 0) + 1
        if rec["score_a"] == 1.0:
            pair_wins[(a, b)] = pair_wins.get((a, b), 0) + 1
        elif rec["score_a"] == 0.0:
            pair_wins[(b, a)] = pair_wins.get((b, a), 0) + 1

    # Short names for matrix display
    short = {n: n[:12] for n in all_names}
    header = f"{'':>14}" + "".join(f"{short[n]:>14}" for n in all_names)
    print(header)
    for row in all_names:
        cells = []
        for col in all_names:
            if row == col:
                cells.append(f"{'---':>14}")
            else:
                w = pair_wins.get((row, col), 0)
                g = pair_games.get((row, col), 0)
                if g > 0:
                    pct = f"{w}/{g} ({w/g:.0%})"
                    cells.append(f"{pct:>14}")
                else:
                    cells.append(f"{'n/a':>14}")
        print(f"{short[row]:>14}" + "".join(cells))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full round-robin tournament")
    parser.add_argument(
        "--study",
        action="store_true",
        help="Focused study: MCTS-All vs Heuristic/Vanilla/Random (200 iters default)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use only 5 seeds (for quick testing)",
    )
    parser.add_argument(
        "--seeds", type=int, default=None,
        help="Override number of seeds per matchup",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Override MCTS iterations per move",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default 1 = sequential)",
    )
    args = parser.parse_args()

    if args.study:
        # Focused study mode: MCTS-All vs 3 key opponents
        mcts_iterations = args.iterations if args.iterations is not None else 200
        n_seeds = args.seeds if args.seeds is not None else (5 if args.quick else 10)
        results_dir = os.environ.get("RESULTS_DIR", "study_results")
        agent_names = ["Random", "Heuristic", "MCTS-Vanilla", "MCTS-All"]
        mode_label = "FOCUSED STUDY"
    else:
        # Full round-robin ablation
        mcts_iterations = args.iterations if args.iterations is not None else int(
            os.environ.get("MCTS_ITERATIONS", "50")
        )
        n_seeds = args.seeds if args.seeds is not None else (5 if args.quick else NUM_SEEDS)
        results_dir = RESULTS_DIR
        agent_names = list(AGENT_CONFIGS.keys())
        mode_label = "FULL ROUND-ROBIN"

    seeds = list(range(n_seeds))

    print(f"\n{'='*60}")
    print(f"  {mode_label}")
    print(f"{'='*60}")
    print(f"  Seeds per matchup : {n_seeds}")
    print(f"  MCTS iterations   : {mcts_iterations}")
    print(f"  Max turns per game: {MAX_TURNS}")
    print(f"  Workers           : {args.workers}")
    print(f"  Agents            : {', '.join(agent_names)}")
    print(f"  Results directory  : {results_dir}")
    print(f"{'='*60}\n")

    results = run_tournament(agent_names, seeds, mcts_iterations, num_workers=args.workers)
    print_summary(results)
    save_results(results, results_dir)
