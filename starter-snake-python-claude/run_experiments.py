#!/usr/bin/env python3
"""
run_experiments.py  --  Idempotent BattleSnake MCTS experiment runner.

Runs two experiment families and saves all results as CSVs:
  1. TOURNAMENT  : full round-robin among Random, Heuristic, MCTS-Vanilla, MCTS-All
     - Covers all pairwise comparisons including every MCTS variant vs Heuristic
  2. HYPERPARAMS : for each MCTS-All parameter variant vs [Random, Heuristic, MCTS-Vanilla]
     - iterations sweep: [25, 50, 100]
     - rollout_depth sweep: [4, 8, 12]
     - exploration sweep: [0.6, 1.05, 1.4]
     - prior_bonus_scale sweep: [1.5, 4.0, 8.0]

Idempotency: any (agent_a, agent_b, seed) already in game_records.csv is SKIPPED.
If the process is interrupted, re-running continues from exactly where it stopped.

Answers the following assignment questions:
  Q1: How does heuristic compare to random? (tournament: Heuristic vs Random)
  Q2: What changes are needed for vanilla MCTS? (tournament: MCTS-Vanilla vs Heuristic)
  Q3: What is the effect of heuristic rollouts vs random rollouts?
      (tournament: MCTS-All vs MCTS-Vanilla)
  Q4: How do MCTS improvements compare overall? (full tournament win matrix)
  Q5: What are the effects of MCTS hyperparameters?
      (hyperparameter sweeps: iterations, rollout_depth, exploration, prior_bonus_scale)
  Q6: Which agent performs best? (ELO + TrueSkill from tournament)

Usage:
    # Run everything (default: 40 seeds, 50 MCTS iters):
    python run_experiments.py

    # Run only the tournament:
    python run_experiments.py --mode tournament

    # Run only the hyperparameter sweeps:
    python run_experiments.py --mode hyperparams

    # Custom seeds/iterations:
    python run_experiments.py --seeds 40 --iterations 50 --workers 4

    # Resume after interruption - just re-run the same command:
    python run_experiments.py
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Path setup — must happen before importing game modules
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

try:
    import trueskill
    _TRUESKILL_AVAILABLE = True
except ImportError:
    _TRUESKILL_AVAILABLE = False
    print("[WARNING] trueskill not installed — TrueSkill ratings will be skipped.")
    print("          Install with:  pip install trueskill")

# Import from tournament.py (existing infrastructure)
from tournament import (
    AgentSpec,
    named_agent_specs,
    _run_single_game,
    update_elo,
)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = 40
DEFAULT_ITERATIONS = 50
DEFAULT_MAX_TURNS = 200
DEFAULT_RESULTS_DIR = "results"

# Agents in the main round-robin tournament
TOURNAMENT_AGENTS = ["Random", "Heuristic", "MCTS-Vanilla", "MCTS-All"]

# All MCTS variants for the Heuristic comparison suite
MCTS_ALL_VARIANTS = [
    "MCTS-Vanilla",
    "MCTS-HeurRollout",
    "MCTS-Prior",
    "MCTS-Opponent",
    "MCTS-RAVE",
    "MCTS-UCB1Tuned",
    "MCTS-All",
]

# Hyperparameter sweep definitions (no lambdas — must be picklable)
HYPERPARAMETER_SWEEPS = [
    {
        "family":      "hyper_iterations",
        "param":       "iterations",
        "values":      [25, 50, 100],
    },
    {
        "family":      "hyper_rollout_depth",
        "param":       "rollout_depth",
        "values":      [4, 8, 12],
    },
    {
        "family":      "hyper_exploration",
        "param":       "exploration",
        "values":      [0.6, 1.05, 1.4],
    },
    {
        "family":      "hyper_prior_bonus",
        "param":       "prior_bonus_scale",
        "values":      [1.5, 4.0, 8.0],
    },
]


def _build_overrides(param_name: str, value) -> Dict[str, Any]:
    """Build config_overrides dict for a given hyperparameter."""
    if param_name == "iterations":
        return {}           # iterations is passed as mcts_iterations, not a config key
    elif param_name == "rollout_depth":
        return {"rollout_depth": int(value)}
    elif param_name == "exploration":
        return {"exploration": float(value)}
    elif param_name == "prior_bonus_scale":
        return {"prior_bonus_scale": float(value)}
    return {}


def _sanitize(v) -> str:
    """Turn a float/int value into a filesystem-safe string."""
    return str(v).replace(".", "p").replace("-", "m")


# ---------------------------------------------------------------------------
# CSV + idempotency helpers
# ---------------------------------------------------------------------------

def load_completed(records_path: str) -> Set[Tuple[str, str, int]]:
    """Return set of (agent_a, agent_b, seed) already in game_records.csv."""
    if not os.path.exists(records_path):
        return set()
    try:
        with open(records_path, newline="") as f:
            return {(r["agent_a"], r["agent_b"], int(r["seed"])) for r in csv.DictReader(f)}
    except Exception as e:
        print(f"[WARN] Could not read {records_path}: {e}")
        return set()


def is_done(agent_a: str, agent_b: str, seed: int, completed: Set) -> bool:
    """Return True if this exact matchup+seed has already been played."""
    return (agent_a, agent_b, seed) in completed or (agent_b, agent_a, seed) in completed


def append_records(records_path: str, new_records: List[dict]) -> None:
    """Append new rows to game_records.csv, writing header only for new files."""
    if not new_records:
        return
    os.makedirs(os.path.dirname(os.path.abspath(records_path)), exist_ok=True)
    file_exists = os.path.exists(records_path)
    fieldnames = list(new_records[0].keys())
    with open(records_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_records)


def load_records(records_path: str) -> List[dict]:
    """Load all rows from game_records.csv."""
    if not os.path.exists(records_path):
        return []
    with open(records_path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Ratings computation
# ---------------------------------------------------------------------------

def compute_ratings(records: List[dict], agent_names: List[str]) -> Dict[str, Any]:
    """Compute ELO and TrueSkill ratings from a list of game records."""
    elo: Dict[str, float] = {n: 1500.0 for n in agent_names}

    ts_ratings = {}
    if _TRUESKILL_AVAILABLE:
        ts_env = trueskill.TrueSkill()
        ts_ratings = {n: ts_env.create_rating() for n in agent_names}

    # Deterministic ordering ensures consistent rating history
    sorted_records = sorted(
        records,
        key=lambda r: (r["agent_a"], r["agent_b"], int(r["seed"]))
    )

    for rec in sorted_records:
        na, nb = rec["agent_a"], rec["agent_b"]
        if na not in elo or nb not in elo:
            continue
        score_a = float(rec["score_a"])
        elo[na], elo[nb] = update_elo(elo[na], elo[nb], score_a)

        if _TRUESKILL_AVAILABLE and na in ts_ratings and nb in ts_ratings:
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

    ts_mu: Dict[str, float] = {}
    ts_sigma: Dict[str, float] = {}
    ts_score: Dict[str, float] = {}
    if _TRUESKILL_AVAILABLE:
        for n in agent_names:
            if n in ts_ratings:
                r = ts_ratings[n]
                ts_mu[n] = r.mu
                ts_sigma[n] = r.sigma
                ts_score[n] = r.mu - 3.0 * r.sigma
    else:
        for n in agent_names:
            ts_mu[n] = 0.0
            ts_sigma[n] = 0.0
            ts_score[n] = 0.0

    return {
        "elo":          elo,
        "ts_mu":        ts_mu,
        "ts_sigma":     ts_sigma,
        "ts_score":     ts_score,
    }


def save_ratings(results_dir: str, ratings: Dict, agent_names: List[str]) -> None:
    """Write ratings.csv to results_dir."""
    os.makedirs(results_dir, exist_ok=True)
    elo = ratings["elo"]
    ts_score = ratings.get("ts_score", {})
    ts_mu = ratings.get("ts_mu", {})
    ts_sigma = ratings.get("ts_sigma", {})

    sorted_names = sorted(agent_names, key=lambda n: -elo.get(n, 0.0))
    path = os.path.join(results_dir, "ratings.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["agent", "elo", "ts_mu", "ts_sigma", "ts_conservative"]
        )
        writer.writeheader()
        for name in sorted_names:
            writer.writerow({
                "agent":          name,
                "elo":            round(elo.get(name, 1500.0), 2),
                "ts_mu":          round(ts_mu.get(name, 0.0), 4),
                "ts_sigma":       round(ts_sigma.get(name, 0.0), 4),
                "ts_conservative": round(ts_score.get(name, 0.0), 4),
            })
    print(f"  [saved] {path}")


def write_matchup_summary(results_dir: str) -> None:
    """Aggregate per-matchup stats from game_records.csv → matchup_summary.csv."""
    records_path = os.path.join(results_dir, "game_records.csv")
    if not os.path.exists(records_path):
        return

    records = load_records(records_path)
    summary: Dict[Tuple, dict] = defaultdict(lambda: {
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

    out_path = os.path.join(results_dir, "matchup_summary.csv")
    fieldnames = [
        "agent_a", "agent_b", "games",
        "wins_a", "wins_b", "win_rate_a", "win_rate_b",
        "avg_turns", "avg_survival_a", "avg_survival_b",
        "avg_final_length_a", "avg_final_length_b",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (agent_a, agent_b), row in sorted(summary.items()):
            g = row["games"]
            wa = row["wins_a"]
            writer.writerow({
                "agent_a":              agent_a,
                "agent_b":              agent_b,
                "games":                g,
                "wins_a":               round(wa, 3),
                "wins_b":               round(g - wa, 3),
                "win_rate_a":           round(wa / g, 4) if g else 0,
                "win_rate_b":           round((g - wa) / g, 4) if g else 0,
                "avg_turns":            round(sum(row["turns"]) / g, 3) if g else 0,
                "avg_survival_a":       round(sum(row["survival_a"]) / g, 3) if g else 0,
                "avg_survival_b":       round(sum(row["survival_b"]) / g, 3) if g else 0,
                "avg_final_length_a":   round(sum(row["length_a"]) / g, 3) if g else 0,
                "avg_final_length_b":   round(sum(row["length_b"]) / g, 3) if g else 0,
            })
    print(f"  [saved] {out_path}")


def save_metadata(
    results_dir: str,
    agent_specs: List[AgentSpec],
    seeds: List[int],
    iterations: int,
    max_turns: int,
    extra: Optional[Dict] = None,
) -> None:
    """Write metadata.json to results_dir."""
    os.makedirs(results_dir, exist_ok=True)
    metadata = {
        "agent_specs":      [s.to_dict() for s in agent_specs],
        "num_agents":       len(agent_specs),
        "seeds":            list(seeds),
        "num_seeds":        len(seeds),
        "mcts_iterations":  iterations,
        "max_turns":        max_turns,
    }
    if extra:
        metadata.update(extra)
    path = os.path.join(results_dir, "metadata.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# Bootstrap helper — seed results dir from an existing game_records.csv
# ---------------------------------------------------------------------------

def bootstrap_from(src_csv: str, results_dir: str) -> int:
    """
    Copy src_csv into results_dir/game_records.csv if the target does not yet
    exist.  Returns number of records bootstrapped (0 if target already exists).
    """
    dst = os.path.join(results_dir, "game_records.csv")
    if os.path.exists(dst):
        existing = load_records(dst)
        print(f"  [bootstrap] {dst} already exists ({len(existing)} records) — skipping copy.")
        return 0
    if not os.path.exists(src_csv):
        print(f"  [bootstrap] Source not found: {src_csv}")
        return 0
    os.makedirs(results_dir, exist_ok=True)
    import shutil
    shutil.copy2(src_csv, dst)
    records = load_records(dst)
    print(f"  [bootstrap] Copied {len(records)} records from {src_csv}  →  {dst}")
    return len(records)


# ---------------------------------------------------------------------------
# Per-matchup incremental game runner
# ---------------------------------------------------------------------------

def run_matchup_incremental(
    spec_a: AgentSpec,
    spec_b: AgentSpec,
    seeds: List[int],
    iterations: int,
    completed: Set,
    max_turns: int,
    workers: int = 1,
) -> List[dict]:
    """
    Run only the seeds that are NOT yet in `completed`.
    Returns a list of new game record dicts.
    """
    pending = [s for s in seeds if not is_done(spec_a.name, spec_b.name, s, completed)]
    if not pending:
        return []

    work_items = [
        (spec_a.to_dict(), spec_b.to_dict(), seed, iterations, max_turns)
        for seed in pending
    ]

    t0 = time.time()
    if workers > 1:
        # Use 'fork' explicitly — avoids the macOS/Python 3.8+ spawn-deadlock where
        # worker processes hang re-importing modules before the first task runs.
        import multiprocessing as _mp
        ctx = _mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            results = pool.map(_run_single_game, work_items)
    else:
        results = [_run_single_game(item) for item in work_items]

    elapsed = time.time() - t0
    wins_a = sum(1 for r in results if r["score_a"] == 1.0)
    print(f"    Played {len(results)} games in {elapsed:.1f}s  "
          f"({spec_a.name} won {wins_a}/{len(results)})")
    return results


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------

def run_tournament(
    agent_specs:  List[AgentSpec],
    seeds:        List[int],
    iterations:   int,
    results_dir:  str,
    workers:      int = 1,
    max_turns:    int = DEFAULT_MAX_TURNS,
) -> None:
    """
    Full round-robin tournament with idempotent resume.
    New games are appended to game_records.csv as they complete.
    Ratings are recomputed from the full dataset at the end.
    """
    os.makedirs(results_dir, exist_ok=True)
    records_path = os.path.join(results_dir, "game_records.csv")

    matchups = list(combinations(agent_specs, 2))
    total_needed = len(matchups) * len(seeds)

    print(f"\n{'='*65}")
    print(f"  TOURNAMENT")
    print(f"  Agents    : {', '.join(s.name for s in agent_specs)}")
    print(f"  Seeds     : {len(seeds)}  |  Iterations: {iterations}  |  Workers: {workers}")
    print(f"  Total     : {len(matchups)} matchups × {len(seeds)} seeds = {total_needed} games")
    print(f"  Output    : {results_dir}")
    print(f"{'='*65}")

    completed = load_completed(records_path)
    skipped = sum(
        1 for (sa, sb) in matchups
        for seed in seeds
        if is_done(sa.name, sb.name, seed, completed)
    )
    print(f"  Already done : {skipped} / {total_needed} games")

    for spec_a, spec_b in matchups:
        pending_count = sum(
            1 for s in seeds if not is_done(spec_a.name, spec_b.name, s, completed)
        )
        if pending_count == 0:
            print(f"  [skip] {spec_a.name} vs {spec_b.name}  (all {len(seeds)} seeds done)")
            continue

        print(f"\n  --- {spec_a.name} vs {spec_b.name}  ({pending_count} seeds pending) ---")
        new_records = run_matchup_incremental(
            spec_a, spec_b, seeds, iterations, completed, max_turns, workers
        )
        if new_records:
            append_records(records_path, new_records)
            for rec in new_records:
                completed.add((rec["agent_a"], rec["agent_b"], int(rec["seed"])))

    # Finalize: recompute ratings from full dataset
    print(f"\n  Finalizing ratings from {records_path} ...")
    all_records = load_records(records_path)
    agent_names = [s.name for s in agent_specs]
    ratings = compute_ratings(all_records, agent_names)
    save_ratings(results_dir, ratings, agent_names)
    write_matchup_summary(results_dir)
    save_metadata(results_dir, agent_specs, seeds, iterations, max_turns,
                  extra={"experiment": "tournament"})

    print(f"\n  Tournament complete. {len(all_records)} game records total.")
    _print_console_summary(all_records, ratings)


# ---------------------------------------------------------------------------
# Hyperparameter sweep runner
# ---------------------------------------------------------------------------

def run_hyperparameter_sweep(
    family_dir:      str,
    param_name:      str,
    param_values:    List,
    seeds:           List[int],
    base_iterations: int,
    workers:         int,
    max_turns:       int,
) -> None:
    """
    Run one hyperparameter sweep family.
    Each parameter value gets its own subfolder.
    Baseline agents: Random, Heuristic, MCTS-Vanilla (unchanged).
    Focus agent: MCTS-All with the parameter value overridden.
    """
    os.makedirs(family_dir, exist_ok=True)
    baseline_specs = named_agent_specs(["Random", "Heuristic", "MCTS-Vanilla"])

    print(f"\n{'='*65}")
    print(f"  SWEEP: {param_name}  values={param_values}")
    print(f"  Output: {family_dir}")
    print(f"{'='*65}")

    for value in param_values:
        agent_name = f"MCTS-All-{param_name}-{_sanitize(value)}"
        overrides = _build_overrides(param_name, value)
        agent_spec = AgentSpec(
            name=agent_name,
            agent_type="mcts",
            config_name="ALL_IMPROVEMENTS_CONFIG",
            config_overrides=overrides,
        )

        # For iterations sweep, the value IS the iteration count
        iters = int(value) if param_name == "iterations" else base_iterations

        run_dir = os.path.join(family_dir, f"{param_name}_{_sanitize(value)}")
        os.makedirs(run_dir, exist_ok=True)
        records_path = os.path.join(run_dir, "game_records.csv")

        all_specs = baseline_specs + [agent_spec]
        matchups = list(combinations(all_specs, 2))
        total_needed = len(matchups) * len(seeds)

        completed = load_completed(records_path)
        pending_total = sum(
            1 for (sa, sb) in matchups
            for s in seeds
            if not is_done(sa.name, sb.name, s, completed)
        )

        print(f"\n  [{param_name}={value}]  agent={agent_name}  "
              f"({total_needed - pending_total}/{total_needed} done)")

        if pending_total == 0:
            print(f"    [skip] All seeds complete.")
        else:
            for spec_a, spec_b in matchups:
                pending_seeds = [
                    s for s in seeds
                    if not is_done(spec_a.name, spec_b.name, s, completed)
                ]
                if not pending_seeds:
                    continue
                print(f"    {spec_a.name} vs {spec_b.name}  ({len(pending_seeds)} seeds)")
                new_records = run_matchup_incremental(
                    spec_a, spec_b, seeds, iters, completed, max_turns, workers
                )
                if new_records:
                    append_records(records_path, new_records)
                    for rec in new_records:
                        completed.add((rec["agent_a"], rec["agent_b"], int(rec["seed"])))

        # Finalize ratings for this run
        all_records = load_records(records_path)
        if all_records:
            agent_names = [s.name for s in all_specs]
            ratings = compute_ratings(all_records, agent_names)
            save_ratings(run_dir, ratings, agent_names)
            write_matchup_summary(run_dir)
            save_metadata(
                run_dir, all_specs, seeds, iters, max_turns,
                extra={
                    "experiment":       "hyperparameter_sweep",
                    "parameter_name":   param_name,
                    "parameter_value":  value,
                    "focus_agent":      agent_name,
                    "base_config_name": "ALL_IMPROVEMENTS_CONFIG",
                },
            )

    # Write family-level sweep summary
    _write_sweep_summary(family_dir, param_name)


def _write_sweep_summary(family_dir: str, param_name: str) -> None:
    """
    Aggregate per-parameter-value metrics into a single summary.csv
    at the family level.  Readable by plot_all.py for sweep plots.
    """
    rows = []

    for entry in sorted(os.listdir(family_dir)):
        child = os.path.join(family_dir, entry)
        if not os.path.isdir(child):
            continue

        metadata_path = os.path.join(child, "metadata.json")
        records_path  = os.path.join(child, "game_records.csv")
        ratings_path  = os.path.join(child, "ratings.csv")

        if not all(os.path.exists(p) for p in [metadata_path, records_path, ratings_path]):
            continue

        with open(metadata_path) as f:
            metadata = json.load(f)

        focus_agent    = metadata.get("focus_agent", "")
        param_value    = metadata.get("parameter_value", 0)

        records = load_records(records_path)
        with open(ratings_path, newline="") as f:
            ratings_rows = list(csv.DictReader(f))

        rating_row = next((r for r in ratings_rows if r["agent"] == focus_agent), None)
        if rating_row is None:
            continue

        # Per-focus-agent metrics
        wins = 0.0
        games = 0
        survival = []
        pair_wins:  Dict[Tuple, float] = defaultdict(float)
        pair_games: Dict[Tuple, int]   = defaultdict(int)

        for rec in records:
            a, b = rec["agent_a"], rec["agent_b"]
            score_a = float(rec["score_a"])
            pair_games[(a, b)] += 1
            pair_games[(b, a)] += 1
            pair_wins[(a, b)] += score_a
            pair_wins[(b, a)] += 1.0 - score_a

            if a == focus_agent:
                wins += score_a
                games += 1
                survival.append(float(rec["turns_survived_a"]))
            elif b == focus_agent:
                wins += (1.0 - score_a)
                games += 1
                survival.append(float(rec["turns_survived_b"]))

        row: Dict[str, Any] = {
            "run_dir":          entry,
            "parameter_name":   param_name,
            "parameter_value":  float(param_value),
            "focus_agent":      focus_agent,
            "elo":              float(rating_row["elo"]),
            "ts_conservative":  float(rating_row["ts_conservative"]),
            "games":            games,
            "wins":             round(wins, 3),
            "win_rate":         round(wins / games, 4) if games else 0.0,
            "avg_survival":     round(sum(survival) / len(survival), 3) if survival else 0.0,
        }

        for opp in ["Heuristic", "MCTS-Vanilla", "Random"]:
            key = (focus_agent, opp)
            g = pair_games.get(key, 0)
            w = pair_wins.get(key, 0.0)
            col_name = f"vs_{opp.lower().replace('-', '_')}_win_rate"
            row[col_name] = round(w / g, 4) if g else 0.0

        rows.append(row)

    if not rows:
        print(f"  [WARN] No sweep runs found under {family_dir}")
        return

    rows.sort(key=lambda r: float(r["parameter_value"]))
    summary_path = os.path.join(family_dir, "summary.csv")
    fieldnames = list(rows[0].keys())
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [saved] {summary_path}")


# ---------------------------------------------------------------------------
# Heuristic vs all MCTS variants — focused ladder run
# ---------------------------------------------------------------------------

def run_vs_heuristic(
    seeds:        List[int],
    iterations:   int,
    results_dir:  str,
    workers:      int = 1,
    max_turns:    int = DEFAULT_MAX_TURNS,
) -> None:
    """
    Run Heuristic against every MCTS variant.
    Saves into the same results_dir/game_records.csv as the main tournament,
    so all data stays in one file and ratings cover all known agents.

    This is idempotent: any (Heuristic, MCTS-X, seed) already in the CSV is skipped.
    Bootstrapping from small_study_40_seeds first means MCTS-Vanilla and MCTS-All
    matchups are already done and will be skipped automatically.
    """
    os.makedirs(results_dir, exist_ok=True)
    records_path = os.path.join(results_dir, "game_records.csv")

    heuristic_spec  = named_agent_specs(["Heuristic"])[0]
    mcts_specs      = named_agent_specs(MCTS_ALL_VARIANTS)
    total_needed    = len(mcts_specs) * len(seeds)

    print(f"\n{'='*65}")
    print(f"  HEURISTIC vs MCTS SUITE")
    print(f"  MCTS agents : {', '.join(s.name for s in mcts_specs)}")
    print(f"  Seeds       : {len(seeds)}  |  Iterations: {iterations}  |  Workers: {workers}")
    print(f"  Total max   : {len(mcts_specs)} matchups × {len(seeds)} seeds = {total_needed} games")
    print(f"  Output      : {results_dir}")
    print(f"{'='*65}")

    completed = load_completed(records_path)
    skipped = sum(
        1 for spec in mcts_specs
        for seed in seeds
        if is_done(heuristic_spec.name, spec.name, seed, completed)
    )
    new_needed = total_needed - skipped
    print(f"  Already done : {skipped} / {total_needed}  →  {new_needed} games to run")

    for mcts_spec in mcts_specs:
        pending_count = sum(
            1 for s in seeds
            if not is_done(heuristic_spec.name, mcts_spec.name, s, completed)
        )
        if pending_count == 0:
            print(f"  [skip] Heuristic vs {mcts_spec.name}  (all {len(seeds)} seeds done)")
            continue

        print(f"\n  --- Heuristic vs {mcts_spec.name}  ({pending_count} seeds pending) ---")
        new_records = run_matchup_incremental(
            heuristic_spec, mcts_spec, seeds, iterations, completed, max_turns, workers
        )
        if new_records:
            append_records(records_path, new_records)
            for rec in new_records:
                completed.add((rec["agent_a"], rec["agent_b"], int(rec["seed"])))

    # Recompute ratings from the full combined dataset
    print(f"\n  Finalizing ratings ...")
    all_records = load_records(records_path)
    # Discover all agents actually in the file (may include bootstrapped agents too)
    all_agent_names = sorted(
        {rec["agent_a"] for rec in all_records} | {rec["agent_b"] for rec in all_records}
    )
    ratings = compute_ratings(all_records, all_agent_names)
    save_ratings(results_dir, ratings, all_agent_names)
    write_matchup_summary(results_dir)

    print(f"\n  Done. {len(all_records)} total records in {records_path}")
    _print_console_summary(all_records, ratings)


# ---------------------------------------------------------------------------
# Console summary helper
# ---------------------------------------------------------------------------

def _print_console_summary(records: List[dict], ratings: Dict) -> None:
    elo = ratings["elo"]
    ts_score = ratings.get("ts_score", {})

    all_names = sorted(elo, key=lambda n: -elo[n])
    print("\n" + "─" * 55)
    print(f"  {'Agent':<24} {'ELO':>8}  {'TS-Conservative':>16}")
    print("─" * 55)
    for name in all_names:
        print(f"  {name:<24} {elo[name]:>8.1f}  {ts_score.get(name, 0.0):>16.3f}")
    print("─" * 55)

    wins:  Dict[str, int] = defaultdict(int)
    games: Dict[str, int] = defaultdict(int)
    for rec in records:
        a, b = rec["agent_a"], rec["agent_b"]
        games[a] += 1
        games[b] += 1
        if float(rec["score_a"]) == 1.0:
            wins[a] += 1
        elif float(rec["score_a"]) == 0.0:
            wins[b] += 1

    print("\n  Win rates:")
    for name in sorted(games, key=lambda n: -(wins.get(n, 0) / max(1, games[n]))):
        g = games[name]
        w = wins.get(name, 0)
        print(f"    {name:<24} {w}/{g}  ({w/g:.1%})")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Idempotent BattleSnake MCTS experiment runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Re-running the same command safely resumes from where it stopped.\n"
            "After this script completes, run:  python plot_all.py\n\n"
            "Quick-start with existing data:\n"
            "  python run_experiments.py --mode vs_heuristic --bootstrap-from small_study_40_seeds/game_records.csv --workers 4"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["tournament", "vs_heuristic", "hyperparams", "all"],
        default="all",
        help=(
            "'tournament'    = round-robin [Random, Heuristic, MCTS-Vanilla, MCTS-All]; "
            "'vs_heuristic'  = Heuristic vs every MCTS variant (fastest, fills the ladder); "
            "'hyperparams'   = MCTS hyperparameter sweeps; "
            "'all'           = tournament + vs_heuristic + hyperparams."
        ),
    )
    parser.add_argument(
        "--seeds", type=int, default=DEFAULT_SEEDS,
        help="Seeds per matchup for tournament / vs_heuristic modes.",
    )
    parser.add_argument(
        "--hyper-seeds", type=int, default=None,
        help="Seeds per matchup for hyperparameter sweeps (defaults to --seeds).",
    )
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS,
        help="MCTS iterations per move (for non-iteration sweeps).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel worker processes (1 = sequential, safer on all platforms).",
    )
    parser.add_argument(
        "--max-turns", type=int, default=DEFAULT_MAX_TURNS,
        help="Maximum turns per game before forced termination.",
    )
    parser.add_argument(
        "--results-dir", default=DEFAULT_RESULTS_DIR,
        help="Base directory for all output CSVs.",
    )
    parser.add_argument(
        "--bootstrap-from", default=None, metavar="CSV_PATH",
        help=(
            "Path to an existing game_records.csv to seed results/tournament/ with. "
            "Skips already-completed matchups automatically. "
            "Example: --bootstrap-from small_study_40_seeds/game_records.csv"
        ),
    )
    args = parser.parse_args()

    seeds       = list(range(args.seeds))
    hyper_seeds = list(range(args.hyper_seeds if args.hyper_seeds is not None else args.seeds))
    base_dir    = os.path.join(os.getcwd(), args.results_dir)
    tour_dir    = os.path.join(base_dir, "tournament")

    print("\n" + "=" * 65)
    print("  BattleSnake Experiment Runner")
    print("=" * 65)
    print(f"  mode            : {args.mode}")
    print(f"  seeds           : {args.seeds}")
    print(f"  hyper_seeds     : {len(hyper_seeds)}")
    print(f"  iterations      : {args.iterations}")
    print(f"  max_turns       : {args.max_turns}")
    print(f"  workers         : {args.workers}")
    print(f"  results_dir     : {base_dir}")
    if args.bootstrap_from:
        print(f"  bootstrap_from  : {args.bootstrap_from}")
    print("=" * 65)

    # ── Optional bootstrap ───────────────────────────────────────────────────
    if args.bootstrap_from:
        bootstrap_from(args.bootstrap_from, tour_dir)

    # ── Tournament (4-agent round-robin) ─────────────────────────────────────
    if args.mode in ("tournament", "all"):
        specs = named_agent_specs(TOURNAMENT_AGENTS)
        run_tournament(
            agent_specs=specs,
            seeds=seeds,
            iterations=args.iterations,
            results_dir=tour_dir,
            workers=args.workers,
            max_turns=args.max_turns,
        )

    # ── Heuristic vs full MCTS ladder ────────────────────────────────────────
    if args.mode in ("vs_heuristic", "all"):
        run_vs_heuristic(
            seeds=seeds,
            iterations=args.iterations,
            results_dir=tour_dir,
            workers=args.workers,
            max_turns=args.max_turns,
        )

    # ── Hyperparameter sweeps ────────────────────────────────────────────────
    if args.mode in ("hyperparams", "all"):
        for sweep in HYPERPARAMETER_SWEEPS:
            family_dir = os.path.join(base_dir, sweep["family"])
            run_hyperparameter_sweep(
                family_dir=family_dir,
                param_name=sweep["param"],
                param_values=sweep["values"],
                seeds=hyper_seeds,
                base_iterations=args.iterations,
                workers=args.workers,
                max_turns=args.max_turns,
            )

    print("\n" + "=" * 65)
    print("  All experiments complete.")
    print(f"  Results saved in: {base_dir}")
    print("  Run  python plot_all.py  to generate all plots.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
