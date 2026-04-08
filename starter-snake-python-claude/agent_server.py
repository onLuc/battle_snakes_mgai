"""
Agent server: supports multiple MCTS configs via mode string.
"""

import random
import typing

from MCTS import (
    ALL_IMPROVEMENTS_CONFIG,
    HEURISTIC_ROLLOUT_CONFIG,
    MCTSConfig,
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


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------

MCTS_CONFIGS: typing.Dict[str, MCTSConfig] = {
    "mcts":                   ALL_IMPROVEMENTS_CONFIG,
    "mcts_vanilla":           VANILLA_CONFIG,
    "mcts_heuristic_rollout": HEURISTIC_ROLLOUT_CONFIG,
    "mcts_prior":             PRIOR_GUIDED_CONFIG,
    "mcts_opponent":          OPPONENT_AWARE_CONFIG,
    "mcts_rave":              RAVE_CONFIG,
    "mcts_ucb1_tuned":        UCB1_TUNED_CONFIG,
    "mcts_all":               ALL_IMPROVEMENTS_CONFIG,
}

# Per-mode appearance
_MODE_APPEARANCE: typing.Dict[str, typing.Tuple[str, str, str]] = {
    # mode            color       head            tail
    "random":         ("#888888", "default",      "default"),
    "heuristic":      ("#D97706", "sand-worm",    "round-bum"),
    "mcts":           ("#00CC00", "evil",         "bolt"),
    "mcts_vanilla":   ("#3B82F6", "dead",         "skinny"),
    "mcts_heuristic_rollout": ("#8B5CF6", "smile", "freckled"),
    "mcts_prior":     ("#EC4899", "workout",      "round-bum"),
    "mcts_opponent":  ("#F59E0B", "bfl-96",       "curled"),
    "mcts_rave":      ("#10B981", "replit",       "fat-rattle"),
    "mcts_ucb1_tuned":("#EF4444", "caffeine",     "rattle"),
    "mcts_all":       ("#00CC00", "evil",         "bolt"),
}


def info(mode: str) -> typing.Dict:
    color, head, tail = _MODE_APPEARANCE.get(mode, ("#00CC00", "evil", "bolt"))
    return {
        "apiversion": "1",
        "author": f"Agent-{mode}",
        "color": color,
        "head": head,
        "tail": tail,
    }


def start(game_state: typing.Dict) -> None:
    pass


def end(game_state: typing.Dict) -> None:
    pass


# ---------------------------------------------------------------------------
# Move selection
# ---------------------------------------------------------------------------

def choose_move(
    game_state: typing.Dict, mode: str
) -> typing.Tuple[str, typing.Dict]:
    state = make_state_from_game(game_state)
    legal = get_legal_moves(state, state["you_id"])
    heuristic_choice, heuristic_scores = heuristic_move(game_state)

    if not legal:
        return fallback_move(state, state["you_id"]), {
            "mode": "fallback",
            "heuristic_scores": heuristic_scores,
        }

    # Random agent
    if mode == "random":
        return random.choice(legal), {
            "mode": "random",
            "legal": legal,
        }

    # Pure heuristic agent
    if mode == "heuristic":
        return heuristic_choice, {
            "mode": "heuristic",
            "heuristic_scores": heuristic_scores,
        }

    # MCTS agents
    cfg = MCTS_CONFIGS.get(mode, ALL_IMPROVEMENTS_CONFIG)
    mcts_choice, mcts_stats = mcts_move(game_state, config=cfg)

    if mcts_choice not in legal:
        mcts_choice = heuristic_choice
    else:
        # Heuristic safety fallback: if MCTS avg < 0 and heuristic strongly disagrees
        child_stats = mcts_stats.get("children", {}).get(mcts_choice, {})
        mcts_avg = child_stats.get("avg")
        chosen_heuristic = heuristic_scores.get(mcts_choice, float("-inf"))
        best_heuristic = heuristic_scores.get(heuristic_choice, float("-inf"))
        if (
            heuristic_choice in legal
            and mcts_choice != heuristic_choice
            and mcts_avg is not None
            and mcts_avg < 0
            and best_heuristic - chosen_heuristic > 60
        ):
            mcts_choice = heuristic_choice

    return mcts_choice, {
        "mode": mode,
        "heuristic_scores": heuristic_scores,
        "mcts": mcts_stats,
        "heuristic_choice": heuristic_choice,
    }


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------

def make_handlers(mode: str) -> typing.Dict:
    def move_handler(game_state: typing.Dict) -> typing.Dict:
        next_move, debug = choose_move(game_state, mode)
        print(f"{mode.upper()} MOVE {game_state['turn']}: {next_move} | {debug}")
        return {"move": next_move}

    return {
        "info": lambda: info(mode),
        "start": start,
        "move": move_handler,
        "end": end,
    }


def run_agent(mode: str) -> None:
    from server import run_server
    run_server(make_handlers(mode))
