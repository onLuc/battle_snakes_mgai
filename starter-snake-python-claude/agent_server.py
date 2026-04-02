import typing

from MCTS import mcts_move
from agent_core import fallback_move, get_legal_moves, heuristic_move, make_state_from_game


def info(mode: str) -> typing.Dict:
    if mode == "heuristic":
        color = "#D97706"
        head = "sand-worm"
        tail = "round-bum"
        author = "HeuristicKing"
    else:
        color = "#00CC00"
        head = "evil"
        tail = "bolt"
        author = "MCTSKing"

    return {
        "apiversion": "1",
        "author": author,
        "color": color,
        "head": head,
        "tail": tail,
    }


def start(game_state: typing.Dict):
    pass


def end(game_state: typing.Dict):
    pass


def choose_move(game_state: typing.Dict, mode: str):
    state = make_state_from_game(game_state)
    legal = get_legal_moves(state, state["you_id"])
    heuristic_choice, heuristic_scores = heuristic_move(game_state)

    if not legal:
        return fallback_move(state, state["you_id"]), {
            "mode": "fallback",
            "heuristic_scores": heuristic_scores,
        }

    if mode == "heuristic":
        return heuristic_choice, {
            "mode": "heuristic",
            "heuristic_scores": heuristic_scores,
        }

    mcts_choice, mcts_stats = mcts_move(game_state)
    if mcts_choice not in legal:
        mcts_choice = heuristic_choice
    else:
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
        "mode": "mcts",
        "heuristic_scores": heuristic_scores,
        "mcts": mcts_stats,
        "heuristic_choice": heuristic_choice,
    }


def make_move_handler(mode: str):
    def move(game_state: typing.Dict) -> typing.Dict:
        next_move, debug = choose_move(game_state, mode)
        print(f"{mode.upper()} MOVE {game_state['turn']}: {next_move} | {debug}")
        return {"move": next_move}

    return move


def make_handlers(mode: str):
    move_handler = make_move_handler(mode)
    return {
        "info": lambda: info(mode),
        "start": start,
        "move": move_handler,
        "end": end,
    }


def run_agent(mode: str):
    from server import run_server

    run_server(make_handlers(mode))
