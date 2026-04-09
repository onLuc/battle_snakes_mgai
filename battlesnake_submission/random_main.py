import random
import typing

from agent_core import fallback_move, get_legal_moves, make_state_from_game


def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "RandomSnake",
        "color": "#2563EB",
        "head": "default",
        "tail": "default",
    }


def start(game_state: typing.Dict):
    pass


def end(game_state: typing.Dict):
    pass


def move(game_state: typing.Dict) -> typing.Dict:
    state = make_state_from_game(game_state)
    legal = get_legal_moves(state, state["you_id"])
    if legal:
        next_move = random.choice(legal)
    else:
        next_move = fallback_move(state, state["you_id"])
    print(f"RANDOM MOVE {game_state['turn']}: {next_move} | legal={legal}")
    return {"move": next_move}


if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
