# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com


import random
import typing
from typing import AnyStr
from MCTS_old import mcts_move

SNAKE_TACTIC = "MCTS"

# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "Denominator",
        "color": "#00CC00",
        "head": "smile",
        "tail": "bolt",
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


def get_legal_moves(game_state: typing.Dict) -> list[AnyStr]:
    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        is_move_safe["left"] = False
    elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
        is_move_safe["right"] = False
    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        is_move_safe["down"] = False
    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        is_move_safe["up"] = False

    # Prevent your Battlesnake from moving out of bounds
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']

    if my_head["x"] == board_width - 1:
        is_move_safe["right"] = False
    if my_head["x"] == 0:
        is_move_safe["left"] = False
    if my_head["y"] == board_height - 1:
        is_move_safe["up"] = False
    if my_head["y"] == 0:
        is_move_safe["down"] = False

    # Prevent your Battlesnake from colliding with itself or others
    # all_snakes includes yourself, so self-collision is implicitly handled
    all_snakes = game_state['board']['snakes']

    for snake in all_snakes:
        for body_part in snake['body']:
            if my_head["x"] + 1 == body_part["x"] and my_head["y"] == body_part["y"]:
                is_move_safe["right"] = False
            if my_head["x"] - 1 == body_part["x"] and my_head["y"] == body_part["y"]:
                is_move_safe["left"] = False
            if my_head["y"] + 1 == body_part["y"] and my_head["x"] == body_part["x"]:
                is_move_safe["up"] = False
            if my_head["y"] - 1 == body_part["y"] and my_head["x"] == body_part["x"]:
                is_move_safe["down"] = False

    # Prevent your Battlesnake from running into hazards
    hazards = game_state['board']['hazards']

    for hazard in hazards:
        if hazard["x"] == my_head["x"] + 1 and hazard["y"] == my_head["y"]:
            is_move_safe["right"] = False
        elif hazard["x"] == my_head["x"] - 1 and hazard["y"] == my_head["y"]:
            is_move_safe["left"] = False
        elif hazard["y"] + 1 == my_head["y"] and hazard["x"] == my_head["x"]:
            is_move_safe["up"] = False
        elif hazard["y"] - 1 == my_head["y"] and hazard["x"] == my_head["x"]:
            is_move_safe["down"] = False

    # Are there any safe moves left?
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            safe_moves.append(move)

    return safe_moves

# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:

    safe_moves = get_legal_moves(game_state)
    if len(safe_moves) == 0:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    # Choose a random move from the safe ones
    print(f"Safe moves {safe_moves}")
    next_move = random.choice(safe_moves)
    if SNAKE_TACTIC == "MCTS":
        next_move = mcts_move(game_state)

    # TODO: Step 4 - Move towards food instead of random, to regain health and survive longer
    # food = game_state['board']['food']

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
