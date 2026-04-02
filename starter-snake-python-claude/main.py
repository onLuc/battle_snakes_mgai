import os

from agent_server import choose_move, run_agent

AGENT_MODE = os.environ.get("SNAKE_TACTIC", "mcts").strip().lower()


def move(game_state):
    next_move, debug = choose_move(game_state, AGENT_MODE)
    print(f"{AGENT_MODE.upper()} MOVE {game_state['turn']}: {next_move} | {debug}")
    return {"move": next_move}


if __name__ == "__main__":
    run_agent(AGENT_MODE)
