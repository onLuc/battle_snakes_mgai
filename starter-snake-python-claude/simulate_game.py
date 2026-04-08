"""
Headless BattleSnake game simulator.

Does NOT import from MCTS.py — agents are passed as plain callables.
"""

import random
import uuid
from typing import Callable, Dict, List, Optional, Tuple

from agent_core import advance_state, fallback_move, get_legal_moves

# ---------------------------------------------------------------------------
# Hazard schedule (hz_hazard_pits ruleset)
# ---------------------------------------------------------------------------

# Default pit center positions for 11x11 board
_PIT_CENTERS_11x11 = [(3, 3), (3, 7), (7, 3), (7, 7)]


def _get_hazard_stacks(turn: int) -> int:
    """
    Hazard pit stack schedule:
      - turn < 26: 0 stacks
      - turn 26–100: stacks = min(4, 1 + (turn-26)//25)
      - turn 101–175: max 4 stacks (held for 75 turns)
      - turn 176+: drain every 25 turns
    """
    if turn < 26:
        return 0
    ramp_stacks = min(4, 1 + (turn - 26) // 25)
    if turn < 176:
        return ramp_stacks
    # After turn 176 drain at same rate
    drained = (turn - 176) // 25
    return max(0, 4 - drained)


def _compute_hazards(turn: int, pit_positions: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    """Return {(x,y): stack_count} for active hazards."""
    stacks = _get_hazard_stacks(turn)
    if stacks == 0 or not pit_positions:
        return {}
    return {pos: stacks for pos in pit_positions}


# ---------------------------------------------------------------------------
# Food management
# ---------------------------------------------------------------------------

def _spawn_food(
    state: dict,
    rng: random.Random,
    min_food: int = 2,
    chance: float = 0.25,
) -> dict:
    """Attempt to spawn food; modifies state in-place and returns it."""
    occupied = set()
    for snake in state["snakes"]:
        for segment in snake["body"]:
            occupied.add(segment)
    occupied |= state["food"]

    width = state["width"]
    height = state["height"]
    all_cells = [(x, y) for x in range(width) for y in range(height)]
    free_cells = [c for c in all_cells if c not in occupied]

    if not free_cells:
        return state

    need_spawn = len(state["food"]) < min_food or rng.random() < chance
    if need_spawn:
        pos = rng.choice(free_cells)
        state["food"].add(pos)

    return state


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------

def _init_state(
    snake_names: List[str],
    width: int,
    height: int,
    rng: random.Random,
) -> dict:
    """Create the initial internal state dict."""
    n = len(snake_names)

    # Standard spawn positions (for 11x11)
    if n <= 2:
        base_spawns = [(1, height - 2), (width - 2, 1)]
    else:
        base_spawns = [
            (1, height - 2),
            (width - 2, height - 2),
            (width - 2, 1),
            (1, 1),
        ]

    # Extend if we somehow have more snakes than corners
    while len(base_spawns) < n:
        base_spawns.append((width // 2, height // 2))

    spawn_positions = base_spawns[:n]
    rng.shuffle(spawn_positions)

    snakes = []
    for i, name in enumerate(snake_names):
        x, y = spawn_positions[i]
        body = [(x, y), (x, y), (x, y)]
        snakes.append({
            "id": f"snake-{i}",
            "name": name,
            "health": 100,
            "body": body,
        })

    state: dict = {
        "turn": 0,
        "width": width,
        "height": height,
        "food": set(),
        "hazards": {},
        "snakes": snakes,
        "you_id": snakes[0]["id"],  # placeholder, overridden per-agent call
    }

    # Place initial food (min 2, no random chance at turn 0)
    _spawn_food(state, rng, min_food=2, chance=0.0)

    return state


# ---------------------------------------------------------------------------
# Internal → API conversion
# ---------------------------------------------------------------------------

def _internal_to_api(state: dict, snake_id: str, game_id: str = "sim") -> dict:
    """Convert internal state to the BattleSnake API dict format."""

    def _seg(x: int, y: int) -> Dict:
        return {"x": x, "y": y}

    # Build hazards list: repeat entry for each stack
    hazard_list = []
    for (x, y), stacks in state["hazards"].items():
        for _ in range(stacks):
            hazard_list.append(_seg(x, y))

    board_snakes = []
    for snake in state["snakes"]:
        body_list = [_seg(x, y) for (x, y) in snake["body"]]
        board_snakes.append({
            "id": snake["id"],
            "name": snake["name"],
            "health": snake["health"],
            "body": body_list,
            "head": body_list[0] if body_list else _seg(0, 0),
            "length": len(snake["body"]),
            "latency": "0",
            "shout": "",
            "squad": "",
            "customizations": {"color": "#888888", "head": "default", "tail": "default"},
        })

    # Find "you" snake
    you_snake = next((s for s in board_snakes if s["id"] == snake_id), None)
    if you_snake is None and board_snakes:
        you_snake = board_snakes[0]

    food_list = [_seg(x, y) for (x, y) in state["food"]]

    return {
        "game": {
            "id": game_id,
            "ruleset": {
                "name": "standard",
                "version": "v1.2.3",
                "settings": {
                    "foodSpawnChance": 25,
                    "minimumFood": 2,
                    "hazardDamagePerTurn": 14,
                },
            },
            "timeout": 500,
            "source": "simulation",
        },
        "turn": state["turn"],
        "board": {
            "height": state["height"],
            "width": state["width"],
            "food": food_list,
            "hazards": hazard_list,
            "snakes": board_snakes,
        },
        "you": you_snake if you_snake else {
            "id": snake_id,
            "name": "dead",
            "health": 0,
            "body": [],
            "head": _seg(0, 0),
            "length": 0,
            "latency": "0",
            "shout": "",
            "squad": "",
            "customizations": {},
        },
    }


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def run_game(
    agents: List[Tuple[str, Callable]],
    width: int = 11,
    height: int = 11,
    seed: Optional[int] = None,
    max_turns: int = 300,
    pit_positions: Optional[List[Tuple[int, int]]] = None,
) -> dict:
    """
    Run a headless BattleSnake game.

    Parameters
    ----------
    agents  : list of (name_str, callable(game_state_dict) -> move_str)
    width, height : board dimensions
    seed    : random seed for reproducibility
    max_turns : cap on simulation length
    pit_positions : hazard pit centres (defaults to 11x11 standard if width==11)

    Returns
    -------
    dict with keys:
        "turns"   : int
        "results" : list of result dicts sorted by placement
    """
    rng = random.Random(seed)

    if pit_positions is None and width == 11 and height == 11:
        pit_positions = _PIT_CENTERS_11x11
    elif pit_positions is None:
        pit_positions = []

    snake_names = [name for name, _fn in agents]
    state = _init_state(snake_names, width, height, rng)

    # Map snake_id → agent callable
    agent_map: Dict[str, Callable] = {}
    name_map: Dict[str, str] = {}
    for i, (name, fn) in enumerate(agents):
        sid = f"snake-{i}"
        agent_map[sid] = fn
        name_map[sid] = name

    death_turn: Dict[str, int] = {}
    last_known: Dict[str, dict] = {}  # snake_id → snapshot dict

    alive_ids = set(agent_map.keys())

    for turn in range(max_turns):
        # Update hazards
        state["hazards"] = _compute_hazards(turn, pit_positions)

        # Snapshot living snakes
        for snake in state["snakes"]:
            last_known[snake["id"]] = {
                "length": len(snake["body"]),
                "health": snake["health"],
            }

        # Stop if 1 or 0 snakes remain
        if len(state["snakes"]) <= 1:
            break

        # Collect moves from all alive agents
        planned_moves: Dict[str, str] = {}
        for snake in state["snakes"]:
            sid = snake["id"]
            fn = agent_map.get(sid)
            if fn is None:
                planned_moves[sid] = fallback_move(state, sid)
                continue
            try:
                game_api = _internal_to_api(state, sid)
                move_str = fn(game_api)
                legal = get_legal_moves(state, sid)
                if move_str not in legal:
                    move_str = fallback_move(state, sid)
                planned_moves[sid] = move_str
            except Exception:
                planned_moves[sid] = fallback_move(state, sid)

        prev_ids = {s["id"] for s in state["snakes"]}

        # Advance state (no stochastic food — we handle food separately)
        state = advance_state(state, planned_moves=planned_moves, stochastic=False)

        # Detect newly dead snakes
        current_ids = {s["id"] for s in state["snakes"]}
        newly_dead = prev_ids - current_ids
        for sid in newly_dead:
            if sid not in death_turn:
                death_turn[sid] = turn + 1  # died after advancing

        # Spawn food
        state = _spawn_food(state, rng, min_food=2, chance=0.25)

    final_turn = state["turn"]

    # Build results
    final_survivors = {s["id"]: s for s in state["snakes"]}
    all_results = []

    for sid in agent_map:
        alive = sid in final_survivors
        if alive:
            turns_survived = final_turn
            final_length = len(final_survivors[sid]["body"])
            final_health = final_survivors[sid]["health"]
        else:
            turns_survived = death_turn.get(sid, 0)
            snap = last_known.get(sid, {})
            final_length = snap.get("length", 3)
            final_health = snap.get("health", 0)

        all_results.append({
            "id": sid,
            "name": name_map[sid],
            "alive": alive,
            "turns_survived": turns_survived,
            "final_length": final_length,
            "final_health": final_health,
            "placement": None,  # assigned below
        })

    # Sort: alive first, then turns_survived desc, then length desc
    all_results.sort(
        key=lambda r: (
            not r["alive"],
            -r["turns_survived"],
            -r["final_length"],
        )
    )

    for rank, r in enumerate(all_results, start=1):
        r["placement"] = rank

    return {
        "turns": final_turn,
        "results": all_results,
    }
