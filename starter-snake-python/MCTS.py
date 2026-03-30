import random
import copy
import math
import typing
from typing import AnyStr

# =========================
# STATE REPRESENTATION
# =========================

# board: https://docs.battlesnake.com/api/objects/board
# body:  https://docs.battlesnake.com/api/objects/battlesnake
# game:  https://docs.battlesnake.com/api/objects/game
class State:
    def __init__(self, game_state):
        self.is_dead = False
        self.you = copy.deepcopy(game_state["you"]["body"]) # list of dicts with items x: int,y: int.
        self.snakes = [copy.deepcopy(s["body"]) for s in game_state["board"]["snakes"]] # list of dicts with items id: string.
        self.hazards = game_state['board']['hazards'] # list of dicts with items x: int,y: int.
        self.food = copy.deepcopy(game_state["board"]["food"]) # list of dicts with items x: int,y: int.
        self.board_width = game_state["board"]["width"] # int
        self.board_height = game_state["board"]["height"] # int
        self.health = 100

    def copy(self):
        return copy.deepcopy(self)


# =========================
# GAME LOGIC (SIMULATION)
# =========================

def get_legal_moves_state(state: State)  -> list[AnyStr]:
    is_move_safe = {"up": True, "down": True, "left": True, "right": True}
    my_head = state.you[0]  # Coordinates of your head
    my_neck = state.you[1]  # Coordinates of your "neck"

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        is_move_safe["left"] = False
    elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
        is_move_safe["right"] = False
    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        is_move_safe["down"] = False
    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        is_move_safe["up"] = False

    # Prevent your Battlesnake from moving out of bounds
    if my_head["x"] == state.board_width - 1:
        is_move_safe["right"] = False
    if my_head["x"] == 0:
        is_move_safe["left"] = False
    if my_head["y"] == state.board_height - 1:
        is_move_safe["up"] = False
    if my_head["y"] == 0:
        is_move_safe["down"] = False

    # Prevent your Battlesnake from colliding with itself or others
    # all_snakes includes yourself, so self-collision is implicitly handled
    for snake in state.snakes:
        for body_part in snake:
            if my_head["x"] + 1 == body_part["x"] and my_head["y"] == body_part["y"]:
                is_move_safe["right"] = False
            if my_head["x"] - 1 == body_part["x"] and my_head["y"] == body_part["y"]:
                is_move_safe["left"] = False
            if my_head["y"] + 1 == body_part["y"] and my_head["x"] == body_part["x"]:
                is_move_safe["up"] = False
            if my_head["y"] - 1 == body_part["y"] and my_head["x"] == body_part["x"]:
                is_move_safe["down"] = False

    # Prevent your Battlesnake from running into hazards
    for hazard in state.hazards:
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


def next_state(state, sim_move):
    new_state = copy.deepcopy(state)

    # --- MOVE ALL SNAKES ---
    for snake in new_state.snakes:

        possible_moves = ["up", "down", "left", "right"]
        # Simulate the other snakes randomly TODO: If heuristic other snakes, use here
        move = random.choice(possible_moves)
        # If the snake is our own, do the given simulated move; sim_move
        if snake == new_state.you:
            move = sim_move

        head = snake[0]

        if move == "up":
            new_head = {"x": head["x"], "y": head["y"] + 1}
        elif move == "down":
            new_head = {"x": head["x"], "y": head["y"] - 1}
        elif move == "left":
            new_head = {"x": head["x"] - 1, "y": head["y"]}
        else:
            new_head = {"x": head["x"] + 1, "y": head["y"]}

        snake.insert(0, new_head)

        # --- FOOD ---
        ate_food = False
        for food in new_state.food:
            if food["x"] == new_head["x"] and food["y"] == new_head["y"]:
                ate_food = True
                new_state.food.remove(food)
                new_state.health = 100
                break

        if not ate_food:
            snake.pop()
            if snake == new_state.you:
                new_state.health -= 1

    # --- COLLISIONS ---
    head = new_state.you[0]

    # wall collision
    if head["x"] < 0 or head["x"] >= new_state.board_width:
        new_state.is_dead = True
    if head["y"] < 0 or head["y"] >= new_state.board_height:
        new_state.is_dead = True

    # self collision
    if head in new_state.you[1:]:
        new_state.is_dead = True

    # collision with other snakes
    for snake in new_state.snakes:
        if snake == new_state.you:
            continue
        if head in snake:
            new_state.is_dead = True

    # starvation
    if new_state.health <= 0:
        new_state.is_dead = True

    return new_state


def is_terminal(state: State):
    # TODO:
    # return True if:
    # - you died
    # - or game ended
    return state.is_dead


def get_reward(state: State):
    # TODO:
    # simplest:
    # +1 = alive
    # -1 = dead
    if state.is_dead:
        return -1
    return len(state.you)


# =========================
# MCTS NODE
# =========================

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move

        self.children = []
        self.untried_moves = get_legal_moves_state(state)
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0


# =========================
# MCTS CORE
# =========================

def select(node):
    """
    Selection step using UCB1
    """
    while node.children:
        node = max(node.children, key=ucb_score)
    return node


def ucb_score(child, C=1.4):
    if child.visits == 0:
        return float("inf")

    return (child.value / child.visits) + C * math.sqrt(math.log(child.parent.visits) / child.visits)


def expand(node):
    move = node.untried_moves.pop()
    new_state = next_state(node.state, move)
    child = Node(new_state, parent=node, move=move)
    node.children.append(child)
    return child


def simulate(state: State):
    """
    Rollout (play randomly until terminal)
    """
    rollout_state = state.copy()

    # TODO: limit rollout depth (important!)
    for _ in range(20):
        if is_terminal(rollout_state):
            break

        moves = get_legal_moves_state(rollout_state)
        if not moves:
            rollout_state.is_dead = True
            break

        move = random.choice(moves)
        rollout_state = next_state(rollout_state, move)

    return get_reward(rollout_state)


def backpropagate(node, reward):
    """
    Backpropagation step
    """
    while node:
        node.visits += 1
        node.value += reward
        node = node.parent


def mcts(root_state, iterations=200):
    root = Node(root_state)

    for _ in range(iterations): # should make this into while < 1 sec

        # 1. Selection
        node = select(root)

        # 2. Expansion
        if not is_terminal(node.state):
            node = expand(node)

        # 3. Simulation
        reward = simulate(node.state)

        # 4. Backpropagation
        backpropagate(node, reward)

    # Choose best move
    best_child = max(root.children, key=lambda c: c.visits)

    return best_child.move


# =========================
# AGENT
# =========================

def mcts_move(game_state):
    state = State(game_state)
    return mcts(state)