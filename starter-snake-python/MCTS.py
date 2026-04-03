import random
import copy
import math
import time
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
        # self.you = copy.deepcopy(game_state["you"]["body"]) # list of dicts with items x: int,y: int.
        your_id = game_state["you"]["id"]
        self.you_index = next(i for i, s in enumerate(game_state["board"]["snakes"]) if s["id"] == your_id)
        self.snakes = [copy.deepcopy(s["body"]) for s in game_state["board"]["snakes"]] # list of dicts with items id: string.
        self.you = self.snakes[self.you_index]
        self.hazards = game_state['board']['hazards'] # list of dicts with items x: int,y: int.
        self.food = copy.deepcopy(game_state["board"]["food"]) # list of dicts with items x: int,y: int.
        self.board_width = game_state["board"]["width"] # int
        self.board_height = game_state["board"]["height"] # int
        self.health = game_state["you"]["health"]

    def copy(self):
        return copy.deepcopy(self)


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

    def uct(self):
        # Finds max option, given that the node is not fully expanded yet
        # Also makes sure there is no division by 0 (= inf)
        options = []
        for child in self.children:
            if child.visits == 0:
                ucb = float("inf")
            else:
                ucb = (child.value / child.visits) + 1.4 * math.sqrt(math.log(self.visits+1) / child.visits)
            options.append((child, ucb))

        # Returns random action if >1 child nodes share same max value
        max_ucb = max(options, key=lambda x: x[1])[1]
        best = [opt for opt in options if opt[1] == max_ucb]
        return random.choice(best)[0]


    def __repr__(self):
        return self.move, self.value, self.visits

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
    elif my_neck["y"] > my_head["y"]:  # Neck is below head, don't move down
        is_move_safe["down"] = False
    elif my_neck["y"] < my_head["y"]:  # Neck is above head, don't move up
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
    new_state = state.copy()
    print(f"new state you_index: {new_state.you_index}")
    print(f"new state you: {new_state.you}")
    # new_heads = {}
    # --- MOVE ALL SNAKES ---
    for i, snake in enumerate(new_state.snakes):
        # If the snake is our own, do the given simulated move; sim_move
        if i == new_state.you_index:
            move = sim_move
            # after mutation:
            new_state.you = new_state.snakes[i]
        else:
            # TODO: If heuristic other snakes, use here. Also, snakes can make illegal moves now
            move = random.choice(["up", "down", "left", "right"])

        head = snake[0]
        if move == "up":
            new_head = {"x": head["x"], "y": head["y"] + 1}
        elif move == "down":
            new_head = {"x": head["x"], "y": head["y"] - 1}
        elif move == "left":
            new_head = {"x": head["x"] - 1, "y": head["y"]}
        elif move == "right":
            new_head = {"x": head["x"] + 1, "y": head["y"]}

        snake.insert(0, new_head)
        # new_heads[i] = new_head

        # --- FOOD ---
    for i, snake in enumerate(new_state.snakes):
        new_head = snake[0]

        ate_food = False
        for f in new_state.food[:]:
            if f["x"] == new_head["x"] and f["y"] == new_head["y"]:
                new_state.food.remove(f)
                ate_food = True
                if i == new_state.you_index:
                    new_state.health = 100

        if not ate_food:
            snake.pop()  # Tail moves forward
            if i == new_state.you_index:
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
    # (same thing?)
    # This fucked it up, because it's a nested list
    # if len(state.snakes)[0] == 1:
    #     print(state.snakes)
    #     return True
    return state.is_dead


def get_reward(state: State, turns_survived: int):
    if state.is_dead:
        return turns_survived * 0.1  # Small reward for surviving longer

    # Large reward for being alive + length + health bonus
    return 100 + (len(state.you) * 10) + state.health

# =========================
# MCTS CORE
# =========================

def expand(node):
    '''
    Picks random move to expand if no node after parent has been explored
    '''
    random_index = random.randrange(len(node.untried_moves))
    move = node.untried_moves.pop(random_index)
    new_state = next_state(node.state, move)
    child = Node(new_state, parent=node, move=move)
    node.children.append(child)
    return child

def simulate(state: State):
    """
    Rollout (play randomly until terminal)
    """
    rollout_state = state.copy()
    turns = 0

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
        turns += 1

    return get_reward(rollout_state, turns)


def backpropagate(node, reward):
    """
    Backpropagation step
    """
    # print(node.parent)
    while node.parent is not None:
        node.visits += 1
        node.value += reward
        node = node.parent


def mcts(root_state, deadline=.9):
    root = Node(root_state)

    deadline = time.time() + deadline
    while time.time() < deadline:
        node = root
        # 1. Selection
        # aka selecting child of highest uct, given that each node is fully expanded
        a = 0
        while node.is_fully_expanded() and node.children:
            a+=1
            node = node.uct()
        # if a > 0:
        #     print(a)

        # 2. Expansion
        # The node we have arrived at, has untried moves which we want to expand, the overwritten node is a random child
        # print(node.is_fully_expanded(), is_terminal(node.state))
        # print( "=============")
        if not node.is_fully_expanded() and not is_terminal(node.state):
            # print("expanding")
            node = expand(node)

        # 3. Simulation (rollout)
        reward = simulate(node.state)

        # 4. Backpropagation
        backpropagate(node, reward)

    # Choose best move
    if not root.children:
        return random.choice(get_legal_moves_state(root_state))
    best_child = max(root.children, key=lambda c: c.visits)

    return best_child.move


# =========================
# AGENT
# =========================

def mcts_move(game_state):
    state = State(game_state)
    return mcts(state)