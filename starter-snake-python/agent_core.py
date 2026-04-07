import copy
import math
import random
from collections import deque

MOVES = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

MOVE_ORDER = ("up", "down", "left", "right")
INF_NEG = float("-inf")


def coord(point):
    return (point["x"], point["y"])


def in_bounds(pos, width, height):
    x, y = pos
    return 0 <= x < width and 0 <= y < height


def add_move(pos, move_name):
    dx, dy = MOVES[move_name]
    return (pos[0] + dx, pos[1] + dy)


def clone_state(state):
    return {
        "turn": state["turn"],
        "width": state["width"],
        "height": state["height"],
        "food": set(state["food"]),
        "hazards": dict(state["hazards"]),
        "snakes": [
            {
                "id": snake["id"],
                "name": snake["name"],
                "health": snake["health"],
                "body": list(snake["body"]),
            }
            for snake in state["snakes"]
        ],
        "you_id": state["you_id"],
    }


def make_state_from_game(game_state):
    snakes = []
    for snake in game_state["board"]["snakes"]:
        snakes.append(
            {
                "id": snake["id"],
                "name": snake.get("name", snake["id"]),
                "health": int(snake["health"]),
                "body": [coord(segment) for segment in snake["body"]],
            }
        )

    hazards = {}
    for hazard in game_state["board"]["hazards"]:
        pos = coord(hazard)
        hazards[pos] = hazards.get(pos, 0) + 1

    return {
        "turn": int(game_state["turn"]),
        "width": int(game_state["board"]["width"]),
        "height": int(game_state["board"]["height"]),
        "food": {coord(food) for food in game_state["board"]["food"]},
        "hazards": hazards,
        "snakes": snakes,
        "you_id": game_state["you"]["id"],
    }


def get_snake(state, snake_id):
    for snake in state["snakes"]:
        if snake["id"] == snake_id:
            return snake
    return None


def snake_length(snake):
    return len(snake["body"])


def tail_stays_next_turn(snake):
    body = snake["body"]
    if len(body) < 2:
        return True
    return body[-1] == body[-2]


def moving_body_cells(state):
    occupied = set()
    for snake in state["snakes"]:
        body = snake["body"]
        last_index = len(body) - 1
        keep_tail = tail_stays_next_turn(snake)
        for index, segment in enumerate(body):
            if index == last_index and not keep_tail:
                continue
            occupied.add(segment)
    return occupied


def enemy_head_zones(state, snake_id):
    zones = {}
    for snake in state["snakes"]:
        if snake["id"] == snake_id:
            continue
        for move_name in MOVE_ORDER:
            pos = add_move(snake["body"][0], move_name)
            if in_bounds(pos, state["width"], state["height"]):
                zones.setdefault(pos, []).append(snake)
    return zones


def flood_fill(start, blocked, width, height, limit=None):
    if start in blocked or not in_bounds(start, width, height):
        return 0
    queue = deque([start])
    visited = {start}
    count = 0
    while queue:
        pos = queue.popleft()
        count += 1
        if limit is not None and count >= limit:
            return count
        for move_name in MOVE_ORDER:
            nxt = add_move(pos, move_name)
            if (
                in_bounds(nxt, width, height)
                and nxt not in blocked
                and nxt not in visited
            ):
                visited.add(nxt)
                queue.append(nxt)
    return count


def distance_map(starts, blocked, width, height, limit=None):
    result = {}
    queue = deque()
    for start in starts:
        if not in_bounds(start, width, height):
            continue
        result[start] = 0
        queue.append(start)
    while queue:
        pos = queue.popleft()
        dist = result[pos]
        if limit is not None and dist >= limit:
            continue
        for move_name in MOVE_ORDER:
            nxt = add_move(pos, move_name)
            if (
                in_bounds(nxt, width, height)
                and nxt not in blocked
                and nxt not in result
            ):
                result[nxt] = dist + 1
                queue.append(nxt)
    return result


def nearest_food_distance(state, snake_id):
    snake = get_snake(state, snake_id)
    if snake is None or not state["food"]:
        return None
    blocked = moving_body_cells(state) - {snake["body"][-1]}
    distances = distance_map([snake["body"][0]], blocked, state["width"], state["height"], limit=32)
    best = None
    for food in state["food"]:
        if food in distances:
            if best is None or distances[food] < best:
                best = distances[food]
    return best


def get_legal_moves(state, snake_id):
    snake = get_snake(state, snake_id)
    if snake is None:
        return []

    blocked = moving_body_cells(state)
    tail = snake["body"][-1]
    blocked_without_own_tail = set(blocked)
    blocked_without_own_tail.discard(tail)

    legal = []
    for move_name in MOVE_ORDER:
        nxt = add_move(snake["body"][0], move_name)
        if not in_bounds(nxt, state["width"], state["height"]):
            continue
        if nxt in blocked_without_own_tail:
            continue
        legal.append(move_name)
    return legal


def fallback_move(state, snake_id):
    snake = get_snake(state, snake_id)
    if snake is None:
        return "up"
    head = snake["body"][0]
    for move_name in MOVE_ORDER:
        if in_bounds(add_move(head, move_name), state["width"], state["height"]):
            return move_name
    return "up"


def local_move_score(state, snake_id, move_name):
    snake = get_snake(state, snake_id)
    if snake is None:
        return INF_NEG

    width = state["width"]
    height = state["height"]
    head = snake["body"][0]
    nxt = add_move(head, move_name)
    if not in_bounds(nxt, width, height):
        return INF_NEG

    blocked = moving_body_cells(state)
    tail = snake["body"][-1]
    blocked.discard(tail)
    if nxt in blocked:
        return INF_NEG

    post_health = snake["health"] - 1
    if nxt in state["food"]:
        post_health = 100
    if nxt in state["hazards"]:
        post_health -= 14 * state["hazards"][nxt]
    if post_health <= 0:
        return INF_NEG

    temp_blocked = set(blocked)
    temp_blocked.add(head)
    reachable = flood_fill(nxt, temp_blocked, width, height, limit=width * height)
    if reachable == 0:
        return INF_NEG

    score = reachable * 4.0
    if reachable < snake_length(snake):
        score -= 120 + (snake_length(snake) - reachable) * 18

    danger = enemy_head_zones(state, snake_id)
    for enemy in danger.get(nxt, []):
        enemy_len = snake_length(enemy)
        if enemy_len >= snake_length(snake):
            score -= 150
        else:
            score += 20

    open_neighbors = 0
    for next_name in MOVE_ORDER:
        adj = add_move(nxt, next_name)
        if in_bounds(adj, width, height) and adj not in temp_blocked:
            open_neighbors += 1
    score += open_neighbors * 8

    if nxt in state["hazards"]:
        score -= 35 * state["hazards"][nxt]
        score -= max(0, 45 - post_health)

    if nxt in state["food"]:
        if snake["health"] < 35:
            score += 120
        elif snake["health"] < 60:
            score += 70
        else:
            score += 25

    food_dist = nearest_food_distance(state, snake_id)
    if food_dist is not None:
        urgency = 3.5 if snake["health"] < 25 else 1.8 if snake["health"] < 50 else 0.5
        score -= food_dist * urgency

    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    score -= abs(nxt[0] - center[0]) + abs(nxt[1] - center[1])

    for enemy in state["snakes"]:
        if enemy["id"] == snake_id:
            continue
        dist = abs(nxt[0] - enemy["body"][0][0]) + abs(nxt[1] - enemy["body"][0][1])
        if snake_length(enemy) >= snake_length(snake) and dist <= 2:
            score -= (3 - dist) * 16
        elif snake_length(enemy) < snake_length(snake) and dist <= 2:
            score += (3 - dist) * 8

    return score


def choose_greedy_move(state, snake_id, stochastic=False):
    legal = get_legal_moves(state, snake_id)
    if not legal:
        return fallback_move(state, snake_id)

    scored = [(move_name, local_move_score(state, snake_id, move_name)) for move_name in legal]
    scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
    if not stochastic or len(scored) == 1:
        return scored[0][0]

    top_score = scored[0][1]
    shortlist = [item for item in scored if item[1] >= top_score - 12]
    return random.choice(shortlist)[0]


def select_moves_for_turn(state, planned_moves=None, stochastic=False):
    planned_moves = planned_moves or {}
    selected = {}
    for snake in state["snakes"]:
        sid = snake["id"]
        if sid in planned_moves:
            selected[sid] = planned_moves[sid]
        else:
            selected[sid] = choose_greedy_move(state, sid, stochastic=stochastic)
    return selected


def advance_state(state, planned_moves=None, stochastic=False):
    planned_moves = planned_moves or {}
    new_state = clone_state(state)
    selected_moves = select_moves_for_turn(state, planned_moves=planned_moves, stochastic=stochastic)
    eaten_food = set()

    for snake in new_state["snakes"]:
        sid = snake["id"]
        new_head = add_move(snake["body"][0], selected_moves[sid])
        ate_food = new_head in new_state["food"]
        snake["body"].insert(0, new_head)
        if not ate_food:
            snake["body"].pop()
            snake["health"] -= 1
        else:
            snake["health"] = 100
            eaten_food.add(new_head)
        if new_head in new_state["hazards"]:
            snake["health"] -= 14 * new_state["hazards"][new_head]

    dead_ids = set()

    for snake in new_state["snakes"]:
        head = snake["body"][0]
        if (
            not in_bounds(head, new_state["width"], new_state["height"])
            or snake["health"] <= 0
        ):
            dead_ids.add(snake["id"])

    heads = {}
    for snake in new_state["snakes"]:
        if snake["id"] in dead_ids:
            continue
        heads.setdefault(snake["body"][0], []).append(snake)

    for position, contenders in heads.items():
        if len(contenders) < 2:
            continue
        max_length = max(snake_length(snake) for snake in contenders)
        winners = [snake for snake in contenders if snake_length(snake) == max_length]
        if len(winners) > 1:
            for snake in contenders:
                dead_ids.add(snake["id"])
        else:
            for snake in contenders:
                if snake_length(snake) < max_length:
                    dead_ids.add(snake["id"])

    body_cells = set()
    for snake in new_state["snakes"]:
        for segment in snake["body"][1:]:
            body_cells.add(segment)

    for snake in new_state["snakes"]:
        if snake["id"] in dead_ids:
            continue
        if snake["body"][0] in body_cells:
            dead_ids.add(snake["id"])

    new_state["snakes"] = [snake for snake in new_state["snakes"] if snake["id"] not in dead_ids]
    new_state["food"] -= eaten_food
    new_state["turn"] += 1
    return new_state


def territory_score(state, snake_id):
    you = get_snake(state, snake_id)
    if you is None:
        return -999

    blocked = moving_body_cells(state)
    blocked.discard(you["body"][-1])
    width = state["width"]
    height = state["height"]

    my_distances = distance_map([you["body"][0]], blocked, width, height, limit=24)
    enemy_maps = []
    for enemy in state["snakes"]:
        if enemy["id"] == snake_id:
            continue
        enemy_blocked = set(blocked)
        enemy_blocked.discard(enemy["body"][-1])
        enemy_maps.append(distance_map([enemy["body"][0]], enemy_blocked, width, height, limit=24))

    score = 0
    for x in range(width):
        for y in range(height):
            pos = (x, y)
            if pos not in my_distances:
                continue
            my_dist = my_distances[pos]
            better = True
            tied = False
            for enemy_map in enemy_maps:
                enemy_dist = enemy_map.get(pos)
                if enemy_dist is None:
                    continue
                if enemy_dist < my_dist:
                    better = False
                    break
                if enemy_dist == my_dist:
                    tied = True
            if better and not tied:
                score += 1
            elif better:
                score += 0.2
    return score


def evaluate_state(state, snake_id):
    snake = get_snake(state, snake_id)
    if snake is None:
        return -1_000_000.0
    if len(state["snakes"]) == 1 and state["snakes"][0]["id"] == snake_id:
        return 1_000_000.0 + snake_length(snake) * 100 + snake["health"]

    blocked = moving_body_cells(state)
    blocked.discard(snake["body"][-1])
    area = flood_fill(
        snake["body"][0],
        blocked - {snake["body"][0]},
        state["width"],
        state["height"],
        limit=state["width"] * state["height"],
    )
    safe_moves = len(get_legal_moves(state, snake_id))
    food_dist = nearest_food_distance(state, snake_id)
    territory = territory_score(state, snake_id)
    head = snake["body"][0]

    score = 0.0
    score += 260.0
    score += snake_length(snake) * 35.0
    score += snake["health"] * 1.2
    score += area * 6.0
    score += safe_moves * 22.0
    score += territory * 4.0

    if food_dist is not None:
        if snake["health"] < 20:
            score -= food_dist * 14.0
        elif snake["health"] < 45:
            score -= food_dist * 6.0
        else:
            score -= food_dist * 1.2

    if head in state["hazards"]:
        score -= state["hazards"][head] * 55.0

    for enemy in state["snakes"]:
        if enemy["id"] == snake_id:
            continue
        enemy_dist = abs(head[0] - enemy["body"][0][0]) + abs(head[1] - enemy["body"][0][1])
        if snake_length(enemy) >= snake_length(snake) and enemy_dist <= 2:
            score -= (3 - enemy_dist) * 25.0
        elif snake_length(enemy) < snake_length(snake) and enemy_dist <= 2:
            score += (3 - enemy_dist) * 12.0

    return score


def is_terminal_state(state, snake_id):
    snake = get_snake(state, snake_id)
    if snake is None:
        return True
    return len(state["snakes"]) == 1 and state["snakes"][0]["id"] == snake_id


def fast_evaluate_state(state, snake_id):
    snake = get_snake(state, snake_id)
    if snake is None:
        return -1_000_000.0
    if len(state["snakes"]) == 1 and state["snakes"][0]["id"] == snake_id:
        return 1_000_000.0 + snake_length(snake) * 100 + snake["health"]

    blocked = moving_body_cells(state)
    blocked.discard(snake["body"][-1])
    area = flood_fill(
        snake["body"][0],
        blocked - {snake["body"][0]},
        state["width"],
        state["height"],
        limit=min(40, state["width"] * state["height"]),
    )
    safe_moves = len(get_legal_moves(state, snake_id))
    head = snake["body"][0]

    score = 0.0
    score += snake_length(snake) * 28.0
    score += snake["health"] * 1.1
    score += area * 5.5
    score += safe_moves * 18.0

    if head in state["hazards"]:
        score -= state["hazards"][head] * 45.0

    for enemy in state["snakes"]:
        if enemy["id"] == snake_id:
            continue
        enemy_dist = abs(head[0] - enemy["body"][0][0]) + abs(head[1] - enemy["body"][0][1])
        if snake_length(enemy) >= snake_length(snake) and enemy_dist <= 2:
            score -= (3 - enemy_dist) * 22.0
        elif snake_length(enemy) < snake_length(snake) and enemy_dist <= 2:
            score += (3 - enemy_dist) * 10.0

    return score


def heuristic_move_scores(state, snake_id):
    legal = get_legal_moves(state, snake_id)
    if not legal:
        return {}

    scores = {}
    for move_name in legal:
        local = local_move_score(state, snake_id, move_name)
        future = advance_state(state, planned_moves={snake_id: move_name}, stochastic=False)
        total = local + 0.85 * evaluate_state(future, snake_id)
        if get_snake(future, snake_id) is None:
            total = INF_NEG
        else:
            next_legal = get_legal_moves(future, snake_id)
            if not next_legal:
                total -= 180
            if len(future["snakes"]) < len(state["snakes"]):
                total += 50
        scores[move_name] = total
    return scores


def choose_heuristic_move(state, snake_id):
    scores = heuristic_move_scores(state, snake_id)
    if not scores:
        return fallback_move(state, snake_id), {}
    best_move = max(scores.items(), key=lambda item: (item[1], item[0]))[0]
    return best_move, scores


def heuristic_move(game_state):
    state = make_state_from_game(game_state)
    return choose_heuristic_move(state, state["you_id"])


def rollout_move(state, snake_id):
    legal = get_legal_moves(state, snake_id)
    if not legal:
        return fallback_move(state, snake_id)

    scored = []
    for move_name in legal:
        score = local_move_score(state, snake_id, move_name)
        scored.append((move_name, score))
    scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
    top_score = scored[0][1]
    shortlist = [item for item in scored if item[1] >= top_score - 18]
    return random.choice(shortlist)[0]
