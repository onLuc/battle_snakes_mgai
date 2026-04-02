import os
import math
import random
import time
from itertools import product

from agent_core import (
    advance_state,
    evaluate_state,
    fallback_move,
    get_legal_moves,
    heuristic_move_scores,
    is_terminal_state,
    make_state_from_game,
    rollout_move,
)


class Node:
    def __init__(self, state, snake_id, parent=None, move=None, priors=None):
        self.state = state
        self.snake_id = snake_id
        self.parent = parent
        self.move = move
        self.children = []
        self.priors = priors if priors is not None else heuristic_move_scores(state, snake_id)
        self.untried_moves = [
            move_name
            for move_name, _score in sorted(
                self.priors.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )
        ]
        self.visits = 0
        self.value = 0.0

    def is_terminal(self):
        return is_terminal_state(self.state, self.snake_id)

    def fully_expanded(self):
        return not self.untried_moves

    def expand(self):
        move_name = self.untried_moves.pop(0)
        next_state = advance_state(
            self.state,
            planned_moves={self.snake_id: move_name},
            stochastic=True,
        )
        child = Node(next_state, self.snake_id, parent=self, move=move_name)
        self.children.append(child)
        return child

    def best_child(self, exploration=1.05):
        best_score = float("-inf")
        best_nodes = []
        log_parent = math.log(max(1, self.visits))
        prior_values = list(self.priors.values())
        min_prior = min(prior_values) if prior_values else 0.0
        max_prior = max(prior_values) if prior_values else 1.0
        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = child.value / child.visits
                explore = exploration * math.sqrt(log_parent / child.visits)
                raw_prior = self.priors.get(child.move, 0.0)
                if max_prior > min_prior:
                    normalized_prior = (raw_prior - min_prior) / (max_prior - min_prior)
                else:
                    normalized_prior = 0.5
                prior_bonus = normalized_prior * 18.0 / (1 + child.visits)
                score = exploit + explore + prior_bonus
            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)
        return random.choice(best_nodes)


def likely_moves_for_snake(state, snake_id, top_n=2):
    scores = heuristic_move_scores(state, snake_id)
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    best_score = ranked[0][1]
    selected = []
    for move_name, score in ranked:
        if len(selected) >= top_n and score < best_score - 35:
            break
        if len(selected) < top_n or score >= best_score - 18:
            selected.append((move_name, score))
    return selected[:top_n] if selected else ranked[:top_n]


def opponent_joint_responses(state, snake_id, max_joint=6):
    opponents = [snake for snake in state["snakes"] if snake["id"] != snake_id]
    if not opponents:
        return [({}, 0.0)]

    move_options = []
    for opponent in opponents:
        ranked_moves = likely_moves_for_snake(state, opponent["id"], top_n=2)
        if not ranked_moves:
            continue
        move_options.append((opponent["id"], ranked_moves))

    if not move_options:
        return [({}, 0.0)]

    joint = []
    ids = [item[0] for item in move_options]
    option_lists = [item[1] for item in move_options]
    for combo in product(*option_lists):
        planned = {}
        score_sum = 0.0
        for opponent_id, (move_name, score) in zip(ids, combo):
            planned[opponent_id] = move_name
            score_sum += score
        joint.append((planned, score_sum))

    joint.sort(key=lambda item: item[1], reverse=True)
    return joint[:max_joint]


def opponent_aware_root_scores(state, snake_id):
    legal = get_legal_moves(state, snake_id)
    if not legal:
        return {}

    opponent_responses = opponent_joint_responses(state, snake_id)
    root_scores = {}

    for move_name in legal:
        scenario_values = []
        for response_moves, _response_score in opponent_responses:
            planned = {snake_id: move_name, **response_moves}
            next_state = advance_state(state, planned_moves=planned, stochastic=False)
            scenario_values.append(evaluate_state(next_state, snake_id))

        if not scenario_values:
            next_state = advance_state(state, planned_moves={snake_id: move_name}, stochastic=False)
            scenario_values.append(evaluate_state(next_state, snake_id))

        scenario_values.sort()
        worst = scenario_values[0]
        average = sum(scenario_values) / len(scenario_values)
        median = scenario_values[len(scenario_values) // 2]
        root_scores[move_name] = (0.55 * worst) + (0.30 * average) + (0.15 * median)

    return root_scores


def rollout(state, snake_id, depth=8):
    rollout_state = state
    for _ in range(depth):
        if is_terminal_state(rollout_state, snake_id):
            break
        move_name = rollout_move(rollout_state, snake_id)
        rollout_state = advance_state(
            rollout_state,
            planned_moves={snake_id: move_name},
            stochastic=True,
        )
    return evaluate_state(rollout_state, snake_id)


def backpropagate(node, reward):
    current = node
    while current is not None:
        current.visits += 1
        current.value += reward
        current = current.parent


def tree_policy(root):
    node = root
    while not node.is_terminal():
        if not node.fully_expanded():
            return node.expand()
        if not node.children:
            return node
        node = node.best_child()
    return node


def mcts(root_state, snake_id, time_budget=0.92):
    root_priors = opponent_aware_root_scores(root_state, snake_id)
    root = Node(root_state, snake_id, priors=root_priors)
    if not root.untried_moves:
        return fallback_move(root_state, snake_id), {"iterations": 0, "children": {}}

    deadline = time.time() + time_budget
    iterations = 0
    while time.time() < deadline:
        leaf = tree_policy(root)
        reward = rollout(leaf.state, snake_id)
        backpropagate(leaf, reward)
        iterations += 1

    if not root.children:
        move_name = fallback_move(root_state, snake_id)
        return move_name, {"iterations": iterations, "children": {}}

    best_child = max(
        root.children,
        key=lambda child: (
            child.value / child.visits if child.visits else float("-inf"),
            child.visits,
        ),
    )
    stats = {
        "iterations": iterations,
        "root_priors": {move: round(score, 2) for move, score in root_priors.items()},
        "children": {
            child.move: {
                "visits": child.visits,
                "avg": round(child.value / child.visits, 2) if child.visits else None,
            }
            for child in root.children
        },
    }
    return best_child.move, stats


def mcts_move(game_state, time_budget=None):
    state = make_state_from_game(game_state)
    if time_budget is None:
        time_budget = float(os.environ.get("MCTS_BUDGET", "0.92"))
    return mcts(state, state["you_id"], time_budget=time_budget)
