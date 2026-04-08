"""
MCTS with 6 configurable improvements:
  1. Prior-guided expansion      (Expansion)
  2. PUCT selection              (Selection)
  3. Opponent-aware root scoring (Selection)
  4. Heuristic rollouts          (Simulation)
  5. RAVE / AMAF                 (Backprop + Selection)
  6. UCB1-Tuned                  (Selection)

All improvements are controlled by MCTSConfig.
"""

import math
import os
import random
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

from agent_core import (
    advance_state,
    evaluate_state,
    fallback_move,
    get_legal_moves,
    heuristic_move_scores,
    is_terminal_state,
    make_state_from_game,
    mcts_evaluate_state,
    rollout_move,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MCTSConfig:
    """Controls which MCTS improvements are active."""

    # Improvement 1 & 2: Prior-guided expansion + PUCT selection
    use_priors: bool = True
    prior_bonus_scale: float = 4.0    # multiplier for prior bonus (was 18.0 — too aggressive at low iteration counts)
    puct_decay: bool = True            # bonus decays with visits

    # Improvement 3: Opponent-aware root scoring (minimax blend)
    use_opponent_model: bool = True
    minimax_top_n: int = 2            # opponent moves considered

    # Improvement 4: Heuristic rollouts
    use_heuristic_rollout: bool = True
    rollout_depth: int = 8

    # Improvement 5: RAVE / AMAF
    use_rave: bool = False
    rave_k: float = 500.0             # controls RAVE-UCT blend speed

    # Improvement 6: UCB1-Tuned
    use_ucb1_tuned: bool = False

    # Generic exploration constant
    exploration: float = 1.05

    def copy_with(self, **kwargs) -> "MCTSConfig":
        import copy
        obj = copy.copy(self)
        for k, v in kwargs.items():
            setattr(obj, k, v)
        return obj


# Preset configurations ---------------------------------------------------

VANILLA_CONFIG = MCTSConfig(
    use_priors=False,
    use_opponent_model=False,
    use_heuristic_rollout=False,
    use_rave=False,
    use_ucb1_tuned=False,
)

HEURISTIC_ROLLOUT_CONFIG = MCTSConfig(
    use_priors=False,
    use_opponent_model=False,
    use_heuristic_rollout=True,
    use_rave=False,
    use_ucb1_tuned=False,
)

PRIOR_GUIDED_CONFIG = MCTSConfig(
    use_priors=True,
    use_opponent_model=False,
    use_heuristic_rollout=True,
    use_rave=False,
    use_ucb1_tuned=False,
)

OPPONENT_AWARE_CONFIG = MCTSConfig(
    use_priors=True,
    use_opponent_model=True,
    use_heuristic_rollout=True,
    use_rave=False,
    use_ucb1_tuned=False,
)

RAVE_CONFIG = MCTSConfig(
    use_priors=True,
    use_opponent_model=True,
    use_heuristic_rollout=True,
    use_rave=True,
    use_ucb1_tuned=False,
)

UCB1_TUNED_CONFIG = MCTSConfig(
    use_priors=True,
    use_opponent_model=True,
    use_heuristic_rollout=True,
    use_rave=False,
    use_ucb1_tuned=True,
)

ALL_IMPROVEMENTS_CONFIG = MCTSConfig(
    use_priors=True,
    use_opponent_model=True,
    use_heuristic_rollout=True,
    use_rave=False,            # RAVE + UCB1-Tuned conflict (both modify selection);
    use_ucb1_tuned=True,       # UCB1-Tuned performed better individually (73.3% vs 60%)
)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    __slots__ = (
        "state", "snake_id", "parent", "move", "children",
        "priors", "untried_moves",
        "visits", "value", "value_sq",
        "rave_visits", "rave_value",
    )

    def __init__(
        self,
        state: dict,
        snake_id: str,
        parent: Optional["Node"] = None,
        move: Optional[str] = None,
        priors: Optional[Dict[str, float]] = None,
        config: Optional[MCTSConfig] = None,
    ):
        self.state = state
        self.snake_id = snake_id
        self.parent = parent
        self.move = move
        self.children: List["Node"] = []

        if priors is not None:
            # Root node: caller provides full heuristic / opponent-aware priors
            self.priors = priors
        elif config is not None and config.use_priors and parent is None:
            # Root without explicit priors — compute them (rare path)
            self.priors = heuristic_move_scores(state, snake_id)
        else:
            # Child nodes or non-prior configs: use cheap legal-move lookup.
            # heuristic_move_scores is too expensive to call on every node in
            # the tree — it dominates the MCTS budget and starves the search.
            legal = get_legal_moves(state, snake_id)
            self.priors = {m: 0.0 for m in legal}

        # Order untried moves by prior score (best first)
        self.untried_moves: List[str] = [
            move_name
            for move_name, _score in sorted(
                self.priors.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )
        ]

        self.visits: int = 0
        self.value: float = 0.0
        self.value_sq: float = 0.0   # for UCB1-Tuned variance

        # RAVE / AMAF tables keyed by move string
        self.rave_visits: Dict[str, int] = {}
        self.rave_value: Dict[str, float] = {}

    def is_terminal(self) -> bool:
        return is_terminal_state(self.state, self.snake_id)

    def fully_expanded(self) -> bool:
        return not self.untried_moves

    def expand(self, config: MCTSConfig) -> "Node":
        move_name = self.untried_moves.pop(0)
        next_state = advance_state(
            self.state,
            planned_moves={self.snake_id: move_name},
            stochastic=True,
        )
        child = Node(
            next_state,
            self.snake_id,
            parent=self,
            move=move_name,
            config=config,
        )
        self.children.append(child)
        return child

    def best_child(self, config: MCTSConfig) -> "Node":
        best_score = float("-inf")
        best_nodes: List["Node"] = []
        log_parent = math.log(max(1, self.visits))

        prior_values = list(self.priors.values())
        min_prior = min(prior_values) if prior_values else 0.0
        max_prior = max(prior_values) if prior_values else 1.0

        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                n = child.visits
                exploit = child.value / n

                # UCB1-Tuned or standard exploration
                if config.use_ucb1_tuned:
                    variance = child.value_sq / n - (exploit ** 2)
                    variance = max(0.0, variance)
                    tuned = variance + math.sqrt(2.0 * log_parent / n)
                    explore = config.exploration * math.sqrt(
                        log_parent / n * min(0.25, tuned)
                    )
                else:
                    explore = config.exploration * math.sqrt(log_parent / n)

                # PUCT prior bonus (decays with visits)
                prior_bonus = 0.0
                if config.use_priors:
                    raw_prior = self.priors.get(child.move, 0.0)
                    if max_prior > min_prior:
                        norm_prior = (raw_prior - min_prior) / (max_prior - min_prior)
                    else:
                        norm_prior = 0.5
                    if config.puct_decay:
                        prior_bonus = norm_prior * config.prior_bonus_scale / (1 + n)
                    else:
                        prior_bonus = norm_prior * config.prior_bonus_scale

                # RAVE / AMAF blend
                if config.use_rave:
                    rv = self.rave_visits.get(child.move, 0)
                    rval = self.rave_value.get(child.move, 0.0)
                    if rv > 0:
                        beta = math.sqrt(
                            config.rave_k / (3.0 * n + config.rave_k)
                        )
                        rave_exploit = rval / rv
                        exploit = (1.0 - beta) * exploit + beta * rave_exploit

                score = exploit + explore + prior_bonus

            if score > best_score:
                best_score = score
                best_nodes = [child]
            elif score == best_score:
                best_nodes.append(child)

        return random.choice(best_nodes)


# ---------------------------------------------------------------------------
# Opponent modeling (kept from original)
# ---------------------------------------------------------------------------

def likely_moves_for_snake(
    state: dict, snake_id: str, top_n: int = 2
) -> List[Tuple[str, float]]:
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


def opponent_joint_responses(
    state: dict, snake_id: str, max_joint: int = 6
) -> List[Tuple[Dict[str, str], float]]:
    opponents = [s for s in state["snakes"] if s["id"] != snake_id]
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


def opponent_aware_root_scores(state: dict, snake_id: str) -> Dict[str, float]:
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
            scenario_values.append(mcts_evaluate_state(next_state, snake_id))

        if not scenario_values:
            next_state = advance_state(
                state, planned_moves={snake_id: move_name}, stochastic=False
            )
            scenario_values.append(mcts_evaluate_state(next_state, snake_id))

        scenario_values.sort()
        worst = scenario_values[0]
        average = sum(scenario_values) / len(scenario_values)
        median = scenario_values[len(scenario_values) // 2]
        root_scores[move_name] = 0.55 * worst + 0.30 * average + 0.15 * median

    return root_scores


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout(
    state: dict, snake_id: str, config: MCTSConfig
) -> Tuple[float, List[str]]:
    """
    Simulate a game from `state`.
    Returns (reward, moves_made_list) where moves_made_list tracks moves
    taken by snake_id during the rollout (used for RAVE / AMAF updates).
    """
    rollout_state = state
    moves_made: List[str] = []

    for _ in range(config.rollout_depth):
        if is_terminal_state(rollout_state, snake_id):
            break

        if config.use_heuristic_rollout:
            move_name = rollout_move(rollout_state, snake_id)
        else:
            legal = get_legal_moves(rollout_state, snake_id)
            if not legal:
                move_name = fallback_move(rollout_state, snake_id)
            else:
                move_name = random.choice(legal)

        moves_made.append(move_name)
        rollout_state = advance_state(
            rollout_state,
            planned_moves={snake_id: move_name},
            stochastic=True,
        )

    reward = mcts_evaluate_state(rollout_state, snake_id)
    return reward, moves_made


# ---------------------------------------------------------------------------
# Backpropagation
# ---------------------------------------------------------------------------

def backpropagate(
    node: Node, reward: float, moves_in_rollout: List[str], use_rave: bool
) -> None:
    """
    Walk up the tree, updating visit counts, values, and RAVE tables.
    AMAF: for every ancestor, update rave_visits/rave_value for ALL moves
    that appeared in the rollout.
    """
    current: Optional[Node] = node
    while current is not None:
        current.visits += 1
        current.value += reward
        current.value_sq += reward * reward

        if use_rave and moves_in_rollout:
            for m in moves_in_rollout:
                current.rave_visits[m] = current.rave_visits.get(m, 0) + 1
                current.rave_value[m] = current.rave_value.get(m, 0.0) + reward

        current = current.parent


# ---------------------------------------------------------------------------
# Tree policy
# ---------------------------------------------------------------------------

def tree_policy(root: Node, config: MCTSConfig) -> Node:
    node = root
    while not node.is_terminal():
        if not node.fully_expanded():
            return node.expand(config)
        if not node.children:
            return node
        node = node.best_child(config)
    return node


# ---------------------------------------------------------------------------
# Main MCTS routine
# ---------------------------------------------------------------------------

def mcts(
    root_state: dict,
    snake_id: str,
    config: MCTSConfig,
    time_budget: float = 0.92,
    max_iterations: Optional[int] = None,
) -> Tuple[str, dict]:
    """
    Run MCTS from root_state for snake_id.

    Stopping criterion (checked in this order):
      1. If *max_iterations* is given, run exactly that many iterations
         (time_budget is ignored).
      2. Otherwise fall back to wall-clock *time_budget* seconds.

    Returns (move_str, stats_dict).
    """
    # Build root priors
    if config.use_opponent_model:
        root_priors = opponent_aware_root_scores(root_state, snake_id)
    elif config.use_priors:
        root_priors = heuristic_move_scores(root_state, snake_id)
    else:
        legal = get_legal_moves(root_state, snake_id)
        root_priors = {m: 0.0 for m in legal}

    root = Node(root_state, snake_id, priors=root_priors, config=config)

    if not root.untried_moves and not root.children:
        fb = fallback_move(root_state, snake_id)
        return fb, {"iterations": 0, "children": {}}

    iterations = 0

    if max_iterations is not None:
        # ---- iteration-based budget ----
        while iterations < max_iterations:
            leaf = tree_policy(root, config)
            reward, moves_made = rollout(leaf.state, snake_id, config)
            backpropagate(leaf, reward, moves_made, config.use_rave)
            iterations += 1
    else:
        # ---- time-based budget (original behaviour) ----
        deadline = time.time() + time_budget
        while time.time() < deadline:
            leaf = tree_policy(root, config)
            reward, moves_made = rollout(leaf.state, snake_id, config)
            backpropagate(leaf, reward, moves_made, config.use_rave)
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


def mcts_move(
    game_state: dict,
    config: Optional[MCTSConfig] = None,
    time_budget: Optional[float] = None,
    max_iterations: Optional[int] = None,
) -> Tuple[str, dict]:
    """
    Convenience wrapper: parse game_state and run MCTS.

    If *max_iterations* is provided it takes precedence over *time_budget*.
    Reads MCTS_BUDGET env var if neither is given.
    Default config: ALL_IMPROVEMENTS_CONFIG.
    """
    state = make_state_from_game(game_state)
    if config is None:
        config = ALL_IMPROVEMENTS_CONFIG
    if max_iterations is None and time_budget is None:
        time_budget = float(os.environ.get("MCTS_BUDGET", "0.92"))
    return mcts(
        state,
        state["you_id"],
        config=config,
        time_budget=time_budget or 999.0,
        max_iterations=max_iterations,
    )
