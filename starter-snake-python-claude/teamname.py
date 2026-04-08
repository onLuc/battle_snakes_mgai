"""
Contest agent — MCTS with all improvements (prior-guided expansion, PUCT selection,
opponent-aware root scoring, heuristic rollouts, RAVE/AMAF, UCB1-Tuned).

Run: python teamname.py
"""
from agent_server import run_agent

if __name__ == "__main__":
    run_agent("mcts")
