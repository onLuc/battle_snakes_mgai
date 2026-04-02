"""
Legacy compatibility wrapper.

The previous MCTS implementation in this file had a separate, inaccurate rules
model. It now re-exports the fixed implementation from MCTS.py so older imports
continue to work without drifting away from the actual game logic.
"""

from MCTS import mcts, mcts_move

__all__ = ["mcts", "mcts_move"]
