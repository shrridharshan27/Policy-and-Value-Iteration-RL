"""
Value Iteration for 2D GridWorld with Goal & Trap States.
"""
from typing import Dict, Tuple
import numpy as np

NUM_ROWS, NUM_COLS = 3, 3
GOAL_STATE = (0, 2)
TRAP_STATE = (1, 2)
END_STATES = [GOAL_STATE, TRAP_STATE]

MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT = 0, 1, 2, 3
DIRECTION_SYMBOL = {MOVE_UP: "↑", MOVE_DOWN: "↓", MOVE_LEFT: "←", MOVE_RIGHT: "→"}
ALL_MOVES = [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT]

def move(row: int, col: int, direction: int) -> Tuple[int, int]:
    if direction == MOVE_UP:
        return (max(row - 1, 0), col)
    elif direction == MOVE_DOWN:
        return (min(row + 1, NUM_ROWS - 1), col)
    elif direction == MOVE_LEFT:
        return (row, max(col - 1, 0))
    else:
        return (row, min(col + 1, NUM_COLS - 1))

def run_value_iteration(gamma: float = 0.9, epsilon: float = 1e-5) -> Tuple[np.ndarray, Dict[Tuple[int, int], str]]:
    grid_V = np.zeros((NUM_ROWS, NUM_COLS))
    reward_grid = np.full((NUM_ROWS, NUM_COLS), -1.0)
    reward_grid[GOAL_STATE] = 10.0
    reward_grid[TRAP_STATE] = -10.0

    while True:
        biggest_change = 0.0
        snapshot = grid_V.copy()

        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                if (r, c) in END_STATES:
                    continue
                q_vals = [
                    reward_grid[r, c] + gamma * snapshot[move(r, c, d)[0], move(r, c, d)[1]]
                    for d in ALL_MOVES
                ]
                grid_V[r, c] = max(q_vals)
                biggest_change = max(biggest_change, abs(grid_V[r, c] - snapshot[r, c]))

        if biggest_change < epsilon:
            break

    grid_policy = {}
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            if (r, c) == GOAL_STATE:
                grid_policy[(r, c)] = "GOAL"
            elif (r, c) == TRAP_STATE:
                grid_policy[(r, c)] = "TRAP"
            else:
                q_vals = [
                    reward_grid[r, c] + gamma * grid_V[move(r, c, d)[0], move(r, c, d)[1]]
                    for d in ALL_MOVES
                ]
                grid_policy[(r, c)] = DIRECTION_SYMBOL[int(np.argmax(q_vals))]

    return grid_V, grid_policy
