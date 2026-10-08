"""
Policy Iteration for 1D Stochastic GridWorld.
"""
from typing import Dict, Tuple
import numpy as np

def run_policy_iteration(threshold: float = 1e-6, discount: float = 0.9) -> Tuple[np.ndarray, Dict[int, str]]:
    all_states = [0, 1, 2, 3]
    all_actions = [0, 1]  # 0 = Left, 1 = Right

    transitions = {
        0: {0: [(1.0, 0)],           1: [(0.8, 1), (0.2, 0)]},
        1: {0: [(0.8, 0), (0.2, 1)], 1: [(0.8, 2), (0.2, 1)]},
        2: {0: [(0.8, 1), (0.2, 2)], 1: [(0.8, 3), (0.2, 2)]},
        3: {0: [(1.0, 3)],           1: [(1.0, 3)]}
    }
    rewards = np.array([-1.0, -1.0, -1.0, 10.0])
    state_values = np.zeros(len(all_states))
    current_policy = {s: 1 for s in all_states}  # Initial policy: all Right

    while True:
        # Phase 1: Policy Evaluation
        while True:
            max_change = 0.0
            for s in all_states:
                if s == 3:  # Skip absorbing terminal
                    continue
                prev_val = state_values[s]
                act = current_policy[s]
                state_values[s] = rewards[s] + discount * sum(
                    p * state_values[ns] for p, ns in transitions[s][act]
                )
                max_change = max(max_change, abs(prev_val - state_values[s]))
            if max_change < threshold:
                break

        # Phase 2: Policy Improvement
        is_stable = True
        for s in all_states:
            if s == 3:
                continue
            prev_act = current_policy[s]
            scores = [
                rewards[s] + discount * sum(p * state_values[ns] for p, ns in transitions[s][a])
                for a in all_actions
            ]
            best_act = int(np.argmax(scores))
            current_policy[s] = best_act
            if best_act != prev_act:
                is_stable = False

        if is_stable:
            break

    readable_policy = {s: ("Right" if current_policy[s] == 1 else "Left") for s in all_states[:-1]}
    readable_policy[3] = "Exit"
    return state_values, readable_policy
