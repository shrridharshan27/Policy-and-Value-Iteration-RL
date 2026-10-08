"""
Main Execution Script for Policy Iteration & Value Iteration
Shrri Dharshan D R - 23BPS1090
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dp.policy_iteration import run_policy_iteration
from dp.value_iteration_grid import run_value_iteration, NUM_ROWS, NUM_COLS, GOAL_STATE, TRAP_STATE

def main():
    print("=" * 60)
    print("   REINFORCEMENT LEARNING LAB 4: POLICY & VALUE ITERATION")
    print("=" * 60)

    # 1. Policy Iteration
    pi_values, pi_policy = run_policy_iteration()
    print("\n=== 1. Policy Iteration Results ===")
    print("State Values :", np.round(pi_values, 2))
    print("Optimal Policy:", pi_policy)

    # 2. Value Iteration
    vi_grid, vi_policy = run_value_iteration()
    print("\n=== 2. Value Iteration Results ===")
    print("Converged Value Grid (3x3):")
    print(np.round(vi_grid, 1))
    print("\nOptimal Policy Map:")
    for r in range(NUM_ROWS):
        print([vi_policy[(r, c)] for c in range(NUM_COLS)])

    # Plot 1: Policy Iteration Converged Values
    plt.figure(figsize=(7, 4))
    bars = plt.bar([f"State {i}\n({pi_policy[i]})" for i in range(4)], pi_values,
                   color=['#2b5c8f', '#457b9d', '#e76f51', '#2a9d8f'], edgecolor='black', width=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.2f}", ha='center', fontweight='bold')
    plt.title('Policy Iteration: Converged State Values', fontsize=12, fontweight='bold')
    plt.ylabel('State Value V(s)', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot1_path = os.path.join("assets", "policy_iteration_values.png")
    plt.savefig(plot1_path, dpi=150)
    plt.close()

    # Plot 2: 2D GridWorld Value Heatmap with Policy Arrows
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(vi_grid, cmap='YlGnBu', origin='upper')
    plt.colorbar(im, ax=ax, label='State Value V(s)')

    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            val_text = f"{vi_grid[r, c]:.1f}"
            pol_text = vi_policy[(r, c)]
            if (r, c) == GOAL_STATE:
                ax.text(c, r, f"GOAL\n+10.0\n({val_text})", ha='center', va='center',
                        color='green', fontweight='bold', fontsize=11, bbox=dict(boxstyle="square", fc="#d4edda", ec="green"))
            elif (r, c) == TRAP_STATE:
                ax.text(c, r, f"TRAP\n-10.0\n({val_text})", ha='center', va='center',
                        color='red', fontweight='bold', fontsize=11, bbox=dict(boxstyle="square", fc="#f8d7da", ec="red"))
            else:
                ax.text(c, r, f"{pol_text}\n{val_text}", ha='center', va='center',
                        color='black', fontweight='bold', fontsize=14)

    ax.set_xticks(range(NUM_COLS))
    ax.set_yticks(range(NUM_ROWS))
    ax.set_title('3x3 GridWorld: Value Heatmap & Optimal Policy Map', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plot2_path = os.path.join("assets", "gridworld_value_policy.png")
    plt.savefig(plot2_path, dpi=150)
    plt.close()

    print(f"\nPlots saved to {plot1_path} and {plot2_path}")

if __name__ == "__main__":
    main()
