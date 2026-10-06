"""
Measuring / diagnosing DQN training progress.
Reproduces the diagnostics from Section 5.1 ("Training and Stability") and
Figure 2 of Mnih et al., 2013.

The paper's key observation: raw per-episode reward is a very noisy metric
during RL training (small policy changes -> large changes in which states
get visited), so they track a second, much smoother metric alongside it:
the network's own average predicted max-Q on a FIXED held-out set of
states, collected once before training starts.

Run after dqn_train.py has produced a dqn_training_history_*.npz file:
    python3 measure_dqn.py [path_to_history.npz]
(defaults to dqn_training_history_breakout.npz if no path is given)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_average(x, window):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    history_path = sys.argv[1] if len(sys.argv) > 1 else "dqn_training_history_breakout.npz"
    tag = os.path.basename(history_path).replace("dqn_training_history_", "").replace(".npz", "")
    data = np.load(history_path)
    episode_rewards = data["episode_rewards"]
    avg_q_history = data["avg_q_history"]
    loss_history = data["loss_history"]

    episodes = np.arange(1, len(episode_rewards) + 1)
    window = max(10, len(episode_rewards) // 40)
    smoothed_reward = moving_average(episode_rewards, window)
    smoothed_episodes = episodes[window - 1:] if len(episode_rewards) >= window else episodes

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    axes[0].plot(episodes, episode_rewards, color="#B5C4D4", linewidth=0.8, label="raw")
    axes[0].plot(smoothed_episodes, smoothed_reward, color="#2E86E8", linewidth=2.2,
                 label=f"{window}-episode moving avg")
    axes[0].set_title("Reward per Episode\n(noisy metric, per Section 5.1)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode reward")
    axes[0].axhline(0, color="grey", linewidth=0.6, linestyle="--")
    axes[0].legend()

    axes[1].plot(episodes, avg_q_history, color="#F29B2E", linewidth=1.2)
    axes[1].set_title("Average Max Predicted Q\n(held-out state set, per Section 5.1)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Avg max Q")

    if len(loss_history) > 0:
        steps = np.arange(1, len(loss_history) + 1)
        log_loss = np.log10(np.clip(loss_history, 1e-8, None))
        axes[2].plot(steps, log_loss, color="#8E44AD", linewidth=0.6)
        axes[2].set_title("Training Loss (log10 scale)\nper SGD step")
        axes[2].set_xlabel("Training step")
        axes[2].set_ylabel("log10(MSE loss)")

    plt.tight_layout()
    out_path = f"dqn_training_metrics_{tag}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    # Console summary
    print("\n--- Summary ---")
    print(f"Episodes trained         : {len(episode_rewards)}")
    print(f"Mean reward (first 10%)  : {episode_rewards[:len(episode_rewards)//10].mean():+.3f}")
    print(f"Mean reward (last 10%)   : {episode_rewards[-len(episode_rewards)//10:].mean():+.3f}")
    print(f"Final avg max Q          : {avg_q_history[-1]:+.3f}")
    print(f"Max |avg max Q| observed : {np.max(np.abs(avg_q_history)):.3f}  "
          f"(large spikes here indicate Q-value instability -- see guide)")


if __name__ == "__main__":
    main()
