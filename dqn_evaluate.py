"""
Load a trained DQN's weights and run it in the environment.
Reproduces the paper's Section 5.3 evaluation convention: run the trained
network with an epsilon-greedy policy at epsilon = 0.05 (not epsilon = 0)
for a fixed number of episodes/steps, and report the average score.

Usage:
    python3 dqn_evaluate.py                                   # defaults: dqn_breakout.pt, ALE/Breakout-v5, 5 episodes
    python3 dqn_evaluate.py --weights dqn_breakout.pt --episodes 10
    python3 dqn_evaluate.py --weights dqn_toy_catch.pt --toy    # evaluate a toy-env checkpoint
    python3 dqn_evaluate.py --record                            # also saves a GIF of episode 1
"""

import argparse

import numpy as np
import torch
from PIL import Image

from dqn_model import DQN
from atari_preprocessing import AtariPreprocessor


def make_env(game_id: str, use_toy: bool):
    if use_toy:
        from toy_pixel_env import SimpleCatchEnv
        return SimpleCatchEnv(seed=123)
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    return gym.make(game_id, render_mode="rgb_array")


def to_tensor(phi_hwc: np.ndarray) -> torch.Tensor:
    x = phi_hwc.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # (1, 4, 84, 84)
    return x


def load_trained_net(weights_path: str, num_actions: int) -> DQN:
    """
    This is the "reuse weights" step. Two things must match what was
    used during training, or load_state_dict will fail/mismatch:
      1. The exact same architecture (DQN class, same in_channels).
      2. The exact same num_actions the checkpoint was trained with
         (different games have different action counts -- loading
         Breakout's 4-action weights into a game with 6 actions will
         raise a shape-mismatch error on the final layer).
    """
    net = DQN(in_channels=4, num_actions=num_actions)
    state_dict = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()  # disables dropout/batchnorm training behavior (harmless
                # no-op for this architecture, but correct practice)
    return net


def select_action_eval(net: DQN, phi_hwc: np.ndarray, epsilon: float, num_actions: int) -> int:
    """
    Paper, Section 5.3: evaluation uses an epsilon-greedy policy with
    epsilon = 0.05, NOT a fully greedy (epsilon = 0) policy. A purely
    greedy policy on a deterministic emulator can get stuck replaying the
    exact same looping trajectory forever; a small amount of randomness
    avoids that and better reflects real play.
    """
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)
    with torch.no_grad():
        q_values = net(to_tensor(phi_hwc))
        return int(torch.argmax(q_values, dim=1).item())


def run_episode(env, net, preprocessor, epsilon, num_actions, record=False):
    obs, info = env.reset()
    preprocessor.reset()
    phi = preprocessor.step(obs)

    frames = [obs] if record else None
    total_reward = 0.0
    done = False
    steps = 0

    while not done:
        action = select_action_eval(net, phi, epsilon, num_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        phi = preprocessor.step(obs)
        total_reward += reward
        steps += 1
        if record:
            frames.append(obs)

    return total_reward, steps, frames


def save_gif(frames, path: str, fps: int = 30):
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"Saved gameplay recording to {path} ({len(images)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Run a trained DQN checkpoint.")
    parser.add_argument("--weights", default="dqn_breakout.pt", help="path to a .pt checkpoint")
    parser.add_argument("--game", default="ALE/Breakout-v5", help="gymnasium Atari game id")
    parser.add_argument("--toy", action="store_true", help="use toy_pixel_env instead of real Atari")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.05, help="paper's Section 5.3 evaluation epsilon")
    parser.add_argument("--record", action="store_true", help="save a GIF of the first episode")
    parser.add_argument("--record-path", default="dqn_gameplay.gif")
    args = parser.parse_args()

    env = make_env(args.game, args.toy)
    num_actions = env.action_space.n
    preprocessor = AtariPreprocessor()

    print(f"Loading weights from {args.weights} ({num_actions} actions)...")
    net = load_trained_net(args.weights, num_actions)

    rewards = []
    for ep in range(1, args.episodes + 1):
        record_this_one = args.record and ep == 1
        reward, steps, frames = run_episode(
            env, net, preprocessor, args.epsilon, num_actions, record=record_this_one
        )
        rewards.append(reward)
        print(f"Episode {ep}: reward = {reward:.1f}, steps = {steps}")
        if record_this_one:
            save_gif(frames, args.record_path)

    print(f"\nAverage reward over {args.episodes} episodes (epsilon={args.epsilon}): "
          f"{np.mean(rewards):.2f} (+/- {np.std(rewards):.2f})")

    env.close()


if __name__ == "__main__":
    main()
