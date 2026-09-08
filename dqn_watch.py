"""
Watch a trained DQN play, live, in a human-visible window, with a
performance overlay (step, action, reward, running average) updated in
real time.

`dqn_evaluate.py` already reports scores numerically and can save a GIF
after the fact; this script instead pops up an OpenCV window and renders
each frame as it happens, so you can watch the trained model play like a
human would watch it, while its performance stats update on top of the
video. Same checkpoint-loading and epsilon=0.05 evaluation convention as
`dqn_evaluate.py` (paper Section 5.3) -- this file only adds the live
display, it does not re-implement the evaluation logic.

Usage:
    python3 dqn_watch.py                                  # defaults: dqn_breakout.pt, ALE/Breakout-v5
    python3 dqn_watch.py --weights dqn_breakout.pt --episodes 3
    python3 dqn_watch.py --weights dqn_toy_catch.pt --toy  # watch the toy-env checkpoint instead
    python3 dqn_watch.py --scale 4 --fps 20                # bigger window, slower playback

Press 'q' or Esc in the video window to stop early.
"""

import argparse

import cv2
import numpy as np

from atari_preprocessing import AtariPreprocessor
from dqn_evaluate import make_env, load_trained_net, select_action_eval

ACTION_NAMES_BY_SIZE = {
    3: {0: "LEFT", 1: "STAY", 2: "RIGHT"},                       # toy_pixel_env
    4: {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"},             # ALE/Breakout-v5
}


def draw_overlay(frame_rgb: np.ndarray, scale: int, lines: list[str]) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    big = cv2.resize(
        frame_bgr,
        (frame_bgr.shape[1] * scale, frame_bgr.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    for i, line in enumerate(lines):
        y = 18 + i * 18
        # black outline then white fill, so text stays readable over any background
        cv2.putText(big, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(big, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return big


def watch_episode(env, net, preprocessor, epsilon, num_actions, ep_idx, total_eps,
                   window_name, scale, delay_ms, action_names, running_rewards):
    obs, info = env.reset()
    preprocessor.reset()
    phi = preprocessor.step(obs)

    episode_reward = 0.0
    step = 0
    done = False

    while not done:
        action = select_action_eval(net, phi, epsilon, num_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        phi = preprocessor.step(obs)
        episode_reward += reward
        step += 1

        running_avg = np.mean(running_rewards) if running_rewards else 0.0
        lines = [
            f"Episode {ep_idx}/{total_eps}  step {step}",
            f"action: {action_names.get(action, action)}",
            f"reward this step: {reward:+.1f}   episode total: {episode_reward:+.1f}",
            f"avg over {len(running_rewards)} finished ep(s): {running_avg:+.2f}",
        ]
        frame = draw_overlay(obs, scale, lines)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord("q"), 27):  # 'q' or Esc
            return episode_reward, step, True

    return episode_reward, step, False


def main():
    parser = argparse.ArgumentParser(description="Watch a trained DQN play live, with a stats overlay.")
    parser.add_argument("--weights", default="dqn_breakout.pt", help="path to a .pt checkpoint")
    parser.add_argument("--game", default="ALE/Breakout-v5", help="gymnasium Atari game id")
    parser.add_argument("--toy", action="store_true", help="use toy_pixel_env instead of real Atari")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.05, help="paper's Section 5.3 evaluation epsilon")
    parser.add_argument("--scale", type=int, default=3, help="window upscale factor (raw frame is 210x160)")
    parser.add_argument("--fps", type=int, default=30, help="playback speed")
    args = parser.parse_args()

    env = make_env(args.game, args.toy)
    num_actions = env.action_space.n
    preprocessor = AtariPreprocessor()
    action_names = ACTION_NAMES_BY_SIZE.get(num_actions, {})

    print(f"Loading weights from {args.weights} ({num_actions} actions)...")
    net = load_trained_net(args.weights, num_actions)

    window_name = f"DQN playing {'toy-catch' if args.toy else args.game} (q or Esc to quit)"
    delay_ms = max(1, int(1000 / args.fps))

    running_rewards = []
    for ep in range(1, args.episodes + 1):
        reward, steps, quit_requested = watch_episode(
            env, net, preprocessor, args.epsilon, num_actions, ep, args.episodes,
            window_name, args.scale, delay_ms, action_names, running_rewards,
        )
        running_rewards.append(reward)
        print(f"Episode {ep}: reward = {reward:.1f}, steps = {steps}")
        if quit_requested:
            print("Stopped early by user.")
            break

    print(f"\nAverage reward over {len(running_rewards)} episode(s) (epsilon={args.epsilon}): "
          f"{np.mean(running_rewards):.2f} (+/- {np.std(running_rewards):.2f})")

    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
