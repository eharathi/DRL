"""
DQN training loop -- reproduces Algorithm 1 ("Deep Q-learning with
Experience Replay") from Mnih et al., "Playing Atari with Deep
Reinforcement Learning" (DeepMind, 2013), Section 4/5.

Runs against the REAL Atari environment (gymnasium + ale-py + ROMs, now
installed) by default. Set USE_TOY_ENV = True below to fall back to the
dependency-free `toy_pixel_env.SimpleCatchEnv` for fast iteration/testing
on a machine without the Atari stack installed.

Usage:
    python3 dqn_train.py                     # train from scratch
    python3 dqn_train.py --resume             # continue from the existing dqn_<tag>.pt
                                               # checkpoint + history, running more episodes
                                               # on top of what's already been learned
    python3 dqn_train.py --resume --episodes 40   # resume, but only run 40 more episodes
"""

import argparse
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn_model import DQN
from atari_preprocessing import AtariPreprocessor


# ---------------------------------------------------------------------------
# Environment selection
# ---------------------------------------------------------------------------
USE_TOY_ENV = False          # True = SimpleCatchEnv (no installs needed), False = real Atari
GAME_ID = "ALE/Breakout-v5"  # one of the paper's 7 games; also try Pong, Seaquest, etc.


def make_env():
    if USE_TOY_ENV:
        from toy_pixel_env import SimpleCatchEnv
        return SimpleCatchEnv(seed=42)
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    return gym.make(GAME_ID, render_mode="rgb_array")


# ---------------------------------------------------------------------------
# Hyperparameters
# (Where the paper doesn't state an exact value, a commonly used default
#  from the wider DQN literature is used instead -- flagged below. Values
#  marked "reduced" are scaled down from the paper's real settings so a
#  demo run finishes in minutes rather than the paper's ~1 week on 2013
#  hardware; raise them for a closer reproduction if you have the time
#  and compute to spare.)
# ---------------------------------------------------------------------------
NUM_EPISODES = 20 if not USE_TOY_ENV else 400
                               # paper: trained on 10M frames (tens of thousands
                               # of episodes). Real Atari episodes are much
                               # longer than the toy game's, so far fewer
                               # episodes are used here to keep runtime
                               # reasonable (~10-15 min on a laptop CPU).
REPLAY_CAPACITY = 20_000 if not USE_TOY_ENV else 5_000
                               # paper: 1,000,000 -- reduced for feasibility
MIN_REPLAY_BEFORE_TRAIN = 1_000 if not USE_TOY_ENV else 500
                               # start learning once the buffer has some data
BATCH_SIZE = 32                # matches the paper
GAMMA = 0.99                   # discount factor (paper doesn't give exact value; 0.99 is standard)
EPS_START = 1.0                 # matches the paper
EPS_END = 0.1                   # matches the paper
EPS_DECAY_STEPS = 20_000 if not USE_TOY_ENV else 3_000
                               # paper: anneals over first 1,000,000 frames -- reduced here
LEARNING_RATE = 2.5e-4            # not stated exactly in the 2013 paper; common RMSProp default
TARGET_EVAL_STATES = 64             # size of the held-out set used to track avg max-Q (Section 5.1)
LOG_EVERY = 5 if not USE_TOY_ENV else 25   # print progress every N episodes
CLIP_REWARDS = True                          # paper (Section 5): clip all rewards to {-1, 0, +1}


class ReplayMemory:
    """
    Stores transitions (phi_t, a_t, r_t, phi_t+1, done) and returns random
    minibatches -- this is the core "experience replay" mechanism
    (Section 4) that breaks correlation between consecutive samples and
    reuses each experience in many weight updates.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, phi, action, reward, next_phi, done):
        self.buffer.append((phi, action, reward, next_phi, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        phis, actions, rewards, next_phis, dones = zip(*batch)
        return (
            np.stack(phis),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_phis),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def to_tensor_batch(phi_batch_hwcN: np.ndarray) -> torch.Tensor:
    """
    Converts a batch of preprocessed states from the pipeline's
    channels-last, uint8 layout (N, 84, 84, 4) into the channels-first,
    normalized float layout PyTorch conv layers expect (N, 4, 84, 84).
    See DQN_Architecture_Guide.md Step 9 for why this conversion exists.
    """
    x = phi_batch_hwcN.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(0, 3, 1, 2)
    return x


def clip_reward(reward: float) -> float:
    """
    Paper, Section 5: 'we fixed all positive rewards to be 1 and all
    negative rewards to be -1, leaving 0 rewards unchanged.' This keeps
    error-derivative scale consistent across games with very different
    score magnitudes (e.g. Breakout's small per-brick points vs.
    Seaquest's large per-kill points), at the cost of losing information
    about reward magnitude.
    """
    if reward > 0:
        return 1.0
    if reward < 0:
        return -1.0
    return 0.0


def epsilon_by_step(step: int) -> float:
    """Linear annealing from EPS_START to EPS_END over EPS_DECAY_STEPS,
    then held fixed -- exactly the schedule described in Section 5."""
    frac = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_START + frac * (EPS_END - EPS_START)


def select_action(net: DQN, phi_hwc: np.ndarray, epsilon: float, num_actions: int) -> int:
    """epsilon-greedy behaviour policy (Section 2 / Algorithm 1)."""
    if random.random() < epsilon:
        return random.randrange(num_actions)
    with torch.no_grad():
        x = to_tensor_batch(phi_hwc[None, ...])   # add batch dim
        q_values = net(x)
        return int(torch.argmax(q_values, dim=1).item())


def collect_eval_states(env, preprocessor, n_states: int):
    """
    Runs a purely random policy for a bit and stores a fixed set of states.
    Used only to track "average predicted max Q" over a FIXED held-out set,
    exactly as described in Section 5.1 -- this is the paper's more stable
    alternative to noisy per-episode reward for monitoring training
    progress.
    """
    states = []
    obs, info = env.reset()
    preprocessor.reset()
    phi = preprocessor.step(obs)
    states.append(phi)
    while len(states) < n_states:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        phi = preprocessor.step(obs)
        states.append(phi)
        if terminated or truncated:
            obs, info = env.reset()
            preprocessor.reset()
            phi = preprocessor.step(obs)
    return np.stack(states[:n_states])


def compute_avg_max_q(net: DQN, eval_states: np.ndarray) -> float:
    with torch.no_grad():
        x = to_tensor_batch(eval_states)
        q_values = net(x)
        max_q = q_values.max(dim=1).values
        return float(max_q.mean().item())


def train(num_episodes: int = NUM_EPISODES, resume_weights_path: str | None = None):
    """
    resume_weights_path: if given and the file exists, load it as the
    network's STARTING weights (the "reuse weights" step, same
    load_state_dict convention as dqn_evaluate.py -- see
    DQN_SaveLoad_Guide.md) instead of the usual random Kaiming init. This
    lets you run additional training iterations on top of an already
    trained checkpoint and observe further improvement, rather than
    re-learning from scratch every time.
    """
    env = make_env()
    num_actions = env.action_space.n
    print(f"Environment: {'toy SimpleCatchEnv' if USE_TOY_ENV else GAME_ID} "
          f"| actions = {num_actions}")

    preprocessor = AtariPreprocessor()
    net = DQN(in_channels=4, num_actions=num_actions)

    # global_step drives epsilon_by_step()'s linear anneal from EPS_START.
    # A freshly initialized network should explore a lot (epsilon starts
    # at EPS_START = 1.0). A network resumed from a trained checkpoint
    # already knows how to act, so exploration is started at the paper's
    # floor (EPS_END = 0.1) instead of re-exploring randomly from scratch.
    global_step = 0
    if resume_weights_path is not None:
        if os.path.exists(resume_weights_path):
            print(f"Resuming from trained weights: {resume_weights_path}")
            net.load_state_dict(torch.load(resume_weights_path, map_location="cpu"))
            global_step = EPS_DECAY_STEPS
        else:
            print(f"--resume given but {resume_weights_path} does not exist yet; "
                  f"starting from a freshly initialized network instead.")

    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    replay = ReplayMemory(REPLAY_CAPACITY)

    print("Collecting a fixed held-out evaluation set (random policy)...")
    eval_states = collect_eval_states(env, preprocessor, TARGET_EVAL_STATES)

    episode_rewards = []
    avg_q_history = []
    loss_history = []

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        preprocessor.reset()
        phi = preprocessor.step(obs)

        episode_reward = 0.0
        done = False

        while not done:
            epsilon = epsilon_by_step(global_step)
            action = select_action(net, phi, epsilon, num_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_phi = preprocessor.step(obs)

            train_reward = clip_reward(reward) if CLIP_REWARDS else reward

            # Algorithm 1: store transition (phi_t, a_t, r_t, phi_t+1)
            replay.push(phi, action, train_reward, next_phi, float(terminated))

            phi = next_phi
            episode_reward += reward  # log the TRUE (unclipped) score, per Section 5.3's evaluation convention
            global_step += 1

            # Algorithm 1: sample random minibatch and do one SGD step
            if len(replay) >= max(MIN_REPLAY_BEFORE_TRAIN, BATCH_SIZE):
                phis, actions, rewards, next_phis, dones = replay.sample(BATCH_SIZE)

                states_t = to_tensor_batch(phis)
                next_states_t = to_tensor_batch(next_phis)
                actions_t = torch.from_numpy(actions).long()
                rewards_t = torch.from_numpy(rewards).float()
                dones_t = torch.from_numpy(dones).float()

                # Bellman target y_j (Eq. in Algorithm 1):
                #   y_j = r_j                              if terminal
                #   y_j = r_j + gamma * max_a' Q(s', a'; theta)   otherwise
                # NOTE: the 2013 paper's Algorithm 1 does NOT use a
                # separate frozen target network (that stabilization
                # trick was added in the 2015 Nature follow-up paper).
                # We reproduce the ORIGINAL algorithm here: the max_a'
                # term uses the SAME network, but wrapped in no_grad()
                # so gradients don't flow through the bootstrap target,
                # matching "theta_i-1 held fixed" during the loss step.
                with torch.no_grad():
                    next_q = net(next_states_t)
                    max_next_q = next_q.max(dim=1).values
                    targets = rewards_t + GAMMA * max_next_q * (1.0 - dones_t)

                current_q_all = net(states_t)
                current_q = current_q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

                loss = loss_fn(current_q, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())

        episode_rewards.append(episode_reward)
        avg_q = compute_avg_max_q(net, eval_states)
        avg_q_history.append(avg_q)

        if episode % LOG_EVERY == 0:
            recent_reward = np.mean(episode_rewards[-LOG_EVERY:])
            recent_loss = np.mean(loss_history[-200:]) if loss_history else float("nan")
            print(f"Episode {episode:4d} | "
                  f"avg reward (last {LOG_EVERY}) = {recent_reward:+.2f} | "
                  f"avg max Q = {avg_q:6.3f} | "
                  f"eps = {epsilon_by_step(global_step):.3f} | "
                  f"loss = {recent_loss:.4f}")

    env.close()
    return net, episode_rewards, avg_q_history, loss_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train (or continue training) the DQN.")
    parser.add_argument("--resume", action="store_true",
                         help="load the existing dqn_<tag>.pt checkpoint and "
                              "dqn_training_history_<tag>.npz as the starting point, "
                              "instead of training from scratch")
    parser.add_argument("--episodes", type=int, default=None,
                         help="episodes to run THIS invocation (default: NUM_EPISODES above; "
                              "with --resume, these are additional episodes on top of what "
                              "was already trained)")
    args = parser.parse_args()

    tag = "toy_catch" if USE_TOY_ENV else GAME_ID.split("/")[-1].replace("-v5", "").lower()
    weights_path = f"dqn_{tag}.pt"
    history_path = f"dqn_training_history_{tag}.npz"

    num_episodes = args.episodes if args.episodes is not None else NUM_EPISODES
    resume_weights_path = weights_path if args.resume else None

    net, episode_rewards, avg_q_history, loss_history = train(num_episodes, resume_weights_path)
    episode_rewards = np.array(episode_rewards)
    avg_q_history = np.array(avg_q_history)
    loss_history = np.array(loss_history)

    if args.resume and os.path.exists(history_path):
        prev = np.load(history_path)
        n_prev = len(prev["episode_rewards"])
        episode_rewards = np.concatenate([prev["episode_rewards"], episode_rewards])
        avg_q_history = np.concatenate([prev["avg_q_history"], avg_q_history])
        loss_history = np.concatenate([prev["loss_history"], loss_history])
        print(f"\nAppended {num_episodes} new episodes onto {n_prev} previously trained "
              f"episodes (cumulative total: {len(episode_rewards)}).")

    torch.save(net.state_dict(), weights_path)
    np.savez(history_path,
             episode_rewards=episode_rewards,
             avg_q_history=avg_q_history,
             loss_history=loss_history)

    print(f"\nSaved trained weights to {weights_path}")
    print(f"Saved training history to {history_path}")
    print(f"Run: python3 measure_dqn.py {history_path}")
