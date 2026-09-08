"""
A minimal, dependency-free "catch the ball" pixel environment.

Why this exists: training a real DQN on real Atari (gymnasium + ale-py +
ROM downloads) was intentionally NOT installed in this workflow. This toy
environment renders 210x160x3 RGB frames and exposes the same API shape as
a gymnasium environment, so:

  1. You can learn/verify the FULL DQN training loop (replay buffer,
     epsilon-greedy, Bellman target, SGD update, reward/Q tracking) end
     to end, right now, with zero extra installs.
  2. Swapping this out for a real Atari game later is a ~2 line change
     (see the bottom of dqn_train.py) because the interface matches:
       env.action_space.n
       obs, info = env.reset()
       obs, reward, terminated, truncated, info = env.step(action)
       frame = env.render()          # (210, 160, 3) uint8 RGB

Game rules: a paddle sits near the bottom of the screen. A ball starts at
the top at a random x-position and falls with a random horizontal drift.
The agent moves the paddle left / stays / right each step. Episode ends
when the ball reaches the paddle's row:
    +1 reward if the paddle overlaps the ball's x-position (catch)
    -1 reward otherwise (miss)
This is a deliberately simple, fast-to-learn control task -- enough to
demonstrate that the training loop actually reduces loss and improves
reward, without needing millions of frames or a GPU.
"""

import numpy as np


class _DiscreteActionSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(self.n)


class SimpleCatchEnv:
    WIDTH, HEIGHT = 160, 210
    SCOREBAR_H = 20
    PADDLE_W, PADDLE_H = 24, 6
    PADDLE_Y = HEIGHT - 20
    BALL_SIZE = 4
    PADDLE_STEP = 8
    BALL_VY = 6

    ACTIONS = {0: -1, 1: 0, 2: 1}  # left, stay, right (direction multiplier)

    def __init__(self, max_steps: int = 60, seed: int | None = None):
        self.action_space = _DiscreteActionSpace(n=3)
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self.paddle_x = self.WIDTH // 2 - self.PADDLE_W // 2
        self.ball_x = self.WIDTH // 2
        self.ball_y = self.SCOREBAR_H
        self.ball_vx = 0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self.paddle_x = self.WIDTH // 2 - self.PADDLE_W // 2
        self.ball_x = int(self._rng.integers(10, self.WIDTH - 10))
        self.ball_y = self.SCOREBAR_H
        self.ball_vx = int(self._rng.choice([-3, -2, 2, 3]))
        return self.render(), {}

    def step(self, action: int):
        self._t += 1

        direction = self.ACTIONS[int(action)]
        self.paddle_x += direction * self.PADDLE_STEP
        self.paddle_x = int(np.clip(self.paddle_x, 0, self.WIDTH - self.PADDLE_W))

        self.ball_y += self.BALL_VY
        self.ball_x += self.ball_vx
        if self.ball_x <= 0 or self.ball_x >= self.WIDTH:
            self.ball_vx *= -1
            self.ball_x = int(np.clip(self.ball_x, 0, self.WIDTH))

        terminated = False
        reward = 0.0

        reached_paddle_row = self.ball_y >= self.PADDLE_Y
        truncated = self._t >= self.max_steps

        if reached_paddle_row:
            caught = (self.paddle_x - self.BALL_SIZE <= self.ball_x
                      <= self.paddle_x + self.PADDLE_W + self.BALL_SIZE)
            reward = 1.0 if caught else -1.0
            terminated = True

        obs = self.render()
        return obs, reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        frame = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        frame[:self.SCOREBAR_H, :, :] = (200, 200, 200)          # scoreboard
        frame[self.SCOREBAR_H:, :, :] = (20, 60, 20)              # playing field

        # paddle (white rectangle)
        py0, py1 = self.PADDLE_Y, self.PADDLE_Y + self.PADDLE_H
        px0, px1 = self.paddle_x, self.paddle_x + self.PADDLE_W
        frame[py0:py1, px0:px1, :] = (255, 255, 255)

        # ball (red square)
        by0 = max(0, self.ball_y - self.BALL_SIZE)
        by1 = min(self.HEIGHT, self.ball_y + self.BALL_SIZE)
        bx0 = max(0, self.ball_x - self.BALL_SIZE)
        bx1 = min(self.WIDTH, self.ball_x + self.BALL_SIZE)
        frame[by0:by1, bx0:bx1, :] = (220, 30, 30)

        return frame

    def close(self):
        pass


def _self_test():
    env = SimpleCatchEnv(seed=0)
    obs, info = env.reset()
    print("Initial frame shape:", obs.shape, obs.dtype)

    total_reward = 0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()
    print("Ran 200 random steps successfully. Cumulative reward:", total_reward)


if __name__ == "__main__":
    _self_test()
