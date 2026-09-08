"""
Atari Frame Preprocessing Pipeline
Reproduces Section 4.1 ("Preprocessing and Model Architecture") of:
Mnih et al., "Playing Atari with Deep Reinforcement Learning", DeepMind, 2013.

Pipeline (per the paper):
    1. Capture a raw Atari frame: 210 x 160 pixels, RGB, 128-color palette.
    2. Convert RGB -> grayscale.
    3. Down-sample the grayscale frame to 110 x 84.
    4. Crop an 84 x 84 region that captures the playing area (square input
       required by the GPU 2D-convolution implementation used in the paper).
    5. Stack the last 4 preprocessed frames -> final input tensor of
       shape (84, 84, 4), giving the network a short motion history.

This file is split into two independent parts so it can run with OR without
the Atari environment installed:

  PART A - AtariPreprocessor: the actual pixel-processing pipeline (steps 2-5).
           Only needs numpy + opencv. Fully self-contained and testable.

  PART B - Environment setup + raw pixel capture (step 1), using the
           standard modern stack: gymnasium + ale-py.
           This part is OPTIONAL and only runs if those packages are
           installed (see requirements.txt / README).

Run this file directly to see a self-test using a synthetic frame:
    python3 atari_preprocessing.py
"""

from collections import deque
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# PART A: Preprocessing pipeline (paper Section 4.1)
# ---------------------------------------------------------------------------

class AtariPreprocessor:
    """
    Implements phi(s_t) from the paper: converts a history of raw RGB frames
    into an 84x84x4 stacked, grayscale, cropped tensor.
    """

    RAW_HEIGHT, RAW_WIDTH = 210, 160      # native Atari 2600 frame size
    RESIZE_HEIGHT, RESIZE_WIDTH = 110, 84  # step 3: down-sample target
    CROP_SIZE = 84                         # step 4: final square crop
    STACK_SIZE = 4                         # step 5: number of frames stacked

    def __init__(self):
        # Holds the last STACK_SIZE preprocessed (84x84) grayscale frames.
        self.frame_stack = deque(maxlen=self.STACK_SIZE)

    # --- Step 2: RGB -> grayscale -----------------------------------------
    @staticmethod
    def to_grayscale(frame_rgb: np.ndarray) -> np.ndarray:
        assert frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3, (
            f"Expected an HxWx3 RGB frame, got shape {frame_rgb.shape}"
        )
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    # --- Step 3: down-sample to 110x84 -------------------------------------
    def downsample(self, frame_gray: np.ndarray) -> np.ndarray:
        return cv2.resize(
            frame_gray,
            (self.RESIZE_WIDTH, self.RESIZE_HEIGHT),  # cv2 takes (width, height)
            interpolation=cv2.INTER_LINEAR,
        )

    # --- Step 4: crop 84x84 playing area ------------------------------------
    def crop(self, frame_110x84: np.ndarray) -> np.ndarray:
        # The paper crops out the score/border area and keeps the playing
        # field. We take the bottom 84 rows of the 110-row image, which is
        # the common convention (score bar sits at the top of most Atari
        # games). Adjust `top` if a specific game's HUD sits elsewhere.
        top = self.RESIZE_HEIGHT - self.CROP_SIZE  # 110 - 84 = 26
        return frame_110x84[top:top + self.CROP_SIZE, 0:self.CROP_SIZE]

    # --- Combine steps 2-4 for a single incoming frame ----------------------
    def preprocess_single_frame(self, frame_rgb: np.ndarray) -> np.ndarray:
        gray = self.to_grayscale(frame_rgb)
        resized = self.downsample(gray)
        cropped = self.crop(resized)
        assert cropped.shape == (self.CROP_SIZE, self.CROP_SIZE)
        return cropped

    # --- Step 5: push a new frame and return the stacked 84x84x4 tensor ----
    def step(self, frame_rgb: np.ndarray) -> np.ndarray:
        processed = self.preprocess_single_frame(frame_rgb)
        self.frame_stack.append(processed)

        # Pad with copies of the first frame until we have STACK_SIZE frames
        # (only happens at the very start of an episode).
        while len(self.frame_stack) < self.STACK_SIZE:
            self.frame_stack.append(processed)

        stacked = np.stack(self.frame_stack, axis=-1)  # -> (84, 84, 4)
        assert stacked.shape == (self.CROP_SIZE, self.CROP_SIZE, self.STACK_SIZE)
        return stacked

    def reset(self):
        self.frame_stack.clear()


# ---------------------------------------------------------------------------
# PART B: Environment setup + raw pixel capture (step 1) - OPTIONAL
# ---------------------------------------------------------------------------

def make_atari_env(game_id: str = "ALE/Breakout-v5"):
    """
    Sets up the Atari environment and returns it, ready to emit raw
    210x160x3 RGB frames on every step/reset.

    Requires (NOT installed automatically by this script):
        pip install "gymnasium[atari]" ale-py "autorom[accept-rom-license]"
        python3 -m AutoROM --accept-license   # downloads the Atari ROMs once

    See README_atari_preprocessing.md for the full step-by-step setup guide.
    """
    import gymnasium as gym  # local import: only needed if this function is called

    env = gym.make(game_id, render_mode="rgb_array")
    return env


def capture_raw_frame(env) -> np.ndarray:
    """Returns the current raw RGB frame straight from the emulator (step 1)."""
    frame = env.render()
    assert frame.shape == (210, 160, 3), f"Unexpected frame shape: {frame.shape}"
    return frame


def demo_with_real_environment(game_id: str = "ALE/Breakout-v5", n_steps: int = 8):
    """
    End-to-end demo: sets up the real Atari environment, plays a few random
    steps, captures raw pixels, and runs them through AtariPreprocessor.
    Only works if gymnasium + ale-py + ROMs are installed.
    """
    env = make_atari_env(game_id)
    preprocessor = AtariPreprocessor()

    obs, info = env.reset()
    preprocessor.reset()

    stacked_state = None
    for _ in range(n_steps):
        raw_frame = capture_raw_frame(env)          # step 1: raw pixels
        stacked_state = preprocessor.step(raw_frame)  # steps 2-5
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
            preprocessor.reset()

    env.close()
    return stacked_state


# ---------------------------------------------------------------------------
# Self-test using a synthetic frame (no external RL dependencies required)
# ---------------------------------------------------------------------------

def _make_synthetic_atari_frame() -> np.ndarray:
    """
    Builds a fake 210x160x3 RGB frame that looks a bit like an Atari screen
    (colored score bar on top + a colored 'ball' in the play area), purely
    so the preprocessing pipeline can be exercised without a real emulator.
    """
    frame = np.zeros((210, 160, 3), dtype=np.uint8)
    frame[:20, :, :] = (200, 200, 200)          # fake score bar (top)
    frame[20:, :, :] = (20, 60, 20)             # fake playing field (green)
    cv2.circle(frame, (80, 120), 6, (255, 0, 0), -1)  # fake ball
    return frame


def _self_test():
    print("Running preprocessing self-test on a synthetic frame...\n")
    preprocessor = AtariPreprocessor()

    raw = _make_synthetic_atari_frame()
    print(f"Step 1  - Raw frame captured        : shape={raw.shape}, dtype={raw.dtype}")

    gray = preprocessor.to_grayscale(raw)
    print(f"Step 2  - Grayscale conversion       : shape={gray.shape}, dtype={gray.dtype}")

    resized = preprocessor.downsample(gray)
    print(f"Step 3  - Down-sampled to 110x84     : shape={resized.shape}")

    cropped = preprocessor.crop(resized)
    print(f"Step 4  - Cropped to 84x84           : shape={cropped.shape}")

    # Feed 4 slightly different synthetic frames to exercise the frame stack.
    preprocessor.reset()
    for i in range(4):
        f = _make_synthetic_atari_frame()
        cv2.circle(f, (80 + i * 5, 120), 6, (255, 0, 0), -1)  # move the "ball"
        stacked = preprocessor.step(f)

    print(f"Step 5  - Stacked last 4 frames       : shape={stacked.shape} "
          f"(expected (84, 84, 4))")
    print("\nSelf-test passed: pipeline produces the expected 84x84x4 tensor.")

    return raw, gray, resized, cropped, stacked


if __name__ == "__main__":
    _self_test()
