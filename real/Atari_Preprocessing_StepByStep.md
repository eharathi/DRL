# Atari Frame Preprocessing — Step-by-Step Guide

Reproduces **Section 4.1 ("Preprocessing and Model Architecture")** of
Mnih et al., *"Playing Atari with Deep Reinforcement Learning"* (DeepMind, 2013).

Goal: turn a raw Atari 2600 screen (210×160 RGB) into the 84×84×4 tensor
that the DQN's convolutional network actually consumes.

Companion files (share all three together):
- `atari_preprocessing.py` — the working code
- `atari_preprocessing_stages.png` — visual proof of each stage
- `Atari_Preprocessing_StepByStep.md` — this guide

---

## 0. What you'll build

```
Raw frame (210x160x3, RGB)
   -> Grayscale (210x160)
   -> Down-sample (110x84)
   -> Crop (84x84)
   -> Stack last 4 frames (84x84x4)  <-- final input to the network
```

---

## Step 1 — Set up your Python environment

You need two independent things:

**A. Image-processing packages (required, lightweight)**
```bash
pip install numpy opencv-python matplotlib
```

**B. The Atari environment itself (optional — only needed to capture pixels
from a real game instead of a test image)**
```bash
pip install "gymnasium[atari]" ale-py "autorom[accept-rom-license]"
python3 -m AutoROM --accept-license
```
`AutoROM` downloads the actual Atari ROM binaries once (a few MB) so the
emulator (`ale-py`) has games to run. You only need to do this once per
machine.

> If you only want to verify/replicate the *preprocessing math* (steps 2–5
> below), you can skip 1B entirely — `atari_preprocessing.py` includes a
> synthetic-frame self-test that needs no emulator at all.

---

## Step 2 — Set up the environment and capture a raw frame

```python
import gymnasium as gym

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
obs, info = env.reset()

raw_frame = env.render()          # <-- this is the raw pixel capture
print(raw_frame.shape)            # (210, 160, 3)  uint8 RGB
```

This is **exactly** the input the paper describes: *"a vector of raw pixel
values representing the current screen,"* 210×160, RGB, from the Atari
2600 emulator (via the Arcade Learning Environment).

---

## Step 3 — Convert RGB → Grayscale

Color carries little useful signal for control (most Atari sprites are
flat-colored), so the first reduction drops it.

```python
import cv2

def to_grayscale(frame_rgb):
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

gray = to_grayscale(raw_frame)   # (210, 160)
```

---

## Step 4 — Down-sample to 110 × 84

Reduces resolution/compute while keeping the frame's aspect ratio close to
the original, per the paper.

```python
def downsample(frame_gray):
    return cv2.resize(frame_gray, (84, 110), interpolation=cv2.INTER_LINEAR)
    # cv2.resize takes (width, height) -> (84, 110)

resized = downsample(gray)       # (110, 84)
```

### Why 110×84, and not exactly half of 210×160?

Exact half of 210×160 would be **105×80** — but the paper uses **110×84**.
Neither dimension is a clean 2× reduction, which is the tell that the
target wasn't "shrink by half"; it was "hit a specific final square size."

Work backward from the real goal: a **square 84×84** input (required
because the paper uses the GPU 2D-convolution implementation from
Krizhevsky et al.'s ImageNet paper, which only accepts square inputs).

1. **Fix the width to land exactly on 84.**
   `scale = 84 / 160 = 0.525`
2. **Apply that same scale factor to the height**, to preserve the
   original aspect ratio instead of distorting the image:
   `210 x 0.525 = 110.25 -> rounds to 110`

That reproduces the paper's exact numbers. The payoff: after this resize,
the **width is already 84 — done, no horizontal cropping needed.** The
height is deliberately left taller (110, not 84) so that Step 5 can crop
out exactly 84 rows containing the play field and discard the
scoreboard/border strip (the leftover `110 - 84 = 26` rows). This is why
the pipeline *crops* the height afterward instead of *padding* it — the
resize intentionally overshoots in one dimension only.

If a naive exact-half scale (105×80) had been used instead, the width
would already be *narrower* than the required 84, forcing padding with
fake pixels instead of a clean crop of real image content.

### Why 84 specifically (not 80, 96, 100, ...)?

84 isn't arbitrary — it's chosen to tile perfectly through the first two
convolutional layers of the network (8×8 kernel/stride 4, then 4×4
kernel/stride 2):

| Input size | Conv1 output (8×8, stride 4) | Conv2 output (4×4, stride 2) | Tiles evenly? |
|---|---|---|---|
| **84×84** | 20×20 | **9×9** | Yes — the last filter window ends exactly on the last pixel; nothing is dropped |
| 80×80 (naive half) | 19×19 | 8×8, with 1 leftover row/col | No — a stride-2 kernel can't evenly cover a 19-wide map; one edge pixel is never touched by any filter |

So 84 was picked because, combined with the (8,4)/(4,2) kernel/stride
pattern, it produces clean, fully-tiled feature maps (20×20 → 9×9,
flattened to 2592 values feeding the 256-unit fully-connected layer) with
zero wasted edge pixels — not merely because it's "close to half of 160."

> **Caveat:** the paper itself never spells out this arithmetic — it only
> states the numbers "110×84" and "84×84," and explains the crop is needed
> because the GPU convolution implementation expects square inputs. The
> derivation above is a reasoned reconstruction, not a quote from the
> authors — but it reproduces their exact published numbers precisely,
> which is fairly strong circumstantial evidence for this being the logic
> behind the choice.

---

## Step 5 — Crop to a square 84 × 84 playing region

The paper crops to a square only because the GPU convolution routine they
used required square inputs. The crop should keep the actual play field and
drop the scoreboard/border.

```python
def crop(frame_110x84):
    top = 110 - 84   # = 26
    return frame_110x84[top:top + 84, 0:84]

cropped = crop(resized)          # (84, 84)
```

> **Note:** the exact crop offset is game-dependent (where the HUD sits
> varies by game). The snippet above takes the *bottom* 84 rows, which
> works for games with a top scoreboard (e.g. Breakout, Space Invaders).
> Inspect a sample frame for your chosen game and adjust `top` if needed.

---

## Step 6 — Stack the last 4 frames

A single frame is a static image — it can't show a ball's direction or an
enemy's velocity. Stacking the last 4 preprocessed frames gives the network
implicit motion information without needing a recurrent architecture.

```python
from collections import deque
import numpy as np

frame_stack = deque(maxlen=4)

def stack_frames(new_cropped_frame):
    frame_stack.append(new_cropped_frame)
    while len(frame_stack) < 4:          # pad at episode start
        frame_stack.append(new_cropped_frame)
    return np.stack(frame_stack, axis=-1)  # (84, 84, 4)

state = stack_frames(cropped)
print(state.shape)   # (84, 84, 4)  <- this is what feeds the CNN
```

---

## Step 7 — Run the full pipeline end-to-end

All of steps 3–6 are already implemented as a single reusable class in
`atari_preprocessing.py` (`AtariPreprocessor`). To replicate:

```bash
python3 atari_preprocessing.py
```

Expected console output:
```
Step 1  - Raw frame captured        : shape=(210, 160, 3), dtype=uint8
Step 2  - Grayscale conversion       : shape=(210, 160), dtype=uint8
Step 3  - Down-sampled to 110x84     : shape=(110, 84)
Step 4  - Cropped to 84x84           : shape=(84, 84)
Step 5  - Stacked last 4 frames       : shape=(84, 84, 4) (expected (84, 84, 4))

Self-test passed: pipeline produces the expected 84x84x4 tensor.
```

This self-test uses a synthetic generated frame (a fake scoreboard + a
moving colored ball) so it runs with **no Atari ROMs or emulator required**
— useful for anyone replicating this who just wants to check the pixel math.

---

## Step 8 — (Optional) Run it against a real Atari game

Once the packages from Step 1B are installed:

```python
from atari_preprocessing import demo_with_real_environment

final_state = demo_with_real_environment("ALE/Breakout-v5", n_steps=8)
print(final_state.shape)   # (84, 84, 4)
```

This plays a few random steps in real Breakout, capturing and preprocessing
each frame exactly as in Steps 2–6.

---

## Step 9 — Visualize the stages (for your presentation/report)

```python
import matplotlib.pyplot as plt
from atari_preprocessing import _self_test

raw, gray, resized, cropped, stacked = _self_test()

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
axes[0].imshow(raw);                      axes[0].set_title("1. Raw RGB\n210x160x3")
axes[1].imshow(gray, cmap="gray");        axes[1].set_title("2. Grayscale\n210x160")
axes[2].imshow(resized, cmap="gray");     axes[2].set_title("3. Down-sampled\n110x84")
axes[3].imshow(cropped, cmap="gray");     axes[3].set_title("4. Cropped\n84x84")
axes[4].imshow(stacked[:, :, -1], cmap="gray"); axes[4].set_title("5. Latest of 4-stack")
for ax in axes: ax.axis("off")
plt.tight_layout()
plt.savefig("atari_preprocessing_stages.png", dpi=150)
```

Result: `atari_preprocessing_stages.png` (already generated and included).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: gymnasium` | Only needed for Step 8 (real env). Steps 1–7 run without it. |
| `AutoROM` license prompt hangs | Use the non-interactive flag: `python3 -m AutoROM --accept-license` |
| Cropped frame cuts off the paddle/player | Adjust the `top` offset in Step 5 for that specific game's HUD height |
| Colors look inverted in matplotlib | `cv2` reads/writes BGR by default for file I/O; this pipeline works directly on in-memory RGB arrays from `env.render()`, so no channel-order fix is needed here |

---

## Summary Table

| Step | Operation | Input shape | Output shape |
|---|---|---|---|
| 1 | Capture raw frame | — | 210×160×3 |
| 2 | RGB → Grayscale | 210×160×3 | 210×160 |
| 3 | Down-sample | 210×160 | 110×84 |
| 4 | Crop | 110×84 | 84×84 |
| 5 | Stack last 4 frames | 84×84 (×4) | 84×84×4 |

This 84×84×4 tensor is exactly what feeds the first convolutional layer of
the DQN described in the paper (16 filters, 8×8, stride 4).
