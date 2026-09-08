# DQN Architecture — Step-by-Step Implementation Guide

Reproduces the network architecture from **Section 4.1** of
Mnih et al., *"Playing Atari with Deep Reinforcement Learning"* (DeepMind, 2013)
(the slide-6 architecture in the presentation deck).

Companion file: `dqn_model.py` (working, shape-verified PyTorch implementation).

---

## 0. What you're building

```
Input: 84x84x4 stacked frames (from atari_preprocessing.py)
   -> Conv1: 16 filters, 8x8, stride 4  + ReLU   -> 20x20x16
   -> Conv2: 32 filters, 4x4, stride 2  + ReLU   ->  9x 9x32
   -> Flatten                                    -> 2592
   -> FC:    256 units + ReLU                    -> 256
   -> Output: linear, 1 unit per action           -> num_actions (raw Q-values)
```

Every step below explains **why** that specific choice was made, not just what it is.

---

## Step 1 — Set up your environment

```bash
pip install torch
```
(TensorFlow/Keras would work equally well; PyTorch is used here because its
explicit `forward()` method makes it easy to print intermediate shapes,
which is the whole point of this guide.)

---

## Step 2 — Define the input layer: 84×84×4

```python
import torch.nn as nn

in_channels = 4   # the 4 stacked frames from atari_preprocessing.py
```

**Why 4 input channels, not 3 (like RGB)?** Each of the 4 stacked grayscale
frames is treated as one input "channel," exactly like Red/Green/Blue are 3
channels in a color image. This is what lets a plain feed-forward CNN
implicitly perceive motion — see the earlier discussion on frame stacking.
The convolution filters in Step 3 will have depth 4 to match.

---

## Step 3 — First convolutional layer: 16 filters, 8×8, stride 4

```python
conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
```

**Why a big 8×8 kernel with a big stride (4), instead of small 3×3 filters
like modern vision CNNs use?**
- Atari objects (paddles, balls, aliens) are large, blocky, low-detail
  sprites — an 8×8 receptive field at the input resolution is already
  enough to capture a small object's full shape in one glance.
- A large stride (4) aggressively shrinks the spatial resolution early
  (84→20), which keeps compute manageable — this network had to run fast
  enough to process ~10 million training frames on 2013-era hardware.

**Why 16 filters?** 16 is a deliberately *small* number of feature detectors
for the first layer — enough to learn basic primitives (edges, blobs,
simple shapes) without over-parameterizing a network that has to train
from a noisy, non-stationary RL signal (unlike supervised learning with
millions of clean labels).

**Output shape check:** `(84 - 8) / 4 + 1 = 20` → **20×20×16**, confirmed by
the self-test in `dqn_model.py`.

**Why ReLU right after?**
```python
relu = nn.ReLU()
x = relu(conv1(input_tensor))
```
ReLU (`max(0, x)`) was the standard nonlinearity choice at the time (cited
from Jarrett et al. 2009 / Nair & Hinton 2010 in the paper's references) —
cheap to compute and avoids the vanishing-gradient problems of sigmoid/tanh,
important for a network trained with noisy RL gradients over millions of
steps.

---

## Step 4 — Second convolutional layer: 32 filters, 4×4, stride 2

```python
conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
```

**Why smaller kernel/stride than layer 1?** By this point the input is
already a compact 20×20 feature map (not raw pixels), so a smaller 4×4
window is enough to combine layer-1's simple features (edges/blobs) into
more complex ones (e.g., "paddle shape," "ball-plus-trail"). The reduced
stride (2 instead of 4) shrinks resolution more gently, since there's much
less spatial redundancy left to discard at this depth.

**Why double the filter count (16 → 32)?** A common CNN design pattern:
as spatial resolution shrinks, the *channel* dimension grows, so the total
representational capacity per layer is roughly preserved even as the
map gets smaller.

**Output shape check:** `(20 - 4) / 2 + 1 = 9` → **9×9×32**, confirmed by
the self-test.

(Recall from the earlier discussion: this is exactly why the input was
84, not 80 — an 84-pixel input tiles perfectly through both of these
conv layers with zero leftover edge pixels; an 80-pixel input would not.)

---

## Step 5 — Flatten before the fully-connected layer

```python
x = x.flatten(start_dim=1)   # (batch, 32, 9, 9) -> (batch, 2592)
```

`32 * 9 * 9 = 2592`. Fully-connected layers expect a flat vector per
sample, so the 3D feature map (channels × height × width) is unrolled into
one long vector. No learnable parameters here — it's a reshape, not a layer.

---

## Step 6 — Fully-connected layer: 256 ReLU units

```python
fc1 = nn.Linear(2592, 256)
x = relu(fc1(x))
```

**Why a fully-connected layer at all, after two conv layers?** The conv
layers extract *local, translation-invariant* visual features (a ball
looks the same regardless of where it is on screen). But deciding the
*value of an action* requires reasoning about the *global* configuration —
where the paddle is *relative to* the ball, not just "there is a ball
somewhere." A fully-connected layer lets every spatial feature location
influence every output unit, which is necessary for this global reasoning.

**Why 256 units specifically?** Not derived from any formula — an
empirically chosen capacity: large enough to represent a useful summary of
the 2592-dimensional conv output, small enough to keep the network fast and
avoid overfitting to the limited, noisy RL reward signal.

---

## Step 7 — Output layer: one linear unit per action, NO activation

```python
out = nn.Linear(256, num_actions)   # num_actions = 4..18 depending on the game
q_values = out(x)                    # NO relu/sigmoid here
```

**Why one output per action, instead of one network call per (state,
action) pair?** This is an explicit design choice the paper calls out
(Section 4.1): some prior approaches fed `(state, action)` into the network
and got back a single scalar Q-value, requiring a separate forward pass
*per action* to find the best one — cost scales linearly with the number
of actions. Here, the network takes only the *state* as input and produces
*all* Q-values in a single forward pass — both the training update
(`max_a' Q(s', a')` in the Bellman target) and acting (`argmax_a Q(s,a)`)
need exactly this "all actions at once" output.

**Why no activation function on the output?** Q-values are unconstrained
real numbers — the paper's reward clipping makes per-step rewards ∈ {-1, 0,
+1}, but *discounted sums* of many such rewards can still be any real
number, positive or negative, and can exceed 1 in magnitude (e.g., a long
successful episode accumulates many +1's). A ReLU would incorrectly forbid
negative Q-values; a sigmoid/tanh would incorrectly cap the magnitude.
A plain linear layer is the only mathematically correct choice here.

---

## Step 8 — Run the full pipeline and verify shapes

All of Steps 2–7 are implemented as a single reusable `DQN` class in
`dqn_model.py`. To replicate:

```bash
python3 dqn_model.py
```

Expected output:
```
Input                : (8, 4, 84, 84)  (expected (batch, 4, 84, 84))
After Conv1 + ReLU    : (8, 16, 20, 20)  (expected (batch, 16, 20, 20))
After Conv2 + ReLU    : (8, 32, 9, 9)  (expected (batch, 32, 9, 9))
After Flatten         : (8, 2592)  (expected (batch, 2592))
After FC1 + ReLU      : (8, 256)  (expected (batch, 256))
Output (Q-values)     : (8, 4)  (expected (batch, 4))

Total trainable parameters: 677,172

Self-test passed: all shapes match the paper's architecture.
```

The parameter count (~677K) is tiny by modern standards (compare to
today's vision models with hundreds of millions+) — a deliberate result of
the paper's design choices (few, large-stride conv filters; a single modest
FC layer) needed to make training on 2013 hardware over 10 million frames
feasible at all.

---

## Step 9 — A note on connecting this to preprocessing

`DQN.forward()` expects a float tensor of shape `(batch, 4, 84, 84)` with
values normalized to roughly `[0, 1]`. The output of `AtariPreprocessor`
(from `atari_preprocessing.py`) is a `uint8` array of shape `(84, 84, 4)`
(height, width, channels-last). Two conversions are needed to bridge them:

```python
import torch

def to_model_input(stacked_frames_hw4_uint8):
    # (84, 84, 4) uint8 -> (4, 84, 84) float32 in [0, 1]
    x = stacked_frames_hw4_uint8.astype("float32") / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1)   # HWC -> CHW
    return x.unsqueeze(0)  # add batch dimension -> (1, 4, 84, 84)
```

This conversion is already included in `dqn_train.py` (next section) —
noted here so the shape convention (channels-last from preprocessing vs.
channels-first for PyTorch) doesn't cause a silent bug when you wire the
two files together.

---

## Summary Table

| Layer | Type | Params | Output shape | Key hyperparameter reasoning |
|---|---|---|---|---|
| Input | — | — | 84×84×4 | 4 stacked frames = short motion history |
| Conv1 | Conv2D + ReLU | 8×8, stride 4, 16 filters | 20×20×16 | large receptive field for large blocky sprites; aggressive downsampling for speed |
| Conv2 | Conv2D + ReLU | 4×4, stride 2, 32 filters | 9×9×32 | combines layer-1 features; input size (84) chosen so this tiles with zero leftover pixels |
| Flatten | Reshape | — | 2592 | bridges conv features to FC layer |
| FC1 | Linear + ReLU | 256 units | 256 | global reasoning over relative object positions |
| Output | Linear (no activation) | num_actions units | num_actions | one Q-value per action, single forward pass, unconstrained real values |

Next: `DQN_Training_Guide.md` covers training this network with experience
replay and measuring its learning progress, matching Section 5 of the paper.
