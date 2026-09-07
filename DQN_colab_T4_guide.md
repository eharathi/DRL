# Running Faster on Google Colab (T4 GPU) — Training, Evaluating, Watching

Status: **not yet applied** — a saved reference, same as `DQN_GPU_Speedup_Guide.md`.
This file only covers what's *different* on Colab with a T4; the actual
`device` plumbing (the `get_device()` helper, and the call-site changes in
`dqn_train.py` / `dqn_evaluate.py` / `dqn_watch.py`) is exactly what's already
written up in `DQN_GPU_Speedup_Guide.md` — apply that first. On a Colab T4
runtime, `get_device()` will resolve to `"cuda"` automatically (no branching
needed for T4 specifically).

---

## 1. One-time Colab environment setup

Colab VMs are ephemeral — every fresh session needs this again:

```python
# Runtime > Change runtime type > T4 GPU, first.

!pip install "gymnasium[atari]" ale-py "autorom[accept-rom-license]"
!python3 -m AutoROM --accept-license   # downloads Atari ROMs, non-interactive
!pip install opencv-python-headless    # cv2 without a GUI backend (see Section 4)

import torch
print("CUDA available:", torch.cuda.is_available())   # should be True on a T4 runtime
print(torch.cuda.get_device_name(0))                    # "Tesla T4"
```

Get the project files onto the Colab VM (upload the folder, or `git clone` /
mount Google Drive) so `dqn_train.py`, `dqn_model.py`, etc. are importable
from the notebook's working directory.

**Persist checkpoints across sessions.** Colab sessions disconnect (idle
timeout / max runtime), and the VM's local disk is wiped between sessions.
Mount Drive and point the training output at it, or download
`dqn_breakout.pt` / `dqn_training_history_breakout.npz` at the end of each
session — otherwise `--resume` (added earlier in `dqn_train.py`) has nothing
to resume *from* next time:

```python
from google.colab import drive
drive.mount('/content/drive')
# then run training with weights_path / history_path pointed at
# /content/drive/MyDrive/... (both are derived from GAME_ID's tag in
# dqn_train.py today; simplest is to copy the .pt/.npz files to/from Drive
# around each run rather than changing the hardcoded paths).
```

Because a T4 session can still hit Colab's runtime limits well before a
very large `NUM_EPISODES` finishes, run training in chunks across sessions:

```bash
!python3 dqn_train.py --episodes 200            # first session
!python3 dqn_train.py --resume --episodes 200    # next session, continues from checkpoint
```

---

## 2. Training — what actually gets faster on a T4, and what doesn't

Once `get_device()` (from `DQN_GPU_Speedup_Guide.md`) is applied and resolves
to `cuda`:

- **The batch-32 SGD step** (forward + backward through `net` on
  `states_t`/`next_states_t`) is what benefits most — a T4 has far more raw
  throughput than a laptop CPU or this Mac's MPS backend, so this should show
  a clear, measurable speedup, unlike the "might not help" caveat noted for
  MPS on a network this small.
- **Action selection** (`select_action`, batch size 1, once per env step)
  still won't benefit much — CPU→GPU transfer overhead for a single 84×84×4
  frame can dominate the actual compute. This is inherent to acting
  one-step-at-a-time in a single environment; it's not specific to Colab.
- **Environment stepping itself** (ALE emulation, `AtariPreprocessor`'s
  `cv2` resize/grayscale) runs on CPU regardless of the GPU — a T4 doesn't
  speed this part up at all. On Colab this can dominate wall-clock more than
  it does locally, since Colab's *CPU* allocation (not just the GPU) is
  often weaker per-core than a modern laptop.

Two safe, numerics-preserving additions worth making alongside the basic
`device` plumbing, specifically because they only pay off on a real CUDA
GPU like the T4 (harmless no-ops on CPU/MPS, so no need to branch on
platform):

```python
torch.backends.cudnn.benchmark = True
```
Add this once near the top of `dqn_train.py`. Every forward pass here uses
the same fixed input shape (`(32, 4, 84, 84)` for training, `(1, 4, 84, 84)`
for action selection) — cuDNN's autotuner picks the fastest convolution
algorithm for that exact shape after a few warm-up iterations. Only helps on
CUDA; on CPU/MPS it's a harmless no-op.

```python
states_t = torch.from_numpy(phis).to(device, non_blocking=True)
```
`non_blocking=True` lets the host→device copy overlap with other work
*if* the source tensor is in pinned memory — a small, safe win on CUDA,
no effect on CPU/MPS. Not worth the added complexity of pinning the replay
buffer's memory for this project's scale, but worth knowing the flag is a
free no-op to add wherever `.to(device)` already appears from the base GPU
guide.

**Explicitly out of scope (per prior discussion):** batch size and mixed
precision (`torch.cuda.amp`) both directly change training numerics, in a
project that already has known instability from the missing target network
(see `DQN_Training_Guide.md` Step 8). Skip both — keep batch size at the
paper's 32, and skip AMP — so a T4 run stays numerically comparable to a
CPU/MPS run of the same code.

---

## 3. Evaluating — already Colab-safe, no changes needed

`dqn_evaluate.py` only ever uses `gym.make(..., render_mode="rgb_array")` —
this returns raw pixel arrays from the ALE emulator entirely in software, it
never opens a window or needs an X server. It already runs fine, unmodified,
in a headless Colab notebook. Once the base `device` plumbing from
`DQN_GPU_Speedup_Guide.md` is applied, running:

```bash
!python3 dqn_evaluate.py --weights dqn_breakout.pt --episodes 10
```
will automatically pick up the T4 via `get_device()` for the (batch-size-1)
inference forward pass — same "may not be a big win, batch size 1" caveat
as action-selection during training, but still correct and no slower than
CPU.

---

## 4. Watching — `dqn_watch.py`'s live `cv2.imshow()` window won't work in Colab

Colab has no OS display, so `cv2.imshow(...)` (used by `dqn_watch.py` for the
live video window) will fail there (`cv2.error: ... not implemented`, no GTK/
Cocoa backend). Rather than reworking `dqn_watch.py` for headless/inline
rendering, **reuse what already exists**: `dqn_evaluate.py --record` already
saves a GIF of episode 1 (`save_gif()`, no extra dependencies) — that's
already the Colab-friendly path, since it never opens a window.

```bash
!python3 dqn_evaluate.py --weights dqn_breakout.pt --episodes 5 --record --record-path dqn_gameplay.gif
```

Then, in a notebook cell, display the saved GIF inline:

```python
from IPython.display import Image
Image(filename="dqn_gameplay.gif")
```

This gives you the same "watch the trained model play" outcome as
`dqn_watch.py`'s live window, just after the episode finishes rather than
frame-by-frame in real time — no code changes needed to either
`dqn_evaluate.py` or `dqn_watch.py` for this to work today.

---

## Summary Table

| Task | Change needed on Colab T4 | Extra vs. local (MPS) setup |
|---|---|---|
| One-time setup | Install `gymnasium[atari]`, `ale-py`, `AutoROM`, accept ROM license | Not needed locally (already installed per `DQN_Training_Guide.md` Step 0) |
| Persist checkpoints | Mount Drive or download `.pt`/`.npz` each session | Not needed locally (disk persists) |
| Apply GPU support | `get_device()` from `DQN_GPU_Speedup_Guide.md` (resolves to `cuda`) | Same helper, different resolved device |
| Extra safe speedup | `torch.backends.cudnn.benchmark = True`, `non_blocking=True` transfers | CUDA-only wins, harmless no-ops elsewhere |
| Skip | Mixed precision (AMP), larger batch size | Changes numerics — not applied per prior decision |
| Evaluate | No changes — already headless-safe (`render_mode="rgb_array"`) | Same |
| Watch | Use `dqn_evaluate.py --record` + `IPython.display.Image`, not `dqn_watch.py`'s live window | Locally, `dqn_watch.py`'s `cv2.imshow` window works directly |
