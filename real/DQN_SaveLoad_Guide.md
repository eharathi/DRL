# Saving, Loading, and Reusing Trained DQN Weights

Companion file: `dqn_evaluate.py` (loads a checkpoint and runs the agent).

---

## 1. How weights are already being saved (in `dqn_train.py`)

```python
torch.save(net.state_dict(), weights_path)   # e.g. "dqn_breakout.pt"
```

**Why `net.state_dict()` and not `net` itself?**
`state_dict()` returns just an ordered dictionary of tensor name → tensor
value (every conv/linear layer's weights and biases) — a plain data
structure with no dependency on the class definition. Saving the whole
`net` object with `torch.save(net, ...)` instead would *pickle the class
definition too*, which is fragile: if `dqn_model.py` changes at all later
(even a comment-only refactor that changes line numbers), unpickling an
old full-object checkpoint can break. Saving/loading only the
`state_dict` is the standard, robust PyTorch convention.

After every training run, you'll have:
- `dqn_breakout.pt` — the weights (a plain dict of tensors, ~2.7 MB for this architecture)

This file is *portable*: copy it anywhere, share it, back it up — it has
no dependency on the training script that produced it, only on the `DQN`
class shape (input channels, layer sizes, number of actions) matching.

---

## 2. How to load weights back (the "reuse" step)

```python
from dqn_model import DQN
import torch

num_actions = 4   # MUST match what the checkpoint was trained with (Breakout = 4)
net = DQN(in_channels=4, num_actions=num_actions)   # 1. build an identical, freshly-initialized network
net.load_state_dict(torch.load("dqn_breakout.pt", map_location="cpu"))  # 2. copy trained weights in
net.eval()   # 3. set evaluation mode
```

Three steps, in order, and **all three matter**:

1. **Rebuild the exact same architecture first.** `state_dict` is just
   numbers — PyTorch has to know which tensor goes into which layer, and
   it matches purely by name (`conv1.weight`, `fc1.bias`, etc.) and shape.
   If you construct `DQN` with the wrong `num_actions` (say, 6 instead of
   4), `load_state_dict` will raise a shape-mismatch error on the output
   layer — this is a *good* thing, it stops you from silently loading
   weights into an incompatible network.
2. **`map_location="cpu"`** makes the checkpoint load correctly even if it
   was originally saved from a GPU session and you're now on a CPU-only
   machine (not the case in this project so far, but it's the standard
   defensive default).
3. **`net.eval()`** switches off training-only behavior (dropout,
   batch-norm running-stats updates). This particular architecture has
   neither, so it's a no-op here — but it's correct practice to always
   call it before using a network purely for inference, since forgetting
   it is a common, hard-to-notice bug in architectures that do have such
   layers.

All of this is already implemented in `load_trained_net()` inside
`dqn_evaluate.py` — the snippet above is exactly what that function does.

---

## 3. Running (reusing) the trained weights

```bash
python3 dqn_evaluate.py --weights dqn_breakout.pt --episodes 5
```

What this does, step by step (see `dqn_evaluate.py`):
1. Creates the environment (`ALE/Breakout-v5` by default — pass `--toy`
   to instead evaluate a `toy_pixel_env` checkpoint).
2. Loads the trained network as in Section 2.
3. Runs `--episodes` full episodes using an **epsilon-greedy policy with
   epsilon = 0.05** — not epsilon = 0 (pure greedy).
4. Prints the reward and step count per episode, then the average ± std
   dev across all episodes.

**Why epsilon = 0.05 at evaluation time, not 0?** This is a direct
reproduction of the paper's own evaluation convention (Section 5.3): *"we
follow the evaluation strategy ... and report the average score obtained
by running an epsilon-greedy policy with epsilon = 0.05."* A fully greedy
policy (epsilon = 0) risks getting stuck: Atari emulation can be
near-deterministic, so a purely greedy agent can loop the exact same
sequence of actions forever if it ever revisits a state it's seen before.
A small amount of randomness breaks these loops and gives a more
realistic estimate of real play.

Example run against the checkpoint already trained in this project
(only 80 episodes — a short demo run, not the paper's full 10M-frame
training, so scores are modest):

```
Loading weights from dqn_breakout.pt (4 actions)...
Episode 1: reward = 0.0, steps = 122
Episode 2: reward = 0.0, steps = 122
Episode 3: reward = 1.0, steps = 150

Average reward over 3 episodes (epsilon=0.05): 0.33 (+/- 0.47)
```

---

## 4. Watching the trained agent play

```bash
python3 dqn_evaluate.py --weights dqn_breakout.pt --record --record-path dqn_gameplay.gif
```

`--record` captures every raw frame of episode 1 and saves it as an
animated GIF (`dqn_gameplay.gif`) using Pillow — no extra dependencies
(no `ffmpeg` needed). Useful for including a short gameplay clip in a
presentation or report. Verified output: a 98-frame, 160×210 GIF.

---

## 5. Evaluating a *different* game or checkpoint

Since `num_actions` and the environment are both read dynamically, you
can reuse `dqn_evaluate.py` for any checkpoint trained via `dqn_train.py`,
as long as you pass the matching game id:

```bash
# if you trained on Pong:
python3 dqn_train.py            # after setting GAME_ID = "ALE/Pong-v5" in dqn_train.py
python3 dqn_evaluate.py --weights dqn_pong.pt --game ALE/Pong-v5 --episodes 5

# to evaluate the toy-environment checkpoint instead:
python3 dqn_evaluate.py --weights dqn_toy_catch.pt --toy --episodes 5
```

If `--game` doesn't match the game the checkpoint was actually trained
on, loading will still succeed as long as both games have the *same
number of actions* — but the agent will play badly, since the learned
weights encode strategy for a different game entirely. There's no
automatic check for this beyond the action-count match, so keep track of
which checkpoint belongs to which game (the filename tagging in
`dqn_train.py`, e.g. `dqn_breakout.pt` vs `dqn_pong.pt`, exists exactly to
prevent this mix-up).

---

## Summary Table

| Task | Command / Code |
|---|---|
| Save weights (already happens at end of training) | `torch.save(net.state_dict(), "dqn_breakout.pt")` |
| Rebuild the same architecture | `DQN(in_channels=4, num_actions=N)` |
| Load weights into it | `net.load_state_dict(torch.load(path)); net.eval()` |
| Run the trained agent, get average score | `python3 dqn_evaluate.py --weights dqn_breakout.pt --episodes 5` |
| Record a gameplay clip | `python3 dqn_evaluate.py --record` |
| Evaluate a toy-env checkpoint | `python3 dqn_evaluate.py --toy --weights dqn_toy_catch.pt` |
