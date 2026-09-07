# Speeding Up Training with a GPU (CUDA / Apple Silicon MPS)

Status: **not yet applied** — this is a saved reference for a future update.
None of `dqn_train.py`, `dqn_evaluate.py`, or `dqn_watch.py` currently move
any tensor off the CPU.

---

## Why training is slow, and what this does / doesn't fix

`dqn_train.py` runs a gradient step on almost every environment step once the
replay buffer is warmed up (`MIN_REPLAY_BEFORE_TRAIN`). Wall-clock time for
`NUM_EPISODES` scales with total environment steps × per-step forward +
backward cost. Moving the network and its tensors to a GPU reduces the
per-step compute cost; it does **not** reduce the number of steps or
episodes, so this is exactly "keep `NUM_EPISODES` large, cut per-episode
time" — no algorithmic changes (epsilon schedule, replay buffer, or the
every-step training trigger from Algorithm 1 stay exactly as they are).

**Honest caveat:** this network is tiny (16/32-filter convs, 256-unit FC),
and action-selection forward passes run with batch size 1. GPU/MPS transfer
overhead can make single-sample forward passes *slower* than CPU, even
though the batch-32 training step should benefit. Time a short run
before/after applying this to confirm it actually helps on your hardware —
don't assume it's a win without measuring, in the same spirit as the
"honest limitations" discussion in `DQN_Training_Guide.md` Step 8.

This machine (checked via `torch.cuda.is_available()` /
`torch.backends.mps.is_available()`): CUDA not available, MPS (Apple
Silicon GPU) available.

---

## 1. Add one shared device-selection helper

Add to `dqn_model.py` (imported by all three scripts already), so CUDA/MPS/CPU
detection isn't duplicated three times:

```python
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

Auto-detecting (CUDA > MPS > CPU) means the same code works unchanged on this
Mac (MPS) and on an NVIDIA machine (CUDA) later.

---

## 2. `dqn_train.py` — move the network and training tensors to device

- Import `get_device` from `dqn_model`.
- In `train()`: compute `device = get_device()` once, print it (e.g.
  `Using device: mps`), and do:
  ```python
  net = DQN(in_channels=4, num_actions=num_actions).to(device)
  ```
- Resume path (`--resume`): load the checkpoint directly onto the target
  device instead of CPU-then-move:
  ```python
  net.load_state_dict(torch.load(resume_weights_path, map_location=device))
  ```
- `select_action()` and `compute_avg_max_q()` both call `to_tensor_batch(...)`
  then run a forward pass — give each an extra `device` parameter and move
  the tensor with `.to(device)` right after building it, before the forward
  pass.
- Main training loop: after building `states_t, next_states_t, actions_t,
  rewards_t, dones_t`, move each to `device` before they're used in the
  Bellman-target / current-Q / loss computation.

---

## 3. `dqn_evaluate.py` — same device plumbing (also reused by `dqn_watch.py`)

- Import `get_device` from `dqn_model`.
- `load_trained_net(weights_path, num_actions, device)`: build
  `DQN(...).to(device)`, then
  `torch.load(weights_path, map_location=device)`, `load_state_dict`, `.eval()`.
- `select_action_eval(net, phi_hwc, epsilon, num_actions, device)`: move the
  tensor from `to_tensor(...)` to `device` before the forward pass.
- `run_episode(...)`: accept and thread through `device` to
  `select_action_eval`.
- `main()`: compute `device = get_device()` once, print it, pass it to
  `load_trained_net` and `run_episode`.

---

## 4. `dqn_watch.py` — update call sites for the new `device` parameter

- `main()`: compute `device = get_device()` (imported from `dqn_model`),
  print it, pass it to `load_trained_net(args.weights, num_actions, device)`.
- `watch_episode(...)`: accept `device` and pass it through to
  `select_action_eval(net, phi, epsilon, num_actions, device)`.

No other files need changes — `atari_preprocessing.py`, `toy_pixel_env.py`,
and `measure_dqn.py` never touch model tensors.

---

## Verification checklist (once applied)

1. `python3 -c "import ast; ast.parse(open('dqn_train.py').read())"` (and the
   same for `dqn_evaluate.py`, `dqn_watch.py`) — confirms syntax.
2. `python3 dqn_train.py --help` still shows `--resume` / `--episodes` — no
   argparse regressions.
3. Quick toy-env smoke test (fast, no ROM dependency) confirming:
   - `get_device()` reports the expected device (e.g. `mps`).
   - `next(net.parameters()).device` matches.
   - A couple of episodes of training complete with a non-NaN, sane loss.
4. `python3 dqn_evaluate.py --weights dqn_toy_catch.pt --toy --episodes 2` and
   `python3 dqn_watch.py --weights dqn_toy_catch.pt --toy --episodes 1` both
   run against the new device-aware `load_trained_net`.
5. Time a short real-Atari run before/after (e.g. `--episodes 10`) to get an
   actual wall-clock comparison, and note honestly if the GPU path turns out
   not to help for a network this small.
