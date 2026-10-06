# DRL – DQN on Atari Breakout

PyTorch reproduction of Mnih et al., "Playing Atari with Deep Reinforcement
Learning" (2013), plus three stability changes from the 2015 Nature DQN paper
(frozen target network, Huber loss, gradient-norm clipping). `dqn_train_backup.py`
is the unmodified original Algorithm 1 version, kept for comparison.

## Files required to replicate

Minimum for training on a new machine (put in one folder):

| File | Purpose |
|---|---|
| `dqn_train.py` | Training loop |
| `dqn_model.py` | `DQN` network class |
| `atari_preprocessing.py` | `AtariPreprocessor` (grayscale, resize, 4-frame stack) |

Optional:

| File | Purpose |
|---|---|
| `dqn_evaluate.py` | Evaluate a trained checkpoint (also imported by `dqn_watch.py`) |
| `dqn_watch.py` | Watch the agent play |
| `measure_dqn.py` | Plot / measure a training-history `.npz` |
| `toy_pixel_env.py` | Dependency-free toy env, only when `USE_TOY_ENV = True` |
| `dqn_breakout.pt` + `dqn_training_history_breakout.npz` | Only to use `--resume` or to evaluate without retraining (copy both together) |
| `DQN_Paper_Reproduction.ipynb` | Notebook version |
| `dqn_train_backup.py` | Original Algorithm 1 snapshot, for comparison |

Not needed: guides (`*.md`), PDFs, PPTX, images, `__pycache__`, `.DS_Store`.

## Dependencies

No `requirements.txt` yet. The code imports: `torch`, `gymnasium`, `ale-py`,
`numpy`, `opencv-python` (`cv2`), `matplotlib`, `pillow`. Python 3.10+ is
required (`str | None` type hints).

Versions on the original machine (from `pip list`; confirm with `pip freeze`
in your training environment): torch 2.11.0, gymnasium 1.3.0, ale-py 0.12.1,
numpy 2.1.1, opencv-python 4.11.0.86, matplotlib 3.10.1, pillow 11.1.0.

```
python3 -m venv .venv && source .venv/bin/activate
pip install torch gymnasium ale-py numpy opencv-python matplotlib pillow
```

For a GPU, install the matching CUDA build of torch. As written, the training
code runs on CPU only (no `.to(device)` calls).

## Usage

```
python3 dqn_train.py                          # train from scratch
python3 dqn_train.py --resume                 # continue from saved weights + history
python3 dqn_train.py --resume --episodes 40   # resume, run only 40 more episodes
python3 measure_dqn.py dqn_training_history_breakout.npz
```

Game and scale are set at the top of `dqn_train.py` (`GAME_ID`, `NUM_EPISODES`,
`REPLAY_CAPACITY`, `EPS_DECAY_STEPS`, ...). Defaults are reduced from the
paper's settings so a run finishes in minutes to hours, not a week.

## Saved outputs (all written to the working directory)

The tag is derived from the game: `ALE/Breakout-v5` -> `breakout`
(`toy_catch` for the toy env).

| File | Contents |
|---|---|
| `dqn_<tag>.pt` e.g. `dqn_breakout.pt` | Trained network weights |
| `dqn_training_history_<tag>.npz` e.g. `dqn_training_history_breakout.npz` | Training curves |
| `dqn_training_metrics_<tag>.png` | Plot of the curves (made by the measure/plot step) |

Other files in the repo, such as `dqn_breakout_paper_notebook.pt` and its
`.npz`, come from the notebook run and are separate from the script's output.

### What is a `.pt` file?

A PyTorch file written with `torch.save(net.state_dict(), path)`. It holds
only the network's weights (a `state_dict`: layer name -> tensor), not the
model code. To use it, build `DQN(in_channels=4, num_actions=...)` from
`dqn_model.py` and call `net.load_state_dict(torch.load(path, map_location="cpu"))`.
The trainer's `--resume` and `dqn_evaluate.py` both do this. Only the live
network is saved; the frozen target network and optimizer state are not.

### What is a `.npz` file?

A NumPy archive written with `np.savez`, containing three arrays:

- `episode_rewards`: true (unclipped) score per episode
- `avg_q_history`: average max-Q on the fixed held-out states, per episode
- `loss_history`: loss per gradient step

Load with `np.load(path)`. With `--resume`, new results are appended to the
existing arrays.

## Replay memory is NOT saved

The replay buffer is a `deque` in RAM (`ReplayMemory` in `dqn_train.py`,
capacity `REPLAY_CAPACITY`) and is discarded when the script exits. Neither
the `.pt` nor the `.npz` contains it. So `--resume` restores the weights and
history, but starts with an empty buffer that refills (`MIN_REPLAY_BEFORE_TRAIN`
steps) before learning resumes. It also resumes with exploration epsilon at the
floor (0.1) and a fresh optimizer. This is not an exact continuation of the
previous run.
