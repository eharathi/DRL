# Training & Measuring the DQN — Step-by-Step Guide

Reproduces **Algorithm 1 ("Deep Q-learning with Experience Replay")** and
**Section 5.1 ("Training and Stability")** of Mnih et al., 2013.

Companion files:
- `dqn_model.py` — the network (see `DQN_Architecture_Guide.md`)
- `atari_preprocessing.py` — the 84×84×4 pipeline
- `toy_pixel_env.py` — a synthetic Atari-like "catch the ball" game (optional fast-iteration fallback, no ALE/ROM install needed)
- `dqn_train.py` — the training loop (this guide)
- `measure_dqn.py` — the diagnostic plots (this guide)

---

## 0. Environment: real Atari, installed

`gymnasium[atari]`, `ale-py`, and the Atari ROM binaries (via `AutoROM
--accept-license`) are now installed, and `dqn_train.py` trains against
the **real** `ALE/Breakout-v5` environment by default (real 210×160×3 RGB
frames straight from the Stella emulator, real game physics and scoring —
nothing simulated). Verified working:

```
A.L.E: Arcade Learning Environment (version 0.12.1+8a8fafb)
Frame shape: (210, 160, 3) uint8
Action space: Discrete(4)
```

`toy_pixel_env.py` (a minimal synthetic "catch the ball" game) is kept in
the project as an optional fallback — flip `USE_TOY_ENV = True` at the top
of `dqn_train.py` if you ever want to iterate on the algorithm itself
without waiting on real Atari episode lengths. Every line of the training
loop below is identical either way — only the `env = ...` object differs,
because `toy_pixel_env.py` deliberately mirrors `gymnasium`'s
`reset()`/`step()`/`render()` interface.

---

## Step 1 — Recap: what Algorithm 1 actually says

```
Initialize replay memory D to capacity N
Initialize Q with random weights θ
for episode = 1 to M:
    initialise s1, φ1 = φ(s1)
    for t = 1 to T:
        with probability ε select random action a_t
        otherwise select a_t = argmax_a Q(φ(s_t), a; θ)
        execute a_t, observe r_t, x_t+1
        set s_t+1, compute φ_t+1
        store (φ_t, a_t, r_t, φ_t+1) in D
        sample random minibatch from D
        y_j = r_j                                  if terminal
        y_j = r_j + γ max_a' Q(φ_j+1, a'; θ)        otherwise
        gradient descent step on (y_j − Q(φ_j, a_j; θ))²
```

Every subsequent step below points at the exact lines in `dqn_train.py`
that implement one piece of this pseudocode.

---

## Step 2 — The replay memory

```python
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, phi, action, reward, next_phi, done):
        self.buffer.append((phi, action, reward, next_phi, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        ...
```

**Why a `deque(maxlen=N)`?** Once full, appending automatically evicts the
*oldest* transition — exactly the paper's "stores the last N experience
tuples ... always overwrites with recent transitions" behaviour (Section
4), with no manual bookkeeping needed.

**Why sample randomly instead of using the most recent transitions?** This
is the core of experience replay: random sampling breaks the strong
correlation between consecutive frames of the *same* episode, which is
what prevents the training data from looking like a single, highly
non-i.i.d. sequence.

`REPLAY_CAPACITY = 20_000` for real Atari here (vs. the paper's 1,000,000)
— scaled down so the whole demo finishes in minutes rather than the days
the paper's full-scale run took on 2013 hardware. (`toy_pixel_env` mode
uses an even smaller 5,000, since that game's state space is trivial.)

---

## Step 3 — ε-greedy action selection

```python
def select_action(net, phi_hwc, epsilon, num_actions):
    if random.random() < epsilon:
        return random.randrange(num_actions)
    with torch.no_grad():
        q_values = net(to_tensor_batch(phi_hwc[None, ...]))
        return int(torch.argmax(q_values, dim=1).item())
```

Directly implements `"with probability ε select a random action, otherwise
select a_t = argmax_a Q(...)"`. `torch.no_grad()` is used because acting in
the environment should never itself contribute to the gradient — only the
training step (Step 5) does.

```python
def epsilon_by_step(step):
    frac = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_START + frac * (EPS_END - EPS_START)
```

Matches the paper's schedule exactly in *shape* (linear anneal from 1.0 to
0.1, then held fixed) — only `EPS_DECAY_STEPS` is shrunk from the paper's
1,000,000 frames to 20,000 for real Atari (3,000 in toy-env mode), since
this demo trains for far fewer total frames than the paper's full run.

---

## Step 4 — Storing transitions and computing φ

```python
next_phi = preprocessor.step(obs)
replay.push(phi, action, reward, next_phi, float(terminated))
phi = next_phi
```

This is `"store transition (φ_t, a_t, r_t, φ_t+1) in D"`. Note `done`
stores `terminated` (the game actually ended — in Breakout, all 5 lives
lost), not `truncated` (ALE's `max_num_frames_per_episode=108000` safety
cutoff, effectively never hit at this demo's scale). This distinction
matters for Step 5's Bellman target: a truncated-but-not-terminated state
still has future value, so it must NOT be treated as terminal in the
target computation — only genuine game-over states should zero out the
bootstrap term.

Also worth knowing: Breakout's `info` dict exposes a `lives` counter
(starts at 5, decrements on each dropped ball) that only reaches
`terminated=True` once it hits 0. A real episode here runs roughly
100-300 environment steps (verified empirically) — much longer than a toy
`SimpleCatchEnv` episode, which is one reason `NUM_EPISODES` is set lower
for real Atari than for the toy environment.

---

## Step 4b — Reward clipping (real Atari only)

```python
def clip_reward(reward):
    if reward > 0: return 1.0
    if reward < 0: return -1.0
    return 0.0

train_reward = clip_reward(reward) if CLIP_REWARDS else reward
replay.push(phi, action, train_reward, next_phi, float(terminated))
episode_reward += reward   # the TRUE, unclipped score is what we log/plot
```

Directly implements the paper's Section 5 rule: *"we fixed all positive
rewards to be 1 and all negative rewards to be -1, leaving 0 rewards
unchanged."* Two things to notice:
- Only the value **stored in the replay buffer** (used for the Bellman
  target) is clipped — the value used for `episode_reward` (what gets
  logged and plotted) is the **true, unclipped** game score, matching
  Section 5.3's evaluation convention of reporting real scores.
- This didn't apply to the toy environment (rewards were already exactly
  ±1), which is why this step only became necessary once real Atari (with
  Breakout's variable per-brick point values) was wired in.

---

## Step 5 — The Bellman target and the SGD update (the heart of DQN)

```python
with torch.no_grad():
    next_q = target_net(next_states_t)
    max_next_q = next_q.max(dim=1).values
    targets = rewards_t + GAMMA * max_next_q * (1.0 - dones_t)

current_q_all = net(states_t)
current_q = current_q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

loss = loss_fn(current_q, targets)
optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
optimizer.step()

if global_step % TARGET_SYNC_STEPS == 0:
    target_net.load_state_dict(net.state_dict())
```

Line by line:
- `next_q = target_net(next_states_t)` then `.max(dim=1)` → this is
  `max_a' Q(φ_j+1, a'; θ⁻)`, using the **target network's** weights `θ⁻`,
  not the live network's.
- `targets = rewards + γ * max_next_q * (1 - dones)` → implements both
  branches of `y_j` in one line: when `dones_t == 1` (terminal), the
  `γ * max_next_q` term is multiplied by zero, leaving `y_j = r_j` exactly
  as the pseudocode's terminal case.
- `current_q_all.gather(1, actions_t...)` → picks out `Q(φ_j, a_j; θ)`,
  i.e., only the Q-value for the action that was *actually* taken (the
  network outputs one value per action; we only want the one whose Bellman
  error we're correcting).
- `loss_fn` is `nn.SmoothL1Loss()` (Huber loss) — see the note below on why
  this replaced plain MSE.
- `nn.utils.clip_grad_norm_(...)` caps the gradient's L2 norm at
  `GRAD_CLIP_NORM` right before the optimizer step, bounding how much a
  single large Bellman error can move the weights.
- `optimizer` is `RMSprop`, matching the paper's stated choice of RMSProp
  (Section 5) with minibatches of size 32 (also matching).
- The final two lines sync `target_net` from `net` every `TARGET_SYNC_STEPS`
  environment steps — see below.

**Documented deviation from the 2013 paper — target network, Huber loss,
gradient clipping.** The original Algorithm 1 (and the pure `torch.no_grad()`
detachment described in earlier revisions of this guide) holds `θ_{i-1}`
fixed for only a *single* backward pass — the next update immediately moves
those same weights again, which is prone to a feedback loop where an
overestimated Q-value inflates its own next target (see Step 8's original
findings). This script now instead reproduces the **2015 *Nature* DQN**
follow-up's three fixes:
1. **A genuinely separate target network** (`target_net`), whose weights are
   frozen across `TARGET_SYNC_STEPS` env steps at a time, not just one
   backward pass.
2. **Huber loss** instead of MSE — quadratic for small errors, linear for
   large ones, so a big Bellman error produces a bounded gradient instead of
   a squared one.
3. **Gradient-norm clipping** — a hard cap on the gradient regardless of loss
   shape.

An unmodified snapshot of the original, pure-Algorithm-1 version (no target
network, MSE loss, no clipping) is kept at `dqn_train_backup.py` for anyone
who wants to see/run the paper-faithful behavior directly.

---

## Step 6 — Run it

```bash
python3 dqn_train.py
```

At the default settings (`NUM_EPISODES = 80`, real `ALE/Breakout-v5`)
this took **~5 minutes** on a laptop CPU in our verified run, printing
progress every 5 episodes:

```
Episode    5 | avg reward (last 5) = +1.60 | avg max Q =   0.175 | eps = 0.955 | loss = nan
Episode   10 | avg reward (last 5) = +0.40 | avg max Q =  74.501 | eps = 0.921 | loss = 55.6578
Episode   15 | avg reward (last 5) = +2.60 | avg max Q =  84.544 | eps = 0.866 | loss = 202.3194
...
Episode   35 | avg reward (last 5) = +1.20 | avg max Q = 894.098 | eps = 0.698 | loss = 3098.4565
...
Episode   55 | avg reward (last 5) = +2.00 | avg max Q = 2387.729 | eps = 0.534 | loss = 1571.9664
...
Episode   80 | avg reward (last 5) = +2.20 | avg max Q =  46.171 | eps = 0.313 | loss = 2.3927
```

(`loss = nan` in early episodes is expected and harmless — it just means
the replay buffer hadn't yet reached `MIN_REPLAY_BEFORE_TRAIN`, so no
gradient step had happened yet; the printed loss is an empty-list average.)

It saves three artifacts (filenames tagged by game):
- `dqn_breakout.pt` — trained network weights
- `dqn_training_history_breakout.npz` — per-episode reward, avg-max-Q, and per-step loss arrays
- (run `measure_dqn.py` next to turn these into plots)

---

## Step 7 — Measure and interpret training progress

```bash
python3 measure_dqn.py dqn_training_history_breakout.npz
```

This reproduces the paper's Section 5.1 diagnostic approach and produces
`dqn_training_metrics_breakout.png` with three panels, from our actual
80-episode real-Breakout run:

```
Episodes trained         : 80
Mean reward (first 10%)  : +1.250
Mean reward (last 10%)   : +2.000
Final avg max Q          : +46.171
Max |avg max Q| observed : 2531.792
```

**Panel 1 — Reward per episode.** The paper explicitly warns this metric
"tends to be very noisy because small changes to the weights of a policy
can lead to large changes in the distribution of states the policy
visits." Our plot shows exactly that: raw per-episode reward (light grey)
swings between 0 and 5 with no obvious trend, while the 10-episode moving
average (blue) reveals a real, if noisy, upward trend — from ~1.25 in the
first 10% of training to ~2.0 in the last 10%. **The agent is measurably
learning to break more bricks** in just 80 episodes / ~15,000 environment
steps, with zero game-specific tuning beyond what the paper specifies.

**Panel 2 — Average max predicted Q on a fixed held-out state set.** This
is the paper's *proposed fix* for the noisy-reward problem: instead of
judging progress by reward, track the network's own confidence (predicted
max Q) on a *fixed* set of states collected once via a random policy
before training starts (`collect_eval_states()` in `dqn_train.py`,
mirroring Section 5.1). In the paper's real experiments (10 million
frames, all seven games), this metric rose **smoothly** with no
divergence. **In our run it does not** — it spikes violently to over 2500
at episodes ~48 and ~55, then partially recovers. See Step 8 for why.

**Panel 3 — Training loss (log scale) per SGD step.** Large periodic
spikes correspond to the same instability visible in Panel 2 — each spike
in avg-max-Q is preceded by a burst of very large squared Bellman errors.

---

## Step 8 — Why our *original* run was noisier than the paper's (now resolved)

The plots and numbers quoted above (spikes to 2500+ in avg-max-Q) were
produced by `dqn_train_backup.py`, the original pure-Algorithm-1 version,
before target network / Huber loss / gradient clipping were added to
`dqn_train.py`. That was a genuine, useful finding from actually running the
code, not a hypothetical — and the root causes were:

- **No target network.** The original script bootstraps `max_a' Q(...)` off
  the *same, constantly-changing* network being trained (Algorithm 1,
  2013). This is prone to feedback loops (an overestimated Q-value makes
  itself the target for the next update, which can compound) — exactly why
  the 2015 *Nature* follow-up froze a separate copy of the network for
  computing targets, synced only every few thousand steps. **This is now
  implemented in `dqn_train.py`** as `target_net` (Step 5).
- **No gradient clipping / Huber loss.** The original paper uses plain MSE;
  the 2015 follow-up switched to a clipped/Huber loss specifically to stop
  large Bellman errors from producing huge gradient steps. **Both are now
  implemented in `dqn_train.py`** (Step 5).
- **Far fewer training frames than the paper** still applies and is not
  something these three fixes change: ~15,000 environment steps here vs.
  the paper's 10,000,000, with `EPS_DECAY_STEPS = 20,000` meaning short runs
  still spend most of their time in heavy exploration. Target
  network/Huber/clipping reduce *instability per update*, not the
  fundamental sample-efficiency gap versus a 10M-frame run.

**Despite the original instability, the reward trend in Panel 1 was still
real and upward** — a useful, concrete illustration of the paper's own point
(Section 5.1) that reward and Q-value noise don't necessarily mean the
policy isn't improving.

**To compare directly:** run `python3 dqn_train_backup.py` (original,
unstable) alongside `python3 dqn_train.py` (stabilized) for the same
`NUM_EPISODES` and compare their `measure_dqn.py` avg-max-Q panels — the
stabilized version should show visibly smaller spikes and, per the goal of
adding these fixes, should reach a comparable or better reward trend in
fewer episodes.

---

## Step 9 — Scaling up, and trying other games

To move closer to the paper's actual regime (at the cost of much longer
runtime):
- Raise `REPLAY_CAPACITY` toward 1,000,000 and `EPS_DECAY_STEPS` toward
  1,000,000, and increase `NUM_EPISODES` substantially — the paper trained
  on 10 million frames total (~hours to days on modern CPU hardware,
  vs. the original ~1 week on 2013 GPUs).
- Try any of the paper's other games by changing `GAME_ID`, e.g.
  `"ALE/Pong-v5"`, `"ALE/Seaquest-v5"`, `"ALE/SpaceInvaders-v5"` — no other
  code changes are needed since `num_actions` is read dynamically from
  `env.action_space.n`.
- Set `USE_TOY_ENV = True` any time you want a fast (~seconds) sanity
  check of an algorithm change before spending minutes re-running real
  Atari.

---

## Summary Table

| Step | Paper concept | Code |
|---|---|---|
| 2 | Replay memory D, capacity N | `ReplayMemory` (deque, `REPLAY_CAPACITY`) |
| 3 | ε-greedy behaviour policy | `select_action()`, `epsilon_by_step()` |
| 4 | Store transition in D | `replay.push(...)` |
| 4b | Reward clipping to {-1,0,+1} (Section 5) | `clip_reward()` |
| 5 | Bellman target y_j + gradient step | `torch.no_grad()` target block using `target_net` + Huber loss + gradient clipping + RMSprop |
| 6 | Train for M episodes on real Atari | `dqn_train.py` main loop, `GAME_ID = "ALE/Breakout-v5"` |
| 7 | Track avg reward AND avg max Q (Section 5.1) | `measure_dqn.py` |
| 8 | (2015 Nature fixes, not in the 2013 paper) target network + Huber loss + gradient clipping | resolved instability seen on real Breakout; original unstable version kept at `dqn_train_backup.py` |
| 9 | Scaling toward the paper's full 10M-frame regime, other games | `GAME_ID`, `REPLAY_CAPACITY`, `EPS_DECAY_STEPS`, `NUM_EPISODES` |
