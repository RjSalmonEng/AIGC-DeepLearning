"""Starter code for Lab 09: Training an RL Agent.

Run this file once, unmodified, to confirm your environment works. It plays one random episode,
prints the average return of a random policy over 50 episodes, and saves a picture. It takes a few
seconds. You write everything about the agents: the DQN update, the replay buffer, the target network,
the ablations, the policy-gradient agent, the checks and every experiment. The starter contains none
of them.

What is provided
    CartPole              the control task, in plain numpy, no downloads. Four numbers describe the state:
                          cart position, cart velocity, pole angle, pole angular velocity. Two actions:
                          0 pushes left, 1 pushes right. Reward is +1 for every step the pole stays up.
                          step() returns (next_state, reward, terminated, truncated). terminated means the
                          pole fell or the cart left the track. truncated means the 200-step time limit
                          was reached. These are different events and the difference matters.
    QNet                  a plain network that maps a state to one value per action. No training code.
    mlp                   a plain tanh network, for a policy (2 outputs) or a state-value function (1 output)
    probe_states()        a fixed set of 500 states, collected by a random policy, for measuring how large
                          your Q-values are
    smooth_curve()        turns a list of (step, episode return) into a learning curve on a fixed grid
    final_return()        mean return of the episodes that ended in the last 2,000 steps
    steps_to_reach()      first grid step at which the smoothed return reaches a level
    plot_curves()         one panel of learning curves, every seed drawn, with the mean in bold
    set_seed()            seeds numpy and torch together

What you write
    the replay buffer, the epsilon schedule, the DQN loss and update, the target network, the switches that
    turn each of them off, REINFORCE or PPO, and every check and experiment.

Requirements: torch, numpy, matplotlib. No downloads. A laptop CPU is enough.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUDGET = 20000        # environment steps given to every agent in the lab
GRID = 500            # learning curves are read every 500 steps
WINDOW = 2000         # and averaged over the episodes that ended in the last 2,000 steps


class CartPole:
    """Cart and pole. An episode ends when the pole tilts past 12 degrees, the cart leaves the track
    (terminated), or max_steps is reached (truncated)."""
    n_actions = 2
    obs_dim = 4

    def __init__(self, seed=0, max_steps=200):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.g, self.mc, self.mp, self.l, self.force, self.tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
        self.theta_lim = 12 * 2 * math.pi / 360
        self.x_lim = 2.4

    def reset(self):
        self.s = self.rng.uniform(-0.05, 0.05, size=4)
        self.t = 0
        return self.s.astype(np.float32)

    def step(self, a):
        x, xd, th, thd = self.s
        f = self.force if a == 1 else -self.force
        ct, st = math.cos(th), math.sin(th)
        tm = self.mc + self.mp
        pml = self.mp * self.l
        tmp = (f + pml * thd ** 2 * st) / tm
        tha = (self.g * st - ct * tmp) / (self.l * (4.0 / 3.0 - self.mp * ct ** 2 / tm))
        xa = tmp - pml * tha * ct / tm
        x, xd = x + self.tau * xd, xd + self.tau * xa
        th, thd = th + self.tau * thd, thd + self.tau * tha
        self.s = np.array([x, xd, th, thd])
        self.t += 1
        terminated = bool(abs(x) > self.x_lim or abs(th) > self.theta_lim)
        truncated = self.t >= self.max_steps
        return self.s.astype(np.float32), 1.0, terminated, truncated


# ---------------------------------------------------------------- networks (no training code)
class QNet(nn.Module):
    """state -> one value per action."""
    def __init__(self, obs_dim=4, n_actions=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


def mlp(i, o, hidden=64):
    """A tanh network. mlp(4, 2) is a policy that outputs two logits; mlp(4, 1) is a state-value function."""
    return nn.Sequential(nn.Linear(i, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, o))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def probe_states(n=500, seed=123):
    """A fixed set of states from a random policy, the same for every student and every run."""
    env = CartPole(seed)
    rng = np.random.default_rng(seed)
    out, s = [], env.reset()
    while len(out) < n:
        out.append(s)
        s, _, term, trunc = env.step(int(rng.integers(2)))
        if term or trunc:
            s = env.reset()
    return torch.tensor(np.array(out))


# ---------------------------------------------------------------- measuring learning curves
def smooth_curve(episodes, steps=BUDGET, grid=GRID, window=WINDOW):
    """episodes: list of (environment step at which the episode ended, episode return).
    Returns the mean return of the episodes that ended in the last `window` steps, read every `grid` steps.
    A grid point with no finished episode is nan."""
    ts = np.array([e[0] for e in episodes])
    rs = np.array([e[1] for e in episodes])
    out = []
    for g in range(grid, steps + 1, grid):
        m = (ts > g - window) & (ts <= g)
        out.append(rs[m].mean() if m.any() else np.nan)
    return np.array(out)


def final_return(episodes, steps=BUDGET, window=WINDOW):
    """Mean return of the episodes that ended in the last `window` steps of the budget."""
    return float(smooth_curve(episodes, steps, grid=steps, window=window)[-1])


def steps_to_reach(curve, level, grid=GRID):
    """First grid step at which a smoothed curve is at or above `level`, or None if it never gets there."""
    hit = np.where(np.nan_to_num(curve, nan=-1) >= level)[0]
    return int((hit[0] + 1) * grid) if len(hit) else None


def plot_curves(ax, curves, title, color="#1C3D6E", steps=BUDGET, grid=GRID):
    """curves: dict seed -> smoothed curve. Every seed is drawn thin, the mean is bold."""
    xs = np.arange(grid, steps + 1, grid)
    for c in curves.values():
        ax.plot(xs, c, color=color, alpha=0.35, lw=1)
    ax.plot(xs, np.nanmean(np.array(list(curves.values())), axis=0), color=color, lw=2.2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_ylim(0, 205)
    ax.grid(alpha=0.25)


if __name__ == "__main__":
    env = CartPole(0)
    rng = np.random.default_rng(0)
    rets = []
    for _ in range(50):
        s, total = env.reset(), 0.0
        while True:
            s, r, term, trunc = env.step(int(rng.integers(2)))
            total += r
            if term or trunc:
                break
        rets.append(total)
    print(f"random policy: mean return {np.mean(rets):.1f} over 50 episodes (best possible is 200)")
    print("probe states:", tuple(probe_states().shape))
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(rets, bins=15, color="#1C3D6E")
    ax.set_title("random policy returns")
    fig.tight_layout()
    fig.savefig("starter_check.png", dpi=120)
    print("saved starter_check.png")
