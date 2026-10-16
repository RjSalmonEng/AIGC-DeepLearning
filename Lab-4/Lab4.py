# Lab 04: What Recurrence Can and Cannot Remember
# Script version of Lab4.ipynb. Run from the Lab-4 folder: python Lab4.py

import os
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from lab04_starter import RecallNet, make_recall_batch, train_recall

SEEDS = [0, 1, 2]
os.makedirs("figures", exist_ok=True)


accuracy, _ = train_recall("rnn", seq_len=10, seed=0)
print(f"plain RNN, 10 steps: {accuracy:.3f}")


def print_gate_blocks(kind, gate_bias):
    rnn = RecallNet(kind, gate_bias=gate_bias).rnn
    H = rnn.hidden_size
    total = (rnn.bias_ih_l0 + rnn.bias_hh_l0).detach()
    names = {"lstm": ["input", "forget", "cell", "output"], "gru": ["reset", "update", "new"]}[kind]
    for i, name in enumerate(names):
        block = total[i * H:(i + 1) * H]
        print(f"{kind} {name:>6} [{i * H}:{(i + 1) * H}]  min {block.min():.3f}  max {block.max():.3f}")


torch.manual_seed(0)
print_gate_blocks("lstm", 3.0)
print()
print_gate_blocks("gru", 3.0)


CONFIGS = {
    "RNN": ("rnn", None),
    "LSTM default": ("lstm", None),
    "LSTM forget bias 3": ("lstm", 3.0),
    "GRU default": ("gru", None),
    "GRU update bias 3": ("gru", 3.0),
}


def run_seeds(kind, seq_len, gate_bias=None, clip=1.0):
    return [train_recall(kind, seq_len, seed=s, gate_bias=gate_bias, clip=clip)[0] for s in SEEDS]


def print_row(name, accs):
    seeds = "  ".join(f"{a:.3f}" for a in accs)
    print(f"{name:<20} {seeds}   mean {sum(accs) / len(accs):.3f}")


benchmark = {}
for seq_len in [10, 30, 60]:
    print(f"{'length ' + str(seq_len):<20} seed0  seed1  seed2")
    for name, (kind, bias) in CONFIGS.items():
        benchmark[(name, seq_len)] = run_seeds(kind, seq_len, bias)
        print_row(name, benchmark[(name, seq_len)])
    print()


def grad_through_time(kind, forget_bias=None, seq_len=60, seeds=range(5)):
    norms = []
    for seed in seeds:
        torch.manual_seed(seed)
        rnn = {"rnn": nn.RNN, "lstm": nn.LSTM}[kind](8, 32, batch_first=True)
        if forget_bias is not None:
            with torch.no_grad():
                rnn.bias_ih_l0[32:64].fill_(forget_bias / 2)
                rnn.bias_hh_l0[32:64].fill_(forget_bias / 2)
        x = torch.randn(16, seq_len, 8, requires_grad=True)
        out, _ = rnn(x)
        out[:, -1].pow(2).sum().backward()
        norms.append(x.grad.norm(dim=(0, 2)))
    # flip so index 0 is the last step (0 steps back from the loss)
    return torch.stack(norms).median(dim=0).values.flip(0)


grads = {
    "RNN": grad_through_time("rnn"),
    "LSTM default": grad_through_time("lstm"),
    "LSTM forget bias 3": grad_through_time("lstm", forget_bias=3.0),
}

print(f"{'steps back':<20}" + "".join(f"{k:>10}" for k in [0, 9, 29, 59]))
for name, g in grads.items():
    print(f"{name:<20}" + "".join(f"{g[k].item():>10.1e}" for k in [0, 9, 29, 59]))

plt.figure(figsize=(8, 4.5))
for name, g in grads.items():
    plt.semilogy(g, label=name)
plt.xlabel("Steps back from the loss")
plt.ylabel("Gradient norm (median of 5 seeds)")
plt.title("Gradient reaching each input step of a 60-step sequence")
plt.legend()
plt.tight_layout()
plt.savefig("figures/gradient_through_time.png", dpi=150)
plt.show()


sweep = {}
print(f"{'forget bias':<20} seed0  seed1  seed2")
for bias in [0.0, 1.0, 2.0, 3.0]:
    sweep[bias] = run_seeds("lstm", 30, gate_bias=bias)
    print_row(f"bias {bias:.0f}", sweep[bias])


clipping = {
    "clip on (1.0)": run_seeds("rnn", 30, clip=1.0),
    "clip off": run_seeds("rnn", 30, clip=None),
}
print(f"{'':<20} seed0  seed1  seed2")
for name, accs in clipping.items():
    print_row(name, accs)


def grad_norms_during_training(seq_len=30, seed=0, steps=600):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 100)
    model = RecallNet("rnn")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    norms = []
    for _ in range(steps):
        x, y = make_recall_batch(64, seq_len, generator)
        loss = nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        # max_norm=inf only measures the norm, it never rescales
        norms.append(nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item())
        optimizer.step()
    return torch.tensor(norms)


for seed in SEEDS:
    norms = grad_norms_during_training(seed=seed)
    print(f"seed {seed}: median {norms.median():.3f}  max {norms.max():.3f}  "
          f"updates above 1.0: {(norms > 1.0).sum().item()} of {len(norms)}")


lstm = nn.LSTM(64, 128, batch_first=True)
timing = {}
for T in [50, 100, 200, 400]:
    x = torch.randn(32, T, 64)
    best = float("inf")
    for _ in range(3):
        start = time.time()
        out, _ = lstm(x)
        out.sum().backward()
        best = min(best, time.time() - start)
    timing[T] = best
    print(f"T = {T:>3}: {best:.4f}s   {best / T * 1000:.3f} ms per step")


long_run = run_seeds("lstm", 200, gate_bias=3.0)
print(f"{'':<20} seed0  seed1  seed2")
print_row("LSTM bias 3, T=200", long_run)
