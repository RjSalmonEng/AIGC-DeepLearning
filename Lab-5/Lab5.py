# Lab 05: Attention Against Recurrence
# Script version of Lab5.ipynb. Run from the Lab-5 folder: python Lab5.py


import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from lab05_starter import (AttentionReadout, OrderClassifier, make_order_batch, make_recall_batch,
                           train_order, train_recall)

SEEDS = [0, 1, 2]
os.makedirs("figures", exist_ok=True)


accuracy, _ = train_recall("attention", seq_len=30, seed=0)
print(f"attention readout, 30 steps: {accuracy:.3f}")
accuracies = train_order("sinusoidal", seed=0, steps=300, test_lens=(16,))
print(f"order task, 2 blocks, 300 updates: {accuracies[0]:.3f}")


def causal_mask(length):
    # row i (the query) may look at columns 0..i (the keys)
    return torch.tril(torch.ones(length, length, dtype=torch.bool))


def scaled_dot_product_attention(q, k, v, mask=None):
    """softmax(q k^T / sqrt(d_k)) v.
    q: (..., T_q, d_k)   k: (..., T_k, d_k)   v: (..., T_k, d_v)
    mask: boolean, broadcastable to (..., T_q, T_k); False entries are blocked.
    Returns (output, weights)."""
    d_k = k.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)          # (..., T_q, T_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))       # before the softmax
    weights = torch.softmax(scores, dim=-1)                     # over the keys
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "a row of weights doesn't sum to 1"
    if mask is not None:
        assert (weights.masked_select(~mask.expand_as(weights)) == 0).all(), "a masked position got weight"
    return weights @ v, weights


torch.manual_seed(0)
q, k, v = torch.randn(2, 4, 5, 8), torch.randn(2, 4, 7, 8), torch.randn(2, 4, 7, 8)
mine, _ = scaled_dot_product_attention(q, k, v)
ref = F.scaled_dot_product_attention(q, k, v)
print(f"no mask, 5 queries 7 keys: match {torch.allclose(mine, ref, atol=1e-5)}  max diff {(mine - ref).abs().max():.1e}")

s = torch.randn(2, 4, 6, 8)
mine, weights = scaled_dot_product_attention(s, s, s, mask=causal_mask(6))
ref = F.scaled_dot_product_attention(s, s, s, is_causal=True)
print(f"causal mask, 6 steps:      match {torch.allclose(mine, ref, atol=1e-5)}  max diff {(mine - ref).abs().max():.1e}")
print("largest weight above the diagonal:", weights[..., ~causal_mask(6)].abs().max().item())
print(weights[0, 0].numpy().round(2))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_head = d_model, n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)  # (batch, heads, T, d_head)

    def forward(self, x, context=None, mask=None):
        context = x if context is None else context
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(context))
        v = self.split_heads(self.w_v(context))
        out, self.last_weights = scaled_dot_product_attention(q, k, v, mask=mask)
        batch, _, length, _ = out.shape
        out = out.transpose(1, 2).reshape(batch, length, self.d_model)  # concatenate the heads
        return self.w_o(out)


def copy_from_reference(mine, reference):
    d = mine.d_model
    W, b = reference.in_proj_weight, reference.in_proj_bias  # stacked as query, key, value
    with torch.no_grad():
        for lin, i in ((mine.w_q, 0), (mine.w_k, 1), (mine.w_v, 2)):
            lin.weight.copy_(W[i * d:(i + 1) * d])
            lin.bias.copy_(b[i * d:(i + 1) * d])
        mine.w_o.weight.copy_(reference.out_proj.weight)
        mine.w_o.bias.copy_(reference.out_proj.bias)


mha = MultiHeadAttention(32, 4)
reference = nn.MultiheadAttention(32, 4, batch_first=True)
copy_from_reference(mha, reference)

x = torch.randn(3, 6, 32)
context = torch.randn(3, 9, 32)
cases = {
    "self-attention": (mha(x), reference(x, x, x)[0]),
    # PyTorch's attn_mask is True where blocked, the opposite of mine
    "causal self-attention": (mha(x, mask=causal_mask(6)), reference(x, x, x, attn_mask=~causal_mask(6))[0]),
    "cross-attention, 6 to 9": (mha(x, context=context), reference(x, context, context)[0]),
}
for name, (a, b) in cases.items():
    print(f"{name:<24} match {torch.allclose(a, b, atol=1e-5)}  max diff {(a - b).abs().max():.1e}")


def softmax_wrong_axis(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-2) @ v


def mask_after_softmax(q, k, v, mask=None):
    weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1]), dim=-1)
    return (weights * mask if mask is not None else weights) @ v


class WrongSplit(MultiHeadAttention):
    def split_heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, self.n_heads, length, self.d_head)  # same shape, wrong order


ref = F.scaled_dot_product_attention(s, s, s, is_causal=True)
print("softmax over the queries:", torch.allclose(softmax_wrong_axis(s, s, s, causal_mask(6)), ref, atol=1e-5))
print("mask after the softmax:  ", torch.allclose(mask_after_softmax(s, s, s, causal_mask(6)), ref, atol=1e-5))
wrong = WrongSplit(32, 4)
copy_from_reference(wrong, reference)
print("heads split in the wrong order:", torch.allclose(wrong(x), reference(x, x, x)[0], atol=1e-5))


count = lambda m: sum(p.numel() for p in m.parameters())
for h in (1, 2, 4, 8):
    print(f"{h} heads: {count(MultiHeadAttention(32, h))} parameters")


perm = torch.randperm(6)
print("permutation:", perm.tolist())
print("layer(shuffled x) == shuffled layer(x):", torch.allclose(mha(x[:, perm]), mha(x)[:, perm], atol=1e-6))
print("layer(shuffled x) == layer(x):         ", torch.allclose(mha(x[:, perm]), mha(x), atol=1e-6))


def print_row(name, accs):
    seeds = "  ".join(f"{a:.3f}" for a in accs)
    print(f"{name:<22} {seeds}   mean {sum(accs) / len(accs):.3f}")


recall = {}
for seq_len in [30, 100, 400]:
    print(f"{'length ' + str(seq_len):<22} seed0  seed1  seed2")
    for kind in ["attention", "lstm"]:
        recall[(kind, seq_len)] = [train_recall(kind, seq_len, seed=s, attention_fn=scaled_dot_product_attention)
                                   for s in SEEDS]
        print_row(kind, [acc for acc, _ in recall[(kind, seq_len)]])
    print_row("weight on step 0", [w for _, w in recall[("attention", seq_len)]])
    print()


def train_readout(seq_len, seed, steps=600):
    # same loop as train_recall in the starter, but returns the model
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 100)
    model = AttentionReadout(scaled_dot_product_attention)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        x, y = make_recall_batch(64, seq_len, generator)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


readout = train_readout(100, seed=0)
x_val, y_val = make_recall_batch(1000, 100, torch.Generator().manual_seed(999))
rows = torch.arange(1000)
with torch.no_grad():
    acc = (readout(x_val).argmax(1) == y_val).float().mean().item()
    w = readout.last_weights[:, 0]
    print(f"accuracy {acc:.3f}  weight on step 0 {w[:, 0].mean():.3f}  "
          f"largest weight on any other step {w[:, 1:].max():.4f}")

    where = torch.randint(0, 100, (1000,), generator=torch.Generator().manual_seed(5))
    moved = x_val.clone()
    moved[rows, 0], moved[rows, where] = x_val[rows, where], x_val[rows, 0]
    acc = (readout(moved).argmax(1) == y_val).float().mean().item()
    print(f"label moved to a random step: accuracy {acc:.3f}  "
          f"weight on its new step {readout.last_weights[rows, 0, where].mean():.3f}")


lengths = [30, 100, 400]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for kind, marker in [("attention", "o"), ("lstm", "s")]:
    for i, seed in enumerate(SEEDS):
        accs = [recall[(kind, T)][i][0] for T in lengths]
        axes[0].plot(lengths, accs, marker=marker, alpha=0.7,
                     color="C0" if kind == "attention" else "C1", label=kind if i == 0 else None)
axes[0].axhline(0.25, color="gray", linestyle="--", label="chance")
axes[0].set_xscale("log")
axes[0].set_xticks(lengths, [str(T) for T in lengths])
axes[0].set_xlabel("Sequence length")
axes[0].set_ylabel("Validation accuracy")
axes[0].set_title("Recall accuracy, every seed")
axes[0].legend()

for i, seed in enumerate(SEEDS):
    axes[1].plot(lengths, [recall[("attention", T)][i][1] for T in lengths], marker="o", label=f"seed {seed}")
axes[1].plot(lengths, [1 / T for T in lengths], color="gray", linestyle="--", label="uniform (1/T)")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xticks(lengths, [str(T) for T in lengths])
axes[1].set_xlabel("Sequence length")
axes[1].set_ylabel("Mean weight on step 0")
axes[1].set_title("Attention weight on the label step")
axes[1].legend()
plt.tight_layout()
plt.savefig("figures/recall_vs_length.png", dpi=150)
plt.show()


print(f"{'':<22} seed0       seed1       seed2")
print(f"{'':<22} 16    32    16    32    16    32")
positions = {}
for pos in ["none", "learned", "sinusoidal"]:
    positions[pos] = [train_order(pos, s, test_lens=(16, 32), attention_cls=MultiHeadAttention) for s in SEEDS]
    print(f"{pos:<22} " + "  ".join(f"{a:.3f} {b:.3f}" for a, b in positions[pos]))


print(f"{'':<12} {'seed0':^17}  {'seed1':^17}  {'seed2':^17}")
print(f"{'':<12} " + "  ".join(["16    32    48"] * 3))
mixed = {}
for pos in ["none", "learned", "sinusoidal"]:
    mixed[pos] = [train_order(pos, s, mixed_lens=(8, 16, 24), test_lens=(16, 32, 48), attention_cls=MultiHeadAttention)
                  for s in SEEDS]
    print(f"{pos:<12} " + "  ".join(" ".join(f"{a:.3f}" for a in accs) for accs in mixed[pos]))


def train_order_model(pos, seed, mixed_lens, steps=800, lr=2e-3):
    # same loop as train_order in the starter, but returns the model
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 100)
    model = OrderClassifier(pos, attention_cls=MultiHeadAttention)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        length = mixed_lens[torch.randint(len(mixed_lens), (1,), generator=generator).item()]
        x, y = make_order_batch(64, length, generator)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


x32, y32 = make_order_batch(1000, 32, torch.Generator().manual_seed(999))
late = ((x32 == 1) | (x32 == 2))[:, 24:].any(dim=1)
print(f"sequences with a marker at step 24 or later: {late.float().mean():.3f}")
for pos in ["learned", "sinusoidal"]:
    model = train_order_model(pos, seed=0, mixed_lens=(8, 16, 24))
    with torch.no_grad():
        correct = model(x32).argmax(1) == y32
    print(f"{pos:<11} all {correct.float().mean():.3f}   both markers before 24 {correct[~late].float().mean():.3f}"
          f"   a marker at 24+ {correct[late].float().mean():.3f}")


STACKS = {
    "post-norm, no warmup": ("post", 0),
    "post-norm, warmup 100": ("post", 100),
    "pre-norm, no warmup": ("pre", 0),
}
stacks = {}
print(f"{'':<22} seed0  seed1  seed2")
for name, (norm, warmup) in STACKS.items():
    stacks[name] = [train_order("sinusoidal", s, depth=12, norm=norm, warmup=warmup, steps=400, lr=3e-3,
                                test_lens=(16,), attention_cls=MultiHeadAttention)[0] for s in SEEDS]
    print_row(name, stacks[name])


_, y16 = make_order_batch(1000, 16, torch.Generator().manual_seed(999))
print(f"share of label 1 in the length-16 validation set: {y16.float().mean():.3f}")


torch.manual_seed(0)
lstm = nn.LSTM(64, 64, batch_first=True)
attention = MultiHeadAttention(64, 4)
layers = {"lstm": lambda x: lstm(x)[0], "attention": attention}

timing = {}
print(f"{'length':<8}{'lstm':>10}{'attention':>12}{'ratio':>8}{'weights MB':>12}")
for T in [100, 200, 400, 800]:
    x = torch.randn(32, T, 64)
    for name, layer in layers.items():
        best = float("inf")
        for _ in range(3):
            start = time.time()
            layer(x).sum().backward()
            best = min(best, time.time() - start)
        timing[(name, T)] = best
    weights_mb = attention.last_weights.numel() * 4 / 1e6  # (batch, heads, T, T) in float32
    print(f"{T:<8}{timing[('lstm', T)]:>9.4f}s{timing[('attention', T)]:>11.4f}s"
          f"{timing[('attention', T)] / timing[('lstm', T)]:>8.1f}{weights_mb:>12.1f}")


lengths = [100, 200, 400, 800]
plt.figure(figsize=(7, 4.5))
for name in ["lstm", "attention"]:
    plt.loglog(lengths, [timing[(name, T)] for T in lengths], marker="o", label=name)
plt.xticks(lengths, [str(T) for T in lengths])
plt.xlabel("Sequence length")
plt.ylabel("Forward + backward time (s, best of 3)")
plt.title("One LSTM layer against one attention layer, width 64, batch 32")
plt.legend()
plt.tight_layout()
plt.savefig("figures/timing_vs_length.png", dpi=150)
plt.show()
