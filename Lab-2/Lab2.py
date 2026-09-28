# Lab 02: Making Training Work
# Script version of Lab2.ipynb. Run from the Lab-2 folder: python Lab2.py

import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

# ---- Copied unchanged from lab02_starter.py ----

class BrokenNet(nn.Module):
    def __init__(self, depth=8, width=32, in_features=20):
        super().__init__()
        layers = []
        n_in = in_features
        for _ in range(depth):
            layers.append(nn.Linear(n_in, width))
            n_in = width
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(width, 1)

        for layer in self.hidden:
            nn.init.normal_(layer.weight, mean=0.0, std=0.3)
            nn.init.constant_(layer.bias, -2.0)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.out(x)


def make_toy_classification(n=256, in_features=20, seed=0):
    """A small synthetic binary classification task: a fixed random
    linear boundary in a 20-dimensional input space. Simple enough that
    a correctly trained network of this size should solve it easily.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_features, generator=g)
    true_w = torch.randn(in_features, 1, generator=g)
    logits = x @ true_w
    y = (logits > 0).float().squeeze(-1)
    return x, y

# ---- The starter's __main__ block, unchanged ----
torch.manual_seed(0)
model = BrokenNet()
x, y = make_toy_classification(seed=0)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("Training BrokenNet for 50 epochs with plain SGD...")
for epoch in range(50):
    optimizer.zero_grad()
    preds = model(x).squeeze(-1)
    loss = nn.functional.binary_cross_entropy_with_logits(preds, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0 or epoch == 49:
        print(f"epoch {epoch:3d}   loss = {loss.item():.4f}")

print("\nFor reference, a model that has learned nothing at all scores")
print("about 0.693 on this loss (ln 2) — check whether the number above")
print("actually moved away from that.")


def per_layer_grad_norms(model):
    return [layer.weight.grad.abs().mean().item() for layer in model.hidden]


def gradient_check(model):
    model.zero_grad()
    preds = model(x).squeeze(-1)
    loss = nn.functional.binary_cross_entropy_with_logits(preds, y)
    loss.backward()
    return per_layer_grad_norms(model)


torch.manual_seed(0)
model = BrokenNet()
grads = gradient_check(model)
for i, g in enumerate(grads):
    print(f"layer {i}: mean |gradient| = {g:.2e}")



dead = []
h = x
with torch.no_grad():
    for i, layer in enumerate(model.hidden):
        h = torch.relu(layer(h))
        dead.append((h == 0).float().mean().item() * 100)
        print(f"layer {i}: {dead[-1]:.1f}% of ReLU outputs are zero")

plt.figure(figsize=(7, 4))
plt.bar(range(8), dead)
plt.xlabel("Layer (0 = closest to input)")
plt.ylabel("% of ReLU outputs that are zero")
plt.title("BrokenNet: dead ReLUs per layer")
plt.tight_layout()
plt.savefig("figures/stage0_dead_relus.png", dpi=150)
plt.show()



results = {}  # stage name -> list of losses
grad_results = {}  # stage name -> per-layer gradient before training


def train(model, optimizer, epochs=50):
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(x).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(preds, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def plot_losses(names, title, filename):
    plt.figure(figsize=(7, 4.5))
    for name in names:
        plt.plot(results[name], label=name)
    plt.axhline(0.693, color="gray", linestyle="--", label="0.693 (guessing)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{filename}", dpi=150)
    plt.show()


torch.manual_seed(0)
model = BrokenNet()
grad_results["Stage 0: baseline"] = gradient_check(model)
results["Stage 0: baseline"] = train(model, torch.optim.SGD(model.parameters(), lr=0.1))
print(f"final loss: {results['Stage 0: baseline'][-1]:.4f}")
plot_losses(["Stage 0: baseline"], "Stage 0: baseline", "stage0_baseline.png")



torch.manual_seed(0)
model = BrokenNet()
grad_results["Stage 1: Adam"] = gradient_check(model)
results["Stage 1: Adam"] = train(model, torch.optim.Adam(model.parameters(), lr=0.01))
print(f"final loss: {results['Stage 1: Adam'][-1]:.4f}")
plot_losses(["Stage 0: baseline", "Stage 1: Adam"], "Stage 1: Adam instead of SGD", "stage1_adam.png")



torch.manual_seed(0)
model = BrokenNet()
for layer in model.hidden:
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

grad_results["Stage 2: He weights"] = gradient_check(model)
results["Stage 2: He weights"] = train(model, torch.optim.SGD(model.parameters(), lr=0.1))
print(f"final loss: {results['Stage 2: He weights'][-1]:.4f}")
print(f"layer 0 gradient: {grad_results['Stage 2: He weights'][0]:.2e}")
plot_losses(["Stage 0: baseline", "Stage 2: He weights"], "Stage 2: He weight initialization", "stage2_he_weights.png")



torch.manual_seed(0)
model = BrokenNet()
for layer in model.hidden:
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    nn.init.zeros_(layer.bias)

grad_results["Stage 3: He init"] = gradient_check(model)
results["Stage 3: He init"] = train(model, torch.optim.SGD(model.parameters(), lr=0.1))
print(f"final loss: {results['Stage 3: He init'][-1]:.4f}")
for i, g in enumerate(grad_results["Stage 3: He init"]):
    print(f"layer {i}: mean |gradient| = {g:.2e}")
plot_losses(["Stage 0: baseline", "Stage 3: He init"], "Stage 3: He weights and zero bias", "stage3_zero_bias.png")



class BrokenNetBN(BrokenNet):
    """Same as BrokenNet (same layers and same initialization), with BatchNorm before each ReLU."""

    def __init__(self, depth=8, width=32, in_features=20):
        super().__init__(depth, width, in_features)
        self.norms = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(depth)])

    def forward(self, x):
        for layer, norm in zip(self.hidden, self.norms):
            x = torch.relu(norm(layer(x)))
        return self.out(x)



torch.manual_seed(0)
model = BrokenNetBN()

grad_results["Stage 4: BatchNorm"] = gradient_check(model)
results["Stage 4: BatchNorm"] = train(model, torch.optim.SGD(model.parameters(), lr=0.1))
print(f"final loss: {results['Stage 4: BatchNorm'][-1]:.4f}")
for i, g in enumerate(grad_results["Stage 4: BatchNorm"]):
    print(f"layer {i}: mean |gradient| = {g:.2e}")
plot_losses(["Stage 0: baseline", "Stage 3: He init", "Stage 4: BatchNorm"],
            "Stage 4: BatchNorm with the original initialization", "stage4_batchnorm.png")



torch.manual_seed(0)
model = BrokenNetBN()
for layer in model.hidden:
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    nn.init.zeros_(layer.bias)

grad_results["Stage 5: He init + BatchNorm"] = gradient_check(model)
results["Stage 5: He init + BatchNorm"] = train(model, torch.optim.SGD(model.parameters(), lr=0.1))
print(f"final loss: {results['Stage 5: He init + BatchNorm'][-1]:.4f}")
plot_losses(["Stage 3: He init", "Stage 4: BatchNorm", "Stage 5: He init + BatchNorm"],
            "Stage 5: He init and BatchNorm together", "stage5_combined.png")



torch.manual_seed(0)
model = BrokenNetBN()
for layer in model.hidden:
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    nn.init.zeros_(layer.bias)

grad_results["Stage 6: + Adam"] = gradient_check(model)
results["Stage 6: + Adam"] = train(model, torch.optim.Adam(model.parameters(), lr=0.01))
print(f"final loss: {results['Stage 6: + Adam'][-1]:.4f}")
plot_losses(["Stage 1: Adam", "Stage 5: He init + BatchNorm", "Stage 6: + Adam"],
            "Stage 6: Adam on the fixed network", "stage6_adam_fixed.png")



plt.figure(figsize=(9, 5))
for name, losses in results.items():
    plt.plot(losses, label=name)
plt.axhline(0.693, color="gray", linestyle="--", label="0.693 (guessing)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss for every stage")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/all_stages.png", dpi=150)
plt.show()

print(f"{'Stage':<30} | {'Final loss':>10}")
print("-" * 43)
for name, losses in results.items():
    print(f"{name:<30} | {losses[-1]:>10.4f}")



plt.figure(figsize=(7, 4.5))
for name in ["Stage 0: baseline", "Stage 3: He init", "Stage 4: BatchNorm", "Stage 5: He init + BatchNorm"]:
    plt.plot(grad_results[name], marker="o", label=name)
plt.xlabel("Layer (0 = closest to input)")
plt.ylabel("Mean |gradient|")
plt.title("Gradient per layer before training")
plt.legend()
plt.tight_layout()
plt.savefig("figures/gradients_per_layer.png", dpi=150)
plt.show()

