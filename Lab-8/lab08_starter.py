"""Starter code for Lab 08: Two Ways to Generate.

Run this file once, unmodified, to confirm your environment works. It trains a short GAN-free baseline
(the VAE from Module 07), scores its samples, and saves a picture. You write everything about the
GAN and the diffusion model: the losses, the training loops, the forward process, the samplers,
the checks and the experiments.

What is provided
    load_task()                 digits data (8 by 8 images, pixels in [0, 1], flattened to 64 values)
    ring_of_gaussians()         two-dimensional toy data with eight modes, and mode_coverage() to count them
    Generator, Discriminator    plain networks, no training code. The discriminator returns a logit
                                (a real-valued score, which is what a Wasserstein critic needs too)
    Denoiser                    a network that takes a noisy input, a step number and optionally a label
    BETAS                       the noise schedule used in the lab: 100 steps from 1e-4 to 0.1
    VAE, train_vae              the baseline from Module 07, for piece 5 only
    train_feature_net, get_features, frechet_distance, precision_recall, nearest_train_distance,
    score_samples               the sample-quality measures from Module 08, Part 5
    save_grid                   save a grid of 8 by 8 images as a picture

What you write
    the GAN losses (original, non-saturating, Wasserstein with a gradient penalty) and their training
    loops, the closed-form forward process, the noise-prediction loss, the ancestral and the
    deterministic samplers, and every check and experiment.

Requirements: torch, numpy, scipy, scikit-learn, matplotlib. No downloads. A laptop CPU is enough.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from sklearn.datasets import load_digits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- data
def load_task(seed=0, test_fraction=0.2):
    """Return (X_train, y_train, X_test, y_test). Pixels are scaled to [0, 1] and flattened to 64 values."""
    d = load_digits()
    X = (d.data / 16.0).astype(np.float32)
    y = d.target.astype(np.int64)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_fraction)
    te, tr = idx[:n_test], idx[n_test:]
    return (torch.tensor(X[tr]), torch.tensor(y[tr]), torch.tensor(X[te]), torch.tensor(y[te]))


def one_hot(y, n=10):
    return F.one_hot(y, n).float()


def ring_of_gaussians(n, n_modes=8, radius=2.0, std=0.05, seed=0):
    """n points from a mixture of n_modes Gaussians evenly spaced on a circle. Returns (points, mode index)."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randint(0, n_modes, (n,), generator=g)
    ang = 2 * math.pi * k.float() / n_modes
    centers = radius * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    return centers + std * torch.randn(n, 2, generator=g), k


def mode_coverage(samples, n_modes=8, radius=2.0, std=0.05, tol=6.0):
    """Return (modes covered, share of samples within tol*std of a mode centre).
    A mode counts as covered if at least 1 percent of the samples land near it."""
    ang = 2 * math.pi * torch.arange(n_modes).float() / n_modes
    c = radius * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
    near = torch.cdist(samples, c).min(1)
    ok = near.values < tol * std
    counts = torch.bincount(near.indices[ok], minlength=n_modes)
    return int((counts >= 0.01 * len(samples)).sum()), float(ok.float().mean())


# ---------------------------------------------------------------- networks for the GAN and the diffusion model
def _mlp(d_in, d_out, d_hidden, layers, act):
    mods, d = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(d, d_hidden), act()]
        d = d_hidden
    mods.append(nn.Linear(d, d_out))
    return nn.Sequential(*mods)


class Generator(nn.Module):
    """Noise (and an optional one-hot label) in, a sample out. out_act='sigmoid' for images in [0, 1], None for coordinates."""

    def __init__(self, d_noise=8, d_out=2, d_hidden=64, n_classes=0, out_act=None):
        super().__init__()
        self.d_noise, self.n_classes, self.out_act = d_noise, n_classes, out_act
        self.net = _mlp(d_noise + n_classes, d_out, d_hidden, 2, nn.ReLU)

    def forward(self, z, c=None):
        out = self.net(z if c is None else torch.cat([z, c], 1))
        return torch.sigmoid(out) if self.out_act == "sigmoid" else out


class Discriminator(nn.Module):
    """A sample (and an optional label) in, one real number out: a logit. Apply a sigmoid yourself where you need a probability."""

    def __init__(self, d_in=2, d_hidden=64, n_classes=0):
        super().__init__()
        self.net = _mlp(d_in + n_classes, 1, d_hidden, 2, lambda: nn.LeakyReLU(0.2))

    def forward(self, x, c=None):
        return self.net(x if c is None else torch.cat([x, c], 1)).squeeze(1)


class Denoiser(nn.Module):
    """Takes a noisy input x, the step t (an integer per example) and optionally a one-hot label; returns a vector the size of x.
    You decide what it is trained to predict."""

    def __init__(self, d_data=2, d_hidden=128, T=100, n_classes=0):
        super().__init__()
        self.T, self.n_classes = T, n_classes
        self.net = nn.Sequential(nn.Linear(d_data + 16 + n_classes, d_hidden), nn.SiLU(),
                                 nn.Linear(d_hidden, d_hidden), nn.SiLU(),
                                 nn.Linear(d_hidden, d_hidden), nn.SiLU(),
                                 nn.Linear(d_hidden, d_data))

    def forward(self, x, t, c=None):
        freqs = torch.exp(torch.linspace(0, math.log(100.0), 8))
        a = (t.float()[:, None] / self.T) * freqs[None]
        h = [x, torch.cat([torch.sin(a), torch.cos(a)], 1)] + ([] if c is None else [c])
        return self.net(torch.cat(h, 1))


BETAS = torch.linspace(1e-4, 0.1, 100)


# ---------------------------------------------------------------- the Module 07 baseline, for piece 5
class VAE(nn.Module):
    """The baseline VAE from Module 07: 8-dimensional code, one hidden layer of 64 units on each side, Bernoulli decoder (logits)."""

    def __init__(self, d_in=64, d_hidden=64, d_latent=8):
        super().__init__()
        self.d_latent = d_latent
        self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU())
        self.mu, self.logvar = nn.Linear(d_hidden, d_latent), nn.Linear(d_hidden, d_latent)
        self.dec = nn.Sequential(nn.Linear(d_latent, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_in))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.dec(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, seed=9):
        z = torch.randn(n, self.d_latent, generator=torch.Generator().manual_seed(seed))
        return torch.sigmoid(self.dec(z))


def train_vae(model, X, epochs=200, lr=2e-3, batch_size=64, seed=0):
    """Plain VAE training at beta 1 (reconstruction summed over pixels, KL summed over dimensions)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch_size):
            xb = X[perm[i:i + batch_size]]
            logits, mu, logvar = model(xb)
            rec = F.binary_cross_entropy_with_logits(logits, xb, reduction="none").sum(1).mean()
            kl = (0.5 * (mu ** 2 + torch.exp(logvar) - 1 - logvar)).sum(1).mean()
            opt.zero_grad(); (rec + kl).backward(); opt.step()
    return model


# ---------------------------------------------------------------- measuring generated images
class FeatureNet(nn.Module):
    """A small classifier trained on real digits. Its hidden layer plays the role Inception plays in FID."""

    def __init__(self, d_in=64, d_feat=32):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, d_feat), nn.ReLU())
        self.head = nn.Linear(d_feat, 10)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x))


def train_feature_net(X, y, epochs=60, seed=0):
    torch.manual_seed(seed)
    net = FeatureNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            loss = F.cross_entropy(net(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def get_features(net, X):
    return net.features(X).numpy().astype(np.float64)


def frechet_distance(fa, fb):
    """Frechet distance between Gaussians fitted to two sets of features (the quantity behind FID)."""
    mu_a, mu_b = fa.mean(0), fb.mean(0)
    eps = 1e-6 * np.eye(fa.shape[1])
    ca, cb = np.cov(fa, rowvar=False) + eps, np.cov(fb, rowvar=False) + eps
    covmean = linalg.sqrtm(ca @ cb).real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca) + np.trace(cb) - 2 * np.trace(covmean))


def _knn_radius(Fm, k):
    d = np.linalg.norm(Fm[:, None] - Fm[None], axis=-1)
    return np.sort(d, axis=1)[:, k]


def precision_recall(real, fake, k=3):
    """Precision: share of generated points inside the real data's neighbourhood. Recall: the reverse. (k-nearest-neighbour balls.)"""
    r_real, r_fake = _knn_radius(real, k), _knn_radius(fake, k)
    d = np.linalg.norm(fake[:, None] - real[None], axis=-1)
    return float((d <= r_real[None]).any(1).mean()), float((d.T <= r_fake[None]).any(1).mean())


def nearest_train_distance(samples, train):
    """Pixel-space distance from each generated image to its nearest training image. Near zero means copying."""
    return torch.cdist(samples, train).min(1).values


def score_samples(samples, test_features, net, train, k=3):
    """Score generated images (pixels in [0, 1]) against the real test features.
    Returns the Frechet distance, precision, recall and the mean distance to the nearest training image."""
    s = samples.clamp(0, 1)
    f = get_features(net, s)
    p, r = precision_recall(test_features, f, k=k)
    return {"frechet": frechet_distance(test_features, f), "precision": p, "recall": r,
            "nearest_train": nearest_train_distance(s, train).mean().item()}


def save_grid(images, path, ncols=10, title=None):
    """Save images (N by 64, pixels in [0, 1]) as a grid picture."""
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 0.8, nrows * 0.8 + (0.4 if title else 0)), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i].detach().clamp(0, 1).reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- check that the environment works
if __name__ == "__main__":
    torch.manual_seed(0)
    Xtr, ytr, Xte, yte = load_task()
    print("train", tuple(Xtr.shape), "test", tuple(Xte.shape))
    ring, _ = ring_of_gaussians(2000)
    print("modes covered by real ring data:", mode_coverage(ring)[0])
    print("alpha_bar at the last step:", float(torch.cumprod(1 - BETAS, 0)[-1]))

    vae = train_vae(VAE(), Xtr, epochs=30, seed=0)
    net = train_feature_net(Xtr, ytr)
    fte = get_features(net, Xte)
    samples = vae.sample(359)
    print("baseline VAE (30 epochs):", {k: round(v, 3) for k, v in score_samples(samples, fte, net, Xtr).items()})
    save_grid(samples[:30], "starter_check.png", ncols=10, title="baseline VAE samples (30 epochs)")
    print("saved starter_check.png")
