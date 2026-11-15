"""Starter code for Lab 07: Latent Spaces.

Run this file once, unmodified, to confirm your environment works. It trains a plain autoencoder
and prints how well it reconstructs and how its samples score. You write everything about the
variational autoencoder: the model, the KL term, the loss, the checks and the experiments.

What is provided
    load_task()               digits data (8 by 8 images, pixels in [0, 1], flattened to 64 values)
    AE                        a plain autoencoder (encode, decode, forward)
    train(model, loss_fn, X)  one training loop that works for any loss you write
    train_feature_net, get_features, frechet_distance, precision_recall, nearest_train_distance,
    score_samples             the sample-quality measures from Module 07, Part 5
    save_grid                 save a grid of 8 by 8 images as a picture

What you write
    a VAE class, the closed-form KL, the loss with a weight on the KL term, a conditional VAE,
    the active-unit count, interpolation, and every experiment.

Requirements: torch, numpy, scipy, scikit-learn, matplotlib. No downloads. A laptop CPU is enough.
"""
import numpy as np
import torch
import torch.nn as nn
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
    return torch.nn.functional.one_hot(y, n).float()

# ---------------------------------------------------------------- plain autoencoder
class AE(nn.Module):
    """Encoder, bottleneck of size d_latent, decoder. With linear=True there are no hidden layers or activations."""

    def __init__(self, d_in=64, d_hidden=64, d_latent=2, linear=False):
        super().__init__()
        if linear:
            self.enc = nn.Linear(d_in, d_latent)
            self.dec = nn.Linear(d_latent, d_in)
        else:
            self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_latent))
            self.dec = nn.Sequential(nn.Linear(d_latent, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_in))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        return self.decode(self.encode(x))



def train(model, loss_fn, X, y=None, epochs=200, lr=2e-3, batch_size=64, seed=0):
    """Train any model with Adam. loss_fn(model, x_batch, y_batch) must return one scalar tensor (the batch mean).
    y is passed through to loss_fn as a batch of labels, or None. Returns the mean loss of each epoch."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        total = 0.0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(model, X[idx], None if y is None else y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        hist.append(total / len(X))
    return hist


def ae_loss(model, xb, yb):
    """Mean squared reconstruction error: the loss for the plain autoencoder."""
    return ((model(xb) - xb) ** 2).mean()


@torch.no_grad()
def reconstruction_mse(model, X):
    """Mean squared error of rebuilding X, for an AE (a model whose forward(x) returns the rebuilt image)."""
    return ((model(X) - X) ** 2).mean().item()


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
            loss = nn.functional.cross_entropy(net(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


@torch.no_grad()
def get_features(net, X):
    return net.features(X).numpy().astype(np.float64)


def frechet_distance(fa, fb):
    """Frechet distance between Gaussians fitted to two sets of features (the quantity behind FID)."""
    mu_a, mu_b = fa.mean(0), fb.mean(0)
    eps = 1e-6 * np.eye(fa.shape[1])
    ca, cb = np.cov(fa, rowvar=False) + eps, np.cov(fb, rowvar=False) + eps
    covmean = linalg.sqrtm(ca @ cb)
    covmean = covmean.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca) + np.trace(cb) - 2 * np.trace(covmean))


def _knn_radius(F, k):
    d = np.linalg.norm(F[:, None] - F[None], axis=-1)
    return np.sort(d, axis=1)[:, k]


def precision_recall(real, fake, k=3):
    """Precision: share of generated points that land inside the real data's neighbourhood.
    Recall: share of real points that land inside the generated data's neighbourhood. (k-nearest-neighbour balls.)"""
    r_real, r_fake = _knn_radius(real, k), _knn_radius(fake, k)
    d = np.linalg.norm(fake[:, None] - real[None], axis=-1)
    precision = float((d <= r_real[None]).any(1).mean())
    recall = float((d.T <= r_fake[None]).any(1).mean())
    return precision, recall


def nearest_train_distance(samples, train):
    """Pixel-space distance from each generated image to its nearest training image. Near zero means copying."""
    d = torch.cdist(samples, train)
    return d.min(1).values


def score_samples(samples, test_features, net, train, k=3):
    """Score a set of generated images (pixels in [0, 1]) against the real test features.
    Returns the Frechet distance, precision, recall and the mean distance to the nearest training image."""
    s = samples.clamp(0, 1)
    f = get_features(net, s)
    fd = frechet_distance(test_features, f)
    p, r = precision_recall(test_features, f, k=k)
    return {"frechet": fd, "precision": p, "recall": r,
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

    ae = AE(d_latent=8)
    train(ae, ae_loss, Xtr, epochs=100)
    print("autoencoder test reconstruction MSE:", round(reconstruction_mse(ae, Xte), 4))

    net = train_feature_net(Xtr, ytr)
    with torch.no_grad():
        print("feature network accuracy on real test images:", round((net(Xte).argmax(1) == yte).float().mean().item(), 3))
    fte = get_features(net, Xte)
    samples = ae.decode(torch.randn(len(Xte), 8)).detach()
    print("autoencoder samples from N(0, I):", {k: round(v, 3) for k, v in score_samples(samples, fte, net, Xtr).items()})
    save_grid(samples[:30], "starter_check.png", ncols=10, title="autoencoder, decoded from N(0, I)")
    print("saved starter_check.png")
