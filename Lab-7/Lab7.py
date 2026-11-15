# Lab 07: Latent Spaces
# Script version of Lab7.ipynb. Run from the Lab-7 folder: python Lab7.py


import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import lab07_starter as S

SEEDS = [0, 1, 2]
LATENT_DIM = 8
HIDDEN_DIM = 64
EPOCHS, LEARNING_RATE, BATCH_SIZE = 200, 2e-3, 64
N_SAMPLES = 359  # every Frechet distance in this notebook uses 359 images
os.makedirs("figures", exist_ok=True)
results = {}
notebook_start = time.perf_counter()


starter_start = time.perf_counter()
starter_run = subprocess.run([sys.executable, "lab07_starter.py"], capture_output=True, text=True, check=True)
print(starter_run.stdout)
print(f"starter took {time.perf_counter() - starter_start:.1f}s")


Xtr, ytr, Xte, yte = S.load_task()
print("train", tuple(Xtr.shape), "test", tuple(Xte.shape))
print(f"pixel range: {Xtr.min().item():.2f} to {Xtr.max().item():.2f}")
print("train images per digit:", torch.bincount(ytr, minlength=10).tolist())
print("test images per digit: ", torch.bincount(yte, minlength=10).tolist())

feature_net = S.train_feature_net(Xtr, ytr)
test_features = S.get_features(feature_net, Xte)
with torch.no_grad():
    feature_net_accuracy = (feature_net(Xte).argmax(1) == yte).float().mean().item()
print(f"feature network test accuracy: {feature_net_accuracy:.3f}   features per image: {test_features.shape[1]}")
results["feature_net_accuracy"] = feature_net_accuracy


class VAE(nn.Module):
    def __init__(self, d_in=64, d_hidden=HIDDEN_DIM, d_latent=LATENT_DIM):
        super().__init__()
        self.d_latent = d_latent
        self.enc_body = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU())
        self.enc_mean = nn.Linear(d_hidden, d_latent)
        self.enc_logvar = nn.Linear(d_hidden, d_latent)
        self.dec = nn.Sequential(nn.Linear(d_latent, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_in))

    def encode(self, x):
        hidden = self.enc_body(x)
        return self.enc_mean(hidden), self.enc_logvar(hidden)

    def reparameterize(self, mean, logvar):
        noise = torch.randn_like(mean)               # the only random part, so gradients flow through mean and std
        return mean + torch.exp(0.5 * logvar) * noise

    def decode(self, z):
        return self.dec(z)                           # logits, one per pixel

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    @torch.no_grad()
    def sample(self, n, generator=None):
        z = torch.randn(n, self.d_latent, generator=generator)
        return torch.sigmoid(self.decode(z))         # pixel probabilities, in [0, 1]


def kl_to_standard_normal(mean, logvar):
    """Closed-form KL(N(mean, exp(logvar)) || N(0, I)), summed over code dimensions. One value per image."""
    return 0.5 * (mean ** 2 + torch.exp(logvar) - 1.0 - logvar).sum(dim=1)


def reconstruction_term(logits, x):
    """Negative Bernoulli log-likelihood of the pixels, summed over the 64 pixels. One value per image."""
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)


def vae_terms(model, xb):
    logits, mean, logvar = model(xb)
    return reconstruction_term(logits, xb), kl_to_standard_normal(mean, logvar)


def make_vae_loss(beta):
    def vae_loss(model, xb, yb):
        recon, kl = vae_terms(model, xb)
        return (recon + beta * kl).mean()            # mean over the batch
    return vae_loss


torch.manual_seed(0)
vae = VAE()
print(vae)
print("parameters:", sum(p.numel() for p in vae.parameters()))


torch.manual_seed(0)
check_means = 1.5 * torch.randn(6, LATENT_DIM)
check_logvars = torch.randn(6, LATENT_DIM)
closed_form_kl = kl_to_standard_normal(check_means, check_logvars)

posterior = torch.distributions.Normal(check_means, torch.exp(0.5 * check_logvars))
prior = torch.distributions.Normal(0.0, 1.0)
draws = posterior.sample((200_000,))                                      # (draws, rows, dims)
log_ratio = (posterior.log_prob(draws) - prior.log_prob(draws)).sum(-1)    # (draws, rows)
monte_carlo_kl = log_ratio.mean(0)
standard_error = log_ratio.std(0) / np.sqrt(len(draws))

print(f"{'row':>3} {'closed form':>12} {'Monte Carlo':>12} {'diff':>8} {'diff / SE':>9} {'averaged (wrong)':>17}")
for row in range(len(check_means)):
    diff = (closed_form_kl[row] - monte_carlo_kl[row]).item()
    print(f"{row:>3} {closed_form_kl[row]:12.4f} {monte_carlo_kl[row]:12.4f} {diff:8.4f} "
          f"{diff / standard_error[row].item():9.2f} {closed_form_kl[row].item() / LATENT_DIM:17.4f}")
max_z = ((closed_form_kl - monte_carlo_kl).abs() / standard_error).max().item()
print(f"largest |diff| in standard errors: {max_z:.2f}")
results["check_kl"] = {"closed_form": closed_form_kl.tolist(), "monte_carlo": monte_carlo_kl.tolist(),
                       "standard_error": standard_error.tolist()}


def gradient_norms(model, xb, sampling, include_kl=False):
    model.zero_grad()
    mean, logvar = model.encode(xb)
    if sampling == "trick":
        z = model.reparameterize(mean, logvar)
    elif sampling == "detached":
        z = model.reparameterize(mean, logvar).detach()
    elif sampling == "no_grad":
        with torch.no_grad():
            z = model.reparameterize(mean, logvar)
    loss = reconstruction_term(model.decode(z), xb).mean()
    if include_kl:
        loss = loss + kl_to_standard_normal(mean, logvar).mean()
    if loss.requires_grad:
        loss.backward()

    def norm(layer):
        return 0.0 if layer.weight.grad is None else layer.weight.grad.norm().item()
    return {"encoder mean": norm(model.enc_mean), "encoder logvar": norm(model.enc_logvar),
            "decoder output": norm(model.dec[-1])}


torch.manual_seed(0)
vae = VAE()
batch = Xtr[:64]
gradient_check = {}
for sampling, include_kl in [("trick", False), ("detached", False), ("no_grad", False), ("detached", True)]:
    torch.manual_seed(1)
    label = f"{sampling}, {'recon + KL' if include_kl else 'recon only'}"
    gradient_check[label] = gradient_norms(vae, batch, sampling, include_kl)
    print(f"{label:<22}", "  ".join(f"{k}: {v:.6f}" for k, v in gradient_check[label].items()))
results["check_gradients"] = gradient_check


class DetachedSampleVAE(VAE):
    def reparameterize(self, mean, logvar):
        return super().reparameterize(mean, logvar).detach()


@torch.no_grad()
def vae_reconstruction_mse(model, X):
    mean, _ = model.encode(X)
    return ((torch.sigmoid(model.decode(mean)) - X) ** 2).mean().item()


detach_demo = {}
for name, model_class in [("trick", VAE), ("detached", DetachedSampleVAE)]:
    torch.manual_seed(0)
    model = model_class()
    history = S.train(model, make_vae_loss(1.0), Xtr, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=0)
    with torch.no_grad():
        recon, kl = vae_terms(model, Xte)
    detach_demo[name] = {"first_loss": history[0], "last_loss": history[-1], "test_kl": kl.mean().item(),
                         "test_mse_from_mean": vae_reconstruction_mse(model, Xte)}
    print(f"{name:<9} loss {history[0]:.2f} -> {history[-1]:.2f}   test KL {kl.mean():.3f}   "
          f"test MSE from posterior mean {detach_demo[name]['test_mse_from_mean']:.4f}")
print(f"MSE of predicting the mean training image: {((Xtr.mean(0) - Xte) ** 2).mean():.4f}")
results["check_detached_training"] = detach_demo


torch.manual_seed(0)
vae = VAE()
S.train(vae, make_vae_loss(1.0), Xtr, epochs=20, lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=0)  # a partly trained model
check_batch = Xte[:64]

with torch.no_grad():
    torch.manual_seed(5)
    my_loss = make_vae_loss(1.0)(vae, check_batch, None).item()

    torch.manual_seed(5)
    mean, logvar = vae.encode(check_batch)
    std = torch.exp(0.5 * logvar)
    z = mean + std * torch.randn_like(mean)
    log_likelihood = torch.distributions.Bernoulli(logits=vae.decode(z), validate_args=False).log_prob(check_batch).sum(1)
    library_kl = torch.distributions.kl_divergence(torch.distributions.Normal(mean, std),
                                                   torch.distributions.Normal(0.0, 1.0)).sum(1)
    negative_elbo = -(log_likelihood - library_kl).mean().item()
print(f"my loss at beta 1:        {my_loss:.6f}")
print(f"-ELBO, torch.distributions: {negative_elbo:.6f}   difference {abs(my_loss - negative_elbo):.2e}")

with torch.no_grad():
    n_draws = 2000
    posterior = torch.distributions.Normal(mean, std)
    z_draws = posterior.sample((n_draws,))                                              # (draws, 64, 8)
    log_px = torch.distributions.Bernoulli(logits=vae.decode(z_draws), validate_args=False).log_prob(check_batch).sum(-1)
    sampled_elbo = log_px + prior.log_prob(z_draws).sum(-1) - posterior.log_prob(z_draws).sum(-1)
    per_draw_loss = torch.stack([make_vae_loss(1.0)(vae, check_batch, None) for _ in range(n_draws)])
print(f"over {n_draws} draws: my loss {per_draw_loss.mean():.4f} +/- {per_draw_loss.std() / np.sqrt(n_draws):.4f}   "
      f"-ELBO (sampled form) {-sampled_elbo.mean(1).mean():.4f} +/- {sampled_elbo.mean(1).std() / np.sqrt(n_draws):.4f}")
results["check_elbo"] = {"my_loss": my_loss, "negative_elbo": negative_elbo}


def train_vae(beta, seed, model_class=None, y=None):
    torch.manual_seed(seed)
    model = (model_class or VAE)()
    S.train(model, make_vae_loss(beta), Xtr, y=y, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=seed)
    return model.eval()


def train_ae(seed):
    torch.manual_seed(seed)
    model = S.AE(d_hidden=HIDDEN_DIM, d_latent=LATENT_DIM)
    S.train(model, S.ae_loss, Xtr, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=seed)
    return model.eval()


@torch.no_grad()
def ae_prior_samples(model, n, seed):
    z = torch.randn(n, LATENT_DIM, generator=torch.Generator().manual_seed(seed))
    return model.decode(z).clamp(0, 1)


def vae_prior_samples(model, n, seed):
    return model.sample(n, generator=torch.Generator().manual_seed(seed))


def score(samples):
    return S.score_samples(samples, test_features, feature_net, Xtr)


ae_models, vae_models, comparison = {}, {}, {}
for seed in SEEDS:
    ae_models[seed] = train_ae(seed)
    vae_models[seed] = train_vae(1.0, seed)
    with torch.no_grad():
        ae_codes = ae_models[seed].encode(Xtr)
        vae_means, vae_logvars = vae_models[seed].encode(Xtr)
    comparison[seed] = {
        "ae": {"test_mse": S.reconstruction_mse(ae_models[seed], Xte),
               **score(ae_prior_samples(ae_models[seed], N_SAMPLES, seed=100 + seed)),
               "code_std": ae_codes.std(0).tolist(), "code_mean": ae_codes.mean(0).tolist()},
        "vae": {"test_mse": vae_reconstruction_mse(vae_models[seed], Xte),
                **score(vae_prior_samples(vae_models[seed], N_SAMPLES, seed=100 + seed)),
                "code_std": vae_means.std(0).tolist(), "code_mean": vae_means.mean(0).tolist(),
                "posterior_std": torch.exp(0.5 * vae_logvars).mean(0).tolist()},
    }

print(f"{'model':<5} {'seed':>4} {'test MSE':>9} {'Frechet':>8} {'precision':>9} {'recall':>7} {'nearest train':>13}")
for name in ["ae", "vae"]:
    for seed in SEEDS:
        row = comparison[seed][name]
        print(f"{name.upper():<5} {seed:>4} {row['test_mse']:9.4f} {row['frechet']:8.2f} {row['precision']:9.3f} "
              f"{row['recall']:7.3f} {row['nearest_train']:13.3f}")
results["ae_vs_vae"] = comparison


def format_row(values):
    return " ".join(f"{v:6.2f}" for v in values)


print(" " * 30 + " ".join(f"{'dim ' + str(d):>6}" for d in range(LATENT_DIM)))
for seed in SEEDS:
    print(f"AE   seed {seed}  std of code         {format_row(comparison[seed]['ae']['code_std'])}")
    print(f"AE   seed {seed}  mean of code        {format_row(comparison[seed]['ae']['code_mean'])}")
for seed in SEEDS:
    print(f"VAE  seed {seed}  std of mean         {format_row(comparison[seed]['vae']['code_std'])}")
    print(f"VAE  seed {seed}  avg posterior std   {format_row(comparison[seed]['vae']['posterior_std'])}")
    total_variance = np.array(comparison[seed]['vae']['code_std']) ** 2 + np.array(comparison[seed]['vae']['posterior_std']) ** 2
    print(f"VAE  seed {seed}  sqrt(total var)     {format_row(np.sqrt(total_variance))}")


def save_rows(rows, path, title=None):
    """rows: list of (label, images N by 64). One labelled row per entry."""
    ncols = max(len(images) for _, images in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(ncols * 0.75 + 1.6, len(rows) * 0.8 + 0.3), squeeze=False)
    for r, (label, images) in enumerate(rows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c < len(images):
                ax.imshow(images[c].detach().clamp(0, 1).reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


with torch.no_grad():
    shown = Xte[:12]
    vae_means_shown, _ = vae_models[0].encode(shown)
    save_rows([("test images", shown),
               ("AE rebuilt", ae_models[0](shown)),
               ("VAE rebuilt", torch.sigmoid(vae_models[0].decode(vae_means_shown))),
               ("AE from N(0, I)", ae_prior_samples(ae_models[0], 12, seed=100)),
               ("VAE from N(0, I)", vae_prior_samples(vae_models[0], 12, seed=100))],
              "figures/ae_vs_vae_samples.png", title="Autoencoder against VAE at beta 1, seed 0")
print("saved figures/ae_vs_vae_samples.png")


@torch.no_grad()
def code_coverage(model, kind, seed):
    if kind == "ae":
        train_codes, test_codes = model.encode(Xtr), model.encode(Xte)
    else:
        torch.manual_seed(seed)
        train_codes = model.reparameterize(*model.encode(Xtr))
        test_codes = model.reparameterize(*model.encode(Xte))
    prior_codes = torch.randn(N_SAMPLES, LATENT_DIM, generator=torch.Generator().manual_seed(100 + seed))
    prior_gap = torch.cdist(prior_codes, train_codes).min(1).values.mean().item()
    test_gap = torch.cdist(test_codes, train_codes).min(1).values.mean().item()
    correlation = torch.corrcoef(train_codes.T) - torch.eye(LATENT_DIM)
    return {"prior_to_train": prior_gap, "test_to_train": test_gap, "ratio": prior_gap / test_gap,
            "max_abs_correlation": correlation.abs().max().item()}


print(f"{'model':<5} {'seed':>4} {'N(0,I) -> train':>16} {'test -> train':>14} {'ratio':>6} {'max |corr|':>11}")
for name, models in [("ae", ae_models), ("vae", vae_models)]:
    for seed in SEEDS:
        coverage = code_coverage(models[seed], name, seed)
        comparison[seed][name]["coverage"] = coverage
        print(f"{name.upper():<5} {seed:>4} {coverage['prior_to_train']:16.3f} {coverage['test_to_train']:14.3f} "
              f"{coverage['ratio']:6.2f} {coverage['max_abs_correlation']:11.2f}")


BETAS = [0.1, 0.5, 1.0, 2.0, 4.0]


@torch.no_grad()
def active_dimensions(model, threshold=0.01):
    mean, _ = model.encode(Xtr)
    return int((mean.var(0) > threshold).sum().item())


@torch.no_grad()
def evaluate_vae(model, seed):
    torch.manual_seed(1000 + seed)
    recon, kl = vae_terms(model, Xte)
    return {"reconstruction": recon.mean().item(), "kl": kl.mean().item(),
            "active": active_dimensions(model),
            **score(vae_prior_samples(model, N_SAMPLES, seed=100 + seed))}


sweep_start = time.perf_counter()
sweep_models, sweep = {}, {}
for beta in BETAS:
    for seed in SEEDS:
        sweep_models[beta, seed] = vae_models[seed] if beta == 1.0 else train_vae(beta, seed)
        sweep[beta, seed] = evaluate_vae(sweep_models[beta, seed], seed)
print(f"sweep took {time.perf_counter() - sweep_start:.0f}s")

print(f"{'beta':>4} {'seed':>4} {'recon':>7} {'KL':>6} {'active':>6} {'Frechet':>8} {'precision':>9} {'recall':>7}")
for beta in BETAS:
    for seed in SEEDS:
        row = sweep[beta, seed]
        print(f"{beta:>4} {seed:>4} {row['reconstruction']:7.2f} {row['kl']:6.2f} {row['active']:>6} "
              f"{row['frechet']:8.2f} {row['precision']:9.3f} {row['recall']:7.3f}")
results["beta_sweep"] = {f"{beta}_{seed}": row for (beta, seed), row in sweep.items()}


FINE_BETAS = [2.5, 3.0, 3.5]
for beta in FINE_BETAS:
    for seed in SEEDS:
        sweep_models[beta, seed] = train_vae(beta, seed)
        sweep[beta, seed] = evaluate_vae(sweep_models[beta, seed], seed)

print(f"{'beta':>4} {'seed':>4} {'recon':>7} {'KL':>6} {'active':>6} {'Frechet':>8} {'precision':>9} {'recall':>7}")
for beta in [2.0] + FINE_BETAS + [4.0]:
    for seed in SEEDS:
        row = sweep[beta, seed]
        print(f"{beta:>4} {seed:>4} {row['reconstruction']:7.2f} {row['kl']:6.2f} {row['active']:>6} "
              f"{row['frechet']:8.2f} {row['precision']:9.3f} {row['recall']:7.3f}")
results["beta_sweep"] = {f"{beta}_{seed}": row for (beta, seed), row in sweep.items()}


with torch.no_grad():
    for seed in SEEDS:
        collapsed_samples = vae_prior_samples(sweep_models[4.0, seed], N_SAMPLES, seed=100 + seed)
        pixel_spread = collapsed_samples.std(0).max().item()
        gap_to_mean_image = (collapsed_samples.mean(0) - Xtr.mean(0)).abs().max().item()
        posterior_means, posterior_logvars = sweep_models[4.0, seed].encode(Xtr)
        print(f"seed {seed}: largest pixel std across samples {pixel_spread:.5f}   "
              f"largest gap to the mean training image {gap_to_mean_image:.4f}   "
              f"posterior means within {posterior_means.abs().max():.4f} of 0, "
              f"posterior std {torch.exp(0.5 * posterior_logvars).min():.3f} to {torch.exp(0.5 * posterior_logvars).max():.3f}")


rows = [(f"beta {beta:g} ({sweep[beta, 0]['active']} active)", vae_prior_samples(sweep_models[beta, 0], 12, seed=100))
        for beta in BETAS + FINE_BETAS if beta != 4.0]
rows.sort(key=lambda row: float(row[0].split()[1]))
rows.append((f"beta 4 ({sweep[4.0, 0]['active']} active)", vae_prior_samples(sweep_models[4.0, 0], 12, seed=100)))
rows.append(("mean train image", Xtr.mean(0, keepdim=True)))
save_rows(rows, "figures/beta_sweep_samples.png", title="VAE prior samples by KL weight, seed 0")
print("saved figures/beta_sweep_samples.png")


pair_rng = np.random.RandomState(0)
pairs = []
while len(pairs) < 40:
    a, b = pair_rng.choice(len(Xte), 2, replace=False)
    if yte[a] != yte[b]:
        pairs.append((a, b))
pairs = np.array(pairs)
steps = torch.linspace(0, 1, 10)[:, None]                      # (10, 1)


@torch.no_grad()
def build_paths(kind, model=None):
    """Returns (pairs, 10 steps, 64 pixels)."""
    image_a, image_b = Xte[pairs[:, 0]], Xte[pairs[:, 1]]
    if kind == "pixels":
        return (1 - steps[None]) * image_a[:, None] + steps[None] * image_b[:, None]
    if kind == "vae":
        code_a, code_b = model.encode(image_a)[0], model.encode(image_b)[0]
    else:
        code_a, code_b = model.encode(image_a), model.encode(image_b)
    codes = (1 - steps[None]) * code_a[:, None] + steps[None] * code_b[:, None]       # (pairs, 10, 8)
    decoded = model.decode(codes)
    return torch.sigmoid(decoded) if kind == "vae" else decoded.clamp(0, 1)


@torch.no_grad()
def path_measures(paths):
    probabilities = torch.softmax(feature_net(paths.reshape(-1, 64)), dim=1).reshape(len(paths), 10, 10)
    confidence, predicted = probabilities.max(dim=2)                                  # (pairs, 10)
    changes = (predicted[:, 1:] != predicted[:, :-1]).sum(1).float()
    endpoints = torch.tensor(yte[pairs].numpy())
    third_class = ((predicted != endpoints[:, :1]) & (predicted != endpoints[:, 1:])).sum(1).float()
    return {"confidence": confidence.mean().item(), "changes": changes.mean().item(),
            "one_change": (changes == 1).float().mean().item(), "third_class_steps": third_class.mean().item()}


interpolation = {}
for seed in SEEDS:
    interpolation[seed] = {"vae": path_measures(build_paths("vae", vae_models[seed])),
                           "ae": path_measures(build_paths("ae", ae_models[seed])),
                           "pixels": path_measures(build_paths("pixels"))}

print(f"{'path':<7} {'seed':>4} {'confidence':>10} {'changes':>8} {'share with 1 change':>20} {'third-class steps':>18}")
for kind in ["vae", "ae", "pixels"]:
    for seed in SEEDS:
        row = interpolation[seed][kind]
        print(f"{kind:<7} {seed:>4} {row['confidence']:10.3f} {row['changes']:8.2f} {row['one_change']:20.2f} "
              f"{row['third_class_steps']:18.2f}")
results["interpolation"] = interpolation


path_rows = []
for pair_index in range(3):
    a, b = pairs[pair_index]
    path_rows += [(f"VAE  {yte[a].item()} to {yte[b].item()}", build_paths("vae", vae_models[0])[pair_index]),
                  (f"AE  {yte[a].item()} to {yte[b].item()}", build_paths("ae", ae_models[0])[pair_index]),
                  (f"pixels  {yte[a].item()} to {yte[b].item()}", build_paths("pixels")[pair_index])]
save_rows(path_rows, "figures/latent_paths.png", title="10-step paths between test digits, seed 0")
print("saved figures/latent_paths.png")


@torch.no_grad()
def path_breakdown(paths):
    probabilities = torch.softmax(feature_net(paths.reshape(-1, 64)), dim=1).reshape(len(paths), 10, 10)
    confidence, predicted = probabilities.max(dim=2)
    endpoints = torch.tensor(yte[pairs].numpy())
    ends_correct = ((predicted[:, 0] == endpoints[:, 0]) & (predicted[:, -1] == endpoints[:, 1])).float().mean().item()
    return {"ends_correct": ends_correct, "middle_confidence": confidence[:, 3:7].mean().item()}


print(f"{'path':<7} {'seed':>4} {'both ends right':>16} {'middle-4 confidence':>20}")
for kind in ["vae", "ae", "pixels"]:
    for seed in SEEDS:
        model = {"vae": vae_models, "ae": ae_models}.get(kind, {}).get(seed)
        breakdown = path_breakdown(build_paths(kind, model))
        interpolation[seed][kind].update(breakdown)
        print(f"{kind:<7} {seed:>4} {breakdown['ends_correct']:16.2f} {breakdown['middle_confidence']:20.3f}")


class ConditionalVAE(VAE):
    def __init__(self, n_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        d_in, d_hidden = self.enc_body[0].in_features, self.enc_body[0].out_features
        self.enc_body = nn.Sequential(nn.Linear(d_in + n_classes, d_hidden), nn.ReLU())
        self.dec[0] = nn.Linear(self.d_latent + n_classes, d_hidden)

    def encode(self, x, labels):
        return super().encode(torch.cat([x, S.one_hot(labels, self.n_classes)], dim=1))

    def decode(self, z, labels):
        return super().decode(torch.cat([z, S.one_hot(labels, self.n_classes)], dim=1))

    def forward(self, x, labels):
        mean, logvar = self.encode(x, labels)
        return self.decode(self.reparameterize(mean, logvar), labels), mean, logvar

    @torch.no_grad()
    def sample(self, labels, generator=None):
        z = torch.randn(len(labels), self.d_latent, generator=generator)
        return torch.sigmoid(self.decode(z, labels))


def make_cvae_loss(beta):
    def cvae_loss(model, xb, yb):
        logits, mean, logvar = model(xb, yb)
        return (reconstruction_term(logits, xb) + beta * kl_to_standard_normal(mean, logvar)).mean()
    return cvae_loss


def train_cvae(beta, seed):
    torch.manual_seed(seed)
    model = ConditionalVAE()
    S.train(model, make_cvae_loss(beta), Xtr, y=ytr, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=seed)
    return model.eval()


@torch.no_grad()
def classify(images):
    return feature_net(images).argmax(1)


cvae_models, conditioning = {}, {}
requested = torch.arange(10).repeat_interleave(200)
for seed in SEEDS:
    cvae_models[seed] = train_cvae(1.0, seed)
    with torch.no_grad():
        torch.manual_seed(1000 + seed)
        logits, mean, logvar = cvae_models[seed](Xte, yte)
        test_kl = kl_to_standard_normal(mean, logvar).mean().item()
        active = int((cvae_models[seed].encode(Xtr, ytr)[0].var(0) > 0.01).sum().item())
    predicted = classify(cvae_models[seed].sample(requested, generator=torch.Generator().manual_seed(200 + seed)))
    per_digit = [(predicted[requested == digit] == digit).float().mean().item() for digit in range(10)]
    unconditional_counts = torch.bincount(classify(vae_prior_samples(vae_models[seed], 1000, seed=300 + seed)),
                                          minlength=10).tolist()
    conditioning[seed] = {"per_digit": per_digit, "overall": (predicted == requested).float().mean().item(),
                          "test_kl": test_kl, "active": active, "unconditional_counts": unconditional_counts}

print("share of 200 conditional samples classified as the requested digit")
print(f"{'seed':>4} " + " ".join(f"{d:>5}" for d in range(10)) + f" {'all':>6} {'test KL':>8} {'active':>6}")
for seed in SEEDS:
    row = conditioning[seed]
    print(f"{seed:>4} " + " ".join(f"{share:5.3f}" for share in row["per_digit"])
          + f" {row['overall']:6.3f} {row['test_kl']:8.2f} {row['active']:>6}")

print("\nclass counts of 1,000 unconditional beta 1 VAE samples (uniform would be 100 each)")
print(f"{'seed':>4} " + " ".join(f"{d:>5}" for d in range(10)) + f" {'max/min':>8}")
for seed in SEEDS:
    counts = conditioning[seed]["unconditional_counts"]
    print(f"{seed:>4} " + " ".join(f"{c:>5}" for c in counts) + f" {max(counts) / max(min(counts), 1):8.1f}")
print(f"{'train':>4} " + " ".join(f"{c:>5}" for c in torch.bincount(ytr, minlength=10).tolist()))
results["conditioning"] = conditioning


conditional_rows = [(f"asked for {digit}", cvae_models[0].sample(torch.full((10,), digit), generator=torch.Generator().manual_seed(digit)))
                    for digit in range(10)]
save_rows(conditional_rows, "figures/cvae_samples.png", title="Conditional VAE at beta 1, prior samples per digit, seed 0")
print("saved figures/cvae_samples.png")


def mean_pairwise_distance(images):
    distances = torch.cdist(images, images)
    n = len(images)
    return (distances.sum() / (n * (n - 1))).item()


real_spread = np.mean([mean_pairwise_distance(Xtr[ytr == digit]) for digit in range(10)])
print(f"real training images:            {real_spread:.3f}")
for seed in SEEDS:
    samples = cvae_models[seed].sample(requested, generator=torch.Generator().manual_seed(200 + seed))
    conditional_spread = np.mean([mean_pairwise_distance(samples[requested == digit]) for digit in range(10)])
    unconditional = vae_prior_samples(vae_models[seed], 1000, seed=300 + seed)
    unconditional_labels = classify(unconditional)
    unconditional_spread = np.mean([mean_pairwise_distance(unconditional[unconditional_labels == digit]) for digit in range(10)])
    conditioning[seed]["within_digit_spread"] = conditional_spread
    conditioning[seed]["unconditional_within_digit_spread"] = unconditional_spread
    print(f"seed {seed}  conditional VAE: {conditional_spread:.3f}   unconditional VAE: {unconditional_spread:.3f}")
results["conditioning"]["real_within_digit_spread"] = real_spread


def six_sets(seed):
    generator = torch.Generator().manual_seed(500 + seed)
    subset = torch.randperm(len(Xtr), generator=generator)[:N_SAMPLES]
    real = Xtr[subset]
    return {
        "real train": real,
        "copier": real + 0.02 * torch.randn(real.shape, generator=generator),
        "mean + noise": Xtr.mean(0) + 0.1 * torch.randn(real.shape, generator=generator),
        "collapsed VAE": vae_prior_samples(sweep_models[4.0, seed], N_SAMPLES, seed=100 + seed),
        "VAE beta 1": vae_prior_samples(vae_models[seed], N_SAMPLES, seed=100 + seed),
        "autoencoder": ae_prior_samples(ae_models[seed], N_SAMPLES, seed=100 + seed),
    }


metric_table = {}
for seed in SEEDS:
    for name, images in six_sets(seed).items():
        metric_table[name, seed] = score(images)

print(f"{'set':<14} {'seed':>4} {'Frechet':>8} {'precision':>9} {'recall':>7} {'nearest train':>13}")
for name in ["real train", "copier", "mean + noise", "collapsed VAE", "VAE beta 1", "autoencoder"]:
    for seed in SEEDS:
        row = metric_table[name, seed]
        print(f"{name:<14} {seed:>4} {row['frechet']:8.2f} {row['precision']:9.3f} {row['recall']:7.3f} {row['nearest_train']:13.3f}")
held_out_gap = S.nearest_train_distance(Xte, Xtr).mean().item()
print(f"\nreal test images to their nearest training image: {held_out_gap:.3f}")
results["metric_table"] = {f"{name}_{seed}": row for (name, seed), row in metric_table.items()}
results["test_to_nearest_train"] = held_out_gap


subset_sizes = [30, 60, 120, 359, 1000]
size_study = {}
size_generator = torch.Generator().manual_seed(7)
for size in subset_sizes:
    distances = []
    for _ in range(10):
        subset = torch.randperm(len(Xtr), generator=size_generator)[:size]
        distances.append(S.frechet_distance(test_features, S.get_features(feature_net, Xtr[subset])))
    size_study[size] = distances

print(f"{'size':>5} {'mean':>7} {'std':>6} {'min':>7} {'max':>7}")
for size in subset_sizes:
    d = np.array(size_study[size])
    print(f"{size:>5} {d.mean():7.2f} {d.std():6.2f} {d.min():7.2f} {d.max():7.2f}")
results["frechet_vs_size"] = {str(size): distances for size, distances in size_study.items()}


fig, ax = plt.subplots(figsize=(6, 3.6))
means = [np.mean(size_study[size]) for size in subset_sizes]
ax.plot(subset_sizes, means, "o-", color="black", label="real training subsets (mean of 10)")
for size in subset_sizes:
    ax.scatter([size] * 10, size_study[size], color="black", alpha=0.25, s=10)
for label, values, color in [("VAE beta 0.5", [sweep[0.5, s]["frechet"] for s in SEEDS], "tab:green"),
                             ("VAE beta 1", [sweep[1.0, s]["frechet"] for s in SEEDS], "tab:blue")]:
    ax.axhspan(min(values), max(values), color=color, alpha=0.2, label=f"{label} at 359, 3 seeds")
ax.axvline(N_SAMPLES, color="gray", linestyle=":", linewidth=1)
ax.set_xscale("log")
ax.set_xticks(subset_sizes)
ax.set_xticklabels(subset_sizes)
ax.set_xlabel("number of real training images scored")
ax.set_ylabel("Fréchet distance to the 359 test images")
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig("figures/frechet_vs_size.png", dpi=150)
plt.close(fig)
print("saved figures/frechet_vs_size.png")


results["total_seconds"] = time.perf_counter() - notebook_start
with open("lab7_results.json", "w") as f:
    json.dump(results, f, indent=1, default=float)
print(f"whole notebook: {results['total_seconds'] / 60:.1f} minutes on {torch.get_num_threads()} CPU threads, torch {torch.__version__}")
