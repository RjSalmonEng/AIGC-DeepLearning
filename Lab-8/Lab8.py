# Lab 08: Two Ways to Generate
# Script version of Lab8.ipynb. Run from the Lab-8 folder: python Lab8.py  (or --pieces 1 2 ...)

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import lab08_starter as S

SEEDS = [0, 1, 2]
SAMPLE_SEED = 9            # noise seed for every evaluation draw (matches S.VAE.sample's default)
N_SAMPLES = 359            # every Frechet distance, precision and recall uses 359 images
DEVICE = "cpu"
BATCH_SIZE = 128
GAN_STEPS = 4000           # generator steps
GAN_BETAS = (0.5, 0.999)
STANDARD_LR_G, STANDARD_LR_D = 5e-4, 2e-4
WGAN_LR, WGAN_CRITIC_STEPS, GP_WEIGHT = 5e-4, 3, 10.0
DIFFUSION_STEPS = 4000
RING_HIDDEN, RING_LR = 128, 2e-3
DIGIT_HIDDEN, DIGIT_LR = 256, 1e-3
T_STEPS = 100
DETERMINISTIC_KS = [5, 10, 25, 50, 100]
os.makedirs("figures", exist_ok=True)
results = {}
run_start = time.perf_counter()

print("python", platform.python_version(), "| torch", torch.__version__, "| threads", torch.get_num_threads(),
      "| venv kernel:", sys.prefix.endswith(".venv"))


Xtr, ytr, Xte, yte = S.load_task()
print("train", tuple(Xtr.shape), "test", tuple(Xte.shape))
print(f"pixel range: {Xtr.min().item():.2f} to {Xtr.max().item():.2f}")
print("train images per digit:", torch.bincount(ytr, minlength=10).tolist())
print("test images per digit: ", torch.bincount(yte, minlength=10).tolist())

feature_net = S.train_feature_net(Xtr, ytr)       # seed 0 inside; trained ONCE, reused for every score
test_features = S.get_features(feature_net, Xte)
with torch.no_grad():
    feature_net_accuracy = (feature_net(Xte).argmax(1) == yte).float().mean().item()
test_shares = torch.bincount(yte, minlength=10).double() / len(yte)
print(f"feature network test accuracy: {feature_net_accuracy:.3f}")


def score_images(images, expected_count=N_SAMPLES):
    """Score images that are ALREADY in [0, 1] (diffusion samples must go through to_unit_range first).
    Returns the starter's four scores plus nearest_train_min, digit_counts, digits_present, tv."""
    assert len(images) == expected_count, f"expected {expected_count} images, got {len(images)}"
    assert images.min() >= 0 and images.max() <= 1, "images must be in [0, 1] before scoring"
    images = images.detach()
    scores = S.score_samples(images, test_features, feature_net, Xtr)
    distances = S.nearest_train_distance(images, Xtr)
    with torch.no_grad():
        predicted = feature_net(images).argmax(1)
    counts = torch.bincount(predicted, minlength=10)
    shares = counts.double() / len(images)
    scores.update({"nearest_train_min": float(distances.min()),
                   "digit_counts": [int(c) for c in counts],
                   "digits_present": int((counts > 0).sum()),
                   "tv": float(0.5 * (shares - test_shares).abs().sum())})
    return scores


# Reference block: what the scores look like for real images
test_to_train = S.nearest_train_distance(Xte, Xtr)
subset_generator = torch.Generator().manual_seed(0)
train_subset = Xtr[torch.randperm(len(Xtr), generator=subset_generator)[:N_SAMPLES]]
test_digit_counts = [int(c) for c in torch.bincount(yte, minlength=10)]
results["feature_net_accuracy"] = feature_net_accuracy
results["reference"] = {
    "test_nearest_train_mean": float(test_to_train.mean()),
    "test_nearest_train_min": float(test_to_train.min()),
    "test_digit_counts": test_digit_counts,
    "train_subset_score": score_images(train_subset),   # nearest-train is 0 here: these images are training images
}
print("reference: real test images to nearest training image, mean "
      f"{results['reference']['test_nearest_train_mean']:.3f}, min {results['reference']['test_nearest_train_min']:.3f}")
print("reference: 359 real training images scored as if generated:",
      {k: (round(v, 3) if isinstance(v, float) else v) for k, v in results["reference"]["train_subset_score"].items()})


def _reduce(values, reduction):
    assert reduction in ("mean", "sum")
    return values.mean() if reduction == "mean" else values.sum()


def discriminator_loss(real_logits, fake_logits, reduction="mean"):
    """-log D(real) - log(1 - D(fake)) with D = sigmoid(logit). Uses log sigmoid(s) = -softplus(-s) and log(1 - sigmoid(s)) = -softplus(s)."""
    return _reduce(F.softplus(-real_logits), reduction) + _reduce(F.softplus(fake_logits), reduction)


def generator_loss_original(fake_logits, reduction="mean"):
    """The original generator loss, log(1 - D(G(z))). The generator minimizes this one."""
    return _reduce(-F.softplus(fake_logits), reduction)


def generator_loss_nonsaturating(fake_logits, reduction="mean"):
    """The non-saturating generator loss, -log D(G(z))."""
    return _reduce(F.softplus(-fake_logits), reduction)


def optimal_discriminator_gap(seed):
    """Train a 1-D discriminator on N(0,1) against a fixed N(1.5,1) generator; return the largest gap to the optimal D and where it is."""
    torch.manual_seed(seed)
    disc = S.Discriminator(d_in=1, d_hidden=32)
    optimizer = torch.optim.Adam(disc.parameters(), lr=3e-3)           # default betas (0.9, 0.999), stated
    for _ in range(3000):
        real = torch.randn(256, 1)
        fake = torch.randn(256, 1) + 1.5                                # fixed generator N(1.5, 1)
        loss = discriminator_loss(disc(real), disc(fake))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    grid = torch.linspace(-2.0, 3.5, 551)                               # step 0.01, endpoints included
    p_data = torch.exp(-grid ** 2 / 2) / math.sqrt(2 * math.pi)
    p_gen = torch.exp(-(grid - 1.5) ** 2 / 2) / math.sqrt(2 * math.pi)
    optimal = p_data / (p_data + p_gen)
    with torch.no_grad():
        learned = torch.sigmoid(disc(grid[:, None]))
    gaps = (learned - optimal).abs()
    return float(gaps.max()), float(grid[gaps.argmax()])


ring_points, _ = S.ring_of_gaussians(5000)          # data seed 0 (starter default)


def saturation_check(seed):
    """Shift a fresh generator's output by 6, train D for 300 steps, then measure the gradient each generator loss sends to the output."""
    torch.manual_seed(seed)
    generator = S.Generator(d_noise=8, d_out=2)                         # hidden 64, starter default
    with torch.no_grad():
        generator.net[-1].bias += torch.tensor([6.0, 6.0])              # shift the output by 6 in both coordinates
    disc = S.Discriminator()                                            # d_in=2, hidden 64
    optimizer = torch.optim.Adam(disc.parameters(), lr=2e-3, betas=(0.5, 0.9))
    for _ in range(300):
        real = ring_points[torch.randint(0, len(ring_points), (128,))]
        with torch.no_grad():
            fake = generator(torch.randn(128, 8))
        loss = discriminator_loss(disc(real), disc(fake))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        generated = generator(torch.randn(1000, 8))
    generated = generated.clone().requires_grad_(True)                  # leaf: "the generator's output"
    logits = disc(generated)
    out = {"mean_d_of_g": float(torch.sigmoid(logits).mean())}
    for name, loss_fn in [("original", generator_loss_original), ("nonsaturating", generator_loss_nonsaturating)]:
        grad, = torch.autograd.grad(loss_fn(logits, reduction="sum"), generated, retain_graph=True)
        out[f"grad_norm_{name}"] = float(grad.norm(dim=1).mean())      # per-sample norm, then mean
    return out


def gradient_penalty(critic, real, fake, mix=None):
    """Penalty BEFORE the weight 10: mean over examples of (||grad_x critic(x_hat)|| - 1)^2,
    x_hat = mix * real + (1 - mix) * fake, one mix value per example (shape [N, 1]).
    Returns (penalty, per_example_norms)."""
    if mix is None:
        mix = torch.rand(len(real), 1)
    assert mix.shape == (len(real), 1)
    points = (mix * real + (1 - mix) * fake).detach().requires_grad_(True)
    scores = critic(points)
    grads, = torch.autograd.grad(scores.sum(), points, create_graph=True)   # create_graph: the penalty must train the critic
    norms = grads.norm(dim=1)                                                # per example, NOT over the batch
    return ((norms - 1) ** 2).mean(), norms


def train_gan(kind, seed, steps=GAN_STEPS):
    """kind 'standard' or 'wgan_gp'. Returns (generator, history, train_seconds)."""
    torch.manual_seed(seed)                                            # BEFORE construction
    generator = S.Generator(d_noise=16, d_out=64, d_hidden=128, out_act="sigmoid")
    critic = S.Discriminator(d_in=64, d_hidden=128)
    if kind == "standard":
        lr_g, lr_d, critic_steps = STANDARD_LR_G, STANDARD_LR_D, 1
    else:
        lr_g, lr_d, critic_steps = WGAN_LR, WGAN_LR, WGAN_CRITIC_STEPS
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=GAN_BETAS)
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr_d, betas=GAN_BETAS)
    history = {"critic_norm": [], "score_gap": [], "critic_norm_per_update": [], "score_gap_per_update": [],
               "d_real_prob": [], "d_fake_prob": [], "d_loss": [], "g_loss": []}
    start = time.perf_counter()
    for step in range(steps):
        step_norms, step_gaps, step_d_losses = [], [], []
        for _ in range(critic_steps):
            real = Xtr[torch.randint(0, len(Xtr), (BATCH_SIZE,))]       # sampled with replacement
            with torch.no_grad():
                fake = generator(torch.randn(BATCH_SIZE, 16))
            real_scores, fake_scores = critic(real), critic(fake)
            if kind == "standard":
                d_loss = discriminator_loss(real_scores, fake_scores)
            else:
                penalty, norms = gradient_penalty(critic, real, fake)
                d_loss = fake_scores.mean() - real_scores.mean() + GP_WEIGHT * penalty
                step_norms.append(float(norms.mean()))
                step_gaps.append(float(real_scores.mean() - fake_scores.mean()))   # before this update, without the penalty
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            step_d_losses.append(float(d_loss))
        fake_scores_for_g = critic(generator(torch.randn(BATCH_SIZE, 16)))
        g_loss = generator_loss_nonsaturating(fake_scores_for_g) if kind == "standard" else -fake_scores_for_g.mean()
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()
        history["d_loss"].append(float(np.mean(step_d_losses)))
        history["g_loss"].append(float(g_loss))
        if kind == "standard":
            history["d_real_prob"].append(float(torch.sigmoid(real_scores).mean()))
            history["d_fake_prob"].append(float(torch.sigmoid(fake_scores).mean()))
        else:
            history["critic_norm"].append(float(np.mean(step_norms)))
            history["score_gap"].append(float(np.mean(step_gaps)))
            history["critic_norm_per_update"].extend(step_norms)
            history["score_gap_per_update"].extend(step_gaps)
    return generator, history, time.perf_counter() - start


@torch.no_grad()
def sample_gan(generator, n=N_SAMPLES):
    z = torch.randn(n, 16, generator=torch.Generator().manual_seed(SAMPLE_SEED))   # "decoded from N(0, I)"
    return generator(z)


def time_draw(draw_fn, repeats=5):
    """Median wall-clock seconds of draw_fn() over `repeats` calls (each draws all 359 samples)."""
    durations = []
    for _ in range(repeats):
        start = time.perf_counter(); draw_fn(); durations.append(time.perf_counter() - start)
    return float(np.median(durations))


def score_row(label, seed, scores, extra=""):
    return (f"{label:10s} {seed:4d} {scores['frechet']:8.2f} {scores['precision']:10.3f} {scores['recall']:8.3f} "
            f"{scores['digits_present']:7d} {scores['nearest_train']:8.3f} {extra}")


BETAS = S.BETAS
ALPHAS = 1 - BETAS
ALPHA_BARS = torch.cumprod(ALPHAS, 0)


def q_sample(x0, t, noise):
    """Closed-form forward process. t: long tensor [N] of 0-based indices."""
    alpha_bar = ALPHA_BARS[t].to(x0.dtype)[:, None]
    return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise


def noise_prediction_loss(model, x0):
    """Squared error between the true noise and the model's output at one random step per example (mean over batch and dims)."""
    t = torch.randint(0, T_STEPS, (len(x0),))
    noise = torch.randn_like(x0)
    return ((model(q_sample(x0, t, noise), t) - noise) ** 2).mean()


def train_diffusion(data, d_hidden, lr, seed, cosine, steps=DIFFUSION_STEPS):
    torch.manual_seed(seed)                                                # BEFORE construction
    model = S.Denoiser(d_data=data.shape[1], d_hidden=d_hidden, T=T_STEPS)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.0) if cosine else None
    losses, learning_rates = [], []
    start = time.perf_counter()
    for _ in range(steps):
        x0 = data[torch.randint(0, len(data), (BATCH_SIZE,))]
        loss = noise_prediction_loss(model, x0)
        learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss))
    return model, losses, time.perf_counter() - start, learning_rates


@torch.no_grad()
def sample_ancestral(model, n, d_data, seed=SAMPLE_SEED):
    """Ancestral sampler with variance sigma_t^2 = beta_t. 100 network evaluations; the last step (t = 0) adds no noise."""
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_data, generator=rng)                              # x at index 99 ~ N(0, I)
    for t in range(T_STEPS - 1, -1, -1):                                   # 99, 98, ..., 0
        eps = model(x, torch.full((n,), t, dtype=torch.long))
        mean = (x - BETAS[t] / (1 - ALPHA_BARS[t]).sqrt() * eps) / ALPHAS[t].sqrt()
        if t > 0:
            x = mean + BETAS[t].sqrt() * torch.randn(n, d_data, generator=rng)
        else:
            x = mean                                                       # no noise at the last step
    return x


def to_unit_range(x):
    """[-1, 1] -> [0, 1], then clip. Apply BEFORE any scoring of digit diffusion samples."""
    return ((x + 1) / 2).clamp(0, 1)


def clipped_share(x):
    """Share of pixels that fall outside [0, 1] after rescaling (and so get clipped)."""
    y = (x + 1) / 2
    return float(((y < 0) | (y > 1)).float().mean())


def deterministic_timesteps(k):
    steps = torch.linspace(T_STEPS - 1, 0, k).round().long()
    assert steps[0] == T_STEPS - 1 and steps[-1] == 0 and len(set(steps.tolist())) == k
    assert bool((steps[:-1] > steps[1:]).all())
    return steps


@torch.no_grad()
def sample_deterministic(model, n, d_data, k, seed=SAMPLE_SEED, return_last_x0=False):
    """Deterministic sampler on k steps t_0 > t_1 > ... taken from 99 down to 0. k network evaluations; no clipping of x0 (not in the handout formula)."""
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_data, generator=rng)                 # SAME x_T as sample_ancestral with the same seed
    steps = deterministic_timesteps(k)
    for i, t in enumerate(steps.tolist()):
        alpha_bar = ALPHA_BARS[t]
        alpha_bar_next = ALPHA_BARS[steps[i + 1]] if i + 1 < k else torch.tensor(1.0)   # 1 after the last step
        eps = model(x, torch.full((n,), t, dtype=torch.long))
        x0_estimate = (x - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt()
        x = alpha_bar_next.sqrt() * x0_estimate + (1 - alpha_bar_next).sqrt() * eps
    return (x, x0_estimate) if return_last_x0 else x


def run_starter():
    starter_start = time.perf_counter()
    starter_run = subprocess.run([sys.executable, "lab08_starter.py"], capture_output=True, text=True, check=True)
    print(starter_run.stdout)
    print(f"starter took {time.perf_counter() - starter_start:.1f}s")


def piece1(results):
    print("\n=== piece 1: GAN losses and three checks ===")
    torch.manual_seed(0)
    logits_real = torch.randn(1000, dtype=torch.float64) * 4
    logits_fake = torch.randn(1000, dtype=torch.float64) * 4
    naive_d = -(torch.log(torch.sigmoid(logits_real)).mean() + torch.log(1 - torch.sigmoid(logits_fake)).mean())
    naive_gmm = torch.log(1 - torch.sigmoid(logits_fake)).mean()
    naive_gns = -torch.log(torch.sigmoid(logits_fake)).mean()
    loss_rows = [("discriminator loss", discriminator_loss(logits_real, logits_fake), naive_d),
                 ("generator loss, original", generator_loss_original(logits_fake), naive_gmm),
                 ("generator loss, non-saturating", generator_loss_nonsaturating(logits_fake), naive_gns)]
    check_losses = {"rows": {}}
    print(f"{'loss':32s} {'mine':>20s} {'definition':>20s} {'abs diff':>10s}")
    for name, mine, definition in loss_rows:
        diff = float((mine - definition).abs())
        check_losses["rows"][name] = {"mine": float(mine), "definition": float(definition), "abs_diff": diff}
        print(f"{name:32s} {float(mine):20.12f} {float(definition):20.12f} {diff:10.2e}")
    check_losses["max_abs_diff"] = max(row["abs_diff"] for row in check_losses["rows"].values())
    check_losses["pass"] = check_losses["max_abs_diff"] < 1e-10
    print("max abs diff", f"{check_losses['max_abs_diff']:.2e}", "| pass (< 1e-10):", check_losses["pass"])

    stability_logits = torch.tensor([-40.0, 0.0, 40.0])
    naive_stability = torch.log(1 - torch.sigmoid(stability_logits))
    mine_stability = -F.softplus(stability_logits)
    check_losses["stability_float32"] = {"logits": stability_logits.tolist(), "naive_log_one_minus_sigmoid": naive_stability.tolist(),
                                         "mine": mine_stability.tolist()}
    print("float32 stability, logits", stability_logits.tolist())
    print("  naive log(1 - sigmoid(s)):", naive_stability.tolist())
    print("  mine  -softplus(s):       ", mine_stability.tolist())
    results["check_losses"] = check_losses

    check_optimal_discriminator = {}
    print(f"{'seed':>4s} {'largest gap':>12s} {'at x':>7s}")
    for seed in SEEDS:
        gap, location = optimal_discriminator_gap(seed)
        check_optimal_discriminator[str(seed)] = {"largest_gap": gap, "at_x": location, "pass": gap < 0.05}
        print(f"{seed:4d} {gap:12.5f} {location:7.2f}   pass (< 0.05): {gap < 0.05}")
    results["check_optimal_discriminator"] = check_optimal_discriminator

    check_saturation = {}
    print(f"{'seed':>4s} {'mean D(G(z))':>14s} {'orig grad norm':>16s} {'nonsat grad norm':>18s} {'ratio':>10s}")
    for seed in SEEDS:
        row = saturation_check(seed)
        row["ratio_nonsat_over_orig"] = row["grad_norm_nonsaturating"] / row["grad_norm_original"]
        row["pass"] = bool(row["mean_d_of_g"] < 0.01 and row["grad_norm_original"] < 1e-2
                           and row["grad_norm_nonsaturating"] > 0.1 and row["ratio_nonsat_over_orig"] > 100)
        check_saturation[str(seed)] = row
        print(f"{seed:4d} {row['mean_d_of_g']:14.5f} {row['grad_norm_original']:16.5f} {row['grad_norm_nonsaturating']:18.5f} {row['ratio_nonsat_over_orig']:10.1f}"
              f"   | D(G(z)) {row['mean_d_of_g']:.3e}, orig {row['grad_norm_original']:.3e}, nonsat {row['grad_norm_nonsaturating']:.3e}   pass: {row['pass']}")
    results["check_saturation"] = check_saturation


def piece2(results, kinds=("standard", "wgan_gp"), with_checks=True):
    print("\n=== piece 2: gradient penalty, standard GAN against WGAN-GP ===")
    if with_checks:
        torch.manual_seed(0)
        real_batch, fake_batch = Xtr[:128], torch.rand(128, 64)
        direction = torch.randn(64)
        direction = direction / direction.norm()
        check_gradient_penalty = {"linear": []}
        print(f"{'target norm':>11s} {'norm min':>9s} {'norm max':>9s} {'my penalty':>12s} {'expected':>9s} {'abs diff':>10s} {'whole-batch norm (wrong)':>26s}")
        for target_norm, expected in [(1.0, 0.0), (3.0, 4.0)]:
            weight = target_norm * direction
            penalty, norms = gradient_penalty(lambda x: x @ weight, real_batch, fake_batch)
            wrong = float((weight.expand(128, 64).norm() - 1) ** 2)          # norm over the whole batch: the mistake the handout warns about
            diff = abs(float(penalty) - expected)
            tolerance = 1e-6 if expected == 0.0 else 1e-4
            check_gradient_penalty["linear"].append({"target_norm": target_norm, "norm_min": float(norms.min()), "norm_max": float(norms.max()),
                                                     "penalty": float(penalty), "expected": expected, "abs_diff": diff,
                                                     "whole_batch_wrong": wrong, "pass": diff < tolerance})
            print(f"{target_norm:11.1f} {float(norms.min()):9.6f} {float(norms.max()):9.6f} {float(penalty):12.8f} {expected:9.1f} {diff:10.2e} {wrong:26.2f}   pass: {diff < tolerance}")


        def penalty_on_one_set(critic, points_in):
            """The same formula applied directly to one set of points (no interpolation), for the direction check."""
            points = points_in.clone().requires_grad_(True)
            grads, = torch.autograd.grad(critic(points).sum(), points)
            return float(((grads.norm(dim=1) - 1) ** 2).mean())


        torch.manual_seed(1)
        probe = S.Discriminator(d_in=64, d_hidden=128)
        penalty_mix_one = float(gradient_penalty(probe, real_batch, fake_batch, mix=torch.ones(128, 1))[0])
        penalty_mix_zero = float(gradient_penalty(probe, real_batch, fake_batch, mix=torch.zeros(128, 1))[0])
        penalty_random_mix = float(gradient_penalty(probe, real_batch, fake_batch)[0])
        direct_real, direct_fake = penalty_on_one_set(probe, real_batch), penalty_on_one_set(probe, fake_batch)
        direction_check = {"mix_one": penalty_mix_one, "real_alone": direct_real, "mix_zero": penalty_mix_zero, "fake_alone": direct_fake,
                           "random_mix": penalty_random_mix,
                           "diff_mix_one": abs(penalty_mix_one - direct_real), "diff_mix_zero": abs(penalty_mix_zero - direct_fake)}
        direction_check["pass"] = bool(direction_check["diff_mix_one"] < 1e-6 and direction_check["diff_mix_zero"] < 1e-6
                                       and abs(penalty_random_mix - direct_real) > 1e-6 and abs(penalty_random_mix - direct_fake) > 1e-6)
        check_gradient_penalty["direction"] = direction_check
        print(f"direction check, nonlinear probe critic: mix=1 {penalty_mix_one:.8f} vs real alone {direct_real:.8f} (diff {direction_check['diff_mix_one']:.1e}); "
              f"mix=0 {penalty_mix_zero:.8f} vs fake alone {direct_fake:.8f} (diff {direction_check['diff_mix_zero']:.1e}); random mix {penalty_random_mix:.8f}; pass: {direction_check['pass']}")
        results["check_gradient_penalty"] = check_gradient_penalty

    gan_models = {}
    results["gan_digits"], results["wgan_logs"], results["standard_logs"] = {}, {}, {}
    print(f"{'kind':10s} {'seed':>4s} {'Frechet':>8s} {'precision':>10s} {'recall':>8s} {'digits':>7s} {'nn mean':>8s} {'train s':>8s}")
    for kind in kinds:
        for seed in SEEDS:
            generator, history, train_seconds = train_gan(kind, seed)
            gan_models[(kind, seed)] = generator
            scores = score_images(sample_gan(generator))
            scores["train_seconds"] = train_seconds
            scores["sample_seconds"] = time_draw(lambda: sample_gan(generator))
            scores["network_evaluations"] = 1
            results["gan_digits"][f"{kind}_{seed}"] = scores
            thin = lambda values: [float(v) for v in values[::10]]
            if kind == "wgan_gp":
                results["wgan_logs"][str(seed)] = {
                    "critic_norm_first100": float(np.mean(history["critic_norm"][:100])), "critic_norm_last500": float(np.mean(history["critic_norm"][-500:])),
                    "score_gap_first100": float(np.mean(history["score_gap"][:100])), "score_gap_last500": float(np.mean(history["score_gap"][-500:])),
                    "critic_norm_first100_updates": float(np.mean(history["critic_norm_per_update"][:100])),
                    "critic_norm_last500_updates": float(np.mean(history["critic_norm_per_update"][-500:])),
                    "score_gap_first100_updates": float(np.mean(history["score_gap_per_update"][:100])),
                    "score_gap_last500_updates": float(np.mean(history["score_gap_per_update"][-500:])),
                    "critic_norm_every10": thin(history["critic_norm"]), "score_gap_every10": thin(history["score_gap"]),
                    "g_loss_every10": thin(history["g_loss"])}
            else:
                results["standard_logs"][str(seed)] = {
                    "d_real_prob_last500": float(np.mean(history["d_real_prob"][-500:])), "d_fake_prob_last500": float(np.mean(history["d_fake_prob"][-500:])),
                    "d_loss_last500": float(np.mean(history["d_loss"][-500:])), "g_loss_last500": float(np.mean(history["g_loss"][-500:])),
                    "d_real_prob_every10": thin(history["d_real_prob"]), "d_fake_prob_every10": thin(history["d_fake_prob"]),
                    "d_loss_every10": thin(history["d_loss"]), "g_loss_every10": thin(history["g_loss"])}
            print(f"{kind:10s} {seed:4d} {scores['frechet']:8.2f} {scores['precision']:10.3f} {scores['recall']:8.3f} "
                  f"{scores['digits_present']:7d} {scores['nearest_train']:8.3f} {train_seconds:8.1f}")
            if not np.isfinite(scores["frechet"]):
                print("   NaN or infinite score in this run: reported, not rerun")

    print("\ndigit counts per run (digits 0 to 9):")
    for key, scores in results["gan_digits"].items():
        print(f"{key:12s}", scores["digit_counts"], " tv", f"{scores['tv']:.3f}")
    print(f"{'test set':12s}", test_digit_counts)

    if "wgan_gp" in kinds:
        print("\nWGAN-GP windows (mean critic gradient norm on interpolates; mean score gap real minus generated)")
        print(f"{'seed':>4s} {'unit':>14s} {'norm first100':>14s} {'norm last500':>13s} {'gap first100':>13s} {'gap last500':>12s}")
        for seed in SEEDS:
            log = results["wgan_logs"][str(seed)]
            print(f"{seed:4d} {'generator step':>14s} {log['critic_norm_first100']:14.4f} {log['critic_norm_last500']:13.4f} {log['score_gap_first100']:13.4f} {log['score_gap_last500']:12.4f}")
            print(f"{seed:4d} {'critic update':>14s} {log['critic_norm_first100_updates']:14.4f} {log['critic_norm_last500_updates']:13.4f} {log['score_gap_first100_updates']:13.4f} {log['score_gap_last500_updates']:12.4f}")
    if "standard" in kinds:
        print("\nstandard GAN, last 500 steps: mean D(real), mean D(fake), D loss, G loss")
        for seed in SEEDS:
            log = results["standard_logs"][str(seed)]
            print(f"{seed:4d} {log['d_real_prob_last500']:.3f} {log['d_fake_prob_last500']:.3f} {log['d_loss_last500']:.3f} {log['g_loss_last500']:.3f}")

    for kind, title, file_name in [("standard", "Standard GAN (non-saturating), rows: seeds 0, 1, 2", "figures/gan_standard_samples.png"),
                                   ("wgan_gp", "WGAN-GP, rows: seeds 0, 1, 2", "figures/gan_wgangp_samples.png")]:
        if kind in kinds:
            S.save_grid(torch.cat([sample_gan(gan_models[(kind, seed)])[:10] for seed in SEEDS]), file_name, ncols=10, title=title)
    return gan_models


def piece3(results, digits_only=False):
    print("\n=== piece 3: forward process, ring, digits ===")
    if not digits_only:
        print("alpha_bar[9, 10, 49, 50, 98, 99] =", [round(float(ALPHA_BARS[i]), 6) for i in (9, 10, 49, 50, 98, 99)])
        torch.manual_seed(0)
        start_values = torch.full((100_000, 1), 1.5, dtype=torch.float64)
        x = start_values.clone()
        chain = {}
        for t in range(100):
            x = ALPHAS[t].double().sqrt() * x + BETAS[t].double().sqrt() * torch.randn_like(x)
            applied = t + 1
            if applied in (10, 11, 50, 51, 99, 100):
                chain[applied] = (float(x.mean()), float(x.std()))

        forward_rows = []
        print(f"{'convention':10s} {'label':>10s} {'n':>4s} {'chain mean':>11s} {'chain std':>10s} {'closed mean':>12s} {'closed std':>11s} {'exact mean':>11s} {'exact std':>10s} {'wrong-alpha mean':>17s} {'wrong std':>10s} ok")
        for convention, label, applied, index in [("literal", "10 steps", 10, 9), ("literal", "50 steps", 50, 49), ("literal", "99 steps", 99, 98),
                                                  ("index", "t = 10", 11, 10), ("index", "t = 50", 51, 50), ("index", "t = 99", 100, 99)]:
            full_index = torch.full((100_000,), index, dtype=torch.long)
            closed = q_sample(start_values, full_index, torch.randn_like(start_values))
            exact_mean, exact_std = 1.5 * float(ALPHA_BARS[index].sqrt()), float((1 - ALPHA_BARS[index]).sqrt())
            wrong = ALPHAS[index].double().sqrt() * start_values + (1 - ALPHAS[index].double()).sqrt() * torch.randn_like(start_values)   # per-step alpha instead of the cumulative product
            chain_mean, chain_std = chain[applied]
            closed_mean, closed_std = float(closed.mean()), float(closed.std())
            worst = max(abs(chain_mean - exact_mean), abs(chain_std - exact_std), abs(closed_mean - exact_mean), abs(closed_std - exact_std),
                        abs(chain_mean - closed_mean), abs(chain_std - closed_std))
            wrong_mean, wrong_std = float(wrong.mean()), float(wrong.std())
            wrong_fails = max(abs(wrong_mean - exact_mean), abs(wrong_std - exact_std)) > 0.015
            forward_rows.append({"convention": convention, "label": label, "applied_steps": applied, "alpha_bar_index": index,
                                 "chain_mean": chain_mean, "chain_std": chain_std, "closed_mean": closed_mean, "closed_std": closed_std,
                                 "exact_mean": exact_mean, "exact_std": exact_std, "wrong_alpha_mean": wrong_mean, "wrong_alpha_std": wrong_std,
                                 "worst_deviation": worst, "pass": bool(worst < 0.015), "wrong_alpha_fails": bool(wrong_fails)})
            print(f"{convention:10s} {label:>10s} {applied:4d} {chain_mean:11.4f} {chain_std:10.4f} {closed_mean:12.4f} {closed_std:11.4f} {exact_mean:11.4f} {exact_std:10.4f} "
                  f"{wrong_mean:17.4f} {wrong_std:10.4f} pass {worst < 0.015} (worst {worst:.4f}); wrong-alpha fails: {wrong_fails}")
        results["check_forward_process"] = forward_rows
        results["alpha_bar_last_step"] = float(ALPHA_BARS[-1])
        results["signal_amplitude_last_step"] = float(ALPHA_BARS[-1].sqrt())
        print(f"alpha_bar at the last step {float(ALPHA_BARS[-1]):.6f}; share of the original signal's amplitude left {float(ALPHA_BARS[-1].sqrt()):.4f}")

        results["ring"] = {}
        results["ring_real_reference"] = dict(zip(("modes_covered", "share_near_mode"), S.mode_coverage(S.ring_of_gaussians(2000)[0])))
        print("real ring data (2,000 points): modes covered, share near a mode =", results["ring_real_reference"])
        ring_samples = {}
        print(f"{'schedule':10s} {'seed':>4s} {'modes':>6s} {'near-mode share':>16s} {'loss first100':>14s} {'loss last200':>13s} {'train s':>8s}")
        for label, cosine in [("constant", False), ("cosine", True)]:
            for seed in SEEDS:
                model, losses, train_seconds, learning_rates = train_diffusion(ring_points, RING_HIDDEN, RING_LR, seed, cosine=cosine)
                samples = sample_ancestral(model, 2000, 2)
                modes, share = S.mode_coverage(samples)
                ring_samples[(label, seed)] = (samples, modes, share)
                results["ring"][f"{label}_{seed}"] = {"modes_covered": modes, "share_near_mode": share, "loss_first100": float(np.mean(losses[:100])),
                                                      "loss_last200": float(np.mean(losses[-200:])), "train_seconds": train_seconds,
                                                      "extra_not_required": cosine}
                print(f"{label:10s} {seed:4d} {modes:6d} {share:16.3f} {np.mean(losses[:100]):14.4f} {np.mean(losses[-200:]):13.4f} {train_seconds:8.1f}")
                if cosine and seed == 0:
                    print(f"   cosine learning rate at steps 0, 2000, 3999: {learning_rates[0]:.2e}, {learning_rates[2000]:.2e}, {learning_rates[3999]:.2e}")

        figure, axes = plt.subplots(1, 4, figsize=(14, 3.8))
        real_ring = S.ring_of_gaussians(2000)[0]
        axes[0].scatter(real_ring[:, 0], real_ring[:, 1], s=2)
        axes[0].set_title(f"real ring (2,000 points)\nmodes {results['ring_real_reference']['modes_covered']}, near-mode share {results['ring_real_reference']['share_near_mode']:.3f}", fontsize=9)
        for column, seed in enumerate(SEEDS, start=1):
            samples, modes, share = ring_samples[("constant", seed)]
            axes[column].scatter(samples[:, 0], samples[:, 1], s=2)
            axes[column].set_title(f"diffusion, seed {seed}\nmodes {modes}, near-mode share {share:.3f}", fontsize=9)
        for axis in axes:
            axis.set_xlim(-3, 3); axis.set_ylim(-3, 3); axis.set_aspect("equal")
        figure.savefig("figures/ring_diffusion_samples.png", dpi=150, bbox_inches="tight")
        plt.close(figure)

    digits_scaled = Xtr * 2 - 1                      # [0, 1] -> [-1, 1] for training
    diffusion_models = {}
    results["diffusion_digits"] = {}
    print(f"{'seed':>4s} {'Frechet':>8s} {'precision':>10s} {'recall':>8s} {'digits':>7s} {'nn mean':>8s} {'nn min':>7s} {'TV':>6s} {'loss first100':>14s} {'loss last200':>13s} {'clipped':>8s} {'train s':>8s}")
    for seed in SEEDS:
        model, losses, train_seconds, learning_rates = train_diffusion(digits_scaled, DIGIT_HIDDEN, DIGIT_LR, seed, cosine=True)
        diffusion_models[seed] = model
        raw = sample_ancestral(model, N_SAMPLES, 64)
        scores = score_images(to_unit_range(raw))
        scores.update({"loss_first100": float(np.mean(losses[:100])), "loss_last200": float(np.mean(losses[-200:])), "train_seconds": train_seconds,
                       "sample_seconds": time_draw(lambda: sample_ancestral(model, N_SAMPLES, 64)), "network_evaluations": 100,
                       "clipped_share": clipped_share(raw), "final_learning_rate": learning_rates[-1]})
        results["diffusion_digits"][str(seed)] = scores
        print(f"{seed:4d} {scores['frechet']:8.2f} {scores['precision']:10.3f} {scores['recall']:8.3f} {scores['digits_present']:7d} {scores['nearest_train']:8.3f} "
              f"{scores['nearest_train_min']:7.3f} {scores['tv']:6.3f} {scores['loss_first100']:14.4f} {scores['loss_last200']:13.4f} {scores['clipped_share']:8.3f} {train_seconds:8.1f}")
        if seed == 0:
            print(f"   cosine learning rate at steps 0, 2000, 3999: {learning_rates[0]:.2e}, {learning_rates[2000]:.2e}, {learning_rates[3999]:.2e}")
            wrong_scores = S.score_samples(raw, test_features, feature_net, Xtr)        # wrong on purpose: scoring in [-1, 1] without rescaling
            results["unscaled_trap_seed0"] = wrong_scores
            print("   wrong on purpose, seed 0 scored without rescaling to [0, 1]:", {k: round(v, 3) for k, v in wrong_scores.items()})
    print(f"share of the original signal's amplitude left at the last step: {float(ALPHA_BARS[-1].sqrt()):.4f}")
    print("\ndigit counts per seed:")
    for seed in SEEDS:
        print(seed, results["diffusion_digits"][str(seed)]["digit_counts"])
    print("test", test_digit_counts)
    S.save_grid(torch.cat([to_unit_range(sample_ancestral(diffusion_models[seed], N_SAMPLES, 64))[:10] for seed in SEEDS]),
                "figures/diffusion_digits_samples.png", ncols=10, title="Diffusion (ancestral, 100 steps), rows: seeds 0, 1, 2")
    return diffusion_models


def piece4(results, diffusion_models, dep_results):
    print("\n=== piece 4: steps against sampler ===")
    results["deterministic_timesteps"] = {str(k): deterministic_timesteps(k).tolist() for k in DETERMINISTIC_KS}
    for k in DETERMINISTIC_KS:
        schedule = results["deterministic_timesteps"][str(k)]
        print(f"k = {k:3d}: {schedule if k <= 25 else str(schedule[:6])[:-1] + ', ..., ' + str(schedule[-3:])[1:]}  ({len(set(schedule))} distinct)")

    # sanity checks on the sampler
    final_x, final_x0 = sample_deterministic(diffusion_models[0], N_SAMPLES, 64, 100, return_last_x0=True)
    repeat_x = sample_deterministic(diffusion_models[0], N_SAMPLES, 64, 100)
    results["check_deterministic_sampler"] = {"last_step_x_minus_x0_max_abs": float((final_x - final_x0).abs().max()),
                                              "repeat_max_abs_diff": float((final_x - repeat_x).abs().max())}
    print("k = 100: max |x after last step - x0 estimate| =", results["check_deterministic_sampler"]["last_step_x_minus_x0_max_abs"],
          "| two calls, max abs difference =", results["check_deterministic_sampler"]["repeat_max_abs_diff"])

    results["steps_vs_sampler"] = {}
    for seed in SEEDS:
        model = diffusion_models[seed]
        seed_row = {"ancestral": {k: v for k, v in dep_results["diffusion_digits"][str(seed)].items() if k not in ("loss_first100", "loss_last200", "train_seconds", "final_learning_rate")}}
        for k in DETERMINISTIC_KS:
            raw = sample_deterministic(model, N_SAMPLES, 64, k)
            scores = score_images(to_unit_range(raw))
            scores.update({"clipped_share": clipped_share(raw), "network_evaluations": k,
                           "sample_seconds": time_draw(lambda: sample_deterministic(model, N_SAMPLES, 64, k))})
            seed_row[f"deterministic_{k}"] = scores
        results["steps_vs_sampler"][str(seed)] = seed_row

    print(f"\n{'seed':>4s} {'sampler':>14s} {'evals':>6s} {'Frechet':>8s} {'precision':>10s} {'recall':>8s} {'clipped':>8s} {'sample s':>9s}")
    for seed in SEEDS:
        for name, scores in results["steps_vs_sampler"][str(seed)].items():
            print(f"{seed:4d} {name:>14s} {scores['network_evaluations']:6d} {scores['frechet']:8.2f} {scores['precision']:10.3f} {scores['recall']:8.3f} {scores['clipped_share']:8.3f} {scores['sample_seconds']:9.4f}")

    # attribution: total = sampler part + step part, exactly
    results["steps_attribution"] = {}
    for metric in ("recall", "frechet"):
        print(f"\nattribution of the {metric} gap (deterministic k minus ancestral 100): total = sampler part + step part")
        print(f"{'seed':>5s} {'k':>4s} {'total':>9s} {'sampler':>9s} {'step':>9s}")
        for seed in SEEDS + ["mean"]:
            for k in DETERMINISTIC_KS:
                if seed == "mean":
                    rows = [results["steps_attribution"][metric][str(s)][str(k)] for s in SEEDS]
                    entry = {part: float(np.mean([row[part] for row in rows])) for part in ("total", "sampler", "step")}
                    results["steps_attribution"][metric].setdefault("mean", {})[str(k)] = entry
                else:
                    ancestral_value = results["steps_vs_sampler"][str(seed)]["ancestral"][metric]
                    det_100_value = results["steps_vs_sampler"][str(seed)]["deterministic_100"][metric]
                    det_k_value = results["steps_vs_sampler"][str(seed)][f"deterministic_{k}"][metric]
                    entry = {"total": det_k_value - ancestral_value, "sampler": det_100_value - ancestral_value, "step": det_k_value - det_100_value}
                    assert abs(entry["total"] - entry["sampler"] - entry["step"]) < 1e-12
                    results["steps_attribution"].setdefault(metric, {}).setdefault(str(seed), {})[str(k)] = entry
                print(f"{str(seed):>5s} {k:4d} {entry['total']:9.4f} {entry['sampler']:9.4f} {entry['step']:9.4f}")

    # sampling-noise spread: ancestral and deterministic k = 100, sample seeds 9, 10, 11
    results["sampling_noise_spread"] = {}
    print(f"\nsampling-noise spread over sample seeds 9, 10, 11 (range = max - min)")
    print(f"{'seed':>4s} {'sampler':>14s} {'recall values':>26s} {'recall range':>13s} {'Frechet values':>26s} {'Frechet range':>14s}")
    for seed in SEEDS:
        model = diffusion_models[seed]
        for name, draw in [("ancestral", lambda s: sample_ancestral(model, N_SAMPLES, 64, seed=s)),
                           ("deterministic_100", lambda s: sample_deterministic(model, N_SAMPLES, 64, 100, seed=s))]:
            scored = [score_images(to_unit_range(draw(sample_seed))) for sample_seed in (9, 10, 11)]
            recalls, frechets = [s["recall"] for s in scored], [s["frechet"] for s in scored]
            results["sampling_noise_spread"][f"{name}_{seed}"] = {"recall": recalls, "frechet": frechets,
                                                                  "recall_range": max(recalls) - min(recalls), "frechet_range": max(frechets) - min(frechets)}
            print(f"{seed:4d} {name:>14s} {str([round(r, 3) for r in recalls]):>26s} {max(recalls) - min(recalls):13.3f} {str([round(f, 2) for f in frechets]):>26s} {max(frechets) - min(frechets):14.2f}")

    # figure
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, label in [(axes[0], "recall", "Recall"), (axes[1], "frechet", "Frechet distance")]:
        for seed in SEEDS:
            color = f"C{seed}"
            row = results["steps_vs_sampler"][str(seed)]
            axis.plot(DETERMINISTIC_KS, [row[f"deterministic_{k}"][metric] for k in DETERMINISTIC_KS], marker="o", color=color, label=f"seed {seed}, deterministic")
            axis.scatter([100], [row["ancestral"][metric]], marker="o", facecolors="none", edgecolors=color, s=90, zorder=3)
            axis.axhline(row["ancestral"][metric], color=color, linestyle="--", linewidth=1, label=f"seed {seed}, ancestral (100)")
        axis.set_xscale("log"); axis.set_xticks(DETERMINISTIC_KS); axis.set_xticklabels([str(k) for k in DETERMINISTIC_KS])
        axis.minorticks_off(); axis.set_xlabel("network evaluations k"); axis.set_ylabel(label); axis.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    figure.suptitle("Deterministic sampler (k evaluations) vs ancestral (100 evaluations)")
    figure.savefig("figures/steps_vs_sampler.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def piece5(results, gan_models, diffusion_models, dep_results):
    print("\n=== piece 5: three families, one scoring ===")
    results["three_families"] = {}
    families = {}
    for seed in SEEDS:
        gan_scores = dep_results["gan_digits"][f"wgan_gp_{seed}"]
        families[f"wgan_gp_{seed}"] = dict(gan_scores)
        diffusion_scores = dep_results["diffusion_digits"][str(seed)]
        families[f"diffusion_{seed}"] = {k: v for k, v in diffusion_scores.items() if k not in ("loss_first100", "loss_last200", "final_learning_rate")}
        torch.manual_seed(seed)                                   # BEFORE constructing the VAE, so the seed sets the initial weights
        vae = S.VAE()
        start = time.perf_counter()
        S.train_vae(vae, Xtr, seed=seed)
        vae_train_seconds = time.perf_counter() - start
        vae_scores = score_images(vae.sample(N_SAMPLES))          # decoded probabilities, sample seed 9 inside
        vae_scores.update({"train_seconds": vae_train_seconds, "sample_seconds": time_draw(lambda: vae.sample(N_SAMPLES)), "network_evaluations": 1})
        families[f"vae_{seed}"] = vae_scores
        if seed == 0:
            vae_samples_seed0 = vae.sample(N_SAMPLES)
    results["three_families"] = families

    print(f"torch threads: {torch.get_num_threads()}")
    columns = f"{'family':10s} {'seed':>5s} {'Frechet':>8s} {'precision':>10s} {'recall':>8s} {'nn mean':>8s} {'nn min':>7s} {'TV':>6s} {'train s':>8s} {'evals':>6s} {'sample s':>9s}"
    print(columns)
    for family in ("vae", "wgan_gp", "diffusion"):
        rows = [families[f"{family}_{seed}"] for seed in SEEDS]
        for seed, scores in zip(SEEDS, rows):
            print(f"{family:10s} {seed:5d} {scores['frechet']:8.2f} {scores['precision']:10.3f} {scores['recall']:8.3f} {scores['nearest_train']:8.3f} {scores['nearest_train_min']:7.3f} "
                  f"{scores['tv']:6.3f} {scores['train_seconds']:8.1f} {scores['network_evaluations']:6d} {scores['sample_seconds']:9.4f}")
        mean_std = {key: (float(np.mean([r[key] for r in rows])), float(np.std([r[key] for r in rows], ddof=1)))
                    for key in ("frechet", "precision", "recall", "nearest_train", "nearest_train_min", "tv", "train_seconds", "sample_seconds")}
        results["three_families"][f"{family}_mean_std_ddof1"] = mean_std
        print(f"{family:10s} {'mean':>5s} {mean_std['frechet'][0]:8.2f} {mean_std['precision'][0]:10.3f} {mean_std['recall'][0]:8.3f} {mean_std['nearest_train'][0]:8.3f} "
              f"{mean_std['nearest_train_min'][0]:7.3f} {mean_std['tv'][0]:6.3f} {mean_std['train_seconds'][0]:8.1f} {'':6s} {mean_std['sample_seconds'][0]:9.4f}")
        print(f"{'':10s} {'std':>5s} {mean_std['frechet'][1]:8.2f} {mean_std['precision'][1]:10.3f} {mean_std['recall'][1]:8.3f} {mean_std['nearest_train'][1]:8.3f} "
              f"{mean_std['nearest_train_min'][1]:7.3f} {mean_std['tv'][1]:6.3f} {mean_std['train_seconds'][1]:8.1f} {'':6s} {mean_std['sample_seconds'][1]:9.4f}   (std uses ddof = 1)")
    print(f"\nreference: real test images to nearest training image, mean {results['reference']['test_nearest_train_mean']:.3f}, min {results['reference']['test_nearest_train_min']:.3f}")
    reference_scores = results['reference']['train_subset_score']
    print(f"reference: 359 real training images scored as generated: Frechet {reference_scores['frechet']:.2f}, precision {reference_scores['precision']:.3f}, recall {reference_scores['recall']:.3f}, TV {reference_scores['tv']:.3f}, nearest-train min {reference_scores['nearest_train_min']:.3f}")
    print("digit counts, seed 0:")
    for family in ("vae", "wgan_gp", "diffusion"):
        print(f"{family:10s}", families[f"{family}_0"]["digit_counts"])
    print(f"{'test set':10s}", results["reference"]["test_digit_counts"])

    diffusion_seed0 = to_unit_range(sample_ancestral(diffusion_models[0], N_SAMPLES, 64))
    S.save_grid(torch.cat([vae_samples_seed0[:10], sample_gan(gan_models[("wgan_gp", 0)])[:10], diffusion_seed0[:10]]),
                "figures/three_families_samples.png", ncols=10, title="Rows: VAE, WGAN-GP, diffusion (seed 0)")


def save_results(results, out_path):
    results["total_seconds"] = time.perf_counter() - run_start
    results["environment"] = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                              "threads": torch.get_num_threads(), "device": DEVICE}
    results["settings"] = {name: globals()[name] for name in ["SEEDS", "SAMPLE_SEED", "N_SAMPLES", "DEVICE", "BATCH_SIZE", "GAN_STEPS", "GAN_BETAS",
                                                               "STANDARD_LR_G", "STANDARD_LR_D", "WGAN_LR", "WGAN_CRITIC_STEPS", "GP_WEIGHT",
                                                               "DIFFUSION_STEPS", "RING_HIDDEN", "RING_LR", "DIGIT_HIDDEN", "DIGIT_LR", "T_STEPS", "DETERMINISTIC_KS"]}


    def json_default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        raise TypeError(f"cannot serialise {type(value)}")


    with open(out_path, "w") as results_file:
        json.dump(results, results_file, indent=1, default=json_default)
    print(f"saved {out_path}; whole notebook took {results['total_seconds'] / 60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description="Lab 08: Two Ways to Generate")
    parser.add_argument("--pieces", type=int, nargs="+", default=[1, 2, 3, 4, 5], choices=[1, 2, 3, 4, 5])
    parser.add_argument("--out", default=None, help="results json; written when all five pieces run, or when given explicitly")
    parser.add_argument("--run-starter", action="store_true", help="also run lab08_starter.py in a subprocess")
    args = parser.parse_args()
    pieces = sorted(set(args.pieces))
    if pieces == [1, 2, 3, 4, 5] or args.run_starter:
        run_starter()
    gan_models, diffusion_models = None, None
    scratch = {}                  # holds scores of any model trained only as a dependency
    if 1 in pieces:
        piece1(results)
    if 2 in pieces:
        gan_models = piece2(results)
    if 3 in pieces:
        diffusion_models = piece3(results)
    if 4 in pieces or 5 in pieces:
        if 4 in pieces and 5 not in pieces and diffusion_models is None:
            print("piece 4 needs the digit diffusion models: training them first (piece 3, digits only)")
            diffusion_models = piece3(scratch, digits_only=True)
        if 5 in pieces and gan_models is None:
            print("piece 5 needs the WGAN-GP models: training them first (piece 2, WGAN-GP only)")
            gan_models = piece2(scratch, kinds=("wgan_gp",), with_checks=False)
        if 5 in pieces and diffusion_models is None:
            print("piece 5 needs the digit diffusion models: training them first (piece 3, digits only)")
            diffusion_models = piece3(scratch, digits_only=True)
        dep_results = {key: value for key, value in scratch.items() if key in ("gan_digits", "diffusion_digits")}
        dep_results.update({key: results[key] for key in ("gan_digits", "diffusion_digits") if key in results})
        if 4 in pieces:
            piece4(results, diffusion_models, dep_results)
        if 5 in pieces:
            piece5(results, gan_models, diffusion_models, dep_results)
    out_path = args.out or ("lab8_results.json" if pieces == [1, 2, 3, 4, 5] else None)
    if out_path:
        save_results(results, out_path)


if __name__ == "__main__":
    main()
