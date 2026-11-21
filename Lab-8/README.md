# Two Ways to Generate

A GAN and a diffusion model written from scratch on the starter's 8 by 8 digits, each checked against something I can work out exactly, then compared with each other and with a VAE.

`lab08_starter.py` is the course starter, unchanged. It supplies the data (1,438 training and 359 test images), the networks, the noise schedule, the VAE and the sample scoring. Every trained model uses seeds 0, 1 and 2. Every Fréchet distance, precision and recall uses the same feature network and 359 generated images, and every evaluation draw starts from sample seed 9, so runs I compare start from the same noise.

## Run

```bash
pip install -r requirements.txt
python lab08_starter.py          # check the unmodified starter runs
python Lab8.py                   # all five pieces, the figures and lab8_results.json
python Lab8.py --pieces 2        # or one piece at a time (1 to 5)
```

`Lab8.ipynb` is the same code with its outputs and my notes, in five pieces:

1. The GAN losses, checked against their definitions, the optimal discriminator and the saturation of the original loss.
2. The gradient penalty, then a standard GAN against WGAN-GP.
3. Diffusion: the forward process, the loss and the sampler, checked, then trained on a ring and on the digits.
4. Steps against sampler: how few denoising steps I can use.
5. WGAN-GP, diffusion and the VAE scored the same way.

## Prediction (written before running piece 2)

I wrote this before running anything in this lab except the unmodified starter. What I am going on is the handout, the theory of the two losses, and what I saw in Lab 7: the starter's 30-epoch VAE scores a Fréchet distance of 31.1, precision 0.847 and recall 0.201 on this feature network, so recall for a weak generator on these digits can be low even when the samples look fine. The test set has 359 images with digit counts 27, 35, 36, 29, 30, 40, 44, 39, 38, 41 for digits 0 to 9.

**Higher recall:** WGAN-GP. I expect a recall of about 0.3 to 0.6 for WGAN-GP and about 0.1 to 0.35 for the standard GAN, with the standard GAN's precision at least as high as WGAN-GP's, since a generator that has dropped modes can still be sharp on the ones it kept.

**Digit counts:** for the standard GAN I expect 6 to 9 of the 10 digits to appear, with a very uneven spread, some digit with 80 or more of the 359 samples and one or two with under 10. A few digits that are easy for a small network (1, 0, 7) could dominate. For WGAN-GP I expect all 10 digits to appear, with counts nearer the test mix (roughly 20 to 60 each), though not equal to it.

**Why:** when the discriminator is winning, the non-saturating loss still gives the generator a useful gradient, but the discriminator only says real or fake. A generator can satisfy it by settling on a few digits the discriminator has not yet learned to reject, and the discriminator then chases them one at a time, so modes get dropped. The critic in WGAN-GP gives an unbounded score whose gradient does not vanish when the two sets are apart, and the gradient penalty keeps that score smooth, so every generated point is pushed toward the real data wherever it is. Three critic steps per generator step keep the critic close to its optimum. That should cover more of the data, at the cost of less sharp samples (a lower precision) early on.

**Stability and gradient behaviour:** I expect the critic's gradient norm on the interpolated points to start well away from 1 (somewhere above 1, maybe 1 to 3, because the critic is untrained) and settle close to 1 by the end of the run (0.8 to 1.2), since that is what the penalty pulls it toward. I expect the score gap (real minus generated) to be large early, several units, while the generator is poor, and to shrink to about 1 to 3 by the last 500 steps as the generator improves. I expect all three standard GAN runs to train without a crash, but one of the three seeds to look clearly worse than the others. I do not expect a WGAN-GP collapse.

**Piece 4:** I expect the deterministic sampler to lose recall as k drops, with 5 steps clearly worse than 100, and with the deterministic sampler at 100 steps about as good as the ancestral one, so most of any gap would come from the number of steps rather than the kind of sampler.
