# Latent Spaces

A VAE written from scratch on 8 by 8 digits, compared with a plain autoencoder, varied in the weight on the KL term, probed, and measured.

## Conventions

The reconstruction term is the Bernoulli negative log-likelihood summed over the 64 pixels, and the KL term is summed over the 8 code dimensions, so each is one number per image. The loss is the mean over the images in the batch of `reconstruction + beta * KL`. Every experiment uses this.

## Prediction (written before running the sweep)

What I already know from the beta 1 runs: on all three seeds the VAE uses 5 of its 8 dimensions (the other three have a posterior-mean std of 0.02 to 0.04), the KL on the test images is about 3.5 nats, and the prior samples have a Fréchet distance of 13 to 15. A fully collapsed model (the detached-sample run) has a loss of about 27, against about 24 for the working one, so the code only buys a few nats of reconstruction.

**Active dimensions:** 8 or 7 at beta 0.1, about 6 at 0.5, 5 at 1, 2 or 3 at 2, and 0 or 1 at 4. Each dimension has to save more than beta times its KL in reconstruction to stay on, and the first dimensions (which digit, how thick) save a lot more than the later ones, so they switch off one at a time as beta goes up.

**Collapse:** at beta 4, on at least one seed. Since the whole code only saves about 6 to 7 nats of reconstruction at beta 1, I don't think there's enough left to pay four times the KL. I expect a collapsed model to put out the same blurry average digit for every draw.

**Best samples:** beta 1, with 0.5 close. Below that the KL is too weak to make the codes fill N(0, I), so prior draws land in gaps like they did for the autoencoder and precision drops. Above 1 the samples get blurrier and lose the less common digits, so recall drops before precision does.
