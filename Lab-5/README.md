# Attention Against Recurrence

Attention built from scratch, then tested against recurrence, positions and normalization on small tasks with known answers.

## Prediction (written before running the mixed-length training)

Trained on 16 only, learned got 0.603 to 0.623 at length 32 and sinusoidal got 0.629 to 0.748.

**Learned:** a small improvement at 32, around 0.7 to 0.8. Lengths 8, 16 and 24 only train rows 0 to 23, so rows 24 to 31 stay random. About 44% of length-32 sequences have a marker there, and I expect those to be close to a coin flip. Lower again at 48.

**Sinusoidal:** a bigger improvement at 32, possibly above 0.9. Positions 24 to 31 still have defined values that follow the same pattern as the trained ones. Lower at 48, since those positions are further from training.
