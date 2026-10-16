# What Recurrence Can and Cannot Remember

Measuring gradients through time and comparing recurrent cells on a delayed-recall task.

## Prediction

I expect clipping to make no difference for the plain RNN at length 30. All three seeds should stay near chance (0.25) with clipping on and with it off.

My gradient measurement shows the plain RNN's gradient 29 steps back is about 3.1e-08, against 5.8 at the last step. The label is 29 steps back from the loss, so almost no gradient reaches it. Clipping only shrinks gradients that are too big. It can't make a vanished gradient bigger, so I don't think turning it off or on will change anything.
