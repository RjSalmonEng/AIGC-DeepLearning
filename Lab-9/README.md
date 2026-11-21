# Training an RL Agent

DQN, two broken versions of it, REINFORCE and PPO, written from scratch on CartPole with the same budget and three seeds each.

The task is the course CartPole. The state is four numbers (cart position and velocity, pole angle and angular velocity) and there are two actions (push left or right). Every step pays +1. An episode is terminated when the pole falls past 12 degrees or the cart leaves the track, and truncated when it reaches 200 steps, so the best possible return is 200. Those are different events, and the difference matters for the DQN target.

`lab09_starter.py` is the course starter, unchanged: the environment, the networks, the probe states and the curve helpers. Every agent gets exactly 20,000 environment steps on seeds 0, 1 and 2, and every learning curve, final return and "steps to 150" comes from the starter's `smooth_curve`, so all agents are measured the same way.

## Run

```bash
pip install -r requirements.txt
python lab09_starter.py          # check the unmodified starter runs
python Lab9.py                   # all five pieces, the figures and lab9_results.json
python Lab9.py checks            # or one piece: checks, dqn, ablations, policy_gradient, compare
```

`Lab9.ipynb` is the same code with its outputs and my notes, in five pieces:

1. The losses (DQN target, Huber, PPO clip, return-to-go, GAE), each checked against numbers I worked out by hand, plus a fixed-point test of the target network.
2. DQN on three seeds.
3. DQN with no target network, with no replay, and (extra) with a buffer of only 1,000.
4. REINFORCE with a baseline and PPO on the same budget.
5. All six agents compared under a rule I wrote down before looking at the table.

## Prediction (written before running piece 3)

I had already run pieces 1 and 2 (the checks and full DQN) when I wrote this, and I had not run any of the ablations. I had also test-run the policy-gradient code in a scratch copy to make sure it works, which does not bear on these predictions. Full DQN gave final returns of 88.7, 145.6 and 88.3 on seeds 0, 1 and 2. Only seed 2 reached a smoothed return of 150, at step 7,500 (seeds 0 and 1 never did), the mean over the probe states of the largest action value was 54.9, 54.7 and 55.8 at step 20,000, and the largest absolute Q I ever saw was 58.4, 58.1 and 61.1, all under the bound of 100.

**No target network.** I expect it to hurt a lot. I predict final returns of about 10 to 60 on every seed (I would not be surprised by a collapse to random-policy level, about 22), and that no seed reaches a smoothed return of 150. I predict the Q-values blow up: the mean largest action value and the largest absolute Q both pass 100 on all three seeds, probably between step 3,000 and step 10,000, and the largest absolute Q ever reaches somewhere between 1e3 and 1e6. The reason is that every update moves its own target. With weights shared across states, raising Q for one state raises Q for its neighbours, including the next states used in the target, and the max over actions picks whichever estimate is highest, so errors compound upward. Huber loss and the gradient clip at 10 limit how big one step is but not how far the drift can go over thousands of steps. I know the one-state fixed-point check shows no target network behaving fine, but that has no generalisation across states, which is what I think drives the blow-up.

**No replay (latest 64 transitions, target network kept).** I expect it to hurt, but less than removing the target network. I predict final returns of about 40 to 100, so on average maybe 30 to 60 points below full DQN, and "never" for steps to 150 on at least two of the three seeds. I predict the Q-values stay under 100 (mean largest action value somewhere between 30 and 70, largest absolute Q ever between 40 and 90), because the target network is still there and the bootstrapped values stay anchored, but I expect them to be noisier than for full DQN. The reason is that 64 consecutive steps are nearly the same trajectory, so each update fits one short stretch of the state space and undoes some of what the previous one learned. Each transition is still reused about 64 times, so I do not think the cost is lack of reuse, I think it is correlated, narrow data. I am less sure about this one than about the no-target run.

**Optional, a buffer of only 1,000 transitions.** I predict it lands close to full DQN, with final returns of about 60 to 130 and Q-values under 100, because 1,000 random samples from the last 1,000 steps are much less correlated than 64 consecutive ones and the target network is still there. I expect it to be a bit worse than full DQN because it forgets old states, for example the early part of an episode that it has learned to survive.

## Decision rule (written before looking at the table)

I count two configurations as different on a measure only if their three per-seed values do not overlap: every seed of one is strictly better than every seed of the other (for final return, the lowest of one above the highest of the other; for steps to 150 and wall-clock time, the slowest of one below the fastest of the other). A seed that never reaches 150 counts as slower than any seed that does; two seeds that never reach it are tied. Equal endpoints count as overlap. Anything else is "cannot tell". I apply it to all 15 pairs of the six agents on final return, steps to 150 and wall-clock time, and to the 6 pairs of DQN configurations on the largest |Q| ever seen. I say a configuration exceeds the bound of 100 only if all three seeds do. I will not change this rule after seeing the table.

## Results

Filled in after the runs.
