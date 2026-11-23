# Training an RL Agent

DQN, two broken versions of it, REINFORCE and PPO, written from scratch on CartPole with the same budget and three seeds each.

CartPole's state is four numbers (cart position and velocity, pole angle and angular velocity) and there are two actions, push left or push right. Every step pays +1. An episode is terminated when the pole falls past 12 degrees or the cart leaves the track, and truncated when it reaches 200 steps, so the best possible return is 200. Those are different events, and the difference matters for the DQN target. Every agent gets exactly 20,000 environment steps on seeds 0, 1 and 2, and every learning curve, final return and "steps to 150" is read from the same smoothed curve, so all six agents are measured the same way.

## Run

```bash
pip install -r requirements.txt
python Lab9.py
```

`lab09_starter.py` is the course starter, unchanged. It supplies the environment, the networks, the 500 probe states and the curve helpers, and my code imports from it. `Lab9.ipynb` has the code with its outputs and my notes, and `Lab9.py` is the same code as a script. `python Lab9.py dqn` runs one part on its own (`checks`, `dqn`, `ablations`, `policy_gradient` or `compare`). Every number below comes from that run and is also saved in `lab9_results.json`.

## Prediction (written before running the ablations)

Full DQN had finished at 88.7, 145.6 and 88.3, and its largest Q on the probe states was 58 to 61, under the bound of 100.

**No target network:** I expect it to hurt a lot, with final returns of 10 to 60 and no seed reaching 150. I expect Q to blow up past 100 on every seed, somewhere between steps 3,000 and 10,000, and to reach 1e3 to 1e6. Every update moves its own target, and because the weights are shared across states, raising Q for one state raises it for the next states in the target too. The max over actions then picks whichever estimate is highest, so the errors compound upward.

**No replay (latest 64 transitions):** I expect it to hurt, but less, with final returns of 40 to 100 and at least two seeds never reaching 150. Q should stay under 100 because the target network is still there. 64 consecutive steps are nearly the same stretch of one episode, so each update fits a narrow slice of states and undoes some of the last one.

**A buffer of only 1,000:** close to full DQN, with final returns of 60 to 130 and Q under 100, maybe a little worse because it forgets older states.

## Decision rule (written before looking at the comparison)

Two agents count as different on a measure only if their three seeds don't overlap: every seed of one is strictly better than every seed of the other. A seed that never reaches 150 counts as slower than any that does. Anything else is "cannot tell". I apply this to all 15 pairs of agents on final return, steps to 150 and wall-clock time, and to the 4 DQN setups on their largest Q.

## Results

### Checks

| Check | Result |
|---|---|
| DQN target: terminal, non-terminal, time limit, both | 1.0, 7.3, 7.3, 1.0 (only a real terminal stops the bootstrap) |
| Huber loss at 0.5, 3, −3 against PyTorch | 0.125, 2.5, 2.5, exact match |
| Fixed point (true value 10), with a target network | 9.98 after 3,000 updates, first within 0.1 at update 2,101 |
| Fixed point, without a target network | settled at 10.00 by about update 130 |
| PPO clip at ratio 1.3 and 0.7, advantage ±1 | 1.2, −1.3, 0.7, −0.8, exact match |
| Return-to-go, rewards 1, 1, 1 at discount 0.5 | 1.75, 1.5, 1.0 |
| GAE at episode ends | truncated bootstraps, terminated doesn't, nothing leaks across episodes |

![Fixed point with and without a target network](figures/fixed_point.png)

The fixed point without a target network didn't diverge. A single state that bootstraps from itself just oscillates and settles, so this check can't show the instability DQN has across many states. What it does show is the cost of the target network: about 2,100 updates to converge instead of about 130.

### DQN and the ablations

| Setup | Final return (seeds 0 / 1 / 2) | Steps to 150 | Largest Q ever | Above 100? |
|---|---|---|---|---|
| Full DQN | 88.7 / 145.6 / 88.3 | never / never / 7,500 | 58.4 / 58.1 / 61.1 | no |
| No target network | 9.6 / 9.6 / 9.7 | never | 5.68e7 / 7.47e7 / 7.03e7 | yes |
| No replay (latest 64) | 111.7 / 110.5 / 172.6 | 7,500 / never / 7,000 | 60.3 / 63.9 / 59.7 | no |
| Buffer of 1,000 | 100.8 / 59.5 / 176.2 | never / never / 7,500 | 57.6 / 58.0 / 55.6 | no |

![Q on the probe states against the bound of 100](figures/q_probe_vs_bound.png)

Full DQN's Q-values stayed near 55 to 61, under the bound of 100 and under 86.6, the most a 200-step episode is worth. Only the no-target run broke the bound, and it did on every seed.

**Against my prediction:**
- **No target network hurts a lot: right, and by more than I said.** Final returns of 9.6 to 9.7, worse than a random policy (22.2). Q was already past 100 at the first reading at step 1,000 and reached 5.7e7 to 7.5e7, about two orders of magnitude above my top guess.
- **No replay hurts: wrong.** It finished at 111.7, 110.5 and 172.6, a higher mean than full DQN (131.6 against 107.5), and two seeds reached 150. The Q part held: it stayed under 100.
- **A buffer of 1,000 lands near full DQN: partly right.** Two of three seeds fell inside 60 to 130, and Q stayed under 100.

No replay isn't really "no replay". Training on the latest 64 with one update per step uses each transition about 64 times in a row, about the same reuse as full DQN (63.8 against 62.4 uses per transition). So data used only once never gets tested here, and I didn't test why the no-replay runs held up.

### Policy gradient

| Agent | Final return (seeds 0 / 1 / 2) | Steps to 150 | Updates | Seconds per run |
|---|---|---|---|---|
| REINFORCE + baseline | 196.2 / 174.5 / 146.7 | 13,000 / 12,000 / 13,500 | 26 / 26 / 27 | about 4 |
| PPO | 133.2 / 186.7 / 178.2 | 10,500 / 10,500 / 11,000 | 912 each | about 10 |

Every DQN setup took about 24 s per run, because it updates on every step. PPO's ratio was exactly 1 on the first minibatch of every rollout and had moved off 1 by the last, with about 3% of samples clipped, so the old log-probabilities were kept rather than recomputed.

### Comparison

![Learning curves of all six agents](figures/learning_curves_all_agents.png)

| Measure | Pairs that differ (of 15) | Which |
|---|---|---|
| Final return | 6 | every agent over no-target (5), and REINFORCE over full DQN (146.7 against 145.6) |
| Steps to 150 | 3 | REINFORCE and PPO over no-target, PPO over REINFORCE |
| Wall-clock | 12 | the 9 pairs with a policy-gradient agent, plus 3 DQN pairs about a second apart |

The 3 DQN wall-clock differences are about a second on 24 s of identical work, and a second run of the same notebook flagged different pairs, so I read them as timing noise. Everything else is "cannot tell".

## Conclusion

**Best final return:** REINFORCE with a baseline (196.2, 174.5, 146.7). It's the only agent that beats full DQN on the rule, but only by 1.1 on the closest seeds, it was the slowest to reach 150, and I can't separate it from PPO.

**Quickest to a working policy:** PPO, which reached 150 at steps 10,500 to 11,000 and beat REINFORCE there. Its final return isn't clearly better than anything except the no-target DQN.

**Cheapest to run:** REINFORCE at about 4 s, then PPO at about 10 s, against about 24 s for every DQN setup.

**Full DQN:** I wouldn't pick it for this task. It only beat its own broken version. In its favour, it's the only agent whose Q-values I could check against a known bound, and it can reuse old data.

**What my numbers establish:**
- The target network matters: without it Q explodes and the agent does worse than random.
- The policy-gradient agents are several times cheaper per run than DQN here.

**What they don't establish:**
- An order among full DQN, no replay and the 1,000 buffer. Their seeds overlap on every measure.
- Why no replay held up, since it still reuses each transition about 64 times.

**Limits:**
- One small task, small networks, a fixed 20,000-step budget, three seeds and settings I didn't tune per agent.
- I picked Huber delta 1, PPO entropy coefficient 0 and PPO discount 0.99 myself.
- REINFORCE doesn't bootstrap at the 200-step limit, which is exactly what the DQN target is careful about.
- Q on the probe states is read every 1,000 steps, so a spike between readings would be missed.
