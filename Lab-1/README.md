# Lab 1

A two-layer neural network trained to predict salary from years of experience. The project compares a baseline Adam learning rate of `0.001` with one controlled variation: a learning rate of `0.01`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From the repository directory, run:

```bash
python Lab1.py
```

The script loads `Salary_dataset.csv`, prepares the data, trains both runs for 100 epochs, prints the per-epoch losses, and displays the baseline and comparison loss curves.

## Network and data

The network is an `nn.Module` with one input, a hidden layer of 10 units, a ReLU activation, and one linear output. It uses mean squared error loss and the Adam optimizer. The dataset contains salary and years-of-experience observations. Missing target rows are removed; missing input values are median-imputed. Input and target values are standardized using statistics fitted only on the training split.

Training data is wrapped in a `TensorDataset` and loaded in batches of 8 with a `DataLoader`. The model and each batch are moved to CUDA when it is available, otherwise to the CPU. Each batch follows the required five steps: clear gradients, forward pass, calculate loss, backpropagate, and update parameters.

## Example results

| Run | Learning rate | Epoch 1 loss | Epoch 100 loss |
| --- | ---: | ---: | ---: |
| Baseline | 0.001 | 0.7413 | 0.0390 |
| Variation | 0.01 | 0.6556 | 0.0341 |

These values come from the recorded run in `Lab1.ipynb`. Small numerical differences can occur across hardware and PyTorch versions.

## What changed and why

I updated the learning rate, it was increased `0.001` to `0.01`. A larger learning rate makes Adam take larger parameter-update steps, so I expected the variation to reduce its loss faster. The result matched that expectation: the variation reached a low loss within the first several epochs, while the baseline decreased more gradually. The variation showed small oscillations after reaching the low-loss region, which is consistent with larger steps moving around the minimum. A much larger learning rate could overshoot the useful region and make training unstable.

## GEN AI Acknowledgement:
Gen AI was used to create the boilerplate code for this project; by that i mean the `READMe.md` and `requirements.txt` files. along with the gitignore,