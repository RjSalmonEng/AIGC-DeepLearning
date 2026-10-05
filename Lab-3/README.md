# From Frozen Features to Fine-Tuning

Fine-tuning a pretrained ResNet18 on EuroSAT, comparing feature extraction against fine-tuning.

## Setup

```bash
cd Lab-3
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python Lab3.py
```

This downloads EuroSAT into `data/` on the first run, trains both models and prints the comparison. Plots are saved in `figures/`. The same code with outputs is in `Lab3.ipynb`.

## Dataset

EuroSAT (RGB): satellite images in 10 land-use classes. I used 200 training and 50 validation images per class (2,000 train, 500 val), split once with seed 0. I picked it because satellite images look very different from ImageNet.

## What I did

- `train_transform` and `eval_transform` are copied from the starter. Training images only use `train_transform` and validation images only use `eval_transform`.
- **Feature extraction:** froze all of ResNet18, replaced `fc` with a 10-class layer and trained only that (Adam, lr 1e-3).
- **Fine-tuning:** same starting weights, but `layer4` is unfrozen. One optimizer with two parameter groups: `layer4` at lr 1e-4 and `fc` at lr 1e-3.
- Both runs use the same split, seed, augmentation and 5 epochs.

The frozen backbone is kept in `eval()` mode. If it were left in `train()` mode, the BatchNorm layers would keep updating their running mean and variance from my data even though the weights are frozen, so the features would slowly change.

## Results

| Run | val accuracy | train time | trainable params |
|---|---|---|---|
| Feature extraction | 0.892 | 282.0s | 5,130 |
| Fine-tuning | 0.944 | 354.7s | 8,398,858 |

## Conclusion

Fine-tuning did better. With the backbone frozen, the head can only use the features ResNet18 learned for ImageNet photos. Satellite classes like crop types, or highways and rivers, need features that ImageNet never had to learn. Unfreezing `layer4` let those features change for this dataset, and the training loss went down to 0.28 compared to 0.53. The smaller learning rate on `layer4` keeps it from losing the pretrained weights while the new head learns.

## Known limitations

- Only one seed and 5 epochs, so the gap might change with more runs.
- Only a small part of EuroSAT was used.
- I did not test a run without augmentation.
