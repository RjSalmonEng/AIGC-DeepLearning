# Fine-Tune with LoRA

SmolLM2-135M from the Hugging Face Hub, measured zero-shot on the emotion task, then fine-tuned fully and with my own LoRA at ranks 1, 4 and 16.

The model stays a language model. Each prompt ends in `Emotion:` and the answer is one label token (sadness, joy, love, anger, fear, surprise). Chance is 1/6. Every training cell uses seeds 0, 1 and 2, the same 600 training examples, the same 500 test examples, 120 steps at batch 8 and AdamW.

## Run

Everything runs on Google Colab with a GPU runtime (Runtime > Change runtime type > T4 GPU).

1. Upload `Lab6.ipynb` to Colab (File > Upload notebook).
2. Run all. The first cells install `transformers peft datasets`, fetch `lab06_starter.py` (from this repo, or an upload box if that fails) and run the starter once, unmodified.

`lab06_starter.py` is the course starter, unchanged. My code is in the notebook and imports from it.

## Prediction (written before running the sweep)

**Rank 16 against full fine-tuning:** I expect rank 16 to land within about 2 points of full fine-tuning, inside the spread of the seeds. The model already reads emotion in a tweet; the job is to route that to six label tokens at one position. That should be a small change, and a rank-16 update to every q_proj and v_proj is plenty for it. With only 120 steps, full fine-tuning also can't use most of its extra freedom.

**Rank 1:** I don't expect it to be clearly worse. Maybe 2 to 5 points under rank 16, with more spread between seeds. With alpha = 2r the update is scaled by 2 at every rank, so rank 1 isn't trained more weakly, it just has one direction per layer. That is still 60 layers' worth of directions, which I think is enough for a six-way choice. If rank 1 does fall well behind, it means the task needs more than one direction per layer to move.

## Results

_To be filled in from the Colab run._
