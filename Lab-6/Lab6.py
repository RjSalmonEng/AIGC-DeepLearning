"""Lab 6: Fine-Tune with LoRA, script version of Lab6.ipynb.

Run on a GPU with lab06_starter.py next to this file:
    pip install -U transformers peft datasets pandas matplotlib
    python Lab6.py
"""


# ----------------------------------------------------------------------
# Lab 6: Fine-Tune with LoRA
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Colab setup
# ----------------------------------------------------------------------

import os

assert os.path.exists("lab06_starter.py"), "put lab06_starter.py next to Lab6.py"


# ----------------------------------------------------------------------
# Does the starter run as given?
# ----------------------------------------------------------------------

import copy
import gc
import json
import time

import matplotlib.pyplot as plt
import pandas as pd
import peft
import torch
import torch.nn as nn
import transformers
import datasets

import lab06_starter as S

print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| datasets", datasets.__version__)
print("device:", S.DEVICE, "|", torch.cuda.get_device_name(0) if S.DEVICE == "cuda" else "no GPU")

STEPS = 120
BATCH_SIZE = 8
SEEDS = [0, 1, 2]
RANKS = [1, 4, 16]
N_TRAIN, N_TEST, N_VALIDATION = 600, 500, 300
os.makedirs("figures", exist_ok=True)
results = {"steps": STEPS, "batch_size": BATCH_SIZE, "device": torch.cuda.get_device_name(0) if S.DEVICE == "cuda" else "cpu"}


# ----------------------------------------------------------------------
# Zero-shot baseline and answer-only loss
# ----------------------------------------------------------------------

base_model, tokenizer = S.load_model()
train_examples, test_examples, label_ids = S.load_task("emotion", n_train=N_TRAIN, n_test=N_TEST, tokenizer=tokenizer)
label_words = S.TASKS["emotion"]["words"]

print("label tokens:", {word: tokenizer.decode([token_id]) for word, token_id in zip(label_words, label_ids)})
print("example prompt:", repr(tokenizer.decode(train_examples[0]["prompt_ids"])),
      "answer:", repr(tokenizer.decode([train_examples[0]["answer_id"]])))


@torch.no_grad()
def predict_labels(model, examples, batch_size=32):
    """Same scoring as S.evaluate, but returns the predicted label for every example."""
    model.eval()
    label_tensor = torch.tensor(label_ids, device=S.DEVICE)
    predictions = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        input_ids, attention_mask, prompt_lens = S.pad_batch(chunk, tokenizer.pad_token_id)
        logits = model(input_ids=input_ids.to(S.DEVICE), attention_mask=attention_mask.to(S.DEVICE)).logits
        last_prompt_scores = logits[torch.arange(len(chunk)), (prompt_lens - 1).to(S.DEVICE)]
        predictions += last_prompt_scores[:, label_tensor].argmax(dim=-1).tolist()
    return predictions


zero_shot_accuracy = S.evaluate(base_model, tokenizer, test_examples, label_ids)
zero_shot_predictions = predict_labels(base_model, test_examples)
test_label_counts = pd.Series([label_words[example["label"]] for example in test_examples]).value_counts()
zero_shot_counts = pd.Series([label_words[prediction] for prediction in zero_shot_predictions]).value_counts()
chance = 1 / len(label_ids)
majority_rate = test_label_counts.iloc[0] / len(test_examples)

print(f"zero-shot accuracy: {zero_shot_accuracy:.3f}")
print(f"chance: {chance:.3f} | always '{test_label_counts.index[0]}': {majority_rate:.3f}")
print(pd.DataFrame({"true labels": test_label_counts, "zero-shot predictions": zero_shot_counts}).fillna(0).astype(int))
results.update(zero_shot_accuracy=zero_shot_accuracy, chance=chance, majority_label=test_label_counts.index[0],
               majority_rate=float(majority_rate))


# ----------------------------------------------------------------------
# Scoring only the answer
# ----------------------------------------------------------------------

def answer_only_labels(input_ids, attention_mask, prompt_lens):
    """Score only the answer token. Everything else, prompt and padding, is -100."""
    labels = torch.full_like(input_ids, -100)
    rows = torch.arange(input_ids.shape[0])
    labels[rows, prompt_lens] = input_ids[rows, prompt_lens]
    return labels


def off_by_one_labels(input_ids, attention_mask, prompt_lens):
    """The silent mistake from the brief: scores the last prompt token instead of the answer."""
    labels = torch.full_like(input_ids, -100)
    rows = torch.arange(input_ids.shape[0])
    labels[rows, prompt_lens - 1] = input_ids[rows, prompt_lens - 1]
    return labels


def check_answer_only(labels_fn, batch):
    input_ids, attention_mask, prompt_lens = S.pad_batch(batch, tokenizer.pad_token_id)
    labels = labels_fn(input_ids, attention_mask, prompt_lens)
    scored_per_row = (labels != -100).sum(dim=1)
    assert torch.all(scored_per_row == 1), f"scored positions per row: {scored_per_row.tolist()}"
    for row, example in enumerate(batch):
        assert torch.all(labels[row, :prompt_lens[row]] == -100), f"row {row} scores a prompt token"
        assert labels[row, prompt_lens[row]].item() == example["answer_id"], f"row {row} does not score the answer"
    return labels


check_batch = train_examples[:8]
input_ids, attention_mask, prompt_lens = S.pad_batch(check_batch, tokenizer.pad_token_id)
print("prompt lengths in the check batch:", prompt_lens.tolist())

answer_labels = check_answer_only(answer_only_labels, check_batch)
print("answer_only_labels passed: one scored position per row, no prompt token scored")
print("scored tokens:", [tokenizer.decode([token_id]) for token_id in answer_labels[answer_labels != -100].tolist()])

try:
    check_answer_only(off_by_one_labels, check_batch)
except AssertionError as error:
    shifted_labels = off_by_one_labels(input_ids, attention_mask, prompt_lens)
    shifted_tokens = [tokenizer.decode([token_id]) for token_id in shifted_labels[shifted_labels != -100].tolist()]
    print("off-by-one version fails:", error, "| it scores", shifted_tokens[:3])

all_token_labels = S.all_tokens_labels(input_ids, attention_mask, prompt_lens)
print("starter's all_tokens_labels scores per row:", (all_token_labels != -100).sum(dim=1).tolist())

with torch.no_grad():
    batch_ids, batch_mask = input_ids.to(S.DEVICE), attention_mask.to(S.DEVICE)
    model_output = base_model(input_ids=batch_ids, attention_mask=batch_mask, labels=answer_labels.to(S.DEVICE))
    last_prompt_logits = model_output.logits[torch.arange(len(check_batch)), (prompt_lens - 1).to(S.DEVICE)]
    answer_tokens = torch.tensor([example["answer_id"] for example in check_batch], device=S.DEVICE)
    manual_loss = nn.functional.cross_entropy(last_prompt_logits, answer_tokens)
print(f"HF loss {model_output.loss.item():.6f} | manual loss at last prompt position {manual_loss.item():.6f}")
assert torch.allclose(model_output.loss, manual_loss, atol=1e-5)


# ----------------------------------------------------------------------
# LoRA from scratch, verified
# ----------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        base_weight = base.weight
        bound = 1 / base.in_features ** 0.5
        self.lora_A = nn.Parameter(
            (torch.rand(r, base.in_features, device=base_weight.device, dtype=base_weight.dtype) * 2 - 1) * bound)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base_weight.device, dtype=base_weight.dtype))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merged(self):
        """A plain nn.Linear holding W + (alpha / r) B A."""
        merged_layer = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None,
                                 device=self.base.weight.device, dtype=self.base.weight.dtype)
        with torch.no_grad():
            merged_layer.weight.copy_(self.base.weight + self.scaling * self.lora_B @ self.lora_A)
            if self.base.bias is not None:
                merged_layer.bias.copy_(self.base.bias)
        return merged_layer


def inject_lora(model, r, alpha, target_modules=("q_proj", "v_proj")):
    """Freeze every weight, then swap each target nn.Linear for a LoRALinear. Returns the wrapped layers."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = [(name, module) for name, module in model.named_modules()
               if name.split(".")[-1] in target_modules and isinstance(module, nn.Linear)]
    wrapped_layers = {}
    for name, module in targets:
        parent_name, child_name = name.rsplit(".", 1)
        lora_layer = LoRALinear(module, r, alpha)
        setattr(model.get_submodule(parent_name), child_name, lora_layer)
        wrapped_layers[name] = lora_layer
    return wrapped_layers


# ----------------------------------------------------------------------
# Checking one layer before trusting the whole model
# ----------------------------------------------------------------------

torch.manual_seed(0)
check_rank, check_alpha = 4, 8
q_proj_copy = copy.deepcopy(base_model.model.layers[0].self_attn.q_proj)
lora_layer = LoRALinear(q_proj_copy, check_rank, check_alpha)
sample_input = torch.randn(3, 11, q_proj_copy.in_features, device=S.DEVICE)

print("identical to the wrapped layer at init:", torch.equal(lora_layer(sample_input), q_proj_copy(sample_input)))
trainable_count = sum(parameter.numel() for parameter in lora_layer.parameters() if parameter.requires_grad)
expected_count = check_rank * (q_proj_copy.in_features + q_proj_copy.out_features)
print(f"trainable {trainable_count} | r * (in + out) = {check_rank} * ({q_proj_copy.in_features} + {q_proj_copy.out_features}) = {expected_count}")
assert trainable_count == expected_count

with torch.no_grad():
    lora_layer.lora_B.normal_(0, 0.02)
two_path_output = lora_layer(sample_input)
merged_output = lora_layer.merged()(sample_input)
print(f"merged vs two-path, B nonzero: max diff {(merged_output - two_path_output).abs().max().item():.2e}")
assert torch.allclose(merged_output, two_path_output, atol=1e-4)
print(f"and the update is no longer zero: max |two-path - base| {(two_path_output - q_proj_copy(sample_input)).abs().max().item():.2e}")

from peft import LoraConfig, get_peft_model


class OneLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.q_proj = layer

    def forward(self, x):
        return self.q_proj(x)


peft_holder = get_peft_model(OneLayer(copy.deepcopy(base_model.model.layers[0].self_attn.q_proj)),
                             LoraConfig(r=check_rank, lora_alpha=check_alpha, target_modules=["q_proj"],
                                        lora_dropout=0.0, bias="none"))
peft_layer = peft_holder.base_model.model.q_proj
with torch.no_grad():
    peft_layer.lora_A["default"].weight.copy_(lora_layer.lora_A)
    peft_layer.lora_B["default"].weight.copy_(lora_layer.lora_B)
peft_output = peft_layer(sample_input)
print(f"mine vs peft, same A and B: max diff {(two_path_output - peft_output).abs().max().item():.2e} | peft scaling {peft_layer.scaling['default']}")
assert torch.allclose(two_path_output, peft_output, atol=1e-4)

wrong_scaling_output = q_proj_copy(sample_input) + (sample_input @ lora_layer.lora_A.T @ lora_layer.lora_B.T) * check_alpha
print(f"scaled by alpha instead of alpha/r: max diff to peft {(wrong_scaling_output - peft_output).abs().max().item():.2e}")


# ----------------------------------------------------------------------
# Does it hold across the whole model?
# ----------------------------------------------------------------------

torch.manual_seed(0)
my_lora_model = copy.deepcopy(base_model)
my_wrapped_layers = inject_lora(my_lora_model, r=check_rank, alpha=check_alpha)
batch_ids, batch_mask = input_ids.to(S.DEVICE), attention_mask.to(S.DEVICE)
with torch.no_grad():
    base_logits = base_model(input_ids=batch_ids, attention_mask=batch_mask).logits
    my_logits = my_lora_model(input_ids=batch_ids, attention_mask=batch_mask).logits
print(f"{len(my_wrapped_layers)} layers wrapped | adapted model at init vs pretrained: identical = {torch.equal(base_logits, my_logits)}")

with torch.no_grad():
    for wrapped_layer in my_wrapped_layers.values():
        wrapped_layer.lora_B.normal_(0, 0.02)
peft_model = S.apply_lora(copy.deepcopy(base_model), r=check_rank, alpha=check_alpha)
with torch.no_grad():
    for module_name, wrapped_layer in my_wrapped_layers.items():
        peft_module = peft_model.get_submodule("base_model.model." + module_name)
        peft_module.lora_A["default"].weight.copy_(wrapped_layer.lora_A)
        peft_module.lora_B["default"].weight.copy_(wrapped_layer.lora_B)
    my_logits = my_lora_model(input_ids=batch_ids, attention_mask=batch_mask).logits
    peft_logits = peft_model(input_ids=batch_ids, attention_mask=batch_mask).logits
print(f"whole model, mine vs peft with B nonzero: max logit diff {(my_logits - peft_logits).abs().max().item():.2e}")
assert torch.allclose(my_logits, peft_logits, atol=1e-3)


# ----------------------------------------------------------------------
# How many parameters am I actually training?
# ----------------------------------------------------------------------

parameter_rows = []
for rank in RANKS:
    counting_model = copy.deepcopy(base_model)
    counted_layers = inject_lora(counting_model, r=rank, alpha=2 * rank)
    trainable_count, total_count = S.count_params(counting_model)
    formula_count = sum(rank * (layer.base.in_features + layer.base.out_features) for layer in counted_layers.values())
    assert trainable_count == formula_count
    parameter_rows.append(dict(rank=rank, wrapped_layers=len(counted_layers), trainable=trainable_count,
                               formula_sum=formula_count, total=total_count,
                               percent_of_model=100 * trainable_count / total_count))
    del counting_model
parameter_table = pd.DataFrame(parameter_rows)
first_attention = base_model.model.layers[0].self_attn
print(f"q_proj in/out ({first_attention.q_proj.in_features}, {first_attention.q_proj.out_features}), "
      f"v_proj in/out ({first_attention.v_proj.in_features}, {first_attention.v_proj.out_features}), "
      f"layers {len(base_model.model.layers)}")
print(parameter_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
results["lora_parameters"] = parameter_rows

del my_lora_model, peft_model, peft_holder, base_model
gc.collect()
torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# Rank sweep against full fine-tuning
# ----------------------------------------------------------------------

search_train_examples, validation_examples, _ = S.load_task("emotion", n_train=N_TRAIN, n_test=N_VALIDATION,
                                                            tokenizer=tokenizer, eval_split="validation")
assert [example["prompt_ids"] for example in search_train_examples] == [example["prompt_ids"] for example in train_examples], \
    "training sample changed"
test_prompts = {tuple(example["prompt_ids"]) for example in test_examples}
print("validation prompts that also appear in test:",
      sum(tuple(example["prompt_ids"]) in test_prompts for example in validation_examples))


def run_experiment(method, seed, learning_rate, eval_examples, rank=None, labels_fn=answer_only_labels):
    """One run on a freshly loaded model. Same data, steps, batch size and optimizer for every method."""
    torch.manual_seed(seed)
    model, _ = S.load_model()
    if method == "lora":
        inject_lora(model, r=rank, alpha=2 * rank)
    run_info = S.train(model, tokenizer, train_examples, steps=STEPS, lr=learning_rate, batch_size=BATCH_SIZE,
                       seed=seed, labels_fn=labels_fn)
    run_info["accuracy"] = S.evaluate(model, tokenizer, eval_examples, label_ids)
    run_info.update(method=method if method == "full" else f"lora r={rank}", rank=rank, seed=seed,
                    learning_rate=learning_rate, loss=labels_fn.__name__)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  {run_info['method']:<10} seed {seed} lr {learning_rate:<7g} {run_info['loss']:<20} acc {run_info['accuracy']:.3f} "
          f"loss {run_info['first_loss']:.3f} -> {run_info['last_loss']:.3f} ({run_info['seconds']:.0f}s)")
    return run_info


search_grid = {"full": [1e-5, 3e-5, 1e-4, 3e-4], "lora": [1e-4, 3e-4, 1e-3, 3e-3]}
search_runs = []
for method, learning_rates in search_grid.items():
    for learning_rate in learning_rates:
        search_runs.append(run_experiment(method, 0, learning_rate, validation_examples,
                                          rank=4 if method == "lora" else None))

search_table = pd.DataFrame(search_runs)[["method", "learning_rate", "accuracy", "first_loss", "last_loss"]]
print(search_table.rename(columns={"accuracy": "validation_accuracy"}).to_string(index=False))
chosen_lr = {"full": float(search_table[search_table.method == "full"].sort_values("accuracy").iloc[-1].learning_rate),
             "lora": float(search_table[search_table.method == "lora r=4"].sort_values("accuracy").iloc[-1].learning_rate)}
print("chosen learning rates:", chosen_lr)
results["lr_search"] = search_table.to_dict("records")
results["chosen_lr"] = chosen_lr


# ----------------------------------------------------------------------
# The sweep
# ----------------------------------------------------------------------

sweep_runs = []
sweep_start = time.time()
for seed in SEEDS:
    sweep_runs.append(run_experiment("full", seed, chosen_lr["full"], test_examples))
    for rank in RANKS:
        sweep_runs.append(run_experiment("lora", seed, chosen_lr["lora"], test_examples, rank=rank))
print(f"sweep took {(time.time() - sweep_start) / 60:.1f} min")
results["sweep"] = sweep_runs

method_order = ["full"] + [f"lora r={rank}" for rank in RANKS]
seed_columns = [f"seed {seed}" for seed in SEEDS]
sweep_frame = pd.DataFrame(sweep_runs)
accuracy_table = sweep_frame.pivot(index="method", columns="seed", values="accuracy").loc[method_order]
accuracy_table.columns = seed_columns
accuracy_table["mean"] = accuracy_table[seed_columns].mean(axis=1)
accuracy_table["std"] = accuracy_table[seed_columns].std(axis=1)
accuracy_table["mean - zero-shot"] = accuracy_table["mean"] - zero_shot_accuracy
accuracy_table["mean - chance"] = accuracy_table["mean"] - chance
accuracy_table["learning rate"] = [f"{chosen_lr['full']:g}"] + [f"{chosen_lr['lora']:g}"] * len(RANKS)
accuracy_table["trainable"] = sweep_frame.groupby("method").trainable.first().loc[method_order]
accuracy_table["s/step"] = sweep_frame.groupby("method").seconds_per_step.mean().loc[method_order]
print(f"zero-shot {zero_shot_accuracy:.3f} | chance {chance:.3f} | majority '{results['majority_label']}' {majority_rate:.3f}")
print(accuracy_table.to_string(float_format=lambda value: f"{value:.3f}"))
print()
final_loss_table = sweep_frame.pivot(index="method", columns="seed", values="last_loss").loc[method_order]
final_loss_table.columns = [f"final loss seed {seed}" for seed in SEEDS]
print(final_loss_table.to_string(float_format=lambda value: f"{value:.3f}"))


# ----------------------------------------------------------------------
# Does scoring the prompt help or hurt?
# ----------------------------------------------------------------------

all_token_runs = [run_experiment("lora", seed, chosen_lr["lora"], test_examples, rank=16, labels_fn=S.all_tokens_labels)
                  for seed in SEEDS]
results["all_token_r16"] = all_token_runs
loss_comparison = pd.DataFrame({
    "answer-only": [run["accuracy"] for run in sweep_runs if run["method"] == "lora r=16"],
    "all-token": [run["accuracy"] for run in all_token_runs]}, index=seed_columns)
loss_comparison["all-token - answer-only"] = loss_comparison["all-token"] - loss_comparison["answer-only"]
loss_comparison.loc["mean"] = loss_comparison.mean()
print(loss_comparison.to_string(float_format=lambda value: f"{value:.3f}"))

figure, axis = plt.subplots(figsize=(7, 4))
plot_labels = method_order + ["lora r=16\nall-token"]
for position, method in enumerate(method_order):
    seed_accuracies = sweep_frame[sweep_frame.method == method].accuracy
    axis.scatter([position] * len(seed_accuracies), seed_accuracies, color="tab:blue", zorder=3)
    axis.hlines(seed_accuracies.mean(), position - 0.25, position + 0.25, color="tab:blue")
all_token_accuracies = [run["accuracy"] for run in all_token_runs]
axis.scatter([len(method_order)] * len(all_token_accuracies), all_token_accuracies, color="tab:orange", zorder=3)
axis.hlines(sum(all_token_accuracies) / len(all_token_accuracies), len(method_order) - 0.25, len(method_order) + 0.25,
            color="tab:orange")
axis.axhline(zero_shot_accuracy, color="grey", linestyle="--", label=f"zero-shot {zero_shot_accuracy:.3f}")
axis.axhline(majority_rate, color="grey", linestyle="-.", label=f"always '{results['majority_label']}' {majority_rate:.3f}")
axis.axhline(chance, color="grey", linestyle=":", label=f"chance {chance:.3f}")
axis.set_xticks(range(len(plot_labels)), plot_labels)
axis.set_ylabel("test accuracy (each dot is a seed)")
axis.set_title(f"SmolLM2-135M on emotion, {STEPS} steps x batch {BATCH_SIZE}")
axis.legend(fontsize=8, loc="lower right")
figure.tight_layout()
figure.savefig("figures/accuracy_by_method.png", dpi=150)
plt.show()


# ----------------------------------------------------------------------
# Memory: predict, then measure
# ----------------------------------------------------------------------

config = transformers.AutoConfig.from_pretrained(S.MODEL_NAME)
hidden_size = config.hidden_size
head_dim = hidden_size // config.num_attention_heads
kv_dim = config.num_key_value_heads * head_dim
params_per_layer = (hidden_size * hidden_size + 2 * hidden_size * kv_dim + hidden_size * hidden_size
                    + 3 * hidden_size * config.intermediate_size + 2 * hidden_size)
embedding_params = config.vocab_size * hidden_size
base_parameter_count = (embedding_params * (1 if config.tie_word_embeddings else 2)
                        + config.num_hidden_layers * params_per_layer + hidden_size)
print(f"hidden {hidden_size}, kv dim {kv_dim}, layers {config.num_hidden_layers}, vocab {config.vocab_size}, "
      f"tied embeddings {config.tie_word_embeddings} -> {base_parameter_count:,} parameters")

prediction_rows = []
for method in method_order:
    if method == "full":
        trainable_count, total_count = base_parameter_count, base_parameter_count
    else:
        rank = int(method.split("=")[1])
        trainable_count = config.num_hidden_layers * rank * ((hidden_size + hidden_size) + (hidden_size + kv_dim))
        total_count = base_parameter_count + trainable_count
    prediction_rows.append(dict(method=method, trainable=trainable_count, predicted_weights=4 * total_count,
                                predicted_grads=4 * trainable_count, predicted_adam=8 * trainable_count,
                                predicted_total=4 * total_count + 12 * trainable_count))
prediction_table = pd.DataFrame(prediction_rows).set_index("method")
print(prediction_table.to_string())
results["memory_prediction"] = prediction_table.reset_index().to_dict("records")

to_mb = lambda byte_count: byte_count / 2 ** 20
memory_columns = ["weight_bytes", "grad_bytes", "optimizer_bytes", "peak_cuda_bytes"]
seed_spread = sweep_frame.groupby("method")[memory_columns].agg(["min", "max"])
for column in ["weight_bytes", "grad_bytes", "optimizer_bytes"]:
    assert (seed_spread[column]["min"] == seed_spread[column]["max"]).all(), f"{column} differs across seeds"
measured = sweep_frame.groupby("method")[memory_columns].first().loc[method_order]
measured["measured_total"] = measured.weight_bytes + measured.grad_bytes + measured.optimizer_bytes

memory_table = pd.DataFrame({
    "weights pred MB": to_mb(prediction_table.predicted_weights), "weights meas MB": to_mb(measured.weight_bytes),
    "grads pred MB": to_mb(prediction_table.predicted_grads), "grads meas MB": to_mb(measured.grad_bytes),
    "adam pred MB": to_mb(prediction_table.predicted_adam), "adam meas MB": to_mb(measured.optimizer_bytes),
    "total pred MB": to_mb(prediction_table.predicted_total), "total meas MB": to_mb(measured.measured_total),
    "pred - meas bytes": prediction_table.predicted_total - measured.measured_total,
    "peak CUDA MB": to_mb(measured.peak_cuda_bytes),
    "peak / pred": measured.peak_cuda_bytes / prediction_table.predicted_total})
print(memory_table.to_string(float_format=lambda value: f"{value:.2f}"))
grads_and_adam = measured.grad_bytes + measured.optimizer_bytes
print(f"\nfull vs lora r=16, training state: {measured.measured_total['full'] / measured.measured_total['lora r=16']:.2f}x, "
      f"grads + adam: {grads_and_adam['full'] / grads_and_adam['lora r=16']:.0f}x")
results["memory_measured"] = measured.reset_index().to_dict("records")

widest_sequence = max(len(example["prompt_ids"]) + 1 for example in train_examples)
mean_sequence = sum(len(example["prompt_ids"]) + 1 for example in train_examples) / len(train_examples)
logits_bytes = BATCH_SIZE * widest_sequence * config.vocab_size * 4
activation_bytes = BATCH_SIZE * widest_sequence * hidden_size * 4 * 20 * config.num_hidden_layers
print(f"sequence width: mean {mean_sequence:.1f}, max {widest_sequence} tokens")
print(f"one fp32 logits tensor at batch {BATCH_SIZE} x {widest_sequence} x {config.vocab_size}: {to_mb(logits_bytes):.1f} MB")
print(f"saved activations, rough (batch x width x hidden x 4 bytes x ~20 tensors x {config.num_hidden_layers} layers): "
      f"{to_mb(activation_bytes):.1f} MB")
results["logits_mb"] = to_mb(logits_bytes)

figure, axis = plt.subplots(figsize=(7, 4))
bar_positions = list(range(len(method_order)))
bar_width = 0.27
axis.bar([position - bar_width for position in bar_positions], to_mb(prediction_table.predicted_total), bar_width,
         label="predicted (weights + grads + Adam)")
axis.bar(bar_positions, to_mb(measured.measured_total), bar_width, label="measured by starter")
axis.bar([position + bar_width for position in bar_positions], to_mb(measured.peak_cuda_bytes), bar_width,
         label="peak CUDA allocated")
axis.set_xticks(bar_positions, method_order)
axis.set_ylabel("MB")
axis.set_title("Training-state memory, fp32")
axis.legend(fontsize=8)
figure.tight_layout()
figure.savefig("figures/memory_by_method.png", dpi=150)
plt.show()


# ----------------------------------------------------------------------
# Save the results
# ----------------------------------------------------------------------

with open("lab6_results.json", "w") as results_file:
    json.dump(results, results_file, indent=1, default=float)
print("saved lab6_results.json")
