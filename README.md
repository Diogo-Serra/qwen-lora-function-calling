# Qwen LoRA Function Calling

A local toolkit for fine-tuning and evaluating **Qwen** text-generation models with **LoRA**. It measures a baseline, trains an adapter, and compares every checkpoint against a held-out test set, so training progress is evidence-based rather than assumed.

The reference task: given a request and a list of available functions, the model must generate the name of the correct function to call. The same train/evaluate/compare flow works for any dataset under `src/data/<task_name>/` and any Qwen checkpoint under `src/model/`.

Every run preserves the original base model, creates one directory per training session, saves a checkpoint per epoch, and evaluates every version on the same held-out test set.


## The task

Given this prompt:

```text
Request: "Add 14 and 28"
Available functions:
- fn_add_numbers: Add two numbers together and return their sum.
- fn_greet: Generate a greeting message for a person by name.
...
The function to call is:
```

the model should generate `fn_add_numbers`. This is implemented as **causal text generation**, not classification: fine-tuning makes the correct function name a more likely continuation of the prompt.

The first six labels are `fn_add_numbers`, `fn_greet`, `fn_reverse_string`, `fn_get_square_root`, `fn_substitute_string_with_regex`, and `fn_unknown`. Additional labels should introduce controlled difficulty: similar descriptions, more parameters, overlapping vocabulary, and ambiguous cases.

## Core concepts

- **Base model**: the default is **Qwen2.5-0.5B** (~500M parameters), downloaded automatically by `make install` into `src/model/Qwen2.5-0.5B`. Larger Qwen checkpoints can be added manually for more capacity.
- **Tokenizer**: converts text into integer tokens; the model predicts token IDs, so prompt format (punctuation, function order, final cue) affects results.
- **Fine-tuning / SFT**: this repo uses supervised fine-tuning - each prompt has a known completion. Prompt tokens are masked (`-100`) so only completion tokens contribute to the loss.
- **LoRA / PEFT**: LoRA freezes the base weights and trains small adapter matrices in the attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), using far less memory than full fine-tuning. Current settings: `r=8`, `lora_alpha=16`, `lora_dropout=0.05`, bias frozen. PEFT is the Hugging Face library that attaches, trains, saves, and merges the adapter.
- **Function calling**: this repo trains only the routing choice. Argument extraction, schema validation, constrained decoding, and error handling remain the application's responsibility.

Stack: **Transformers** (model/tokenizer/`Trainer`), **Datasets** (JSONL + seeded split), **PEFT** (LoRA), **Accelerate** (device handling), **PyTorch** (execution), **TensorBoard** (loss curves), **safetensors** (weight storage).

## Repository layout

```text
qwen-lora-function-calling/
├── src/
│   ├── __main__.py       # entry point for `python -m src`
│   ├── cli/              # menu.py - interactive terminal menu
│   ├── evaluation/       # evaluate.py (single model), compare.py (all checkpoints)
│   ├── training/         # train.py, export_model.py
│   ├── data/
│   │   ├── raw/          # drop unconverted downloads here (untracked)
│   │   └── function_name/# converted dataset + fixed test set (not shipped)
│   └── model/            # local pretrained weights (e.g. Qwen2.5-0.5B)
├── run.sh                # terminal application launcher
├── output/               # generated artifacts, ignored by Git
│   ├── evaluations/<base-model>/
│   ├── models/<base-model>/       # exported merged models
│   └── runs/<base-model>/<session>/
│       ├── latest -> <session>/
│       ├── adapter_model.safetensors, adapter_config.json
│       ├── run_config.json
│       ├── checkpoint-*/
│       └── logs/
├── Makefile
└── requirements.txt
```

A **base model** owns multiple **sessions**; each session owns multiple **checkpoints**. Different base models (e.g. `Qwen2.5-0.5B` vs `Qwen3-0.6B`) get separate `output/runs/<model>/` trees, so the menu only lists sessions and adapters compatible with the selected model.

## Installation

Python 3.10+ recommended. CPU training works but is slow; a CUDA GPU is much faster.

```bash
cd fine_tunning
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Or `make install`, which also downloads the default model (`Qwen/Qwen2.5-0.5B`, <1GB) into `src/model/Qwen2.5-0.5B`. Verify it downloaded correctly:

```bash
ls -lh src/model/Qwen2.5-0.5B/model.safetensors
```


## Usage

```bash
./run.sh          # interactive menu
./run.sh --list   # list discovered models, datasets, and runs without opening the menu
```

The menu discovers local models, datasets, runs, and checkpoints, and previews the exact Python command before running it. Workflows:

1. Train a LoRA adapter
2. Evaluate one model or checkpoint
3. Compare all checkpoints in a session
4. Open TensorBoard loss curves
5. View a saved JSON report
6. Show artifact locations
7. Export a standalone model

## Dataset format

JSONL, one JSON object per line:

```json
{"prompt":"Request: \"Add 3 and 7\"\nAvailable functions:\n- fn_add_numbers: Add two numbers together and return their sum.\n- fn_unknown: Use this when the request does not clearly match any available function.\nThe function to call is:\n","completion":"fn_add_numbers\n"}
```

No dataset ships with this project. Drop a raw file into `src/data/raw/`, convert it to the format above, and save it under `src/data/<task>/`.

- **Training set** updates LoRA weights.
- **Validation set** is a seeded 20% split of the training file.
- **Test set** is a fixed, held-out file named `function_name_test.jsonl`; it is never used for training, only for evaluation.

Never copy test examples into training data - this is **data leakage** and inflates scores without real generalization.

## Workflow

1. **Baseline**: evaluate the untouched base model on the fixed test set (`output/evaluations/<base-model>/base.json`).
2. **Train**: run **Train a LoRA adapter**; each run creates `output/runs/<base-model>/<session>/` with the adapter, config, checkpoints, and logs. `latest` points to the newest session.
3. **Evaluate / compare**: evaluate one state, or use **Compare all checkpoints** to score the base model, every checkpoint, and the final adapter together (`comparison.json`).
4. **Inspect loss**: open TensorBoard (`http://localhost:6006`). `train/loss` and `eval/loss` should both fall; validation rising while training falls suggests overfitting. Loss and exact-match accuracy tell different stories - check both.

Equivalent direct training command:

```bash
.venv/bin/python -m src.training.train \
  --model_name src/model/Qwen2.5-0.5B \
  --train_file src/data/function_name/function_name_dataset.jsonl \
  --output_dir output/runs/qwen2-5-0-5b/functions-v1 \
  --epochs 5 --batch_size 4 --lr 2e-4 --seed 42
```

| Flag | Default | Meaning |
|---|---:|---|
| `--epochs` | 8 | Full passes over training data |
| `--batch_size` | 4 | Examples per optimizer update |
| `--lr` | `2e-4` | Optimizer update size |
| `--seed` | 42 | Split/initialization randomness |
| `--model_name` | `src/model/Qwen2.5-0.5B` | Base model path or Hub ID |
| `--max_length` | 256 | Max token sequence length |
| `--eval_ratio` | 0.2 | Validation split fraction |

Change one variable at a time; keep the base model, test set, and seed fixed for fair comparisons.

## Export

**Export a standalone model** merges an adapter into its base model for deployment (`output/models/<base-model>/`), producing a regular Transformers checkpoint that can be loaded with `AutoModelForCausalLM.from_pretrained(...)` by any downstream application.

## Common problems

- **CUDA out of memory**: reduce batch size or sequence length; consider gradient accumulation or QLoRA.
- **Loss falls but test accuracy doesn't**: check data diversity, leakage, and overfitting.
- **One label dominates predictions**: check class balance and conflicting examples.
- **Session directory exists**: choose a different run name; sessions are never overwritten.

The goal is to understand how data quality, data quantity, training choices, model capacity, and evaluation discipline interact in a complete local LLM fine-tuning workflow - not just to produce a high score.
