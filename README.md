# Local Qwen Fine-tuning

This repository is a general-purpose local toolkit for fine-tuning and evaluating **Qwen** text-generation models. It measures a baseline, trains a LoRA adapter, and compares every checkpoint against a held-out test set, so training progress is evidence-based rather than assumed.

The current example task is narrow and measurable: teach **Qwen3-0.6B** to read a request and select one valid function name. It is the reference task used to validate the whole pipeline end to end, but the same train/evaluate/compare flow applies to any other task you point it at, by adding a new dataset under `src/data/<task_name>/` and, optionally, a different Qwen checkpoint under `src/model/`.

The project preserves the original model, creates a directory for every training run, saves each epoch checkpoint, and evaluates every version on the same held-out test set. This lets us measure what training changed instead of trusting one final score.

> The model selects a function name. The companion `Call_Me_Maybe` application, whose `llm_sdk` loads Qwen text-generation models directly, still constrains output, extracts arguments, validates types, and handles `fn_unknown`. Fine-tuning improves the model's choice; it does not replace application safety.

> **Model scope:** this toolkit focuses on the Qwen family (Qwen2, Qwen3, …). Training is architecture-agnostic under the hood and will run against other causal LMs (GPT-2, Llama-style models) for quick smoke tests, but defaults, docs, and future work assume Qwen.

## Learning path

1. Measure untouched Qwen on a fixed benchmark.
2. Bring in a function-selection dataset in the expected JSONL format (for example, one downloaded from Hugging Face Datasets).
3. Train a LoRA adapter while preserving the base model.
4. Compare the base model, every epoch checkpoint, and the final adapter.
5. Increase dataset size until additional data stops producing meaningful gains.
6. Add harder, more confusable functions only after this baseline is understood.

You will learn what pretrained models, tokenizers, supervised fine-tuning, LoRA, checkpoints, hyperparameters, training loss, validation loss, and held-out evaluation mean. You will also see which Hugging Face tools are used, why they are used, and what alternatives exist.

## The mission

Given this text:

```text
Request: "Add 14 and 28"
Available functions:
- fn_add_numbers: Add two numbers together and return their sum.
- fn_greet: Generate a greeting message for a person by name.
...
The function to call is:
```

the model should continue with:

```text
fn_add_numbers
```

This resembles classification, but it is implemented as **causal text generation**. A causal language model predicts the next token repeatedly. Fine-tuning makes the correct function name a more likely continuation of the prompt.

The first six labels are:

- `fn_add_numbers`
- `fn_greet`
- `fn_reverse_string`
- `fn_get_square_root`
- `fn_substitute_string_with_regex`
- `fn_unknown`

These simple functions establish the data and evaluation process. Later functions should introduce controlled difficulty: similar descriptions, more parameters, overlapping vocabulary, multilingual requests, and genuinely ambiguous cases.

## Core concepts

### Pretrained model

A pretrained model has already learned language patterns from a large general corpus. Training one from random weights would need far more data and compute than this project. We start from the local weights in `src/model/qwen3_0_6b/` and specialize them.

The base is **Qwen3-0.6B**, roughly 600 million parameters. Its configuration has 28 transformer layers and a hidden size of 1024. It is small enough for local experiments but capable enough to understand varied requests. Larger models can be more capable, but require more memory, storage, energy, and inference time.

### Tokenizer and tokens

A tokenizer converts text into integer **tokens** understood by the model. A token may be a word, part of a word, punctuation, or whitespace. The model predicts token IDs, not characters or semantic labels directly. Prompt format matters because changing punctuation, function order, or the final cue changes the token context.

### Fine-tuning and SFT

**Fine-tuning** continues training a pretrained model on a narrower task. This repository uses **supervised fine-tuning (SFT)**: each prompt has a known correct completion.

During loss calculation, prompt tokens receive the mask value `-100`, so only completion tokens contribute to the objective. The model learns the desired answer rather than being rewarded for reproducing the input.

Fine-tuning changes statistical behavior. It does not guarantee correctness, store facts like a database, or remove the need for validation. Bad or repetitive data can teach bad behavior very efficiently.

### LoRA and PEFT

**LoRA**, Low-Rank Adaptation, freezes the original Qwen weights and trains small added matrices in selected attention projections. This uses far less memory and storage than changing every parameter.

**PEFT**, Parameter-Efficient Fine-Tuning, is the Hugging Face library used to attach, train, save, load, and merge the LoRA adapter.

Current LoRA settings:

- `r=8`: rank, or capacity, of the low-rank update
- `lora_alpha=16`: update scaling
- `lora_dropout=0.05`: regularization during training
- target modules: `q_proj`, `k_proj`, `v_proj`, and `o_proj`
- bias: frozen

A higher rank can learn more but uses more memory and may overfit. **Full fine-tuning** updates every model parameter and costs much more. **QLoRA** keeps a quantized base model under LoRA to reduce memory further, but adds quantization dependencies and complexity. It is a useful later experiment.

### Function calling

Function calling maps natural language to a structured tool invocation. This repository trains only the routing choice. Complete function calling also needs argument extraction, JSON/schema validation, constrained decoding, permission checks, and error handling. The application remains responsible for those deterministic guarantees.

## Hugging Face stack

Hugging Face is an ecosystem, not one tool.

| Tool | What it does here | Alternatives |
|---|---|---|
| Transformers | Loads Qwen/tokenizer; supplies generation and `Trainer` | TRL, a custom PyTorch loop, Axolotl, Unsloth |
| Datasets | Reads JSONL and creates a seeded train/validation split | Python, pandas, PyTorch `Dataset` |
| PEFT | Implements and stores LoRA adapters | Full fine-tuning, prompt tuning, other adapter libraries |
| Accelerate | Handles devices underneath `Trainer` | Custom device code, DeepSpeed, PyTorch FSDP |
| PyTorch | Runs tensors, gradients, and model operations | JAX or TensorFlow stacks |
| TensorBoard | Visualizes training and validation loss | Weights & Biases, MLflow, CSV/custom plots |
| safetensors | Stores weights without executable pickle data | PyTorch `.bin` files |

`Trainer` is convenient for teaching because it provides optimization, evaluation, checkpointing, and logging. A custom training loop offers more control but requires more code and more opportunities for mistakes.

## Repository layout

```text
fine_tunning/
├── src/                                  # all executable product code and inputs
│   ├── evaluation/
│   │   ├── evaluate.py                   # base or adapter evaluation
│   │   └── compare.py                    # checkpoint comparison
│   ├── publishing/
│   │   └── huggingface.py                # controlled Hub upload
│   ├── training/
│   │   ├── train.py                      # seeded LoRA training
│   │   └── export_model.py               # standalone model export
│   ├── data/
│   │   ├── raw/                          # drop unconverted downloads here (untracked)
│   │   └── function_name/                # one folder per training task; empty until you add data
│   │       ├── function_name_dataset.jsonl   # your converted training data (not shipped)
│   │       └── function_name_test.jsonl      # your fixed, held-out benchmark (not shipped)
│   └── model/                            # local pretrained model weights
│       ├── Qwen2.5-0.5B/                 # smallest Qwen, downloaded automatically by `make install`
│       └── ...                          # other Qwen checkpoints you download manually, e.g. Qwen3-0.6B for the reference task
├── run.sh                                # terminal application launcher
├── output/                               # all generated artifacts
│   ├── datasets/<base-model>/            # datasets you place here, e.g. from Hugging Face
│   ├── evaluations/<base-model>/         # untouched baseline reports
│   ├── models/<base-model>/              # exported merged models
│   └── runs/
│       └── <base-model>/
│           ├── latest -> <session>/
│           └── <session>/                # one training experiment
├── Makefile
└── requirements.txt
```

`src/` contains code and project inputs. `src/data/` holds one subfolder per training task, each with its seed and held-out files; a future task adds a sibling folder such as `src/data/<task-name>/` without touching this one. `src/model/` holds the unchanged pretrained weights. `output/` contains reproducible local artifacts and is ignored by Git; the application creates its subfolders as needed. Archive or publish a selected adapter separately only when it is intentionally part of a release. The local base weights remain unchanged; training creates LoRA adapters, not replacement base models.

### Models, sessions, and checkpoints

A **base model** owns multiple **training sessions**, and each session owns multiple **checkpoints**:

```text
output/runs/qwen3-0-6b/
├── latest -> 20260915-143012/
├── 20260915-143012/
│   ├── adapter_model.safetensors          # final LoRA adapter
│   ├── adapter_config.json
│   ├── run_config.json                    # model, data, settings, timestamp
│   ├── checkpoint-300/                    # intermediate training state
│   ├── checkpoint-600/
│   ├── evaluations/
│   └── logs/
└── another-experiment/
```

Training Qwen3-1.7B later creates a separate `output/runs/qwen3-1-7b/` parent. Its sessions and `latest` link cannot be confused with Qwen3-0.6B. Datasets placed under `output/datasets/` and exported models use the same model grouping.

After a base model is selected, the terminal menu lists only training sessions owned by that model. This prevents accidentally loading a Qwen3-0.6B adapter onto a different architecture.

The final adapter files are needed to evaluate or export that trained result. Checkpoint folders are needed to compare intermediate performance or resume training. Optimizer and scheduler files inside checkpoints are resume state, not another full base model. None of these generated files is required to read or run the source code, and `output/` is intentionally excluded from Git.

## Installation

Python 3.10 or newer is recommended. CPU training is possible but slow; an NVIDIA CUDA GPU is substantially faster.

```bash
cd fine_tunning
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Or:

```bash
make install
```

Model weights are not committed to Git; `src/model/` is listed in `.gitignore` because these files can be multiple gigabytes each. To keep the repository small and the first run fast, `make install` only downloads one small default model automatically: `Qwen/Qwen2.5-0.5B` (500M parameters, under 1 GB) into `src/model/Qwen2.5-0.5B`. It is the smallest official Qwen checkpoint, small enough to train and smoke-test quickly right after cloning.

Verify the download produced real weights, not a small placeholder:

```bash
ls -lh src/model/Qwen2.5-0.5B/model.safetensors
```

Larger Qwen checkpoints used for more capable results are **not** downloaded automatically. Fetch them yourself with the same Hugging Face CLI, already installed as part of `requirements.txt`:

```bash
.venv/bin/hf download Qwen/Qwen3-0.6B --local-dir src/model/qwen3_0_6b
```

Any other Hugging Face causal LM can be added the same way, for example `Qwen/Qwen3-1.7B` when you want more capacity than the 0.5B checkpoint.

A Hugging Face account is unnecessary for public models such as these. Hub model IDs require internet access, and gated models require `hf auth login` first.

## Start the terminal lab

Launch the guided application from the repository root:

```bash
./run.sh
```

`make run` is only a convenience alias. The Makefile intentionally contains only installation, launching, and cache-cleaning targets; experiment logic belongs to the Python application.

The menu uses Questionary for keyboard navigation and discovers available resources instead of requiring paths to be memorized:

- local base and exported models
- seed and downloaded JSONL datasets
- completed training runs
- every checkpoint inside a selected run
- evaluation and comparison reports

The available workflows are:

1. Train a LoRA adapter.
2. Evaluate one model or checkpoint.
3. Compare all checkpoints in a session.
4. Open TensorBoard loss curves.
5. View a saved JSON report.
6. Show artifact locations.
7. Export a standalone model.
8. Publish to Hugging Face.

Before starting work, the application prints the exact Python command and asks for confirmation. This keeps the convenient menu educational: learners can see which module and arguments perform each operation. `Ctrl+C` stops the current command and returns control without hiding its exit status.

Every prompt has a persistent status bar at the bottom. It shows:

```text
Task | Current step | Model | Dataset | Session | Checkpoint/state | Settings | Output | TensorBoard URL
```

`Task` is currently always **Function selection**. `Model` means the selected base or generator model; it is loaded only after command confirmation. `Session` identifies one training experiment, while `State` distinguishes the untouched base, final adapter, one checkpoint, or all checkpoints. `Output` is the exact destination for the report, run, or exported model.

The fixed benchmark is deliberately excluded from the normal training-dataset list to reduce accidental leakage. It appears first when choosing evaluation data. Advanced users can still enter custom paths explicitly, so responsibility remains visible rather than hidden.

To verify discovery without opening the interactive menu:

```bash
./run.sh --list
```

## Dataset format

Datasets use **JSONL**, one complete JSON object per line:

```json
{"prompt":"Request: \"Add 3 and 7\"\nAvailable functions:\n- fn_add_numbers: Add two numbers together and return their sum.\n- fn_greet: Generate a greeting message for a person by name.\n- fn_reverse_string: Reverse a string and return the reversed result.\n- fn_get_square_root: Calculate the square root of a number.\n- fn_substitute_string_with_regex: Replace all occurrences matching a regex pattern in a string.\n- fn_unknown: Use this when the request does not clearly match any available function.\nThe function to call is:\n","completion":"fn_add_numbers\n"}
```

JSONL is easy to stream and append. Keep the runtime prompt structure aligned with the training prompt.

## Bringing in a dataset

No dataset ships with this project; `src/data/function_name/` starts empty (see its `README.md`). Source a dataset (for example from [Hugging Face Datasets](https://huggingface.co/datasets)) and convert it to the JSONL `{"prompt": ..., "completion": ...}` format shown above, matching the prompt template shown above with your own set of function labels.

Drop the original, unconverted file into `src/data/raw/` first (ignored by Git; see `src/data/raw/README.md`). Ask Copilot CLI to convert it from there into the expected JSONL format.

Place the converted file under `src/data/<task>/` (for example `src/data/function_name/`) or `output/datasets/<base-model>/` so the menu can discover it. The **Train a LoRA adapter** and **Evaluate one model or checkpoint** workflows list every `*.jsonl` file found in those locations, excluding the fixed test set (an exact-named `function_name_test.jsonl`) from the training list.

Keep these roles separate when preparing data:

- **Training set** updates LoRA weights.
- **Validation set** is a seeded 20% split of the training file and monitors loss after each epoch.
- **Test set** is your held-out `function_name_test.jsonl`; `Trainer` never sees it, and it is used only for final comparisons.

Do not copy test requests into training data. This is **data leakage**: the score improves because the answers were seen, not because the model generalized.

A handful of examples is only enough to smoke-test the workflow, not to claim production quality: with few examples per class, one mistake can swing a class score by dozens of percentage points. Real benchmarks need many reviewed examples, boundary cases, multilingual inputs, and realistic distributions.

## Reproducible training workflow

### 1. Save the untouched baseline

Open **Evaluate one model or checkpoint**, select your base model and the fixed test dataset, then choose **Untouched base model**. This writes all predictions and metrics to `output/evaluations/<base-model>/base.json`.

### 2. Train a new adapter

Open **Train a LoRA adapter**, select Qwen and the prepared dataset, then choose the run name and hyperparameters. The menu previews the complete training command before asking for confirmation.

Each invocation creates one session folder, such as `output/runs/qwen3-0-6b/20260915-143012`. A non-empty session is never reused. After successful training, `output/runs/qwen3-0-6b/latest` points to that model's newest session.

Each training session contains:

```text
run_config.json             settings, split sizes, and dataset SHA-256
adapter_config.json         LoRA configuration and base reference
adapter_model.safetensors   final adapter weights
README.md                   generated model card for sharing
checkpoint-*/               adapter and trainer state after every epoch
logs/                       TensorBoard event files
tokenizer files             tokenizer needed for reproduction
```

A **checkpoint** is a saved training state. It lets us compare intermediate learning and potentially resume work. Every epoch checkpoint is retained, so disk use grows with epochs.

The SHA-256 identifies the exact training-file bytes. The random seed makes splitting and initialization repeatable as far as the hardware and PyTorch operations allow.

### 3. Evaluate and compare

Open **Evaluate one model or checkpoint** to choose one exact model state. Results are saved under the selected session's `evaluations/` folder.

Open **Compare all checkpoints in a session** to evaluate untouched Qwen, every `checkpoint-*`, and the final adapter on one selected test set. It prints a table and writes `comparison.json` inside that session.

The final adapter and last checkpoint may score identically. They have different purposes: a checkpoint includes trainer state; the root adapter is the clean final artifact.

### 4. Inspect loss

Choose **Open TensorBoard loss curves**, select the base model and training session, then select a port. The status bar shows both the selected session's log directory and the local URL, usually `http://localhost:6006`. Press `Ctrl+C` in the terminal to stop TensorBoard and return to the menu. TensorBoard is a local web viewer; the model files remain in `output/runs/...`.

- `train/loss` measures error on data updating weights.
- `eval/loss` measures error on the validation split after each epoch.
- Both falling usually indicates useful learning.
- Train falling while validation rises suggests overfitting.
- Flat loss may mean a low learning rate, malformed labels, or a task mismatch.
- Unstable rising loss may mean the learning rate is too high or data is problematic.

Loss measures token confidence. Exact-match accuracy asks whether the complete generated function name is correct. Use both.

## Hyperparameters

The **Train a LoRA adapter** workflow asks for run name, epochs, batch size, learning rate, and seed. Its advanced prompt also exposes maximum sequence length and validation ratio. For automation, its equivalent direct command is:

```bash
.venv/bin/python -m src.training.train \
  --model_name src/model/qwen3_0_6b \
  --train_file output/datasets/qwen3-0-6b/qwen3-functions-v1.jsonl \
  --output_dir output/runs/qwen3-0-6b/qwen3-functions-v1 \
  --epochs 5 \
  --batch_size 4 \
  --lr 2e-4 \
  --seed 42
```

| Setting | CLI flag | Default | Meaning | Adjustment |
|---|---|---:|---|---|
| Run name | prompt only | UTC timestamp | Session folder name | Use a descriptive name for planned trials |
| Epochs | `--epochs` | 8 | Full passes over training data | Stop earlier when validation worsens |
| Batch size | `--batch_size` | 4 | Examples before an optimizer update | Increase for throughput; reduce for memory errors |
| Learning rate | `--lr` | `2e-4` | Optimizer update size | Lower if unstable; cautiously raise if flat |
| Seed | `--seed` | 42 | Split and initialization randomness | Fix for comparisons; vary to estimate uncertainty |
| Base model | `--model_name` | `gpt2` | Local path or Hub ID | Always set explicitly by the menu; change only for deliberate model comparisons |
| Training data | `--train_file` | seed JSONL | Training/validation source | Always set explicitly by the menu; use reviewed balanced data |

`src/training/train.py` also exposes `--max_length` (default 256) and `--eval_ratio` (default 0.2), available from the training workflow's advanced prompt. Longer sequences use more memory. A larger validation ratio stabilizes validation estimates but leaves fewer training examples.

For fair experiments, change one major variable at a time. Keep the base model, test set, prompt format, and seed fixed. Repeat important results over several seeds before trusting a small difference.

## Reading and recording results

Each evaluation is saved as `evaluations/<state>.json` inside the session or base-model folder, for example `evaluations/final.json` or `evaluations/checkpoint-300.json`. It includes every expected and predicted label. `comparison.json` captures base-to-checkpoint progression inside the compared session.

Record measured values, never invented examples:

| Run | Accepted training examples | Epochs | LR | Seed | Base accuracy | Best checkpoint | Final accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Add after checkpoint comparison | - | - | - | - | - | - | - |

Overall accuracy can hide failure. Inspect per-class values and false positives. In function routing, choosing a dangerous function instead of `fn_unknown` may matter more than average accuracy.

## Export for the application

A LoRA adapter needs its base model during inference. Exporting merges both into a standalone model copy:

Choose **Export a standalone model**, then select its base model, training session, and final adapter or checkpoint. The application writes the merged model under that base model's folder in `output/models/`.

Then:

```bash
cd ../Call_Me_Maybe
uv run python -m src \
  --model ../fine_tunning/output/models/qwen3-0-6b/my-export
```

Merging simplifies deployment but consumes full-model storage. Keep adapters when one base model serves several specializations.

## Locate, copy, and publish models

Choose **Show artifact locations** to select a base model, training session, and checkpoint or final adapter. The menu prints absolute, copyable paths for:

- the base model
- the complete training session
- selected adapter and weight file
- `run_config.json`
- TensorBoard logs
- evaluation reports
- the parent folder for merged exports

A LoRA adapter is the small result in `adapter_model.safetensors` plus `adapter_config.json`. It still requires the exact compatible base model. A checkpoint also contains optimizer, scheduler, random-state, and trainer files so training can resume. Those resume files are not necessary for inference.

A merged export contains the base model plus adapter update. It consumes much more disk space, but its directory can be copied to another environment and loaded like a normal local Transformers model.

### Publish to Hugging Face

Authenticate directly in your terminal first so no token passes through the menu:

```bash
.venv/bin/hf auth login
```

Then choose **Publish to Hugging Face**. The menu supports:

- **LoRA adapter**: small upload; checkpoint folders, optimizer state, logs, and evaluations are excluded automatically; users must also obtain the declared base model.
- **Merged model**: large standalone upload from `output/models/<base-model>/`.

Enter a repository ID such as `username/qwen3-function-selector`, choose public or private visibility, inspect the exact command, and confirm. Publishing creates the model repository when needed and uploads the selected folder.

Each completed training session receives a generated model card containing its base model, dataset path and hash, split sizes, hyperparameters, intended use, and limitations. Before making a repository public, add measured evaluation results and verify the base-model license, dataset rights, privacy, safety limitations, and model card wording.

## Other approaches

- **Prompt/few-shot examples**: no training cost, but consume context and may be inconsistent.
- **Constrained decoding**: guarantees allowed output forms; it complements fine-tuning and is already used by the application.
- **Full fine-tuning**: maximum flexibility at much higher memory/storage cost.
- **QLoRA**: lower memory through quantization, with more complexity.
- **Instruction-tuned base**: may route better before task-specific training.
- **Larger model**: potentially stronger understanding at greater operational cost.
- **Distillation**: use a stronger teacher to train the small model.
- **RAG**: useful for frequently changing knowledge, but unnecessary for six fixed labels.

The practical design is usually hybrid: a capable small model, reviewed data, constrained output, deterministic validation, and an explicit unknown path.

## Common problems

- **CUDA out of memory**: reduce batch size or sequence length; later explore gradient accumulation or QLoRA.
- **Loss falls but test accuracy does not**: inspect diversity, leakage, ambiguity, and overfitting.
- **One label dominates predictions**: check class balance and conflicting examples.
- **Session directory exists**: choose a different run name; existing sessions are never overwritten.
- **Comparison is slow**: each checkpoint requires a separate inference pass.

The goal is not simply to produce a high score. It is to understand how data quality, amount of data, training choices, model capacity, and evaluation discipline interact in a complete local LLM experiment.
