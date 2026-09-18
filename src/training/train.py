import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


# LoRA target module names differ per base model architecture. GPT-2 style
# models expose fused "c_attn"/"c_proj" layers, while Qwen/Llama style models
# expose separate q/k/v/o and MLP gate/up/down projections.
LORA_TARGET_MODULES = {
    "gpt2": ["c_attn", "c_proj"],
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def resolve_target_modules(model) -> list[str]:
    model_type = getattr(model.config, "model_type", "").lower()
    if model_type in LORA_TARGET_MODULES:
        return LORA_TARGET_MODULES[model_type]
    # Fall back to the GPT-2 default to preserve previous behavior.
    return LORA_TARGET_MODULES["gpt2"]


FUNCTIONS = [
    "fn_add_numbers",
    "fn_greet",
    "fn_reverse_string",
    "fn_get_square_root",
    "fn_substitute_string_with_regex",
    "fn_unknown",
]


def build_prompt(example: dict) -> str:
    prompt = example["prompt"]
    if not prompt.endswith("\n"):
        prompt += "\n"
    return prompt


def tokenize_example(
    example: dict,
    tokenizer,
    max_length: int = 256,
):
    prompt = build_prompt(example)
    completion = example["completion"].strip() + "\n"

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    completion_ids = tokenizer(
        completion,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def portable_reference(value: str) -> str:
    path = Path(value)
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return value


def write_model_card(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_hash: str,
    train_examples: int,
    validation_examples: int,
) -> None:
    model_name = portable_reference(args.model_name)
    train_file = portable_reference(args.train_file)
    base_model = json.dumps(model_name)
    card = f"""---
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- function-calling
- function-selection
---

# Function Selection LoRA Adapter

This LoRA adapter specializes `{model_name}` for selecting one function
name from the function definitions included in each prompt.

## Training

- Dataset: `{train_file}`
- Dataset SHA-256: `{dataset_hash}`
- Training examples: {train_examples}
- Validation examples: {validation_examples}
- Epochs: {args.epochs}
- Batch size: {args.batch_size}
- Learning rate: {args.lr}
- Maximum sequence length: {args.max_length}
- Validation ratio: {args.eval_ratio}
- Seed: {args.seed}

Only completion tokens contributed to loss. The base weights remained frozen;
LoRA targeted the model's attention projections.

## Intended Use

Use this adapter with its recorded base model for function-name selection.
Validate generated names and arguments in application code.

## Limitations

The adapter is task-specific and inherits limitations from its base model and
training data. Review held-out evaluation results before deployment. It does
not guarantee safe function execution or valid arguments.
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a small causal LM for function selection."
    )
    parser.add_argument("--model_name", type=str, default="src/model/Qwen2.5-0.5B")
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help=(
            "Path to a training JSONL file, e.g. "
            "src/data/function_name/<name>_dataset.jsonl"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/runs/qwen2-5-0-5b/manual-run",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--eval_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = Path(args.train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Dataset not found: {train_path}")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {output_dir}. "
            "Choose a new run name to preserve previous results."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    ds = load_dataset("json", data_files=str(train_path), split="train")
    ds = ds.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_ds = ds["train"]
    eval_ds = ds["test"]

    dataset_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    recorded_args = vars(args) | {
        "model_name": portable_reference(args.model_name),
        "train_file": portable_reference(args.train_file),
        "output_dir": portable_reference(args.output_dir),
    }
    run_config = recorded_args | {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_file_sha256": dataset_hash,
        "train_examples": len(train_ds),
        "validation_examples": len(eval_ds),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = train_ds.map(
        lambda ex: tokenize_example(ex, tokenizer, args.max_length),
        batched=False,
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda ex: tokenize_example(ex, tokenizer, args.max_length),
        batched=False,
        remove_columns=eval_ds.column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=resolve_target_modules(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # TrainingArguments no longer accepts logging_dir (transformers >= 5.0);
    # TensorBoardCallback reads this env var instead.
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(output_dir / "logs")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            report_to="tensorboard",
            load_best_model_at_end=False,
            remove_unused_columns=False,
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        # DataCollatorWithPadding only pads input_ids/attention_mask, not the
        # per-token "labels" sequence used here to mask the prompt (-100).
        # DataCollatorForSeq2Seq pads labels too (with -100), which is what
        # this prompt/completion masking setup needs.
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            label_pad_token_id=-100,
        ),
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_model_card(
        output_dir,
        args,
        dataset_hash,
        len(train_ds),
        len(eval_ds),
    )
    print(f"Training complete. Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
