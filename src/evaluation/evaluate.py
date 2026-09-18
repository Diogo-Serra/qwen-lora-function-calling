import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_function_name(generated_text: str) -> str:
    stripped = generated_text.strip()
    if not stripped:
        return ""
    return stripped.splitlines()[0].split()[0]


def evaluate_model(
    base_model_name: str,
    eval_file: str,
    adapter_dir: str | None = None,
    max_new_tokens: int = 10,
    verbose: bool = True,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = load_dataset("json", data_files=eval_file, split="train")
    correct = 0
    class_totals = defaultdict(int)
    class_correct = defaultdict(int)
    predictions = []

    with torch.no_grad():
        for example in dataset:
            prompt = example["prompt"]
            expected = example["completion"].strip()
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generated[0, encoded.input_ids.shape[1]:]
            generated_text = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
            )
            predicted = extract_function_name(generated_text)

            is_correct = predicted == expected
            correct += int(is_correct)
            class_totals[expected] += 1
            class_correct[expected] += int(is_correct)
            predictions.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": is_correct,
                }
            )
            if verbose:
                request = prompt.splitlines()[0]
                print(f"{request} | expected={expected} predicted={predicted}")

    total = len(dataset)
    result = {
        "model": adapter_dir or base_model_name,
        "base_model": base_model_name,
        "adapter": adapter_dir,
        "eval_file": eval_file,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "per_class_accuracy": {
            label: class_correct[label] / class_totals[label]
            for label in sorted(class_totals)
        },
        "predictions": predictions,
    }
    print(f"Accuracy: {result['accuracy']:.2%} ({correct}/{total})")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a base model or fine-tuned LoRA adapter."
    )
    parser.add_argument("--base_model", type=str, default="src/model/Qwen2.5-0.5B")
    parser.add_argument("--adapter_dir", type=str)
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help=(
            "Path to an evaluation JSONL file, e.g. "
            "src/data/function_name/<name>_test.jsonl"
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--output", type=str)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = evaluate_model(
        base_model_name=args.base_model,
        eval_file=args.eval_file,
        adapter_dir=args.adapter_dir,
        max_new_tokens=args.max_new_tokens,
        verbose=not args.quiet,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
