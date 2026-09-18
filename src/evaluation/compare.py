import argparse
import gc
import json
from pathlib import Path

import torch

from .evaluate import evaluate_model


def checkpoint_step(path: Path) -> int:
    return int(path.name.removeprefix("checkpoint-"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base Qwen with every checkpoint in one run."
    )
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument(
        "--eval_file",
        type=str,
        default="src/data/function_name/function_name_test.jsonl",
    )
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Trained adapter not found in: {run_dir}")

    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=checkpoint_step,
    )
    candidates = [("base", None)]
    candidates.extend((path.name, str(path)) for path in checkpoints)
    candidates.append(("final", str(run_dir)))

    results = []
    for name, adapter_dir in candidates:
        print(f"\nEvaluating {name}...")
        result = evaluate_model(
            base_model_name=args.base_model,
            eval_file=args.eval_file,
            adapter_dir=adapter_dir,
            verbose=False,
        )
        results.append(
            {
                "name": name,
                "adapter": adapter_dir,
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
                "per_class_accuracy": result["per_class_accuracy"],
            }
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = (
        Path(args.output) if args.output else run_dir / "comparison.json"
    )
    output_path.write_text(
        json.dumps({"results": results}, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nModel                 Accuracy")
    print("--------------------  --------")
    for result in results:
        print(f"{result['name']:<20}  {result['accuracy']:>7.2%}")
    print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()
