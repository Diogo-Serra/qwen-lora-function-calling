import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained LoRA adapter into a local model directory."
        )
    )
    parser.add_argument("--base_model", type=str, default="src/model/Qwen2.5-0.5B")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Directory containing the trained LoRA adapter.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../Call_Me_Maybe/models/function_selector_qwen3_0_6b",
        help=(
            "Where to save the exported full model folder."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    model = model.merge_and_unload()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Exported model to: {output_dir}")
    print("You can now run the app with:")
    print(f"  uv run python -m src --model {output_dir}")


if __name__ == "__main__":
    main()
