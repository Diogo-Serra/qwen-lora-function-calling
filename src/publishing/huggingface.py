import argparse
from pathlib import Path

from huggingface_hub import HfApi


ADAPTER_IGNORE_PATTERNS = [
    "checkpoint-*",
    "checkpoint-*/**",
    "evaluations/*",
    "evaluations/**",
    "logs/*",
    "logs/**",
    "runs/*",
    "runs/**",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a trained adapter or merged model to Hugging Face."
    )
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--adapter_only",
        action="store_true",
        help="Exclude checkpoints, optimizer state, logs, and evaluations.",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Artifact folder not found: {folder}")

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=(
            ADAPTER_IGNORE_PATTERNS if args.adapter_only else None
        ),
    )
    print(f"Published model: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
