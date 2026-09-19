import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import questionary
from questionary import Choice, Separator, Style


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = SRC_DIR / "data"
MODEL_DIR = SRC_DIR / "model"
OUTPUT_DIR = PROJECT_ROOT / "output"
RUNS_DIR = OUTPUT_DIR / "runs"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
EVALUATION_OUTPUT_DIR = OUTPUT_DIR / "evaluations"
DEFAULT_TEST_FILE = DATA_DIR / "function_name" / "function_name_test.jsonl"

BACK = "__back__"  # Choice value meaning "return to the previous step"

MENU_STYLE = Style(
    [
        ("qmark", "fg:#2aa198 bold"),
        ("question", "bold"),
        ("answer", "fg:#268bd2 bold"),
        ("pointer", "fg:#2aa198 bold"),
        ("highlighted", "fg:#002b36 bg:#93a1a1 bold"),
        ("selected", "fg:#859900"),
        ("instruction", "fg:#657b83"),
        ("separator", "fg:#b58900 bold"),
        ("disabled", "fg:#586e75 italic"),
    ]
)

BANNER_STYLE = "fg:#268bd2 bold"


def section(title: str) -> Separator:
    """A visually distinct, non-selectable heading inside a choice list."""
    return Separator(f"\u2500\u2500 {title.upper()} \u2500\u2500")


def print_banner(title: str) -> None:
    rule = "\u2500" * (len(title) + 4)
    questionary.print(f"\n{rule}", style=BANNER_STYLE)
    questionary.print(f"  {title}", style=BANNER_STYLE)
    questionary.print(f"{rule}", style=BANNER_STYLE)


@dataclass
class SessionContext:
    workflow: str = "Main menu"
    model: str | None = None
    dataset: Path | None = None
    run: Path | None = None
    checkpoint: str | None = None
    parameters: str | None = None
    output: Path | str | None = None
    tensorboard_url: str | None = None

    def begin(self, workflow: str) -> None:
        self.workflow = workflow
        self.model = None
        self.dataset = None
        self.run = None
        self.checkpoint = None
        self.parameters = None
        self.output = None


SESSION = SessionContext()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def short_value(value: Path | str) -> str:
    if isinstance(value, Path):
        return relative(value)
    candidate = Path(value)
    if candidate.exists():
        return relative(candidate)
    return value


def status_toolbar() -> str:
    """Recap only what has actually been picked so far; nothing pending."""
    lines = [f" Step: {SESSION.workflow}"]
    fields = [
        ("Model", SESSION.model),
        ("Dataset", SESSION.dataset),
        ("Session", SESSION.run),
        ("Checkpoint", SESSION.checkpoint),
        ("Settings", SESSION.parameters),
        ("Output", SESSION.output),
        ("TensorBoard", SESSION.tensorboard_url),
    ]
    for label, value in fields:
        if value is None:
            continue
        display = value if label in ("Checkpoint", "Settings", "TensorBoard") else short_value(value)
        lines.append(f" {label}: {display}")
    return "\n".join(lines)


def prompt_options() -> dict:
    return {
        "style": MENU_STYLE,
        "bottom_toolbar": status_toolbar,
    }


def ask_select(message: str, choices: list, show_back: bool = True):
    """Ask an arrow-key select question; None means cancelled or 'Back'."""
    items = list(choices)
    if show_back:
        items.extend([Separator(), Choice("\u25c0 Back", BACK)])
    # questionary.select() already reserves "bottom_toolbar" internally
    # (for its own validation messages), so the recap is printed above
    # the prompt instead of passed in as a kwarg.
    recap = status_toolbar()
    if recap.strip():
        questionary.print(recap, style="fg:#657b83")
    selected = questionary.select(
        message,
        choices=items,
        style=MENU_STYLE,
    ).ask()
    return None if selected in (None, BACK) else selected


def ask_confirm(message: str, default: bool = True) -> bool | None:
    return questionary.confirm(message, default=default, **prompt_options()).ask()


def press_enter_to_continue(message: str = "Press any key to continue...") -> None:
    questionary.press_any_key_to_continue(message, **prompt_options()).ask()


def explain_action(title: str, *lines: str) -> None:
    questionary.print(f"\n{title}", style="fg:#859900 bold")
    questionary.print("\u2500" * len(title), style="fg:#859900")
    for line in lines:
        print(line)
    print()


def model_slug(model_name: str) -> str:
    short_name = model_name.rstrip("/").rsplit("/", maxsplit=1)[-1]
    slug = re.sub(r"[^a-z0-9]+", "-", short_name.lower()).strip("-")
    return slug or "model"


def discover_models() -> list[Path]:
    candidates = []
    for root in (MODEL_DIR, MODEL_OUTPUT_DIR):
        if root.exists():
            candidates.extend(root.rglob("config.json"))
    return sorted({path.parent.resolve() for path in candidates})


def discover_datasets() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted({path.resolve() for path in DATA_DIR.rglob("*.jsonl")})


def training_datasets() -> list[Path]:
    test_path = DEFAULT_TEST_FILE.resolve()
    return [path for path in discover_datasets() if path != test_path]


def evaluation_datasets() -> list[Path]:
    datasets = discover_datasets()
    test_path = DEFAULT_TEST_FILE.resolve()
    return sorted(datasets, key=lambda path: path != test_path)


def discover_runs(base_model: str | None = None) -> list[Path]:
    search_root = (
        RUNS_DIR / model_slug(base_model) if base_model else RUNS_DIR
    )
    if not search_root.exists():
        return []
    runs = []
    for config_path in search_root.rglob("adapter_config.json"):
        path = config_path.parent
        if path.name == "latest" or path.name.startswith("checkpoint-"):
            continue
        runs.append(path.resolve())
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def discover_reports() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []
    report_names = {
        "base_qwen.json",
        "comparison.json",
        "evaluation.json",
    }
    reports = [
        path
        for path in OUTPUT_DIR.rglob("*.json")
        if path.name in report_names
        or path.name.endswith((".audit.json", ".report.json"))
        or "evaluations" in path.parts
    ]
    return sorted(reports, key=lambda path: str(path))


def ask_text(message: str, default: str = "") -> str | None:
    return questionary.text(message, default=default, **prompt_options()).ask()


def ask_number(
    message: str,
    default: str,
    value_type: type = int,
) -> str | None:
    def validate(value: str) -> bool | str:
        try:
            number = value_type(value)
        except ValueError:
            return f"Enter a valid {value_type.__name__}."
        if number <= 0:
            return "Enter a value greater than zero."
        return True

    return questionary.text(
        message,
        default=default,
        validate=validate,
        **prompt_options(),
    ).ask()


def select_path(
    message: str,
    paths: list[Path],
    allow_custom: bool = True,
) -> Path | None:
    choices = [Choice(relative(path), path) for path in paths]
    if allow_custom:
        choices.append(Choice("Enter another path", "custom"))
    if not choices:
        print("No matching files were found.")
        return None
    selected = ask_select(message, choices)
    if selected is None:
        return None
    if selected == "custom":
        entered = ask_text("Path:")
        return Path(entered) if entered else None
    return selected


def select_model(message: str = "Select a model") -> str | None:
    models = discover_models()
    choices = [Choice(relative(path), relative(path)) for path in models]
    choices.append(Choice("Enter a Hugging Face model ID or path", "custom"))
    selected = ask_select(message, choices)
    if selected == "custom":
        selected = ask_text("Model ID or path:")
    if selected:
        SESSION.model = selected
    return selected


def select_run(base_model: str) -> Path | None:
    selected = select_path(
        "Select a training session",
        discover_runs(base_model),
        False,
    )
    if selected is not None:
        SESSION.run = selected
    return selected


def select_adapter(run_dir: Path) -> tuple[str, Path] | None:
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.removeprefix("checkpoint-")),
    )
    choices = [Choice("Final adapter", ("final", run_dir))]
    choices.extend(
        Choice(f"Checkpoint {path.name.removeprefix('checkpoint-')}",
               (path.name, path))
        for path in checkpoints
    )
    selected = ask_select("Select the model state to evaluate", choices)
    if selected is not None:
        SESSION.checkpoint = selected[0]
    return selected


def run_command(command: list[str], description: str) -> bool:
    print(f"\nCurrent selection\n{status_toolbar()}")
    print(f"\n{description}")
    print(f"$ {shlex.join(command)}\n")
    confirmed = ask_confirm("Run this command?", default=True)
    if not confirmed:
        return False
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    except KeyboardInterrupt:
        print("\nCommand interrupted.")
        return False
    if completed.returncode == 0:
        print("\nCompleted successfully.")
        return True
    print(f"\nCommand failed with exit code {completed.returncode}.")
    return False


def module_command(module: str, *arguments: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(value) for value in arguments)]


def action_train() -> None:
    SESSION.begin("Train LoRA adapter")
    explain_action(
        "Train a function-selection model",
        "The base model stays unchanged. Each training session receives its",
        "own adapter, configuration, epoch checkpoints, and TensorBoard logs.",
    )
    model = select_model("Select the base model to fine-tune")
    if not model:
        return
    dataset = select_path("Select the training dataset", training_datasets())
    if dataset is None:
        return
    SESSION.dataset = dataset
    default_name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = ask_text("Run name:", default_name)
    epochs = ask_number("Epochs:", "8")
    batch_size = ask_number("Batch size:", "4")
    learning_rate = ask_number("Learning rate:", "0.0002", float)
    seed = ask_number("Random seed:", "42")
    if not all((run_name, epochs, batch_size, learning_rate, seed)):
        return
    safe_run_name = Path(run_name).name
    if safe_run_name in (".", "..", "latest"):
        print("Choose a specific run name other than 'latest'.")
        return
    model_runs_dir = RUNS_DIR / model_slug(model)
    run_dir = model_runs_dir / safe_run_name
    SESSION.run = run_dir
    SESSION.checkpoint = "new session"
    SESSION.parameters = (
        f"{epochs} epochs; batch {batch_size}; LR {learning_rate}; seed {seed}"
    )
    SESSION.output = run_dir
    command = module_command(
        "src.training.train",
        "--model_name",
        model,
        "--train_file",
        dataset,
        "--output_dir",
        run_dir,
        "--epochs",
        epochs,
        "--batch_size",
        batch_size,
        "--lr",
        learning_rate,
        "--seed",
        seed,
    )
    if ask_confirm("Configure sequence length and validation split?", default=False):
        max_length = ask_number("Maximum sequence length:", "256")
        eval_ratio = ask_number("Validation ratio:", "0.2", float)
        if not max_length or not eval_ratio:
            return
        command.extend(
            [
                "--max_length",
                max_length,
                "--eval_ratio",
                eval_ratio,
            ]
        )
        SESSION.parameters += (
            f"; max length {max_length}; validation {eval_ratio}"
        )
    if run_command(command, "Train a new LoRA adapter"):
        model_runs_dir.mkdir(parents=True, exist_ok=True)
        latest = model_runs_dir / "latest"
        latest.unlink(missing_ok=True)
        latest.symlink_to(run_dir.resolve(), target_is_directory=True)
        print("\nTraining artifacts")
        print(f"Session      : {run_dir.resolve()}")
        adapter_weights = (run_dir / "adapter_model.safetensors").resolve()
        print(f"Final adapter: {adapter_weights}")
        print(f"Checkpoints  : {run_dir.resolve()}/checkpoint-*")
        print(f"Logs         : {(run_dir / 'logs').resolve()}")
        print(f"Latest link  : {latest.resolve()}")
        print(
            "Next steps   : evaluate, compare checkpoints, or open "
            "TensorBoard."
        )


def action_evaluate() -> None:
    SESSION.begin("Evaluate model")
    explain_action(
        "Evaluate exact-match accuracy",
        "Use the held-out test set to compare the untouched base model, a",
        "final adapter, or one checkpoint from a training session.",
    )
    model = select_model("Select the base model")
    if not model:
        return
    test_file = select_path(
        "Select the evaluation dataset",
        evaluation_datasets(),
    )
    if test_file is None:
        return
    SESSION.dataset = test_file
    target = ask_select(
        "What should be evaluated?",
        ["Untouched base model", "Training run or checkpoint"],
    )
    command = module_command(
        "src.evaluation.evaluate",
        "--base_model",
        model,
        "--eval_file",
        test_file,
    )
    if target == "Untouched base model":
        SESSION.checkpoint = "base model"
        report = EVALUATION_OUTPUT_DIR / model_slug(model) / "base.json"
    elif target == "Training run or checkpoint":
        run_dir = select_run(model)
        if run_dir is None:
            return
        selected = select_adapter(run_dir)
        if selected is None:
            return
        state_name, adapter_dir = selected
        report = run_dir / "evaluations" / f"{state_name}.json"
        command.extend(["--adapter_dir", adapter_dir])
    else:
        return
    SESSION.output = report
    command.extend(["--output", report])
    run_command(command, "Evaluate exact-match function selection")


def action_compare() -> None:
    SESSION.begin("Compare checkpoints")
    explain_action(
        "Compare one training session",
        "Runs the same held-out evaluation on the base model, every saved",
        "checkpoint, and the final adapter, then writes comparison.json.",
    )
    model = select_model("Select the base model")
    if not model:
        return
    run_dir = select_run(model)
    if run_dir is None:
        return
    test_file = select_path(
        "Select the evaluation dataset",
        evaluation_datasets(),
    )
    if test_file is None:
        return
    SESSION.dataset = test_file
    SESSION.checkpoint = "all checkpoints"
    SESSION.output = run_dir / "comparison.json"
    command = module_command(
        "src.evaluation.compare",
        "--base_model",
        model,
        "--run_dir",
        run_dir,
        "--eval_file",
        test_file,
    )
    run_command(command, "Compare base, checkpoints, and final adapter")


def action_tensorboard() -> None:
    SESSION.begin("View training curves")
    explain_action(
        "Open TensorBoard",
        "TensorBoard reads the selected session's logs and serves curves in",
        "your browser. Keep this terminal open; press Ctrl+C to stop it.",
    )
    model = select_model("Select the base model")
    if not model:
        return
    run_dir = select_run(model)
    if run_dir is None:
        return
    port = ask_number("TensorBoard port:", "6006")
    if not port:
        return
    SESSION.tensorboard_url = f"http://localhost:{port}"
    SESSION.output = run_dir / "logs"
    command = module_command(
        "tensorboard.main",
        "--logdir",
        run_dir,
        "--port",
        port,
    )
    run_command(command, "Start TensorBoard; press Ctrl+C to return")
    SESSION.tensorboard_url = None


def action_view_report() -> None:
    SESSION.begin("View saved report")
    report = select_path(
        "Select a saved JSON report",
        discover_reports(),
        False,
    )
    if report is None:
        return
    SESSION.output = report
    try:
        content = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read {relative(report)}: {error}")
        return
    print(f"\n{relative(report)}\n")
    print(json.dumps(content, indent=2))


def action_export() -> None:
    SESSION.begin("Export trained model")
    explain_action(
        "Export a standalone model",
        "Merges one LoRA adapter into its base model. The exported folder is",
        "larger, but can be copied and loaded without the separate adapter.",
    )
    model = select_model("Select the base model")
    if not model:
        return
    run_dir = select_run(model)
    if run_dir is None:
        return
    selected = select_adapter(run_dir)
    if selected is None:
        return
    state_name, adapter_dir = selected
    base_slug = model_slug(model)
    default_name = f"{run_dir.name}-{state_name}"
    name = ask_text("Export folder name:", default_name)
    if not name:
        return
    command = module_command(
        "src.training.export_model",
        "--base_model",
        model,
        "--adapter_dir",
        adapter_dir,
        "--output_dir",
        MODEL_OUTPUT_DIR / base_slug / Path(name).name,
    )
    SESSION.output = MODEL_OUTPUT_DIR / base_slug / Path(name).name
    run_command(command, "Merge and export a standalone model")


def action_show_artifacts() -> None:
    SESSION.begin("Locate trained artifacts")
    explain_action(
        "Locate a trained model",
        "Select a session and model state to display exact copyable paths.",
        "A LoRA adapter is small but still requires its original base model.",
        "A merged export is larger but is easier to move to another system.",
    )
    model = select_model("Select the base model")
    if not model:
        return
    run_dir = select_run(model)
    if run_dir is None:
        return
    selected = select_adapter(run_dir)
    if selected is None:
        return
    state_name, adapter_dir = selected
    SESSION.output = adapter_dir
    export_root = MODEL_OUTPUT_DIR / model_slug(model)
    checkpoints = list(run_dir.glob("checkpoint-*"))

    print("\nSelected artifact locations")
    print("===========================")
    print(f"Base model       : {model}")
    print(f"Training session : {run_dir}")
    print(f"Selected state   : {state_name}")
    print(f"Adapter folder   : {adapter_dir}")
    print(f"Adapter weights  : {adapter_dir / 'adapter_model.safetensors'}")
    print(f"Session metadata : {run_dir / 'run_config.json'}")
    print(f"TensorBoard logs : {run_dir / 'logs'}")
    print(f"Evaluations      : {run_dir / 'evaluations'}")
    print(f"Saved checkpoints: {len(checkpoints)}")
    print(f"Merged exports   : {export_root}")
    print("\nUse 'Export a trained model' for a standalone portable folder.")


def print_resources() -> None:
    groups = {
        "Models": discover_models(),
        "Datasets": discover_datasets(),
        "Training runs": discover_runs(),
        "JSON reports": discover_reports(),
    }
    for heading, paths in groups.items():
        questionary.print(
            f"{heading.upper()} ({len(paths)})", style="fg:#b58900 bold"
        )
        for path in paths:
            print(f"  {relative(path)}")
        print()


def interactive_menu() -> None:
    actions = {
        "train": action_train,
        "evaluate": action_evaluate,
        "compare": action_compare,
        "tensorboard": action_tensorboard,
        "locations": action_show_artifacts,
        "report": action_view_report,
        "export": action_export,
    }
    choices = [
        section("Training"),
        Choice("Train a LoRA adapter", "train"),
        Separator(),
        section("Evaluation"),
        Choice("Evaluate one model or checkpoint", "evaluate"),
        Choice("Compare all checkpoints in a session", "compare"),
        Choice("Open TensorBoard loss curves", "tensorboard"),
        Choice("View a saved JSON report", "report"),
        Separator(),
        section("Deployment"),
        Choice("Show artifact locations", "locations"),
        Choice("Export a standalone model", "export"),
        Separator(),
        Choice("Exit", "exit"),
    ]
    while True:
        SESSION.begin("Main menu")
        print_banner("Local LLM Fine-tuning Lab")
        selection = ask_select("Choose a workflow", choices, show_back=False)
        if selection in (None, "exit"):
            print("Goodbye.")
            return
        actions[selection]()
        press_enter_to_continue("Press Enter to return to the main menu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive local LLM fine-tuning lab."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered resources and exit without opening the menu.",
    )
    args = parser.parse_args()
    if args.list:
        print_resources()
        return
    interactive_menu()
