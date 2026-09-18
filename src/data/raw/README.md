# Raw datasets (unconverted)

Drop datasets here in whatever format you got them: a Hugging Face dataset
export, CSV, plain JSON, parquet, a zipped folder, etc. Nothing in this
folder is read by the menu or by training/evaluation directly.

Ask Copilot CLI to convert a file here into the project's JSONL format
(`{"prompt": ..., "completion": ...}`, matching
`src/data/function_name/function_name_dataset.jsonl`) and save the result
under `src/data/<task>/` or `output/datasets/<base-model>/` so it shows up
in the **Train a LoRA adapter** / **Evaluate one model or checkpoint**
menus.

Contents of this folder are not committed to Git (see `.gitignore`); only
this README and `.gitkeep` are tracked.
