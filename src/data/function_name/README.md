# Function-selection dataset

This folder holds the converted, ready-to-train JSONL files for the
function-selection task. No dataset ships here by default; add your own
before training.

Put two files here:

- `<name>_dataset.jsonl` — training data. Split into train/validation
  automatically by `--eval_ratio` during training.
- `function_name_test.jsonl` — a fixed, held-out benchmark. Never used
  for training, only for evaluation and comparison. Use this exact file
  name so the menu recognizes it, lists it first when picking an
  evaluation dataset, and excludes it from the training dataset list.

Each line is one JSON object:

```json
{"prompt":"Request: \"Add 3 and 7\"\nAvailable functions:\n- fn_add_numbers: Add two numbers together and return their sum.\n- fn_greet: Generate a greeting message for a person by name.\n- fn_unknown: Use this when the request does not clearly match any available function.\nThe function to call is:\n","completion":"fn_add_numbers\n"}
```

To bring in a real dataset (for example from
[Hugging Face Datasets](https://huggingface.co/datasets)): drop the raw,
unconverted file into `src/data/raw/` first, then ask Copilot CLI to
convert it into the format above and save the result here.

Do not copy the fixed test file's examples into the training file — that
is data leakage and inflates evaluation scores without real learning.
