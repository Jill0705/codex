# Repository Guidelines

## Project Structure & Module Organization

The active codebase lives in `work/aspnet_experiments/`. The training entry point is `train.py`; model, data, classifier, loss, and utility code live in `aspnet_lt/`. PowerShell experiment runners and exporters are in `scripts/`. Generated outputs go to `runs/`, logs to `logs/`, and paper-ready CSV summaries to `tables/`. Dataset files belong in `data/`; do not commit large datasets, checkpoints, or `runs/*/best.pt` unless requested.

## Build, Test, and Development Commands

Use the project directory before running commands:

```powershell
cd C:\Users\x2472\Documents\Codex\2026-07-04\new-chat-2\work\aspnet_experiments
```

Install lightweight dependencies with `pip install -r requirements.txt`. Install PyTorch separately for the target CUDA/Python environment.

Useful checks:

```powershell
python -m py_compile train.py aspnet_lt\*.py
python train.py --dataset synthetic --model adaptive_proto --num-classes 10 --epochs 1 --batch-size 64 --synthetic-size 512 --device cpu
.\scripts\export_paper_tables.ps1
.\scripts\export_ccfb_tables.ps1
```

For GPU experiments, prefer `C:\Users\x2472\miniconda3\envs\torch\python.exe`, which the scripts already use.

## Coding Style & Naming Conventions

Write Python with 4-space indentation, clear function names, and small helpers for reusable experiment logic. Keep CLI flags descriptive and lowercase, such as `--proto-budget`, `--allocation`, and `--imb-factor`. Use ASCII unless an existing file requires otherwise. Prefer vectorized PyTorch operations over Python loops in training-critical paths.

## Testing Guidelines

Before long GPU runs, run a CPU synthetic smoke test and `py_compile`. New experiment scripts should skip runs whose `metrics.csv` already reached the target epoch. Preserve exporter fields: `val_acc`, `many_acc`, `medium_acc`, `few_acc`, `prototype_counts.json`, and `args.json`.

## Commit & Pull Request Guidelines

No reliable git history is available here, so use concise imperative messages, for example `Add balanced softmax baseline script`. PRs should summarize the experiment purpose, changed paths, smoke-test commands, and new generated tables. Do not mix unrelated paper-note edits with training-code changes.

## Agent Workflow Guidelines

Follow a cautious, Karpathy-inspired workflow:

- Think before coding: state assumptions, surface ambiguity, and ask when the task has multiple plausible interpretations.
- Simplicity first: implement the minimum code needed; avoid speculative abstractions, unused configurability, and broad rewrites.
- Surgical changes: touch only files required by the request, match existing style, and remove only dead code introduced by your own edits.
- Goal-driven execution: define a concrete verification step for each change, such as `py_compile`, a synthetic smoke test, or a table export.

Keep the paper's current main line focused on adaptive sub-prototype allocation and prototype budget efficiency. Treat EMA, confidence gating, Hungarian matching, and logit adjustment as negative or supplemental modules unless new evidence justifies revisiting them.
