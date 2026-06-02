# ALFWorld Plain ReAct Baseline

Minimal ReAct agent for ALFWorld text environments. The baseline intentionally excludes
LangGraph, long-term memory, reflection, replanning, and fine-tuning.

## Smoke Test

Run the deterministic mock environment without installing ALFWorld or configuring an API:

```powershell
python main.py --mock --clear-results
python evaluator.py
```

## Real ALFWorld Run

The text-only ALFWorld dependency chain requires Linux. On Windows, run it inside WSL.
From an Ubuntu terminal:

```bash
cd "/mnt/c/Users/Mr.Orange/Desktop/agent/git-workspace/agent/ReAct_baseline"
bash scripts/setup_text_baseline.sh
source ~/.venvs/react-baseline/bin/activate
python real_env_smoke_test.py
```

The setup script installs the text-only Python dependencies and downloads ALFWorld data
to `~/.cache/alfworld/`. The dataset and virtual environment stay local and are not
committed to Git.

TextWorld supports Python 3.9 through 3.12. Do not use Python 3.13 or newer for the
real ALFWorld environment. The setup script automatically selects a supported interpreter.

Configure an OpenAI-compatible LLM endpoint before a real agent run:

```bash
export ALFWORLD_CONFIG_PATH="$PWD/configs/base_config.yaml"
export ALFWORLD_SPLIT="eval_out_of_distribution"
export LLM_PROVIDER="openai_compatible"
export LLM_MODEL="your-model"
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"
python main.py --episodes 10 --max-steps 30 --clear-results
```

Generated JSONL logs and `evaluation_report.json` are written to `results/` and ignored by Git.
