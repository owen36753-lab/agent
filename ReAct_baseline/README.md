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

Install ALFWorld and PyYAML in your Python environment, then configure:

```powershell
$env:ALFWORLD_CONFIG_PATH = "C:\path\to\alfworld\config.yaml"
$env:ALFWORLD_SPLIT = "eval_out_of_distribution"
$env:LLM_PROVIDER = "openai_compatible"
$env:LLM_MODEL = "your-model"
$env:LLM_API_KEY = "your-api-key"
$env:LLM_BASE_URL = "https://api.openai.com/v1"
python main.py --episodes 10 --max-steps 30 --clear-results
```

Generated JSONL logs and `evaluation_report.json` are written to `results/` and ignored by Git.
