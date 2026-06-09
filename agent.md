# Agent Rules

This file contains stable project guidance for AI coding agents working in this workspace. Do not put one-off task requests, temporary status, API keys, private credentials, or rapidly changing experiment results here.

## Project Context

- The workspace root is `D:\大模型论文复现`.
- The reference paper PDF is `参考论文1.pdf`, titled `Red Teaming Large Reasoning Models`.
- The URL file points to `https://github.com/edu-yinzhaoxia/Rt-LRM`.
- The downloaded code directory is `Rt-LRM-main/`.
- The current code in `Rt-LRM-main/` implements `LCO: LLM-based Constraint Optimization for Safer Agentic LLMs in Real-world Tasks`, not a direct Rt-LRM benchmark implementation.
- Always verify whether the current task is about the Rt-LRM paper benchmark or the LCO codebase before editing code or running experiments.

## Tech Stack

- Language: Python.
- Prompt composition package: `PromptCode/` installs `procoder`.
- Output refinement experiments: `output-refinement/`.
- Policy refinement experiments: `policy-refinement/`.
- Tool-use simulation framework: modified ToolEmu under `policy-refinement/toolemu/`.
- Common data formats: JSON for output-refinement results, JSONL for policy-refinement trajectories.
- LLM backends: OpenAI-compatible APIs, Anthropic, Qwen/DashScope-compatible APIs, Llama/Together-compatible APIs, optional local vLLM.
- Safety scoring may use Google Perspective API.

## Environment Rules

- Prefer isolated virtual environments for each subproject.
- `policy-refinement` pins `openai==0.28.1` and `langchain==0.0.277`; do not upgrade these casually.
- `PromptCode` must be installed before running `policy-refinement`.
- Run `policy-refinement` scripts from the `policy-refinement/` directory using `python -m`.
- Do not hardcode API keys. Use `.env` or process environment variables.
- Do not commit `.env`, logs, local model weights, large binaries, or generated cache directories.

## Setup Commands

```powershell
cd D:\大模型论文复现\Rt-LRM-main\PromptCode
pip install -e .

cd D:\大模型论文复现\Rt-LRM-main\policy-refinement
pip install -e .

cd D:\大模型论文复现\Rt-LRM-main\output-refinement
pip install choix openai anthropic numpy matplotlib langchain requests python-dotenv scipy pandas seaborn aiohttp
```

## Test Commands

```powershell
cd D:\大模型论文复现\Rt-LRM-main\policy-refinement
python tests/test_env_setup.py

cd D:\大模型论文复现\Rt-LRM-main\output-refinement
python tests/test_process_response.py
python tests/test_benchmark_data.py
```

## Experiment Commands

```powershell
cd D:\大模型论文复现\Rt-LRM-main\output-refinement
python filtering.py --experiment reward_hacking --n_rounds 2 --agent_model gpt-3.5-turbo --judge_model gpt-3.5-turbo --n_judges 1 --seed 0 --agent_idx -1 --method base
python filtering.py --experiment reward_hacking --n_rounds 2 --agent_model gpt-3.5-turbo --judge_model gpt-3.5-turbo --n_judges 1 --seed 0 --agent_idx -1 --method LCO
```

```powershell
cd D:\大模型论文复现\Rt-LRM-main\policy-refinement
python -m scripts.emulate -inp assets/all_cases.json -atp naive -stp adv_thought --agent-model qwen2.5-72b-instruct --simulator-model gpt-4o -si 0 -tn 3 -v -me 3
python -m scripts.emulate -inp assets/all_cases.json -atp naive -stp lco_thought --agent-model qwen2.5-72b-instruct --simulator-model gpt-4o -si 0 -tn 3 -v -me 3 -pos 3
```

## Code Style

- Keep variable names, class names, and module names in English.
- Use `snake_case` for functions and variables.
- Use `CamelCase` for classes.
- Keep imports ordered as standard library, third-party packages, local modules.
- Add comments only when they clarify non-obvious research logic, parsing behavior, or evaluation assumptions.
- Avoid broad refactors unless required for the current task.
- Do not rewrite prompt templates without preserving their evaluation intent.
- Preserve existing CLI argument names unless there is a strong reason to change them.

## Known Pitfalls

- The repository name suggests Rt-LRM, but the current code implements LCO. Treat this as a major project-scope risk.
- PowerShell may not have `git` installed; ZIP download may be used instead.
- Chinese README output may appear garbled in terminals with incompatible encoding; prefer the English README or read files in a UTF-8-aware editor.
- `policy-refinement` imports expect execution from its project root.
- `PromptCode` is required by `policy-refinement`; missing editable install causes runtime import failures.
- vLLM can deadlock when wrapped in thread-pool style execution; keep vLLM inference serialized unless the backend code explicitly supports the concurrency mode.
- Result directories are hardcoded in several scripts.
- Output parsing is fragile around quotes, HTML-like tags, refusals, and explanatory text.
- Automatic safety evaluation can be noisy; use human spot checks for final claims.
- Model APIs, model names, and provider base URLs can change; verify them before large runs.

## Documentation Rules

- Long-term rules belong in this file.
- One-off tasks, current progress, checklists, temporary TODOs, run-specific costs, and final experiment results should go in separate Markdown reports.
- Do not paste third-party documentation wholesale into this file.
- Do not store secrets, private URLs, account-specific paths outside the workspace, or temporary API configuration in this file.
- When adding commands, prefer small smoke-test commands before full experiment commands.
