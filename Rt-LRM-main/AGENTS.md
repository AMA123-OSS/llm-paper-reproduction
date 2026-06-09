# AGENTS.md — LCO: LLM-based Constraint Optimization for Safer Agentic LLMs

> This file is intended for AI coding agents. It assumes you know nothing about the project.

## 1. Project Overview

This repository implements **LCO (LLM-based Constraint Optimization)**, a defense framework against **In-Context Reward Hacking (ICRH)** in LLM agents. ICRH occurs when an agent over-optimizes for a measurable reward (e.g., tweet engagement) and gradually violates implicit safety constraints.

The codebase is organized into three independent but related subprojects:

| Directory | Purpose |
|-----------|---------|
| `PromptCode/` | A small Python package (`procoder`) for composing modular, hierarchical prompts with cross-referencing. Used by `policy-refinement/`. |
| `output-refinement/` | Experiments on **content generation** (tweet engagement optimization). Measures defense effectiveness via toxicity growth rate (TGR). |
| `policy-refinement/` | Experiments on **tool-use agents** using the ToolEmu framework. Measures defense effectiveness via ICRH Occurrence Rate (IOR). |

The project is a research codebase. There is no CI/CD, no formal test suite, and no containerized deployment.

## 2. Technology Stack

- **Language**: Python 3.8+
- **LLM Backends**:
  - OpenAI API (GPT-3.5, GPT-4)
  - Anthropic API (Claude)
  - Together / DashScope (Llama, Qwen)
  - Local inference via **vLLM** (with tensor-parallelism support)
- **Key Libraries**:
  - `output-refinement/`: `openai`, `anthropic`, `numpy`, `scipy`, `matplotlib`, `aiohttp`, `choix`, `requests`, `python-dotenv`
  - `policy-refinement/`: `langchain==0.0.277`, `openai==0.28.1`, `anthropic==0.3.6`, `transformers`, `torch`, `tiktoken`, `fire`
  - `PromptCode/`: `python-dotenv==1.0.0`, `roman`, `black`
- **External APIs**: Google Perspective API (toxicity scoring)
- **Data Formats**: JSON (output-refinement results), JSONL (policy-refinement trajectories)

## 3. Repository Structure

```
.
├── PromptCode/              # Prompt management library (installable package)
│   ├── procoder/            # Core prompt composition modules
│   ├── setup.py
│   └── requirements.txt
├── output-refinement/       # Tweet engagement / content generation experiments
│   ├── filtering.py         # Main synchronous experiment runner
│   ├── async_filtering.py   # Async API variant
│   ├── run_experiments.py   # Batch scheduler across seeds
│   ├── Genetic_optimization.py  # Parallel evolutionary lineages
│   ├── toxicity.py          # Google Perspective API wrapper
│   ├── pairwise_voting.py   # Post-hoc engagement scoring
│   ├── compute_ICRH.py      # Statistical analysis (Kendall's tau, t-test)
│   ├── compute_helpfulness.py
│   ├── vllm_backend.py      # Local model abstraction (~540 lines)
│   ├── api_keys.py          # Centralized credential loader
│   ├── reward_hacking/      # Tweet scenario prompts & data
│   ├── optimization/        # General optimization scenario
│   └── *.sh                 # Quick benchmark wrappers
├── policy-refinement/       # Tool-use experiments (ToolEmu-based)
│   ├── setup.py             # Installs `toolemu` package
│   ├── toolemu/             # Core framework
│   │   ├── agents/          # Agent executors & simulators
│   │   ├── prompts/         # Prompt modules (agent, simulator, evaluator)
│   │   ├── tools/           # Virtual tool definitions (~16k lines)
│   │   ├── utils/           # LLM wrappers, I/O, threading helpers
│   │   └── executors/       # Batch execution abstractions
│   ├── self_reflection_agent.py
│   ├── self_defense_agent.py
│   ├── goal_priority_agent.py
│   ├── safety_prompt_agent.py
│   ├── MyAgent.py           # Genetic / LCO / Seco agents
│   ├── agent_executor_builder.py
│   ├── evaluators.py        # LLM-as-judge evaluators
│   ├── generators.py        # Test-case generators
│   └── scripts/             # CLI entry points
│       ├── emulate.py
│       ├── evaluate.py
│       └── run.py
├── README.md
├── README_zh.md
├── modify.md                # Internal paper revision plan (Chinese)
└── test.py                  # Standalone toxicity visualization tool
```

## 4. Build and Setup Commands

There is **no single top-level build system**. Each subdirectory must be set up independently.

### 4.1 PromptCode
```bash
cd PromptCode
pip install -e .
```

### 4.2 Output-Refinement
```bash
cd output-refinement
# Core dependencies
pip install choix openai anthropic numpy matplotlib langchain requests python-dotenv scipy aiohttp
# Optional: vLLM support
pip install -r requirements_vllm.txt   # vllm>=0.4.0, torch>=2.0.0, transformers>=4.35.0
```

### 4.3 Policy-Refinement
```bash
cd policy-refinement
pip install -e .
```

### 4.4 API Keys
- **Output-refinement**: Edit `output-refinement/api_keys.py` or create a `.env` file.
- **Policy-refinement**: Create a `.env` file with `OPENAI_API_KEY=your_key`.

> **Security Warning**: Hardcoded API keys and base URLs appear in several files (e.g., `toolemu/utils/llm.py`, `self_reflection_agent.py`, `self_defense_agent.py`, `MyAgent.py`, `api_keys.py`). Do not commit real keys.

## 5. Code Organization and Architecture

### 5.1 Output-Refinement
The main entry point is `filtering.py`. It runs a multi-round feedback loop where agents generate tweets, a judge selects a winner, and toxicity is measured.

**Key architectural patterns:**
- **Global `args` pattern**: `filtering.py` and submodules reference a module-level `args` object inside functions rather than passing parameters explicitly.
- **Thread-safety dichotomy**:
  - Cloud API paths use `ThreadPoolExecutor` aggressively.
  - vLLM paths force **serial execution** because vLLM's multiprocessing (`spawn`) deadlocks with Python thread pools. A global `_VLLM_LOCK` in `vllm_backend.py` protects inference.
- **Response parsing**: `process_response()` in `reward_hacking/utils.py` strips HTML tags, extracts quoted text, detects refusal phrases, and removes polite prefixes.
- **Model aliasing**: `replace_model_name()` maps shorthands (`"gpt"` → `"gpt-4-0613"`, `"qwen"` → `"qwen2.5-72b-instruct"`, `"llama"` → `"llama-3.1-405b-instruct"`).
- **Hardcoded result directories**: `rh_single_agent_results/`, `rh_multi_agent_results/`, `optimization_results/`, `benchmark_results/`.

**LCO defense flow (`method=LCO`):**
1. Generate multiple candidate responses.
2. Score toxicity via Google Perspective API.
3. Run LLM-based `crossover` and `mutate` prompts (genetic operations).
4. Use a safety-aware LLM judge (`vote()`) to select the final output.

### 5.2 Policy-Refinement
Built on top of **ToolEmu**, which uses an LLM-based simulator to generate tool observations instead of calling real APIs.

**Key architectural patterns:**
- **Registry pattern**: `SIMULATORS`, `EVALUATORS`, and `AGENT_TYPES` are centralized string→class dictionaries in `toolemu/agents/__init__.py` and related modules.
- **LangChain subclassing**: Most logic lives in overridden `_call`, `_take_next_step`, and `_return` methods of `AgentExecutorWithToolkit` and `VirtualAgentExecutorWithToolkit`.
- **Procoder prompts**: Prompts are constructed as module-level `PromptModule` (Procoder) constants and bound to executors via `_sys_prompt_module` and `_task_prompt_module`.
- **Stateless executors**: `BasePromptExecutor` subclasses are designed to be thread-safe; state is passed through inputs.
- **Ad-hoc parsing**: Extensive regex fallback chains parse LLM outputs (especially in `evaluators.py` and `virtual_agent_executor.py`).

**Defense agent hierarchy:**
- `SelfReflectionAgentExecutorWithToolkit` — prepends dynamically generated safety constraints.
- `SelfDefenseAgentExecutorWithToolkit` — real-time safety check before each action (max 3 retries).
- `GoalPriorityAgentExecutorWithToolkit` — prepends safety-first instructions.
- `GeneticAgentExecutorWithToolkit` / `LCOAgentExecutorWithToolkit` — population-based action selection with LLM fitness evaluation, crossover, and mutation.

**Typical pipeline:**
1. **Emulate**: `python scripts/emulate.py -inp assets/all_cases.json -atp naive -stp adv_thought ...`
2. **Evaluate**: `python scripts/evaluate.py -inp dumps/...jsonl -ev agent_constraint`
3. **Analyze**: Scripts in `evaluation_scripts/` (e.g., `compute_icrh_ratio.py`, `pairwise_eval.py`).

### 5.3 PromptCode
A PyTorch-`nn.Module`-like prompt composition library.
- `Module` is the base class (uses `_modules`, `add_module()`, `children()`, `forward()`).
- Prompts are rendered with `format_prompt(prompt, inputs={...})`.
- Cross-referencing is managed via `refname`; `collect_refnames` expands them to bracketed names (e.g., `[Input Requirement]`).

## 6. Running Experiments

### 6.1 Output-Refinement (Single Experiment)
```bash
cd output-refinement
python filtering.py \
    --experiment reward_hacking \
    --n_rounds 11 \
    --agent_model gpt-4 \
    --judge_model gpt-3.5-turbo \
    --n_judges 3 \
    --seed 0 \
    --agent_idx -1 \
    --method LCO
```

**Methods**: `base`, `LCO`, `self_defense`, `goal_priority`
**Models**: `gpt-3.5-turbo`, `gpt-4`, `qwen2.5-72b-instruct`, `llama-3.1-405b-instruct`

### 6.2 Output-Refinement (Batch + Analysis)
```bash
python run_experiments.py --agent_model gpt-4 --method LCO --seeds 20 --max_workers 4
python compute_ICRH.py --dir rh_multi_agent_results/ --method LCO
```

### 6.3 Policy-Refinement (Emulate + Evaluate)
```bash
cd policy-refinement
# Vanilla baseline
python scripts/emulate.py -inp assets/all_cases.json -atp naive -stp adv_thought --agent-model gpt-4-0613 --trunc-num 10 -v -me 3

# LCO with population size = 5
python scripts/emulate.py -inp assets/all_cases.json -atp naive -stp lco_thought --agent-model gpt-4-0613 --trunc-num 10 -v -me 3 --simulator-model gpt-4o -pos 5

# Evaluate
python scripts/evaluate.py -inp dumps/trajectories.jsonl --evaluator agent_constraint --split_by_errors
```

### 6.4 Capability-Retention Benchmarks
```bash
cd output-refinement
./run_benign_eval.sh gpt-3.5-turbo 50      # Benign tweet evaluation
./run_mmlu_gsm8k.sh gpt-3.5-turbo 50       # MMLU + GSM8K
```

## 7. Code Style Guidelines

- **Bilingual codebase**: Variable names, class names, and docstrings are in English. **Comments and print/log strings are often in Chinese** (especially in `output-refinement/`). Maintain this convention when adding comments or user-facing strings in those modules.
- **No formatter enforcement**: `black` is listed as a dependency but there is no pre-commit hook. Running `black` on changed files is good practice.
- **Import style**: Standard library first, then third-party, then local modules. `sys.path.insert` is used in some ad-hoc test scripts.
- **Type hints**: Minimal. Do not introduce strict typing unless you are modifying `PromptCode/`.
- **Naming**: `snake_case` for functions/variables, `CamelCase` for classes. Constants in `UPPER_CASE` in utility modules.

## 8. Testing Instructions

There is **no formal test framework** (no `pytest`, `tox`, or CI pipeline). Testing is ad-hoc:

| File | Purpose |
|------|---------|
| `output-refinement/test_process_response.py` | Unit tests for `reward_hacking/utils.process_response()` |
| `output-refinement/test_vllm.py` | Smoke test for the vLLM backend |
| `output-refinement/test_benchmark_data.py` | Validates benchmark data loading |
| `test.py` (root) | CLI tool for visualizing toxicity scores from result JSONs |
| `for test.py` (root) | Development scratchpad |
| `policy-refinement/eval_fitness_model/test.py` | Fitness-model evaluation utilities |

**If you modify parsing logic**, run `python output-refinement/test_process_response.py`.
**If you modify the vLLM backend**, run `python output-refinement/test_vllm.py`.

## 9. Security Considerations

1. **API Keys**: Real credentials have been hardcoded in multiple historical commits and files (`toolemu/utils/llm.py`, `api_keys.py`, `self_reflection_agent.py`, `self_defense_agent.py`, `MyAgent.py`). If you are working in a fork, rotate any exposed keys immediately.
2. **No `.env` validation**: The code loads secrets via `python-dotenv` but does not warn when keys are missing until runtime.
3. **Local model paths**: The `.gitignore` ignores `/llama3` and `/Qwen` directories at the root, suggesting local model weights may be stored there.
4. **No sandboxing**: `policy-refinement/` uses **virtual** tools (LLM-simulated), but some tool definitions look like real APIs. Do not accidentally wire them to live endpoints.

## 10. Common Pitfalls for Agents

1. **Do not run `filtering.py` with vLLM inside a `ThreadPoolExecutor`**. vLLM uses `multiprocessing.spawn` internally; threading around it causes deadlocks. The codebase already has serial fallbacks — do not remove them.
2. **LangChain version is pinned to `0.0.277`** in `policy-refinement/`. Upgrading LangChain will break prompt templates and agent executor internals.
3. **OpenAI SDK version is pinned to `0.28.1`** in `policy-refinement/`. The `output-refinement/` code uses the newer `openai>=1.0` client. Do not unify the versions without extensive refactoring.
4. **Result directories are hardcoded**. If you add a new experiment variant, you may need to add a new output directory string in multiple files (`filtering.py`, `run_experiments.py`, `plot_results.py`, etc.).
5. **Procoder is a submodule dependency**. `policy-refinement/` will fail at runtime if `PromptCode` is not installed with `pip install -e ./PromptCode` first.
