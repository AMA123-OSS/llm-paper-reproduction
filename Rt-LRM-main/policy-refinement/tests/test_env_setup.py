"""
Environment validation script for policy-refinement.
Run with: python test_env_setup.py
"""
import sys

def test_basic_info():
    print("=" * 60)
    print("Policy-Refinement Conda Environment Test")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print()

def test_procoder():
    import procoder
    from procoder.functional import format_prompt
    from procoder.prompt import NamedBlock, Sequential
    print("✓ PromptCode (procoder) imported successfully")

def test_dependencies():
    import langchain
    import openai
    import anthropic
    import torch
    import transformers
    import numpy
    import scipy
    import matplotlib
    import tiktoken
    import fire
    import dotenv
    print(f"✓ langchain=={langchain.__version__}")
    print(f"✓ openai=={openai.__version__}")
    print(f"✓ anthropic=={anthropic.__version__}")
    print(f"✓ torch=={torch.__version__}")
    print(f"✓ transformers=={transformers.__version__}")
    print(f"✓ numpy=={numpy.__version__}")
    print(f"✓ scipy=={scipy.__version__}")
    print(f"✓ matplotlib=={matplotlib.__version__}")
    print(f"✓ tiktoken=={tiktoken.__version__}")
    print(f"✓ fire installed")
    print(f"✓ python-dotenv installed")

def test_toolemu_agents():
    from toolemu.agents import SIMULATORS, AGENT_TYPES, AgentExecutorWithToolkit
    from toolemu.agents.virtual_agent_executor import (
        StandardVirtualAgentExecutorWithToolkit,
        AdversarialVirtualAgentExecutorWithToolkit,
    )
    from toolemu.agents import (
        LCOAgentExecutorWithToolkit,
        GeneticAgentExecutorWithToolkit,
        SelfReflectionAgentExecutorWithToolkit,
        SelfDefenseAgentExecutorWithToolkit,
        GoalPriorityAgentExecutorWithToolkit,
        SafetyPromptAgentExecutorWithToolkit,
    )
    print(f"✓ toolemu.agents imported successfully")
    print(f"  Simulators: {list(SIMULATORS.keys())}")
    print(f"  Agent types: {AGENT_TYPES}")

def test_toolemu_evaluators():
    from toolemu.evaluators import (
        EVALUATORS,
        AgentConstraintEvaluator,
        AgentHelpfulnessEvaluator,
    )
    print(f"✓ toolemu.evaluators imported successfully")
    print(f"  Evaluators: {list(EVALUATORS.keys())}")

def test_toolemu_builder():
    from toolemu.agent_executor_builder import build_agent_executor
    print("✓ toolemu.agent_executor_builder imported successfully")

def test_toolemu_utils():
    from toolemu.utils import load_openai_llm_with_args, llm_register_args
    from toolemu.executors import FuncExecutor
    from toolemu.dataloader import DataLoader
    print("✓ toolemu.utils / executors / dataloader imported successfully")

def main():
    test_basic_info()
    tests = [
        test_procoder,
        test_dependencies,
        test_toolemu_agents,
        test_toolemu_evaluators,
        test_toolemu_builder,
        test_toolemu_utils,
    ]
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            return 1
    print()
    print("=" * 60)
    print("All environment tests PASSED!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
