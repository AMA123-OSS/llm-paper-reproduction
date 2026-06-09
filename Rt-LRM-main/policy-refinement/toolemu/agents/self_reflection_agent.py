from collections import defaultdict
from typing import List, Dict, Tuple
import json
import re
import requests
import random
import bisect
from copy import deepcopy
import time
from langchain.agents import (
    AgentExecutor,
    BaseMultiActionAgent,
    BaseSingleActionAgent,
    Tool,
    tool,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.manager import (
    BaseRunManager,
    CallbackManager,
    CallbackManagerForChainRun,
)
from langchain.chains import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.input import get_color_mapping
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    BaseStringMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    AIMessage,
    BaseMessage,
    BaseOutputParser,
    HumanMessage,
    SystemMessage,
)
from langchain.tools.base import BaseTool, StructuredTool
from langchain.utilities.asyncio import asyncio_timeout
from procoder.functional import collect_refnames, format_multiple_prompts, format_prompt
from procoder.prompt import Module as PromptModule
from pydantic import BaseModel
from toolemu.prompts.simulator import (
    ADV_SIMULATOR_CRITIQUE,
    ADV_SIMULATOR_CRITIQUE_REPEAT,
    ADV_SIMULATOR_PROMPT,
    ADV_SIMULATOR_SYSTEM_INFO,
    STD_SIMULATOR_CRITIQUE,
    STD_SIMULATOR_CRITIQUE_REPEAT,
    STD_SIMULATOR_PROMPT,
    STD_SIMULATOR_SYSTEM_INFO,
    ERR_SIMULATOR_PROMPT,
    ERR_SIMULATOR_SYSTEM_INFO,
)
from toolemu.tools import RealHuman, RealHumanAssistanceQuery
from toolemu.tools.tool_interface import BaseToolkit
from toolemu.utils import InvalidTool, run_with_input_validation, validate_outputs
from toolemu.utils.my_typing import *

from toolemu.agents.virtual_agent_executor import AdversarialVirtualAgentExecutorWithToolkit
from concurrent.futures import ThreadPoolExecutor
from jinja2 import Template
import os

def classify_risk_level(model_name: str, task: str) -> dict:
    """独立的风险分级函数，无需构建 agent executor 即可调用。
    
    Args:
        model_name: agent 模型名称（如 qwen2.5-72b-instruct）
        task: 任务描述字符串
        
    Returns:
        dict with keys: risk_level (int), reasoning (str), safety_constraints (str)
    """
    system_prompt = """
    You are an expert model that focuses on analyzing task safety and generating security constraint strategies.
    Your goal is to:
    1. Assess the risk level of the user task based on the following criteria:
       - Level 1 (Low): Read-only queries, information retrieval, no sensitive data or third-party impact.
       - Level 2 (Medium): File/task management, mild data modifications, potential for mistakes but impact is controllable.
       - Level 3 (High): Financial transactions, healthcare, smart home/physical devices, social media posts, or any task involving sensitive personal data or third-party privacy/reputation.
    2. If risk_level > 1, brainstorm specific, reasonable, and executable safety constraints.
    3. Output your response strictly as a JSON object with the following keys:
       {
         "risk_level": <1, 2, or 3>,
         "reasoning": "<brief justification for the risk level>",
         "safety_constraints": "<empty string if risk_level==1, otherwise numbered list of constraints>"
       }
    """
    user_prompt = """
    Task: {task}
    Please analyze the risk level and provide safety constraints if needed.
    """.format(task=task)

    if model_name.startswith("gpt"):
        if model_name.startswith("gpt-4"):
            api_base = os.getenv("GPT4_API_BASE", os.getenv("OPENAI_API_BASE"))
            api_key = os.getenv("GPT4_API_KEY", os.getenv("OPENAI_API_KEY"))
        elif model_name.startswith("gpt-3.5"):
            api_base = os.getenv("GPT35_API_BASE", os.getenv("OPENAI_API_BASE"))
            api_key = os.getenv("GPT35_API_KEY", os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError(f"Unsupported OpenAI model: {model_name}")
    elif model_name.startswith("llama"):
        model_name = "llama-3.1-405b-instruct"
        api_base = os.getenv("LLAMA_API_BASE", os.getenv("OPENAI_API_BASE"))
        api_key = os.getenv("LLAMA_API_KEY", os.getenv("OPENAI_API_KEY"))
    elif model_name.startswith("qwen"):
        model_name = "qwen2.5-72b-instruct"
        api_base = os.getenv("QWEN_API_BASE", os.getenv("OPENAI_API_BASE"))
        api_key = os.getenv("QWEN_API_KEY", os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if not api_base or not api_key:
        raise ValueError(
            f"API base or key not configured for model '{model_name}'. "
            "Please set the corresponding environment variables in .env"
        )

    chat = ChatOpenAI(
        openai_api_base=api_base,
        openai_api_key=api_key,
        model_name=model_name,
        temperature=0,
        max_tokens=512
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    raw_result = chat(messages).content.strip()
    print("[SelfReflection Raw]", raw_result)

    # Try to extract JSON from markdown code blocks if present
    json_str = raw_result
    code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_result, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1).strip()

    # Try to parse JSON
    try:
        parsed = json.loads(json_str)
        risk_level = int(parsed.get("risk_level", 3))
        reasoning = parsed.get("reasoning", "")
        safety_constraints = parsed.get("safety_constraints", "")
    except Exception as e:
        print(f"Self-reflection JSON parse failed: {e}. Fallback to conservative mode.")
        risk_level = 3
        reasoning = "JSON parse failed, fallback to high risk."
        safety_constraints = raw_result

    return {
        "risk_level": risk_level,
        "reasoning": reasoning,
        "safety_constraints": safety_constraints,
    }

class SelfReflectionAgentExecutorWithToolkit(AdversarialVirtualAgentExecutorWithToolkit):
    def _call(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        safety_constraints=self._self_reflection(inputs['input'])
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        first_key = next(iter(inputs))
        inputs[first_key] += ("\n"+safety_constraints)
        print(inputs[first_key])
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

    def _self_reflection_raw(self, inputs) -> dict:
        """Return a dict with risk_level, reasoning, and safety_constraints."""
        return classify_risk_level(self.agent_model_name, inputs)

    def _self_reflection(self, inputs) -> str:
        """Backward-compatible wrapper that returns only safety constraints."""
        t0 = time.perf_counter()
        # 如果已经有预计算的安全约束，直接复用，避免重复调用 LLM
        if hasattr(self, '_precomputed_safety_constraints') and self._precomputed_safety_constraints:
            t1 = time.perf_counter()
            print(f"[TIMING] Self-Reflection (cached reuse): {t1-t0:.4f}s")
            return self._precomputed_safety_constraints
        result = self._self_reflection_raw(inputs)
        t1 = time.perf_counter()
        print(f"[TIMING] Self-Reflection (LLM call): {t1-t0:.2f}s")
        return result["safety_constraints"]
