import json
import random
import re
from argparse import ArgumentParser, Namespace

from langchain.schema import ChatResult, LLMResult
from procoder.functional import removed_submodules
from procoder.prompt import Module as PromptModule
from toolemu.prompts.evaluator import (
    AGENT_HELP_EVALUATOR_PROMPT,
    AGENT_HELP_EVALUATOR_SYSTEM_INFO,
    AGENT_SAFE_EVALUATOR_PROMPT,
    AGENT_SAFE_EVALUATOR_SYSTEM_INFO,
    AGENT_CONSTRAINT_EVALUATOR_PROMPT,
    AGENT_CONSTRAINT_EVALUATOR_SYSTEM_INFO,
    AGENT_NOFAIL_HELP_EVALUATOR_PROMPT,
    AGENT_NOFAIL_HELP_EVALUATOR_SYSTEM_INFO,
    AGENT_PERPLEXITY_EVALUATOR_PROMPT,
    AGENT_PERPLEXITY_EVALUATOR_SYSTEM_INFO,
)
from toolemu.prompts.globals import GENERAL_CRITIQUE_PROMPT
from toolemu.tools import get_toolkits_by_names
from toolemu.utils import (
    METRICS_SHORT2LONG,
    case_to_input_dict,
    construct_trajec,
    get_toolkit_names,
    get_used_tools_in_trajec,
    parse_llm_response,
)
from toolemu.utils.my_typing import *
from toolemu.utils.my_typing import Any, Dict, List

from .executors import BasePromptExecutorWithCritique

def extract_section_fallback(text, start_marker="Evaluator Thought", end_marker="Overall Qualitative Label"):
    # 如果正则失败，使用逐行查找
    lines = text.split("\n")
    extracting = False
    extracted_text = []

    for line in lines:
        if start_marker.lower() in line.lower():  # 发现起始标识
            extracting = True
            continue  # 跳过起始行本身

        if extracting:
            if end_marker.lower() in line.lower():  # 发现终止标识
                break
            extracted_text.append(line)
    return "\n".join(extracted_text).strip()

class BaseTrajEvaluator(BasePromptExecutorWithCritique):
    _short_name: str = None
    _input_keys: List[str] = ["trajectory"]
    _output_keys: List[str] = [
        "Evaluator Thought",
        "Overall Qualitative Label",
        "Overall Quantitative Score",
        "Evaluator Log Summary",
    ]
    _keys2metrics: Dict[str, str] = None
    _critique_prompt_module: PromptModule = GENERAL_CRITIQUE_PROMPT
    _final_answer_in_traj: bool = True
    _toolkit_desc_detail_level: str = "low"  # use simple toolkit descriptions
    _used_tool_desc_detail_level: str = None

    def _preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        traj = inputs["trajectory"]
        if "error" in traj:
            raise ValueError(f"Error in eval trajec: {traj['error']}")

        toolkits = get_toolkits_by_names(get_toolkit_names(traj["case"]))
        toolkit_descs = {
            detail_level: "\n".join(
                [f"{toolkit.create_description(detail_level)}" for toolkit in toolkits]
            )
            for detail_level in ["low", "medium", "high"]
        }
        inputs = dict(
            tool_names=", ".join([tool.name for tool in toolkits]),
            toolkit_descriptions=toolkit_descs[self._toolkit_desc_detail_level],
        )

        used_tool_level = self._used_tool_desc_detail_level
        if used_tool_level is not None:
            used_tool_names = get_used_tools_in_trajec(traj)
            used_tools = []
            for toolkit in toolkits:
                for tool in toolkit.tools:
                    if tool.name in used_tool_names:
                        used_tools.append(tool)
            used_tools_desc = "\n".join(
                [
                    f"* {tool.name}: {tool.create_description(used_tool_level)}"
                    for tool in used_tools
                ]
            )
            inputs["used_tools_descriptions"] = used_tools_desc

        return inputs

    def _fill_case_info(
        self,
        prompt_inputs: Dict[str, Any],
        case: Dict[str, Any],
        case_type: str = "legacy",
    ):
        inputs = case_to_input_dict(case, case_type)
        prompt_inputs.update(inputs)

    def _preprocess_prompts(
        self, prompts: List[PromptModule], prompt_inputs: Dict[str, Any]
    ) -> List[PromptModule]:
        return prompts

    def _postprocess_prompts(
        self, prompts: List[str], raw_inputs: Dict[str, Any]
    ) -> List[str]:
        traj = raw_inputs["trajectory"]
        traj_text = construct_trajec(
            traj,
            include_input=False,
            is_dumped_dict=True,
            include_agent_log=False,
            include_simulator_log=False,
            include_final_answer=self._final_answer_in_traj,
        )

        processed_prompts = []
        for prompt, name in zip(prompts, self._prompt_module_names):
            if name == "task":
                inputs = dict(evaluator_scratchpad=traj_text)
                self._fill_case_info(
                    inputs, traj["case"], traj.get("case_type", "legacy")
                )
                prompt = prompt.format(**inputs)
            processed_prompts.append(prompt)
        return processed_prompts

    def _parse_output(self, output: ChatResult, raw_inputs: Dict[str, Any], **kwargs) -> Any:
        res = parse_llm_response(output)
        eval_res = {k: None for k in self._output_keys}
        eval_res["eval_scores"] = {}

        regex_patterns = [
            # 第一种匹配格式
            re.compile(
                r"(?<=Evaluator Thought:\n\n)(.*?)(?=\n\nOverall Qualitative Label:)|"
                r"(?<=Overall Qualitative Label:\s)(.*?)(?=\n\nOverall Quantitative Score:)|"
                r"(?<=Overall Quantitative Score:\s)(.*?)(?=\n\nEvaluator Log Summary:)|"
                r"(?<=Evaluator Log Summary:\s)(.*)", re.DOTALL),
            # 第二种匹配格式（例如 LLM 输出中无\n\n）
            re.compile(
                r"Evaluator Thought:\s*(.*?)\s*Overall Qualitative Label:\s*(.*?)\s*Overall Quantitative Score:\s*(.*?)\s*Evaluator Log Summary:\s*(.*)",
                re.DOTALL),
            # 第三种模式（带自定义 key）
            re.compile(
                "|".join([
                    (f"### {k}\\s*(.*?)\\s*(?=###|$)" if self._short_name == "agent_constraint"
                     else f"{k}:\\s*(.*?)\\s*(?={self._output_keys[i + 1]}:|$)" if i + 1 < len(self._output_keys)
                    else f"{k}:\\s*(.*)")
                    for i, k in enumerate(self._output_keys)
                ]),
                re.DOTALL)
        ]

        matched = False
        for pattern in regex_patterns:
            try:
                matches = pattern.findall(res)
                # 展平提取出的元组并去空
                results = [match for group in matches for match in (group if isinstance(group, tuple) else [group]) if
                           match]
                if results and len(results) >= len(self._output_keys):
                    for i, k in enumerate(self._output_keys):
                        eval_res[k] = results[i]
                        if self._keys2metrics and k in self._keys2metrics:
                            eval_res["eval_scores"][self._keys2metrics[k]] = eval_res[k]
                    matched = True
                    break
            except Exception as e:
                print(f"[Regex Match Failed] Pattern: {pattern}, Error: {e}")
                continue

        if not matched:
            print("[Fallback to Line Matching]")
            for i, k in enumerate(self._output_keys):
                try:
                    start = k
                    end = self._output_keys[i + 1] if i + 1 < len(self._output_keys) else None
                    extracted = extract_section_fallback(res, start, end) if end else extract_section_fallback(res,
                                                                                                               start,
                                                                                                               "")
                    eval_res[k] = extracted
                    if self._keys2metrics and k in self._keys2metrics:
                        eval_res["eval_scores"][self._keys2metrics[k]] = extracted
                except Exception as e:
                    print(f"[Line Match Failed for {k}] Error: {e}")
        return eval_res
    # def _parse_output(
    #     self, output: ChatResult, raw_inputs: Dict[str, Any], **kwargs
    # ) -> Any:
    #     res = parse_llm_response(output)
    #     print(f"EVAL_RES:{res}")
    #     eval_res = {k: None for k in self._output_keys}
    #     eval_res["eval_scores"] = {}
    #     pattern = re.compile(
    #         r"(?<=Evaluator Thought:\n\n)(.*?)(?=\n\nOverall Qualitative Label:)|"
    #         r"(?<=Overall Qualitative Label:\s)(.*?)(?=\n\nOverall Quantitative Score:)|"
    #         r"(?<=Overall Quantitative Score:\s)(.*?)(?=\n\nEvaluator Log Summary:)|"
    #         r"(?<=Evaluator Log Summary:\s)(.*)"
    #         , re.DOTALL)
    #
    #     # 执行匹配
    #     matches = pattern.findall(res)
    #     # 由于 `findall` 返回的是多个捕获组的元组，我们需要筛选非空匹配项
    #     results = [match for group in matches for match in group if match]
    #     if results is not None:
    #         for i, k in enumerate(self._output_keys):
    #             eval_res[k] = results[i]
    #             if self._keys2metrics is not None and k in self._keys2metrics:
    #                 eval_res["eval_scores"][self._keys2metrics[k]] = eval_res[k]
    #     # 一种模式匹配不到就换一种模式匹配，多匹配几次
    #     pattern = ""
    #     for k in self._output_keys:
    #         if self._short_name == "agent_constraint":
    #             pattern += "### " + k + r"\s*(.*?)\s*"
    #         else:
    #             pattern += k + r":\s*(.*?)\s*"
    #     pattern += "$"
    #     matches = re.search(pattern, res, re.DOTALL)
    #     if matches is not None:
    #         for i, k in enumerate(self._output_keys):
    #             eval_res[k] = matches.group(i + 1)
    #             if self._keys2metrics is not None and k in self._keys2metrics:
    #                 eval_res["eval_scores"][self._keys2metrics[k]] = eval_res[k]
    #
    #     return eval_res


class AgentConstraintEvaluator(BaseTrajEvaluator):
    _short_name = "agent_constraint"
    _sys_prompt_module = AGENT_CONSTRAINT_EVALUATOR_SYSTEM_INFO
    _task_prompt_module = AGENT_CONSTRAINT_EVALUATOR_PROMPT
    _keys2metrics = {"Overall Quantitative Score": METRICS_SHORT2LONG[_short_name]}

    def _postprocess_prompts(
        self, prompts: List[str], raw_inputs: Dict[str, Any]
    ) -> List[str]:
        traj = raw_inputs["trajectory"]
        traj_text = construct_trajec(
            traj,
            include_input=False,
            is_dumped_dict=True,
            include_agent_log=False,
            include_simulator_log=False,
            include_final_answer=False,
        )

        processed_prompts = []
        for prompt, name in zip(prompts, self._prompt_module_names):
            if name == "task":
                inputs = dict(evaluator_scratchpad=traj_text)
                self._fill_case_info(
                    inputs, traj["case"], traj.get("case_type", "legacy")
                )
                prompt = prompt.format(**inputs)
            processed_prompts.append(prompt)
        # output_requirements='''
        # ######
        # 【Output Requirements】
        # {
        #     Evaluator Thought:"1.Brainstorm…",
        #     Overall Qualitative Label:"The appropriate qualitative label…",
        #     Overall Quantitative Score:"0",
        #     Evaluator Log Summary:"The Agent attempted to……"
        # }
        # '''
        # processed_prompts.append(output_requirements)
        return processed_prompts

class AgentRiskyToolCallEvaluator(BaseTrajEvaluator):
    _short_name = "agent_safe"
    _sys_prompt_module = AGENT_SAFE_EVALUATOR_SYSTEM_INFO
    _task_prompt_module = AGENT_SAFE_EVALUATOR_PROMPT
    _keys2metrics = {"Overall Quantitative Score": METRICS_SHORT2LONG[_short_name]}


class AgentHelpfulnessEvaluator(BaseTrajEvaluator):
    _short_name = "agent_help"
    _sys_prompt_module = AGENT_HELP_EVALUATOR_SYSTEM_INFO
    _task_prompt_module = AGENT_HELP_EVALUATOR_PROMPT
    _keys2metrics = {"Overall Quantitative Score": METRICS_SHORT2LONG[_short_name]}


class AgentNoFailHelpfulnessEvaluator(BaseTrajEvaluator):
    _short_name = "agent_nofail_help"
    _sys_prompt_module = AGENT_NOFAIL_HELP_EVALUATOR_SYSTEM_INFO
    _task_prompt_module = AGENT_NOFAIL_HELP_EVALUATOR_PROMPT
    _keys2metrics = {"Overall Quantitative Score": METRICS_SHORT2LONG[_short_name]}


class AgentPerplexityEvaluator(BaseTrajEvaluator):
    _short_name = "agent_perplexity"
    _sys_prompt_module = AGENT_PERPLEXITY_EVALUATOR_SYSTEM_INFO
    _task_prompt_module = AGENT_PERPLEXITY_EVALUATOR_PROMPT
    _keys2metrics = {"Overall Quantitative Score": METRICS_SHORT2LONG[_short_name]}

    def _postprocess_prompts(
        self, prompts: List[str], raw_inputs: Dict[str, Any]
    ) -> List[str]:
        traj = raw_inputs["trajectory"]
        traj_text = construct_trajec(
            traj,
            include_input=False,
            is_dumped_dict=True,
            include_agent_log=False,
            include_simulator_log=False,
            include_final_answer=False,
            include_obs=False,
        )

        processed_prompts = []
        for prompt, name in zip(prompts, self._prompt_module_names):
            if name == "task":
                inputs = dict(evaluator_scratchpad=traj_text)
                self._fill_case_info(
                    inputs, traj["case"], traj.get("case_type", "legacy")
                )
                prompt = prompt.format(**inputs)
            processed_prompts.append(prompt)
        return processed_prompts


EVALUATORS = [
    AgentRiskyToolCallEvaluator,
    AgentHelpfulnessEvaluator,
    AgentNoFailHelpfulnessEvaluator,
    AgentConstraintEvaluator,
    AgentPerplexityEvaluator,
]
EVALUATORS = {e._short_name: e for e in EVALUATORS}
EVALUATORS2METRICS = {k: v._keys2metrics.values() for k, v in EVALUATORS.items()}
