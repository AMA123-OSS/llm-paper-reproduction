"""
Adaptive emulation script with risk-aware routing.

Routes each case dynamically based on Self-Reflection risk level:
  - Level 1 (Low)  -> Vanilla (adv_thought, pos=1)
  - Level 2 (Med)  -> LCO light (lco_thought, pos=1)
  - Level 3 (High) -> LCO full (lco_thought, pos=3)

Usage: python -m scripts.emulate_adaptive -inp <input_cases_file> -atp <agent_type>
"""

import argparse
import os
import random
import time

import openai
import openai.error
from dotenv import load_dotenv
from toolemu.agent_executor_builder import build_agent_executor
from toolemu.agents import AGENT_TYPES, SIMULATORS
from toolemu.dataloader import DataLoader
from toolemu.executors import FuncExecutor
from toolemu.utils import (
    case_to_input_dict,
    filter_keys,
    get_toolkit_names,
    llm_register_args,
    load_openai_llm_with_args,
    replace_agent_action_with_list,
)
from toolemu.agents.self_reflection_agent import classify_risk_level
import jsonl2json

load_dotenv()
ROLES = ["agent", "simulator"]

parser = argparse.ArgumentParser()
for role in ROLES:
    llm_register_args(parser, prefix=role)
DataLoader.register_args(parser)
FuncExecutor.register_args(
    parser, default_num_retries=5, default_batch_size=1, default_timeout=1800
)
parser.add_argument(
    "--output-file-prefix",
    "-outpre",
    type=str,
    default="./dumps/traj_adaptive",
)
parser.add_argument(
    "--output-file-suffix",
    "-outsuf",
    type=str,
    default="",
)
parser.add_argument(
    "--agent-type",
    "-atp",
    type=str,
    default="naive",
    choices=AGENT_TYPES,
)
parser.add_argument("--max-iterations", "-mi", type=int, default=10)
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--random-seed", "-seed", type=int, default=42)
parser.add_argument("--num_critique_steps", "-ncs", type=int, default=0)
parser.add_argument("--p_error", "-pe", type=float, default=0.0)
parser.add_argument("--max_errors", "-me", type=int, default=0)
parser.add_argument("--allow_all_tools", "-aat", action="store_true")
parser.add_argument("--cases", type=str, default="")

args = parser.parse_args()
random.seed(args.random_seed)


def main():
    if args.agent_model_name.startswith("gpt-3.5-turbo-instruct"):
        args.agent_max_tokens = 1024
    llms = {role: load_openai_llm_with_args(args, prefix=role) for role in ROLES}
    if args.cases == "":
        cases = DataLoader.from_args(args, return_mode="with_idx", item_name="case")
    else:
        idx = [int(i) for i in args.cases.split(",")]
        cases = DataLoader.from_args(
            args, return_mode="with_idx", item_name="case", selected_indexes=idx
        )
    runner = FuncExecutor.from_args(args)
    output_path = (
        f"{args.output_file_prefix}_{args.agent_model_name}_{args.agent_type}{args.output_file_suffix}.jsonl"
    )

    def generate_trajectory(case_with_idx):
        case_idx, case = case_with_idx["idx"], case_with_idx["item"]
        t_overall_start = time.perf_counter()

        # ---------- Step 1: Risk Classification via Self-Reflection ----------
        t0 = time.perf_counter()
        task_input = case_to_input_dict(case)["input"]
        reflection_result = classify_risk_level(args.agent_model_name, task_input)
        risk_level = reflection_result["risk_level"]
        reasoning = reflection_result["reasoning"]
        safety_constraints = reflection_result["safety_constraints"]
        t1 = time.perf_counter()
        print(f"[TIMING][Case {case_idx}] Risk Classification: {t1-t0:.2f}s")
        print(f"[Case {case_idx}] Risk Level: {risk_level}, Reasoning: {reasoning}")

        # ---------- Step 2: Adaptive Routing ----------
        if risk_level == 1:
            sim_type = "adv_thought"
            pop_size = 1
            sel_size = 1
        elif risk_level == 2:
            sim_type = "lco_thought"
            pop_size = 1
            sel_size = 1
        else:
            sim_type = "lco_thought"
            pop_size = 3
            sel_size = 1

        # ---------- Step 3: Build and run the real executor ----------
        t2 = time.perf_counter()
        agent_executer = build_agent_executor(
            get_toolkit_names(case),
            llms["agent"],
            llms["simulator"],
            agent_type=args.agent_type,
            simulator_type=sim_type,
            verbose=args.verbose,
            max_iterations=args.max_iterations,
            num_critique_steps=args.num_critique_steps,
            p_error=args.p_error,
            max_errors=args.max_errors,
            allow_all_tools=args.allow_all_tools,
            case_idx=case_idx,
            population_size=pop_size,
            selection_size=sel_size,
            agent_model_name=args.agent_model_name,
        )
        # 注入预计算的安全约束，避免 LCO/Seco 内部重复调用 _self_reflection() 生成
        if risk_level > 1 and safety_constraints:
            # 使用 object.__setattr__ 绕过 pydantic 字段校验
            object.__setattr__(agent_executer, '_precomputed_safety_constraints', safety_constraints)
        t3 = time.perf_counter()
        print(f"[TIMING][Case {case_idx}] Build Executor: {t3-t2:.4f}s")

        print(f"[Case {case_idx}] Using simulator={sim_type}, population_size={pop_size}")
        inputs = filter_keys(case_to_input_dict(case), agent_executer.input_keys)
        try:
            t4 = time.perf_counter()
            outputs = agent_executer(inputs)
            t5 = time.perf_counter()
            print(f"[TIMING][Case {case_idx}] Agent Execution: {t5-t4:.2f}s")
            failed_item = None
        except (openai.error.InvalidRequestError) as e:
            print(f"{case_idx}: {str(e)}")
            outputs = {"error": str(e)}
            failed_item = case_with_idx

        t_overall_end = time.perf_counter()
        print(f"[TIMING][Case {case_idx}] Overall: {t_overall_end-t_overall_start:.2f}s")

        print(type(outputs))
        outputs = replace_agent_action_with_list(outputs)  # ad-hoc fix
        outputs["case"] = case
        outputs["case_idx"] = case_idx
        outputs["risk_level"] = risk_level
        outputs["risk_reasoning"] = reasoning
        outputs["safety_constraints"] = safety_constraints
        outputs["adaptive_simulator_type"] = sim_type
        outputs["adaptive_population_size"] = pop_size
        if args.agent_model_name.startswith("claude"):
            outputs["agent"] = f"{llms['agent'].model}_{args.agent_type}"
        else:
            outputs["agent"] = f"{llms['agent'].model_name}_{args.agent_type}"
        return failed_item, outputs

    # trajectory generation and recording pipeline
    runner.run(generate_trajectory, cases, output_path, args=args)
    print(
        "You may want to use scripts to convert the result jsonl file "
        f"{output_path} to json for easier reading."
    )


if __name__ == "__main__":
    main()
