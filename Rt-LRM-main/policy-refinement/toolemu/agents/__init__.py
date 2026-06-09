from toolemu.agents.safety_prompt_agent import SafetyPromptAgentExecutorWithToolkit
from toolemu.agents.self_defense_agent import SelfDefenseAgentExecutorWithToolkit
from toolemu.agents.self_reflection_agent import SelfReflectionAgentExecutorWithToolkit
from toolemu.agents.goal_priority_agent import GoalPriorityAgentExecutorWithToolkit
from .agent_executor import AgentExecutorWithToolkit
from .virtual_agent_executor import (
    AdversarialVirtualAgentExecutorWithToolkit,
    StandardVirtualAgentExecutorWithToolkit,
)
from toolemu.agents.MyAgent import GeneticAgentExecutorWithToolkit, SecoAgentExecutorWithToolkit, LCOAgentExecutorWithToolkit
from .zero_shot_agent_with_toolkit import AGENT_TYPES, ZeroShotAgentWithToolkit

SIMULATORS = {
    "normal": AgentExecutorWithToolkit,
    "std_thought": StandardVirtualAgentExecutorWithToolkit,
    "adv_thought": AdversarialVirtualAgentExecutorWithToolkit,
    "gen_thought":GeneticAgentExecutorWithToolkit,
    "self_defense_thought": SelfDefenseAgentExecutorWithToolkit,
    "safety_prompt_thought": SafetyPromptAgentExecutorWithToolkit,
    "self_reflection_thought":SelfReflectionAgentExecutorWithToolkit,
    "seco_thought":SecoAgentExecutorWithToolkit,
    "lco_thought":LCOAgentExecutorWithToolkit,
    "goal_priority_thought":GoalPriorityAgentExecutorWithToolkit
}
