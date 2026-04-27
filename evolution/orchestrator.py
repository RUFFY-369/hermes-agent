import traceback
import asyncio
import torch
import os
import sys
from typing import List, Dict, Any, Optional

# Evolution System Imports (Internal to hermes-agent)
from evolution.client import SGLangClient, RolloutResponse
from evolution.sandbox import DockerSandbox
from evolution.grpo_trainer import GRPOTrainer
from evolution.opd_trainer import OPDTrainer
from evolution.judge import PRMJudge
from evolution.tinker import TinkerBridgeTrainer

# Ensure ml-intern is in the path
ML_INTERN_PATH = os.path.expanduser("~/NousResearch/ml-intern")
if os.path.exists(ML_INTERN_PATH):
    sys.path.append(ML_INTERN_PATH)

try:
    from agent.core.session import Session
    from agent.core.tools import ToolRouter
    from agent.config import Config
except ImportError as e:
    Session = None

class GASPOrchestrator:
    """
    Guided Asymmetric Self-Play (GASP) Orchestrator for Hermes-Agent.
    """
    def __init__(
        self,
        sgl_client: SGLangClient,
        sandbox: DockerSandbox,
        grpo_trainer: GRPOTrainer,
        opd_trainer: OPDTrainer,
        prm_judge: PRMJudge,
        tinker_bridge: Optional[TinkerBridgeTrainer] = None,
        group_size: int = 64
    ):
        self.client = sgl_client
        self.sandbox = sandbox
        self.grpo = grpo_trainer
        self.opd = opd_trainer
        self.prm = prm_judge
        self.tinker = tinker_bridge
        self.group_size = group_size
        
        if Session:
            self.event_queue = asyncio.Queue()
            self.teacher_config = Config(model_name="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0")
            self.teacher_router = ToolRouter(mcp_servers={}, local_mode=False)
            self.teacher_session = Session(
                event_queue=self.event_queue,
                config=self.teacher_config,
                tool_router=self.teacher_router
            )
        else:
            self.teacher_session = None

    async def _get_teacher_task(self, last_task: str = None) -> str:
        base_task = (
            "Write a Python function called solve() that returns the number of valid N-Queens solutions for N=8. "
            "STRICT CONSTRAINT: You must not use the 'if' keyword anywhere in your code. "
            "Rely entirely on boolean short-circuiting or array masking."
        )
        return base_task

    async def run_iteration(self, lora_path: str = None):
        task = await self._get_teacher_task()
        student_prompt = (
            f"### TASK\n{task}\n\n"
            "### INSTRUCTION\n"
            "Implement the solution in a function called `solve()`. "
            "Provide ONLY the code block.\n\n"
            "### SOLUTION\n```python\ndef solve():\n"
        )
        prompts = [student_prompt] * self.group_size
        student_rollouts = await self.client.generate_group(prompts, lora_path=lora_path)
        
        grading_tasks = []
        for r in student_rollouts:
            if not r.text.strip():
                async def failed_reward(): return -1.0
                grading_tasks.append(failed_reward())
            else:
                grading_tasks.append(self.sandbox.execute_code(r.text))
        
        rewards_list = await asyncio.gather(*grading_tasks)
        rewards = [float(r) for r in rewards_list]
        
        return rewards, student_rollouts, prompts, task
