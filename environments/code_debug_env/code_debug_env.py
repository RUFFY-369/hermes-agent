"""
CodeDebugEnv -- Multi-Turn Python Debugging with Real Tool Execution

A hermes-agent environment where the model uses terminal + file tools to
iteratively debug buggy Python functions from the HumanEvalPack dataset.

Unlike the single-turn Atropos `code_debug_env` (which scores raw model output),
this environment gives the agent a real sandbox with:
    - /workspace/buggy.py  (the buggy function to fix)
    - /workspace/tests.py  (the test suite to pass)

The agent can read files, run tests, edit code, and re-run tests - exactly
how a human developer would debug.

Reward is multi-signal:
    - test_signal  (0.5): Did the tests pass?
    - diagnosis    (0.3): Did the agent read the code and run tests before editing?
    - efficiency   (0.2): How many turns did it take?

Usage:
    # Process mode (local terminal backend)
    export ATROPOS_ALLOW_DUMMY_MANAGED_SERVER=1
    python environments/code_debug_env/code_debug_env.py process \\
        --config environments/code_debug_env/default.yaml

    # Serve mode (with Atropos API server)
    run-api
    python environments/code_debug_env/code_debug_env.py serve \\
        --config environments/code_debug_env/default.yaml
"""

import logging
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repo root is on sys.path for imports
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from datasets import load_dataset

from atroposlib.envs.base import ScoredDataGroup
from atroposlib.envs.server_handling.server_manager import APIServerConfig
from atroposlib.type_definitions import Item

from environments.agent_loop import AgentResult
from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
from environments.tool_context import ToolContext

logger = logging.getLogger(__name__)


# =============================================================================
# Test runner template - uploaded to /workspace/tests.py
# =============================================================================

TEST_RUNNER_TEMPLATE = '''"""Auto-generated test runner for {entry_point}."""
import sys

# Load the function under test from buggy.py
exec(open("/workspace/buggy.py").read())

{test_code}

# Run the check function
try:
    check({entry_point})
    print("ALL TESTS PASSED")
    sys.exit(0)
except AssertionError as e:
    print(f"TEST FAILED: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{e}}")
    sys.exit(1)
'''


class CodeDebugEnvConfig(HermesAgentEnvConfig):
    """Config with defaults for code debugging tasks."""

    pass  # Inherits all fields, overrides defaults in config_init


class CodeDebugEnv(HermesAgentBaseEnv):
    """
    Multi-turn code debugging environment with real tool execution.

    For each task:
    1. Uploads buggy.py and tests.py to the agent's sandbox
    2. The agent uses terminal/file tools to debug and fix the code
    3. Reward is computed by running tests in the sandbox via ToolContext

    This is the agentic version of the Atropos code_debug_env - instead of
    single-turn "output the fix", the model iteratively debugs like a human.
    """

    name = "code-debug-agent"
    env_config_cls = CodeDebugEnvConfig

    @classmethod
    def config_init(cls) -> Tuple[CodeDebugEnvConfig, List[APIServerConfig]]:
        """Default configuration for the code debug environment."""
        env_config = CodeDebugEnvConfig(
            # Terminal + file tools for debugging
            enabled_toolsets=["terminal", "file"],
            disabled_toolsets=None,
            distribution=None,
            # Agent settings
            max_agent_turns=15,
            max_token_length=4096,
            agent_temperature=1.0,
            system_prompt=(
                "You are an expert Python debugger with access to terminal and file tools. "
                "You will be given a workspace with a buggy Python file and a test file. "
                "Your goal is to identify and fix the bug so all tests pass.\n\n"
                "Debugging strategy:\n"
                "1. Read the buggy code to understand its intent\n"
                "2. Run the tests to see which ones fail and why\n"
                "3. Identify the root cause of the failure\n"
                "4. Edit the file to fix the bug\n"
                "5. Run the tests again to verify your fix\n\n"
                "Use the terminal to run commands and file tools to read/write files. "
                "Be precise with your edits - fix only the bug, don't rewrite everything."
            ),
            # Local terminal backend (override to modal/docker for production)
            terminal_backend="local",
            # Dataset
            dataset_name="bigcode/humanevalpack",
            dataset_split="test",
            # Atropos settings
            group_size=4,
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            tool_call_parser="hermes",
            steps_per_eval=50,
            total_steps=200,
            use_wandb=True,
            wandb_name="code-debug-agent",
            ensure_scores_are_not_same=False,
            extra_body={"atropos_inhibit_tools": True},
        )

        server_configs = [
            APIServerConfig(
                base_url="http://localhost:9001/v1",
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                server_type="openai",
                api_key="x",
                health_check=False,
                extra_body={"atropos_inhibit_tools": True},
            )
        ]

        return env_config, server_configs

    async def setup(self):
        """Load HumanEvalPack dataset and prepare train/test splits."""
        logger.info("Loading HumanEvalPack (python) dataset...")
        dataset = load_dataset(
            self.config.dataset_name or "bigcode/humanevalpack",
            "python",
            split=self.config.dataset_split,
        )

        all_items: List[Dict[str, Any]] = []
        for row in dataset:
            all_items.append(
                {
                    "task_id": row["task_id"],
                    "declaration": row["declaration"],
                    "buggy_solution": row["buggy_solution"],
                    "canonical_solution": row["canonical_solution"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                }
            )

        logger.info("Loaded %d problems", len(all_items))

        # Split 80/20 train/test
        random.shuffle(all_items)
        split_idx = int(len(all_items) * 0.8)
        self.train = all_items[:split_idx]
        self.test = all_items[split_idx:]
 
        logger.info("Train: %d, Test: %d", len(self.train), len(self.test))
        self.iter = 0

        # Reward tracking for wandb
        self.reward_buffer: List[float] = []
        self.test_pass_buffer: List[float] = []
        self.diagnosis_buffer: List[float] = []
        self.efficiency_buffer: List[float] = []

    async def get_next_item(self) -> Dict[str, Any]:
        """Cycle through training items."""
        item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return item

    def format_prompt(self, item: Dict[str, Any]) -> str:
        """
        Format the debugging task prompt.

        Tells the agent about the workspace files without revealing
        the exact bug or solution.
        """
        return (
            f"There is a buggy Python function in `/workspace/buggy.py` that needs to be fixed.\n"
            f"The function `{item['entry_point']}` has a bug that causes test failures.\n\n"
            f"A test file is available at `/workspace/tests.py` - run it to see which tests fail.\n\n"
            f"Your task:\n"
            f"1. Read `/workspace/buggy.py` to understand the function\n"
            f"2. Run `python /workspace/tests.py` to see the failures\n"
            f"3. Fix the bug in `/workspace/buggy.py`\n"
            f"4. Run the tests again to verify all tests pass\n\n"
            f"Fix only the bug. Do not rewrite the entire function."
        )

    def _build_buggy_file(self, item: Dict[str, Any]) -> str:
        """Build the contents of /workspace/buggy.py."""
        return item["declaration"] + item["buggy_solution"]

    def _build_test_file(self, item: Dict[str, Any]) -> str:
        """Build the contents of /workspace/tests.py with the test runner."""
        return TEST_RUNNER_TEMPLATE.format(
            entry_point=item["entry_point"],
            test_code=item["test"],
        )

    async def collect_trajectory(
        self, item: Item
    ) -> Tuple[Optional[Union[Any, Dict]], List[Item]]:
        """
        Override to scaffold the workspace before the agent loop runs.

        Uploads buggy.py and tests.py into the sandbox, then delegates
        to the parent's collect_trajectory for the agent loop + reward.
        """
        task_id = str(uuid.uuid4())

        # Get group-level tools
        if self._current_group_tools is None:
            tools, valid_names = self._resolve_tools_for_group()
        else:
            tools, valid_names = self._current_group_tools

        # --- Scaffold the workspace ---
        ctx = ToolContext(task_id)
        try:
            # Create workspace directory
            ctx.terminal("mkdir -p /workspace", timeout=10)

            # Upload buggy.py
            buggy_content = self._build_buggy_file(item)
            ctx.write_file("/workspace/buggy.py", buggy_content)

            # Upload tests.py
            test_content = self._build_test_file(item)
            ctx.write_file("/workspace/tests.py", test_content)
        except Exception as e:
            logger.error("Failed to scaffold workspace: %s", e)
            ctx.cleanup()
            return {"tokens": list(range(128)), "masks": [-100] + list(range(1, 128)),
                    "scores": 0.0, "messages": []}, []

        # --- Run the agent loop (reuse parent logic but with our task_id) ---
        from environments.agent_loop import HermesAgentLoop

        messages: List[Dict[str, Any]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": self.format_prompt(item)})

        # Run agent loop
        if self._use_managed_server():
            try:
                async with self.server.managed_server(
                    tokenizer=self.tokenizer,
                    preserve_think_blocks=bool(self.config.thinking_mode),
                ) as managed:
                    agent = HermesAgentLoop(
                        server=managed,
                        tool_schemas=tools,
                        valid_tool_names=valid_names,
                        max_turns=self.config.max_agent_turns,
                        task_id=task_id,
                        temperature=self.config.agent_temperature,
                        max_tokens=self.config.max_token_length,
                        extra_body=self.config.extra_body,
                    )
                    result = await agent.run(messages)
            except NotImplementedError:
                agent = HermesAgentLoop(
                    server=self.server,
                    tool_schemas=tools,
                    valid_tool_names=valid_names,
                    max_turns=self.config.max_agent_turns,
                    task_id=task_id,
                    temperature=self.config.agent_temperature,
                    max_tokens=self.config.max_token_length,
                    extra_body=self.config.extra_body,
                )
                result = await agent.run(messages)
        else:
            agent = HermesAgentLoop(
                server=self.server,
                tool_schemas=tools,
                valid_tool_names=valid_names,
                max_turns=self.config.max_agent_turns,
                task_id=task_id,
                temperature=self.config.agent_temperature,
                max_tokens=self.config.max_token_length,
                extra_body=self.config.extra_body,
            )
            result = await agent.run(messages)

        # --- Compute reward ---
        only_system_and_user = all(
            msg.get("role") in ("system", "user") for msg in result.messages
        )
        if result.turns_used == 0 or only_system_and_user:
            reward = 0.0
        else:
            # Reuse the same ToolContext (same task_id = same sandbox)
            try:
                reward = await self.compute_reward(item, result, ctx)
            except Exception as e:
                logger.error("compute_reward failed: %s", e)
                reward = 0.0

        # Track tool errors
        if result.tool_errors:
            for err in result.tool_errors:
                self._tool_error_buffer.append({
                    "turn": err.turn,
                    "tool": err.tool_name,
                    "args": err.arguments[:150],
                    "error": err.error[:300],
                    "result": err.tool_result[:300],
                })

        # Cleanup the sandbox
        ctx.cleanup()

        # --- Build ScoredDataItem ---
        nodes = (result.managed_state or {}).get("nodes", [])

        if nodes:
            node = nodes[-1]
            scored_item: Dict[str, Any] = {
                "tokens": node.tokens,
                "masks": node.masked_tokens,
                "scores": reward,
            }
            if hasattr(node, "logprobs") and node.logprobs:
                scored_item["advantages"] = None
                scored_item["ref_logprobs"] = None
        else:
            full_text = "\n".join(
                msg.get("content", "") for msg in result.messages if msg.get("content")
            )
            if self.tokenizer:
                tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
            else:
                tokens = list(range(min(len(full_text) // 4, 128)))

            scored_item = {
                "tokens": tokens,
                "masks": [-100] + tokens[1:],
                "scores": reward,
            }

        scored_item["messages"] = result.messages
        return scored_item, []

    async def compute_reward(
        self, item: Dict[str, Any], result: AgentResult, ctx: ToolContext
    ) -> float:
        """
        Multi-signal reward: test_signal (0.5) + diagnosis (0.3) + efficiency (0.2).

        - test_signal: Run tests in the sandbox. 1.0 if all pass, 0.0 otherwise.
        - diagnosis: Did the agent read the file and run tests before editing?
        - efficiency: Fewer turns = higher reward.
        """
        # --- Test Signal (weight: 0.5) ---
        test_result = ctx.terminal("cd /workspace && python tests.py", timeout=30)
        test_output = test_result.get("output", "")

        if test_result.get("exit_code") == 0 and "ALL TESTS PASSED" in test_output:
            test_signal = 1.0
        elif "TEST FAILED" in test_output:
            # Tests ran but some failed - at least the code compiles
            test_signal = 0.2
        elif "SyntaxError" in test_output or "IndentationError" in test_output:
            # Code doesn't even parse
            test_signal = -0.5
        else:
            # Runtime error
            test_signal = 0.0

        # --- Diagnosis Signal (weight: 0.3) ---
        # Good debugging practice: read the code and run tests before editing
        diagnosis = 0.0
        tool_calls_made = []
        for msg in result.messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    tool_calls_made.append({
                        "name": func.get("name", ""),
                        "args": func.get("arguments", ""),
                    })

        # Check if agent read the buggy file
        read_file = any(
            tc["name"] in ("read_file", "terminal")
            and "buggy.py" in tc["args"]
            for tc in tool_calls_made
        )
        # Check if agent ran tests at any point
        ran_tests = any(
            tc["name"] == "terminal"
            and "tests.py" in tc["args"]
            for tc in tool_calls_made
        )
        # Check if agent edited the file
        edited_file = any(
            tc["name"] in ("write_file", "patch_file", "terminal")
            and ("buggy.py" in tc["args"] and tc["name"] != "terminal"
                 or "buggy.py" in tc["args"] and ">" in tc["args"])
            for tc in tool_calls_made
        )

        if read_file:
            diagnosis += 0.4
        if ran_tests:
            diagnosis += 0.4
        if edited_file:
            diagnosis += 0.2

        # --- Efficiency Signal (weight: 0.2) ---
        max_turns = self.config.max_agent_turns
        efficiency = max(0.0, 1.0 - result.turns_used / max_turns)

        # --- Combined reward ---
        reward = 0.5 * test_signal + 0.3 * diagnosis + 0.2 * efficiency

        # Track components for wandb
        self.test_pass_buffer.append(test_signal)
        self.diagnosis_buffer.append(diagnosis)
        self.efficiency_buffer.append(efficiency)
        self.reward_buffer.append(reward)

        return reward

    async def evaluate(self, *args, **kwargs):
        """Run evaluation on the test split."""
        start_time = time.time()
        # Eval is a lightweight check - just log current stats
        end_time = time.time()

        eval_metrics = {
            "eval/placeholder": 0.0,
        }

        if self.test_pass_buffer:
            eval_metrics["eval/test_pass_rate"] = sum(
                1 for t in self.test_pass_buffer if t == 1.0
            ) / len(self.test_pass_buffer)

        await self.evaluate_log(
            metrics=eval_metrics,
            start_time=start_time,
            end_time=end_time,
        )

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log multi-signal reward components to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        if self.reward_buffer:
            wandb_metrics["train/avg_reward"] = sum(self.reward_buffer) / len(
                self.reward_buffer
            )
            wandb_metrics["train/test_pass_rate"] = sum(
                1 for t in self.test_pass_buffer if t == 1.0
            ) / max(len(self.test_pass_buffer), 1)
            wandb_metrics["train/avg_test_signal"] = sum(
                self.test_pass_buffer
            ) / max(len(self.test_pass_buffer), 1)
            wandb_metrics["train/avg_diagnosis"] = sum(
                self.diagnosis_buffer
            ) / max(len(self.diagnosis_buffer), 1)
            wandb_metrics["train/avg_efficiency"] = sum(
                self.efficiency_buffer
            ) / max(len(self.efficiency_buffer), 1)

            self.reward_buffer = []
            self.test_pass_buffer = []
            self.diagnosis_buffer = []
            self.efficiency_buffer = []

        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    CodeDebugEnv.cli()
