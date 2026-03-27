# CodeDebugEnv (hermes-agent implementation)

A multi-turn RL environment for training and evaluating Hermes agents on code debugging tasks (e.g., HumanEvalPack).

## Features
- **Real Tool Execution**: The agent has access to `terminal`, `read_file`, `write_file`, and `patch`.
- **Execution-based Scoring**: Rewards are calculated based on the actual test pass rate after the agent's fixes.
- **Universal Tool Strategy**: A specialized architecture for stabilizing vLLM inference by bypassing server-side tool-call parsing and using ultra-robust client-side parsing.

## Configuration
The environment is configured via `default.yaml`. Key settings:
- `atropos_inhibit_tools: True`: Bypasses vLLM's internal tool parser to prevent 400/500 errors.
- `system_prompt`: Contains the manual tool documentation for the client-side parser.

## Usage
Run the environment in process mode:
```bash
/opt/conda/envs/hermes_conda/bin/python environments/code_debug_env/code_debug_env.py process \
  --config environments/code_debug_env/default.yaml
```

## Stability Notes
We use a custom `parse_tool_calls_from_text` function in `model_tools.py` to extract tool calls from `<tool_code>` tags. This handles varied model outputs and is much more stable than standard server-side parsing for vLLM 0.6.5.
