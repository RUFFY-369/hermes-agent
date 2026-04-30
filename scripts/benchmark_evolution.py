import asyncio
import os
import sys
import json
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from evolution.client import SGLangClient
from evolution.sandbox import DockerSandbox
from evolution.sync import LoRASyncEngine

TASKS = [
    {
        "name": "N-Queens (No If)",
        "prompt": (
            "Write a Python function called solve() that returns the number of valid N-Queens solutions for N=8. "
            "STRICT CONSTRAINT: You must not use the 'if' keyword anywhere in your code. "
            "Rely entirely on boolean short-circuiting or array masking."
        )
    },
    {
        "name": "Matrix Rotation (No Loops)",
        "prompt": (
            "Write a Python function called solve() that takes a 3x3 matrix (list of lists) and returns it rotated 90 degrees clockwise. "
            "STRICT CONSTRAINT: You must not use any 'for' or 'while' loops. "
            "Use list comprehensions, `zip(*)`, or map/lambda."
        )
    },
    {
        "name": "Fibonacci (No Plus)",
        "prompt": (
            "Write a Python function called solve() that returns the 10th Fibonacci number. "
            "STRICT CONSTRAINT: You must not use the '+' operator. "
            "Use bitwise operations or `sum()` for addition."
        )
    },
    {
        "name": "Prime Sieve (No Range)",
        "prompt": (
            "Write a Python function called solve() that returns all prime numbers up to 50 using a sieve. "
            "STRICT CONSTRAINT: You must not use the 'range()' function. "
            "Use `slice()` or direct list manipulation instead."
        )
    }
]

async def evaluate_adapter(client: SGLangClient, sandbox: DockerSandbox, adapter_path: str = None, adapter_name: str = "active_policy", samples: int = 4):
    print(f"\n🔍 Evaluating: {'Base Model' if not adapter_path else adapter_path}")
    
    results = {}
    
    for task in TASKS:
        print(f"   Task: {task['name']}...", end="", flush=True)
        
        full_prompt = (
            f"### TASK\n{task['prompt']}\n\n"
            "### INSTRUCTION\n"
            "Implement the solution in a function called `solve()`. "
            "Provide ONLY the code block.\n\n"
            "### SOLUTION\n```python\ndef solve():\n"
        )
        
        prompts = [full_prompt] * samples
        rollouts = await client.generate_group(prompts, lora_path=adapter_path if adapter_path else None)
        
        grading_tasks = []
        for r in rollouts:
            if not r.text.strip():
                async def failed_reward(): return -1.0
                grading_tasks.append(failed_reward())
            else:
                grading_tasks.append(sandbox.execute_code(r.text))
        
        rewards = await asyncio.gather(*grading_tasks)
        mean_reward = sum([float(r) for r in rewards]) / len(rewards)
        
        results[task['name']] = mean_reward
        print(f" Done. Mean Reward: {mean_reward:+.2f}")
        
    return results

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, help="Path to LoRA adapter to benchmark")
    parser.add_argument("--samples", type=int, default=8, help="Number of samples per task")
    args = parser.parse_args()

    client = SGLangClient()
    sandbox = DockerSandbox()
    sync = LoRASyncEngine()
    
    try:
        # 1. Benchmark Base Model
        base_results = await evaluate_adapter(client, sandbox, adapter_path=None, samples=args.samples)
        
        # 2. Benchmark Adapter (if provided)
        target_results = None
        if args.adapter:
            # Sync the adapter to SGLang first
            print(f"\n🔄 Syncing adapter {args.adapter} to SGLang...")
            await sync.sync_weights(adapter_path=args.adapter, adapter_name="eval_policy")
            target_results = await evaluate_adapter(client, sandbox, adapter_path=args.adapter, samples=args.samples)

        # 3. Print Comparison Table
        print("\n" + "=" * 60)
        print(f"{'TASK':<30} | {'BASE':<8} | {'TARGET':<8} | {'DELTA':<8}")
        print("-" * 60)
        for task_name in base_results:
            base_r = base_results[task_name]
            target_r = target_results[task_name] if target_results else "N/A"
            delta = (target_r - base_r) if isinstance(target_r, float) else 0.0
            
            target_str = f"{target_r:+.2f}" if isinstance(target_r, float) else "N/A"
            delta_str = f"{delta:+.2f}" if isinstance(target_r, float) else "N/A"
            
            print(f"{task_name:<30} | {base_r:+.2f} | {target_str:<8} | {delta_str:<8}")
        print("=" * 60)

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
