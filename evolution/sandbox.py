import asyncio
import re
import ast
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DockerSandbox:
    """
    H100-Optimized Asynchronous Sandbox.
    Uses non-blocking asyncio subprocesses and Dense Reward Shaping.
    Designed for G=64+ concurrency.
    """
    def __init__(self, image: str = "python:3.11-slim"):
        self.image = image

    async def execute_code(self, code_string: str, timeout: float = 10.0) -> float:
        """
        Executes Python code in an ephemeral container using non-blocking I/O.
        Returns a shaped reward:
        - -1.0: Syntax Error / Fatal Failure
        - -0.5: Timeout / Hang
        - +0.2: Valid Syntax (AST Parsed)
        - +0.5: Compiles and runs but returns non-zero exit code (Logic Error)
        - +1.0: Perfect Run (Exit Code 0)
        """
        # 1. Pre-execution Check (AST Parsing)
        code = re.sub(r"```python\n|```", "", code_string).strip()
        if not code:
            return -1.0
            
        try:
            ast.parse(code)
            reward = 0.2  # Reward for valid syntax
        except SyntaxError:
            return -1.0

        # 2. Async Subprocess Execution
        # We use 'docker run' with strict resource limits
        cmd = [
            "docker", "run", "--rm", "--network", "none",
            "--memory", "256m", "--cpus", "0.5",
            self.image, "python", "-c", code
        ]

        try:
            # Launch without blocking the event loop
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                # Wait for process with timeout
                await asyncio.wait_for(proc.wait(), timeout=timeout)
                exit_code = proc.returncode
            except asyncio.TimeoutError:
                # Cleanup hanging container
                try:
                    # We don't have the container ID easily here, 
                    # but '--rm' and proc.terminate() usually handle it.
                    proc.terminate()
                    await proc.wait()
                except:
                    pass
                return -0.5  # Hang penalty

            # 3. Final Reward Calculation
            if exit_code == 0:
                return 1.0  # Success
            else:
                return reward + 0.3  # 0.5 Total: Logic Error (Runtime)
                
        except Exception as e:
            logger.error(f"Sandbox Error: {e}")
            return -1.0

if __name__ == "__main__":
    async def verify():
        sandbox = DockerSandbox()
        print("🚀 Testing H100-Scale Async Sandbox...")
        
        tasks = [
            sandbox.execute_code("print('Hello')"),
            sandbox.execute_code("import time; time.sleep(20)"), # Timeout
            sandbox.execute_code("this is not python"), # Syntax
            sandbox.execute_code("assert 1 == 0") # Runtime
        ]
        
        results = await asyncio.gather(*tasks)
        print(f"Results: {results}")
        # Expected: [1.0, -0.5, -1.0, 0.5]

    asyncio.run(verify())
