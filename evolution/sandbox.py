import asyncio
import re
import ast
import logging
import tempfile
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DockerSandbox:
    """
    Asynchronous Sandbox for code execution.
    Uses non-blocking asyncio subprocesses and Dense Reward Shaping.
    
    Supports two modes:
    - Docker mode (preferred): Runs code in ephemeral containers with resource limits.
    - Subprocess mode (fallback): Runs code via local Python subprocess when Docker is unavailable.
    """
    def __init__(self, image: str = "python:3.11-slim", use_docker: bool = None):
        self.image = image
        # Auto-detect Docker availability if not explicitly set
        if use_docker is None:
            self.use_docker = self._check_docker()
        else:
            self.use_docker = use_docker
        
        mode = "Docker" if self.use_docker else "Subprocess (fallback)"
        logger.info(f"Sandbox initialized in {mode} mode")

    def _check_docker(self) -> bool:
        """Check if Docker daemon is available."""
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "info"], 
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    async def execute_code(self, code_string: str, timeout: float = 10.0) -> float:
        """
        Executes Python code and returns a shaped reward:
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

        # 2. Execute via Docker or subprocess
        if self.use_docker:
            return await self._execute_docker(code, timeout, reward)
        else:
            return await self._execute_subprocess(code, timeout, reward)

    async def _execute_docker(self, code: str, timeout: float, base_reward: float) -> float:
        """Execute code in Docker container."""
        cmd = [
            "docker", "run", "--rm", "--network", "none",
            "--memory", "256m", "--cpus", "0.5",
            self.image, "python", "-c", code
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
                exit_code = proc.returncode
            except asyncio.TimeoutError:
                try:
                    proc.terminate()
                    await proc.wait()
                except:
                    pass
                return -0.5  # Hang penalty

            if exit_code == 0:
                return 1.0  # Success
            else:
                return base_reward + 0.3  # 0.5 Total: Logic Error (Runtime)
                
        except Exception as e:
            logger.error(f"Docker Sandbox Error: {e}")
            return -1.0

    async def _execute_subprocess(self, code: str, timeout: float, base_reward: float) -> float:
        """Execute code via local Python subprocess (fallback when Docker unavailable)."""
        # Write code to a temp file for safer execution
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="sandbox_")
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                f.write(code)

            proc = await asyncio.create_subprocess_exec(
                "python3", tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                exit_code = proc.returncode
            except asyncio.TimeoutError:
                try:
                    proc.terminate()
                    await proc.wait()
                except:
                    pass
                return -0.5  # Hang penalty

            if exit_code == 0:
                return 1.0  # Success
            else:
                return base_reward + 0.3  # 0.5 Total: Logic Error (Runtime)
                
        except Exception as e:
            logger.error(f"Subprocess Sandbox Error: {e}")
            return -1.0
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass

if __name__ == "__main__":
    async def verify():
        sandbox = DockerSandbox()
        print(f"🚀 Testing Sandbox (Docker={sandbox.use_docker})...")
        
        tasks = [
            sandbox.execute_code("print('Hello')"),
            sandbox.execute_code("import time; time.sleep(20)"),  # Timeout
            sandbox.execute_code("this is not python"),  # Syntax
            sandbox.execute_code("assert 1 == 0")  # Runtime
        ]
        
        results = await asyncio.gather(*tasks)
        print(f"Results: {results}")
        # Expected: [1.0, -0.5, -1.0, 0.5]

    asyncio.run(verify())
