import asyncio
from typing import Optional
from evolution.client import SGLangClient

class PRMJudge:
    """
    Process Reward Model (PRM) for hindsight hint extraction within Hermes-Agent.
    Transforms raw user corrections into actionable logic hints.
    """
    def __init__(self, client: SGLangClient):
        self.client = client

    async def extract_hindsight_hint(
        self, 
        original_prompt: str, 
        failed_trajectory: str, 
        user_correction: str
    ) -> str:
        """
        Generates a 1-sentence hindsight hint via SGLang.
        """
        meta_prompt = f"""
### Task
Original Prompt: {original_prompt}
Failed Attempt: {failed_trajectory}
User Correction: {user_correction}

### Instruction
Extract a 1-sentence "hindsight hint" that explains the core logic error and the fix. 
Focus on the 'Why' and 'How'. Do not be conversational.
Hint:"""

        results = await self.client.generate_group(
            prompts=[meta_prompt], 
            max_new_tokens=64, 
            temperature=0.1
        )
        
        return results[0].text.strip()
