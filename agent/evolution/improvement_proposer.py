"""Improvement Proposer — the AEGIS "Evolver" for the Evolution Engine.

Generates concrete improvement proposals based on failure analysis. Each proposal
is a specific, reviewable change to the agent's capabilities:

- Skill creation/update (SKILL.md files, uses existing skill_manager_tool)
- Tool creation (new Python functions registered in the tool registry)
- Tool modification (targeted code patches to existing tools)
- Prompt modification (strategy changes, better guidance)
- Memory update (persistent knowledge to prevent recurrence)

Safety: proposals are DRAFTS. They must pass through the RegressionGate
before being applied, and destructive changes require human approval.

Inspired by HarnessX's "typed builder operations" and SIA's code-generation pattern.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.evolution.task_definition import ImprovementActionType, TaskDefinition
from agent.evolution.failure_analyzer import FailureAnalysis, FailureCategory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Improvement proposal data model
# ---------------------------------------------------------------------------


@dataclass
class ImprovementProposal:
    """A concrete, reviewable improvement proposal."""

    action_type: ImprovementActionType
    target: str  # What is being changed (skill name, tool name, prompt section, etc.)
    description: str  # Human-readable description of the change
    rationale: str  # Why this change should fix the failure

    # The actual change content
    content: str = ""  # For skills: SKILL.md content; for tools: Python code; for prompts: new text

    # For code modifications: the specific edit
    file_path: str = ""       # Path to the file being modified
    old_string: str = ""      # For patch operations
    new_string: str = ""

    # Metadata
    confidence: float = 0.5  # Estimated probability this fix works
    failure_categories_addressed: List[str] = field(default_factory=list)
    estimated_token_cost: int = 0

    # Safety
    is_destructive: bool = False  # Could break existing functionality
    requires_approval: bool = True
    rollback_instructions: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "target": self.target,
            "description": self.description,
            "rationale": self.rationale,
            "content": self.content,
            "file_path": self.file_path,
            "old_string": self.old_string,
            "new_string": self.new_string,
            "confidence": self.confidence,
            "failure_categories_addressed": self.failure_categories_addressed,
            "estimated_token_cost": self.estimated_token_cost,
            "is_destructive": self.is_destructive,
            "requires_approval": self.requires_approval,
            "rollback_instructions": self.rollback_instructions,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImprovementProposal":
        try:
            action_type = ImprovementActionType(d["action_type"])
        except ValueError:
            action_type = ImprovementActionType.SKILL_CREATE
        return cls(
            action_type=action_type,
            target=d.get("target", ""),
            description=d.get("description", ""),
            rationale=d.get("rationale", ""),
            content=d.get("content", ""),
            file_path=d.get("file_path", ""),
            old_string=d.get("old_string", ""),
            new_string=d.get("new_string", ""),
            confidence=float(d.get("confidence", 0.5)),
            failure_categories_addressed=d.get("failure_categories_addressed", []),
            estimated_token_cost=int(d.get("estimated_token_cost", 0)),
            is_destructive=bool(d.get("is_destructive", False)),
            requires_approval=bool(d.get("requires_approval", True)),
            rollback_instructions=d.get("rollback_instructions", ""),
        )


# ---------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------


class ImprovementProposer:
    """Generates improvement proposals based on failure analysis.

    Uses an LLM (auxiliary model) to propose concrete, actionable fixes.
    Can also generate rule-based proposals for common failure patterns.
    """

    def __init__(self, llm_propose_fn: Optional[Any] = None):
        """Initialize the proposer.

        Args:
            llm_propose_fn: Optional async callable for LLM-based proposal generation.
              Signature: async (prompt: str) -> str (JSON response with proposals)
        """
        self._llm_propose_fn = llm_propose_fn

    def propose(
        self,
        task: TaskDefinition,
        analysis: FailureAnalysis,
        existing_skills: Optional[List[str]] = None,
        existing_tools: Optional[List[str]] = None,
    ) -> List[ImprovementProposal]:
        """Generate improvement proposals for a failed task.

        Args:
            task: The task that failed.
            analysis: The failure analysis from FailureAnalyzer.
            existing_skills: Names of existing skills (to avoid duplicates).
            existing_tools: Names of existing tools (to avoid duplicates).

        Returns:
            Ordered list of improvement proposals (highest confidence first).
        """
        proposals: List[ImprovementProposal] = []

        # Tier 1: Rule-based proposals for common patterns
        rule_proposals = self._rule_based_proposals(analysis, existing_skills, existing_tools)
        proposals.extend(rule_proposals)

        # Tier 2: LLM-generated proposals (if available)
        if self._llm_propose_fn and analysis.findings:
            try:
                llm_proposals = self._llm_generate_proposals(task, analysis, existing_skills, existing_tools)
                proposals.extend(llm_proposals)
            except Exception as e:
                logger.warning("LLM proposal generation failed: %s", e)

        # Deduplicate by target
        proposals = _deduplicate_proposals(proposals)

        # Sort by confidence (highest first)
        proposals.sort(key=lambda p: p.confidence, reverse=True)

        return proposals

    # -- Rule-based proposals ------------------------------------------------

    def _rule_based_proposals(
        self,
        analysis: FailureAnalysis,
        existing_skills: Optional[List[str]] = None,
        existing_tools: Optional[List[str]] = None,
    ) -> List[ImprovementProposal]:
        """Generate proposals from deterministic rules."""
        proposals: List[ImprovementProposal] = []
        existing_skills = existing_skills or []
        existing_tools = existing_tools or []

        for finding in analysis.findings:
            if finding.category == FailureCategory.MISSING_TOOL:
                # Propose a skill that documents workarounds for missing tools
                proposals.append(ImprovementProposal(
                    action_type=ImprovementActionType.SKILL_CREATE,
                    target=f"workaround-{analysis.task_name}",
                    description=f"Skill documenting workarounds for missing tool capability in '{analysis.task_name}'",
                    rationale=f"Agent needed a capability that doesn't exist: {finding.description}",
                    failure_categories_addressed=[finding.category.value],
                    confidence=0.5,
                    requires_approval=False,
                    rollback_instructions="Delete the skill directory",
                ))

            elif finding.category == FailureCategory.PREMATURE_COMPLETION:
                # Propose a prompt modification to enforce verification
                proposals.append(ImprovementProposal(
                    action_type=ImprovementActionType.PROMPT_MODIFY,
                    target="verification_guidance",
                    description="Strengthen verification requirements in agent prompts",
                    rationale="Agent declared success without verifying. Need explicit verification step.",
                    failure_categories_addressed=[finding.category.value],
                    confidence=0.7,
                    is_destructive=False,
                    requires_approval=True,
                    rollback_instructions="Revert prompt changes",
                ))

            elif finding.category == FailureCategory.LOOP:
                proposals.append(ImprovementProposal(
                    action_type=ImprovementActionType.PROMPT_MODIFY,
                    target="loop_detection_guidance",
                    description="Add loop-detection guidance to agent prompts",
                    rationale=f"Agent got stuck in loop: {finding.description}",
                    failure_categories_addressed=[finding.category.value],
                    confidence=0.65,
                    is_destructive=False,
                    requires_approval=True,
                    rollback_instructions="Revert prompt changes",
                ))

            elif finding.category == FailureCategory.EXECUTION_ERROR:
                proposals.append(ImprovementProposal(
                    action_type=ImprovementActionType.SKILL_CREATE,
                    target=f"troubleshoot-{finding.implicated_tools[0] if finding.implicated_tools else 'tools'}",
                    description=f"Troubleshooting guide for common {finding.implicated_tools[0] if finding.implicated_tools else 'tool'} errors",
                    rationale=f"Tool execution errors: {finding.description}",
                    failure_categories_addressed=[finding.category.value],
                    confidence=0.55,
                    requires_approval=False,
                    rollback_instructions="Delete the skill directory",
                ))

        return proposals

    # -- LLM-based proposals -------------------------------------------------

    def _llm_generate_proposals(
        self,
        task: TaskDefinition,
        analysis: FailureAnalysis,
        existing_skills: Optional[List[str]],
        existing_tools: Optional[List[str]],
    ) -> List[ImprovementProposal]:
        """Use an LLM to generate improvement proposals."""
        if not self._llm_propose_fn:
            return []

        import asyncio

        prompt = _build_proposal_prompt(task, analysis, existing_skills or [], existing_tools or [])
        try:
            response = asyncio.get_event_loop().run_until_complete(
                self._llm_propose_fn(prompt)
            )
        except Exception:
            response = asyncio.run(self._llm_propose_fn(prompt))

        return _parse_llm_proposals(response)


# ---------------------------------------------------------------------------
# Proposal prompt
# ---------------------------------------------------------------------------


def _build_proposal_prompt(
    task: TaskDefinition,
    analysis: FailureAnalysis,
    existing_skills: List[str],
    existing_tools: List[str],
) -> str:
    """Build the prompt for LLM-based proposal generation."""
    findings_text = "\n".join(
        f"  [{i+1}] {f.category.value} (confidence: {f.confidence:.0%})\n"
        f"      Description: {f.description}\n"
        f"      Evidence: {f.evidence[:200]}\n"
        f"      Suggested fix category: {f.suggested_fix_category}"
        for i, f in enumerate(analysis.findings)
    )

    failed_checks_text = "\n".join(
        f"  - {c.get('type', 'unknown')}: {c.get('detail', '')[:200]}"
        for c in analysis.failed_checks
    )

    existing_skills_text = "\n".join(f"  - {s}" for s in existing_skills[:30]) if existing_skills else "  (none)"
    existing_tools_text = "\n".join(f"  - {t}" for t in existing_tools[:50]) if existing_tools else "  (none)"

    return f"""You are an improvement agent for an AI agent system. Generate concrete, actionable proposals to fix the failures described below.

FAILED TASK: {task.name}
DESCRIPTION: {task.description}
DOMAIN: {task.domain}

FAILURE ANALYSIS:
{findings_text}

FAILED EVALUATION CHECKS:
{failed_checks_text}

EXISTING SKILLS (don't create duplicates):
{existing_skills_text}

EXISTING TOOLS (don't create duplicates):
{existing_tools_text}

IMPROVEMENT PRIORITIES: {', '.join(analysis.improvement_priorities)}

Generate improvement proposals. Respond with ONLY valid JSON:

{{
  "proposals": [
    {{
      "action_type": "skill_create|skill_patch|tool_create|tool_modify|prompt_modify|memory_update|strategy_change",
      "target": "<name of skill, tool, prompt section, etc.>",
      "description": "<one-line description of the change>",
      "rationale": "<why this addresses the root cause>",
      "content": "<the full SKILL.md content, Python code, or prompt text>",
      "file_path": "<if modifying an existing file, the path>",
      "old_string": "<if patching, the exact text to replace>",
      "new_string": "<if patching, the replacement text>",
      "confidence": 0.0-1.0,
      "failure_categories_addressed": ["<category names>"],
      "is_destructive": true/false,
      "rollback_instructions": "<how to undo this change>"
    }}
  ]
}}

RULES:
- Be concrete. Generate actual SKILL.md content or Python code, not just descriptions.
- For skill_create: include FULL SKILL.md with frontmatter (name, description ≤60 chars, version 0.1.0).
- For tool_create: include complete Python function with type hints and docstring.
- For skill_patch: specify exact old_string and new_string for targeted replacement.
- Don't propose tools/skills that already exist (check the lists above).
- Don't propose changes that would break existing functionality.
- Prioritize the highest-confidence failure categories first.
"""
    return prompt


def _parse_llm_proposals(response: str) -> List[ImprovementProposal]:
    """Parse LLM-generated proposals from JSON response."""
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM proposals")
                return []
        else:
            return []

    proposals = []
    for p in data.get("proposals", []):
        try:
            action_type = ImprovementActionType(p.get("action_type", "skill_create"))
        except ValueError:
            action_type = ImprovementActionType.SKILL_CREATE

        proposals.append(ImprovementProposal(
            action_type=action_type,
            target=p.get("target", ""),
            description=p.get("description", ""),
            rationale=p.get("rationale", ""),
            content=p.get("content", ""),
            file_path=p.get("file_path", ""),
            old_string=p.get("old_string", ""),
            new_string=p.get("new_string", ""),
            confidence=float(p.get("confidence", 0.5)),
            failure_categories_addressed=p.get("failure_categories_addressed", []),
            is_destructive=bool(p.get("is_destructive", False)),
            requires_approval=action_type.value not in ("skill_create", "skill_patch"),
            rollback_instructions=p.get("rollback_instructions", ""),
        ))
    return proposals


def _deduplicate_proposals(proposals: List[ImprovementProposal]) -> List[ImprovementProposal]:
    """Remove duplicate proposals (same action_type + target)."""
    seen: set = set()
    unique = []
    for p in proposals:
        key = (p.action_type.value, p.target)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique
