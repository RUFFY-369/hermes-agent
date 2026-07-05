"""PR Proposer — code fixes become reviewable git branches for maintainers.

Completes the loop: agent usage → failure → analysis → code fix → git branch → PR → merge.

Safe: nothing reaches users without maintainer approval.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CodeFixProposal:
    title: str
    description: str
    file_path: str
    original_code: str
    proposed_code: str
    failure_evidence: Dict[str, Any] = field(default_factory=dict)
    benchmark_results: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    created_at: str = ""

    def to_pr_body(self) -> str:
        e = self.failure_evidence
        b = self.benchmark_results
        diff = "\n".join(difflib.unified_diff(
            self.original_code.splitlines(keepends=True),
            self.proposed_code.splitlines(keepends=True),
            fromfile=f"a/{self.file_path}", tofile=f"b/{self.file_path}",
        ))
        return f"""## HAEE Auto-Proposed Fix

**Failure**: {e.get('failure_type', 'unknown')}
**Occurrences**: {e.get('occurrences', 0)}x across {e.get('sessions', 0)} sessions
**Confidence**: {self.confidence:.0%}

### Evidence
{e.get('description', '')}
```
{e.get('evidence_excerpt', '')}
```

### Change
**File**: `{self.file_path}`

```diff
{diff[:5000]}
```

### Safety
| Gate | Result |
|------|--------|
| Regression | {b.get('gate_verdict', 'N/A')} |
| Baselines | {b.get('baselines_checked', 0)} passed |
| Smoke test | {b.get('smoke_test', 'N/A')} |

### Review Checklist
- [ ] Code addresses root cause
- [ ] No regression on existing functionality
- [ ] Follows Hermes conventions
"""

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "description": self.description,
                "file_path": self.file_path, "confidence": self.confidence,
                "failure_evidence": self.failure_evidence,
                "benchmark_results": self.benchmark_results}


class PRProposer:
    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path.cwd()

    def propose_from_failure(self, failure_analysis: Dict, tool_name: str,
                            proposed_code: str, benchmarks: Optional[Dict] = None
                            ) -> Optional[CodeFixProposal]:
        findings = failure_analysis.get("findings", [])
        if not findings:
            return None
        top = findings[0]
        original = self._read_code(tool_name)
        if not original:
            return None
        proposal = CodeFixProposal(
            title=f"fix({tool_name}): {top.get('category', 'auto-fix')}",
            description=top.get("description", ""),
            file_path=self._resolve_path(tool_name),
            original_code=original, proposed_code=proposed_code,
            failure_evidence={"failure_type": top.get("category"),
                "occurrences": failure_analysis.get("total_occurrences", 1),
                "sessions": failure_analysis.get("total_sessions", 1),
                "description": top.get("description", ""),
                "evidence_excerpt": top.get("evidence", "")[:500]},
            benchmark_results=benchmarks or {},
            confidence=top.get("confidence", 0.5),
            created_at=datetime.now(timezone.utc).isoformat())
        return proposal

    def create_branch(self, proposal: CodeFixProposal) -> Optional[str]:
        branch = f"haee-fix/{re.sub(r'[^a-z0-9-]', '-', proposal.title.lower())[:50].strip('-')}"
        try:
            fp = self.repo_path / proposal.file_path
            fp.write_text(proposal.proposed_code)
            subprocess.run(["git", "checkout", "-b", branch], cwd=self.repo_path,
                          capture_output=True, check=True)
            subprocess.run(["git", "add", str(proposal.file_path)], cwd=self.repo_path,
                          capture_output=True, check=True)
            subprocess.run(["git", "commit", "-m",
                f"{proposal.title}\n\n{proposal.description}\n\n"
                f"HAEE auto-proposed. Confidence: {proposal.confidence:.0%}."],
                cwd=self.repo_path, capture_output=True, check=True)
            return branch
        except subprocess.CalledProcessError:
            return None

    def _read_code(self, tool_name: str) -> Optional[str]:
        for path in [self._resolve_path(tool_name),
                     f"tools/{tool_name}.py", f"tools/{tool_name}_tool.py"]:
            fp = self.repo_path / path
            if fp.exists():
                try: return fp.read_text()
                except: pass
        return None

    def _resolve_path(self, tool_name: str) -> str:
        for c in [f"tools/{tool_name}.py", f"tools/{tool_name}_tool.py"]:
            if (self.repo_path / c).exists():
                return c
        return f"tools/{tool_name}.py"


def propose_code_fix(failure_analysis: Dict, proposed_code: str,
                    tool_name: str, repo_path: Optional[Path] = None) -> Dict[str, Any]:
    proposer = PRProposer(repo_path=repo_path)
    proposal = proposer.propose_from_failure(failure_analysis, tool_name, proposed_code)
    if not proposal:
        return {"error": "Could not generate proposal"}
    result = {"title": proposal.title, "file_path": proposal.file_path,
              "pr_body": proposal.to_pr_body(), "confidence": proposal.confidence}
    branch = proposer.create_branch(proposal)
    if branch:
        result["branch_name"] = branch
    return result
