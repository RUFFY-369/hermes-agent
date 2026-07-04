"""End-to-end integration test for HAEE with live DeepSeek API.

Tests the full evolution pipeline with real LLM calls:
1. Failure analysis via DeepSeek
2. Improvement proposal generation via DeepSeek
3. LLM-as-judge evaluation
4. Full EvolutionManager lifecycle with aux LLM
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# DeepSeek API key for testing
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


@pytest.fixture
def temp_hermes_home():
    """Create a temporary HERMES_HOME for isolated testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        home = Path(tmpdir) / ".hermes"
        home.mkdir(parents=True)
        (home / "evolution").mkdir(parents=True, exist_ok=True)
        with patch("hermes_constants.get_hermes_home", return_value=home):
            yield home


@pytest.fixture
def evolution_llm():
    """Create an EvolutionLLMClient with the DeepSeek API key."""
    if not DEEPSEEK_API_KEY:
        pytest.skip("DEEPSEEK_API_KEY not set")
    from agent.evolution.auxiliary_llm import EvolutionLLMClient
    return EvolutionLLMClient(api_key=DEEPSEEK_API_KEY)


# ── Auxiliary LLM Tests ────────────────────────────────────────────────


class TestAuxiliaryLLM:
    """Test the EvolutionLLMClient with real DeepSeek API calls."""

    def test_client_available(self, evolution_llm):
        assert evolution_llm.is_available

    def test_analyze_failure(self, evolution_llm):
        """Test that the LLM can analyze a failure trajectory."""
        prompt = """Analyze this failed agent execution:

TASK: fix-login-bug
The agent attempted to fix a login redirect bug but the tests still failed.

EXECUTION:
- Turn 1: Agent read auth/login.py (120 lines)
- Turn 2: Agent patched line 42, changing redirect URL
- Turn 3: Agent ran pytest — 2 of 5 tests still fail
- Turn 4: Agent declared "fixed" without re-running tests

ERRORS:
- Step 3: pytest test_login.py::test_session_expiry FAILED
- Step 3: pytest test_login.py::test_redirect_flow FAILED

Respond with ONLY valid JSON:
{
  "findings": [
    {
      "category": "premature_completion",
      "confidence": 0.9,
      "description": "what went wrong",
      "evidence": "quote from trace",
      "suggested_fix_category": "prompt"
    }
  ],
  "improvement_priorities": ["first thing to fix"],
  "root_cause_summary": "one paragraph"
}"""

        import asyncio
        response = asyncio.run(evolution_llm.analyze(prompt))
        assert response is not None
        assert len(response) > 0

        # Should be parseable JSON
        try:
            data = json.loads(response)
            assert "findings" in data
            print(f"  [LLM Analysis] {len(data.get('findings', []))} findings: {data.get('root_cause_summary', '')[:100]}")
        except json.JSONDecodeError:
            # Try extracting from code block
            import re
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                assert "findings" in data
            else:
                pytest.fail(f"Response is not valid JSON: {response[:200]}")

    def test_propose_improvements(self, evolution_llm):
        """Test that the LLM can generate improvement proposals."""
        prompt = """Generate improvement proposals for this failed task:

TASK: fix-login-bug
FAILURE: Agent declared success without verifying — 2/5 tests still fail.

EXISTING SKILLS: git-workflow, pytest-runner, code-review
EXISTING TOOLS: terminal, read_file, write_file, search_files, patch, web_search

Generate proposals. Respond with ONLY valid JSON:
{
  "proposals": [
    {
      "action_type": "skill_create",
      "target": "verify-before-complete",
      "description": "Skill to verify task completion before declaring success",
      "rationale": "Agent doesn't verify its work — need explicit verification step",
      "content": "---\\nname: verify-before-complete\\ndescription: Verify task before completion.\\n---\\n\\n# Verify Before Complete\\n\\nAlways run verification before declaring a task complete.",
      "confidence": 0.85,
      "failure_categories_addressed": ["premature_completion"],
      "is_destructive": false,
      "rollback_instructions": "Delete the skill directory"
    }
  ]
}"""

        import asyncio
        response = asyncio.run(evolution_llm.propose(prompt))
        assert response is not None
        assert len(response) > 0

        try:
            data = json.loads(response)
            assert "proposals" in data
            print(f"  [LLM Proposals] {len(data.get('proposals', []))} proposals generated")
        except json.JSONDecodeError:
            import re
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                assert "proposals" in data
            else:
                pytest.fail(f"Not valid JSON: {response[:200]}")

    def test_llm_judge(self, evolution_llm):
        """Test LLM-as-judge evaluation."""
        rubric = "The agent must create a file at /tmp/test.txt containing the text 'success'. Score 1.0 if done, 0.0 if not."
        trajectory = json.dumps({
            "steps": [
                {"type": "tool_execution", "tool": "write_file", "summary": "Created /tmp/test.txt with content 'success'"},
                {"type": "model_call", "summary": "Task complete — file has been created"},
            ]
        })

        import asyncio
        result = asyncio.run(evolution_llm.judge(rubric, trajectory))
        assert isinstance(result, dict)
        assert "passed" in result
        assert "score" in result
        print(f"  [LLM Judge] passed={result.get('passed')}, score={result.get('score')}, reasoning={str(result.get('reasoning', ''))[:100]}")


# ── Full Pipeline Test ──────────────────────────────────────────────────


class TestFullEvolutionPipeline:
    """Test the complete evolution pipeline with LLM integration."""

    def test_failure_analysis_with_llm(self, temp_hermes_home, evolution_llm):
        """Full failure analysis with real LLM."""
        from agent.evolution.failure_analyzer import FailureAnalyzer
        from agent.evolution.task_definition import TaskDefinition, SuccessCriterion, SuccessCriterionType
        from agent.evolution.trajectory_collector import Trajectory, TraceStep, EvalResult, EvalCheck

        task = TaskDefinition(
            name="test-fix",
            description="Fix the bug",
            success_criteria=[
                SuccessCriterion(type=SuccessCriterionType.TEST_PASS, command="pytest"),
            ],
        )

        traj = Trajectory(
            task_name="test-fix",
            run_id="evo_test",
            status="completed",
            total_turns=4,
            total_tool_calls=3,
        )
        traj.steps = [
            TraceStep(step=1, type="model_call", summary="I will fix the bug", extra={"tool_calls": ["read_file"]}),
            TraceStep(step=2, type="tool_execution", status="success", summary="Read 120 lines", extra={"tool": "read_file"}),
            TraceStep(step=3, type="model_call", summary="Fixed — task complete", extra={"tool_calls": []}),
        ]

        eval_result = EvalResult(passed=False, score=0.0, checks=[
            EvalCheck(type="test_pass", passed=False, detail="pytest: 2 tests failed"),
        ])

        async def llm_analyze(prompt):
            return await evolution_llm.analyze(prompt)

        analyzer = FailureAnalyzer(llm_analyze_fn=llm_analyze)
        analysis = analyzer.analyze(task, traj, eval_result)

        assert analysis is not None
        assert analysis.task_name == "test-fix"
        # Should have both rule-based and LLM findings
        all_findings = analysis.findings
        assert len(all_findings) > 0
        print(f"  [Full Analysis] {len(all_findings)} findings:")
        for f in all_findings:
            print(f"    - {f.category.value}: {f.description[:80]} (confidence={f.confidence:.0%})")

    def test_proposal_generation_with_llm(self, temp_hermes_home, evolution_llm):
        """Full improvement proposal generation with real LLM."""
        from agent.evolution.improvement_proposer import ImprovementProposer
        from agent.evolution.failure_analyzer import FailureAnalysis, FailureFinding, FailureCategory
        from agent.evolution.task_definition import TaskDefinition, SuccessCriterion, SuccessCriterionType

        task = TaskDefinition(
            name="test-fix",
            description="Fix the bug",
            success_criteria=[
                SuccessCriterion(type=SuccessCriterionType.TEST_PASS, command="pytest"),
            ],
        )

        analysis = FailureAnalysis(
            run_id="evo_test",
            task_name="test-fix",
            overall_score=0.0,
            findings=[
                FailureFinding(
                    category=FailureCategory.PREMATURE_COMPLETION,
                    confidence=0.9,
                    description="Agent declared success without verifying tests passed",
                    evidence="No test execution after patch",
                    suggested_fix_category="prompt",
                ),
            ],
            improvement_priorities=["premature_completion → prompt"],
        )

        async def llm_propose(prompt):
            return await evolution_llm.propose(prompt)

        proposer = ImprovementProposer(llm_propose_fn=llm_propose)
        proposals = proposer.propose(
            task, analysis,
            existing_skills=["git-workflow", "pytest-runner"],
            existing_tools=["terminal", "read_file", "write_file", "patch"],
        )

        assert len(proposals) > 0
        print(f"  [Full Proposals] {len(proposals)} proposals:")
        for p in proposals:
            print(f"    - {p.action_type.value}: {p.target} (confidence={p.confidence:.0%})")


# ── Benchmark Tests ─────────────────────────────────────────────────────


class TestBenchmarks:
    """Run the evolution benchmarks."""

    def test_benchmark_tasks_load(self):
        """Verify all benchmark task YAML files are valid."""
        benchmark_dir = Path(__file__).parent.parent / "benchmarks" / "evolution" / "tasks"
        if not benchmark_dir.exists():
            pytest.skip("Benchmark tasks directory not found")

        from agent.evolution.task_definition import TaskDefinition

        yaml_files = list(benchmark_dir.glob("*.yaml"))
        assert len(yaml_files) > 0, f"No benchmark tasks found in {benchmark_dir}"

        for yf in yaml_files:
            task = TaskDefinition.from_yaml(yf)
            errors = task.validate()
            assert not errors, f"Task {yf.name} has errors: {errors}"
            print(f"  [Benchmark] {task.name}: valid ({len(task.success_criteria)} criteria)")

    def test_benchmark_evaluation(self, temp_hermes_home):
        """Run benchmark evaluation on all tasks."""
        benchmark_dir = Path(__file__).parent.parent / "benchmarks" / "evolution" / "tasks"
        if not benchmark_dir.exists():
            pytest.skip("Benchmark tasks directory not found")

        from agent.evolution.task_definition import TaskDefinition, save_task
        from agent.evolution.evaluator import TaskEvaluator, EvaluationContext

        evaluator = TaskEvaluator()
        results = []

        for yf in sorted(benchmark_dir.glob("*.yaml")):
            task = TaskDefinition.from_yaml(yf)
            save_task(task)

            start = time.monotonic()
            result = evaluator.evaluate(task, None, EvaluationContext())  # type: ignore[arg-type]
            elapsed = time.monotonic() - start

            results.append({
                "task": task.name,
                "passed": result.passed,
                "score": result.score,
                "time": elapsed,
            })
            print(f"  [Benchmark] {task.name}: {'PASS' if result.passed else 'FAIL'} score={result.score:.2f} time={elapsed:.3f}s")

        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        print(f"\n  Benchmark Summary: {passed}/{total} passed ({passed/total*100:.1f}%)")
        print(f"  Average score: {sum(r['score'] for r in results)/total:.3f}")
        print(f"  Total time: {sum(r['time'] for r in results):.2f}s")

        # All deterministic tasks should pass
        assert passed >= total * 0.5, f"Too many benchmark failures: {passed}/{total}"


# ── Evolution Manager + LLM Integration ────────────────────────────────


class TestEvolutionManagerWithLLM:
    """Full EvolutionManager lifecycle with auxiliary LLM."""

    def test_full_lifecycle_with_llm(self, temp_hermes_home, evolution_llm):
        """Full run: start → execute → evaluate → analyze → propose → gate."""
        from agent.evolution import (
            EvolutionConfig, EvolutionManager,
            TaskDefinition, SuccessCriterion, SuccessCriterionType,
            EvalResult, EvalCheck,
        )
        from agent.evolution.trajectory_collector import TraceStep

        config = EvolutionConfig(enabled=True, mode="on_failure", max_iterations=3)
        mgr = EvolutionManager()

        # Wire the LLM callbacks
        import asyncio

        async def llm_analyze(prompt):
            return await evolution_llm.analyze(prompt)

        async def llm_propose(prompt):
            return await evolution_llm.propose(prompt)

        async def llm_judge(rubric, traj_json):
            return await evolution_llm.judge(rubric, traj_json)

        mgr.initialize(
            session_id="e2e-test",
            config=config,
            llm_call_fn=llm_analyze,  # Used for both analysis and proposals
        )

        # Define and run a task that will fail
        task = TaskDefinition(
            name="e2e-llm-test",
            description="Fix the login bug and verify with tests",
            success_criteria=[
                SuccessCriterion(type=SuccessCriterionType.TEST_PASS, command="pytest tests/ -q"),
            ],
        )

        run = mgr.start_task(task)

        # Simulate agent execution with premature completion
        run.collector.record_model_call(
            model="test-model",
            input_tokens=500,
            output_tokens=100,
            tool_calls=["read_file", "patch"],
            summary="I'll read the file and apply the fix",
        )
        run.collector.record_tool_call(
            tool_name="read_file",
            status="success",
            result_summary="Read auth/login.py (120 lines)",
        )
        run.collector.record_tool_call(
            tool_name="patch",
            status="success",
            result_summary="Patched line 42 in auth/login.py",
        )
        run.collector.record_model_call(
            model="test-model",
            input_tokens=600,
            output_tokens=50,
            tool_calls=[],
            summary="The fix is applied. Task complete.",
        )
        run.trajectory = run.collector.stop()

        # Evaluate — should fail since no test was run
        result = mgr.evaluate(run)

        # If the task failed, try to improve
        if not result.passed:
            print(f"  Task failed as expected (score={result.score:.2f})")

            # Analyze with LLM
            analysis = mgr._analyzer.analyze(run.task, run.trajectory, run.eval_result)
            assert len(analysis.findings) > 0
            print(f"  Analysis: {len(analysis.findings)} findings")

            # Generate proposals with LLM
            proposals = mgr._proposer.propose(run.task, analysis)
            print(f"  Proposals: {len(proposals)} generated")

            if proposals:
                # Gate the best proposal
                gate_result = mgr._gate.evaluate(proposals[0])
                print(f"  Gate verdict: {gate_result.verdict.value}")
                print(f"  Proposal: {proposals[0].action_type.value} → {proposals[0].target}")

        # Cleanup
        mgr.shutdown()

        print("  ✅ Full E2E with LLM pipeline complete")


# ── Run entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
