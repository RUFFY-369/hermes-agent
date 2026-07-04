# Hermes Agent Evolution Engine (HAEE) — Implementation Plan

## Research Foundation

This design synthesizes three cutting-edge research systems:

1. **HarnessX (Xiaomi Darwin Agent Team, ICLR 2026)**: AEGIS 4-stage evolution pipeline
   (Digester→Planner→Evolver→Critic+Gate), operational mirror mapping RL to symbolic
   harness evolution, variant isolation, deterministic seesaw constraint. Paper only.
   +14.5% avg gain across 5 benchmarks.

2. **SIA (Hexo AI, MIT License)**: Working open-source implementation of 3-agent loop
   (Meta→Target→Feedback). Code-level harness evolution. 23 commits, v0.5.1.

3. **Darwin Gödel Machine / HyperAgents (Meta, ICLR 2026)**: Self-modifying code with
   archive-based evolution. Metacognitive self-modification across domains.

## Gap Analysis

What exists in Hermes Agent: skill creation (SKILL.md only), background review,
curator (cleanup only), trajectory compression (for training data), Atropos RL
(separate framework for model training).

What does NOT exist: built-in evaluation, automatic tool creation, code-level
self-modification, closed-loop autonomous improvement, failure-driven capability
acquisition. The hermes-agent-self-evolution repo (8 commits) is an external
research prototype, not integrated.

## Architecture

### Phase 1: Evolution Harness Foundation (this PR)

```
agent/evolution/
├── __init__.py              # Public API
├── evolution_manager.py     # Central orchestrator (MemoryManager pattern)
├── task_definition.py       # Task spec model + YAML parser
├── trajectory_collector.py  # Trace capture during agent execution
├── evaluator.py             # Multi-method task evaluation
├── failure_analyzer.py      # Digester: root-cause analysis
├── improvement_proposer.py  # Evolver: generate improvements
├── regression_gate.py       # Seesaw constraint + deterministic checks
├── evolution_store.py       # SQLite persistence for evolution history
├── harness_variants.py      # Variant isolation (inspired by HarnessX)
├── evolution_tools.py       # Model-facing tools (evolution_run, etc.)
├── evolution_hooks.py       # Lifecycle hooks for turn-level instrumentation
└── config.py                # Evolution-specific configuration
```

### Phase 2: Self-Modification Capabilities (follow-up)
- Tool code generation and modification
- System prompt optimization  
- MCP server auto-creation
- Continuous improvement loop

### Phase 3: Co-Evolution Flywheel (follow-up)
- Integration with Atropos RL
- Model+harness co-evolution
- Cross-harness GRPO
- Training data generation from evolution traces

## Core Loop

```
1. User defines task + success criteria
2. Agent attempts task (with trajectory capture)
3. Evaluator scores the attempt
4. If failed: FailureAnalyzer identifies root causes
5. ImprovementProposer generates candidate fixes:
   a. New/modified SKILL.md (existing infra)
   b. New tool registration (new capability)
   c. Prompt modification (strategy change)
6. RegressionGate validates against prior successes
7. Agent retries with improvements
8. Loop until success or exhaustion
```

## Safety Architecture (HarnessX-inspired)

1. **Seesaw Constraint**: No improvement may regress any previously-solved task
2. **Deterministic Gates**: All changes must pass:
   - Schema validation (tool signatures intact)
   - Build/smoke test (code imports successfully)
   - Regression test (prior successes still pass)
3. **Human-in-the-Loop**: Destructive changes require approval
4. **Rollback**: Every modification is reversible via git-backed state
5. **Variant Isolation**: Conflicting improvements fork into separate harness variants

## Integration with Existing Hermes Infrastructure

| Existing Component | How HAEE Uses It |
|-------------------|-----------------|
| MemoryManager | Stores evolution state, traces, evaluation results |
| skill_manager_tool | Creates/patches skills from improvement proposals |
| Background review fork | Runs evaluation and improvement asynchronously |
| Curator | Evolution-aware lifecycle management |
| Tool registry | Target for tool creation/modification |
| Plugin system | Evolution hooks as plugin lifecycle hooks |
| Config system | Evolution settings in config.yaml |

## Key Design Decisions

1. **Built-in, not external**: Unlike hermes-agent-self-evolution (separate repo), HAEE
   lives inside the agent runtime. Evolution happens DURING operation, not as a batch job.

2. **Provider pattern**: EvolutionManager follows the same pattern as MemoryManager —
   one manager, pluggable evaluation backends, tools gated on configuration.

3. **Deterministic safety over LLM judgment**: Following HarnessX, all critical decisions
   use deterministic checks. LLMs propose; gates dispose.

4. **Trace-first**: Every evolution decision is grounded in execution traces, not model
   introspection. This prevents the "self-trust problem" identified in the community.

5. **Minimal token overhead**: Evolution runs on a separate model (auxiliary client)
   and only activates on failure. Zero overhead for successful tasks.
