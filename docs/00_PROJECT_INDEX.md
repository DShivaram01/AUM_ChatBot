# AUM Academic Assistant — Project Guide Index

**Status date:** 2026-08-24  
**Guide version:** v2 — updated after the deep-research architecture review  
**Project phase:** Research prototype → provenance-first architecture stabilization

This directory is the shared operating guide for the AUM Academic Assistant. It should be usable by the project owner, Claude Code, Codex, and future engineering agents even if all chat history disappears.

## Required reading order

Before substantial architectural work, read:

1. `project_updates.md` — current state and immediate work queue.
2. `project_scope.md` — product boundary, capabilities, non-goals, success criteria.
3. `engineering_principles.md` — engineering rules and architectural invariants.
4. `AUM_DEEP_RESEARCH_ARCHITECTURE_2026-08-24.md` — evidence-backed architecture reference and source registry.
5. `decision_register.md` — current accepted/deferred/benchmark-first decisions.
6. `source_provenance_and_versioning.md` — authoritative-source, versioning, freshness, and provenance rules.
7. `testing_and_evaluation.md` — software tests + AUM Gold Set + RAG evaluation.
8. `development_workflow.md` — task lifecycle and Definition of Done.
9. `ai_agent_workflow.md` — Claude Code/Codex collaboration rules.
10. `roadmap.md` — staged implementation sequence.
11. `deployment_and_benchmark_plan.md` — local serving and RTX 4090 benchmark plan.
12. `kg_phase0.md` — Knowledge Graph prerequisites; not an implementation selection.
13. `skills_and_roles.md` — engineering/research skills required.
14. `agent_handoff_template.md` — mandatory significant-task handoff.
15. `deep_research_prompt.md` — reusable prompt when existing research needs updating or a new major decision needs evidence.

## Core project statement

The project is not a collection of separate chatbots and is not merely a vector-search RAG demo.

> **AUM Academic Assistant is a capability-oriented retrieval system over authoritative, versioned institutional knowledge. It uses deterministic structured lookup when possible, hybrid text retrieval when appropriate, bounded graph traversal when relationships matter, calibrated abstention when evidence is insufficient, and a local LLM primarily for grounded interpretation and synthesis.**

COS and Housing are the first two implemented knowledge areas, not the final product boundary.

## Direction of travel

```text
User
  ↓
Unified Interface
  ↓
FastAPI / Application Boundary
  ↓
Query Orchestrator
  ├── Router
  ├── Conversation State
  └── Access / Policy Gate
        ↓
Capability Retrieval
  ├── Structured Lookup
  ├── Hybrid RAG
  └── Future KG Traversal
        ↓
Evidence Sufficiency Gate
  ├── Insufficient → abstain / clarify
  └── Sufficient → grounded synthesis
        ↓
Structured Citations / Sources
```

## Immediate principle

Do not broaden the assistant faster than we can measure, version, trace, and safely update it.

The Knowledge Graph is important, but implementation begins only after `kg_phase0.md` exit criteria are satisfied.
