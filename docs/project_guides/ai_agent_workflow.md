# Claude Code + Codex Workflow

Claude Code and Codex are engineering contributors to one live repository.

They should not evolve independent versions of the project.

---

## 1. Required context

Before substantial work, read:

1. `00_PROJECT_INDEX.md`
2. `project_updates.md`
3. `engineering_principles.md`
4. `decision_register.md`
5. task-specific source files
6. latest handoff
7. relevant sections of `AUM_DEEP_RESEARCH_ARCHITECTURE_2026-08-24.md`

For source/provenance, serving, security, routing, evaluation, or KG decisions, the deep-research reference is mandatory context.

---

## 2. Suggested role split

### Claude Code

Useful for:

- repository exploration;
- architecture tracing;
- multi-file refactors;
- integration work;
- long-running debugging;
- design/handoff documentation.

### Codex

Useful for:

- focused implementation;
- tests;
- code review;
- bug isolation;
- targeted refactors;
- benchmark harnesses;
- schema/interface implementation.

Use whichever agent is strongest for the task; the important rule is one shared architecture.

---

## 3. One primary implementation owner

For a significant task:

- one agent implements;
- the other should preferably review, test, or challenge assumptions.

Independent duplicate rewrites should be deliberate experiments, not routine workflow.

---

## 4. Task packet

Give the primary agent:

```text
TASK ID
GOAL
WHY
CURRENT BEHAVIOR
ACCEPTED DECISIONS
FILES TO INSPECT
FILES ALLOWED TO CHANGE
SOURCE/VERSION IMPACT
NON-GOALS
ACCEPTANCE CRITERIA
TESTS/EVALS
HANDOFF FORMAT
```

---

## 5. Architecture guardrails

Agents must not casually:

- put business logic in Gradio;
- create new unbounded globals;
- add another hard-coded domain branch when capability interfaces are appropriate;
- create one global final vector index for all university knowledge;
- interpret similarity as calibrated confidence;
- ask the LLM to regenerate known structured facts;
- remove provenance/version fields for convenience;
- overwrite old source/index versions destructively;
- log sensitive raw content without policy;
- expose protected documents before authorization;
- select a graph database before KG Phase 0;
- route every query through a graph or LLM planner;
- replace Mistral without proving generation is the failure;
- quantize without a measured deployment objective;
- build a 21-node GPU cluster before single-node measurements justify it;
- choose TGI as a new default without re-verifying its maintenance status.

---

## 6. Stale snapshots

A copied file from another agent is a proposed patch, not ground truth.

Always inspect the current live branch.

---

## 7. Pre-change behavior

The agent should:

1. inspect git status/current commit;
2. inspect current files;
3. trace execution/data flow;
4. identify accepted constraints;
5. identify test/eval impact;
6. state plan and risks.

---

## 8. Post-change behavior

Provide:

- changed files;
- new/changed schemas;
- source/version impact;
- tests;
- Gold Set/eval results;
- benchmark results where relevant;
- architectural impact;
- assumptions;
- unresolved risks;
- next recommended action;
- commit hash if committed.

Use `agent_handoff_template.md`.

---

## 9. Research claims

If an agent wants to challenge an accepted evidence-backed recommendation:

- cite the relevant existing source ID if applicable;
- identify what changed;
- provide newer evidence or a project benchmark;
- propose an update to `decision_register.md`.

Confidence alone is not evidence.

---

## 10. KG work

Before graph implementation, agents must read `kg_phase0.md`.

They must not select Neo4j, RDF, ArangoDB, GraphRAG, etc. as the architecture merely because they are familiar tools.

The first questions are:

```text
What is an entity?
What is its immutable ID?
What is an assertion?
Which source/version supports the assertion?
When is it valid?
How are conflicts represented?
Which graph queries must outperform text RAG?
```

---

## 11. Serving work

Preserve direct Transformers as a baseline.

Benchmark alternatives under the same workload.

Do not report capacity from theoretical VRAM alone.

---

## 12. Handoff over chat memory

The repository must contain enough context for a new agent to continue if all prior chat history disappears.
