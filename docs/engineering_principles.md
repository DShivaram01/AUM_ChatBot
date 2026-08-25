# Engineering Principles

These principles are project invariants unless a documented architecture decision explicitly changes them.

---

## 1. Evidence before assumptions

Use:

```text
symptom
→ reproduce
→ instrument
→ hypothesis
→ isolate
→ benchmark/test
→ root cause
→ fix
→ regression case
```

Do not optimize or redesign based on plausible-sounding explanations.

---

## 2. Evaluation precedes architectural escalation

Before replacing retrieval components, serving engines, models, routers, or adding graph machinery:

- establish baseline behavior;
- define failure mode;
- measure;
- compare;
- keep the new component only if it solves a measured problem.

---

## 3. Sources establish institutional truth

The LLM is not:

- the university database;
- the source authority;
- the Knowledge Graph;
- the source of course codes, deadlines, or policy facts.

The LLM primarily interprets and synthesizes evidence.

---

## 4. Provenance is first-class data

Institutional facts must eventually be traceable to:

```text
source_id
document_version_id
evidence location
effective/observed time
authority tier
```

Do not treat citations as presentation-only strings.

---

## 5. Freshness/versioning is part of correctness

An answer retrieved from an old catalog can be fluent and still be wrong.

Do not overwrite the only index in place.

Build candidate snapshots, validate them, then promote them.

---

## 6. Authorization belongs before retrieval

For protected data:

```text
authenticate
→ authorize source access
→ retrieve
→ construct context
→ generate
```

Never retrieve unauthorized content and try to remove it after generation.

---

## 7. Retrieval existence is not evidence sufficiency

Vector search always has nearest neighbors.

Nearest ≠ relevant.

Every answer path must eventually support:

```text
SUPPORTED
PARTIALLY_SUPPORTED
UNSUPPORTED
AMBIGUOUS / CONFLICTING
```

---

## 8. Capability-oriented architecture

Do not encode the final product as:

```text
if cos
elif housing
elif faculty
elif courses
...
```

Capabilities should be discoverable through explicit interfaces and may be selected in combination.

---

## 9. Logical modularity before physical distribution

First define clean service boundaries in code.

Only split them across machines when:

- performance;
- reliability;
- security;
- operations;

justify the complexity.

---

## 10. UI depends on application logic

Desired dependency direction:

```text
Gradio / FastAPI / Electron
            ↓
     Application Service
            ↓
Retrieval / State / Models / Data
```

Business logic must not belong to Gradio.

---

## 11. Prefer explicit state and dependencies

Reduce global mutable state.

Prefer explicit objects such as:

```text
ApplicationContext
AssistantService
CapabilityRegistry
ConversationStore
SourceRegistry
TraceStore
ModelService
```

---

## 12. Use typed schemas at boundaries

Use dataclasses/Pydantic models for:

- source;
- document version;
- chunk/evidence;
- retrieval result;
- route/capability decision;
- assistant request/response;
- conversation state;
- citations;
- trace events.

Avoid free-form dictionaries between architectural layers.

---

## 13. One module, one primary responsibility

Separate:

- ingestion;
- index lifecycle;
- runtime retrieval;
- routing;
- evidence qualification;
- generation;
- UI/API;
- persistence.

Do not allow a new monolith to replace the old monolith.

---

## 14. Structured facts remain structured

Prefer:

```text
structured evidence → deterministic output
retrieved prose → grounded synthesis
```

Do not ask the LLM to regenerate exact known metadata.

---

## 15. Graphs are for relations

Use future KG traversal for questions where typed relationships/multi-hop structure add measurable value.

Keep ordinary RAG for prose-heavy policy/document explanation.

Keep exact lookup for simple structured facts.

Do not route every query through the graph.

---

## 16. Names are not identifiers

Canonical entity IDs must be immutable and independent of display labels.

Aliases, titles, and names can change.

This is mandatory before serious KG work.

---

## 17. Institutional relations can be temporal

Examples:

- prerequisites;
- faculty membership;
- chair/dean roles;
- course offerings;
- program requirements.

Do not silently model changing relations as timeless facts.

---

## 18. Conflicts are data

If two authoritative sources disagree:

- preserve both assertions;
- preserve provenance/version;
- apply a documented authority policy or surface the conflict.

Do not erase disagreement during ingestion.

---

## 19. Observability is a feature

A difficult answer should be explainable by trace:

```text
route
retrieval
scores
source/index versions
evidence-gate decision
generation
citations
latencies
abstention reason
```

Tracing must become privacy-aware when restricted sources appear.

---

## 20. No hidden import-time side effects

Do not unexpectedly:

- load models;
- create runtime directories;
- mutate logging globally;
- run tests;
- download assets;

simply by importing a module.

Use explicit bootstrap/startup.

---

## 21. Configuration is environment-aware data

Prefer:

- `pathlib`;
- project-relative defaults;
- environment overrides;
- explicit runtime profiles.

Avoid hard-coded workstation paths when practical.

---

## 22. Tests are required for completion

A change must be covered at the appropriate levels:

- static checks;
- unit tests;
- integration tests;
- AI-system evaluation;
- runtime smoke test.

---

## 23. Gold Set metrics protect AI behavior

Track subsystem changes using project-specific questions.

Do not rely only on an LLM judge or “the answer looked good.”

---

## 24. Performance claims require comparable benchmarks

Record:

- workload;
- environment;
- configuration;
- latency;
- TTFT;
- throughput;
- VRAM;
- concurrency;
- errors.

A faster isolated token benchmark is not automatically a faster real application.

---

## 25. Direct Transformers is a baseline, not a permanent serving contract

Application code should depend on a model-serving abstraction.

Benchmark production-style serving such as vLLM before adopting it.

---

## 26. Quantization must solve a measured deployment problem

Do not quantize merely because it reduces memory.

Evaluate quality, throughput, TTFT, VRAM, and operational complexity on the same AUM Gold Set/workload.

---

## 27. AI agents are contributors, not authorities

Claude Code and Codex must:

- inspect live files;
- obey accepted decisions;
- show assumptions;
- run tests;
- report unresolved risks;
- avoid fabricating legacy behavior;
- use evidence/benchmarks for architectural claims.

---

## 28. Preserve historical decisions without freezing the architecture

Research snapshots and decision records preserve reasoning.

When evidence changes:

```text
old recommendation
new recommendation
date
reason
new evidence
benchmark impact
migration impact
```

Do not silently rewrite history.

---

## 29. Reversible, reviewable changes

Prefer small, purpose-specific commits that can be:

- inspected;
- tested;
- reverted;
- cherry-picked.

---

## 30. Architecture serves the product

Add abstraction only when it improves:

- correctness;
- extension;
- testing;
- provenance;
- security;
- debuggability;
- deployment;
- understanding.
