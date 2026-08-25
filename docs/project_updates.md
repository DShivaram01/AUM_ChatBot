# Project Updates

**Project:** AUM Academic Assistant  
**Status date:** 2026-08-24  
**Current phase:** Provenance-first architecture stabilization  
**Research reference:** `AUM_DEEP_RESEARCH_ARCHITECTURE_2026-08-24.md`

---

## 1. Current system

The working prototype currently includes:

- local `Mistral-7B-Instruct-v0.3`;
- MiniLM embeddings;
- FAISS dense retrieval;
- BM25 sparse retrieval;
- Reciprocal Rank Fusion;
- CrossEncoder reranking;
- deterministic rendering of important COS metadata;
- grounded LLM explanation/summarization;
- COS symposium data;
- AUM Housing policy data;
- Gradio development UI;
- FastAPI backend;
- Electron client;
- streaming generation paths;
- query tracing / inspector;
- embedding/index caching;
- corpus-aware name matching;
- numbered-result follow-up flow;
- LLM-based COS/Housing routing with heuristic fallback;
- process-wide generation serialization;
- flash-attention investigation and benchmark.

The retrieval core is directionally strong. The primary risk is no longer “can we make RAG work?” but “can we evolve it into a trustworthy institutional information system without creating uncontrolled coupling, stale data, unmeasured regressions, or unsupported answers?”

---

## 2. What we keep

### Hybrid retrieval

Keep:

```text
BM25 + dense retrieval
        ↓
       RRF
        ↓
CrossEncoder reranking
```

Do not replace FAISS/BM25/RRF/reranking for novelty. Replace components only when AUM-specific evaluation shows measurable benefit.

### Deterministic structured facts

Continue inserting structured institutional facts directly from evidence when possible:

- names;
- course codes;
- department/program names;
- years;
- dates;
- policy identifiers;
- prerequisites;
- source/version identifiers.

Use the LLM for explanation and synthesis, not as the institutional database.

### Tracing

Keep and expand tracing to include:

- routing decisions;
- source/version IDs;
- retrieval candidates/ranks;
- fusion and reranker scores;
- evidence-gate features and decision;
- latency by stage;
- model-serving metadata;
- citations;
- abstention reason.

---

## 3. Current resolved engineering lessons

### GPU generation concurrency

Concurrent `generate()` calls on the same model instance created serious stalls and VRAM pressure. The process-wide generation lock fixed the immediate prototype problem.

Long-term lesson: direct serialized generation is a baseline, not the final multi-user serving architecture.

### Flash-attention

The original latency hypothesis was disproven by benchmark. PyTorch SDPA was already active and `flash_attention_2` did not materially improve the measured AUM workload.

Long-term lesson:

> Benchmark the real workload before changing architecture around a performance hypothesis.

### Modularization

The 2,227-line flat backend was successfully split into modules. That migration phase is now complete.

From this point forward, do not preserve legacy `backend.py` structure merely for compatibility when a better interface is justified and protected by tests.

---

## 4. Architecture corrections now considered BUILD NOW

Deep research changes the immediate priority stack.

### 4.1 Source Registry

Every authoritative source needs a stable identity and metadata such as:

```text
source_id
canonical_url
authority_tier
source_type
domain
title
owner
effective dates
catalog/policy version
retrieved_at
content_hash
access_classification
```

### 4.2 Document/version model

The system must distinguish a source from a particular version of that source.

This is required because AUM catalogs, policies, department structures, and pages change over time.

### 4.3 Chunk provenance

Every retrievable chunk should preserve:

```text
chunk_id
source_id
document_version_id
heading_path
page / section / offset
content_hash
extraction timestamp
```

### 4.4 Versioned index snapshots

Do not destructively rebuild the only production index.

Use:

```text
new source/document version
→ candidate index snapshot
→ validation/evals
→ atomic promotion
```

### 4.5 AUM Gold Set

Create a curated evaluation dataset before major retrieval/routing changes.

Initial target from the research: approximately 250–400 queries across current and near-term capabilities.

### 4.6 Formal capability interface

The assistant must evolve from:

```text
COS | Housing
```

toward logical capabilities such as:

```text
Research
People
Policy
Courses / Programs
General AUM
Out of Scope
```

Questions may select more than one capability.

### 4.7 Evidence sufficiency / abstention

Nearest-neighbor retrieval is not proof of relevance.

Introduce an explicit evidence gate with labels such as:

```text
SUPPORTED
PARTIALLY_SUPPORTED
UNSUPPORTED
AMBIGUOUS / CONFLICTING
```

### 4.8 Model-serving abstraction

Application logic should not care whether generation comes from:

- direct Transformers;
- vLLM;
- another local serving engine later.

Direct Transformers remains the correctness baseline.

### 4.9 Structured sources in API responses

Sources must become first-class response data, not text decorations.

---

## 5. Current architecture red flags

### UI/business coupling

FastAPI should not depend on Gradio chat logic.

Target:

```text
              AssistantService
               /           \
            FastAPI       Gradio
              |
           Electron
```

### Excess global state

Gradually replace:

- `_loaded`;
- global indexes;
- global traces;
- query counter;
- pending-session dictionaries;
- UI-injected model globals;

with explicit application/service/state objects.

### Retrieval module overload

Current retrieval code combines ingestion, cache management, PDF parsing, name resolution, sparse/dense search, RRF, and reranking.

Split ingestion/index lifecycle from runtime retrieval as new capabilities are added.

### Housing answerability

Housing can retrieve nearest chunks for unrelated questions. Fix through the generic evidence gate rather than a one-off patch.

### Inconsistent cache versioning

COS currently has stronger manifest validation than Housing. Standardize this before new corpora are added.

### Machine-specific configuration

Move toward project-relative `pathlib` defaults + environment overrides + explicit bootstrap side effects.

---

## 6. Current P0/P1/P2 work queue

### P0 — establish trustworthy foundations

- Source Registry schema;
- DocumentVersion schema;
- chunk provenance schema;
- immutable index-version naming and promotion;
- AUM Gold Set v1;
- trace schema expansion;
- formal retrieval/capability interface;
- extract application business logic from Gradio;
- typed response/evidence/source schemas;
- structured sources in API output;
- software unit/integration test baseline.

### P1 — orchestration and reliability

- logical per-capability collections;
- multi-label routing baseline;
- evidence-gate prototype;
- structured conversation state;
- source-authority/conflict handling;
- model-server abstraction;
- vLLM benchmark harness;
- hardware inventory;
- client streaming cleanup.

### P2 — KG foundation and first graph experiment

- KG Phase 0 ontology/ID/provenance document;
- graph-specific evaluation questions;
- tiny hand-verified sample graph;
- then a College of Sciences research vertical;
- graph-vs-text evaluation.

Do not select the graph database merely because P2 exists.

---

## 7. Deployment direction

The intended university environment includes approximately 21 RTX 4090-equipped lab desktops, but that does not imply a 21-GPU cluster should be built.

Recommended progression:

1. benchmark one RTX 4090;
2. compare current direct Transformers serving vs vLLM;
3. measure real concurrency;
4. if needed, separate application/retrieval and inference hosts;
5. if demand requires more capacity, consider independent single-GPU model replicas;
6. defer multi-machine cluster engineering until operational requirements justify it.

Unknown hardware details must be inventoried rather than guessed.

---

## 8. Knowledge Graph status

The KG vision remains:

> AUM-centered institutional hierarchy for navigation + typed cross-domain relations for reasoning, with College of Sciences as the initial deep-focus branch.

But the graph must not be built as a timeless tree of names.

Before implementation we require:

- canonical IDs;
- source/version provenance;
- temporal validity;
- entity resolution;
- authority/conflict policy;
- relation semantics;
- graph constraints;
- graph-specific evaluation questions.

See `kg_phase0.md`.

---

## 9. Immediate next recommended task sequence

A practical sequence from the current codebase is:

1. establish `pytest` + first regression/eval fixtures;
2. define shared typed schemas (`Source`, `DocumentVersion`, `Evidence`, `AssistantResponse`, routing decision);
3. extract core assistant execution from Gradio into an application service;
4. add Source Registry + versioned document/chunk provenance;
5. expand traces to carry source/index version IDs;
6. add structured sources to API responses;
7. implement generic evidence sufficiency behavior and fix Housing through it;
8. build Gold Set v1 and baseline current performance;
9. introduce capability interfaces and multi-label routing;
10. add structured conversation state;
11. build model-server abstraction and benchmark direct Transformers vs vLLM;
12. begin KG Phase 0 documentation/evaluation design.

---

## 10. Rule for every future task

A task is not complete because the code runs.

It must answer:

1. What user or engineering problem is solved?
2. Which architectural layer owns the behavior?
3. Which source/version contracts are affected?
4. What behavior must remain unchanged?
5. What tests prove correctness?
6. What Gold Set/eval cases prove AI-system behavior?
7. What trace should make failures observable?
8. What migration/backward-compatibility risk exists?
9. What technical debt is added?
10. Does this move the project toward the accepted architecture?
