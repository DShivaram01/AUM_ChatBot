# Deep Research Prompt — Reuse / Refresh Guidance

**Note:** A major architecture deep-research pass was completed on 2026-08-24. Its results are preserved in `AUM_DEEP_RESEARCH_ARCHITECTURE_2026-08-24.md`.

Use the prompt below when:
- a major decision is still unresolved;
- time-sensitive framework/hardware/security guidance needs re-verification;
- new project constraints materially change the architecture;
- a future research pass is needed.

Do not rerun it merely to re-collect already settled information.

---

# Deep Research Prompt — AUM Academic Assistant Architecture

Use this prompt when outside research is needed to support major design choices.

The purpose is **not** to collect fashionable tools. The purpose is to find evidence that helps decide how to build the next stage of the AUM Academic Assistant.

---

## Prompt

I am building a privacy-conscious, grounded university academic assistant for Auburn University at Montgomery.

The current prototype has:

- a local Mistral-7B-Instruct model;
- MiniLM sentence embeddings;
- FAISS;
- BM25;
- Reciprocal Rank Fusion;
- CrossEncoder reranking;
- deterministic structured metadata rendering;
- grounded LLM summarization;
- a Gradio UI;
- FastAPI;
- an Electron client;
- query tracing;
- two current knowledge domains: undergraduate research symposium data and Housing policy.

The project is evolving from a two-domain RAG prototype into a unified academic assistant with future capabilities including:

- research discovery;
- faculty/people;
- courses;
- university policies;
- general institutional information;
- document QA;
- multi-turn conversational follow-ups;
- later, a large institutional Knowledge Graph for relationship-heavy questions.

## Deployment resources and physical constraints

Design recommendations must be realistic for the hardware the university already has or plans to use.

### Planned campus deployment

The intended deployment environment is the AUM Machine Learning / Deep Learning Lab.

Known infrastructure:

- approximately **21 lab desktops equipped with NVIDIA RTX 4090 GPUs**;
- the assistant is intended to run on university-controlled/local infrastructure rather than depending on a commercial hosted LLM API for its core intelligence;
- the current system already runs a local `mistralai/Mistral-7B-Instruct-v0.3` model with CUDA when available;
- the current prototype has already demonstrated that GPU concurrency, VRAM pressure, model loading, streaming, and inference serialization materially affect system behavior.

The exact CPU model, system RAM per workstation, operating-system version, local storage capacity, networking topology, and whether one machine or several machines will act as persistent servers are **not yet documented**.

Treat those missing specifications as an explicit design uncertainty rather than inventing values.

When analyzing deployment architecture, research at least these possibilities:

1. one dedicated RTX 4090 workstation serving the entire chatbot;
2. one primary inference machine plus separate retrieval/application services;
3. multiple lab workstations used as an inference pool;
4. one model server with many thin clients;
5. whether vLLM/TGI/direct Hugging Face serving is appropriate for this scale;
6. whether quantization is beneficial or unnecessary on a 24-GB RTX 4090;
7. realistic concurrent-user capacity, queueing behavior, time-to-first-token, throughput, and VRAM requirements;
8. whether embedding/reranking workloads should share the LLM GPU or execute separately;
9. persistence requirements for FAISS/index data, structured data, sessions, traces, and future graph storage;
10. what minimum CPU RAM, storage, networking, and OS configuration should be recommended once the exact lab inventory is collected.

Clearly separate recommendations that are possible with **one RTX 4090** from recommendations that require coordinating multiple machines.

### Current development context

The current prototype is already being exercised on a local CUDA-capable development machine and has shown approximately 20-GB-class GPU-memory constraints during Mistral testing. That environment should be treated as a development reference point, not automatically assumed to be identical to the final lab deployment.

Do not begin by recommending a complete new architecture.

Instead, conduct a source-driven investigation into the following engineering questions.

### 1. RAG architecture

Research modern best practices for modular RAG systems that support multiple heterogeneous knowledge sources.

Compare:

- single shared index vs per-domain indexes;
- capability-based routing;
- retrieval orchestration;
- query decomposition;
- sparse+dense hybrid retrieval;
- reranking;
- structured lookup alongside vector search.

Focus on systems that need strong provenance and low hallucination.

### 2. Retrieval confidence and abstention

Investigate methods for deciding when retrieved evidence is actually relevant.

Compare:

- absolute similarity thresholds;
- relative score thresholds;
- reranker calibration;
- top-1/top-2 margins;
- query-document entailment;
- answerability classifiers;
- retrieval consistency;
- conformal or calibrated confidence approaches where relevant.

Identify methods realistic for a small university project.

### 3. RAG evaluation

Find authoritative approaches and current tools for evaluating:

- retrieval Recall@K;
- MRR/nDCG;
- answer correctness;
- faithfulness/grounding;
- citation correctness;
- abstention;
- routing;
- conversational follow-ups.

Distinguish research metrics from practical regression testing.

### 4. Multi-domain routing

Research approaches for routing queries across multiple capabilities.

Compare:

- rule systems;
- embedding classifiers;
- small supervised classifiers;
- LLM routers;
- tool/capability selection;
- multi-label routing.

Pay special attention to questions that legitimately require more than one source/capability.

### 5. Conversation memory

Research architectures for reliable multi-turn RAG memory.

Compare:

- raw history injection;
- conversation summaries;
- structured entity state;
- retrieved-result state;
- episodic memory;
- external stores.

Focus on avoiding context pollution and retrieval corruption.

### 6. Local model serving

Research practical local inference architectures for one GPU and low-to-moderate concurrent usage.

Compare, where relevant:

- direct Hugging Face `generate`;
- serialized generation queues;
- vLLM;
- Text Generation Inference;
- llama.cpp or quantized alternatives where applicable.

Discuss:

- VRAM;
- batching;
- streaming;
- cancellation;
- concurrent users;
- throughput;
- operational complexity.

### 7. Data ingestion and provenance

Research best practices for maintaining changing institutional sources.

Include:

- source IDs;
- canonical URLs;
- document versions;
- effective dates;
- content hashes;
- update detection;
- chunk provenance;
- stale-data prevention;
- conflicting versions.

### 8. Knowledge Graph prerequisites and intended institutional shape

Do **not** choose the final graph database, GraphRAG framework, or implementation yet.

First investigate the prerequisites for the Knowledge Graph while taking the intended conceptual structure seriously.

The working vision is an **AUM-centered hierarchical tree + graph**.

At the highest level:

```text
Auburn University at Montgomery (AUM)
│
├── College of Sciences
├── College of Business
├── College of Education
├── College of Liberal Arts and Social Sciences
└── other institutional units
```

The initial deep-focus branch will be the **College of Sciences**.

A simplified conceptual expansion could look like:

```text
AUM
└── College of Sciences
    ├── Departments / academic units
    │   ├── Computer Science
    │   ├── Biology
    │   ├── Chemistry
    │   ├── Mathematics
    │   └── ...
    │
    ├── Majors
    ├── Degrees
    ├── Courses
    ├── Faculty
    ├── Research areas
    ├── Research projects
    ├── Publications
    ├── Specialized research facilities / laboratories
    ├── Student research
    └── Events / symposium activity
```

However, this must **not** remain a strict tree.

Cross-links turn the hierarchy into a true graph. Examples:

```text
Faculty ──TEACHES────────────→ Course
Faculty ──MENTORS────────────→ Student / Project
Faculty ──MEMBER_OF──────────→ Department
Faculty ──RESEARCHES─────────→ Research Area
Project ──USES───────────────→ Research Facility
Project ──RELATED_TO─────────→ Research Area
Project ──PRESENTED_AT───────→ Symposium / Event
Project ──PRODUCED───────────→ Publication
Course ──REQUIRED_FOR────────→ Degree / Major
Course ──PREREQUISITE_OF────→ Course
Degree ──OFFERED_BY──────────→ Department / College
Research Facility ──BELONGS_TO→ Department / College
Publication ──AUTHORED_BY────→ Faculty / Student
```

The intended result is therefore:

> **institutional hierarchy for navigation + cross-domain relations for reasoning.**

Research should determine how best to model this without forcing every relationship into a parent/child tree.

Before implementation, investigate and recommend:

- canonical entity schema;
- canonical IDs;
- hierarchical relationships versus non-hierarchical relationships;
- entity resolution;
- provenance at node and edge level;
- relation types and relation direction;
- temporal/versioned relationships;
- data ownership and authoritative sources;
- deduplication;
- graph constraints;
- graph evaluation questions;
- graph update strategy;
- graph/vector/keyword hybrid retrieval;
- when graph traversal is superior to ordinary RAG;
- when semantic retrieval should locate entry nodes before graph traversal;
- how citations/provenance should survive graph reasoning;
- whether the graph should eventually include all AUM colleges or deepen College of Sciences first.

Also investigate a **phased KG build** rather than constructing the entire university graph at once.

A likely phased strategy to evaluate is:

```text
Phase 1:
AUM → College of Sciences → Departments → Faculty → Research Projects

Phase 2:
Majors + Degrees + Courses + prerequisites

Phase 3:
Research areas + publications + facilities + events

Phase 4:
Cross-college relationships and the remaining AUM hierarchy
```

The research should assess this strategy rather than assuming it is correct.

### 9. Security and privacy

For a university assistant that may eventually touch internal documents, research:

- access control;
- authentication;
- authorization;
- PII handling;
- logging;
- prompt/trace retention;
- data isolation;
- prompt injection against RAG sources;
- document-level permissions.

### 10. Recommended staged architecture

After the evidence review, propose a staged path from the current prototype to a maintainable assistant.

For every recommendation provide:

- why it is needed;
- evidence/source;
- complexity;
- risk;
- whether it should be done now, later, or avoided.

---

## Source requirements

Prioritize:

1. primary research papers;
2. official framework documentation;
3. authoritative engineering documentation;
4. benchmark/evaluation papers;
5. well-documented production engineering reports.

Use recent 2024–2026 literature heavily where appropriate, but include important older foundational work.

Avoid low-quality SEO blog summaries unless they point to a primary source.

---

## Deliverable format

Produce:

1. Executive summary.
2. What the current architecture gets right.
3. What is likely to fail as the project expands.
4. Evidence-backed recommendations.
5. A comparison table of alternative approaches.
6. Recommended near-term architecture.
7. What **not** to build yet.
8. Evaluation strategy.
9. Security/privacy checklist.
10. Knowledge Graph prerequisites and a recommended phased institutional ontology based on the AUM → colleges → College of Sciences deep-focus vision.
11. Deployment architecture options constrained by one or more RTX 4090 lab machines, including a hardware inventory checklist for the currently unknown CPU/RAM/storage/network specifications.
12. 30/60/90-day engineering roadmap.
13. Annotated bibliography with direct links/DOIs.
14. Explicit uncertainties or areas where evidence is weak.

Do not recommend technology merely because it is popular.

Optimize for:

- correctness;
- grounding;
- maintainability;
- debuggability;
- reasonable local compute;
- incremental evolution from the existing codebase.
