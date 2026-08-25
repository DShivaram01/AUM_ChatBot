# Roadmap

This roadmap is aligned with the 2026-08-24 deep-research architecture reference.

The order is intended to reduce irreversible rework.

---

# Phase 0 — Measurable, provenance-first foundation

**Priority: now**

Build in parallel:

### Measurement

- `pytest` baseline;
- AUM Gold Set v1;
- regression cases;
- current retrieval/routing/latency baseline.

### Provenance

- Source Registry;
- DocumentVersion;
- chunk provenance;
- content hashes;
- source authority/access classification;
- immutable index snapshots.

### Contracts

- typed Source/Evidence/Response schemas;
- expanded trace schema;
- formal capability/retrieval interface;
- extract core assistant service from Gradio.

### Serving boundary

- model-server abstraction;
- preserve direct Transformers baseline.

**Exit criteria**

We can identify exactly:

- which source/version supported an answer;
- which index snapshot was used;
- which capability ran;
- whether the change improved/degraded Gold Set metrics;
- how the same request performs over time.

---

# Phase 1 — Evidence-aware multi-capability assistant

**Priority: next**

Build:

- logical capability/domain collections;
- initial capability set;
- multi-label routing baseline;
- evidence sufficiency / abstention;
- structured sources in API/client;
- source conflict handling;
- structured conversation state.

Initial capability direction:

```text
Research
People
Policy
Courses / Programs
General AUM
Out of Scope
```

**Exit criteria**

The assistant reliably distinguishes:

```text
structured exact lookup
single-capability RAG
multi-capability RAG
unanswerable
clarification-needed
conflicting-source case
```

---

# Phase 2 — Local serving benchmark and deployment preparation

In parallel with Phase 1 when hardware is available:

- collect exact lab hardware inventory;
- benchmark direct Transformers vs vLLM on one RTX 4090;
- determine realistic concurrency;
- decide whether embedding/reranking stay on CPU or GPU;
- measure client streaming/TTFT;
- establish monitoring fields.

Do not claim supported concurrency until measured.

---

# Phase 3 — KG Phase 0

No graph database selection yet.

Define:

- canonical entity model;
- immutable IDs;
- entity resolution;
- Source Registry integration;
- assertion/provenance model;
- temporal validity;
- relation vocabulary;
- graph constraints;
- authority/conflict policy;
- 20–30 graph evaluation questions;
- tiny hand-verified sample graph.

**Exit criterion**

The team can represent one institutional fact with:

```text
identity
relation/value
source
document version
evidence location
validity
```

without ambiguity.

---

# Phase 4 — College of Sciences research vertical

Build the first useful graph slice:

```text
AUM
→ College of Sciences
→ Departments
→ Faculty
→ Research Projects
```

Candidate relations:

```text
MEMBER_OF
RESEARCHES
WORKS_ON
MENTORS
PRESENTED_AT
```

Compare graph-assisted results with existing hybrid RAG.

---

# Phase 5 — Academic program graph

Add:

- degree/program;
- major;
- course;
- requirement;
- prerequisite;
- catalog-year semantics.

---

# Phase 6 — Research ecosystem

Add:

- research areas;
- publications;
- facilities;
- events;
- symposium presentations;
- student research.

---

# Phase 7 — Wider AUM expansion

Only after earlier entity/version/update rules survive real changes:

- remaining colleges;
- interdisciplinary programs;
- cross-college relationships;
- institution-wide graph navigation.

---

# Phase 8 — Scale-out only when measured demand requires it

Preferred progression:

```text
one RTX 4090
→ separate app/retrieval host + one inference host
→ independent model replicas if needed
```

Defer:

- 21-node cluster;
- Kubernetes;
- tightly coupled multi-machine tensor parallelism.

---

# Ongoing rule

Do not add architecture because it is impressive.

Add it because a measured AUM failure mode requires it.
