# Project Scope

## 1. Product vision

Build a grounded, privacy-conscious **AUM Academic Assistant** that helps students, faculty, and researchers navigate authoritative university information through one conversational interface.

The assistant should be able to combine information across multiple AUM knowledge sources without requiring the user to know which database, document, index, graph, or subsystem contains the answer.

---

## 2. Product identity

The product is best understood as:

> **an academic information system with a conversational interface**

rather than:

> a generic LLM with AUM documents in a vector database.

Its intelligence comes from:

```text
authoritative data
+ provenance/versioning
+ routing
+ structured lookup
+ hybrid retrieval
+ entity understanding
+ conversation state
+ evidence sufficiency
+ graph relations where justified
+ grounded synthesis
```

---

## 3. Intended user experience

Examples:

- What research projects has this professor supervised?
- Which projects relate to protein language models?
- Which faculty work in a specific research area?
- Which courses support that research direction?
- Who teaches those courses?
- What does Housing policy say about guests?
- Which catalog year supports this degree requirement?
- Explain this uploaded course document.
- Which source supports this answer?
- Compare these two policy versions.
- Which faculty working in cybersecurity teach courses required by this program?

The user should not manually select a domain for ordinary use.

---

## 4. Capability model

### Implemented today

- COS research/project QA;
- COS person/entity lookup;
- Housing policy QA;
- hybrid sparse+dense retrieval;
- reranking;
- grounded summarization.

### Build now / near term

- Source Registry;
- versioned documents/chunks/indexes;
- structured sources/citations;
- evidence sufficiency / abstention;
- formal capability interfaces;
- multi-label routing;
- General AUM / Out of Scope;
- structured conversation state;
- evaluation framework.

### Future

- faculty/people;
- courses/programs;
- academic catalogs;
- policies;
- research discovery;
- document QA;
- events;
- university facilities;
- cross-source questions;
- institutional Knowledge Graph.

---

## 5. Knowledge organization

Avoid both extremes:

### Not the final design

```text
one global vector database containing everything
```

### Also not the final design

```text
separate disconnected chatbot per university domain
```

Preferred direction:

```text
logical capability/domain collections
        ↓
shared retrieval/orchestration interface
        ↓
controlled cross-capability fusion
```

---

## 6. Grounding policy

### A. Structured authoritative facts

Examples:

- faculty identity;
- course code/title;
- degree/program name;
- event date;
- policy section;
- prerequisite;
- catalog year;
- department membership.

Prefer deterministic rendering from structured evidence.

### B. Evidence-backed synthesis

Examples:

- explaining an abstract;
- comparing programs/projects;
- summarizing policy;
- explaining a cross-source relationship.

The LLM may synthesize these only from authorized retrieved evidence.

### C. Unsupported or conflicting information

If evidence is insufficient:

- abstain;
- ask for clarification;
- provide constrained partial support.

If authoritative sources conflict:

- surface the conflict;
- identify source/version;
- avoid silently choosing one without a documented authority rule.

---

## 7. Provenance and freshness scope

Every important institutional claim should eventually be traceable to:

```text
source
document/version
evidence location
retrieval/index version
effective/observed time where relevant
```

The assistant must be able to distinguish:

- current vs older catalog;
- current vs replaced policy;
- historical vs active department/program relation.

---

## 8. Security boundary

Public and restricted knowledge are different security domains.

Before protected data is introduced:

- authentication must exist;
- authorization must apply before retrieval;
- unauthorized content must never enter model context;
- trace/log policy must be privacy-aware;
- document access classification must be stored with source metadata.

Student-record ingestion is outside current scope until AUM governance and FERPA handling are explicitly defined.

---

## 9. Knowledge Graph scope

The long-term KG is:

```text
AUM
→ academic units
→ departments/programs/courses/faculty/research/facilities/events
+ typed cross-links
```

College of Sciences is the first deep-focus branch.

The KG is **not** the universal retrieval mechanism. Use it where relations and multi-hop structure measurably improve answers.

Do not select a graph database until KG Phase 0 is complete.

---

## 10. Deployment scope

Near-term deployment should assume:

- local/university-controlled infrastructure;
- one RTX 4090 as the first production-style benchmark;
- many thin clients;
- logical modularity before physical distribution.

A 21-machine cluster, Kubernetes, or tightly coupled multi-GPU serving is not current scope.

---

## 11. Success criteria

The system succeeds when it can:

- route to one or more appropriate capabilities;
- retrieve authoritative evidence;
- preserve source/version provenance;
- abstain when evidence is insufficient;
- expose structured citations;
- handle multi-turn entity/result references;
- distinguish structured lookup from text RAG from future graph traversal;
- remain testable and observable;
- evolve without domain-specific branch explosion;
- support local deployment with measured performance;
- introduce new sources without silently serving stale indexes.
