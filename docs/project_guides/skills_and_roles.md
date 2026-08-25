# Skills and Roles

The project now requires more than RAG/model integration. The next phase deliberately strengthens architecture, evaluation, provenance, and serving skills.

---

## 1. Software architecture

Use for:

- application/service boundary;
- capability interfaces;
- typed contracts;
- dependency direction;
- state ownership;
- orchestration;
- future graph integration.

Key concepts:

- separation of concerns;
- dependency inversion;
- domain modeling;
- explicit state;
- service boundaries.

---

## 2. Python backend engineering

Use for:

- FastAPI;
- Pydantic/dataclasses;
- async streaming;
- concurrency;
- persistence;
- tests;
- dependency injection.

---

## 3. Information retrieval

Maintain and evaluate:

- BM25;
- dense retrieval;
- FAISS;
- RRF;
- cross-encoder reranking;
- structured/entity lookup;
- metadata filters;
- future logical capability collections.

Metrics:

- Recall@K;
- Precision@K;
- MRR;
- nDCG.

---

## 4. Evidence calibration / answerability

New priority skill.

Build and evaluate an evidence gate using signals such as:

- reranker score;
- score margins;
- sparse/dense agreement;
- source authority;
- support count;
- router probability;
- optional entailment/reformulation consistency.

Learn calibration methods such as:

- logistic regression;
- Platt scaling;
- isotonic regression.

Do not call a score “confidence” without held-out calibration evidence.

---

## 5. Data engineering and provenance

Critical near-term skill.

Use for:

- Source Registry;
- DocumentVersion;
- chunk provenance;
- content hashes;
- update detection;
- immutable snapshots;
- source authority;
- conflicting versions;
- access classification.

Conceptually study W3C PROV-O semantics where useful.

---

## 6. Evaluation engineering

Build:

- AUM Gold Set;
- retrieval labels;
- expected sources/evidence;
- routing labels;
- answerability labels;
- grounding/citation labels;
- multi-turn cases;
- regression harness.

Know the difference between:

- software tests;
- retrieval metrics;
- automated LLM judges;
- human-reviewed control sets.

---

## 7. LLM engineering

Use for:

- grounded explanation;
- synthesis;
- structured output;
- limited routing/decomposition when justified.

Skills:

- prompt contracts;
- token budgeting;
- streaming;
- hallucination controls;
- model-server abstraction.

---

## 8. Conversation/state engineering

Design:

- recent dialogue;
- active entity IDs;
- active program/catalog year;
- previous result sets;
- unresolved ambiguity;
- evidence references.

Avoid relying on raw full-history injection.

---

## 9. Entity/ontology modeling

Required before KG implementation.

Skills:

- canonical IDs;
- aliases;
- entity resolution;
- relation vocabulary;
- domain/range constraints;
- temporal assertions;
- source-level provenance.

---

## 10. Knowledge Graph engineering

Later-stage skill.

Use for relational/multi-hop query classes.

Skills eventually include:

- graph modeling;
- graph query languages;
- bounded traversal;
- hybrid graph/vector retrieval;
- graph evaluation.

Tool/database selection comes after requirements.

---

## 11. GPU / inference serving

Current local hardware makes this important.

Skills:

- VRAM/KV cache reasoning;
- concurrency;
- TTFT;
- batching;
- vLLM benchmarking;
- direct Transformers baseline;
- quantization tradeoffs;
- model replicas;
- cancellation/backpressure.

---

## 12. Security/privacy

Before restricted university data:

- authentication;
- authorization-before-retrieval;
- data classification;
- PII/FERPA awareness;
- trace/log retention;
- prompt injection defense;
- source permissions;
- secret management.

---

## 13. Front-end/client engineering

Current priorities:

- consume structured sources;
- real streaming;
- error/cancellation states;
- trace IDs for debugging;
- avoid UI logic becoming business logic.

UI polish is secondary while core contracts are changing.

---

## 14. DevOps/reproducibility

Build toward:

- environment reproducibility;
- CI;
- tests/evals;
- benchmark scripts;
- artifact packaging;
- monitoring;
- backups;
- deployment profiles.

---

## 15. Research skill

Use deep research only for concrete decisions:

- answerability/calibration;
- routing;
- serving;
- KG retrieval;
- security;
- evaluation.

Do not chase newer frameworks without a measured project need.
