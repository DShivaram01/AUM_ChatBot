# Architecture Decision Register

**Snapshot date:** 2026-08-24

This file records currently accepted project-level decisions. It is intentionally concise. Detailed evidence lives in `AUM_DEEP_RESEARCH_ARCHITECTURE_2026-08-24.md`.

Decision labels:

- KEEP
- BUILD NOW
- BUILD NEXT
- BENCHMARK FIRST
- DEFER
- AVOID

---

## D001 — Preserve hybrid retrieval baseline

**Decision:** KEEP

Current baseline:

```text
BM25 + dense embeddings
→ RRF
→ CrossEncoder reranking
```

Do not replace for novelty.

---

## D002 — Deterministic structured facts

**Decision:** KEEP AND EXPAND

Exact institutional facts should come from structured evidence when possible. LLM performs explanation/synthesis.

---

## D003 — Source Registry and provenance

**Decision:** BUILD NOW

Every source/document/chunk requires stable identity and provenance before broad corpus expansion.

---

## D004 — Versioned immutable source/index lifecycle

**Decision:** BUILD NOW

Do not destructively overwrite the only authoritative document/index state.

---

## D005 — AUM Gold Set and subsystem evaluation

**Decision:** BUILD NOW

Architectural changes must become measurable.

---

## D006 — Capability-oriented logical collections

**Decision:** BUILD NOW

Avoid both one final global vector store and disconnected domain-specific chatbots.

---

## D007 — Multi-label routing

**Decision:** BUILD NOW

Questions may require multiple capabilities. Do not force one intent label.

---

## D008 — Evidence sufficiency / abstention

**Decision:** BUILD NOW

Similarity and RRF scores are not calibrated confidence.

---

## D009 — Structured conversation state

**Decision:** BUILD NEXT

Use recent dialogue + entity/topic/evidence state. Avoid full-history injection as the core memory design.

---

## D010 — Model-serving abstraction

**Decision:** BUILD NOW

Application code should be independent of direct `transformers.generate()`.

---

## D011 — vLLM

**Decision:** BENCHMARK FIRST

Primary production-style serving candidate against direct Transformers on one RTX 4090.

Do not adopt solely from published throughput claims.

---

## D012 — TGI

**Decision:** AVOID AS NEW DEFAULT

Deep research found current Hugging Face documentation marking TGI maintenance mode. Re-verify if reconsidered.

---

## D013 — llama.cpp / quantization

**Decision:** DEFER / BENCHMARK IF NEEDED

Use only if portability, VRAM, longer context, or concurrency requirements justify it.

---

## D014 — One RTX 4090 first deployment benchmark

**Decision:** BUILD / BENCHMARK FIRST

Start operational learning on one GPU before designing a cluster.

---

## D015 — Multi-GPU scale-out

**Decision:** DEFER

If needed, prefer independent model replicas for the current 7B model before tightly coupled multi-machine tensor parallelism.

---

## D016 — Knowledge Graph

**Decision:** BUILD NEXT, AFTER KG PHASE 0

KG is for relational/multi-hop query classes. It does not replace ordinary RAG.

---

## D017 — Graph database/framework selection

**Decision:** DEFER

Do not select before canonical identity, provenance, temporal model, relation semantics, and graph evaluation questions exist.

---

## D018 — Microsoft GraphRAG as default KG

**Decision:** AVOID FOR NOW

Its global corpus-sensemaking objective is not the same as the canonical institutional ontology problem.

---

## D019 — Authorization before retrieval

**Decision:** BUILD BEFORE PROTECTED DATA

Unauthorized content must never enter prompt context.

---

## D020 — Student education records

**Decision:** AVOID / OUT OF CURRENT SCOPE

Do not ingest until AUM governance, authorization, FERPA handling, and retention policies are explicit.

---

## How to change a decision

Add a dated amendment containing:

```text
old decision
new decision
date
reason
new evidence/source IDs
benchmark impact
migration impact
```

Do not silently edit away the previous rationale.
