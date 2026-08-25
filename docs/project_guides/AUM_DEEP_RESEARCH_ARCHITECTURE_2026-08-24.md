# AUM Academic Assistant — Deep Research Architecture Reference

**Research snapshot:** 2026-08-24  
**Project:** AUM Academic Assistant  
**Purpose:** Long-lived project-root reference for architectural decisions, future implementation work, evaluation, deployment, security, and Knowledge Graph planning.  
**Status:** Evidence-backed design guidance; not a frozen implementation specification.

> **Important:** This file should be treated as a research snapshot. Re-verify time-sensitive framework documentation, AUM catalog content, hardware availability, and security requirements before major implementation or production deployment decisions.

---

## 0. How to use this file

This document consolidates the deep-research investigation requested for the AUM Academic Assistant. It is intended to be readable by humans and reusable as project context for Codex, Claude Code, or other engineering agents.

Use it to:

- preserve the reasoning behind major architectural choices;
- avoid repeatedly re-researching settled questions;
- distinguish **evidence-backed recommendations** from **engineering judgments** and **open uncertainties**;
- guide near-term implementation without prematurely locking the project into a specific graph database, agent framework, or distributed serving stack;
- provide a source registry for future verification;
- keep the original research prompt attached to the conclusions that came from it.

### Decision labels used in this document

- **KEEP** — already present and directionally correct.
- **BUILD NOW** — should be implemented before broad expansion.
- **BUILD NEXT** — useful after the core evaluation/provenance foundation exists.
- **BENCHMARK FIRST** — cannot be selected responsibly without project-specific measurements.
- **DEFER** — potentially valuable, but premature now.
- **AVOID** — likely to add complexity without proportional value at this stage.

---

# 1. Executive conclusion

The central conclusion of the investigation is:

> **Do not rebuild the AUM assistant. Evolve it into a modular, provenance-first retrieval system with explicit capabilities, calibrated abstention, evaluation, and a clean model-serving boundary. Build the Knowledge Graph only after its schema, identity, provenance, temporality, and graph-specific evaluation questions exist.**

The current retrieval pipeline is directionally strong. A combination of dense retrieval, BM25, Reciprocal Rank Fusion, and cross-encoder reranking is much closer to a serious RAG architecture than a simple vector-search prototype. The next bottlenecks are unlikely to be solved by replacing FAISS or adding a fashionable agent framework.

The most important next problems are:

1. capability and domain separation;
2. retrieval confidence and abstention;
3. provenance, freshness, and versioning;
4. subsystem-level evaluation;
5. multi-domain routing and cross-capability questions;
6. structured multi-turn state;
7. clean separation between application/retrieval services and LLM serving;
8. canonical institutional identity before graph construction;
9. authorization-aware retrieval before any protected university data is introduced.

For deployment, one RTX 4090 is sufficient to host the current Mistral-7B-Instruct-v0.3 model in BF16, but **model fit is not equivalent to concurrent-user capacity**. The RTX 4090 provides 24 GB VRAM, while the official Mistral checkpoint contains about 14.5 GB of BF16 weights. Remaining VRAM must absorb KV cache, runtime buffers, allocator overhead, and any additional GPU-resident workloads. Real concurrency therefore must be measured.

The first production-style serving system to benchmark is **vLLM** against the current direct Hugging Face/Transformers generation path. Hugging Face currently marks Text Generation Inference (TGI) as maintenance mode, so TGI should not be the default choice for a new AUM deployment. `llama.cpp` remains a useful alternative when low-bit quantization, portability, or CPU/GPU hybrid inference becomes desirable.

---

# 2. What the current architecture gets right

## 2.1 Hybrid retrieval — KEEP

The current combination of:

- MiniLM dense embeddings;
- FAISS;
- BM25;
- Reciprocal Rank Fusion (RRF);
- cross-encoder reranking;

is a sound multi-stage retrieval design.

Sparse and dense retrieval fail in complementary ways:

- BM25 is strong for exact terminology, names, course codes, policy phrases, acronyms, and rare tokens.
- dense retrieval is stronger for paraphrase and semantic similarity.
- RRF is appropriate for combining rankings without pretending that BM25 and cosine similarity scores are calibrated onto the same scale.
- a cross-encoder provides a precision-oriented second stage after broad candidate retrieval.

Recommended conceptual flow:

```text
query
  │
  ├── sparse retrieval ──┐
  │                      ├─> candidate fusion / RRF
  └── dense retrieval ───┘
                            │
                            ▼
                    CrossEncoder reranker
                            │
                            ▼
                    Evidence sufficiency gate
                            │
                            ▼
                    Grounded generation
```

**Recommendation:** preserve this architecture as the baseline. Replace individual components only when AUM-specific evaluation shows measurable benefit.

---

## 2.2 Deterministic structured rendering — KEEP AND EXPAND

Information that already exists in structured form should not be regenerated freely by an LLM.

Examples:

- faculty names and titles;
- course codes;
- degree/program names;
- event dates;
- office/contact metadata;
- policy section identifiers;
- catalog years;
- prerequisites;
- department membership.

The LLM should primarily perform:

- interpretation;
- synthesis;
- explanation;
- summarization;
- conversational response generation.

The system should retain deterministic control over exact institutional facts whenever the data model permits it.

---

## 2.3 Query tracing — KEEP AND EXPAND

Tracing is one of the strongest early engineering decisions because future failures will require answers to questions such as:

- Why was this capability selected?
- Why did this source rank first?
- Which index version was used?
- Which reranker scores were observed?
- Why did the system answer instead of abstaining?
- Which source/version supported a generated claim?

A production-quality trace should eventually include:

```text
trace_id
query_id
session_id (if applicable)
router decision + scores
retrieval candidates + ranks
fusion scores
reranker scores
evidence-gate features + decision
source/version IDs
model server request metadata
latency by stage
answer/citation IDs
abstention reason
```

Avoid storing sensitive raw content by default once restricted sources are introduced; trace design must become privacy-aware.

---

## 2.4 Local inference — KEEP

Local/university-controlled model serving is aligned with the project's privacy-conscious design and future protected-data requirements.

Benefits include:

- control over data flow;
- reduced dependence on commercial hosted APIs;
- predictable serving policy;
- compatibility with local authorization and network boundaries;
- control over inference stack and model versions.

Local deployment does not automatically make the system secure. Access control, trace retention, prompt injection defenses, and data isolation remain separate requirements.

---

# 3. What is likely to fail as the project expands

## 3.1 One undifferentiated index — AVOID AS THE FINAL DESIGN

Housing policy, faculty records, research projects, publications, degree requirements, and course prerequisites have different retrieval, authority, update, and permission characteristics.

One global vector collection eventually creates:

- cross-domain false positives;
- difficult score calibration;
- authority conflicts;
- harder incremental updates;
- permission-filtering problems;
- difficult debugging;
- potential recall dilution as the corpus expands.

However, the opposite extreme—completely independent chatbots per domain—is also poor because legitimate questions span multiple domains.

**Recommended direction:** logical capability/domain collections behind a shared retrieval interface, with multi-label routing and controlled cross-capability fusion.

---

## 3.2 Treating similarity scores as confidence — AVOID

A cosine score such as `0.72` is not a probability that the answer is supported. An RRF score is not a confidence score either.

Scores vary with:

- embedding model;
- corpus composition;
- query type;
- chunk size;
- corpus size;
- reranker;
- newly ingested sources;
- domain.

The project needs an explicit **evidence sufficiency / answerability layer** rather than a fixed global threshold.

---

## 3.3 Full chat-history injection — AVOID AS THE MEMORY STRATEGY

Multi-turn RAG benchmarks show that later turns, topic shifts, unanswerable questions, non-standalone references, and multi-domain interactions remain hard.

Simply injecting the entire conversation risks:

- context pollution;
- stale assumptions;
- retrieval corruption;
- repeated irrelevant content;
- higher KV-cache consumption;
- topic leakage from earlier turns.

Treat conversation state as a separate subsystem.

---

## 3.4 Reconstructing institutional relations from prose every time — AVOID

Questions such as:

> Which faculty member working in cybersecurity teaches a course required by this degree?

are relational. Text retrieval can sometimes answer them, but multi-hop and multi-entity queries become increasingly brittle when every relation must be rediscovered from chunks.

This is the eventual justification for an institutional Knowledge Graph.

The KG should not replace normal RAG. It should handle query classes in which explicit relations provide measurable value.

---

## 3.5 Treating the university website as one internally consistent source — AVOID

AUM source content itself illustrates the need for provenance and versioning.

The current AUM catalog's College of Sciences page lists five academic departments:

- Biology & Environmental Sciences;
- Chemistry;
- Computer Science;
- Mathematics;
- Psychology.

The broader AUM academics page also identifies five colleges at the institution level.

University pages can change, disagree, or lag one another. Therefore, the future graph and structured store should preserve **assertions and provenance**, not silently collapse every page into a single timeless truth.

Recommended representation:

```text
Assertion
  subject_id
  predicate
  object_id / value

Provenance
  source_id
  document_version_id
  evidence_location
  observed_at
  effective_from
  effective_to
  authority_tier
  extraction_method
  review_status
```

---

# 4. Recommended near-term architecture

The recommended architecture is **logically modular before physically distributed**.

A single workstation can host many of these components at first while preserving interfaces that permit later separation.

```text
                    Browser / Electron
                           │
                           ▼
                Reverse Proxy / Auth
                           │
                           ▼
                        FastAPI
                           │
                    Query Orchestrator
                ┌──────────┼──────────┐
                │          │          │
                ▼          ▼          ▼
             Router   Conversation   Policy /
                      State Manager   Access Gate
                │
        ┌───────┼───────────────┐
        ▼       ▼               ▼
  Structured   Hybrid RAG     Future KG
    Lookup     Capability     Traversal
                │
      BM25 ─────┼──── Dense
                ▼
               RRF
                ▼
          CrossEncoder
                ▼
          Evidence Gate
             /      \
        insufficient sufficient
            │          │
            ▼          ▼
         abstain   LLM Model Server
                      │
                      ▼
               Citation Renderer
```

Persist independently:

```text
Source Registry
Document/version store
Canonical structured data
BM25/index snapshots
Vector index snapshots
Conversation state
Query/evaluation traces
Future Knowledge Graph
```

This design allows later service separation without prematurely introducing a distributed-systems burden.

---

# 5. Alternative approaches and recommendation matrix

| Problem | Approach | Recommendation for AUM |
|---|---|---|
| Knowledge organization | One global vector index | **Avoid as final design**; easy now but weak for authority, routing, security, and updates. |
| Knowledge organization | Completely isolated indexes/chatbots | Better isolation, but poor for cross-domain questions. |
| Knowledge organization | Logical per-capability collections behind one interface | **Recommended.** |
| Routing | Rules only | Strong high-precision baseline; insufficient alone as domains grow. |
| Routing | Embedding classifier | Cheap and useful baseline. |
| Routing | Small supervised classifier | **Likely eventual default** once labeled AUM routing data exists. |
| Routing | LLM router | Flexible but slower, more expensive, less deterministic. |
| Routing | Rules + learned multi-label router + fallback | **Recommended.** |
| Confidence | Fixed cosine threshold | **Avoid.** |
| Confidence | Reranker threshold | Useful feature, not sufficient alone. |
| Confidence | Calibrated answerability/evidence model using multiple signals | **Recommended.** |
| Memory | Full raw history | Short conversations only. |
| Memory | Running summary only | Cheap but summary errors become persistent state. |
| Memory | Recent turns + structured entity/topic/evidence state | **Recommended.** |
| KG retrieval | Graph for every query | **Avoid.** |
| KG retrieval | Exact/semantic entry-node retrieval + bounded traversal | **Recommended later.** |
| Serving | Direct Transformers `generate()` | Keep as development/correctness baseline. |
| Serving | vLLM | **Primary production-style benchmark.** |
| Serving | TGI | **Do not select for new deployment**; currently maintenance mode. |
| Serving | llama.cpp | Retain as low-bit/portable/hybrid alternative. |
| Multi-GPU | Tensor-parallel 7B across many 4090s | Usually unnecessary because 7B fits one GPU. |
| Multi-GPU | Independent model replicas | Preferred scale-out model if demand requires it. |

---

# 6. Retrieval confidence and abstention

## 6.1 Build an Evidence Gate — BUILD NOW

The evidence gate should estimate whether the retrieved evidence is sufficient for a grounded answer.

Candidate features:

```text
top reranker score
top1 - top2 reranker margin
dense rank
BM25 rank
dense/BM25 agreement
number of independent supporting chunks
source authority tier
retrieved capability count
router probabilities
query/document entailment score (optional)
retrieval consistency under query reformulation (optional)
```

Create an AUM development set labeled approximately as:

```text
SUPPORTED
PARTIALLY_SUPPORTED
UNSUPPORTED
AMBIGUOUS / CONFLICTING
```

Start with a simple calibrated classifier rather than another generative model:

- logistic regression;
- isotonic calibration;
- Platt scaling;
- a small supervised classifier.

Possible policy:

```text
high evidence confidence
    -> answer normally

medium confidence
    -> constrained answer or clarification

low confidence
    -> abstain

conflicting authoritative sources
    -> surface the conflict and provenance
```

Do not claim mathematically calibrated confidence until calibration has been measured on held-out AUM data.

---

# 7. Multi-domain routing

## 7.1 Use multi-label capability selection — BUILD NOW

A query can legitimately require more than one source/capability.

Example:

> Which Computer Science faculty working in AI are involved with projects presented at the symposium?

Possible capability set:

```text
faculty
research
symposium
```

Do not force every request into exactly one intent label.

Recommended staged router:

```text
Stage 1 — deterministic high-precision rules
  course code patterns -> courses
  explicit housing terms -> housing
  named faculty patterns -> people
  symposium/year terms -> symposium

Stage 2 — lightweight embedding or classifier router

Stage 3 — multi-label capability selection

Stage 4 — query decomposition only for genuinely multi-hop / multi-source questions
```

Do not route every ordinary query through an LLM planner.

---

# 8. Conversation memory

Recommended memory separation:

```text
1. Recent dialogue
   - small number of raw recent turns

2. Conversation state
   - current degree/program
   - referenced course IDs
   - referenced person/entity IDs
   - active catalog year
   - current topic/capability
   - unresolved ambiguity

3. Evidence state
   - source IDs / entity IDs returned in prior turns
   - selected results that pronouns or follow-ups may reference

4. Optional historical summary
   - conversational intent only
   - never authoritative institutional evidence
```

Example:

```text
User: Tell me about the MSCS.
User: What are its prerequisites?
User: Which of those does Dr. X teach?
```

State can resolve this as:

```text
active_program_id = <MSCS canonical ID>
previous_course_set = [...]
active_person_id = <Dr. X canonical ID>
```

rather than embedding the entire transcript and hoping retrieval reconstructs the reference chain.

---

# 9. RAG evaluation strategy

## 9.1 Build an AUM Gold Set — BUILD NOW

Initial target: roughly **250–400 carefully curated queries** across existing and near-term domains.

Suggested record:

```text
query_id
question
query_class
answerable
required_capabilities[]
gold_source_ids[]
gold_evidence_ids[]
gold_claims[]
expected_citations[]
expected_routes[]
catalog_or_policy_version
multi_turn_parent_id
notes
```

### Retrieval metrics

Primary:

- Recall@K

Additional:

- MRR;
- nDCG@K;
- Precision@K.

Always segment results by query class/domain rather than reporting one average only.

### Routing metrics

For multi-label routing:

- exact route-set match;
- micro-F1;
- macro-F1;
- missed-capability rate;
- unnecessary-capability rate.

### Answerability metrics

Track both:

```text
answerable query -> answered correctly
unanswerable query -> rejected correctly
```

Useful error categories:

```text
false answer
false abstention
partial support treated as full support
source conflict ignored
missing capability
retrieval miss
reranker miss
generation hallucination
citation mismatch
```

### Generation / grounding metrics

Measure:

- claim correctness;
- claim-level support;
- answer relevance;
- citation support;
- citation completeness;
- unsupported additions;
- contradictions.

Automated judge frameworks can accelerate regression testing, but keep a stable human-reviewed control set. Automated judges should not be the sole release criterion.

### Multi-turn evaluation

Script cases involving:

- pronouns;
- “the second one” references;
- topic changes;
- return to a previous topic;
- correction of a mistaken premise;
- catalog-year changes;
- unanswerable follow-ups;
- questions requiring multiple capabilities.

CORAL, MTRAG, and the later MTRAG-UN benchmark are useful references for the failure modes to emulate.

---

# 10. Data ingestion and provenance

## 10.1 Build a Source Registry before the KG — BUILD NOW

Suggested source record:

```text
SourceRecord
  source_id
  canonical_url
  authority_tier
  source_type
  domain
  title
  owner
  effective_from
  effective_to
  catalog_year
  retrieved_at
  content_hash
  parser_version
  access_classification
```

Every extracted chunk should preserve:

```text
chunk_id
source_id
document_version_id
heading_path
page/section/offset
text
content_hash
extraction_timestamp
```

## 10.2 Immutable/index-versioned ingestion — BUILD NOW

Recommended process:

```text
source documents
      ↓
fetch / parse
      ↓
validate
      ↓
create candidate document version
      ↓
build candidate indexes
      ↓
run regression + ingestion checks
      ↓
PASS
      ↓
atomically promote candidate snapshot
```

Avoid destructive in-place updates of the only production index.

AUM itself states that catalog provisions can change without actual notice. This makes temporal/version-aware ingestion a real institutional requirement rather than theoretical complexity.

## 10.3 Borrow provenance semantics from W3C PROV-O

The project does not need to adopt RDF immediately. However, concepts such as:

- `wasDerivedFrom`;
- `wasRevisionOf`;
- `hadPrimarySource`;
- `generatedAtTime`;
- `invalidatedAtTime`;

are useful conceptual primitives for a durable source/provenance model.

---

# 11. Knowledge Graph prerequisites and institutional ontology

## 11.1 Institutional root

The current public AUM academics page identifies five colleges:

```text
AUM
├── College of Business
├── College of Education
├── College of Liberal Arts & Social Sciences
├── College of Nursing & Health Sciences
└── College of Sciences
```

University College is also represented in AUM's broader institutional/catalog structures and should be modeled as an academic unit according to its actual institutional role rather than forced into a fixed five-college enum.

**Important modeling rule:** do not encode college or department names into application code as permanent schema fields. They are data.

Use:

```text
AcademicUnit
  unit_id
  unit_type = college | department | center | program | other
  canonical_name
  aliases[]
  parent_unit_id (where hierarchical)
```

---

## 11.2 College of Sciences deep-focus branch

The current AUM catalog lists:

```text
AUM
└── College of Sciences
    ├── Biology & Environmental Sciences
    ├── Chemistry
    ├── Computer Science
    ├── Mathematics
    └── Psychology
```

The planned KG should appear tree-like for navigation but become graph-like through typed cross-links.

---

## 11.3 Recommended canonical entity classes

```text
Institution
AcademicUnit
Department
Person
FacultyMember
Student
Degree
Major
Program
Course
CourseOffering
ResearchArea
ResearchProject
Publication
ResearchFacility
Event
Presentation
Policy
SourceDocument
DocumentVersion
```

Not all classes need to exist in the first implementation. Establish names and semantics during Phase 0, then introduce only those needed by the first vertical slice.

---

## 11.4 Recommended relation vocabulary

Candidate relations:

```text
SUBUNIT_OF
MEMBER_OF
OFFERED_BY
TEACHES
MENTORS
RESEARCHES
WORKS_ON
USES
RELATED_TO
AUTHORED_BY
PRESENTED_AT
PRODUCED
REQUIRED_FOR
PREREQUISITE_OF
LOCATED_IN
AFFILIATED_WITH
```

Relations should have:

- explicit direction;
- inverse semantics where useful;
- domain/range constraints;
- source-level provenance;
- temporal validity where applicable.

---

## 11.5 Canonical IDs — REQUIRED BEFORE KG

Names are not IDs.

Use immutable identifiers independent of display labels, e.g.:

```text
aum:unit:<id>
aum:person:<id>
aum:course:<id>
aum:program:<id>
aum:project:<id>
aum:event:<id>
aum:publication:<id>
```

Then labels and aliases can change independently:

```text
canonical_name = "Department of Computer Science"
aliases = ["Computer Science", "CS"]
```

Canonical ID strategy must also support entity resolution across:

- catalog pages;
- college websites;
- faculty directories;
- research/project data;
- symposium records;
- publications.

---

## 11.6 Assertion-level provenance

A relation should be represented conceptually as an assertion, not just an unqualified edge:

```text
RelationAssertion
  relation_id
  subject_id
  predicate
  object_id
  valid_from
  valid_to
  source_id
  document_version_id
  evidence_location
  observed_at
  extraction_method
  review_status
  confidence (only if meaningfully defined)
```

This permits:

- source conflicts;
- versioned relationships;
- historical queries;
- evidence-backed graph answers;
- audit/debugging.

---

## 11.7 Temporal relationships

Potentially temporal relations include:

- prerequisites;
- program requirements;
- faculty department membership;
- chair/dean roles;
- courses taught;
- project participation;
- facilities and ownership;
- event presentations.

Example:

```text
Course A PREREQUISITE_OF Course B
valid_for_catalog = 2026-2027
source = catalog_version_x
```

Do not flatten changing institutional relationships into timeless edges.

---

# 12. When to use graph traversal vs ordinary RAG

## Graph traversal is appropriate for questions such as:

- Who mentors students working on projects using Lab X?
- Which faculty working in cybersecurity teach courses required by a program?
- Which projects produced publications?
- What prerequisite chain connects Course A to Course D?
- Which symposium presentations connect a student, faculty mentor, project, and research area?

## Ordinary RAG is appropriate for:

- Explain the housing cancellation policy.
- Summarize the academic standing policy.
- What does the handbook say about a procedure?
- Compare two sections of a policy document.

## Structured exact lookup is appropriate for:

- What is the title of CSCI 3000?
- Which department offers this course?
- What office is listed for this faculty member?

## Hybrid retrieval is appropriate for:

- Find faculty doing protein research and summarize their current projects.
- Which facilities are connected to research in area X, and what does AUM say about those facilities?

Recommended hybrid flow:

```text
query
  ↓
exact / semantic entity anchoring
  ↓
bounded graph traversal
  ↓
retrieve source evidence attached to nodes/edges
  ↓
optional text retrieval for richer explanation
  ↓
rerank / evidence gate
  ↓
grounded response with citations
```

Graph traversal should earn its place through evaluation rather than become the universal retrieval mode.

---

# 13. Recommended phased KG build

## Phase 0 — FOUNDATION (add this before the original Phase 1)

Build and review:

- canonical entity model;
- canonical IDs;
- Source Registry;
- document/version model;
- provenance model;
- temporal model;
- relation vocabulary;
- graph constraints;
- entity resolution policy;
- authority/source precedence policy;
- 20–30 graph evaluation questions;
- tiny hand-verified sample graph.

**Exit condition:** the team can represent one institutional fact with identity, source, version, and validity without ambiguity.

---

## Phase 1 — RESEARCH VERTICAL

```text
AUM
 └─ College of Sciences
     └─ Departments
         ├─ Faculty
         └─ Research Projects
```

Useful relations:

```text
MEMBER_OF
RESEARCHES
WORKS_ON
MENTORS
```

Why first: research discovery is relation-heavy and offers clear graph-specific questions without requiring the entire curriculum model.

---

## Phase 2 — ACADEMIC PROGRAM GRAPH

Add:

```text
Majors
Degrees
Programs
Courses
Requirements
Prerequisites
```

Make catalog-year/version semantics mandatory.

---

## Phase 3 — RESEARCH ECOSYSTEM

Add:

```text
Research Areas
Publications
Facilities
Events
Symposium Presentations
Student Research
```

Cross-links create the richer institutional graph envisioned for the project.

---

## Phase 4 — UNIVERSITY EXPANSION

Expand to:

- remaining AUM colleges/units;
- cross-college projects;
- interdisciplinary programs;
- cross-unit faculty/research relations;
- institution-wide navigation/reasoning.

Do not build Phase 4 until the earlier ontology and update strategy survive real data changes.

---

# 14. RTX 4090 deployment analysis

## 14.1 Verified GPU facts

Official NVIDIA RTX 4090 specifications include:

- 24 GB GDDR6X memory;
- 16,384 CUDA cores;
- PCI Express Gen 4 support;
- no NVLink/SLI support.

This is enough VRAM to host the current Mistral-7B-Instruct-v0.3 BF16 weights, but not enough to assume unlimited context/concurrency.

---

## 14.2 Verified Mistral-7B-Instruct-v0.3 configuration

The official Hugging Face configuration reports:

```text
hidden_size = 4096
num_hidden_layers = 32
num_attention_heads = 32
num_key_value_heads = 8
max_position_embeddings = 32768
torch_dtype = bfloat16
```

The repository exposes a consolidated safetensors file of approximately 14.5 GB.

---

## 14.3 KV-cache intuition

A rough BF16 KV-cache payload estimate for this model is:

```text
32 layers
× 8 KV heads
× 128 head dimension
× 2 (K and V)
× 2 bytes
≈ 128 KiB / token / active sequence
```

Approximate raw KV payload per active sequence:

| Context | Approximate KV payload |
|---:|---:|
| 4K tokens | ~0.5 GiB |
| 8K tokens | ~1 GiB |
| 16K tokens | ~2 GiB |
| 32K tokens | ~4 GiB |

These are **not production VRAM requirements**. They exclude runtime buffers, allocator behavior, activations, CUDA/runtime overhead, fragmentation, and serving-engine implementation details.

They are useful only to explain why model fit does not equal concurrent-user capacity.

---

# 15. Local model serving recommendations

## 15.1 Direct Hugging Face / Transformers — KEEP AS BASELINE

Use for:

- development;
- model correctness checks;
- simple one-user experiments;
- comparison baseline during serving benchmarks.

A serialized `generate()` queue is straightforward but may leave batching and KV-memory efficiency on the table under concurrency.

---

## 15.2 vLLM — BENCHMARK FIRST / PRIMARY CANDIDATE

The PagedAttention/vLLM paper addresses memory waste in dynamic KV caches and reports 2–4× higher throughput than the evaluated serving baselines at comparable latency in its experiments.

Current vLLM is the first serving engine to benchmark because the project's deployment needs include:

- batching;
- streaming;
- concurrent requests;
- KV-cache management;
- local GPU serving.

AUM-specific performance still has to be measured on the exact machine and workload.

---

## 15.3 Text Generation Inference — DEFER / DO NOT SELECT FOR NEW DEPLOYMENT

Hugging Face currently states that TGI is in maintenance mode and will primarily accept minor fixes/documentation/lightweight maintenance.

Therefore TGI should not be the default foundation of a new AUM serving architecture.

---

## 15.4 llama.cpp — RETAIN AS AN ALTERNATIVE

Relevant capabilities include:

- NVIDIA CUDA kernels;
- OpenAI-compatible local server;
- CPU/GPU hybrid inference;
- GGUF format;
- 1.5–8-bit quantization options.

Its quantization documentation explicitly notes the trade-off: reduced model size and potentially faster inference at the risk of accuracy/perplexity degradation.

Use it if measurements show value for:

- low-bit deployment;
- portability;
- reduced VRAM;
- CPU/GPU hybrid serving.

---

# 16. Quantization decision

**Do not quantize simply because quantization exists.**

Mistral-7B-Instruct-v0.3 BF16 already fits in one RTX 4090.

Quantization is justified only if it improves the deployment objective, such as:

- more KV-cache headroom;
- greater concurrency;
- longer context;
- lower loading/storage requirements;
- spare GPU memory for other workloads.

Evaluate any quantized candidate on the same AUM gold set and serving workload.

Decision dimensions:

```text
answer quality
retrieval-grounded correctness
latency
TTFT
throughput
VRAM
stability
operational complexity
```

---

# 17. Deployment topology options

## Option A — one RTX 4090 workstation

```text
4090 workstation
├── model server
├── FastAPI
├── retrieval
├── reranker
├── indexes
└── persistent database
```

**Recommended first deployment topology.**

Advantages:

- simplest operations;
- easiest debugging;
- enough to measure real demand;
- no distributed network dependency.

Potential improvement: keep embedding/BM25 and, if performance permits, reranking on CPU to reserve VRAM for generation.

---

## Option B — application/retrieval host + one inference host

```text
Application / Retrieval
        │
        │ LAN
        ▼
RTX 4090 Model Server
```

**Likely best medium-term topology.**

Advantages:

- application restarts do not reload the model;
- clearer observability;
- retrieval does not contend for LLM VRAM;
- straightforward later model-replica expansion.

---

## Option C — multiple independent RTX 4090 model replicas

```text
                 Load Balancer
                /      |       \
             GPU-1   GPU-2    GPU-3
             model   model    model
```

Because the current 7B model fits one GPU, independent replicas are usually a cleaner scale-out pattern than splitting one 7B model across multiple 4090 desktops.

The RTX 4090 does not support NVLink, which further reduces the attraction of tightly coupled multi-machine tensor parallelism for this model.

---

## Option D — convert ~21 lab desktops into a cluster

**DEFER.**

First determine:

- whether machines are student workstations;
- whether they are always on;
- whether persistent server workloads are permitted;
- CPU/RAM/storage consistency;
- network topology and speed;
- administration/patching ownership;
- cooling/power behavior under sustained inference;
- whether a subset can be dedicated.

Twenty-one GPUs represent potential capacity, not automatically a reliable cluster.

---

# 18. Measure concurrent-user capacity instead of guessing

Benchmark at least:

```text
Input context:
1K
4K
8K
16K

Output length:
128
512
1024

Concurrent requests:
1
2
4
8
16
```

Record:

```text
p50 TTFT
p95 TTFT
p99 TTFT
inter-token latency
tokens/sec/request
aggregate tokens/sec
queue time
GPU utilization
VRAM high-water mark
CPU RAM
OOM events
request cancellation behavior
retrieval latency
reranking latency
end-to-end latency
```

Run the same workload through:

```text
current direct Transformers path
vs
vLLM
```

Only after that benchmark should the project claim a supported concurrent-user capacity.

---

# 19. Hardware inventory checklist

Collect these fields for candidate deployment machines:

| Area | Inventory required |
|---|---|
| GPU | exact RTX 4090 vendor/model; VRAM confirmation |
| CPU | model, sockets, cores, threads |
| RAM | capacity, speed, available memory |
| Disk | NVMe/SATA, capacity, free space, read/write performance |
| Network | NIC speed, switch speed, topology, VLAN |
| PCIe | generation and lane configuration |
| OS | distribution/version/kernel |
| NVIDIA | driver version and CUDA compatibility |
| Containers | Docker/Podman availability/policy |
| Operations | machines permitted to remain permanently on |
| Cooling | sustained thermal behavior |
| Power | PSU and UPS availability |
| Security | firewall/VLAN/inbound access rules |
| Persistence | backups and restore procedure |
| Administration | root/admin ownership and patch process |
| Monitoring | GPU/CPU/RAM/disk/service telemetry |
| DNS/TLS | internal hostname/certificate plan |

Provisional engineering target for a dedicated all-in-one server—not a formal model requirement:

- ~64 GB system RAM;
- >=1 TB NVMe;
- stable Linux LTS environment;
- reliable wired LAN.

Final requirements must be based on actual corpus size, concurrent usage, and institutional operational policy.

---

# 20. Security and privacy

Security becomes substantially more important once the assistant touches restricted/internal documents.

FERPA defines personally identifiable information in education records broadly, including direct identifiers, indirect identifiers, and information that can be linked to identify a student.

Recommended principles:

1. Public and restricted knowledge must be separate security domains.
2. Authenticate before retrieval.
3. Apply authorization **during retrieval**, not after generation.
4. Never place unauthorized source text into model context.
5. Store access-control metadata with every protected document/version.
6. Treat retrieved documents as untrusted content, never as instructions.
7. Defend against indirect prompt injection from retrieved content.
8. Separate observability/telemetry from conversation content.
9. Do not log sensitive raw prompts by default.
10. Define trace retention/deletion policy.
11. Encrypt traffic and sensitive persistent data.
12. Keep secrets outside Git and prompts.
13. Audit ingestion and software dependencies.
14. Do not ingest student-level education records until AUM governance, authorization, and FERPA handling are explicitly defined.
15. Keep the raw model server behind the application/API access layer; do not expose it directly to users.

Use OWASP LLM guidance and NIST AI RMF/Generative AI Profile as governance/security references, but adapt controls to AUM policy and actual data classes.

---

# 21. Recommended implementation priorities

| Recommendation | Timing | Complexity | Risk if delayed |
|---|---|---:|---:|
| Source Registry + immutable IDs | **NOW** | Low–Medium | Very high |
| Document/version model | **NOW** | Medium | Very high |
| Versioned index snapshots | **NOW** | Medium | High |
| AUM regression/evaluation dataset | **NOW** | Medium | Very high |
| Formal capability interfaces | **NOW** | Medium | High |
| Logical domain collections | **NOW** | Medium | Medium |
| Multi-label routing baseline | **NOW** | Low–Medium | Medium |
| Evidence/abstention gate | **NOW** | Medium | High |
| Model-serving abstraction | **NOW** | Low | Medium |
| vLLM vs Transformers benchmark | **NOW** | Low–Medium | Medium |
| Structured conversation state | **NEXT** | Medium | Medium |
| KG Phase 0 schema/provenance | **NEXT** | Medium | High if skipped |
| College of Sciences mini-KG | **AFTER PHASE 0** | Medium | Low |
| Multiple GPU replicas | **AFTER LOAD TEST** | Medium | Low |
| Quantized deployment | **ONLY IF JUSTIFIED** | Medium | Low |
| Full AUM KG | **LATER** | Very high | None now |

---

# 22. What not to build yet

Avoid spending the next development cycle on:

- graph database selection debates before KG requirements exist;
- a university-wide ontology;
- Microsoft GraphRAG integration as the default retrieval layer;
- autonomous multi-agent orchestration for routine queries;
- LLM routing for every question;
- generic long-term AI memory infrastructure;
- 21-machine distributed serving;
- Kubernetes;
- automatic LLM-driven entity merging without review/evaluation;
- student-record ingestion;
- replacing FAISS/BM25 for novelty;
- replacing Mistral before proving the generator is the bottleneck.

The graph database is not the current hard problem.

The hard questions are:

```text
What is an entity?
What is its canonical ID?
What is an assertion?
Which source asserted it?
Which source has greater authority?
When is the assertion valid?
How is it updated or invalidated?
What graph query classes must succeed?
How is traversal evaluated?
How does the answer preserve provenance/citations?
```

---

# 23. 30 / 60 / 90-day engineering roadmap

## Days 0–30 — make the assistant measurable

Primary objective: **establish contracts, provenance, and regression measurement before expanding the corpus.**

Build:

```text
Source Registry
DocumentVersion model
chunk provenance
capability interface
retrieval trace schema
Gold Set v1
unanswerable tests
routing tests
index snapshot/version mechanism
model-server abstraction
```

Also:

- benchmark current Mistral serving vs vLLM on one RTX 4090;
- collect complete lab hardware inventory;
- baseline retrieval and answer quality before major changes.

### Exit criterion

The project can say:

> We changed component X; Recall@10 changed from A to B, citation correctness from C to D, p95 latency from E to F, and abstention performance from G to H.

If the project cannot answer that question, additional architecture complexity is premature.

---

## Days 31–60 — multi-domain orchestration

Add or formalize:

```text
faculty/people capability
course/program capability
multi-label router
evidence gate
structured conversation state
cross-capability result fusion
source authority handling
```

Create deliberate cross-domain and multi-turn evaluation questions.

Begin KG Phase 0 documentation only: schema, identity, provenance, relation semantics, temporal model, and evaluation questions.

### Exit criterion

The assistant reliably distinguishes:

```text
exact structured lookup
single-domain RAG
multi-domain RAG
unanswerable
clarification-needed
```

---

## Days 61–90 — Knowledge Graph vertical slice

Build a hand-verifiable graph:

```text
AUM
 → College of Sciences
 → Departments
 → Faculty
 → Research Projects
```

Add selected relations:

```text
MEMBER_OF
RESEARCHES
WORKS_ON
MENTORS
PRESENTED_AT (if symposium data included)
```

Create 20–50 graph-specific evaluation questions.

For every graph-target query class, compare:

```text
current hybrid text RAG
vs
KG-assisted retrieval
```

Retain graph machinery only where it measurably improves correctness, coverage, citation quality, or interpretability.

At that point, the project will have evidence to select a graph database and eventual graph retrieval framework.

---

# 24. Architectural principles to preserve

These are the most durable findings from the investigation:

## Principle A — Retrieval is a capability system, not one database

Different institutional questions should use the retrieval mechanism appropriate to the data and query class.

## Principle B — LLMs synthesize; sources establish truth

The model is not the institutional database, graph, router, or authority.

## Principle C — Provenance is first-class data

Every important fact must be traceable to a source and version.

## Principle D — Abstention is a feature

A trustworthy academic assistant must know when its evidence is insufficient.

## Principle E — Evaluation precedes architectural escalation

New components should be added because they fix measured failure modes.

## Principle F — Graphs are for relations

Use graph traversal when relationship structure matters; keep ordinary RAG for prose/policy/document explanation.

## Principle G — Physical distribution comes after logical modularity

Define interfaces first. Separate machines only when resource or reliability measurements justify it.

## Principle H — Authorization belongs before retrieval

Protected data must never reach an unauthorized prompt context.

---

# 25. Recommended project-level architecture statement

Use this as a concise project design principle:

> **AUM Academic Assistant is a capability-oriented retrieval system over authoritative, versioned institutional knowledge. It uses deterministic structured lookup when possible, hybrid text retrieval when appropriate, bounded graph traversal when relationships matter, calibrated abstention when evidence is insufficient, and a local LLM primarily for language understanding and grounded synthesis—not as the database, router, source of truth, or graph itself.**

The intended Knowledge Graph remains consistent with the original vision:

> **AUM-centered institutional hierarchy for navigation + typed cross-domain relations for reasoning, with College of Sciences as the initial deep-focus branch.**

The additions required to make that vision reliable are:

- canonical identity;
- provenance;
- temporal validity;
- source authority;
- entity resolution;
- graph constraints;
- measurable graph evaluation.

---

# 26. Source registry

This section records the sources used to support the research snapshot. Prefer these canonical/primary links when re-verifying conclusions.

## 26.1 AUM institutional sources

### S-AUM-01 — AUM Academics

- **Title:** Academics | Undergraduate & Graduate Studies at AUM
- **URL:** https://www.aum.edu/academics/
- **Type:** Official institutional webpage
- **Supports:** AUM public academic organization; five colleges.
- **Freshness:** Re-verify before ontology changes.

### S-AUM-02 — AUM College of Sciences catalog

- **Title:** College of Sciences | Auburn University at Montgomery Catalog
- **URL:** https://catalog.aum.edu/about/collegeofsciences
- **Type:** Official catalog
- **Supports:** College of Sciences departments and official catalog structure.
- **Observed departments:** Biology & Environmental Sciences; Chemistry; Computer Science; Mathematics; Psychology.

### S-AUM-03 — AUM catalogs landing page

- **Title:** Catalogs | Undergraduate & Graduate Degree Programs at AUM
- **URL:** https://www.aum.edu/academics/catalogs/
- **Type:** Official institutional webpage
- **Supports:** existence of the 2026–27 digital catalog and explicit statement that catalog provisions may change.

### S-AUM-04 — AUM catalog departments index

- **Title:** Auburn University at Montgomery Catalog — Departments
- **URL:** https://catalog.aum.edu/departments
- **Type:** Official catalog
- **Supports:** department-name discovery and entity-resolution input.

### S-AUM-05 — AUM degrees/programs page

- **Title:** Undergraduate & Graduate Degrees & Certificates | AUM
- **URL:** https://www.aum.edu/academics/degrees/
- **Type:** Official institutional webpage
- **Supports:** degree/program discovery and institutional framing.

---

## 26.2 Local model / serving / hardware sources

### S-HW-01 — NVIDIA RTX 4090 official specifications

- **Title:** GeForce RTX 4090
- **URL:** https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
- **Type:** Official manufacturer documentation
- **Supports:** 24 GB GDDR6X, CUDA core count, PCIe generation, no NVLink.

### S-MODEL-01 — Mistral-7B-Instruct-v0.3 model card

- **Title:** mistralai/Mistral-7B-Instruct-v0.3
- **URL:** https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- **Type:** Official model repository
- **Supports:** model identity, recommended usage, checkpoint metadata.

### S-MODEL-02 — Mistral configuration

- **Title:** Mistral-7B-Instruct-v0.3 config.json
- **URL:** https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/config.json
- **Type:** Official model configuration
- **Supports:** hidden size, layer count, attention heads, KV heads, max positions, BF16 dtype.

### S-MODEL-03 — Mistral repository file listing

- **Title:** Mistral-7B-Instruct-v0.3 files
- **URL:** https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/tree/main
- **Type:** Official model repository
- **Supports:** ~14.5 GB consolidated safetensors weight size.

### S-SERVE-01 — PagedAttention / vLLM paper

- **Title:** Efficient Memory Management for Large Language Model Serving with PagedAttention
- **Authors:** Woosuk Kwon et al.
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2309.06180
- **DOI:** 10.1145/3600006.3613165
- **Type:** Peer-reviewed systems paper (SOSP 2023)
- **Supports:** KV-cache memory-management motivation; vLLM throughput claims in the paper's evaluated setting.

### S-SERVE-02 — Hugging Face TGI documentation

- **Title:** Text Generation Inference
- **URL:** https://huggingface.co/docs/text-generation-inference/index
- **Type:** Official framework documentation
- **Supports:** current maintenance-mode status.

### S-SERVE-03 — llama.cpp README

- **Title:** llama.cpp README
- **URL:** https://github.com/ggml-org/llama.cpp/blob/master/README.md
- **Type:** Official project documentation
- **Supports:** CUDA kernels, CPU/GPU hybrid inference, low-bit support, local OpenAI-compatible server.

### S-SERVE-04 — llama.cpp quantization documentation

- **Title:** llama.cpp quantize README
- **URL:** https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- **Type:** Official project documentation
- **Supports:** GGUF quantization process and size/performance/accuracy trade-off.

---

## 26.3 RAG evaluation / abstention sources

### S-EVAL-01 — RAGAS

- **Title:** RAGAs: Automated Evaluation of Retrieval Augmented Generation
- **Authors:** Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert
- **Venue:** EACL 2024 System Demonstrations
- **URL:** https://aclanthology.org/2024.eacl-demo.16/
- **PDF:** https://aclanthology.org/2024.eacl-demo.16.pdf
- **DOI:** 10.18653/v1/2024.eacl-demo.16
- **Supports:** decomposed automated evaluation of retrieval relevance, faithfulness, and generation quality.

### S-EVAL-02 — ARES

- **Title:** ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems
- **Authors:** Jon Saad-Falcon, Omar Khattab, Christopher Potts, Matei Zaharia
- **Venue:** NAACL 2024
- **URL:** https://aclanthology.org/2024.naacl-long.20/
- **PDF:** https://aclanthology.org/2024.naacl-long.20.pdf
- **DOI:** 10.18653/v1/2024.naacl-long.20
- **Supports:** context relevance, answer faithfulness, answer relevance; lightweight judges combined with human annotations/PPI.

### S-EVAL-03 — UAEval4RAG

- **Title:** Unanswerability Evaluation for Retrieval Augmented Generation
- **Authors:** Xiangyu Peng, Prafulla Kumar Choubey, Caiming Xiong, Chien-Sheng Wu
- **Venue:** ACL 2025
- **URL:** https://aclanthology.org/2025.acl-long.415/
- **PDF:** https://aclanthology.org/2025.acl-long.415.pdf
- **Supports:** explicit unanswerability testing; no one RAG configuration consistently optimizes answerable and unanswerable behavior across datasets.

---

## 26.4 Multi-turn RAG / memory sources

### S-MEM-01 — CORAL

- **Title:** CORAL: Benchmarking Multi-turn Conversational Retrieval-Augmentation Generation
- **Authors:** Yiruo Cheng et al.
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2410.23090
- **Supports:** passage retrieval, response generation, citation labeling, topic shifts, multi-turn conversational RAG failure modes.

### S-MEM-02 — MTRAG

- **Title:** MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems
- **Authors:** Yannis Katsis et al.
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2501.03468
- **Project:** https://github.com/ibm/mt-rag-benchmark
- **Supports:** later turns, non-standalone questions, unanswerable questions, multiple domains; 110 conversations / 842 tasks in the reported benchmark.

### S-MEM-03 — MTRAG-UN

- **Title:** MTRAG-UN: A Benchmark for Open Challenges in Multi-Turn RAG Conversations
- **Authors:** Sara Rosenthal et al.
- **Year:** 2026
- **URL:** https://arxiv.org/abs/2602.23184
- **Project:** https://github.com/IBM/mt-rag-benchmark
- **Supports:** continued difficulty with unanswerable, underspecified, non-standalone, and unclear multi-turn interactions.

---

## 26.5 Routing / federated retrieval sources

### S-ROUTE-01 — RouterRetriever

- **Title:** RouterRetriever: Routing over a Mixture of Expert Embedding Models
- **Authors:** Hyunji Lee, Luca Soldaini, Arman Cohan, Minjoon Seo, Kyle Lo
- **Venue:** AAAI 2025
- **URL:** https://ojs.aaai.org/index.php/AAAI/article/view/33306
- **PDF:** https://ojs.aaai.org/index.php/AAAI/article/download/33306/35461
- **arXiv:** https://arxiv.org/abs/2409.02685
- **Supports:** routing among domain-specific retriever experts and heterogeneous-domain retrieval.

### S-ROUTE-02 — RAGRoute

- **Title:** Efficient Federated Search for Retrieval-Augmented Generation
- **Authors:** Rachid Guerraoui et al.
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2502.19280
- **Supports:** lightweight routing over distributed repositories; reported reductions in unnecessary queries/communication.

### S-ROUTE-03 — Cross-domain routing/planning

- **Title:** Talk to Right Specialists: Iterative Routing in Multi-agent Systems for Question Answering
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2501.07813
- **Supports:** the need to route and plan across knowledge boundaries for cross-domain/multi-hop questions.
- **Use cautiously:** architecture is more agentic than currently recommended for AUM; source is used to support the problem, not to require its full implementation.

---

## 26.6 Graph / Knowledge Graph retrieval sources

### S-KG-01 — GNN-RAG

- **Title:** GNN-RAG: Graph Neural Retrieval for Efficient Large Language Model Reasoning on Knowledge Graphs
- **Authors:** Costas Mavromatis, George Karypis
- **Venue:** Findings of ACL 2025
- **URL:** https://aclanthology.org/2025.findings-acl.856/
- **PDF:** https://aclanthology.org/2025.findings-acl.856.pdf
- **DOI:** 10.18653/v1/2025.findings-acl.856
- **Supports:** graph retrieval value for multi-hop and multi-entity KGQA; reported 8.9–15.5 percentage-point answer-F1 improvements over LLM-based retrieval approaches on those categories.

### S-KG-02 — GeAR

- **Title:** GeAR: Graph-enhanced Agent for Retrieval-augmented Generation
- **Authors:** Zhili Shen et al.
- **Venue:** Findings of ACL 2025
- **URL:** https://aclanthology.org/2025.findings-acl.624/
- **PDF:** https://aclanthology.org/2025.findings-acl.624.pdf
- **DOI:** 10.18653/v1/2025.findings-acl.624
- **Supports:** graph expansion can augment conventional retrievers such as BM25 for multi-hop QA; reported >10% improvement on MuSiQue in the evaluated setting.

### S-KG-03 — GRAG

- **Title:** GRAG: Graph Retrieval-Augmented Generation
- **Authors:** Yuntong Hu et al.
- **Venue:** Findings of NAACL 2025
- **URL:** https://aclanthology.org/2025.findings-naacl.232/
- **DOI:** 10.18653/v1/2025.findings-naacl.232
- **Supports:** networked/graph-structured document retrieval and multi-hop graph reasoning.

### S-KG-04 — Microsoft GraphRAG research paper

- **Title:** From Local to Global: A Graph RAG Approach to Query-Focused Summarization
- **Authors:** Darren Edge et al.
- **Year:** 2024
- **URL:** https://arxiv.org/abs/2404.16130
- **Supports:** graph/community summarization for global corpus-sensemaking questions.
- **Important interpretation:** this is not the same problem as a canonical institutional ontology; use it later for global-sensemaking workloads if needed.

### S-KG-05 — W3C PROV-O

- **Title:** PROV-O: The PROV Ontology
- **URL:** https://www.w3.org/TR/prov-o/
- **Type:** W3C Recommendation
- **Supports:** standardized provenance concepts and vocabulary.

---

## 26.7 Security, privacy, and governance sources

### S-SEC-01 — FERPA PII definition

- **Title:** Personally Identifiable Information for Education Records
- **URL:** https://studentprivacy.ed.gov/content/personally-identifiable-information-education-records
- **Type:** U.S. Department of Education / Student Privacy Policy Office
- **Supports:** broad FERPA-specific PII definition, including direct and indirect identifiers.

### S-SEC-02 — FERPA education-record definition

- **Title:** What is an education record?
- **URL:** https://studentprivacy.ed.gov/faq/what-education-record
- **Type:** U.S. Department of Education
- **Supports:** definition of education records and examples relevant to postsecondary institutions.

### S-SEC-03 — FERPA regulations/reference page

- **Title:** FERPA | Protecting Student Privacy
- **URL:** https://studentprivacy.ed.gov/ferpa
- **Type:** U.S. Department of Education
- **Supports:** statutory/regulatory definitions and disclosure context.

### S-SEC-04 — OWASP LLM Top 10 (2025)

- **Title:** OWASP Top 10 for LLM Applications 2025
- **URL:** https://owasp.org/www-project-top-10-for-large-language-model-applications/
- **PDF used in research:** https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf
- **Supports:** prompt injection, sensitive-information disclosure, vector/embedding weaknesses, and other LLM-application security risks.

### S-SEC-05 — NIST Generative AI Profile

- **Title:** Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile
- **Publication:** NIST AI 600-1
- **URL:** https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf
- **Type:** NIST guidance
- **Supports:** lifecycle risk-management/governance framing for generative AI systems.

---

# 27. Source-to-decision map

| Decision / conclusion | Main supporting sources | Nature of conclusion |
|---|---|---|
| Keep hybrid sparse+dense retrieval + reranking baseline | RAG evaluation/retrieval literature; current architecture | Evidence-informed engineering recommendation |
| Use per-capability logical collections rather than one flat global store | RouterRetriever; RAGRoute; multi-domain routing literature | Evidence-informed engineering recommendation |
| Multi-label routing is needed | RAGRoute; cross-domain routing/planning work; project query patterns | Engineering recommendation |
| Build explicit abstention/evidence gate | UAEval4RAG; ARES; RAGAS | Strongly evidence-backed |
| Evaluate multi-turn behavior separately | CORAL; MTRAG; MTRAG-UN | Strongly evidence-backed |
| Build Source Registry and versioning before KG | AUM catalog change policy; W3C PROV-O; institutional inconsistency risk | Engineering requirement derived from evidence |
| Use KG only for relational/multi-hop query classes | GNN-RAG; GeAR; GRAG | Evidence-backed direction |
| Do not equate Microsoft GraphRAG with institutional KG | GraphRAG paper's global-sensemaking objective vs AUM ontology needs | Scope/architecture interpretation |
| Benchmark vLLM first | vLLM/PagedAttention paper; TGI maintenance status | Strong serving recommendation |
| Do not quantize merely for fit | RTX 4090 VRAM + Mistral weight size; llama.cpp quantization trade-offs | Engineering judgment |
| Prefer GPU replicas over splitting current 7B model across 4090 desktops | current model fit; no RTX 4090 NVLink; operational simplicity | Engineering judgment; benchmark before scaling |
| Authorization must happen before retrieval for protected sources | FERPA; OWASP; security architecture principles | Security requirement |

---

# 28. Explicit uncertainties / facts that must still be measured

The following are intentionally unresolved:

## Deployment capacity

Unknown until measured:

- supported concurrent users;
- p95/p99 latency under realistic AUM workloads;
- safe maximum context under concurrency;
- benefit of vLLM over current direct generation on this hardware;
- whether reranking should run on CPU or GPU.

## Lab hardware

Still required:

- exact CPU models;
- RAM per workstation;
- storage type/capacity;
- NIC/switch speeds;
- persistent-server policy;
- operating system versions;
- which machines can remain powered and dedicated.

## Knowledge Graph implementation

Premature to finalize:

- graph database;
- graph query language;
- GraphRAG framework;
- ingestion/extraction framework;
- final ontology breadth.

These should follow Phase 0 and graph-specific evaluation.

## Model choice

The current generator may eventually become a quality bottleneck, but the project should first measure whether errors originate from:

- retrieval;
- routing;
- missing source data;
- evidence insufficiency;
- context construction;
- generation.

Do not upgrade the LLM merely because larger/newer models exist.

---

# 29. Immediate next engineering tasks

Recommended project-root work queue derived from this research:

```text
P0
- Source Registry schema
- DocumentVersion schema
- chunk provenance schema
- immutable index-version naming
- AUM Gold Set v1
- trace schema expansion
- formal retrieval capability interface

P1
- per-capability logical collections
- multi-label routing baseline
- evidence gate prototype
- structured conversation state
- model-server abstraction
- vLLM benchmark harness

P2
- KG Phase 0 ontology/ID/provenance document
- small College of Sciences research graph
- graph-vs-text evaluation suite
```

---

# 30. Original deep-research specification

The original research prompt is preserved verbatim below so future agents can distinguish the **requested investigation** from the **resulting recommendations**.

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


---

# 31. Maintenance instructions for future agents

When this file is revisited:

1. **Do not silently overwrite historical conclusions.** Add a dated update section when evidence changes.
2. Re-check all sources marked time-sensitive, especially AUM pages and framework documentation.
3. Preserve source IDs (`S-AUM-*`, `S-SERVE-*`, etc.) where possible so code/docs can reference them.
4. Add benchmark results from the actual AUM RTX 4090 machine into a new dated deployment-results section.
5. Add AUM Gold Set metrics before changing retrieval architecture.
6. Do not select a graph database until KG Phase 0 and graph evaluation questions are complete.
7. When a recommendation is changed, record:

```text
old recommendation
new recommendation
date
reason
new evidence
benchmark impact
migration impact
```

8. Treat externally generated summaries as non-authoritative. Source documents and AUM-owned records remain the ground truth.

---

# 32. Research snapshot summary

The research does **not** recommend replacing the project with a new architecture. It recommends turning the current prototype into a disciplined institutional retrieval system by adding the missing layers around it:

```text
current strong retrieval core
        +
source identity / versioning / provenance
        +
capability routing
        +
evidence sufficiency / abstention
        +
subsystem evaluation
        +
structured conversation state
        +
clean local model serving
        +
phased relational KG where justified
        =
maintainable AUM Academic Assistant
```

**End of research snapshot.**
