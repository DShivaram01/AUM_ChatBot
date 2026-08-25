# Knowledge Graph Phase 0 — Prerequisites

**Status:** planning foundation only  
**Do not select a graph database/framework yet.**

---

## 1. Product vision

The intended graph is an AUM-centered institutional hierarchy that becomes a graph through cross-links.

At a high level:

```text
AUM
├── College of Business
├── College of Education
├── College of Liberal Arts & Social Sciences
├── College of Nursing & Health Sciences
├── College of Sciences
└── other academic units as appropriate
```

The first deep-focus branch is College of Sciences.

Current research snapshot lists the College of Sciences departments as:

```text
Biology & Environmental Sciences
Chemistry
Computer Science
Mathematics
Psychology
```

Re-verify AUM sources before implementing ontology changes.

---

## 2. Tree + graph

Navigation may look hierarchical:

```text
AUM
→ College
→ Department
→ Program / Faculty / Course / Research
```

But cross-links make it a graph:

```text
Faculty ──MEMBER_OF────→ Department
Faculty ──TEACHES──────→ Course
Faculty ──MENTORS──────→ Student / Project
Faculty ──RESEARCHES───→ ResearchArea
Project ──USES─────────→ ResearchFacility
Project ──PRESENTED_AT─→ Event
Project ──PRODUCED─────→ Publication
Course ──REQUIRED_FOR──→ Program / Degree
Course ──PREREQUISITE_OF→ Course
Publication ──AUTHORED_BY→ Person
```

Do not force all relations into parent/child edges.

---

## 3. Canonical entity classes to evaluate

Potential classes:

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

Do not implement every class in the first vertical slice.

---

## 4. Canonical IDs

Names are labels, not identifiers.

Candidate pattern:

```text
aum:unit:<id>
aum:person:<id>
aum:course:<id>
aum:program:<id>
aum:project:<id>
aum:event:<id>
aum:publication:<id>
```

IDs must survive label/alias changes.

---

## 5. Entity resolution

Need a policy for matching the same entity across:

- AUM catalog;
- department pages;
- faculty directory;
- research pages;
- COS/symposium data;
- publications.

Do not allow automatic LLM entity merging to become authoritative without evaluation/review.

---

## 6. Relation assertions

A graph relation should conceptually be an assertion with provenance:

```text
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
```

---

## 7. Temporal validity

Relations that may change:

- prerequisites;
- program requirements;
- department membership;
- chair/dean roles;
- teaching;
- project involvement;
- facility ownership;
- event participation.

Temporal/version semantics are mandatory for those relation classes.

---

## 8. Authority/conflict policy

Before graph ingestion, define how to represent:

- conflicting sources;
- superseded catalog versions;
- duplicate people/projects;
- missing/uncertain relations.

Do not erase conflicting assertions during ingestion.

---

## 9. Graph-specific evaluation questions

Before implementation, write 20–30 questions where graph structure might genuinely improve retrieval.

Examples:

- Which faculty working in cybersecurity teach a course required by this program?
- Who mentors projects using Facility X?
- Which projects produced publications?
- What prerequisite chain links Course A to Course D?
- Which symposium presentations connect this faculty member to this research area?

These questions become the graph acceptance set.

---

## 10. Tiny hand-verified sample graph

Before selecting a database, model a tiny slice by hand.

Example:

```text
AUM
→ College of Sciences
→ Computer Science
→ 2–3 faculty
→ 3–5 projects
→ several research areas
```

Attach source/version provenance to every assertion.

If the model is awkward here, a graph database will not fix the ontology.

---

## 11. Phase 0 exit criteria

Do not move to a real graph implementation until the team can answer:

1. What is an entity?
2. What is its canonical ID?
3. What is an assertion?
4. Which source/version supports it?
5. When is it valid?
6. How are aliases/deduplication handled?
7. How are conflicts represented?
8. What relation types exist?
9. Which graph questions must succeed?
10. How will graph-assisted answers cite underlying evidence?

---

## 12. First implementation after Phase 0

Recommended first vertical:

```text
AUM
→ College of Sciences
→ Departments
→ Faculty
→ Research Projects
```

Then evaluate against existing hybrid text RAG.

Graph machinery should remain only where it measurably improves answers.
