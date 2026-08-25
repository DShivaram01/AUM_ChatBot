# Source, Provenance, and Versioning Policy

This is a foundational data contract for the AUM Academic Assistant.

The assistant must distinguish:

> “I retrieved text”  
from  
> “I have authoritative evidence from a known source/version supporting this claim.”

---

## 1. Source Registry

Suggested model:

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

`source_id` should remain stable across versions when the conceptual source is the same.

---

## 2. Document versions

A changed catalog/PDF/page snapshot becomes a new immutable version.

Suggested:

```text
DocumentVersion
  document_version_id
  source_id
  version_label
  retrieved_at
  content_hash
  effective_from
  effective_to
  supersedes_version_id
  parser_version
```

Do not overwrite history.

---

## 3. Chunk provenance

Every retrievable unit should carry:

```text
chunk_id
source_id
document_version_id
heading_path
page
section
offset
text
content_hash
extracted_at
access_classification
```

The answer layer should never have to reconstruct provenance from display text.

---

## 4. Index snapshot identity

Every built retrieval index should have a stable snapshot ID containing or referring to:

```text
index_snapshot_id
capability
source/document versions included
embedding model ID/version
chunking/parser version
build timestamp
content manifest/hash
```

---

## 5. Candidate → promotion lifecycle

```text
new/changed sources
      ↓
immutable document versions
      ↓
candidate chunks
      ↓
candidate BM25/vector indexes
      ↓
validation + Gold Set regression
      ↓
PASS
      ↓
atomic promotion
```

Rollback should be possible by restoring the previous snapshot.

---

## 6. Authority and conflicts

AUM sources can disagree or lag one another.

Do not collapse disagreement silently.

Represent assertions with provenance:

```text
subject
predicate
object/value
source_id
document_version_id
evidence_location
observed_at
valid_from
valid_to
authority_tier
review_status
```

A separate authority policy may eventually decide which source is preferred for particular fact classes.

---

## 7. Time

Temporal metadata is necessary for:

- catalog requirements;
- prerequisites;
- faculty roles;
- program structures;
- policies;
- course offerings;
- facilities/ownership;
- graph relations.

Avoid timeless edges/facts when the institutional relation can change.

---

## 8. Source access classification

Suggested initial classes:

```text
PUBLIC
INTERNAL
RESTRICTED
```

Authorization checks must occur before retrieval for protected classes.

---

## 9. W3C PROV-O concepts

The project does not need RDF immediately, but useful concepts include:

- `wasDerivedFrom`;
- `wasRevisionOf`;
- `hadPrimarySource`;
- `generatedAtTime`;
- `invalidatedAtTime`.

These can inform internal schema semantics without forcing an RDF implementation.

---

## 10. Trace integration

Every answer trace should eventually preserve:

```text
source IDs
document version IDs
index snapshot IDs
selected chunk/evidence IDs
citation IDs
```

---

## 11. Definition of provenance-complete answer

A grounded answer is provenance-complete when a developer can identify:

1. the exact source;
2. the exact version;
3. the exact evidence location;
4. the index snapshot used;
5. the retrieval/rerank decision;
6. the final claim/citation mapping.

This is the target, even if implementation is phased.
