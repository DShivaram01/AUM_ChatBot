# Development Workflow

Use this workflow for significant code, data, retrieval, serving, and architecture changes.

---

## 1. Task definition

Every task should state:

### Goal

What user-visible or engineering problem is solved?

### Current behavior

What happens now?

### Desired behavior

What must change?

### Architectural owner

Which layer owns the behavior?

### Source/data impact

Does this change:

- source identities;
- document versions;
- chunk schema;
- index version;
- entity IDs;
- API schemas?

### Non-goals

What is intentionally untouched?

### Acceptance criteria

What observable behavior proves completion?

### Validation

Which tests/evals/benchmarks are required?

---

## 2. Read accepted project context first

For architecture-affecting work, inspect:

- `project_updates.md`;
- `engineering_principles.md`;
- `decision_register.md`;
- relevant sections of the deep-research architecture reference.

Do not re-litigate accepted decisions without new evidence.

---

## 3. Inspect before editing

Trace:

1. request entry;
2. callers;
3. state;
4. data sources;
5. retrieval path;
6. model call;
7. response consumer;
8. current tests/evals.

Patch the real execution path, not the issue description.

---

## 4. Reproduce bugs first

For a bug:

```text
reproduction
→ capture failure
→ locate layer
→ regression case
→ fix
→ rerun
```

---

## 5. Architecture design note for multi-module changes

Use:

```text
Problem
Accepted constraints
Current execution path
Proposed interfaces
Source/version impact
State impact
Migration sequence
Risks
Rollback
Tests/evals
```

---

## 6. Data ingestion workflow

For institutional source updates:

```text
fetch / receive source
→ create SourceRecord if new
→ create immutable DocumentVersion
→ extract/chunk with provenance
→ validate
→ build candidate indexes
→ run ingestion checks + Gold Set regression
→ promote snapshot
```

Never overwrite authoritative history merely because a webpage changed.

---

## 7. Implementation sequence

Prefer:

```text
schema/interface
→ unit tests
→ core implementation
→ integration tests
→ eval fixtures
→ API/UI integration
→ runtime test
→ docs/decision update
```

---

## 8. AI-system change workflow

For retrieval/routing/evidence/model changes:

1. record baseline metrics;
2. define target failure mode;
3. change one major variable where possible;
4. rerun Gold Set;
5. segment results by query class/capability;
6. inspect regressions;
7. keep/revert based on evidence.

---

## 9. Serving/performance workflow

Benchmark comparable configurations.

For example:

```text
Direct Transformers
vs
vLLM
```

with the same:

- model;
- prompts;
- input lengths;
- output lengths;
- concurrency;
- hardware.

Record TTFT, p50/p95 latency, tokens/sec, queue time, VRAM, CPU RAM, errors.

---

## 10. Git discipline

One substantial task → one branch / isolated commit series.

Suggested:

```text
task/<id>-<short-purpose>
```

Commit messages should describe behavior.

---

## 11. AI-agent integration

Do not overwrite current code from an old agent snapshot.

Use:

```text
diff
→ inspect
→ apply minimal useful patch
→ test against current branch
```

---

## 12. Documentation

Update the appropriate artifact:

- project update;
- issue/fix log;
- decision register;
- architecture note;
- source/provenance policy;
- handoff.

Keep migration archaeology out of production code where possible.

---

## 13. Definition of Done

A significant task is complete when:

- current code was inspected;
- implementation follows accepted architecture;
- tests pass;
- relevant Gold Set/evals pass;
- source/version impact is handled;
- traces remain meaningful;
- no unexplained performance regression exists;
- runtime path is smoke-tested where relevant;
- docs/decision/handoff are updated;
- remaining risks are explicit.
