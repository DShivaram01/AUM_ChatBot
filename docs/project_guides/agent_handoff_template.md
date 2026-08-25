# AI Agent Handoff Template

Use after every significant Claude Code or Codex task.

---

## Task identity

**ID:**  
**Title:**  
**Primary agent:**  
**Date:**  
**Branch:**  
**Commit:**  
**Decision label:** KEEP / BUILD NOW / BUILD NEXT / BENCHMARK FIRST / DEFER / AVOID / N/A

---

## Goal

What problem was this task intended to solve?

---

## Accepted constraints used

Which project principles/decisions were relevant?

---

## Previous behavior

What happened before the change?

---

## Changes made

### Files modified

- 

### Files added

- 

### Files intentionally not changed

- 

---

## Schema / contract impact

Did this change:

- API schemas?
- Source schema?
- DocumentVersion?
- Evidence?
- capability interfaces?
- conversation state?
- trace schema?

---

## Source / provenance / version impact

- new `source_id`s:
- new document/index version:
- migration needed:
- cache/index invalidation:
- authority/access impact:

---

## Architectural impact

Describe dependency/state/service changes.

---

## Tests run

### Static

```text
command
result
```

### Unit

### Integration

### Runtime

---

## AUM Gold Set / AI evals

| Metric | Before | After | Notes |
|---|---:|---:|---|
| Recall@K | | | |
| Routing | | | |
| Abstention | | | |
| Grounding/citations | | | |

If not applicable, explain why.

---

## Performance

| Metric | Before | After |
|---|---:|---:|
| TTFT | | |
| Total latency | | |
| Tokens/sec | | |
| VRAM high-water | | |
| Queue time | | |

---

## Bugs / risks discovered

- 

---

## Assumptions

- 

---

## Unresolved work

- 

---

## Recommended next step

One concrete next action.

---

## Notes for the next agent

Anything easy to miss when reopening the project.
