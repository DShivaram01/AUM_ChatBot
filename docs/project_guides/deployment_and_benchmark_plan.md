# Deployment and Benchmark Plan

## 1. Known environment

Planned AUM environment:

- approximately 21 lab desktops with NVIDIA RTX 4090 GPUs;
- local/university-controlled inference;
- current generator: Mistral-7B-Instruct-v0.3.

Unknown and still to inventory:

- CPU models;
- system RAM;
- storage;
- NIC/switch speed;
- OS/kernel;
- driver/CUDA versions;
- machine availability;
- permission to run persistent services;
- cooling/power/UPS;
- administrative ownership.

Do not invent these values.

---

## 2. First topology to test

Start with one RTX 4090 workstation:

```text
model server
FastAPI
retrieval
indexes
persistent data
```

This is an evaluation/deployment learning topology, not necessarily the permanent production architecture.

---

## 3. Medium-term topology candidate

If measurements justify separation:

```text
Application / Retrieval Host
        │
        │ LAN
        ▼
RTX 4090 Inference Host
```

Benefits:

- app restart does not reload model;
- clearer model-serving boundary;
- easier inference monitoring;
- retrieval does not need LLM VRAM;
- easier later replica scale-out.

---

## 4. Scale-out candidate

If real load requires more capacity:

```text
load balancer
  ├── RTX 4090 replica 1
  ├── RTX 4090 replica 2
  └── RTX 4090 replica N
```

For the current 7B model, independent replicas are the default scale-out idea to evaluate before tightly coupled multi-machine model parallelism.

---

## 5. Deferred topology

Do not currently design:

- 21-machine cluster;
- Kubernetes;
- distributed tensor parallelism;
- complex scheduler.

Potential hardware is not the same as reliable operational infrastructure.

---

## 6. Serving candidates

### Direct Transformers

KEEP as correctness/development baseline.

### vLLM

Primary production-style benchmark candidate.

Evaluate:

- batching;
- TTFT;
- throughput;
- KV-cache behavior;
- concurrency;
- cancellation;
- operational complexity.

### TGI

Do not choose as new default based on the 2026-08-24 research snapshot; re-verify current status if reconsidered.

### llama.cpp

Retain as an alternative if low-bit/portable/hybrid CPU-GPU deployment becomes valuable.

---

## 7. Benchmark matrix

### Input context

```text
1K
4K
8K
16K
```

### Output length

```text
128
512
1024
```

### Concurrent requests

```text
1
2
4
8
16
```

---

## 8. Metrics

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
OOM/error count
cancellation behavior
retrieval latency
reranking latency
end-to-end latency
```

---

## 9. Quality must be measured with performance

If testing:

- quantization;
- model change;
- serving-engine change that alters outputs;

run the same AUM Gold Set.

Optimize:

```text
quality
grounding
latency
TTFT
throughput
VRAM
stability
operational complexity
```

not throughput alone.

---

## 10. Hardware inventory checklist

Collect:

| Area | Required |
|---|---|
| GPU | exact vendor/model + VRAM |
| CPU | model/cores/threads |
| RAM | capacity/speed |
| Disk | NVMe/SATA/capacity/free space |
| Network | NIC/switch speed/topology/VLAN |
| PCIe | generation/lanes |
| OS | distro/version/kernel |
| NVIDIA | driver/CUDA compatibility |
| Containers | Docker/Podman policy |
| Operations | always-on/persistent-service permission |
| Cooling | sustained thermal behavior |
| Power | PSU/UPS |
| Security | firewall/VLAN/inbound rules |
| Persistence | backups/restore |
| Administration | root/admin/patch ownership |
| Monitoring | GPU/CPU/RAM/disk/service telemetry |
| DNS/TLS | internal hostname/certificate plan |

---

## 11. Capacity claims

Do not say “one 4090 supports N users” from theoretical VRAM.

Supported concurrency is an empirical result of:

```text
model
context
output length
serving engine
batching
retrieval overhead
hardware
latency SLO
```

Only publish a capacity number after reproducible load tests.
