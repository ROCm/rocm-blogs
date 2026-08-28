---
blogpost: true
blog_title: "From Load Balancing to Self-Healing: Progressive KV Cache Architecture for DeepSeek-V3.2 on AMD Instinct"
date: "02 Jun 2026"
author: "Matvei Pashkovskii, Tyko Niemi"
thumbnail: ''
tags: "LLM, Performance, Serving, Kubernetes"
category: "Software tools & optimizations"
target_audience: "ML infrastructure engineers, MLOps teams, and platform engineers deploying large-scale LLM inference on AMD Instinct GPUs. Intermediate-to-advanced practitioners familiar with Kubernetes and vLLM who need to optimize multi-instance serving with distributed caching."
key_value_propositions: "Demonstrates a reproducible TTFT improvement for 671B MoE inference on MI300X through progressive KV cache architecture (local → CPU offload → distributed), with chaos-tested resilience proving the system survives pod failures with minimal degradation. Provides production-ready k8s manifests and CLI commands."
language: English
myst:
    html_meta:
        "author": "Matvei Pashkovskii, Tyko Niemi"
        "description lang=en": "3-stage guide to faster DeepSeek-V3.2 inference on MI300X via distributed KV caching with Mooncake."
        "keywords": "AMD Instinct, vLLM, KV cache, Mooncake, DeepSeek-V3, distributed inference, RDMA, ROCm, prefix caching, consistent hashing, chaos engineering"
        "vertical": "AI"
        "amd_category": "Developer Resources"
        "amd_asset_type": "Blog"
        "amd_technical_blog_type": "Tools, Features, and Optimizations"
        "amd_blog_hardware_platforms": "Instinct GPUs"
        "amd_blog_development_tools": "ROCm Software, Open-Source Tools"
        "amd_blog_applications": "AI Inference, Deploying AI at Scale"
        "amd_blog_topic_categories": "Software & Ecosystem"
        "amd_blog_authors": "Matvei Pashkovskii, Tyko Niemi"
---

<!---
Copyright (c) 2026 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->

# From Load Balancing to Self-Healing: Progressive KV Cache Architecture for DeepSeek-V3.2 on AMD Instinct

Running large MoE models like DeepSeek-V3.2 (671B parameters) across multiple AMD Instinct GPUs gives you throughput and redundancy — but each instance's KV cache is local. Pod restarts, scaling events, and load-balancer rebalancing force expensive cold-start prefills of 50-100K tokens. In this guide, you'll build a progressive caching architecture that eliminates this problem in three stages, achieving **3× lower time-to-first-token (TTFT)** and surviving continuous pod failures with minimal degradation. You'll walk away with production-ready Kubernetes manifests, CLI commands, and a chaos-tested resilience proof you can reproduce on your own cluster.

| Stage | Architecture | TTFT (p50) | Throughput | Survives Pod Kill? |
|-------|-------------|-----------|------------|-------------------|
| 1. Baseline | vLLM Router (consistent hash) + 2 instances | 84s | 3,400 tok/s | No |
| 2. CPU Offload | + LMCache CPU (256 GB/node) | 33s (**2.5×**) | 6,400 tok/s | No |
| 3. Distributed | + Mooncake central store | 28s (**3×**) | 7,200 tok/s | **Yes** |

```{note}
All manifests are available in the [companion repository](https://github.com/your-org/vllm-distributed-kv-cache). You can also try the configuration interactively on [recipes.vllm.ai](https://recipes.vllm.ai/) — select DeepSeek-V3 → "Distributed KV Store (Mooncake)" or "Centralized KV Store (Mooncake)" strategy ([PR #500](https://github.com/vllm-project/recipes/pull/500)).
```

## Prerequisites

Before you begin, ensure you have:

- **Hardware**: 2× nodes with 8×AMD Instinct MI300X (192 GB HBM3e each), 512 GB DRAM per node, 100 Gbps private network
- **Software**: Kubernetes cluster, `kubectl` access, nodes labeled by hostname
- **Model**: `deepseek-ai/DeepSeek-V3.2` — 671B MoE (37B active), native FP8, TP=8 per node
- **Image**: `vllm/vllm-openai-rocm:nightly` (includes LMCache)
- **Benchmark tool**: [kv-cache-tester](https://github.com/callanjfox/kv-cache-tester) with 739 Claude Code agentic traces

The stack consists of:

- **[vLLM](https://github.com/vllm-project/vllm)** — inference engine (nightly ROCm build)
- **[vLLM Router](https://docs.vllm.ai/en/latest/serving/router.html)** — load balancer with consistent hashing and K8s service discovery
- **[LMCache](https://github.com/LMCache/LMCache)** — KV cache connector for CPU/remote offloading
- **[Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake)** — distributed KV cache store (TCP/RDMA)

## Stage 1: Session-Affinity Load Balancing

Start with a baseline: two vLLM instances behind a router using `consistent_hash` policy. This ensures requests from the same user always route to the same backend, maximizing local HBM prefix cache hits.

```
                    ┌──────────────────────────────────┐
                    │     vLLM Router (:8080)           │
                    │     consistent_hash routing       │
                    │     K8s service discovery         │
                    └──────────┬──────────┬────────────┘
                               │          │
              ┌────────────────▼──┐  ┌────▼────────────────┐
              │  vLLM Instance A  │  │  vLLM Instance B     │
              │  Node 1           │  │  Node 2              │
              │  8× MI300X, TP=8  │  │  8× MI300X, TP=8    │
              │  HBM prefix cache │  │  HBM prefix cache    │
              └───────────────────┘  └──────────────────────┘
```

### Configure the vLLM worker

Run vLLM with ROCm-optimized settings on each GPU node:

```bash
vllm serve deepseek-ai/DeepSeek-V3.2 \
  --tensor-parallel-size 8 \
  --enable-prefix-caching \
  --enable-expert-parallel \
  --async-scheduling \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --block-size 64 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
```

Set these AMD-specific environment variables on each worker:

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export NCCL_MIN_NCHANNELS=112
```

### Configure the router

Deploy the vLLM Router with consistent-hash routing and Kubernetes service discovery:

```bash
vllm-router \
  --host 0.0.0.0 --port 8080 \
  --service-discovery \
  --selector app=vllm-worker \
  --service-discovery-namespace vllm-serving \
  --service-discovery-port 8000 \
  --policy consistent_hash
```

The full Kubernetes manifest (namespace, RBAC, router, and worker pods) is in [`manifests/stage1-baseline.yaml`](manifests/stage1-baseline.yaml).

### Run the benchmark

Use kv-cache-tester to replay realistic agentic traces:

```bash
git clone https://github.com/callanjfox/kv-cache-tester.git && cd kv-cache-tester

python3 trace_replay_tester.py \
  --api-endpoint http://<router-ip>:30080 \
  --trace-directory traces \
  --start-users 8 --max-users 32 \
  --max-ttft 120.0 --test-duration 1200 \
  --max-context 65536 --warm-prefix-pct 0.5 \
  --timing-strategy think-only --recycle \
  --seed 42 --output-dir ./results/stage1-baseline
```

### Stage 1 results

Under 32 concurrent agentic users with 65K context:

- **TTFT median**: ~84s
- **Input throughput**: ~3,400 tok/s
- **Pod kill recovery**: Complete cache loss — affected users restart cold

Consistent hash helps by keeping users sticky to instances. But under load, HBM fills and evictions are permanent. When the hash ring rebalances (pod restart, scale event), all session affinity is lost and every affected user faces a full cold-start prefill.

## Stage 2: LMCache CPU KV Cache Offloading

Now add LMCache to extend your effective cache from HBM into CPU DRAM. Evicted KV blocks spill to a 256 GB CPU tier (~100μs retrieval) instead of being lost permanently.

```
              ┌────────────────────────────────────┐
              │        vLLM Instance               │
              │  ┌──────────────────────────────┐  │
              │  │ GPU HBM (L1) — FP8 KV cache  │  │
              │  │ ~80 GB effective              │  │
              │  └──────────┬───────────────────┘  │
              │             │ eviction             │
              │  ┌──────────▼───────────────────┐  │
              │  │ CPU DRAM (L2) — 256 GB        │  │
              │  │ LMCache local_cpu             │  │
              │  └──────────────────────────────┘  │
              └────────────────────────────────────┘
```

### What changes from Stage 1

Only three modifications to your worker spec:

```diff
# 1. Add KV transfer flag to vllm serve command:
+  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

# 2. Add LMCache environment variables:
+  PYTHONHASHSEED: "0"          # MANDATORY — cache key consistency across TP workers
+  LMCACHE_LOCAL_CPU: "True"
+  LMCACHE_MAX_LOCAL_CPU_SIZE: "256"   # GB
+  LMCACHE_CHUNK_SIZE: "256"           # tokens per chunk
+  LMCACHE_REMOTE_SERDE: "naive"
+  LMCACHE_NUMA_MODE: "auto"

# 3. Increase pod memory (accommodate 256 GB CPU cache pool):
-  memory: 900Gi
+  memory: 1400Gi
```

The full manifest is in [`manifests/stage2-lmcache-cpu.yaml`](manifests/stage2-lmcache-cpu.yaml).

### Critical details to get right

```{note}
**`PYTHONHASHSEED=0` is mandatory.** LMCache hashes token sequences for cache keys. Without deterministic hashing, TP workers produce different keys for identical prompts — resulting in 0% hit rate with zero errors in logs. This is the most common silent failure mode.
```

- **`LMCACHE_CHUNK_SIZE=256`** — Tokens per cache chunk. 256 balances granularity vs. metadata overhead for agentic workloads where conversations grow incrementally.
- **Memory sizing** — Ensure `LMCACHE_MAX_LOCAL_CPU_SIZE` + model staging + OS ≤ node RAM. For 512 GB nodes, 256 GB cache is safe.

### Stage 2 results

| Metric | Stage 1 | Stage 2 | Delta |
|--------|---------|---------|-------|
| TTFT median | 84s | **33s** | -61% |
| TTFT p95 | 207s | **150s** | -28% |
| Requests completed | 208 | **331** | +59% |
| Input throughput | 3,400 tok/s | **6,400 tok/s** | +88% |

HBM fills within minutes under load. Without LMCache, eviction means permanent loss and full prefill on next request. With LMCache, evicted blocks land in DRAM and get retrieved in ~100μs — the system self-heals under pressure instead of degrading monotonically.

**Remaining problem**: Caches are still per-instance. When a pod dies, its 256 GB of cached KV state dies with it.

## Stage 3: Distributed KV Cache with Mooncake

[Mooncake](https://github.com/kvcache-ai/Mooncake) adds a central KV store that all instances share. When any instance computes a prefix, the KV blocks are written to Mooncake and become available to every other instance. The store runs on a dedicated CPU node and outlives individual vLLM pods.

```
┌─────────────────────────────────────────────────────────────────┐
│                    vLLM Router (:8080)                           │
│               consistent_hash routing                           │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
┌──────────────▼──────────┐  ┌────────────▼──────────────┐
│  vLLM Instance A        │  │  vLLM Instance B           │
│  8× MI300X, TP=8        │  │  8× MI300X, TP=8           │
│  LMCache → Mooncake     │  │  LMCache → Mooncake        │
└──────────────┬──────────┘  └────────────┬──────────────┘
               │          TCP/RDMA         │
               └──────────┬────────────────┘
          ┌───────────────▼────────────────────┐
          │  Mooncake Central Store            │
          │  etcd (metadata) + master (alloc)  │
          │  + MooncakeDistributedStore (data) │
          │  Dedicated node — no GPU needed    │
          └────────────────────────────────────┘
```

### New components

Deploy these on a dedicated CPU node (300+ GB RAM, no GPU required):

- **etcd** — metadata registry (`quay.io/coreos/etcd:v3.6.1`)
- **mooncake_master** — allocation manager (port 50051)
- **MooncakeDistributedStore** — data storage node (256 GB DRAM, TCP or RDMA transport)
- **ClusterIP Service** — stable DNS for etcd + master

### What changes on vLLM workers

```diff
# 1. Replace individual LMCACHE env vars with a config file:
+  LMCACHE_CONFIG_FILE: /shared/lmcache_config.yaml
+  LMCACHE_USE_EXPERIMENTAL: "True"

# 2. Add initContainer that resolves local IP into config template:
+  initContainers:
+    - name: resolve-config
+      command: ["sh", "-c"]
+      args:
+        - |
+          IP=${POD_IP:-$(hostname -I | awk '{print $1}')}
+          sed "s/__LOCAL_IP__/$IP/g" /etc/lmcache/config.template.yaml > /shared/lmcache_config.yaml
```

### LMCache config template

Create this as a Kubernetes ConfigMap shared by all workers:

```yaml
chunk_size: 256
local_device: cpu
remote_url: "mooncakestore://mooncake.vllm-serving.svc.cluster.local:50051/"
remote_serde: naive
local_cpu: true
max_local_cpu_size: 256
extra_config:
  local_hostname: "__LOCAL_IP__"
  metadata_server: "etcd://mooncake.vllm-serving.svc.cluster.local:2379"
  protocol: tcp
  device_name: ""
  master_server_address: "mooncake.vllm-serving.svc.cluster.local:50051"
  global_segment_size: 274877906944   # ~256 GB
  local_buffer_size: 10737418240      # ~10 GB staging
  transfer_timeout: 30
```

### Mooncake store node

Run this Python process on the dedicated CPU node:

```python
from mooncake.store import MooncakeDistributedStore

store = MooncakeDistributedStore()
store.setup({
    "local_hostname": os.environ["POD_IP"],
    "metadata_server": "etcd://127.0.0.1:2379",
    "global_segment_size": str(256 * 1024**3),
    "local_buffer_size": str(4 * 1024**3),
    "protocol": "tcp",
    "rdma_devices": "",
    "master_server_addr": "127.0.0.1:50051",
})
# Runs until SIGTERM
```

The full manifest (Mooncake pod, ConfigMap, initContainer, workers, RBAC) is in [`manifests/stage3-distributed.yaml`](manifests/stage3-distributed.yaml).

### How it works

1. **Instance A** computes 50K tokens of prefill
2. LMCache stores KV locally (DRAM) **and** writes to Mooncake via TCP
3. Mooncake master registers keys; store node holds data
4. **Instance B** gets the next request from the same conversation (hash rebalance or pod restart)
5. LMCache checks Mooncake → hit → retrieves in ~5ms
6. Instance B skips 50K tokens of prefill

### Stage 3 results (steady state)

| Metric | Stage 2 | Stage 3 | Delta |
|--------|---------|---------|-------|
| TTFT median | 33s | **28s** | -15% |
| TTFT p95 | 150s | **95s** | -37% |
| Requests completed | 331 | **380** | +15% |
| Cross-instance hit rate | 0% | **72%** | New |

### Resilience proof: Chaos monkey

The real value of a distributed cache shows under failure. Deploy a chaos monkey that kills the oldest vLLM worker pod every 5 minutes during the benchmark and observe the difference.

With consistent-hash routing, each pod "owns" a set of users. When that pod dies, those users lose their local cache AND get reassigned to a different pod. Without distributed cache, they face a full cold-start. With Mooncake, the cache survives in the central store.

Deploy the chaos monkey (full version in [`manifests/chaos-monkey.yaml`](manifests/chaos-monkey.yaml)):

```yaml
containers:
  - name: chaos
    image: bitnami/kubectl:latest
    command: ["/bin/sh", "-c"]
    args:
      - |
        while true; do
          sleep ${KILL_INTERVAL_SECONDS}
          TARGET=$(kubectl get pods -l app=vllm-worker \
            --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[0].metadata.name}')
          if [ -n "$TARGET" ]; then
            echo "[$(date -u)] Killing: $TARGET"
            kubectl delete pod "$TARGET" --grace-period=0 --force
          fi
        done
    env:
      - name: KILL_INTERVAL_SECONDS
        value: "300"
```

Run the benchmark with chaos active:

```bash
# Start benchmark first
python3 trace_replay_tester.py \
  --api-endpoint http://<router-ip>:30080 \
  --trace-directory traces \
  --start-users 8 --max-users 32 \
  --max-ttft 120.0 --test-duration 1800 \
  --max-context 65536 --warm-prefix-pct 0.5 \
  --seed 42 --output-dir ./results/stage3-chaos

# Deploy chaos monkey while benchmark runs
kubectl apply -f manifests/chaos-monkey.yaml
```

**Results under chaos (pods dying every 5 min)**:

| Metric | Stage 3 (steady) | Stage 3 + chaos | Stage 2 + chaos |
|--------|-----------------|-----------------|-----------------|
| TTFT median | 28s | **31s** | 120s+ |
| TTFT p95 | 95s | **110s** | 230s+ |
| Requests completed | 380 | **355** | ~100 |
| Recovery after kill | — | **<30s** | **5-10 min** |

Stage 3 under continuous pod failures performs at ~90% of its steady-state. Stage 2 under the same chaos collapses to ~30%. The distributed cache transforms pod failures from catastrophic to routine.

## RDMA upgrade (optional)

TCP works everywhere (~5ms per KV transfer). If your cluster has InfiniBand or RoCE v2 NICs, upgrade to RDMA for <1ms transfers. The process involves four steps:

1. **Deploy the Kubernetes RDMA device plugin** — a DaemonSet that exposes `rdma/hca_shared` resources

2. **Discover RDMA interfaces** across all nodes (no SSH needed):

```bash
for node in $(kubectl get nodes -o name | sed 's#node/##'); do
  echo "===== ${node} ====="
  kubectl debug node/${node} -it --image=ubuntu -- chroot /host bash -lc 'ibdev2netdev' || true
done
```

3. **Deploy the plugin** ([`manifests/rdma-infra.yaml`](manifests/rdma-infra.yaml)):

```bash
# Update ifNames in the ConfigMap to match your interfaces, then:
kubectl apply -f manifests/rdma-infra.yaml

# Verify
kubectl get nodes -o json | \
  jq -r '.items[] | [.metadata.name, .status.allocatable["rdma/hca_shared"]] | @tsv'
```

4. **Update LMCache config and pod spec**:

```diff
# lmcache config:
-  protocol: tcp
-  device_name: ""
+  protocol: rdma
+  device_name: mlx5_1

# Pod resources:
+  rdma/hca_shared: 1

# Pod env:
+  MC_MTU: "4096"        # Use 1024 for RoCE v2 with ETH MTU < 4096
+  MC_GID_INDEX: "3"     # Match your cluster's RDMA GID index
```

| Transport | KV Transfer Latency | When to Use |
|-----------|--------------------|-|
| TCP | ~5ms | Default — works everywhere |
| RDMA (RoCE v2) | <1ms | Have Mellanox/Broadcom NICs |
| RDMA (InfiniBand) | <0.5ms | Dedicated HPC clusters |

## Troubleshooting

If you run into issues, check these common problems first:

| Symptom | Cause | Fix |
|---------|-------|-----|
| 0% cache hits on identical prompts | `PYTHONHASHSEED` not 0 | Set `PYTHONHASHSEED=0` on ALL pods |
| `batch_get_buffer` AttributeError | Mooncake < 0.3.6 | `pip install mooncake>=0.3.6` |
| RDMA `received packet mismatch` | Cross-device (mlx5_0 ↔ mlx5_1) | Pin `device_name: mlx5_1` or use TCP |
| `Failed to modify QP to RTR` | RDMA MTU mismatch | Set `MC_MTU=1024` on all nodes |
| LMCache slower than baseline | Python fallback on ROCm | Build: `BUILD_WITH_HIP=1 pip install -e .` |
| Mooncake keys visible but decode misses | Three-phase put timing | Increase `transfer_timeout` or fix RDMA path |

Verify your deployment is working end-to-end:

```bash
# 1. Mooncake store is receiving data
curl http://mooncake-ip:50051/metrics  # PutEnd > 0, PutRevoke = 0

# 2. LMCache is hitting
kubectl logs vllm-node1 | grep "hit tokens"  # Should show N > 0

# 3. Cross-instance cache works
kubectl delete pod vllm-node1 --force  # Kill a pod
# Send same prompt → surviving pod should show cache hit from Mooncake
```

## What's next

Interested in going further? Here are the natural next steps:

- **`MooncakeStoreConnector` (vLLM ≥ 0.21)** — A direct Mooncake connector without the LMCache intermediary. Two modes are available on [recipes.vllm.ai](https://recipes.vllm.ai/): *Distributed* (each vLLM node contributes CPU DRAM to a shared pool) and *Centralized* (a dedicated `mooncake_store_service` owns all cache capacity, survives vLLM restarts). Try it yourself via [PR #500](https://github.com/vllm-project/recipes/pull/500).
- **Multi-node Mooncake store** — Scale to >1 TB distributed cache with StatefulSet replicas
- **NVMe L3 tier** — Add `LMCACHE_LOCAL_DISK=/nvme/lmcache` for terabytes of local cache before hitting the network
- **Smart routing** — Session-aware routing that considers cache locality hints from Mooncake

## Summary

In this guide, you built a progressive KV cache architecture for DeepSeek-V3.2 on AMD Instinct MI300X that delivers 3× lower TTFT and survives continuous pod failures:

- **Stage 1** — Consistent-hash routing keeps users sticky to instances, maximizing local HBM cache hits (84s TTFT baseline)
- **Stage 2** — LMCache CPU offloading spills evicted KV blocks to 256 GB DRAM instead of losing them permanently (33s TTFT, 2.5× improvement)
- **Stage 3** — Mooncake distributed store shares cache across all instances and outlives pod failures (28s TTFT, 3× improvement, chaos-proof)

The chaos monkey test proved the critical difference: Stage 3 under continuous pod failures maintains ~90% of steady-state performance, while Stage 2 under the same conditions collapses to ~30%. A distributed KV cache transforms pod failures from catastrophic events into routine operations.

All manifests, configuration files, and benchmark commands are available in the [companion repository](https://github.com/your-org/vllm-distributed-kv-cache). Clone it, update the node hostnames, and `kubectl apply` to reproduce these results on your own AMD Instinct cluster.

## Disclaimers

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.

THIS INFORMATION IS PROVIDED 'AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

AMD, the AMD Arrow logo, AMD Instinct, ROCm, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2026 Advanced Micro Devices, Inc. All rights reserved.
