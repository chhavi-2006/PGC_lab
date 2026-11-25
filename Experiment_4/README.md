# Experiment 4: Cache Coherence Protocols (MSI, MESI, MOESI) & NUMA Memory Performance

---

## Executive Summary

This experiment investigates how hardware cache coherence protocols, cache-line alignment, and Non-Uniform Memory Access (NUMA) page placement govern multi-core memory performance. We combine a 2-core snooping coherence simulator (**MSI**, **MESI**, and **MOESI** over $1,000$ iterations / $4,000$ memory operations) with empirical OpenMP C benchmarks on Ubuntu Linux measuring **false sharing** across $100,000,000$ counter increments and **512 MB NUMA memory sweeps** (4 passes = $2.0\text{ GB}$ read) under default and `numactl` node-bound policies.

```mermaid
flowchart LR
    subgraph PartA ["1. Coherence Protocol Simulation"]
        direction TB
        M1["MSI: 4,000 Bus Tx<br/>(1,999 Writebacks, 2,000 Upgrades)"]
        M2["MESI: 3,999 Bus Tx<br/>(Silent E->M Upgrade, 1,999 Writebacks)"]
        M3["MOESI: 3,999 Bus Tx<br/>(0 Writebacks, 1,999 C2C Transfers)"]
    end

    subgraph PartB ["2. False Sharing Benchmark (100M)"]
        direction TB
        FS1["Adjacent (Stride = 1)<br/>0.0191s at 4 Threads<br/>Cache-Line Ping-Pong"]
        FS2["Padded (Stride = 8 / 64B)<br/>0.0103s at 4 Threads<br/>1.85x Faster Execution"]
    end

    subgraph PartC ["3. NUMA 512 MB Sweep (4 Passes)"]
        direction TB
        N1["Default Placement (compare_numa)<br/>2 Threads: 0.1097s (18.23 GB/s)"]
        N2["Local Node 0 Bound (numactl)<br/>2 Threads: 0.0526s (38.02 GB/s)"]
    end

    PartA --> PartB --> PartC
```

### Key Findings

> 1. **Zero DRAM Writebacks under MOESI**: In the adjacent-layout simulation, **MOESI eliminates all 1,999 DRAM writebacks** incurred by MSI and MESI by transitioning the modifying core to the **Owned ($O$)** state and servicing peer reads directly via **Cache-to-Cache ($C2C$) transfers**. Separating counters onto distinct cache lines (*padded layout*) drops bus transactions from ~4,000 down to **2–4 total transactions**.
> 2. **Eliminating False Sharing with 64-Byte Padding**: When threads update adjacent `long` counters (`stride = 1`), cross-core invalidations stall scaling at 4 threads (`0.0154s` at 2T $\to$ `0.0191s` at 4T). Isolating each thread's counter on its own 64-byte cache line (`stride = 8`) keeps runtime scaling downward to **0.0103s at 4 threads**—achieving up to a **1.85× speedup** over the adjacent layout.
> 3. **Local NUMA Node Affinity Boosts Memory Throughput**: Under default OS placement in `compare_numa`, the 512 MB 4-pass sweep reaches **17.33–17.64 GB/s** (1T) and **18.23–18.52 GB/s** (2T). Pinning execution and first-touch allocation to local node 0 (`numactl --cpunodebind=0 --membind=0 ./numa_sweep`) improves single-thread latency to **0.1027s (19.47 GB/s)** and scales 2-thread bandwidth up to **38.02 GB/s (`0.0526s`)**.

---

## Table of Contents

1. [Experiment Objectives](#1-experiment-objectives)
2. [Microarchitectural Background](#2-microarchitectural-background)
   - [2.1 Snooping Cache Coherence](#21-snooping-cache-coherence)
   - [2.2 MSI vs. MESI vs. MOESI States](#22-msi-vs-mesi-vs-moesi-states)
   - [2.3 False Sharing Across 64-Byte Cache Lines](#23-false-sharing-across-64-byte-cache-lines)
   - [2.4 NUMA Affinity & First-Touch Allocation](#24-numa-affinity--first-touch-allocation)
3. [Source Code Overview](#3-source-code-overview)
4. [Empirical Results & Terminal Screenshots](#4-empirical-results--terminal-screenshots)
   - [4.1 Coherence Simulator State Traces & Totals](#41-coherence-simulator-state-traces--totals)
   - [4.2 Standalone False Sharing & `numactl` NUMA Sweep Runs](#42-standalone-false-sharing--numactl-numa-sweep-runs)
   - [4.3 Unified `compare_numa` Benchmark — Run 1 & Standalone Checks](#43-unified-compare_numa-benchmark--run-1--standalone-checks)
   - [4.4 Unified `compare_numa` Benchmark — Run 2](#44-unified-compare_numa-benchmark--run-2)
5. [Performance Visualizations](#5-performance-visualizations)
6. [Technical Discussion](#6-technical-discussion)
7. [Build & Execution Guide](#7-build--execution-guide)
8. [Conclusions](#8-conclusions)

---

## 1. Experiment Objectives

- **Coherence Protocol Modeling**: Simulate 2-core snooping coherence under **MSI**, **MESI**, and **MOESI**, comparing bus reads (`BusRd`), upgrades (`BusUpgr`), invalidations, DRAM writebacks, silent $E \to M$ upgrades, and cache-to-cache transfers for shared vs. isolated cache lines.
- **False Sharing Measurement**: Quantify cache-line ping-pong overhead when concurrent OpenMP threads increment adjacent 8-byte counters, and verify performance recovery using 64-byte cache-line padding.
- **NUMA Bandwidth Profiling**: Benchmark a 512 MB memory sweep across 4 sequential passes (`2.0 GB` total read) to evaluate default memory placement against explicit local-node binding with `numactl`.

---

## 2. Microarchitectural Background

### 2.1 Snooping Cache Coherence

Each CPU core maintains private L1/L2 caches. When multiple cores cache copies of the same physical address block, a **snooping coherence protocol** broadcasts read and ownership requests over the shared interconnect so every cache controller observes conflicting writes and preserves a single consistent view of memory.

```mermaid
flowchart TD
    subgraph MultiCore ["Multi-Core Processor"]
        C0["Core 0: Private L1/L2"]
        C1["Core 1: Private L1/L2"]
        C2["Core 2: Private L1/L2"]
        C3["Core 3: Private L1/L2"]
    end

    Bus[["Shared Interconnect / Snooping Bus"]]
    LLC["Shared L3 Cache"]
    DRAM["Main Memory (DRAM)"]

    C0 <--> Bus
    C1 <--> Bus
    C2 <--> Bus
    C3 <--> Bus
    Bus <--> LLC <--> DRAM
```

### 2.2 MSI vs. MESI vs. MOESI States

| State | Protocols | Meaning & Bus Behavior | Clean w.r.t. DRAM? | Coexists in Other Caches? |
| :--- | :--- | :--- | :---: | :---: |
| **M** (Modified) | MSI, MESI, MOESI | Sole valid copy with uncommitted writes; must supply data on peer read. | No (Dirty) | No |
| **O** (Owned) | MOESI | Holds dirty line while allowing shared readers; services peer reads via $C2C$ without DRAM writeback. | No (Dirty) | Yes (as Shared) |
| **E** (Exclusive) | MESI, MOESI | Sole valid clean copy; subsequent write upgrades silently ($E \to M$) with zero bus traffic. | Yes (Clean) | No |
| **S** (Shared) | MSI, MESI, MOESI | Valid read-only copy shared across one or more caches; writes require `BusUpgr`. | Yes (or Owned) | Yes |
| **I** (Invalid) | MSI, MESI, MOESI | Cache line absent or invalidated; any access triggers a miss. | N/A | N/A |

### 2.3 False Sharing Across 64-Byte Cache Lines

Coherence hardware tracks ownership at the **64-byte cache-line granularity** rather than per variable:

```text
+---------------------------------------------------------------+
|                      64-Byte Cache Line                       |
+---------------+---------------+---------------+---------------+
|  Counter 0    |  Counter 1    |  Counter 2    |  Counter 3    |
|   (Thread 0)  |   (Thread 1)  |   (Thread 2)  |   (Thread 3)  |
|    8 Bytes    |    8 Bytes    |    8 Bytes    |    8 Bytes    |
+---------------+---------------+---------------+---------------+
|<------------------- CONTENDED SHARED LINE ------------------->|
```

- **Adjacent (`stride = 1`)**: Independent thread counters occupy consecutive 8-byte slots inside one 64-byte line. Each write invalidates the line in peer cores, causing continuous cache misses (*ping-ponging*).
- **Padded (`stride = 8`)**: Spacing each counter by 8 `long` elements ($8 \times 8 = 64\text{ bytes}$) isolates every thread onto its own cache line so each core keeps its line in state $M$ without invalidations.

### 2.4 NUMA Affinity & First-Touch Allocation

On NUMA architectures, memory latency and bandwidth depend on whether pages reside on the local socket's memory controller or a remote socket across an interconnect. Under Linux's **first-touch policy**, physical pages are mapped to the NUMA node of the thread that first writes to them (`memset`), making `numactl --cpunodebind=0 --membind=0` effective at locking both compute threads and pages to node 0.

---

## 3. Source Code Overview

All source implementations and build rules are located in [`src/`](src/) and follow the lab manual in [`docs/Exp_4_Cache_Coherence_NUMA.pdf`](docs/Exp_4_Cache_Coherence_NUMA.pdf):

| Component | Source File | Purpose |
| :--- | :--- | :--- |
| **Coherence Simulator** | [`src/coherence_sim.py`](src/coherence_sim.py) | 2-core MSI, MESI, and MOESI state-machine and bus-traffic simulator |
| **False Sharing Test** | [`src/false_sharing.c`](src/false_sharing.c) | OpenMP benchmark comparing `adjacent` vs. `padded` counter strides |
| **NUMA Memory Sweep** | [`src/numa_sweep.c`](src/numa_sweep.c) | 512 MB 4-pass sequential cache-line read bandwidth benchmark |
| **Unified Benchmark** | [`src/compare_numa.c`](src/compare_numa.c) | Combined benchmark outputting False Sharing (Table 1) and Sweep (Table 2) |
| **Build Configuration** | [`src/Makefile`](src/Makefile) | Compiles C benchmarks with `gcc -O2 -fopenmp` |

---

## 4. Empirical Results & Terminal Screenshots

### 4.1 Coherence Simulator State Traces & Totals

Running `python3 coherence_sim.py 1000` (simulating 1,000 iterations / 4,000 memory operations) across adjacent and padded layouts:

| Layout | Protocol | Bus Transactions | BusRd | BusRdX | BusUpgr | Invalidations | Memory Writebacks | Cache-to-Cache ($C2C$) | Silent $E \to M$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adjacent** | **MSI** | `4000` | `2000` | `0` | `2000` | `1999` | `1999` | `0` | `0` |
| **Adjacent** | **MESI** | `3999` | `2000` | `0` | `1999` | `1999` | `1999` | `0` | `1` |
| **Adjacent** | **MOESI** | `3999` | `2000` | `0` | `1999` | `1999` | **`0`** | **`1999`** | `1` |
| **Padded** | **MSI** | `4` | `2` | `0` | `2` | `0` | `0` | `0` | `0` |
| **Padded** | **MESI** | `2` | `2` | `0` | `0` | `0` | `0` | `0` | `2` |
| **Padded** | **MOESI** | `2` | `2` | `0` | `0` | `0` | `0` | `0` | `2` |

![Coherence Protocol Simulation](images/01_coherence_sim_trace_and_stats.jpg)

---

### 4.2 Standalone False Sharing & `numactl` NUMA Sweep Runs

Compilation and standalone runs of `false_sharing` and `numa_sweep` pinned to NUMA node 0 (`numactl --cpunodebind=0 --membind=0`):

| Program | Configuration | Measured Time (s) | Effective Bandwidth | Observation |
| :--- | :--- | :---: | :---: | :--- |
| **`false_sharing`** | 1 Thread (`padded`) | `0.0253 s` | - | Single-thread baseline |
| **`false_sharing`** | 2 Threads (`adjacent`) | `0.0159 s` / `0.0183 s` | - | Cross-core invalidation overhead |
| **`false_sharing`** | 2 Threads (`padded`) | `0.0141 s` / `0.0146 s` | - | Up to **1.25× faster** than adjacent |
| **`numa_sweep` (512 MB)** | 1 Thread (Node 0 Bound) | `0.1027 s` | **19.47 GB/s** | Local node single-thread sweep (`sum=33554432`) |
| **`numa_sweep` (512 MB)** | 2 Threads (Node 0 Bound) | `0.0526 s` | **38.02 GB/s** | Dual-thread local node scaling (`sum=33554432`) |

![False Sharing and NUMA Sweep Runs](images/02_false_sharing_numa_standalone_runs.jpg)

---

### 4.3 Unified `compare_numa` Benchmark — Run 1 & Standalone Checks

Standalone checks (`2T adjacent = 0.0162 s`, `2T padded = 0.0115 s` -> **1.41× faster**, `2T numa_sweep` Node 0 bound = `0.1012 s` / **19.76 GB/s**) followed by the first `./compare_numa` run:

#### Table 1: False Sharing Benchmark ($100,000,000$ Increments — Run 1)

| Layout | Threads ($P$) | Final Count | Status | Measured Time (s) | Padded Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **adjacent** | 1 | `100000000` | Correct | `0.0262 s` | Baseline |
| **padded** | 1 | `100000000` | Correct | `0.0235 s` | 1.11× Faster |
| **adjacent** | 2 | `100000000` | Correct | `0.0222 s` | Contention |
| **padded** | 2 | `100000000` | Correct | `0.0178 s` | 1.25× Faster |
| **adjacent** | 4 | `100000000` | Correct | `0.0179 s` | Contention |
| **padded** | 4 | `100000000` | Correct | `0.0129 s` | **1.39× Faster** |

#### Table 2: 512 MB Memory Sweep (4 Passes = $2.0\text{ GB}$ — Run 1)

| Active Threads | Measured Time (s) | Effective Memory Bandwidth (GB/s) |
| :---: | :---: | :---: |
| **1 Thread** | `0.1134 s` | **17.64 GB/s** |
| **2 Threads** | `0.1080 s` | **18.52 GB/s** |

![compare_numa Execution Run 1](images/03_compare_numa_execution_and_sweeps.jpg)

---

### 4.4 Unified `compare_numa` Benchmark — Run 2

Second `./compare_numa` execution highlighting peak 4-thread false-sharing divergence:

#### Table 1: False Sharing Benchmark ($100,000,000$ Increments — Run 2)

| Layout | Threads ($P$) | Final Count | Status | Measured Time (s) | Padded Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **adjacent** | 1 | `100000000` | Correct | `0.0256 s` | Baseline |
| **padded** | 1 | `100000000` | Correct | `0.0217 s` | 1.18× Faster |
| **adjacent** | 2 | `100000000` | Correct | `0.0154 s` | Contention |
| **padded** | 2 | `100000000` | Correct | `0.0149 s` | 1.03× Faster |
| **adjacent** | 4 | `100000000` | Correct | `0.0191 s` | Slowdown (Line Ping-Pong) |
| **padded** | 4 | `100000000` | Correct | `0.0103 s` | **1.85× Faster (Optimal Scaling)** |

#### Table 2: 512 MB Memory Sweep (4 Passes = $2.0\text{ GB}$ — Run 2)

| Active Threads | Measured Time (s) | Effective Memory Bandwidth (GB/s) |
| :---: | :---: | :---: |
| **1 Thread** | `0.1154 s` | **17.33 GB/s** |
| **2 Threads** | `0.1097 s` | **18.23 GB/s** |

![compare_numa Execution Run 2](images/04_compare_numa_multithread_and_numactl.jpg)

---

## 5. Performance Visualizations

### 5.1 Cache Coherence Bus Transactions & Memory Traffic
![Cache Coherence Bus Traffic](images/cache_coherence_bus_traffic.png)

---

### 5.2 False Sharing Execution Time & Padded Speedup
![False Sharing Scaling](images/false_sharing_scaling.png)

---

### 5.3 NUMA Memory Sweep Bandwidth (Default vs. Node 0 Bound)
![NUMA Memory Sweep](images/numa_sweep_latency.png)

---

### 5.4 Consolidated 4-Panel Performance Dashboard
![Master Performance Dashboard](images/performance_comparison_dashboard.png)

---

## 6. Technical Discussion

1. **MOESI Eliminates Dirty Writeback Traffic**: Under MSI and MESI, reading a dirty cache line owned by Core 0 forces a flush to DRAM (`1,999` writebacks over 1,000 iterations). MOESI introduces the **Owned ($O$)** state, allowing Core 0 to retain dirty ownership and serve Core 1 directly via **Cache-to-Cache ($C2C$) transfers** (`1,999 C2C`, `0` writebacks). Meanwhile, MESI and MOESI eliminate `BusUpgr` on private lines via silent $E \to M$ upgrades (`2` total bus transactions in the padded layout vs. `4` in MSI).
2. **False Sharing Bottleneck Under Scaling**: In Run 2 of `./compare_numa`, scaling `adjacent` from 2 to 4 threads causes runtime to regress from `0.0154s` up to `0.0191s` (+24.0% slower) as four cores repeatedly invalidate the shared 64-byte cache line. With 64-byte padding (`padded`), runtime steadily drops from `0.0217s` (1T) $\to$ `0.0149s` (2T) $\to$ `0.0103s` (4T), delivering a **1.85× speedup** at 4 threads.
3. **NUMA Binding & Memory Bandwidth**: Default `compare_numa` sweeps achieve `17.33–17.64 GB/s` (1T) and `18.23–18.52 GB/s` (2T). Explicit node-0 CPU and memory affinity (`numactl --cpunodebind=0 --membind=0`) eliminates remote placement overhead, boosting `numa_sweep` throughput to **19.47 GB/s** (1T) and up to **38.02 GB/s** (`0.0526s`) at 2 threads.

---

## 7. Build & Execution Guide

```bash
# Build C benchmarks
cd Experiment_4/src
make all

# 1. Run cache coherence simulator
python3 coherence_sim.py 1000

# 2. Run standalone false sharing & NUMA sweep tests
OMP_NUM_THREADS=2 ./false_sharing adjacent
OMP_NUM_THREADS=2 ./false_sharing padded
OMP_NUM_THREADS=2 numactl --cpunodebind=0 --membind=0 ./numa_sweep

# 3. Run unified benchmark suite
./compare_numa

# 4. Regenerate charts
cd ..
python3 scripts/generate_charts.py
```

---

## 8. Conclusions

- **Prefer MOESI / C2C Transfers for Producer-Consumer Sharing**: Servicing dirty reads from an `Owned` cache avoids saturating DRAM writeback bandwidth.
- **Align Per-Thread Counters to 64 Bytes**: Padding thread-local accumulators (`stride = 8` for 8-byte `long`s or `alignas(64)`) removes cache-line invalidation traffic and restores multi-core scaling (**1.85× speedup** at 4 threads).
- **Bind Memory & Threads on NUMA Systems**: Pairing Linux's first-touch initialization with `numactl --cpunodebind=0 --membind=0` keeps memory accesses local and maximizes sustained sweep bandwidth.
