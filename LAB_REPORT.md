# LABORATORY EXPERIMENT REPORT

**Course Title:** Parallel and Grid Computing (PGC)  
**Experiment Title:** Performance Analysis of Matrix Multiplication using Sequential, OpenMP, MPI, and CUDA Paradigms  
**Author / Repository Owner:** `chhavi-2006`  
**Repository:** [https://github.com/chhavi-2006/PGC_lab](https://github.com/chhavi-2006/PGC_lab)  
**Date:** September 2026  

---

## 1. Abstract

This laboratory experiment evaluates the performance, speedup, and architectural trade-offs of four parallel computing models for a dense $4000 \times 4000$ matrix multiplication problem. We implemented: (1) a baseline single-threaded C program in WSL2 Ubuntu, (2) an OpenMP shared-memory multi-threaded program, (3) an Open MPI distributed-memory program across a 4-node VM cluster, and (4) a CUDA GPU kernel executed on an NVIDIA GPU. All implementations produced the verified mathematical result $C[0][0] = 4000.00$. Empirical results demonstrate that while the sequential baseline took **348.02 seconds**, OpenMP (8 threads) reduced execution to **132.46 seconds** ($2.63\times$ speedup), Open MPI (4 nodes) reduced execution to **92.98 seconds** ($3.74\times$ speedup), and CUDA GPU acceleration achieved an extraordinary **0.1650 seconds** total phase time ($2109.18\times$ speedup) and **0.1464 seconds** kernel execution time ($2376.51\times$ speedup).

---

## 2. Experimental Objectives

1. Program the dense $4000 \times 4000$ matrix multiplication algorithm ($C = A \times B$) across Sequential, OpenMP, Open MPI, and CUDA GPU paradigms.
2. Verify numerical accuracy by confirming $C[0][0] = 4000.00$ ($A[i][j]=1.0, B[i][j]=1.0$).
3. Measure wall-clock execution time and calculate speedup factors ($S = T_{\text{seq}} / T_{\text{par}}$) and throughput (GFLOPS).
4. Analyze performance limitations related to memory bandwidth, network latency, and SIMT thread scheduling.

---

## 3. System Architecture & Resource Allocation

| Paradigm | Environment | Compute Units | Memory Architecture | Compiler / Toolchain |
| :--- | :--- | :--- | :--- | :--- |
| **Sequential** | WSL2 Ubuntu 22.04 | 1 CPU Core | Single Memory Space | GCC `-O2` |
| **OpenMP** | WSL2 Ubuntu 22.04 | 8 CPU Threads | Shared System RAM | GCC `-O2 -fopenmp` |
| **Open MPI** | 4-Node VM Cluster | 4 VM Processes | Distributed Memory | Open MPI `mpicc -O2` |
| **CUDA GPU** | NVIDIA GPU System | 16,000,000 Threads | Dedicated VRAM | NVIDIA `nvcc -O2` |

---

## 4. Empirical Data & Benchmarking Results

### 4.1 Performance Summary Table

| Computing Model | Resources | Execution Time (s) | Speedup vs Sequential | Throughput (GFLOPS) | Verification $C[0][0]$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Sequential Baseline** | 1 CPU Core | **348.023990** | **1.00×** | **0.37 GFLOPS** | **4000.00** |
| **OpenMP Shared Memory**| 8 Threads | **132.457362** | **2.63×** | **0.97 GFLOPS** | **4000.00** |
| **MPI Distributed** | 4 VM Cluster Nodes | **92.979510** | **3.74×** | **1.38 GFLOPS** | **4000.00** |
| **CUDA GPU (Kernel)** | NVIDIA GPU | **0.146443** | **2376.51×** | **874.06 GFLOPS** | **4000.00** |
| **CUDA GPU (Total Phase)**| NVIDIA GPU | **0.165004** | **2109.18×** | **775.74 GFLOPS** | **4000.00** |

---

## 5. Performance Visualizations

1. **Figure 1 (`images/1_sequential_execution.jpeg`)**: Sequential execution terminal output ($348.023990$s).
2. **Figure 2 (`images/2_openmp_execution.jpeg`)**: OpenMP execution terminal output ($132.457362$s).
3. **Figure 3 (`images/3_openmp_verification.jpeg`)**: OpenMP verification screenshot.
4. **Figure 4 (`images/4_htop_resource_monitor.jpeg`)**: `htop` terminal process monitor during 8-thread execution.
5. **Figure 5 (`images/5_sequential_verification.jpeg`)**: Sequential output verification screenshot.
6. **Figure 6 (`images/execution_time_comparison.png`)**: Logarithmic scale execution time comparison.
7. **Figure 7 (`images/speedup_comparison.png`)**: Speedup factor comparison chart.
8. **Figure 8 (`images/overall_performance_dashboard.png`)**: 4-panel comprehensive performance dashboard.

---

## 6. Discussion & Technical Findings

1. **Sequential Complexity**: Performing $2 \times 4000^3 = 128 \times 10^9$ FLOPs on a single CPU thread incurs severe cache eviction and memory stall cycles.
2. **OpenMP Shared Memory**: Distributing outer loops across 8 threads speeds up computation by $2.63\times$, limited by memory bus contention across shared RAM.
3. **MPI Cluster Scaling**: MPI distributes rows across 4 independent VM nodes. Dedicated memory spaces on each node mitigate memory bus bottlenecks, yielding $3.74\times$ speedup despite network latency.
4. **CUDA GPU Massively Parallel Execution**: CUDA divides matrix evaluation into $62,500$ blocks of $256$ threads ($16,000,000$ threads), achieving an outstanding $2109.18\times$ total phase speedup.

---

## 7. Conclusion

The experiment demonstrates the superiority of CUDA GPU parallelism for compute-intensive matrix operations, delivering a **2109.18× speedup** over the sequential baseline. MPI distributed computing provides scalability across physical nodes, while OpenMP offers easy multi-core parallelization for shared-memory environments.
