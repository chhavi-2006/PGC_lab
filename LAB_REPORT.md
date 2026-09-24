# LABORATORY EXPERIMENT REPORT

**Course Title:** Parallel and Grid Computing (PGC)  
**Experiment Title:** Performance Analysis of Matrix Multiplication using Sequential, OpenMP, MPI, and CUDA Paradigms in C and C++  
**Author / Repository Owner:** `chhavi-2006`  
**Repository:** [https://github.com/chhavi-2006/PGC_lab](https://github.com/chhavi-2006/PGC_lab)  
**Date:** September 2026  

---

## 1. Abstract

This laboratory experiment evaluates the performance, speedup, and architectural trade-offs of four parallel computing models for a dense $4000 \times 4000$ matrix multiplication problem, implemented in both C and C++. We implemented: (1) a baseline single-threaded C/C++ program in WSL2 Ubuntu, (2) an OpenMP shared-memory multi-threaded C/C++ program, (3) an Open MPI distributed-memory C/C++ program across a 4-node VM cluster, and (4) a CUDA GPU kernel executed on an NVIDIA GPU. All implementations produced the verified mathematical result $C[0][0] = 4000.00$. Empirical results demonstrate that while the sequential baseline took **348.02 seconds**, OpenMP (8 threads) reduced execution to **132.46 seconds** ($2.63\times$ speedup), Open MPI (4 nodes) reduced execution to **92.98 seconds** ($3.74\times$ speedup), and CUDA GPU acceleration achieved an extraordinary **0.1650 seconds** total phase time ($2109.18\times$ speedup) and **0.1464 seconds** kernel execution time ($2376.51\times$ speedup).

---

## 2. Experimental Objectives

1. Program the dense $4000 \times 4000$ matrix multiplication algorithm ($C = A \times B$) in both C and C++ across Sequential, OpenMP, Open MPI, and CUDA GPU paradigms.
2. Verify numerical accuracy by confirming $C[0][0] = 4000.00$ ($A[i][j]=1.0, B[i][j]=1.0$).
3. Measure wall-clock execution time and calculate speedup factors ($S = T_{\text{seq}} / T_{\text{par}}$) and throughput (GFLOPS).
4. Analyze performance limitations related to memory bandwidth, network latency, and SIMT thread scheduling.

---

## 3. System Architecture & Source Code Matrix

| Paradigm | C Source File | C++ Source File | Compute Units | Compiler / Toolchain |
| :--- | :--- | :--- | :--- | :--- |
| **Sequential** | `src/matrix_sequential.c` | `src/matrix_sequential.cpp` | 1 CPU Core | `gcc` / `g++` `-O2` |
| **OpenMP** | `src/matrix_openmp.c` | `src/matrix_openmp.cpp` | 8 CPU Threads | `gcc` / `g++` `-O2 -fopenmp` |
| **Open MPI** | `src/matrix_mpi.c` | `src/matrix_mpi.cpp` | 4 VM Processes | `mpicc` / `mpicxx` `-O2` |
| **CUDA GPU** | `src/matrix_cuda.cu` | `src/matrix_cuda.cpp` | 16,000,000 Threads | NVIDIA `nvcc -O2` |

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

### 6.1 Algorithmic Complexity & Operational Intensity
Matrix multiplication of two $4000 \times 4000$ matrices requires $2 N^3 = 128 \times 10^9$ floating-point operations ($128 \text{ GFLOPs}$). The operational intensity $I = N / 12 = 333.33 \text{ FLOPs/byte}$ indicates that the workload is compute-heavy when data resides in L1/L2 caches, but turns into a memory bandwidth bottleneck when cache eviction forces main RAM fetches.

### 6.2 Cache Locality & Serial CPU Execution
In the standard $i-j-k$ loop structure, Matrix A is accessed with unit stride ($\text{stride-}1$), while Matrix B is accessed with non-unit stride ($\text{stride-}N = 32,000$ bytes stride). This non-contiguous access pattern generates severe L1/L2 cache misses on single CPU cores, resulting in a low baseline performance of $0.37 \text{ GFLOPS}$ ($348.02 \text{ seconds}$).

### 6.3 Shared Memory Contention in OpenMP
OpenMP parallelizes the outer loop across $8$ CPU threads. While the workload is evenly divided ($500$ iterations per thread), all 8 cores compete for the same socket's shared L3 cache and memory bus channels. Memory bus saturation limits OpenMP speedup to $2.63\times$ ($132.46 \text{ seconds}$) versus the theoretical $8.0\times$ linear peak.

### 6.4 Distributed Memory Scaling in MPI
MPI assigns $1000$ matrix rows to each node across a 4-node VM cluster. Because each virtual machine runs with its own isolated guest RAM controller, memory bus contention is significantly reduced compared to OpenMP. Despite TCP/IP inter-node communications (`MPI_Scatter` and `MPI_Gather` totaling $128 \text{ MB}$ data), the $O(N^3)$ computational cost dominates the $O(N^2)$ communication cost by $1000:1$, allowing MPI to achieve $3.74\times$ speedup ($92.98 \text{ seconds}$, $93.50\%$ parallel efficiency).

### 6.5 SIMT Massively Parallel Acceleration in CUDA GPU
The CUDA architecture maps the computation across $62,500$ thread blocks of $256$ threads each ($16,000,000$ active logical threads). The NVIDIA GPU achieves a $2109.18\times$ speedup ($0.1650 \text{ seconds}$ total phase time, $874.06 \text{ GFLOPS}$ kernel performance) due to:
- Hardware-level SIMT warp execution ($32$ threads per warp).
- Automatic global VRAM memory coalescing across adjacent threads.
- Zero-cycle warp context switching for memory latency hiding.

---

## 7. Conclusion

The experiment demonstrates the superiority of CUDA GPU parallelism for compute-intensive matrix operations, delivering a **2109.18× speedup** over the sequential baseline. MPI distributed computing provides scalability across physical nodes, while OpenMP offers easy multi-core parallelization for shared-memory environments.
