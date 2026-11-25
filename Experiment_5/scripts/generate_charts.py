import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('images', exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial']

# Empirical measurements from Experiment 5 run on NVIDIA RTX 5060 Ti
cpu_time = 1.154830          # ms
gpu_kernel_time = 0.027536   # ms
gpu_e2e_time = 0.114065      # ms
transfer_time = gpu_e2e_time - gpu_kernel_time  # 0.086529 ms

kernel_speedup = 41.938918   # x
e2e_speedup = 10.124315      # x

# Muted engineering palette
c_cpu = '#B85D43'     # Muted Terracotta
c_e2e = '#2B5C8F'     # Deep Slate Blue
c_kern = '#387864'    # Muted Sage Teal
c_io = '#64748B'      # Slate Gray
c_dark = '#1E293B'    # Slate Dark
c_gray = '#475569'    # Slate Mid

colors = [c_cpu, c_e2e, c_kern]
speedup_colors = [c_kern, c_e2e]

# ============================================================================
# 1. Consolidated 3-Panel Benchmark Dashboard
# ============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.4), facecolor='#FAFBFC')

categories = ['CPU Baseline\n(Single-Core C++)', 'GPU End-to-End\n(Compute + PCIe)', 'GPU Kernel-Only\n(CUDA Compute)']
times = [cpu_time, gpu_e2e_time, gpu_kernel_time]
bars1 = ax1.bar(categories, times, color=colors, width=0.5, edgecolor=c_dark, linewidth=1.2)
ax1.set_yscale('log')
ax1.set_ylim(0.01, 4.0)
ax1.set_title('Forward Pass Execution Latency (Log Scale)', fontsize=12, fontweight='bold', color=c_dark, pad=12)
ax1.set_ylabel('Execution Time (ms)', fontsize=10.5, fontweight='bold', color=c_gray)
ax1.grid(True, which='both', linestyle=':', color='#E2E8F0', alpha=0.9)
ax1.set_facecolor('#FFFFFF')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar, t in zip(bars1, times):
    yval = bar.get_height()
    label = f'{t:.3f} ms' if t >= 0.1 else f'{t:.4f} ms'
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval * 1.25, label,
             ha='center', va='bottom', fontsize=10, fontweight='bold', color=c_dark,
             bbox=dict(boxstyle='round,pad=0.25', fc='#F8FAFC', ec='#CBD5E1', lw=1))

speedup_cats = ['Kernel-Only\n(Compute Speedup)', 'End-to-End\n(With PCIe Transfers)']
speedups = [kernel_speedup, e2e_speedup]
bars2 = ax2.bar(speedup_cats, speedups, color=speedup_colors, width=0.45, edgecolor=c_dark, linewidth=1.2)
ax2.set_title('GPU Speedup vs. CPU Baseline', fontsize=12, fontweight='bold', color=c_dark, pad=12)
ax2.set_ylabel('Speedup Factor (x)', fontsize=10.5, fontweight='bold', color=c_gray)
ax2.set_ylim(0, 50)
ax2.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax2.set_facecolor('#FFFFFF')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for bar, s in zip(bars2, speedups):
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{s:.2f}x',
             ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=c_dark,
             bbox=dict(boxstyle='round,pad=0.25', fc='#F8FAFC', ec='#CBD5E1', lw=1))

kernel_pct = (gpu_kernel_time / gpu_e2e_time) * 100
transfer_pct = (transfer_time / gpu_e2e_time) * 100

breakdown_labels = [f'PCIe Transfers\n({transfer_pct:.1f}%)', f'CUDA Kernels\n({kernel_pct:.1f}%)']
breakdown_sizes = [transfer_time, gpu_kernel_time]
breakdown_colors = [c_io, c_kern]

wedges, texts, autotexts = ax3.pie(
    breakdown_sizes, labels=breakdown_labels, autopct='%1.1f%%',
    startangle=140, colors=breakdown_colors,
    wedgeprops=dict(edgecolor=c_dark, linewidth=1.2),
    textprops=dict(fontweight='bold', color=c_dark, fontsize=10)
)
for at in autotexts:
    at.set_color('#FFFFFF')
    at.set_fontsize(10.5)
ax3.set_title('End-to-End GPU Time Breakdown\n(PCIe I/O vs. Kernel Execution)', fontsize=12, fontweight='bold', color=c_dark, pad=12)

plt.suptitle('Experiment 5: Neural Network Layers in CUDA — Benchmark Summary',
             fontsize=14, fontweight='bold', color=c_dark, y=1.02)
plt.tight_layout()
plt.savefig('images/performance_comparison_dashboard.png', dpi=200, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. Standalone Execution Time Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(8.8, 5.2), facecolor='#FAFBFC')
bars = ax.bar(categories, times, color=colors, width=0.46, edgecolor=c_dark, linewidth=1.2)
ax.set_yscale('log')
ax.set_ylim(0.01, 4.0)
ax.set_title('Two-Layer Forward Pass Execution Latency (Log Scale)', fontsize=13, fontweight='bold', color=c_dark, pad=14)
ax.set_ylabel('Average Time per Forward Pass (ms)', fontsize=10.5, fontweight='bold', color=c_gray)
ax.grid(True, which='both', linestyle=':', color='#E2E8F0', alpha=0.9)
ax.set_facecolor('#FFFFFF')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, t in zip(bars, times):
    yval = bar.get_height()
    label = f'{t:.4f} ms' if t < 0.1 else f'{t:.3f} ms'
    ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.25, label,
            ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=c_dark,
            bbox=dict(boxstyle='round,pad=0.25', fc='#F8FAFC', ec='#CBD5E1', lw=1))

plt.tight_layout()
plt.savefig('images/execution_time_comparison.png', dpi=200, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. Standalone Speedup Analysis
# ============================================================================
fig, ax = plt.subplots(figsize=(8.2, 5.2), facecolor='#FAFBFC')
bars = ax.bar(speedup_cats, speedups, color=speedup_colors, width=0.42, edgecolor=c_dark, linewidth=1.2)
ax.set_title('CUDA Acceleration Speedup Over CPU Baseline', fontsize=13, fontweight='bold', color=c_dark, pad=14)
ax.set_ylabel('Speedup Factor (x)', fontsize=10.5, fontweight='bold', color=c_gray)
ax.set_ylim(0, 50)
ax.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax.set_facecolor('#FFFFFF')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, s in zip(bars, speedups):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{s:.2f}x',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color=c_dark,
            bbox=dict(boxstyle='round,pad=0.25', fc='#F8FAFC', ec='#CBD5E1', lw=1))

ax.axhline(1.0, color=c_cpu, linestyle='--', linewidth=1.5, label='CPU Baseline (1.0x)')
ax.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1', fontsize=10)

plt.tight_layout()
plt.savefig('images/speedup_analysis.png', dpi=200, bbox_inches='tight')
plt.close()

print('Experiment 5 charts generated successfully.')
