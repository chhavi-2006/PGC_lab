#!/usr/bin/env python3
"""
generate_charts.py - Generates performance comparison charts for Experiment 4
Course: Parallel and GPU Computing (26ECAC304) | Experiment 4

Uses empirical benchmark data measured in the WSL Ubuntu lab environment.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('images', exist_ok=True)

# Typography & theme settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial']

# Muted engineering palette
c_blue = '#2B5C8F'      # Deep Slate Blue
c_red = '#B85D43'       # Muted Terracotta (False Sharing)
c_green = '#387864'     # Muted Sage Teal (Padded / Aligned)
c_purple = '#64748B'    # Slate Gray
c_orange = '#C97B4B'    # Warm Amber
c_dark = '#1E293B'      # Slate Dark
c_gray = '#475569'      # Slate Mid

# =========================================================================
# 1. Cache Coherence Protocol Bus Transactions & Memory Traffic Chart
# =========================================================================
protocols = ['MSI', 'MESI', 'MOESI']
adjacent_bus_tx = [4000, 3999, 3999]
padded_bus_tx = [4, 2, 2]
writebacks = [1999, 1999, 0]
c2c_transfers = [0, 0, 1999]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor='#FAFBFC')

x = np.arange(len(protocols))
width = 0.35

rects1 = ax1.bar(x - width/2, adjacent_bus_tx, width, label='Adjacent (False Sharing)', color=c_red, alpha=0.92, edgecolor=c_dark)
rects2 = ax1.bar(x + width/2, padded_bus_tx, width, label='Padded (Cache Aligned)', color=c_green, alpha=0.92, edgecolor=c_dark)

ax1.set_title('Bus Transactions: Adjacent vs. Padded Layout', fontsize=12, fontweight='bold', color=c_dark, pad=12)
ax1.set_xlabel('Coherence Protocol', fontsize=10.5, fontweight='bold', color=c_gray)
ax1.set_ylabel('Total Bus Transactions (Lower is Better)', fontsize=10.5, fontweight='bold', color=c_gray)
ax1.set_xticks(x)
ax1.set_xticklabels(protocols, fontweight='bold')
ax1.set_ylim(0, 4500)
ax1.set_facecolor('#FFFFFF')
ax1.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1')

for r in rects1:
    h = r.get_height()
    ax1.annotate(f'{h}', xy=(r.get_x() + r.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color=c_red)

for r in rects2:
    h = r.get_height()
    ax1.annotate(f'{h}', xy=(r.get_x() + r.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color=c_green)

rects3 = ax2.bar(x - width/2, writebacks, width, label='DRAM Writebacks', color=c_orange, alpha=0.92, edgecolor=c_dark)
rects4 = ax2.bar(x + width/2, c2c_transfers, width, label='Cache-to-Cache Transfers (Owned)', color=c_blue, alpha=0.92, edgecolor=c_dark)

ax2.set_title('DRAM Writebacks vs. Cache-to-Cache Transfers (Adjacent)', fontsize=12, fontweight='bold', color=c_dark, pad=12)
ax2.set_xlabel('Coherence Protocol', fontsize=10.5, fontweight='bold', color=c_gray)
ax2.set_ylabel('Transaction Count', fontsize=10.5, fontweight='bold', color=c_gray)
ax2.set_xticks(x)
ax2.set_xticklabels(protocols, fontweight='bold')
ax2.set_ylim(0, 2300)
ax2.set_facecolor('#FFFFFF')
ax2.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1')

for r in rects3:
    h = r.get_height()
    ax2.annotate(f'{h}', xy=(r.get_x() + r.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color=c_orange)

for r in rects4:
    h = r.get_height()
    ax2.annotate(f'{h}', xy=(r.get_x() + r.get_width()/2, h),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color=c_blue)

plt.suptitle('Cache Coherence Protocol Traffic Analysis (1,000 Iterations / 4,000 Operations)',
             fontsize=13.5, fontweight='bold', color=c_dark, y=1.01)
plt.tight_layout()
plt.savefig('images/cache_coherence_bus_traffic.png', dpi=200, bbox_inches='tight')
plt.close()

# =========================================================================
# 2. Empirical False Sharing Execution Time & Speedup Scaling Chart
# =========================================================================
# Data from compare_numa execution
threads = [1, 2, 4]
adj_time = [0.0256, 0.0154, 0.0191]
pad_time = [0.0217, 0.0149, 0.0103]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor='#FAFBFC')

ax1.plot(threads, adj_time, marker='o', markersize=8, color=c_red, lw=2.5, label='Adjacent Layout (False Sharing)')
ax1.plot(threads, pad_time, marker='s', markersize=8, color=c_green, lw=2.5, linestyle='--', label='Padded Layout (64B Aligned)')

for t, at, pt in zip(threads, adj_time, pad_time):
    ax1.annotate(f'{at:.4f}s', (t, at), xytext=(-10, 8), textcoords='offset points', fontsize=9, fontweight='bold', color=c_red)
    ax1.annotate(f'{pt:.4f}s', (t, pt), xytext=(-10, -15), textcoords='offset points', fontsize=9, fontweight='bold', color=c_green)

ax1.set_title('Execution Time Scaling (100M Shared Increments)', fontsize=12, fontweight='bold', color=c_dark, pad=12)
ax1.set_xlabel('Thread Count (P)', fontsize=10.5, fontweight='bold', color=c_gray)
ax1.set_ylabel('Execution Time in Seconds (Lower is Better)', fontsize=10.5, fontweight='bold', color=c_gray)
ax1.set_xticks(threads)
ax1.set_ylim(0.006, 0.030)
ax1.set_facecolor('#FFFFFF')
ax1.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1')

speedup_advantage = [at / pt for at, pt in zip(adj_time, pad_time)]
x_pos = np.arange(len(threads))

bars = ax2.bar(x_pos, speedup_advantage, width=0.45, color=[c_purple, c_blue, c_green], edgecolor=c_dark, alpha=0.92)
for b, val in zip(bars, speedup_advantage):
    ax2.annotate(f'{val:.2f}x\nFaster', xy=(b.get_x() + b.get_width()/2, val),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c_dark)

ax2.set_title('Padded Layout Speedup Over Adjacent Layout', fontsize=12, fontweight='bold', color=c_dark, pad=12)
ax2.set_xlabel('Thread Count (P)', fontsize=10.5, fontweight='bold', color=c_gray)
ax2.set_ylabel('Speedup Factor (Adjacent / Padded)', fontsize=10.5, fontweight='bold', color=c_gray)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{t} Thread' if t == 1 else f'{t} Threads' for t in threads], fontweight='bold')
ax2.set_ylim(0, 2.3)
ax2.set_facecolor('#FFFFFF')
ax2.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.suptitle('False Sharing Benchmark: Adjacent vs. 64-Byte Padded Layout',
             fontsize=13.5, fontweight='bold', color=c_dark, y=1.01)
plt.tight_layout()
plt.savefig('images/false_sharing_scaling.png', dpi=200, bbox_inches='tight')
plt.close()

# =========================================================================
# 3. NUMA Memory Sweep Throughput & Policy Comparison Chart
# =========================================================================
labels = ['1T Default', '1T Node 0 Bound', '2T Default', '2T Node 0 Bound']
sweep_times = [0.1154, 0.1027, 0.1097, 0.0526]
total_gb = (512 * 4) / 1024.0  # 2.0 GB
sweep_bw = [total_gb / t for t in sweep_times]

fig, ax1 = plt.subplots(figsize=(9.5, 5.2), facecolor='#FAFBFC')

bar_colors = [c_purple, c_blue, c_orange, c_green]
bars = ax1.bar(labels, sweep_bw, color=bar_colors, width=0.5, edgecolor=c_dark, alpha=0.92)

for b, t_val, bw_val in zip(bars, sweep_times, sweep_bw):
    ax1.annotate(f'{bw_val:.2f} GB/s\n({t_val:.4f}s)', xy=(b.get_x() + b.get_width()/2, bw_val),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c_dark)

ax1.set_title('NUMA 512 MB Memory Sweep: Bandwidth & Latency (Default vs. Node 0 Bound)',
              fontsize=12.5, fontweight='bold', color=c_dark, pad=14)
ax1.set_ylabel('Effective Memory Bandwidth (GB/s)', fontsize=10.5, fontweight='bold', color=c_gray)
ax1.set_ylim(0, 44)
ax1.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
ax1.set_facecolor('#FFFFFF')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('images/numa_sweep_latency.png', dpi=200, bbox_inches='tight')
plt.close()

# =========================================================================
# 4. Master 4-Panel Performance Dashboard
# =========================================================================
fig, ((d1, d2), (d3, d4)) = plt.subplots(2, 2, figsize=(13.5, 9.5), facecolor='#FAFBFC')

d1.bar(x - width/2, adjacent_bus_tx, width, label='Adjacent (False Sharing)', color=c_red, alpha=0.92, edgecolor=c_dark)
d1.bar(x + width/2, padded_bus_tx, width, label='Padded (Cache Aligned)', color=c_green, alpha=0.92, edgecolor=c_dark)
d1.set_title('(A) Cache Coherence Bus Transactions (1,000 Iters)', fontsize=11.5, fontweight='bold', color=c_dark)
d1.set_xticks(x)
d1.set_xticklabels(protocols, fontweight='bold')
d1.set_ylabel('Bus Transactions', fontsize=10, fontweight='bold', color=c_gray)
d1.set_ylim(0, 5100)
d1.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
d1.set_facecolor('#FFFFFF')
d1.spines['top'].set_visible(False)
d1.spines['right'].set_visible(False)
d1.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1')

d2.plot(threads, adj_time, marker='o', color=c_red, lw=2.5, label='Adjacent (False Sharing)')
d2.plot(threads, pad_time, marker='s', color=c_green, lw=2.5, linestyle='--', label='Padded (64B Aligned)')
d2.set_title('(B) False Sharing Penalty: 100M Increments', fontsize=11.5, fontweight='bold', color=c_dark)
d2.set_xticks(threads)
d2.set_xlabel('Threads', fontsize=10, fontweight='bold', color=c_gray)
d2.set_ylabel('Execution Time (s)', fontsize=10, fontweight='bold', color=c_gray)
d2.set_ylim(0.008, 0.028)
d2.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
d2.set_facecolor('#FFFFFF')
d2.spines['top'].set_visible(False)
d2.spines['right'].set_visible(False)
d2.legend(frameon=True, facecolor='#F8FAFC', edgecolor='#CBD5E1')

d3.bar(['1 Thread', '2 Threads', '4 Threads'], speedup_advantage, color=[c_purple, c_blue, c_green], width=0.45, edgecolor=c_dark, alpha=0.92)
for i, val in enumerate(speedup_advantage):
    d3.annotate(f'{val:.2f}x Faster', xy=(i, val), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c_dark)
d3.set_title('(C) Speedup Advantage: Padded vs. Adjacent Layout', fontsize=11.5, fontweight='bold', color=c_dark)
d3.set_ylabel('Speedup Factor', fontsize=10, fontweight='bold', color=c_gray)
d3.set_ylim(0, 2.3)
d3.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
d3.set_facecolor('#FFFFFF')
d3.spines['top'].set_visible(False)
d3.spines['right'].set_visible(False)

d4.bar(labels, sweep_bw, color=bar_colors, width=0.5, edgecolor=c_dark, alpha=0.92)
for i, bw in enumerate(sweep_bw):
    d4.annotate(f'{bw:.2f} GB/s', xy=(i, bw), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=c_dark)
d4.set_title('(D) Memory Bandwidth: 512 MB Sweep (4 Passes)', fontsize=11.5, fontweight='bold', color=c_dark)
d4.set_ylabel('Throughput (GB/s)', fontsize=10, fontweight='bold', color=c_gray)
d4.set_ylim(0, 44)
d4.tick_params(axis='x', rotation=12)
d4.grid(True, linestyle=':', color='#E2E8F0', alpha=0.9)
d4.set_facecolor('#FFFFFF')
d4.spines['top'].set_visible(False)
d4.spines['right'].set_visible(False)

plt.suptitle('Experiment 4: Cache Coherence (MSI/MESI/MOESI) & NUMA Performance Dashboard',
             fontsize=14, fontweight='bold', color=c_dark, y=0.995)
plt.tight_layout()
plt.savefig('images/performance_comparison_dashboard.png', dpi=200, bbox_inches='tight')
plt.close()

print("[+] Successfully generated all Experiment 4 benchmark charts in images/")
