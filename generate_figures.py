#!/usr/bin/env python3
"""Generate figures from benchmark_results.json."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_FILE = 'results/benchmark_results.json'
OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)

with open(RESULTS_FILE) as f:
    results = json.load(f)

dims = sorted(set(r['n'] for r in results))
sec_types = sorted(set(r['secret_type'] for r in results))
q = results[0]['q']

# ============================================================
# FIG 1: Success rate vs dimension (per secret type)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, sec in zip(axes, sec_types):
    milp_rate, lll_rate, bkz_rate = [], [], []
    for n in dims:
        sub = [r for r in results if r['n'] == n and r['secret_type'] == sec]
        if sub:
            milp_rate.append(sum(r['milp_success'] for r in sub) / len(sub) * 100)
            lll_rate.append(sum(r['lll_success'] for r in sub) / len(sub) * 100)
            bkz_rate.append(sum(r['bkz_success'] for r in sub) / len(sub) * 100)
    ax.plot(dims, milp_rate, 'o-', label='MILP', color='#1f77b4', linewidth=2)
    ax.plot(dims, lll_rate, 's--', label='LLL', color='#d62728', linewidth=2)
    ax.plot(dims, bkz_rate, '^-', label='BKZ', color='#2ca02c', linewidth=2)
    ax.set_title(f'Secret: {sec}')
    ax.set_xlabel('Dimension $n$')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel('Success Rate (%)')
axes[0].legend(loc='lower left')
plt.suptitle(f'LWE Benchmark (q={q})', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/success_rate.pdf', bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/success_rate.png', bbox_inches='tight', dpi=150)
print(f"Saved: {OUT_DIR}/success_rate.pdf")

# ============================================================
# FIG 2: Time-to-solution (log scale)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
for sec, marker in zip(sec_types, ['o', 's', '^']):
    milp_t = []
    for n in dims:
        sub = [r['milp_time'] for r in results if r['n'] == n and r['secret_type'] == sec]
        milp_t.append(np.mean(sub) if sub else 0)
    ax.semilogy(dims, milp_t, marker=marker, label=f'MILP ({sec})', linewidth=1.8)

ax.set_xlabel('Dimension $n$')
ax.set_ylabel('Time-to-solution (s, log scale)')
ax.set_title('MILP Time-to-Solution vs. Dimension')
ax.grid(True, alpha=0.3, which='both')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/time_to_solution.pdf', bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/time_to_solution.png', bbox_inches='tight', dpi=150)
print(f"Saved: {OUT_DIR}/time_to_solution.pdf")

# ============================================================
# Summary table
# ============================================================
print(f"\n{'n':<5} {'Secret':<10} {'MILP':<10} {'LLL':<10} {'BKZ':<10}")
print("-" * 50)
for n in dims:
    for sec in sec_types:
        sub = [r for r in results if r['n'] == n and r['secret_type'] == sec]
        if sub:
            m = sum(r['milp_success'] for r in sub) / len(sub) * 100
            l = sum(r['lll_success'] for r in sub) / len(sub) * 100
            b = sum(r['bkz_success'] for r in sub) / len(sub) * 100
            print(f"{n:<5} {sec:<10} {m:>6.1f}%    {l:>6.1f}%    {b:>6.1f}%")