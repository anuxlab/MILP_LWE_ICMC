#!/usr/bin/env python3
"""Generate all comparison figures for the paper."""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 11
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

RESULTS_FILE = 'results/benchmark_results.json'
OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(RESULTS_FILE):
    print(f"[WARN] {RESULTS_FILE} not found; skipping figures.")
    sys.exit(0)

with open(RESULTS_FILE) as f:
    results = json.load(f)

if not results:
    print("[WARN] No results in benchmark_results.json; skipping figures.")
    sys.exit(0)

dims = sorted(set(r['n'] for r in results))
sec_types = sorted(set(r['secret_type'] for r in results))
q = results[0]['q']
methods = [
    ('highs_success', 'highs_time', 'MILP-HiGHS', '#1f77b4', 'o', '-'),
    ('cbc_success', 'cbc_time', 'MILP-CBC', '#ff7f0e', 'D', '-'),
    ('lll_success', 'lll_time', 'LLL', '#d62728', 's', '--'),
    ('bkz_success', 'bkz_time', f"BKZ-{results[0].get('bkz_beta', 15)}", '#2ca02c', '^', '-'),
]


def success_rate(records, key):
    return sum(r[key] for r in records) / len(records) * 100 if records else 0.0


def mean_time(records, key):
    return float(np.mean([r[key] for r in records])) if records else 0.0


# ============================================================
# FIGURE 1: Success rate, one panel per secret type
# ============================================================
fig, axes = plt.subplots(1, len(sec_types), figsize=(4.5 * len(sec_types), 4), sharey=True)
if len(sec_types) == 1:
    axes = [axes]

for ax, sec in zip(axes, sec_types):
    for succ_key, _, label, color, marker, ls in methods:
        rates = []
        for n in dims:
            sub = [r for r in results if r['n'] == n and r['secret_type'] == sec]
            rates.append(success_rate(sub, succ_key))
        ax.plot(dims, rates, marker=marker, linestyle=ls, color=color,
                label=label, linewidth=1.8, markersize=5)
    ax.set_title(f'Secret: {sec}')
    ax.set_xlabel('Dimension $n$')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(dims)
axes[0].set_ylabel('Success Rate (%)')
axes[0].legend(loc='lower left', fontsize=8)
plt.suptitle(f'Success Rate vs. Dimension (q={q})', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_success_rate.pdf')
plt.savefig(f'{OUT_DIR}/fig_success_rate.png')
plt.close()
print(f"Saved: {OUT_DIR}/fig_success_rate.pdf")


# ============================================================
# FIGURE 2: Time-to-solution (log scale)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
colors = {'highs': '#1f77b4', 'cbc': '#ff7f0e', 'lll': '#d62728', 'bkz': '#2ca02c'}
markers = {'highs': 'o', 'cbc': 'D', 'lll': 's', 'bkz': '^'}
for method_key, _, label, _, _, _ in methods:
    base = method_key.replace('_success', '')
    for sec in sec_types:
        times = []
        for n in dims:
            sub = [r for r in results if r['n'] == n and r['secret_type'] == sec]
            times.append(mean_time(sub, f'{base}_time'))
        ax.semilogy(dims, [max(t, 1e-4) for t in times],
                    marker=markers[base], color=colors[base],
                    linestyle='-' if sec == 'ternary' else ('--' if sec == 'sparse' else ':'),
                    alpha=0.85, linewidth=1.5, markersize=5,
                    label=f'{label} ({sec})')

ax.set_xlabel('Dimension $n$')
ax.set_ylabel('Mean Time-to-Solution (s, log scale)')
ax.set_title(f'Time-to-Solution Comparison (q={q})')
ax.set_xticks(dims)
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='upper left', fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_time_to_solution.pdf')
plt.savefig(f'{OUT_DIR}/fig_time_to_solution.png')
plt.close()
print(f"Saved: {OUT_DIR}/fig_time_to_solution.pdf")


# ============================================================
# FIGURE 3: HiGHS vs CBC bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(dims))
width = 0.35
highs_times = [mean_time([r for r in results if r['n'] == n], 'highs_time') for n in dims]
cbc_times = [mean_time([r for r in results if r['n'] == n], 'cbc_time') for n in dims]

ax.bar(x - width/2, highs_times, width, label='HiGHS', color='#1f77b4')
ax.bar(x + width/2, cbc_times, width, label='CBC', color='#ff7f0e')
ax.set_xlabel('Dimension $n$')
ax.set_ylabel('Mean Time (s)')
ax.set_title('MILP Solver Comparison: HiGHS vs. CBC')
ax.set_xticks(x)
ax.set_xticklabels(dims)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for i, n in enumerate(dims):
    if cbc_times[i] > 0 and highs_times[i] > 0:
        speedup = cbc_times[i] / highs_times[i]
        ax.text(i, max(highs_times[i], cbc_times[i]) * 1.05,
                f'{speedup:.2f}×', ha='center', fontsize=8, color='green')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_solver_comparison.pdf')
plt.savefig(f'{OUT_DIR}/fig_solver_comparison.png')
plt.close()
print(f"Saved: {OUT_DIR}/fig_solver_comparison.pdf")


# ============================================================
# FIGURE 4: Success rate by secret type (bar chart at max n)
# ============================================================
max_n = max(dims)
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(sec_types))
width = 0.2
for i, (succ_key, _, label, color, _, _) in enumerate(methods):
    rates = []
    for sec in sec_types:
        sub = [r for r in results if r['n'] == max_n and r['secret_type'] == sec]
        rates.append(success_rate(sub, succ_key))
    ax.bar(x + (i - 1.5) * width, rates, width, label=label, color=color)

ax.set_xlabel('Secret Distribution')
ax.set_ylabel('Success Rate (%)')
ax.set_title(f'Success Rate by Secret Type at $n={max_n}$')
ax.set_xticks(x)
ax.set_xticklabels(sec_types)
ax.set_ylim(0, 110)
ax.legend(loc='lower right', ncol=2, fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_success_by_secret.pdf')
plt.savefig(f'{OUT_DIR}/fig_success_by_secret.png')
plt.close()
print(f"Saved: {OUT_DIR}/fig_success_by_secret.pdf")

print("\nAll figures generated.")