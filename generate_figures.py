#!/usr/bin/env python3
"""Generate all comparison figures for the paper (final layout)."""

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

RESULTS_FILE = 'results/benchmark_results.json'
OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(RESULTS_FILE):
    print(f"[WARN] {RESULTS_FILE} not found; skipping figures.")
    sys.exit(0)

with open(RESULTS_FILE) as f:
    results = json.load(f)

if not results:
    print("[WARN] No results; skipping figures.")
    sys.exit(0)

dims = sorted(set(r['n'] for r in results))
sec_types = sorted(set(r['secret_type'] for r in results))
q = results[0]['q']
bkz_beta = results[0].get('bkz_beta', 15)

methods = [
    ('highs_success', 'highs_time', 'MILP-HiGHS', '#1f77b4', 'o', '-'),
    ('cbc_success',   'cbc_time',   'MILP-CBC',  '#ff7f0e', 'D', '-'),
    ('lll_success',   'lll_time',   'LLL',       '#d62728', 's', '--'),
    ('bkz_success',   'bkz_time',   f'BKZ-{bkz_beta}', '#2ca02c', '^', '-'),
]


def success_rate(records, key):
    return sum(r[key] for r in records) / len(records) * 100 if records else 0.0


def mean_time(records, key):
    return float(np.mean([r[key] for r in records])) if records else 0.0


# ============================================================
# FIGURE 1: Success rate — 3 panels, shared legend BELOW
# ============================================================
fig, axes = plt.subplots(1, len(sec_types),
                         figsize=(4.6 * len(sec_types), 4.2),
                         sharey=True)
if len(sec_types) == 1:
    axes = [axes]

for ax, sec in zip(axes, sec_types):
    for succ_key, _, label, color, marker, ls in methods:
        rates = [success_rate([r for r in results
                               if r['n'] == n and r['secret_type'] == sec],
                              succ_key)
                 for n in dims]
        ax.plot(dims, rates, marker=marker, linestyle=ls, color=color,
                label=label, linewidth=1.8, markersize=5)
    ax.set_title(f'Secret: {sec}')
    ax.set_xlabel('Dimension $n$')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(dims)

axes[0].set_ylabel('Success Rate (%)')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, 0.01), frameon=True)
plt.suptitle(f'Success Rate vs. Dimension (q={q})', fontsize=12, y=0.98)
# Explicit margins — no tight bbox
plt.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.20, wspace=0.10)
plt.savefig(f'{OUT_DIR}/fig_success_rate.pdf')
plt.savefig(f'{OUT_DIR}/fig_success_rate.png')
plt.close()
print(f"Saved: {OUT_DIR}/fig_success_rate.pdf")


# ============================================================
# FIGURE 2: Time-to-solution — 3 panels, shared legend BELOW
# Explicit margins ensure the legend doesn't drift up into x-labels.
# ============================================================
fig, axes = plt.subplots(1, len(sec_types),
                         figsize=(4.6 * len(sec_types), 4.2),
                         sharey=True)
if len(sec_types) == 1:
    axes = [axes]

for ax, sec in zip(axes, sec_types):
    for succ_key, time_key, label, color, marker, ls in methods:
        base = succ_key.replace('_success', '')
        times = [mean_time([r for r in results
                            if r['n'] == n and r['secret_type'] == sec],
                           f'{base}_time')
                 for n in dims]
        ax.semilogy(dims, [max(t, 1e-4) for t in times],
                    marker=marker, linestyle=ls, color=color,
                    linewidth=1.8, markersize=5, label=label)
    ax.set_title(f'Secret: {sec}')
    ax.set_xlabel('Dimension $n$')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xticks(dims)

axes[0].set_ylabel('Mean Time-to-Solution (s, log scale)')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, 0.01), frameon=True)
plt.suptitle(f'Time-to-Solution Comparison (q={q})', fontsize=12, y=0.98)
plt.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.20, wspace=0.10)
plt.savefig(f'{OUT_DIR}/fig_time_to_solution.pdf')
plt.savefig(f'{OUT_DIR}/fig_time_to_solution.png')
plt.close()
print(f"Saved: {OUT_DIR}/fig_time_to_solution.pdf")


# ============================================================
# FIGURE 3: HiGHS vs CBC — extra headroom for annotations
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5.0))
x = np.arange(len(dims))
width = 0.35
highs_times = [mean_time([r for r in results if r['n'] == n], 'highs_time') for n in dims]
cbc_times   = [mean_time([r for r in results if r['n'] == n], 'cbc_time')   for n in dims]

ax.bar(x - width/2, highs_times, width, label='HiGHS', color='#1f77b4')
ax.bar(x + width/2, cbc_times,   width, label='CBC',   color='#ff7f0e')

ax.set_xlabel('Dimension $n$')
ax.set_ylabel('Mean Time (s)')
ax.set_title('MILP Solver Comparison: HiGHS vs. CBC', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(dims)
ax.grid(True, alpha=0.3, axis='y')

# -------- KEY FIX: 25% headroom + text anchored at bar top --------
ymax = max(max(highs_times), max(cbc_times))
ax.set_ylim(0, ymax * 1.25)

for i, n in enumerate(dims):
    if cbc_times[i] > 0 and highs_times[i] > 0:
        speedup = cbc_times[i] / highs_times[i]
        bar_top = max(highs_times[i], cbc_times[i])
        ax.text(i, bar_top * 1.02, f'{speedup:.2f}×',
                ha='center', va='bottom', fontsize=9,
                color='green', clip_on=False)

# Legend below, close to the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=2, frameon=True, fontsize=10)

plt.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.20)
plt.savefig(f'{OUT_DIR}/fig_solver_comparison.pdf')
plt.savefig(f'{OUT_DIR}/fig_solver_comparison.png')
plt.close()
print(f"Saved: {OUT_DIR}/fig_solver_comparison.pdf")


# ============================================================
# FIGURE 4: Success by secret type at max n — legend below
# ============================================================
max_n = max(dims)
fig, ax = plt.subplots(figsize=(8, 5.0))
x = np.arange(len(sec_types))
width = 0.2

for i, (succ_key, _, label, color, _, _) in enumerate(methods):
    rates = [success_rate([r for r in results
                           if r['n'] == max_n and r['secret_type'] == sec],
                          succ_key)
             for sec in sec_types]
    ax.bar(x + (i - 1.5) * width, rates, width, label=label, color=color)

ax.set_xlabel('Secret Distribution')
ax.set_ylabel('Success Rate (%)')
ax.set_title(f'Success Rate by Secret Type at $n={max_n}$', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(sec_types)
ax.set_ylim(0, 118)
ax.grid(True, alpha=0.3, axis='y')

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=4, frameon=True, fontsize=9)

plt.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.20)
plt.savefig(f'{OUT_DIR}/fig_success_by_secret.pdf')
plt.savefig(f'{OUT_DIR}/fig_success_by_secret.png')
plt.close()
print(f"Saved: {OUT_DIR}/fig_success_by_secret.pdf")

print("\nAll figures regenerated.")