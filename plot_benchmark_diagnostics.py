#!/usr/bin/env python3
"""
Paired MILP diagnostics: 2-panel figure.

Panel (a) — per-instance HiGHS vs CBC runtimes, log-log, with
            the y=x equality diagonal.
Panel (b) — recovery agreement by secret type: both / only-HiGHS /
            only-CBC / neither.

Reads results/benchmark_results.json.
Writes results/fig_milp_diagnostics.pdf and .png.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 10
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 300

RESULTS_FILE = 'results/benchmark_results.json'
OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)

if not Path(RESULTS_FILE).exists():
    print(f"[WARN] {RESULTS_FILE} not found; skipping MILP diagnostics.")
    sys.exit(0)

with open(RESULTS_FILE) as f:
    results = json.load(f)

if not results:
    print("[WARN] No results; skipping MILP diagnostics.")
    sys.exit(0)

sec_order = ['ternary', 'sparse', 'uniform']
sec_colors = {'ternary': '#1f77b4', 'sparse': '#2ca02c', 'uniform': '#d62728'}
sec_markers = {'ternary': 'o', 'sparse': 's', 'uniform': '^'}

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.5, 3.9))

# ------------------------------------------------------------------
# Panel (a): paired scatter
# ------------------------------------------------------------------
JITTER = 0.03
rng = np.random.default_rng(42)

for sec in sec_order:
    xs, ys = [], []
    for r in results:
        if r['secret_type'] != sec:
            continue
        xs.append(max(r['highs_time'], 1e-3))
        ys.append(max(r['cbc_time'], 1e-3))
    if not xs:
        continue
    xs = np.array(xs) * (1 + JITTER * rng.standard_normal(len(xs)))
    ys = np.array(ys) * (1 + JITTER * rng.standard_normal(len(ys)))
    ax_a.scatter(xs, ys, s=22, alpha=0.55,
                 color=sec_colors[sec], marker=sec_markers[sec],
                 edgecolor='black', linewidth=0.3, label=sec)

lims = [5e-3, 1.5e2]
ax_a.plot(lims, lims, color='black', linestyle='--', linewidth=0.9,
          alpha=0.7, label='$y = x$')
ax_a.set_xscale('log')
ax_a.set_yscale('log')
ax_a.set_xlim(lims)
ax_a.set_ylim(lims)
ax_a.set_xlabel('HiGHS runtime (s)')
ax_a.set_ylabel('CBC runtime (s)')
ax_a.set_title('(a) Paired MILP runtimes')
ax_a.grid(True, alpha=0.3, which='both')
ax_a.legend(loc='lower right', frameon=True, fontsize=7, ncol=2)

# ------------------------------------------------------------------
# Panel (b): recovery agreement
# ------------------------------------------------------------------
both_counts, only_h_counts, only_c_counts, neither_counts = [], [], [], []

for sec in sec_order:
    both = only_h = only_c = neither = 0
    for r in results:
        if r['secret_type'] != sec:
            continue
        h = bool(r['highs_success'])
        c = bool(r['cbc_success'])
        if h and c:        both += 1
        elif h and not c:  only_h += 1
        elif c and not h:  only_c += 1
        else:              neither += 1
    both_counts.append(both)
    only_h_counts.append(only_h)
    only_c_counts.append(only_c)
    neither_counts.append(neither)

totals = [sum(t) for t in zip(both_counts, only_h_counts,
                              only_c_counts, neither_counts)]

def pct(vals):
    return [100 * v / t if t else 0 for v, t in zip(vals, totals)]

x = np.arange(len(sec_order))
width = 0.55
stack_colors = {'both': '#2ca02c', 'only_h': '#1f77b4',
                'only_c': '#ff7f0e', 'neither': '#cccccc'}
stack_labels = {'both': 'Both', 'only_h': 'Only HiGHS',
                'only_c': 'Only CBC', 'neither': 'Neither'}

bottom = np.zeros(len(sec_order))
for key, vals in [
    ('both',    pct(both_counts)),
    ('only_h',  pct(only_h_counts)),
    ('only_c',  pct(only_c_counts)),
    ('neither', pct(neither_counts)),
]:
    ax_b.bar(x, vals, width, bottom=bottom,
             color=stack_colors[key], label=stack_labels[key],
             edgecolor='white', linewidth=0.5)
    bottom += np.array(vals)

ax_b.set_xticks(x)
ax_b.set_xticklabels(sec_order)
ax_b.set_ylabel('Instances (%)')
ax_b.set_title('(b) Recovery agreement')
ax_b.set_ylim(0, 100)
ax_b.grid(True, alpha=0.3, axis='y')
ax_b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
            ncol=4, frameon=True, fontsize=7)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig(f'{OUT_DIR}/fig_milp_diagnostics.pdf', bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig_milp_diagnostics.png', bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/fig_milp_diagnostics.pdf")