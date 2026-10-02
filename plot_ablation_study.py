#!/usr/bin/env python3
"""
Extension 2: Plot two-factor ablation study results.

Reads results/ablation_study.json (merged from six cells).
Writes results/fig_ablation_study.pdf with three panels:
  (a) Grouped bar chart of recovery by (support_mode, E)
  (b) Heatmap of HiGHS recovery over (support_mode × E)
  (c) Interaction plot: recovery vs E, one line per support mode.

Instances with planted_feasible=False are excluded from analysis.
"""

import json
import sys
from pathlib import Path
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

IN = 'results/ablation_study.json'
OUT_DIR = 'results'
Path(OUT_DIR).mkdir(exist_ok=True)

if not Path(IN).exists():
    print(f"[WARN] {IN} not found; skipping ablation figure.")
    sys.exit(0)

with open(IN) as f:
    records = json.load(f)

if not records:
    print(f"[WARN] {IN} is empty; skipping.")
    sys.exit(0)

# Filter planted-infeasible
records_all = records
records = [r for r in records if r.get('planted_feasible', True)]
excluded = len(records_all) - len(records)
if excluded > 0:
    print(f"Excluded {excluded} instances with planted_feasible=False")

if not records:
    print("[WARN] No feasible instances remain; skipping.")
    sys.exit(0)

support_modes = sorted(set(r['support_mode'] for r in records))
error_bounds = sorted(set(r['error_bound'] for r in records))

def recovery_rate(sub, key):
    if not sub:
        return np.nan
    return 100.0 * sum(r[key] for r in sub) / len(sub)

def cell(**kw):
    return [r for r in records if all(r.get(k) == v for k, v in kw.items())]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

# Panel (a): grouped bars
ax = axes[0]
x = []
highs_rates, cbc_rates, labels = [], [], []
for support in support_modes:
    for E in error_bounds:
        sub = cell(support_mode=support, error_bound=E)
        highs_rates.append(recovery_rate(sub, 'highs_success'))
        cbc_rates.append(recovery_rate(sub, 'cbc_success'))
        short = 'cap' if support == 'weight_cap' else 'nocap'
        labels.append(f'{short}\nE={E}')
        x.append(len(x))
x = np.arange(len(x))
width = 0.38
ax.bar(x - width/2, highs_rates, width, label='HiGHS', color='#1f77b4')
ax.bar(x + width/2, cbc_rates, width, label='CBC', color='#ff7f0e')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Recovery rate (%)')
ax.set_title('(a) Recovery by configuration')
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(loc='best', fontsize=8)

# Panel (b): heatmap
ax = axes[1]
grid = np.full((len(support_modes), len(error_bounds)), np.nan)
for i, support in enumerate(support_modes):
    for j, E in enumerate(error_bounds):
        sub = cell(support_mode=support, error_bound=E)
        if sub:
            grid[i, j] = recovery_rate(sub, 'highs_success')

im = ax.imshow(grid, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
ax.set_xticks(range(len(error_bounds)))
ax.set_xticklabels([f'E={E}' for E in error_bounds])
ax.set_yticks(range(len(support_modes)))
ax.set_yticklabels(['weight cap' if s == 'weight_cap' else 'no cap'
                    for s in support_modes])
ax.set_xlabel('Error bound $E$')
ax.set_ylabel('Support constraint')
ax.set_title('(b) HiGHS recovery (%)')
for i in range(len(support_modes)):
    for j in range(len(error_bounds)):
        v = grid[i, j]
        if not np.isnan(v):
            color = 'white' if v < 50 else 'black'
            ax.text(j, i, f'{v:.0f}%', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Panel (c): interaction plot
ax = axes[2]
markers = {'weight_cap': 'o', 'no_cap': 's'}
colors = {'weight_cap': '#1f77b4', 'no_cap': '#d62728'}
styles = {'weight_cap': '-', 'no_cap': '--'}
for support in support_modes:
    ys = []
    for E in error_bounds:
        sub = cell(support_mode=support, error_bound=E)
        ys.append(recovery_rate(sub, 'highs_success'))
    ax.plot(error_bounds, ys,
            marker=markers[support], linestyle=styles[support],
            color=colors[support], linewidth=2.0, markersize=8,
            label='weight cap' if support == 'weight_cap' else 'no cap')
ax.set_xlabel('Error bound $E$')
ax.set_ylabel('Recovery rate (%) (HiGHS)')
ax.set_title('(c) Interaction: support × error bound')
ax.set_ylim(-5, 105)
ax.set_xticks(error_bounds)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_ablation_study.pdf', bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig_ablation_study.png', bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_DIR}/fig_ablation_study.pdf")

# Summary
print("\nRecovery rate (%) by configuration:")
print(f"{'support':>12} {'E':>4} {'n':>4} {'HiGHS':>8} {'CBC':>8}")
for support in support_modes:
    for E in error_bounds:
        sub = cell(support_mode=support, error_bound=E)
        if not sub:
            continue
        h = recovery_rate(sub, 'highs_success')
        c = recovery_rate(sub, 'cbc_success')
        print(f"{support:>12} {E:>4} {len(sub):>4} {h:>7.1f}% {c:>7.1f}%")