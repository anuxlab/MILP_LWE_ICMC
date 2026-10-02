#!/usr/bin/env python3
"""
Extension 2: Plot two-factor ablation study results.

Reads results/ablation_study.json (merged from six cells by the CI
workflow) and produces results/fig_ablation_study.pdf with three
panels:

  (a) Grouped bar chart of recovery rate by (support_mode, E) for
      HiGHS and CBC.
  (b) Heatmap of HiGHS recovery over the (support_mode × E) grid.
  (c) Interaction plot: recovery rate vs. E, one line per support
      mode, showing whether the effect of E depends on the support
      constraint.

Also prints summary tables and paired comparisons for the paper.

Instances with planted_feasible=False are excluded from recovery
analysis, with the count reported.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ------------------------------------------------------------------
# Plot style
# ------------------------------------------------------------------
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 11
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300

IN = 'results/ablation_study.json'
OUT_DIR = 'results'
Path(OUT_DIR).mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
if not Path(IN).exists():
    print(f"[WARN] {IN} not found; skipping ablation figure.")
    sys.exit(0)

with open(IN) as f:
    records = json.load(f)

if not records:
    print(f"[WARN] {IN} is empty; skipping.")
    sys.exit(0)

print(f"Loaded {len(records)} ablation records from {IN}")

# ------------------------------------------------------------------
# Filter: exclude instances where the planted solution is infeasible
# under the ablation error bound. Those instances cannot be recovered
# by construction; including them would bias the comparison.
# ------------------------------------------------------------------
records_all = records
records = [r for r in records if r.get('planted_feasible', True)]
excluded = len(records_all) - len(records)
if excluded > 0:
    print(f"Excluded {excluded} instances with planted_feasible=False")

if not records:
    print("[WARN] No feasible instances remain; skipping figure.")
    sys.exit(0)

# ------------------------------------------------------------------
# Discover grid structure
# ------------------------------------------------------------------
support_modes = sorted(set(r['support_mode'] for r in records))
error_bounds = sorted(set(r['error_bound'] for r in records))
n_dims = sorted(set(r['n'] for r in records))
secrets = sorted(set(r['secret_type'] for r in records))

print(f"  support_modes: {support_modes}")
print(f"  error_bounds:  {error_bounds}")
print(f"  dimensions:    {n_dims}")
print(f"  secrets:       {secrets}")


def recovery_rate(subset, key):
    if not subset:
        return np.nan
    return 100.0 * sum(r[key] for r in subset) / len(subset)


def cell(subset_filter):
    """Filter records by any predicate dict."""
    out = []
    for r in records:
        if all(r.get(k) == v for k, v in subset_filter.items()):
            out.append(r)
    return out


# ------------------------------------------------------------------
# Figure: three panels
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

# ------------------------------------------------------------------
# Panel (a): grouped bar chart of recovery by (support, E)
# ------------------------------------------------------------------
ax = axes[0]

x_groups = []          # one group per (support, E) combination
highs_rates = []
cbc_rates = []
labels = []

for support in support_modes:
    for E in error_bounds:
        sub = cell({'support_mode': support, 'error_bound': E})
        highs_rates.append(recovery_rate(sub, 'highs_success'))
        cbc_rates.append(recovery_rate(sub, 'cbc_success'))
        short_support = 'cap' if support == 'weight_cap' else 'nocap'
        labels.append(f'{short_support}\nE={E}')
        x_groups.append(len(x_groups))

x = np.arange(len(x_groups))
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

# Annotate counts
for i, (h, c) in enumerate(zip(highs_rates, cbc_rates)):
    if not np.isnan(h):
        ax.text(i - width/2, h + 1, f'{h:.0f}', ha='center',
                fontsize=6.5, color='#333333')
    if not np.isnan(c):
        ax.text(i + width/2, c + 1, f'{c:.0f}', ha='center',
                fontsize=6.5, color='#333333')

# ------------------------------------------------------------------
# Panel (b): heatmap of HiGHS recovery over (support × E)
# ------------------------------------------------------------------
ax = axes[1]

grid = np.full((len(support_modes), len(error_bounds)), np.nan)
for i, support in enumerate(support_modes):
    for j, E in enumerate(error_bounds):
        sub = cell({'support_mode': support, 'error_bound': E})
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

# Annotate cell values
for i in range(len(support_modes)):
    for j in range(len(error_bounds)):
        v = grid[i, j]
        if not np.isnan(v):
            color = 'white' if v < 50 else 'black'
            ax.text(j, i, f'{v:.0f}%', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# ------------------------------------------------------------------
# Panel (c): interaction plot — recovery vs E, one line per support
# ------------------------------------------------------------------
ax = axes[2]

markers = {'weight_cap': 'o', 'no_cap': 's'}
colors = {'weight_cap': '#1f77b4', 'no_cap': '#d62728'}
styles = {'weight_cap': '-', 'no_cap': '--'}

for support in support_modes:
    ys = []
    for E in error_bounds:
        sub = cell({'support_mode': support, 'error_bound': E})
        ys.append(recovery_rate(sub, 'highs_success'))
    ax.plot(error_bounds, ys,
            marker=markers[support],
            linestyle=styles[support],
            color=colors[support],
            linewidth=2.0, markersize=8,
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
print(f"\nSaved: {OUT_DIR}/fig_ablation_study.pdf")
print(f"Saved: {OUT_DIR}/fig_ablation_study.png")

# ------------------------------------------------------------------
# Summary tables for the paper
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Recovery rate (%) by configuration")
print("=" * 70)
print(f"{'support':>12} {'E':>4} {'n':>4} {'HiGHS':>8} {'CBC':>8}")
print("-" * 45)
for support in support_modes:
    for E in error_bounds:
        sub = cell({'support_mode': support, 'error_bound': E})
        if not sub:
            continue
        h = recovery_rate(sub, 'highs_success')
        c = recovery_rate(sub, 'cbc_success')
        print(f"{support:>12} {E:>4} {len(sub):>4} {h:>7.1f}% {c:>7.1f}%")

# ------------------------------------------------------------------
# Marginal effects
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Marginal effects — HiGHS recovery (%)")
print("=" * 70)

print("\nEffect of support constraint (avg over E):")
for support in support_modes:
    vals = [recovery_rate(cell({'support_mode': support, 'error_bound': E}),
                          'highs_success')
            for E in error_bounds]
    vals = [v for v in vals if not np.isnan(v)]
    if vals:
        print(f"  {support:>12}: {np.mean(vals):>6.1f}%")

print("\nEffect of error bound (avg over support):")
for E in error_bounds:
    vals = [recovery_rate(cell({'support_mode': s, 'error_bound': E}),
                          'highs_success')
            for s in support_modes]
    vals = [v for v in vals if not np.isnan(v)]
    if vals:
        print(f"  E = {E:>2}: {np.mean(vals):>6.1f}%")

# ------------------------------------------------------------------
# Interaction check
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Interaction check (HiGHS)")
print("=" * 70)

# Difference in recovery between support modes at each E
print("\nDifference (weight_cap − no_cap) in recovery, by E:")
diffs = []
for E in error_bounds:
    sub_cap = cell({'support_mode': 'weight_cap', 'error_bound': E})
    sub_no = cell({'support_mode': 'no_cap', 'error_bound': E})
    r_cap = recovery_rate(sub_cap, 'highs_success')
    r_no = recovery_rate(sub_no, 'highs_success')
    if not np.isnan(r_cap) and not np.isnan(r_no):
        d = r_cap - r_no
        diffs.append(d)
        print(f"  E = {E:>2}: weight_cap = {r_cap:>5.1f}%, "
              f"no_cap = {r_no:>5.1f}%, diff = {d:>+6.1f} pp")

if len(diffs) >= 2:
    spread = max(diffs) - min(diffs)
    print(f"\n  Spread of the difference across E: {spread:.1f} pp")
    if spread > 15:
        print("  → Suggestive of interaction between support and E.")
        print("    Report this as a first-order effect in the tested grid,")
        print("    not as a causal decomposition.")
    else:
        print("  → Difference is roughly constant across E; support")
        print("    and error bound may act independently in this grid.")

# ------------------------------------------------------------------
# Notes for the paper
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Notes for the paper")
print("=" * 70)
if excluded > 0:
    print(f"- {excluded} instances excluded (planted_feasible=False)")
print("- All tested cells are reported; none dropped for being uninformative.")
print("- Do NOT claim causal isolation; the design rules out some")
print("  confoundings but does not separate relaxation, presolve,")
print("  heuristics, and constraint effects.")