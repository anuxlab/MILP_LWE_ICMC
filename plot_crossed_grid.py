#!/usr/bin/env python3
"""
Extension 4: Crossed modulus × noise grid.

Produces results/fig_crossed_grid.pdf with two panels:
  (a) Heatmap of HiGHS recovery rate over (q, σ) for ternary secrets
  (b) Heatmap of HiGHS recovery rate over (q, σ) for sparse secrets

Also prints an interaction summary table for the paper.
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
rcParams['font.size'] = 10
rcParams['figure.dpi'] = 300

IN = 'results/grid_results.json'
OUT = 'results'
os.makedirs(OUT, exist_ok=True)

if not Path(IN).exists():
    print(f"[WARN] {IN} not found; skipping.")
    sys.exit(0)

with open(IN) as f:
    records = json.load(f)

if not records:
    print("[WARN] Empty grid results; skipping.")
    sys.exit(0)

# Discover grid dimensions
qs = sorted(set(r['q'] for r in records))
sigmas = sorted(set(r.get('nominal_sigma', 1.0) for r in records))
secrets = ['ternary', 'sparse']


def cell_rate(q, sigma, secret, key):
    sub = [r for r in records
           if r['q'] == q
           and abs(r.get('nominal_sigma', 1.0) - sigma) < 1e-6
           and r['secret_type'] == secret
           and r.get('planted_feasible', True)]
    if not sub:
        return np.nan
    return 100.0 * sum(r[key] for r in sub) / len(sub)


# ------------------------------------------------------------------
# Figure: 2 heatmaps (ternary, sparse) for HiGHS recovery
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

for ax, sec in zip(axes, secrets):
    grid = np.full((len(sigmas), len(qs)), np.nan)
    for i, s in enumerate(sigmas):
        for j, q in enumerate(qs):
            grid[i, j] = cell_rate(q, s, sec, 'highs_success')

    im = ax.imshow(grid, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=100, origin='lower')
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels([str(q) for q in qs])
    ax.set_yticks(range(len(sigmas)))
    ax.set_yticklabels([f'{s:.1f}' for s in sigmas])
    ax.set_xlabel('Modulus $q$')
    ax.set_ylabel('Absolute noise $\\sigma$')
    ax.set_title(f'HiGHS recovery (%): {sec}')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Crossed modulus × noise grid (recovery rate %)', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_crossed_grid.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig_crossed_grid.png', bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/fig_crossed_grid.pdf")

# ------------------------------------------------------------------
# Interaction summary for the paper
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Interaction summary — HiGHS recovery rate (%)")
print("=" * 70)
for sec in secrets:
    print(f"\nSecret: {sec}")
    header = "sigma  " + "  ".join(f"q={q:<5}" for q in qs)
    print(header)
    print("-" * len(header))
    for s in sigmas:
        row = f"{s:<7}" + "  ".join(
            f"{cell_rate(q, s, sec, 'highs_success'):>6.1f}%" for q in qs
        )
        print(row)

# Compute marginal effects
print("\n" + "=" * 70)
print("Marginal effects")
print("=" * 70)
for sec in secrets:
    print(f"\nSecret: {sec}")
    # Effect of q averaged over sigma
    print("  Effect of q (avg over σ):")
    for q in qs:
        vals = [cell_rate(q, s, sec, 'highs_success')
                for s in sigmas if not np.isnan(cell_rate(q, s, sec, 'highs_success'))]
        if vals:
            print(f"    q = {q:>5}: {np.mean(vals):>6.1f}%")
    # Effect of sigma averaged over q
    print("  Effect of σ (avg over q):")
    for s in sigmas:
        vals = [cell_rate(q, s, sec, 'highs_success')
                for q in qs if not np.isnan(cell_rate(q, s, sec, 'highs_success'))]
        if vals:
            print(f"    σ = {s:>5}: {np.mean(vals):>6.1f}%")