#!/usr/bin/env python3
"""
Modulus and noise sensitivity analysis for the LWE MILP benchmark.

Supports Extensions 3, 4, and 5:

  Extension 3 (modulus sensitivity, two noise policies):
    - Panel (a): success rate vs q under fixed_abs (σ=1)
    - Panel (b): success rate vs q under fixed_rel (σ=q/100)

  Extension 4 (crossed modulus × noise grid):
    - Panel (c): heatmap of recovery over (q, σ) for ternary secrets
    - Panel (d): heatmap of recovery over (q, σ) for sparse secrets
    (Only shown if data with explicit_sigma values is present.)

  Extension 5 (adaptive dimension expansion):
    - Panel (e): frontier bracketing curves for one noise policy
    (Only shown if data with more than 2 dimension values is present.)

The script auto-detects available data and renders the appropriate
number of panels.

Inputs (in priority order):
  - results/grid_results.json      (Extension 4 output)
  - results/scale_results.json     (Extension 3 output)
  - results/benchmark_results.json (baseline + multi-modulus)

Output:
  - results/fig_modulus_scaling.pdf
  - results/fig_modulus_scaling.png
  - Printed summary tables
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

# ------------------------------------------------------------------
# Plot configuration
# ------------------------------------------------------------------
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 11
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300

OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Load data — merge all available sources
# ------------------------------------------------------------------
IN_CANDIDATES = [
    'results/grid_results.json',
    'results/scale_results.json',
    'results/scale_results_with_lp.json',
    'results/benchmark_results.json',
]

records = []
loaded_from = []
for cand in IN_CANDIDATES:
    if Path(cand).exists():
        try:
            with open(cand) as f:
                data = json.load(f)
            if data:
                records.extend(data)
                loaded_from.append(f'{cand} ({len(data)})')
        except Exception as e:
            print(f"[WARN] Could not parse {cand}: {e}")

if not records:
    print(f"[WARN] No input data found among {IN_CANDIDATES}; skipping.")
    sys.exit(0)

print(f"Loaded {len(records)} records from:")
for src in loaded_from:
    print(f"  {src}")

# ------------------------------------------------------------------
# Deduplicate by checkpoint key (last write wins)
# ------------------------------------------------------------------
seen = {}
for r in records:
    key = r.get('key', str(id(r)))
    seen[key] = r
records = list(seen.values())
print(f"After deduplication: {len(records)} unique records")

# ------------------------------------------------------------------
# Discover structure
# ------------------------------------------------------------------
qs = sorted(set(r['q'] for r in records))
dims = sorted(set(r['n'] for r in records))
secs = sorted(set(r['secret_type'] for r in records))

# Determine noise modes present
has_explicit_sigma = any(r.get('explicit_sigma') is not None for r in records)
has_fixed_abs = any(r.get('sigma_mode', 'fixed_abs') == 'fixed_abs'
                    and r.get('explicit_sigma') is None for r in records)
has_fixed_rel = any(r.get('sigma_mode') == 'fixed_rel' for r in records)

# Explicit sigma values (for Extension 4)
explicit_sigmas = sorted(set(
    r['explicit_sigma'] for r in records
    if r.get('explicit_sigma') is not None
))

print(f"\nStructure:")
print(f"  q:                 {qs}")
print(f"  n:                 {dims}")
print(f"  secrets:           {secs}")
print(f"  explicit sigmas:   {explicit_sigmas if explicit_sigmas else 'none'}")
print(f"  fixed_abs present: {has_fixed_abs}")
print(f"  fixed_rel present: {has_fixed_rel}")

# ------------------------------------------------------------------
# Plot styling
# ------------------------------------------------------------------
methods = [
    ('highs_success', 'MILP-HiGHS', '#1f77b4', 'o', '-'),
    ('cbc_success',   'MILP-CBC',  '#ff7f0e', 'D', '-'),
    ('lll_success',   'LLL',       '#d62728', 's', '--'),
    ('bkz_success',   'BKZ-15',    '#2ca02c', '^', '-'),
]
plot_secrets = [s for s in ['ternary', 'sparse'] if s in secs]
if not plot_secrets:
    plot_secrets = secs


def cell_records(q, sec, n=None, sigma_mode=None, explicit_sigma=None,
                 support_mode=None):
    """Filter records by configuration."""
    out = []
    for r in records:
        if r['q'] != q or r['secret_type'] != sec:
            continue
        if n is not None and r['n'] != n:
            continue
        if sigma_mode is not None:
            if r.get('sigma_mode', 'fixed_abs') != sigma_mode:
                continue
        if explicit_sigma is not None:
            if r.get('explicit_sigma') != explicit_sigma:
                continue
        if support_mode is not None:
            if r.get('support_mode', 'weight_cap') != support_mode:
                continue
        out.append(r)
    return out


def success_rate(subset, key='highs_success'):
    if not subset:
        return np.nan
    return 100.0 * sum(r[key] for r in subset) / len(subset)


# ------------------------------------------------------------------
# Determine number of panels
# ------------------------------------------------------------------
n_panels = 0
panels_spec = []

if has_fixed_abs:
    n_panels += 2  # one per secret
    panels_spec.append(('fixed_abs', 'ternary'))
    panels_spec.append(('fixed_abs', 'sparse'))

if has_fixed_rel:
    n_panels += 2
    panels_spec.append(('fixed_rel', 'ternary'))
    panels_spec.append(('fixed_rel', 'sparse'))

has_grid = len(explicit_sigmas) >= 2 and len(qs) >= 2
has_bracketing = len(dims) >= 3

if has_grid:
    n_panels += 2  # two heatmaps

# ------------------------------------------------------------------
# If no data matches, fall back to a single-panel layout
# ------------------------------------------------------------------
if n_panels == 0:
    panels_spec = [('fixed_abs', 'ternary')]
    n_panels = 1

# ------------------------------------------------------------------
# Figure layout
# ------------------------------------------------------------------
ncols = min(3, n_panels)
nrows = (n_panels + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(4.5 * ncols, 4.0 * nrows),
                         squeeze=False)
axes = axes.flatten()

# ------------------------------------------------------------------
# Helper: draw a success-rate-vs-q panel
# ------------------------------------------------------------------
def draw_success_vs_q(ax, sigma_mode, sec):
    for key, label, color, marker, ls in methods:
        xs, ys = [], []
        for q in qs:
            sub = cell_records(q, sec, sigma_mode=sigma_mode)
            # Filter out explicit_sigma records — those belong to grid
            sub = [r for r in sub if r.get('explicit_sigma') is None]
            if sub:
                rate = success_rate(sub, key)
                if not np.isnan(rate):
                    xs.append(q)
                    ys.append(rate)
        if xs:
            ax.plot(xs, ys, marker=marker, linestyle=ls, color=color,
                    label=label, linewidth=1.8, markersize=7)

    ax.set_xscale('log')
    ax.set_xlabel('Modulus $q$ (log scale)')
    ax.set_ylabel('Recovery rate (%)')
    ax.set_title(f'{sigma_mode} — secret: {sec}')
    ax.set_ylim(-5, 105)
    ax.set_xticks(qs)
    ax.set_xticklabels([str(q) for q in qs])
    ax.minorticks_off()
    ax.grid(True, alpha=0.3)

    # Reference lines for deployed schemes
    if 3329 in qs:
        ax.axvline(3329, color='grey', linestyle=':', alpha=0.5)
    if 12289 in qs:
        ax.axvline(12289, color='grey', linestyle=':', alpha=0.5)

    ax.legend(loc='best', fontsize=8, framealpha=0.9)


# ------------------------------------------------------------------
# Draw panels (a), (b), (c), (d) — success rate vs q
# ------------------------------------------------------------------
idx = 0
for sigma_mode, sec in panels_spec:
    if idx >= len(axes):
        break
    draw_success_vs_q(axes[idx], sigma_mode, sec)
    idx += 1

# ------------------------------------------------------------------
# Panel: crossed-grid heatmaps (Extension 4)
# ------------------------------------------------------------------
if has_grid:
    for sec in plot_secrets:
        if idx >= len(axes):
            break
        ax = axes[idx]

        # Build (sigma, q) grid
        grid = np.full((len(explicit_sigmas), len(qs)), np.nan)
        for i, sig in enumerate(explicit_sigmas):
            for j, q in enumerate(qs):
                sub = cell_records(q, sec, explicit_sigma=sig)
                if sub:
                    grid[i, j] = success_rate(sub, 'highs_success')

        im = ax.imshow(grid, aspect='auto', cmap='RdYlGn',
                       vmin=0, vmax=100, origin='lower')
        ax.set_xticks(range(len(qs)))
        ax.set_xticklabels([str(q) for q in qs])
        ax.set_yticks(range(len(explicit_sigmas)))
        ax.set_yticklabels([f'{s:g}' for s in explicit_sigmas])
        ax.set_xlabel('Modulus $q$')
        ax.set_ylabel('Absolute noise $\\sigma$')
        ax.set_title(f'Crossed grid — {sec} (HiGHS %)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        idx += 1

# ------------------------------------------------------------------
# Panel: frontier bracketing (Extension 5)
# ------------------------------------------------------------------
if has_bracketing and idx < len(axes):
    ax = axes[idx]
    for q in qs:
        rates = []
        for n in dims:
            sub = cell_records(q, 'ternary', n=n, sigma_mode='fixed_abs')
            sub = [r for r in sub if r.get('explicit_sigma') is None]
            rates.append(success_rate(sub, 'highs_success') if sub else np.nan)

        # Only plot if at least 2 valid points
        valid = [(n, r) for n, r in zip(dims, rates) if not np.isnan(r)]
        if len(valid) >= 2:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, marker='o', label=f'q={q}', linewidth=1.8)

    ax.set_xlabel('Dimension $n$')
    ax.set_ylabel('Recovery rate (%) (ternary, HiGHS)')
    ax.set_title('Frontier bracketing')
    ax.set_ylim(-5, 105)
    ax.set_xticks(dims)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    idx += 1

# ------------------------------------------------------------------
# Hide unused axes
# ------------------------------------------------------------------
for j in range(idx, len(axes)):
    axes[j].axis('off')

# ------------------------------------------------------------------
# Shared title, layout, save
# ------------------------------------------------------------------
plt.suptitle('Modulus and Noise Sensitivity',
             fontsize=12, y=1.00)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_modulus_scaling.pdf', bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/fig_modulus_scaling.png', bbox_inches='tight')
plt.close()
print(f"\nSaved: {OUT_DIR}/fig_modulus_scaling.pdf")
print(f"Saved: {OUT_DIR}/fig_modulus_scaling.png")

# ------------------------------------------------------------------
# Print summary tables for the paper
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Recovery rate (%) — fixed_abs policy, by (q, n)")
print("=" * 70)
for sec in plot_secrets:
    print(f"\nSecret: {sec}")
    header = "n\\q   " + "  ".join(f"{q:>6}" for q in qs)
    print(header)
    print("-" * len(header))
    for n in dims:
        row = f"{n:<5} "
        for q in qs:
            sub = cell_records(q, sec, n=n, sigma_mode='fixed_abs')
            sub = [r for r in sub if r.get('explicit_sigma') is None]
            rate = success_rate(sub, 'highs_success')
            row += f" {rate:>6.1f}" if not np.isnan(rate) else "    ---"
        print(row)

if has_fixed_rel:
    print("\n" + "=" * 70)
    print("Recovery rate (%) — fixed_rel policy, by (q, n)")
    print("=" * 70)
    for sec in plot_secrets:
        print(f"\nSecret: {sec}")
        header = "n\\q   " + "  ".join(f"{q:>6}" for q in qs)
        print(header)
        print("-" * len(header))
        for n in dims:
            row = f"{n:<5} "
            for q in qs:
                sub = cell_records(q, sec, n=n, sigma_mode='fixed_rel')
                sub = [r for r in sub if r.get('explicit_sigma') is None]
                rate = success_rate(sub, 'highs_success')
                row += f" {rate:>6.1f}" if not np.isnan(rate) else "    ---"
            print(row)

if has_grid:
    print("\n" + "=" * 70)
    print("Crossed grid — recovery rate (%) by (q, σ), HiGHS")
    print("=" * 70)
    for sec in plot_secrets:
        print(f"\nSecret: {sec}")
        header = "σ\\q   " + "  ".join(f"{q:>6}" for q in qs)
        print(header)
        print("-" * len(header))
        for sig in explicit_sigmas:
            row = f"{sig:<5g} "
            for q in qs:
                sub = cell_records(q, sec, explicit_sigma=sig)
                rate = success_rate(sub, 'highs_success')
                row += f" {rate:>6.1f}" if not np.isnan(rate) else "    ---"
            print(row)

# ------------------------------------------------------------------
# Marginal effects for paper text
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Marginal effects")
print("=" * 70)
for sec in plot_secrets:
    print(f"\nSecret: {sec}")

    # Effect of q (fixed_abs, averaged over n)
    print("  Effect of q (fixed_abs, avg over n):")
    for q in qs:
        vals = []
        for n in dims:
            sub = cell_records(q, sec, n=n, sigma_mode='fixed_abs')
            sub = [r for r in sub if r.get('explicit_sigma') is None]
            rate = success_rate(sub, 'highs_success')
            if not np.isnan(rate):
                vals.append(rate)
        if vals:
            print(f"    q = {q:>5}: {np.mean(vals):>6.1f}%")

    # Effect of n (fixed_abs)
    if has_bracketing:
        print("  Effect of n (fixed_abs, avg over q):")
        for n in dims:
            vals = []
            for q in qs:
                sub = cell_records(q, sec, n=n, sigma_mode='fixed_abs')
                sub = [r for r in sub if r.get('explicit_sigma') is None]
                rate = success_rate(sub, 'highs_success')
                if not np.isnan(rate):
                    vals.append(rate)
            if vals:
                print(f"    n = {n:>5}: {np.mean(vals):>6.1f}%")

# ------------------------------------------------------------------
# Feasibility warning
# ------------------------------------------------------------------
infeasible = [r for r in records if r.get('planted_feasible') is False]
if infeasible:
    print(f"\n[WARN] {len(infeasible)} instances have planted_feasible=False.")
    print("       These are excluded from recovery analysis.")
    print("       Re-run with --error-bound scaled to clip_factor × σ.")