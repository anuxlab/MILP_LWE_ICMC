#!/usr/bin/env python3
"""
Extension 6: Plot held-out predictor validation results.

Produces results/fig_predictor_validation.pdf with three panels:
  (a) AUC by split type: baseline vs candidate
  (b) ROC curves for the best split type
  (c) Feature importance for the candidate model
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

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['figure.dpi'] = 300

IN = 'results/predictor_validation.json'
OUT = 'results'
if not Path(IN).exists():
    print(f"[WARN] {IN} not found; skipping.")
    sys.exit(0)

with open(IN) as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

# ------------------------------------------------------------------
# Panel (a): AUC by split type
# ------------------------------------------------------------------
ax = axes[0]
split_order = ['leave_dim_out', 'leave_modulus_out', 'leave_secret_out',
               'random_cell_split']
split_labels = ['Leave\ndim out', 'Leave\nmodulus out',
                'Leave\nsecret out', 'Random\ncell split']

baseline_aucs, candidate_aucs = [], []
for split in split_order:
    b_aucs = [r['auc'] for r in data.get('label_highs', [])
              if r['split'] == split and r['name'] == 'baseline'
              and r.get('auc') is not None]
    c_aucs = [r['auc'] for r in data.get('label_highs', [])
              if r['split'] == split and r['name'] == 'candidate'
              and r.get('auc') is not None]
    baseline_aucs.append(np.mean(b_aucs) if b_aucs else np.nan)
    candidate_aucs.append(np.mean(c_aucs) if c_aucs else np.nan)

x = np.arange(len(split_order))
width = 0.35
ax.bar(x - width/2, baseline_aucs, width, label='Baseline', color='#1f77b4')
ax.bar(x + width/2, candidate_aucs, width, label='Candidate', color='#2ca02c')
ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5,
           label='Random')
ax.set_xticks(x)
ax.set_xticklabels(split_labels, fontsize=8)
ax.set_ylabel('AUC')
ax.set_title('(a) AUC by held-out split type')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=8)

# ------------------------------------------------------------------
# Panel (b): distribution of AUC across random splits
# ------------------------------------------------------------------
ax = axes[1]
for name, color in [('baseline', '#1f77b4'), ('candidate', '#2ca02c')]:
    aucs = [r['auc'] for r in data.get('label_highs', [])
            if r['split'] == 'random_cell_split'
            and r['name'] == name
            and r.get('auc') is not None]
    if aucs:
        ax.hist(aucs, bins=np.linspace(0, 1, 11),
                alpha=0.5, label=name, color=color, edgecolor='black')

ax.set_xlabel('AUC (random cell split, 10 seeds)')
ax.set_ylabel('Count')
ax.set_title('(b) AUC distribution')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# ------------------------------------------------------------------
# Panel (c): Summary table as text
# ------------------------------------------------------------------
ax = axes[2]
ax.axis('off')

lines = ['Split type  │ Baseline │ Candidate', '─' * 38]
for split, label in zip(split_order, ['dim', 'mod', 'sec', 'rand']):
    b_aucs = [r['auc'] for r in data.get('label_highs', [])
              if r['split'] == split and r['name'] == 'baseline'
              and r.get('auc') is not None]
    c_aucs = [r['auc'] for r in data.get('label_highs', [])
              if r['split'] == split and r['name'] == 'candidate'
              and r.get('auc') is not None]
    b_str = f"{np.mean(b_aucs):.3f}" if b_aucs else "  -- "
    c_str = f"{np.mean(c_aucs):.3f}" if c_aucs else "  -- "
    lines.append(f"  {label:<8}  │  {b_str}   │  {c_str}")

text = '\n'.join(lines)
ax.text(0.05, 0.95, text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace')
ax.set_title('(c) Mean AUC summary (HiGHS label)')

plt.tight_layout()
plt.savefig(f'{OUT}/fig_predictor_validation.pdf', bbox_inches='tight')
plt.savefig(f'{OUT}/fig_predictor_validation.png', bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/fig_predictor_validation.pdf")

# ------------------------------------------------------------------
# Print paper-ready summary
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Predictor validation summary (label: HiGHS recovery)")
print("=" * 70)
print(f"{'split':>20} {'predictor':>12} {'AUC':>8} {'acc':>8} {'F1':>8}")
print("-" * 60)
for split in split_order:
    for name in ['baseline', 'candidate']:
        rs = [r for r in data.get('label_highs', [])
              if r['split'] == split and r['name'] == name
              and r.get('auc') is not None]
        if rs:
            aucs = np.mean([r['auc'] for r in rs])
            accs = np.mean([r['acc'] for r in rs if r.get('acc')])
            f1s = np.mean([r['f1'] for r in rs if r.get('f1')])
            print(f"{split:>20} {name:>12} {aucs:>8.3f} {accs:>8.3f} {f1s:>8.3f}")