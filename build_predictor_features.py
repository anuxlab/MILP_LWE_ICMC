#!/usr/bin/env python3
"""
Extension 6: Build pre-solve features for predictor training.

Reads benchmark JSON files, regenerates each instance from its seed,
and computes features that are available BEFORE the MILP solver runs.
The resulting feature table is used by train_and_validate_predictor.py.

Features (all pre-solve):
  Identity:
    n, m, q, sample_ratio, secret_type_id
  Noise policy:
    nominal_sigma, clip_bound, clip_factor
  Instance statistics from b:
    norm_b_l1, norm_b_l2, norm_b_inf, norm_b_mean,
    b_uniformity_score
  Instance statistics from A:
    A_density, A_row_norm_mean, A_row_norm_std, A_max, A_min

Labels (taken from the existing benchmark result):
  label_highs, label_cbc  (0/1)

Explicitly excluded (measured during or after the solve):
  first_feasible_time, first_matching_time, node_count, gap,
  bound_at_root_cutpool_cb, solver_status.

The script is deterministic: given the same inputs, it produces the
same features. This is required for held-out validation to be
meaningful.

Output: results/predictor_features.json
"""

import json
import sys
from pathlib import Path

import numpy as np

# Import the instance generator from lwe_benchmark to guarantee that
# regenerated instances match the originals exactly.
from lwe_benchmark import generate_lwe_instance


# ============================================================
# CONSTANTS
# ============================================================
SECRET_TYPE_MAP = {'ternary': 0, 'sparse': 1, 'uniform': 2}

# Data sources in priority order. All available sources are merged
# and deduplicated by checkpoint key.
INPUT_CANDIDATES = [
    'results/grid_results.json',
    'results/scale_results.json',
    'results/ablation_study.json',
    'results/benchmark_results.json',
]

DEFAULT_CLIP_FACTOR = 5.0
DEFAULT_SIGMA_MODE = 'fixed_abs'


# ============================================================
# FEATURE COMPUTATION
# ============================================================
def compute_features(A, b, q, secret_type, sigma, clip_bound, clip_factor):
    """
    Compute all pre-solve features for one instance.

    All computations use only A, b, and publicly declared parameters.
    No feature requires solving the MILP or running lattice reduction.
    """
    m, n = A.shape

    feats = {}

    # --- Identity ---
    feats['n'] = int(n)
    feats['m'] = int(m)
    feats['q'] = int(q)
    feats['sample_ratio'] = float(m) / float(n)
    feats['secret_type_id'] = int(SECRET_TYPE_MAP.get(secret_type, -1))

    # --- Noise policy ---
    feats['nominal_sigma'] = float(sigma)
    feats['clip_bound'] = int(clip_bound)
    feats['clip_factor'] = float(clip_factor)

    # --- Statistics of b ---
    # Interpret b as centered around 0 mod q: values near 0 or near q
    # are "small", values near q/2 are "large".
    b_int = b.astype(int)
    b_centered = np.minimum(b_int, q - b_int).astype(float)

    feats['norm_b_l1'] = float(np.sum(np.abs(b_centered)))
    feats['norm_b_l2'] = float(np.sqrt(np.sum(b_centered ** 2)))
    feats['norm_b_inf'] = float(np.max(np.abs(b_centered)))
    feats['norm_b_mean'] = float(np.mean(np.abs(b_centered)))
    feats['norm_b_std'] = float(np.std(np.abs(b_centered)))

    # Uniformity score of b mod q (chi-square-like statistic).
    # For a uniform b, this is approximately q-1 in expectation; large
    # deviations indicate non-uniformity.
    counts = np.bincount(b_int % q, minlength=q).astype(float)
    expected = float(m) / float(q)
    if expected > 0:
        feats['b_uniformity_score'] = float(
            np.sum((counts - expected) ** 2 / expected)
        )
    else:
        feats['b_uniformity_score'] = 0.0

    # --- Statistics of A ---
    feats['A_density'] = float(np.mean(A != 0))
    row_norms = np.linalg.norm(A.astype(float), axis=1)
    feats['A_row_norm_mean'] = float(np.mean(row_norms))
    feats['A_row_norm_std'] = float(np.std(row_norms))
    feats['A_max'] = int(A.max())
    feats['A_min'] = int(A.min())

    return feats


# ============================================================
# RECORD PROCESSING
# ============================================================
def resolve_sigma_params(rec):
    """
    Determine (sigma_mode, explicit_sigma, clip_factor) for a record.

    Older records (from the baseline run) may not carry these fields;
    apply defaults that match the baseline generator.
    """
    sigma_mode = rec.get('sigma_mode', DEFAULT_SIGMA_MODE)
    explicit_sigma = rec.get('explicit_sigma', None)
    clip_factor = rec.get('clip_factor', DEFAULT_CLIP_FACTOR)
    return sigma_mode, explicit_sigma, clip_factor


def regenerate_instance(rec):
    """
    Regenerate the instance exactly as the benchmark did.

    Returns (A, b, empirical_sigma). Raises RuntimeError if the record
    is missing required fields.
    """
    required = ['n', 'q', 'm', 'secret_type', 'seed']
    for field in required:
        if field not in rec:
            raise RuntimeError(f"Record missing field: {field}")

    sigma_mode, explicit_sigma, clip_factor = resolve_sigma_params(rec)

    if explicit_sigma is not None:
        A, b, _, _, emp_sigma = generate_lwe_instance(
            rec['n'], rec['q'], rec['m'], rec['secret_type'],
            error_sigma=explicit_sigma,
            clip_factor=clip_factor,
            seed=rec['seed'],
            return_sigma=True,
        )
    else:
        A, b, _, _, emp_sigma = generate_lwe_instance(
            rec['n'], rec['q'], rec['m'], rec['secret_type'],
            sigma_mode=sigma_mode,
            clip_factor=clip_factor,
            seed=rec['seed'],
            return_sigma=True,
        )

    return A, b, emp_sigma


def process_record(rec):
    """
    Compute features and attach labels for one record.

    Returns None if the record cannot be processed (missing labels,
    missing fields, or unsupported configuration).
    """
    # Labels are required
    if 'highs_success' not in rec or 'cbc_success' not in rec:
        return None

    # Regenerate the instance
    try:
        A, b, emp_sigma = regenerate_instance(rec)
    except Exception as e:
        print(f"  [WARN] {rec.get('key', '?')}: {e}")
        return None

    # Determine sigma and clip bound for the feature record
    sigma_mode, explicit_sigma, clip_factor = resolve_sigma_params(rec)
    if explicit_sigma is not None:
        sigma = float(explicit_sigma)
    elif sigma_mode == 'fixed_abs':
        sigma = 1.0
    elif sigma_mode == 'fixed_rel':
        sigma = float(rec['q']) / 100.0
    else:
        sigma = 1.0

    clip_bound = int(np.ceil(clip_factor * sigma))

    # Compute features
    feats = compute_features(
        A, b, rec['q'], rec['secret_type'],
        sigma, clip_bound, clip_factor,
    )

    # Attach labels and identity
    feats['label_highs'] = int(rec['highs_success'])
    feats['label_cbc'] = int(rec['cbc_success'])
    feats['key'] = rec['key']
    feats['empirical_sigma'] = float(emp_sigma)
    feats['support_mode'] = rec.get('support_mode', 'weight_cap')
    feats['sigma_mode'] = sigma_mode
    feats['explicit_sigma'] = explicit_sigma

    return feats


# ============================================================
# DATA LOADING
# ============================================================
def load_and_deduplicate():
    """
    Load records from all available sources and deduplicate by key.

    When the same key appears in multiple files, the last-write-wins
    policy applies (matching the checkpoint system's semantics).
    """
    seen = {}
    loaded_from = []

    for path_str in INPUT_CANDIDATES:
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            if not data:
                continue
            for rec in data:
                key = rec.get('key')
                if key is None:
                    continue
                seen[key] = rec
            loaded_from.append(f'{path_str} ({len(data)})')
        except Exception as e:
            print(f"[WARN] Could not load {path_str}: {e}")

    return list(seen.values()), loaded_from


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("Extension 6: Build predictor features")
    print("=" * 70)

    records, loaded_from = load_and_deduplicate()

    if not records:
        print("[ERROR] No input records found.")
        print(f"        Looked for: {INPUT_CANDIDATES}")
        sys.exit(1)

    print(f"\nLoaded records from:")
    for src in loaded_from:
        print(f"  {src}")
    print(f"After deduplication: {len(records)} unique records")

    # ------------------------------------------------------------------
    # Process each record
    # ------------------------------------------------------------------
    feature_rows = []
    skipped = 0
    for i, rec in enumerate(records):
        feats = process_record(rec)
        if feats is None:
            skipped += 1
        else:
            feature_rows.append(feats)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(records)}] processed")

    print(f"\nProcessed {len(feature_rows)} records "
          f"({skipped} skipped)")

    if not feature_rows:
        print("[ERROR] No features computed.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    out_path = Path('results/predictor_features.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(feature_rows, f, indent=2)
    print(f"\nWrote {len(feature_rows)} feature rows to {out_path}")

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Feature summary")
    print("=" * 70)

    feature_cols = [
        'n', 'm', 'q', 'sample_ratio', 'secret_type_id',
        'nominal_sigma', 'clip_bound', 'clip_factor',
        'norm_b_l1', 'norm_b_l2', 'norm_b_inf', 'norm_b_mean',
        'norm_b_std', 'b_uniformity_score',
        'A_density', 'A_row_norm_mean', 'A_row_norm_std',
        'A_max', 'A_min',
    ]

    print(f"{'feature':>22} {'min':>10} {'median':>10} {'max':>10}")
    print("-" * 55)
    for col in feature_cols:
        vals = [r[col] for r in feature_rows]
        print(f"{col:>22} {min(vals):>10.4g} "
              f"{np.median(vals):>10.4g} {max(vals):>10.4g}")

    # ------------------------------------------------------------------
    # Label distribution
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Label distribution")
    print("=" * 70)
    n_highs_pos = sum(r['label_highs'] for r in feature_rows)
    n_cbc_pos = sum(r['label_cbc'] for r in feature_rows)
    total = len(feature_rows)
    print(f"  HiGHS positive: {n_highs_pos}/{total} "
          f"({100*n_highs_pos/total:.1f}%)")
    print(f"  CBC positive:   {n_cbc_pos}/{total} "
          f"({100*n_cbc_pos/total:.1f}%)")

    # Class balance warning
    if n_highs_pos == 0 or n_highs_pos == total:
        print("\n[WARN] HiGHS label is single-class; cannot train a classifier.")
        print("       Extend the dataset with more diverse configurations.")
    if n_cbc_pos == 0 or n_cbc_pos == total:
        print("\n[WARN] CBC label is single-class; cannot train a classifier.")

    # ------------------------------------------------------------------
    # Configuration coverage
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Configuration coverage")
    print("=" * 70)
    n_dims = sorted(set(r['n'] for r in feature_rows))
    qs = sorted(set(r['q'] for r in feature_rows))
    sigmas = sorted(set(r['nominal_sigma'] for r in feature_rows))
    secrets = sorted(set(r['secret_type_id'] for r in feature_rows))
    print(f"  dimensions:      {n_dims}")
    print(f"  moduli:          {qs}")
    print(f"  sigmas:          {sigmas}")
    print(f"  secret_ids:      {secrets}")

    # ------------------------------------------------------------------
    # Warning if some configurations are sparse
    # ------------------------------------------------------------------
    configs = set(
        (r['q'], r['n'], r['secret_type_id'], r['nominal_sigma'])
        for r in feature_rows
    )
    if len(configs) < 5:
        print(f"\n[WARN] Only {len(configs)} distinct configurations; "
              f"held-out validation may be unreliable.")


if __name__ == '__main__':
    main()