#!/usr/bin/env python3
"""
Extension 2: Two-factor ablation study runner (single-cell).

Runs one (support_mode, error_bound) configuration on a set of base
instances and writes a partial JSON file. The CI workflow runs six
cells in parallel — one per configuration — and merges the outputs.

Configurations (2 × 3 factorial):
    support_mode ∈ {weight_cap, no_cap}
    error_bound  ∈ {5, 20, 30}

Base instances are loaded from results/benchmark_results.json, which
the workflow's prepare job generates. Each instance is regenerated
from its seed so the ablation runs on identical inputs as the
baseline.

Usage (matches the workflow's invocation):

    python run_ablation_study.py \\
        --dimensions 15 \\
        --instances-per-dim 5 \\
        --time-limit 30 \\
        --support-mode weight_cap \\
        --error-bound 5 \\
        --out results/ablation_weight_cap_E5.json

Output per cell:
    A JSON array of records, one per (instance, solver) pair.
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime

from lwe_benchmark import (
    generate_lwe_instance,
    solve_lwe_milp,
    secrets_match,
)


# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_BASE_MANIFEST = 'results/benchmark_results.json'


# ============================================================
# PER-INSTANCE ABLATION RUN
# ============================================================
def run_one_instance(base_rec, support_mode, error_bound, time_limit):
    """
    Rerun one MILP configuration on one base instance.

    Returns a record with success flags for HiGHS and CBC under the
    specified ablation configuration. Instance identity (key, n, q,
    seed, secret_type) is preserved so the merge job can pair cells
    correctly.
    """
    n = base_rec['n']
    q = base_rec['q']
    m = base_rec['m']
    sec_type = base_rec['secret_type']
    seed = base_rec['seed']

    # Regenerate the instance from its seed.
    # The baseline generator uses sigma_mode='fixed_abs' and
    # clip_factor=5.0 by default.
    A, b, s_true, e_true, empirical_sigma = generate_lwe_instance(
        n, q, m, sec_type,
        sigma_mode='fixed_abs',
        clip_factor=5.0,
        seed=seed,
        return_sigma=True,
    )

    max_abs_error = int(abs(e_true).max()) if len(e_true) else 0
    # Under fixed_abs, the natural error bound is 5 (clip factor 1× sigma
    # at sigma=1 gives ±5). The ablation varies E around this value.
    # We record planted_feasible against the ablation E below.
    planted_feasible = bool(max_abs_error <= error_bound)

    # ---- HiGHS ----
    s_h, _, st_h = solve_lwe_milp(
        A, b, q, sec_type, 'highs',
        time_limit=time_limit,
        support_mode=support_mode,
        error_bound=error_bound,
    )
    ok_h = secrets_match(s_h, s_true, q)

    # ---- CBC ----
    s_c, _, st_c = solve_lwe_milp(
        A, b, q, sec_type, 'cbc',
        time_limit=time_limit,
        support_mode=support_mode,
        error_bound=error_bound,
    )
    ok_c = secrets_match(s_c, s_true, q)

    return {
        'key': base_rec['key'],
        'n': n,
        'q': q,
        'm': m,
        'secret_type': sec_type,
        'instance': base_rec['instance'],
        'seed': seed,
        'support_mode': support_mode,
        'error_bound': error_bound,
        'max_abs_error': max_abs_error,
        'empirical_sigma': empirical_sigma,
        'planted_feasible': planted_feasible,
        'highs_success': bool(ok_h),
        'highs_status': st_h,
        'cbc_success': bool(ok_c),
        'cbc_status': st_c,
        'verified': bool(ok_h or ok_c),
        'timestamp': datetime.now().isoformat(),
    }


# ============================================================
# FILTERING
# ============================================================
def filter_base_instances(base_records, dimensions, instances_per_dim,
                          secret_types):
    """
    Filter the base manifest to the requested dimensions, secret
    types, and instance counts.

    The workflow generates a manifest that already matches the
    requested dimensions and instances, but this filter makes the
    script robust to a broader manifest.
    """
    filtered = [
        r for r in base_records
        if r['n'] in dimensions
        and r['secret_type'] in secret_types
    ]

    # Cap instance counts per (n, secret_type) cell
    grouped = {}
    for r in filtered:
        key = (r['n'], r['secret_type'])
        grouped.setdefault(key, []).append(r)

    capped = []
    for key, group in grouped.items():
        group_sorted = sorted(group, key=lambda r: r['instance'])
        capped.extend(group_sorted[:instances_per_dim])

    return capped


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Extension 2: two-factor ablation study (single cell)'
    )
    parser.add_argument('--dimensions', type=int, nargs='+', default=[15],
                        help='Dimensions to test')
    parser.add_argument('--instances-per-dim', type=int, default=5,
                        help='Instances per (dimension, secret_type) cell')
    parser.add_argument('--time-limit', type=int, default=30,
                        help='Per-attack time limit in seconds')
    parser.add_argument('--support-mode', choices=['weight_cap', 'no_cap'],
                        required=True,
                        help='Support information: weight cap or none')
    parser.add_argument('--error-bound', type=int, required=True,
                        choices=[5, 20, 30],
                        help='MILP error bound E')
    parser.add_argument('--secret-types', nargs='+',
                        default=['sparse'],
                        help='Secret types to test (default: sparse)')
    parser.add_argument('--base-manifest', default=DEFAULT_BASE_MANIFEST,
                        help='Path to base instance manifest')
    parser.add_argument('--out', required=True,
                        help='Path for the output JSON file')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Ablation Study Cell (Extension 2)")
    print(f"  support_mode:    {args.support_mode}")
    print(f"  error_bound:     {args.error_bound}")
    print(f"  dimensions:      {args.dimensions}")
    print(f"  instances/dim:   {args.instances_per_dim}")
    print(f"  secret_types:    {args.secret_types}")
    print(f"  time limit:      {args.time_limit}s")
    print(f"  base manifest:   {args.base_manifest}")
    print(f"  output:          {args.out}")
    print("=" * 70)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Load base manifest
    # ------------------------------------------------------------------
    manifest_path = Path(args.base_manifest)
    if not manifest_path.exists():
        print(f"[ERROR] Base manifest not found: {manifest_path}")
        print("        The workflow's prepare job should have created it.")
        sys.exit(1)

    with open(manifest_path) as f:
        base_records = json.load(f)

    print(f"Loaded {len(base_records)} base records from {manifest_path}")

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------
    base_instances = filter_base_instances(
        base_records,
        dimensions=args.dimensions,
        instances_per_dim=args.instances_per_dim,
        secret_types=args.secret_types,
    )

    if not base_instances:
        print("[ERROR] No instances match the filter criteria.")
        sys.exit(1)

    print(f"Filtered to {len(base_instances)} instances "
          f"({len(args.dimensions)} dim × {len(args.secret_types)} secret × "
          f"{args.instances_per_dim} instances)")

    # ------------------------------------------------------------------
    # Run ablation
    # ------------------------------------------------------------------
    results = []
    total = len(base_instances)
    solved = 0
    failed = 0
    infeasible_planted = 0

    for i, base_rec in enumerate(base_instances):
        print(f"\n[{i+1}/{total}] {base_rec['key']}")
        sys.stdout.flush()

        rec = run_one_instance(
            base_rec,
            support_mode=args.support_mode,
            error_bound=args.error_bound,
            time_limit=args.time_limit,
        )

        if not rec['planted_feasible']:
            infeasible_planted += 1
            print(f"  [WARN] max_abs_error={rec['max_abs_error']} > "
                  f"E={args.error_bound}: planted solution infeasible")

        if rec['verified']:
            solved += 1
        else:
            failed += 1

        print(f"  HiGHS={'OK' if rec['highs_success'] else 'X'} | "
              f"CBC={'OK' if rec['cbc_success'] else 'X'}")
        sys.stdout.flush()

        results.append(rec)

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Wrote {len(results)} records to {out_path}")
    print(f"  solved:                {solved}")
    print(f"  failed:                {failed}")
    print(f"  planted_infeasible:    {infeasible_planted}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    if results:
        n_highs = sum(r['highs_success'] for r in results)
        n_cbc = sum(r['cbc_success'] for r in results)
        print(f"\nRecovery rate (HiGHS): "
              f"{100 * n_highs / len(results):.1f}% "
              f"({n_highs}/{len(results)})")
        print(f"Recovery rate (CBC):   "
              f"{100 * n_cbc / len(results):.1f}% "
              f"({n_cbc}/{len(results)})")

        if infeasible_planted > 0:
            print(f"\n[WARN] {infeasible_planted} instances had "
                  f"planted_feasible=False. These are reported but "
                  f"should be excluded from recovery-rate analysis "
                  f"in the paper.")


if __name__ == '__main__':
    main()