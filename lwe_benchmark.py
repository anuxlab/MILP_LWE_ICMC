#!/usr/bin/env python3
"""
LWE Benchmark: MILP (HiGHS + CBC) vs. Lattice Reduction (LLL/BKZ)
Records all solvers' success/time on identical instances for comparison.

Solver priority for 'highs':
  1. HiGHS CLI binary (via HiGHS_CMD, path detected)
  2. Native HiGHS Python API (pulp.HiGHS)
  3. Fallback to CBC (with warning)
"""

import numpy as np
import time
import os
import json
import shutil
import argparse
from datetime import datetime
from pulp import (
    LpProblem, LpVariable, LpInteger, lpSum, LpMinimize,
    PULP_CBC_CMD, HiGHS_CMD, value,
)
from fpylll import IntegerMatrix, LLL, BKZ

# --- Try native HiGHS Python API (pulp >= 3.0) ---
try:
    from pulp import HiGHS as HiGHS_PY
    HAS_HIGHS_PY = True
except ImportError:
    HAS_HIGHS_PY = False


# ============================================================
# HiGHS DETECTION
# ============================================================
def find_highs():
    """Locate the HiGHS CLI executable on the system."""
    path = shutil.which('highs')
    if path is None:
        for candidate in ['/usr/local/bin/highs', '/usr/bin/highs', '/opt/highs/bin/highs']:
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return path


# ============================================================
# 1. LWE INSTANCE GENERATION
# ============================================================
def generate_lwe_instance(n, q, m, secret_type='ternary', error_sigma=1.0, seed=None):
    """Generate a Search-LWE instance (A, b, s_true, e_true)."""
    if seed is not None:
        np.random.seed(seed)

    if secret_type == 'ternary':
        s_true = np.random.choice([-1, 0, 1], size=n)
    elif secret_type == 'sparse':
        s_true = np.zeros(n, dtype=int)
        idx = np.random.choice(n, size=min(3, n), replace=False)
        s_true[idx] = np.random.choice([-1, 1], size=len(idx))
    else:  # uniform
        s_true = np.random.randint(0, q, size=n)

    A = np.random.randint(0, q, size=(m, n))
    e_true = np.round(np.random.normal(0, error_sigma, size=m)).astype(int)
    e_true = np.clip(e_true, -5, 5)
    b = (A @ s_true + e_true) % q
    return A, b, s_true, e_true


# ============================================================
# 2. MILP ATTACK (parameterized solver: 'highs' or 'cbc')
# ============================================================
def solve_lwe_milp(A, b, q, secret_type='ternary', solver='highs',
                   time_limit=60, highs_path=None):
    """
    Solve Search-LWE with MILP.

    Solver selection:
      - solver='highs' + highs_path set  → HiGHS CLI
      - solver='highs' + no CLI          → native HiGHS Python API (if available)
      - solver='highs' + neither         → CBC fallback
      - solver='cbc'                     → CBC
    """
    m, n = A.shape
    ERROR_BOUND = 20  # tight bound; true errors are ~N(0,1)

    prob = LpProblem(f"LWE_{solver}", LpMinimize)

    # Secret bounds based on structure
    if secret_type in ('ternary', 'sparse'):
        s_low, s_high = -1, 1
    else:  # uniform
        s_low, s_high = 0, q - 1

    s_vars = [LpVariable(f"s_{j}", lowBound=s_low, upBound=s_high, cat=LpInteger) for j in range(n)]
    k_vars = [LpVariable(f"k_{i}", lowBound=-100, upBound=100, cat=LpInteger) for i in range(m)]
    e_plus = [LpVariable(f"ep_{i}", lowBound=0, upBound=ERROR_BOUND, cat=LpInteger) for i in range(m)]
    e_minus = [LpVariable(f"em_{i}", lowBound=0, upBound=ERROR_BOUND, cat=LpInteger) for i in range(m)]

    # Structural constraints
    if secret_type == 'ternary':
        s_pos = [LpVariable(f"sp_{j}", cat='Binary') for j in range(n)]
        s_neg = [LpVariable(f"sn_{j}", cat='Binary') for j in range(n)]
        for j in range(n):
            prob += s_vars[j] == s_pos[j] - s_neg[j]
            prob += s_pos[j] + s_neg[j] <= 1
    elif secret_type == 'sparse':
        z_vars = [LpVariable(f"z_{j}", cat='Binary') for j in range(n)]
        for j in range(n):
            prob += s_vars[j] <= z_vars[j]
            prob += s_vars[j] >= -z_vars[j]
        prob += lpSum(z_vars) <= 4

    # Objective: minimize L1 error
    prob += lpSum(e_plus[i] + e_minus[i] for i in range(m))

    # Modular equality constraints
    for i in range(m):
        prob += lpSum(A[i, j] * s_vars[j] for j in range(n)) + e_plus[i] - e_minus[i] - q * k_vars[i] == b[i]

    start = time.time()
    try:
        if solver == 'highs' and highs_path:
            prob.solve(HiGHS_CMD(path=highs_path, msg=False, timeLimit=time_limit))
        elif solver == 'highs' and HAS_HIGHS_PY:
            prob.solve(HiGHS_PY(msg=False, timeLimit=time_limit))
        elif solver == 'highs':
            # No HiGHS in any form — fall back to CBC
            prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=2))
        else:
            prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=2))
        elapsed = time.time() - start
    except Exception as e:
        print(f"    [{solver.upper()} ERROR] {e}")
        return None, time.time() - start, 'ERROR'

    if prob.status == 1:
        s_rec = np.array([value(s_vars[j]) for j in range(n)], dtype=int)
        return s_rec, elapsed, 'SUCCESS'
    return None, elapsed, 'FAIL'


# ============================================================
# 3. LATTICE REDUCTION ATTACK (fpylll)
# ============================================================
def solve_lwe_lattice(A, b, q, method='LLL', beta=20, time_limit=60):
    """Primal attack (uSVP) with LLL or BKZ via fpylll."""
    m, n = A.shape
    d = m + n + 1

    B = IntegerMatrix(d, d)
    for i in range(m):
        B[i, i] = q
    for i in range(n):
        for j in range(m):
            B[m + i, j] = int(A[j, i])
        B[m + i, m + i] = 1
    for j in range(m):
        B[d - 1, j] = int(b[j])
    B[d - 1, d - 1] = 1

    start = time.time()
    try:
        if method == 'LLL':
            LLL.reduction(B, delta=0.99)
        elif method == 'BKZ':
            BKZ.reduction(B, BKZ.Param(block_size=beta, max_loops=4))
        elapsed = time.time() - start
    except Exception as e:
        print(f"    [{method} ERROR] {e}")
        return None, time.time() - start, 'ERROR'

    for i in range(min(5, d)):
        v = [B[i, j] for j in range(d)]
        s_cand = np.array(v[m:m + n], dtype=int) % q
        residual = (A @ s_cand - b) % q
        if np.all(np.minimum(residual, q - residual) <= 5):
            return s_cand, elapsed, 'SUCCESS'

    return None, time.time() - start, 'FAIL'


# ============================================================
# 4. MAIN BENCHMARK
# ============================================================
def run_benchmark(args):
    # Detect HiGHS once
    highs_path = find_highs()

    print("=" * 70)
    print(f"LWE Benchmark (Dual-Solver) — {datetime.now().isoformat()}")
    print(f"HiGHS CLI path:        {highs_path if highs_path else 'NOT FOUND'}")
    print(f"HiGHS Python API:      {'available' if HAS_HIGHS_PY else 'NOT available'}")
    if not highs_path and not HAS_HIGHS_PY:
        print("[WARN] HiGHS not available in any form. 'HiGHS' results will use CBC.")
    print(f"Dimensions:            {args.dimensions}")
    print(f"Secret types:          {args.secret_types}")
    print(f"Instances/config:      {args.instances}")
    print(f"Time limit/attack:     {args.time_limit}s")
    print(f"BKZ β:                 {args.bkz_beta}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    results_file = os.path.join(args.output_dir, 'benchmark_results.json')

    total = len(args.dimensions) * len(args.secret_types) * args.instances
    done = 0

    for n in args.dimensions:
        m = args.sample_ratio * n
        for sec_type in args.secret_types:
            print(f"\n>>> n={n}, q={args.q}, m={m}, secret={sec_type}")

            for inst in range(args.instances):
                seed = args.seed_base + inst
                A, b, s_true, e_true = generate_lwe_instance(
                    n, args.q, m, sec_type, seed=seed
                )

                # --- MILP: HiGHS (CLI → Python API → CBC fallback) ---
                t0 = time.time()
                s_h, _, st_h = solve_lwe_milp(
                    A, b, args.q, sec_type, 'highs',
                    time_limit=args.time_limit, highs_path=highs_path,
                )
                t_h = time.time() - t0
                ok_h = (s_h is not None) and np.array_equal(s_h, s_true)

                # --- MILP: CBC ---
                t0 = time.time()
                s_c, _, st_c = solve_lwe_milp(
                    A, b, args.q, sec_type, 'cbc',
                    time_limit=args.time_limit,
                )
                t_c = time.time() - t0
                ok_c = (s_c is not None) and np.array_equal(s_c, s_true)

                # --- LLL ---
                t0 = time.time()
                s_l, _, _ = solve_lwe_lattice(
                    A, b, args.q, 'LLL', time_limit=args.time_limit
                )
                t_l = time.time() - t0
                ok_l = (s_l is not None) and np.array_equal(s_l, s_true)

                # --- BKZ ---
                t0 = time.time()
                s_b, _, _ = solve_lwe_lattice(
                    A, b, args.q, 'BKZ',
                    beta=args.bkz_beta, time_limit=args.time_limit,
                )
                t_b = time.time() - t0
                ok_b = (s_b is not None) and np.array_equal(s_b, s_true)

                done += 1
                print(f"  [{done:>3}/{total}] inst={inst:>2}: "
                      f"HiGHS={'OK' if ok_h else 'X'}({t_h:>6.2f}s) | "
                      f"CBC={'OK' if ok_c else 'X'}({t_c:>6.2f}s) | "
                      f"LLL={'OK' if ok_l else 'X'}({t_l:>6.2f}s) | "
                      f"BKZ-{args.bkz_beta}={'OK' if ok_b else 'X'}({t_b:>6.2f}s)")

                results.append({
                    'n': n, 'q': args.q, 'm': m, 'secret_type': sec_type,
                    'instance': inst,
                    'highs_success': bool(ok_h),
                    'highs_time': t_h,
                    'highs_status': st_h,
                    'cbc_success': bool(ok_c),
                    'cbc_time': t_c,
                    'cbc_status': st_c,
                    'lll_success': bool(ok_l),
                    'lll_time': t_l,
                    'bkz_success': bool(ok_b),
                    'bkz_time': t_b,
                    'bkz_beta': args.bkz_beta,
                })

                # Incremental save so crashes don't lose progress
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print(f"Benchmark complete — {len(results)} results saved to {results_file}")
    print("=" * 70)

    print(f"\n{'n':<5} {'Secret':<10} {'HiGHS':<9} {'CBC':<9} {'LLL':<9} {'BKZ':<9}")
    print("-" * 55)

    def rate(sub, key):
        return sum(r[key] for r in sub) / len(sub) * 100 if sub else 0.0

    for n in args.dimensions:
        for sec_type in args.secret_types:
            sub = [r for r in results if r['n'] == n and r['secret_type'] == sec_type]
            if sub:
                print(f"{n:<5} {sec_type:<10} "
                      f"{rate(sub, 'highs_success'):>6.1f}%  "
                      f"{rate(sub, 'cbc_success'):>6.1f}%  "
                      f"{rate(sub, 'lll_success'):>6.1f}%  "
                      f"{rate(sub, 'bkz_success'):>6.1f}%")

    # Solver-vs-solver summary
    print("\n" + "=" * 70)
    print("HiGHS vs. CBC Solver Comparison")
    print("=" * 70)
    n_records = len(results)
    highs_wins = sum(1 for r in results
                     if r['highs_time'] < r['cbc_time'] and r['highs_success'])
    cbc_wins = sum(1 for r in results
                   if r['cbc_time'] < r['highs_time'] and r['cbc_success'])
    ties = n_records - highs_wins - cbc_wins
    mean_h = float(np.mean([r['highs_time'] for r in results]))
    mean_c = float(np.mean([r['cbc_time'] for r in results]))
    print(f"  HiGHS mean time: {mean_h:.3f}s  |  CBC mean time: {mean_c:.3f}s")
    print(f"  HiGHS faster:    {highs_wins}/{n_records} "
          f"({highs_wins / n_records * 100:.1f}%)")
    print(f"  CBC faster:      {cbc_wins}/{n_records} "
          f"({cbc_wins / n_records * 100:.1f}%)")
    print(f"  Ties:            {ties}/{n_records}")
    if mean_h > 0:
        print(f"  Overall speedup (CBC/HiGHS): {mean_c / mean_h:.2f}x")

    return results


# ============================================================
# 5. ARGUMENT PARSER
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='LWE Benchmark: MILP (HiGHS/CBC) vs. LLL/BKZ'
    )
    parser.add_argument('--dimensions', type=int, nargs='+',
                        default=[5, 10, 15, 20])
    parser.add_argument('--secret-types', nargs='+',
                        default=['ternary', 'sparse', 'uniform'])
    parser.add_argument('--instances', type=int, default=10)
    parser.add_argument('--q', type=int, default=97)
    parser.add_argument('--sample-ratio', type=int, default=2)
    parser.add_argument('--time-limit', type=int, default=60)
    parser.add_argument('--bkz-beta', type=int, default=15)
    parser.add_argument('--seed-base', type=int, default=1000)
    parser.add_argument('--output-dir', default='results')
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())