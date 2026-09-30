#!/usr/bin/env python3
"""
LWE Benchmark: MILP (CBC) vs. Lattice Reduction (LLL/BKZ)
Optimized for GitHub Actions (Ubuntu).
"""

import numpy as np
import time
import os
import json
import argparse
from datetime import datetime
from pulp import LpProblem, LpVariable, LpInteger, lpSum, LpMinimize, PULP_CBC_CMD, value
from fpylll import IntegerMatrix, LLL, BKZ


# ============================================================
# 1. LWE INSTANCE GENERATION
# ============================================================
def generate_lwe_instance(n, q, m, secret_type='ternary', error_sigma=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if secret_type == 'ternary':
        s_true = np.random.choice([-1, 0, 1], size=n)
    elif secret_type == 'sparse':
        s_true = np.zeros(n, dtype=int)
        idx = np.random.choice(n, size=min(3, n), replace=False)
        s_true[idx] = np.random.choice([-1, 1], size=len(idx))
    else:
        s_true = np.random.randint(0, q, size=n)

    A = np.random.randint(0, q, size=(m, n))
    e_true = np.round(np.random.normal(0, error_sigma, size=m)).astype(int)
    e_true = np.clip(e_true, -5, 5)
    b = (A @ s_true + e_true) % q
    return A, b, s_true, e_true


# ============================================================
# 2. MILP ATTACK (PuLP + CBC)
# ============================================================
def solve_lwe_milp(A, b, q, secret_type='ternary', time_limit=60):
    m, n = A.shape
    prob = LpProblem("LWE_Search", LpMinimize)

    s_vars = [LpVariable(f"s_{j}", lowBound=-q, upBound=q, cat=LpInteger) for j in range(n)]
    k_vars = [LpVariable(f"k_{i}", lowBound=-q, upBound=q, cat=LpInteger) for i in range(m)]
    e_plus = [LpVariable(f"ep_{i}", lowBound=0, upBound=q, cat=LpInteger) for i in range(m)]
    e_minus = [LpVariable(f"em_{i}", lowBound=0, upBound=q, cat=LpInteger) for i in range(m)]

    if secret_type == 'ternary':
        s_pos = [LpVariable(f"sp_{j}", cat='Binary') for j in range(n)]
        s_neg = [LpVariable(f"sn_{j}", cat='Binary') for j in range(n)]
        for j in range(n):
            prob += s_vars[j] == s_pos[j] - s_neg[j]
            prob += s_pos[j] + s_neg[j] <= 1
    elif secret_type == 'sparse':
        z_vars = [LpVariable(f"z_{j}", cat='Binary') for j in range(n)]
        for j in range(n):
            prob += s_vars[j] <= q * z_vars[j]
            prob += s_vars[j] >= -q * z_vars[j]
        prob += lpSum(z_vars) <= 4

    prob += lpSum(e_plus[i] + e_minus[i] for i in range(m))

    for i in range(m):
        prob += lpSum(A[i, j] * s_vars[j] for j in range(n)) + e_plus[i] - e_minus[i] - q * k_vars[i] == b[i]

    start = time.time()
    try:
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit))
        elapsed = time.time() - start
        status = prob.status
    except Exception as e:
        print(f"  [MILP ERROR] {e}")
        return None, time_limit, 'ERROR'

    if status == 1:
        s_rec = np.array([value(s_vars[j]) for j in range(n)], dtype=int)
        return s_rec, elapsed, 'SUCCESS'
    return None, elapsed, 'FAIL'


# ============================================================
# 3. LATTICE REDUCTION ATTACK (fpylll)
# ============================================================
def solve_lwe_lattice(A, b, q, method='LLL', beta=20, time_limit=60):
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
        print(f"  [{method} ERROR] {e}")
        return None, time_limit, 'ERROR'

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
    print("=" * 70)
    print(f"LWE Benchmark — Started at {datetime.now().isoformat()}")
    print(f"Dimensions: {args.dimensions}")
    print(f"Secret types: {args.secret_types}")
    print(f"Instances per config: {args.instances}")
    print(f"Time limit per attack: {args.time_limit}s")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    results_file = os.path.join(args.output_dir, 'benchmark_results.json')

    total_configs = len(args.dimensions) * len(args.secret_types) * args.instances
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

                milp_start = time.time()
                s_milp, _, _ = solve_lwe_milp(A, b, args.q, sec_type, args.time_limit)
                milp_time = time.time() - milp_start
                milp_ok = (s_milp is not None) and np.array_equal(s_milp, s_true)

                lll_start = time.time()
                s_lll, _, _ = solve_lwe_lattice(A, b, args.q, method='LLL', time_limit=args.time_limit)
                lll_time = time.time() - lll_start
                lll_ok = (s_lll is not None) and np.array_equal(s_lll, s_true)

                bkz_start = time.time()
                s_bkz, _, _ = solve_lwe_lattice(A, b, args.q, method='BKZ', beta=args.bkz_beta, time_limit=args.time_limit)
                bkz_time = time.time() - bkz_start
                bkz_ok = (s_bkz is not None) and np.array_equal(s_bkz, s_true)

                done += 1
                print(f"  [{done}/{total_configs}] inst={inst}: "
                      f"MILP={'OK' if milp_ok else 'X'}({milp_time:.2f}s) | "
                      f"LLL={'OK' if lll_ok else 'X'}({lll_time:.2f}s) | "
                      f"BKZ={'OK' if bkz_ok else 'X'}({bkz_time:.2f}s)")

                results.append({
                    'n': n, 'q': args.q, 'm': m, 'secret_type': sec_type,
                    'instance': inst,
                    'milp_success': bool(milp_ok), 'milp_time': milp_time,
                    'lll_success': bool(lll_ok), 'lll_time': lll_time,
                    'bkz_success': bool(bkz_ok), 'bkz_time': bkz_time,
                    'bkz_beta': args.bkz_beta,
                })

                # Save incrementally so crashes don't lose progress
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Benchmark complete — {len(results)} results saved to {results_file}")
    print("=" * 70)

    # Summary table
    print(f"\n{'n':<5} {'Secret':<10} {'MILP':<8} {'LLL':<8} {'BKZ':<8}")
    print("-" * 45)
    for n in args.dimensions:
        for sec_type in args.secret_types:
            sub = [r for r in results if r['n'] == n and r['secret_type'] == sec_type]
            if sub:
                milp_rate = sum(r['milp_success'] for r in sub) / len(sub) * 100
                lll_rate = sum(r['lll_success'] for r in sub) / len(sub) * 100
                bkz_rate = sum(r['bkz_success'] for r in sub) / len(sub) * 100
                print(f"{n:<5} {sec_type:<10} {milp_rate:>5.1f}%  {lll_rate:>5.1f}%  {bkz_rate:>5.1f}%")

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimensions', type=int, nargs='+', default=[5, 10, 15, 20])
    parser.add_argument('--secret-types', nargs='+', default=['ternary', 'sparse', 'uniform'])
    parser.add_argument('--instances', type=int, default=5)
    parser.add_argument('--q', type=int, default=97)
    parser.add_argument('--sample-ratio', type=int, default=2)
    parser.add_argument('--time-limit', type=int, default=60)
    parser.add_argument('--bkz-beta', type=int, default=10)
    parser.add_argument('--seed-base', type=int, default=1000)
    parser.add_argument('--output-dir', default='results')
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())