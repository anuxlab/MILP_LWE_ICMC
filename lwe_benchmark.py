#!/usr/bin/env python3
"""
LWE Benchmark: MILP (HiGHS + CBC) vs. Lattice Reduction (LLL/BKZ)

Robust resumable design:
  - Per-instance atomic checkpointing: each result is written to disk
    immediately after it's computed, so nothing is ever lost.
  - Resume: reruns pick up where the last run stopped.
  - Accepts any feasible MILP solution (not just proven-optimal),
    fixing a known pulp.HiGHS status bug on small MIPs.
  - Per-secret-type time limits: uniform secrets (known-hard) get a
    shorter budget than ternary/sparse.
  - LLL/BKZ verification compares mod q, so signed ternary/sparse
    secrets match their canonical [0, q-1] representatives.
  - Aggregates all checkpoints into results/benchmark_results.json
    at the end of each run.
"""

import numpy as np
import time
import os
import json
import shutil
import argparse
import sys
from datetime import datetime
from pathlib import Path
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
# PATHS & CHECKPOINT HELPERS
# ============================================================
RESULTS_DIR = Path('results')
CHECKPOINT_DIR = RESULTS_DIR / 'checkpoints'
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)


def instance_key(n, q, m, sec_type, inst):
    """Stable key for a single instance."""
    return f'n{n}_q{q}_m{m}_{sec_type}_i{inst}'


def load_checkpoint(key):
    """Load a checkpoint if it exists and is valid."""
    path = CHECKPOINT_DIR / f'{key}.json'
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"  [WARN] Corrupt checkpoint {key}: {e}")
    return None


def save_checkpoint(key, record):
    """Atomically write a checkpoint (write-then-rename)."""
    path = CHECKPOINT_DIR / f'{key}.json'
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(record, f, indent=2)
    tmp.replace(path)  # atomic on POSIX


def rebuild_aggregate():
    """Rebuild results/benchmark_results.json from all checkpoints."""
    records = []
    for path in sorted(CHECKPOINT_DIR.glob('*.json')):
        try:
            with open(path) as f:
                records.append(json.load(f))
        except Exception:
            pass
    with open(RESULTS_DIR / 'benchmark_results.json', 'w') as f:
        json.dump(records, f, indent=2)
    return records


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
# LWE INSTANCE GENERATION
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
# MILP ATTACK
# ============================================================
def solve_lwe_milp(A, b, q, secret_type='ternary', solver='highs',
                   time_limit=60, highs_path=None):
    """
    Solve Search-LWE with MILP.

    Solver selection:
      highs + CLI path       → HiGHS_CMD
      highs + Python API     → pulp.HiGHS
      highs + neither        → CBC fallback
      cbc                    → CBC
    """
    m, n = A.shape
    ERROR_BOUND = 20  # tight bound; true errors are ~N(0,1)

    prob = LpProblem(f"LWE_{solver}", LpMinimize)

    if secret_type in ('ternary', 'sparse'):
        s_low, s_high = -1, 1
    else:
        s_low, s_high = 0, q - 1

    s_vars = [LpVariable(f"s_{j}", lowBound=s_low, upBound=s_high, cat=LpInteger) for j in range(n)]
    k_vars = [LpVariable(f"k_{i}", lowBound=-100, upBound=100, cat=LpInteger) for i in range(m)]
    e_plus = [LpVariable(f"ep_{i}", lowBound=0, upBound=ERROR_BOUND, cat=LpInteger) for i in range(m)]
    e_minus = [LpVariable(f"em_{i}", lowBound=0, upBound=ERROR_BOUND, cat=LpInteger) for i in range(m)]

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

    prob += lpSum(e_plus[i] + e_minus[i] for i in range(m))

    for i in range(m):
        prob += lpSum(A[i, j] * s_vars[j] for j in range(n)) + e_plus[i] - e_minus[i] - q * k_vars[i] == b[i]

    start = time.time()
    try:
        if solver == 'highs' and highs_path:
            prob.solve(HiGHS_CMD(path=highs_path, msg=False, timeLimit=time_limit))
        elif solver == 'highs' and HAS_HIGHS_PY:
            prob.solve(HiGHS_PY(msg=False, timeLimit=time_limit))
        elif solver == 'highs':
            prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=2))
        else:
            prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=2))
        elapsed = time.time() - start
    except Exception as e:
        print(f"    [{solver.upper()} ERROR] {e}")
        return None, time.time() - start, 'ERROR'

    # ---- ROBUST SOLUTION EXTRACTION ----
    # Accept ANY status that produced values for the secret variables.
    # We do NOT require proven optimality; the outer loop verifies
    # whether the recovered secret is actually correct. This fixes
    # the pulp.HiGHS status bug where feasible solutions are marked
    # as non-optimal for small MIPs.
    try:
        vals = [value(v) for v in s_vars]
    except Exception:
        vals = [None] * n

    if all(v is not None for v in vals):
        s_rec = np.array([int(round(v)) for v in vals], dtype=int)
        return s_rec, elapsed, 'SUCCESS'
    return None, elapsed, 'FAIL'


# ============================================================
# LATTICE REDUCTION ATTACK
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

    # Look for the short target vector among the first few basis rows.
    # The recovered candidate is mapped into [0, q-1] via mod q, so
    # verification later must compare mod q as well.
    for i in range(min(5, d)):
        v = [B[i, j] for j in range(d)]
        s_cand = np.array(v[m:m + n], dtype=int) % q
        residual = (A @ s_cand - b) % q
        if np.all(np.minimum(residual, q - residual) <= 5):
            return s_cand, elapsed, 'SUCCESS'

    return None, time.time() - start, 'FAIL'


# ============================================================
# TIME LIMITS PER SECRET TYPE
# ============================================================
def get_time_limit(sec_type, base_limit):
    """Uniform secrets are known-hard for MILP; give them a smaller budget."""
    if sec_type == 'uniform':
        return min(base_limit, 10)
    return base_limit


# ============================================================
# MOD-Q SECRET COMPARISON
# ============================================================
def secrets_match(candidate, truth, q):
    """
    Compare a candidate secret against ground truth, modulo q.

    LLL/BKZ return values in [0, q-1] (after % q), while ternary
    and sparse secrets use signed representatives in {-1, 0, 1}.
    Comparing mod q puts both sides in a common domain:
        -1 mod 97 == 96 mod 97  → True
        0  mod 97 == 0  mod 97  → True
    """
    if candidate is None:
        return False
    return np.array_equal(np.asarray(candidate) % q, np.asarray(truth) % q)


# ============================================================
# MAIN BENCHMARK
# ============================================================
def run_benchmark(args):
    highs_path = find_highs()

    print("=" * 70)
    print(f"LWE Benchmark (Resumable) — {datetime.now().isoformat()}")
    print(f"HiGHS CLI path:    {highs_path if highs_path else 'NOT FOUND'}")
    print(f"HiGHS Python API:  {'available' if HAS_HIGHS_PY else 'NOT available'}")
    print(f"Dimensions:        {args.dimensions}")
    print(f"Secret types:      {args.secret_types}")
    print(f"Instances/config:  {args.instances}")
    print(f"Base time limit:   {args.time_limit}s")
    print(f"BKZ β:             {args.bkz_beta}")
    print(f"Force rerun:       {args.force_rerun}")
    print(f"Checkpoint dir:    {CHECKPOINT_DIR}")
    print("=" * 70)
    sys.stdout.flush()

    total = len(args.dimensions) * len(args.secret_types) * args.instances
    done = 0
    skipped = 0
    solved = 0
    failed = 0
    last_heartbeat = time.time()

    for n in args.dimensions:
        m = args.sample_ratio * n
        for sec_type in args.secret_types:
            tl = get_time_limit(sec_type, args.time_limit)
            print(f"\n>>> n={n}, q={args.q}, m={m}, secret={sec_type} (time limit {tl}s)")
            sys.stdout.flush()

            for inst in range(args.instances):
                done += 1
                key = instance_key(n, args.q, m, sec_type, inst)

                # --- Checkpoint check ---
                cached = None if args.force_rerun else load_checkpoint(key)
                if cached is not None:
                    skipped += 1
                    if cached.get('verified', False):
                        solved += 1
                    print(f"  [{done:>3}/{total}] {key}: RESUMED "
                          f"(HiGHS={'OK' if cached.get('highs_success') else 'X'}, "
                          f"CBC={'OK' if cached.get('cbc_success') else 'X'}, "
                          f"LLL={'OK' if cached.get('lll_success') else 'X'}, "
                          f"BKZ={'OK' if cached.get('bkz_success') else 'X'})")
                    sys.stdout.flush()
                    continue

                # --- Fresh computation ---
                seed = args.seed_base + inst
                A, b, s_true, e_true = generate_lwe_instance(
                    n, args.q, m, sec_type, seed=seed
                )

                # --- HiGHS ---
                t0 = time.time()
                s_h, _, st_h = solve_lwe_milp(A, b, args.q, sec_type, 'highs',
                                               time_limit=tl, highs_path=highs_path)
                t_h = time.time() - t0
                ok_h = secrets_match(s_h, s_true, args.q)

                # --- CBC ---
                t0 = time.time()
                s_c, _, st_c = solve_lwe_milp(A, b, args.q, sec_type, 'cbc',
                                               time_limit=tl)
                t_c = time.time() - t0
                ok_c = secrets_match(s_c, s_true, args.q)

                # --- LLL ---
                t0 = time.time()
                s_l, _, _ = solve_lwe_lattice(A, b, args.q, 'LLL', time_limit=tl)
                t_l = time.time() - t0
                ok_l = secrets_match(s_l, s_true, args.q)

                # --- BKZ ---
                t0 = time.time()
                s_b, _, _ = solve_lwe_lattice(A, b, args.q, 'BKZ',
                                               beta=args.bkz_beta, time_limit=tl)
                t_b = time.time() - t0
                ok_b = secrets_match(s_b, s_true, args.q)

                verified = ok_h or ok_c or ok_l or ok_b
                if verified:
                    solved += 1
                else:
                    failed += 1

                record = {
                    'n': n, 'q': args.q, 'm': m, 'secret_type': sec_type,
                    'instance': inst, 'key': key, 'seed': seed,
                    'highs_success': bool(ok_h), 'highs_time': t_h, 'highs_status': st_h,
                    'cbc_success': bool(ok_c), 'cbc_time': t_c, 'cbc_status': st_c,
                    'lll_success': bool(ok_l), 'lll_time': t_l,
                    'bkz_success': bool(ok_b), 'bkz_time': t_b, 'bkz_beta': args.bkz_beta,
                    'verified': bool(verified),
                    'timestamp': datetime.now().isoformat(),
                }

                save_checkpoint(key, record)

                print(f"  [{done:>3}/{total}] {key}: "
                      f"HiGHS={'OK' if ok_h else 'X'}({t_h:>6.2f}s) | "
                      f"CBC={'OK' if ok_c else 'X'}({t_c:>6.2f}s) | "
                      f"LLL={'OK' if ok_l else 'X'}({t_l:>6.2f}s) | "
                      f"BKZ-{args.bkz_beta}={'OK' if ok_b else 'X'}({t_b:>6.2f}s)")
                sys.stdout.flush()

                # Heartbeat every 60s
                if time.time() - last_heartbeat > 60:
                    print(f"  [HEARTBEAT] done={done} skipped={skipped} "
                          f"solved={solved} failed={failed}")
                    sys.stdout.flush()
                    last_heartbeat = time.time()

    # --- Aggregate ---
    print("\n" + "=" * 70)
    print("Rebuilding aggregate results from checkpoints...")
    records = rebuild_aggregate()
    print(f"Aggregate written: {RESULTS_DIR / 'benchmark_results.json'} ({len(records)} records)")
    print("=" * 70)

    # --- Summary table ---
    def rate(sub, key):
        return sum(r[key] for r in sub) / len(sub) * 100 if sub else 0.0

    print(f"\n{'n':<5} {'Secret':<10} {'HiGHS':<9} {'CBC':<9} {'LLL':<9} {'BKZ':<9}")
    print("-" * 55)
    for n in args.dimensions:
        for sec_type in args.secret_types:
            sub = [r for r in records if r['n'] == n and r['secret_type'] == sec_type]
            if sub:
                print(f"{n:<5} {sec_type:<10} "
                      f"{rate(sub, 'highs_success'):>6.1f}%  "
                      f"{rate(sub, 'cbc_success'):>6.1f}%  "
                      f"{rate(sub, 'lll_success'):>6.1f}%  "
                      f"{rate(sub, 'bkz_success'):>6.1f}%")

    # --- Solver summary ---
    print("\n" + "=" * 70)
    print("HiGHS vs. CBC Summary")
    print("=" * 70)
    n_records = len(records)
    if n_records:
        mean_h = float(np.mean([r['highs_time'] for r in records]))
        mean_c = float(np.mean([r['cbc_time'] for r in records]))
        print(f"  HiGHS mean: {mean_h:.3f}s  |  CBC mean: {mean_c:.3f}s")
        if mean_h > 0:
            print(f"  Speedup (CBC/HiGHS): {mean_c / mean_h:.2f}x")

    print(f"\n  Total: {n_records} records  |  Skipped: {skipped}  |  "
          f"Solved: {solved}  |  Failed: {failed}")

    return records


# ============================================================
# ARGS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='LWE Benchmark: MILP (HiGHS/CBC) vs. LLL/BKZ (resumable)'
    )
    parser.add_argument('--dimensions', type=int, nargs='+', default=[5, 10, 15])
    parser.add_argument('--secret-types', nargs='+',
                        default=['ternary', 'sparse', 'uniform'])
    parser.add_argument('--instances', type=int, default=10)
    parser.add_argument('--q', type=int, default=97)
    parser.add_argument('--sample-ratio', type=int, default=2)
    parser.add_argument('--time-limit', type=int, default=60)
    parser.add_argument('--bkz-beta', type=int, default=15)
    parser.add_argument('--seed-base', type=int, default=1000)
    parser.add_argument('--force-rerun', action='store_true', default=False,
                        help='Ignore checkpoints and rerun all instances')
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())