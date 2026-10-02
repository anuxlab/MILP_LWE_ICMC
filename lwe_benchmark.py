#!/usr/bin/env python3
"""
LWE Benchmark: MILP (HiGHS + CBC) vs. Lattice Reduction (LLL/BKZ)

Core features
-------------
  - Resumable per-instance checkpointing (atomic writes, no data loss)
  - Robust MILP solution extraction (accepts feasible-but-unproven)
  - LLL/BKZ sign-ambiguity handling (tests both (e,s,-1) and (-e,-s,1))
  - Mod-q secret comparison for ternary/sparse secrets
  - Per-secret-type time limits (uniform secrets capped at 10s)

Extension 2 — two-factor ablation
---------------------------------
  --support-mode     weight_cap (Σz_j ≤ w) or no_cap
  --error-bound      explicit E for the MILP model

Extension 3 — modulus sensitivity under two noise policies
-----------------------------------------------------------
  --q-list           multiple moduli in one run
  --error-sigma-mode fixed_abs (σ=1) or fixed_rel (σ=q/100)
  --clip-factor      clipping multiplier for errors

Extension 4 — crossed modulus × noise grid
-------------------------------------------
  --error-sigma      explicit σ value, overrides --error-sigma-mode

Reproducibility enhancements
----------------------------
  - Raw instance artifacts (A, b, s_true, e_true) stored per checkpoint
    as base64-encoded int16/int8 arrays, with SHA-256 hashes
  - Recovered candidate s_hat stored even on failure, with residual norms
  - Per-checkpoint git commit hash for full provenance
  - Instance statistics: rank_A, ||s_true||, ||e_true||, timing breakdown
  - Optional full solver logs via --capture-logs
  --no-artifacts     disable artifact storage (smaller checkpoints only)

NOTE ON SCOPE
-------------
  This benchmark studies plain-LWE instances with dense random matrices.
  It does NOT reproduce the module or ring structure of ML-KEM or NewHope.
"""

import numpy as np
import time
import os
import json
import shutil
import argparse
import sys
import base64
import hashlib
import subprocess
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
# PATHS & GIT PROVENANCE
# ============================================================
RESULTS_DIR = Path('results')
CHECKPOINT_DIR = RESULTS_DIR / 'checkpoints'
LOGS_DIR = RESULTS_DIR / 'logs'
ARTIFACTS_DIR = RESULTS_DIR / 'instances'
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)


def get_git_commit():
    """Return the current git commit hash, or 'unknown' on failure."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


GIT_COMMIT = get_git_commit()


# ============================================================
# ARRAY ENCODING HELPERS
# ============================================================
def encode_array(arr, dtype):
    """Base64-encode a numpy array after casting to dtype."""
    return base64.b64encode(arr.astype(dtype).tobytes()).decode('ascii')


def hash_array(arr, dtype):
    """SHA-256 of an array's raw bytes."""
    return hashlib.sha256(arr.astype(dtype).tobytes()).hexdigest()


def save_instance_artifact(key, A, b, s_true, e_true, q,
                           dtype_A='int16', dtype_s='int8'):
    """
    Persist the raw instance arrays alongside the checkpoint.

    Enables full re-analysis years later without depending on the
    specific NumPy RNG implementation.
    """
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = ARTIFACTS_DIR / f'{key}.json'
    payload = {
        'key': key,
        'q': int(q),
        'A_dtype': dtype_A,
        'A_shape': list(A.shape),
        'A_b64': encode_array(A, dtype_A),
        'A_hash': hash_array(A, dtype_A),
        'b_b64': encode_array(b, dtype_A),
        'b_hash': hash_array(b, dtype_A),
        's_true_b64': encode_array(s_true, dtype_s),
        's_true_hash': hash_array(s_true, dtype_s),
        'e_true_b64': encode_array(e_true, dtype_s),
        'e_true_hash': hash_array(e_true, dtype_s),
    }
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    tmp.replace(path)


def load_instance_artifact(key):
    """Reconstruct instance arrays from a stored artifact."""
    path = ARTIFACTS_DIR / f'{key}.json'
    if not path.exists():
        return None
    with open(path) as f:
        payload = json.load(f)

    A_shape = tuple(payload['A_shape'])
    A = np.frombuffer(base64.b64decode(payload['A_b64']),
                      dtype=payload['A_dtype']).reshape(A_shape)
    b = np.frombuffer(base64.b64decode(payload['b_b64']),
                      dtype=payload['A_dtype'])
    s_true = np.frombuffer(base64.b64decode(payload['s_true_b64']),
                           dtype='int8')
    e_true = np.frombuffer(base64.b64decode(payload['e_true_b64']),
                           dtype='int8')

    # Verify hashes
    if hash_array(A, payload['A_dtype']) != payload['A_hash']:
        print(f"  [WARN] A hash mismatch for {key}")
    if hash_array(b, payload['A_dtype']) != payload['b_hash']:
        print(f"  [WARN] b hash mismatch for {key}")

    return A.astype(int), b.astype(int), s_true.astype(int), e_true.astype(int)


# ============================================================
# CHECKPOINT HELPERS
# ============================================================
def instance_key(n, q, m, sec_type, inst,
                 support_mode='weight_cap', error_bound=20,
                 sigma_mode='fixed_abs', explicit_sigma=None):
    """
    Stable key for a single instance.

    Default configuration produces the original key format for
    backward compatibility. Non-default configurations append a
    suffix so ablation and grid checkpoints do not collide with
    baseline.
    """
    base = f'n{n}_q{q}_m{m}_{sec_type}_i{inst}'
    is_default = (support_mode == 'weight_cap'
                  and error_bound == 20
                  and sigma_mode == 'fixed_abs'
                  and explicit_sigma is None)
    if is_default:
        return base

    parts = [base]
    if support_mode != 'weight_cap':
        parts.append(support_mode)
    if error_bound != 20:
        parts.append(f'E{error_bound}')
    if explicit_sigma is not None:
        sig_str = f'{explicit_sigma:g}'.replace('.', 'p')
        parts.append(f'sig{sig_str}')
    elif sigma_mode != 'fixed_abs':
        parts.append(sigma_mode)
    return '_'.join(parts)


def load_checkpoint(key):
    path = CHECKPOINT_DIR / f'{key}.json'
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"  [WARN] Corrupt checkpoint {key}: {e}")
    return None


def save_checkpoint(key, record):
    path = CHECKPOINT_DIR / f'{key}.json'
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(record, f, indent=2)
    tmp.replace(path)


def rebuild_aggregate():
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
    path = shutil.which('highs')
    if path is None:
        for candidate in ['/usr/local/bin/highs', '/usr/bin/highs',
                          '/opt/highs/bin/highs']:
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return path


# ============================================================
# NOISE POLICY HELPERS
# ============================================================
def get_sigma(q, sigma_mode, explicit_sigma=None):
    if explicit_sigma is not None:
        return float(explicit_sigma)
    if sigma_mode == 'fixed_abs':
        return 1.0
    if sigma_mode == 'fixed_rel':
        return q / 100.0
    raise ValueError(f"Unknown sigma_mode: {sigma_mode}")


def get_clip_bound(sigma, clip_factor):
    return int(np.ceil(clip_factor * sigma))


def get_error_bound_for_sigma(sigma, clip_factor, safety_margin=1):
    return int(np.ceil(clip_factor * sigma)) + safety_margin


# ============================================================
# LWE INSTANCE GENERATION
# ============================================================
def generate_lwe_instance(n, q, m, secret_type='ternary',
                          error_sigma=None, sigma_mode='fixed_abs',
                          clip_factor=5.0, seed=None,
                          return_sigma=False):
    """
    Generate a Search-LWE instance.

    Returns (A, b, s_true, e_true) normally, or
    (A, b, s_true, e_true, empirical_sigma) if return_sigma=True.
    """
    if seed is not None:
        np.random.seed(seed)

    sigma = error_sigma if error_sigma is not None \
            else get_sigma(q, sigma_mode)

    if secret_type == 'ternary':
        s_true = np.random.choice([-1, 0, 1], size=n)
    elif secret_type == 'sparse':
        s_true = np.zeros(n, dtype=int)
        idx = np.random.choice(n, size=min(3, n), replace=False)
        s_true[idx] = np.random.choice([-1, 1], size=len(idx))
    else:
        s_true = np.random.randint(0, q, size=n)

    A = np.random.randint(0, q, size=(m, n))

    e_true = np.round(np.random.normal(0, sigma, size=m)).astype(int)
    clip_bound = get_clip_bound(sigma, clip_factor)
    e_true = np.clip(e_true, -clip_bound, clip_bound)

    b = (A @ s_true + e_true) % q

    if return_sigma:
        return A, b, s_true, e_true, float(np.std(e_true))
    return A, b, s_true, e_true


# ============================================================
# MILP ATTACK
# ============================================================
def solve_lwe_milp(A, b, q, secret_type='ternary', solver='highs',
                   time_limit=60, highs_path=None,
                   support_mode='weight_cap', error_bound=None,
                   sigma_mode='fixed_abs', clip_factor=5.0,
                   explicit_sigma=None, log_path=None):
    """
    Solve Search-LWE with MILP.

    Returns (result_dict, elapsed, status) where result_dict contains
    at least 's_hat', 'residual_l2', 'residual_inf'. Even on failure,
    s_hat may be a near-miss candidate.
    """
    m, n = A.shape

    if error_bound is None:
        sigma = get_sigma(q, sigma_mode, explicit_sigma)
        error_bound = get_error_bound_for_sigma(sigma, clip_factor)
    E = error_bound

    t_model_start = time.time()
    prob = LpProblem(f"LWE_{solver}", LpMinimize)

    if secret_type in ('ternary', 'sparse'):
        s_low, s_high = -1, 1
    else:
        s_low, s_high = 0, q - 1

    s_vars = [LpVariable(f"s_{j}", lowBound=s_low, upBound=s_high,
                         cat=LpInteger) for j in range(n)]
    k_vars = [LpVariable(f"k_{i}", lowBound=-100, upBound=100,
                         cat=LpInteger) for i in range(m)]
    e_plus = [LpVariable(f"ep_{i}", lowBound=0, upBound=E,
                         cat=LpInteger) for i in range(m)]
    e_minus = [LpVariable(f"em_{i}", lowBound=0, upBound=E,
                          cat=LpInteger) for i in range(m)]

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
        if support_mode == 'weight_cap':
            prob += lpSum(z_vars) <= 4

    prob += lpSum(e_plus[i] + e_minus[i] for i in range(m))

    for i in range(m):
        prob += (lpSum(A[i, j] * s_vars[j] for j in range(n))
                 + e_plus[i] - e_minus[i] - q * k_vars[i] == b[i])

    t_model_end = time.time()
    model_build_time = t_model_end - t_model_start

    start = time.time()
    try:
        if solver == 'highs' and highs_path:
            prob.solve(HiGHS_CMD(path=highs_path, msg=False,
                                 timeLimit=time_limit))
        elif solver == 'highs' and HAS_HIGHS_PY:
            prob.solve(HiGHS_PY(msg=False, timeLimit=time_limit))
        elif solver == 'highs':
            prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit,
                                    threads=2))
        else:
            kwargs = {'msg': False, 'timeLimit': time_limit, 'threads': 2}
            if log_path:
                kwargs['logPath'] = str(log_path)
            prob.solve(PULP_CBC_CMD(**kwargs))
        elapsed = time.time() - start
    except Exception as e:
        print(f"    [{solver.upper()} ERROR] {e}")
        return ({'s_hat': None, 'residual_l2': None, 'residual_inf': None,
                 'status': 'ERROR', 'model_build_time': model_build_time},
                time.time() - start, 'ERROR')

    # ---- Robust extraction: accept any status with values ----
    try:
        vals = [value(v) for v in s_vars]
    except Exception:
        vals = [None] * n

    if all(v is not None for v in vals):
        s_rec = np.array([int(round(v)) for v in vals], dtype=int)
        residual = (A @ s_rec - b) % q
        residual = np.minimum(residual, q - residual)
        res_l2 = float(np.linalg.norm(residual))
        res_inf = int(np.max(np.abs(residual))) if len(residual) else 0
        return ({
            's_hat': s_rec,
            'residual_l2': res_l2,
            'residual_inf': res_inf,
            'status': 'FEASIBLE',
            'model_build_time': model_build_time,
        }, elapsed, 'SUCCESS')

    return ({
        's_hat': None, 'residual_l2': None, 'residual_inf': None,
        'status': 'INFEASIBLE', 'model_build_time': model_build_time,
    }, elapsed, 'FAIL')


# ============================================================
# LATTICE REDUCTION ATTACK
# ============================================================
def solve_lwe_lattice(A, b, q, method='LLL', beta=20, time_limit=60,
                      error_bound_for_verification=5):
    m, n = A.shape
    d = m + n + 1

    t_model_start = time.time()
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
    model_build_time = time.time() - t_model_start

    start = time.time()
    try:
        if method == 'LLL':
            LLL.reduction(B, delta=0.99)
        elif method == 'BKZ':
            BKZ.reduction(B, BKZ.Param(block_size=beta, max_loops=4))
        elapsed = time.time() - start
    except Exception as e:
        print(f"    [{method} ERROR] {e}")
        return ({'s_hat': None, 'residual_l2': None, 'residual_inf': None,
                 'status': 'ERROR', 'model_build_time': model_build_time},
                time.time() - start, 'ERROR')

    best_residual = None
    best_s = None
    for i in range(d):
        row = [int(B[i, j]) for j in range(d)]
        norm_sq = sum(x * x for x in row)
        if norm_sq > 500:
            continue
        x_raw = np.array(row[m:m + n], dtype=int)
        for x_try in (x_raw, -x_raw):
            s_cand = x_try % q
            residual = (A @ s_cand - b) % q
            residual = np.minimum(residual, q - residual)
            res_inf = int(np.max(np.abs(residual)))
            if best_residual is None or res_inf < best_residual:
                best_residual = res_inf
                best_s = s_cand
            if res_inf <= error_bound_for_verification:
                return ({
                    's_hat': s_cand,
                    'residual_l2': float(np.linalg.norm(residual)),
                    'residual_inf': res_inf,
                    'status': 'FOUND',
                    'model_build_time': model_build_time,
                }, elapsed, 'SUCCESS')

    return ({
        's_hat': best_s,
        'residual_l2': None if best_s is None else float(np.linalg.norm(
            np.minimum((A @ best_s - b) % q, q - (A @ best_s - b) % q))),
        'residual_inf': best_residual,
        'status': 'NOT_FOUND',
        'model_build_time': model_build_time,
    }, time.time() - start, 'FAIL')


# ============================================================
# TIME LIMITS & COMPARISON
# ============================================================
def get_time_limit(sec_type, base_limit):
    if sec_type == 'uniform':
        return min(base_limit, 10)
    return base_limit


def secrets_match(candidate, truth, q):
    if candidate is None:
        return False
    return np.array_equal(np.asarray(candidate) % q, np.asarray(truth) % q)


# ============================================================
# INSTANCE STATISTICS
# ============================================================
def instance_statistics(A, s_true, e_true, q):
    """Cheap pre-analysis statistics recorded per instance."""
    try:
        rank_A = int(np.linalg.matrix_rank(A))
    except Exception:
        rank_A = -1
    return {
        'rank_A': rank_A,
        'norm_s_true_l2': float(np.linalg.norm(s_true)),
        'norm_e_true_l2': float(np.linalg.norm(e_true)),
        'norm_e_true_inf': int(np.max(np.abs(e_true))) if len(e_true) else 0,
    }


# ============================================================
# MAIN BENCHMARK
# ============================================================
def run_benchmark(args):
    highs_path = find_highs()
    q_values = args.q_list if args.q_list is not None else [args.q]

    print("=" * 70)
    print(f"LWE Benchmark (Resumable) — {datetime.now().isoformat()}")
    print(f"Git commit:        {GIT_COMMIT}")
    print(f"HiGHS CLI path:    {highs_path if highs_path else 'NOT FOUND'}")
    print(f"HiGHS Python API:  {'available' if HAS_HIGHS_PY else 'NOT available'}")
    print(f"Moduli:            {q_values}")
    print(f"Dimensions:        {args.dimensions}")
    print(f"Secret types:      {args.secret_types}")
    print(f"Instances/config:  {args.instances}")
    print(f"Base time limit:   {args.time_limit}s")
    print(f"BKZ β:             {args.bkz_beta}")
    print(f"Support mode:      {args.support_mode}")
    print(f"Error bound:       {args.error_bound if args.error_bound is not None else 'derived from sigma'}")
    print(f"Sigma mode:        {args.error_sigma_mode}")
    print(f"Explicit sigma:    {args.error_sigma if args.error_sigma is not None else 'None'}")
    print(f"Clip factor:       {args.clip_factor}× sigma")
    print(f"Capture logs:      {args.capture_logs}")
    print(f"Store artifacts:   {not args.no_artifacts}")
    print(f"Force rerun:       {args.force_rerun}")
    print(f"Checkpoint dir:    {CHECKPOINT_DIR}")
    print("=" * 70)
    sys.stdout.flush()

    total = (len(q_values) * len(args.dimensions)
             * len(args.secret_types) * args.instances)
    done = 0
    skipped = 0
    solved = 0
    failed = 0
    last_heartbeat = time.time()

    for q in q_values:
        sigma = get_sigma(q, args.error_sigma_mode, args.error_sigma)
        clip_bound = get_clip_bound(sigma, args.clip_factor)
        derived_E = (args.error_bound if args.error_bound is not None
                     else get_error_bound_for_sigma(sigma, args.clip_factor))

        print(f"\n{'#' * 70}")
        print(f"# MODULUS q = {q}")
        print(f"#   sigma = {sigma:.4f}, clip_bound = ±{clip_bound}, "
              f"error_bound E = {derived_E}")
        print(f"{'#' * 70}")

        for n in args.dimensions:
            m = args.sample_ratio * n
            for sec_type in args.secret_types:
                tl = get_time_limit(sec_type, args.time_limit)
                print(f"\n>>> n={n}, q={q}, m={m}, secret={sec_type} "
                      f"(time limit {tl}s)")
                sys.stdout.flush()

                for inst in range(args.instances):
                    done += 1
                    key = instance_key(
                        n, q, m, sec_type, inst,
                        support_mode=args.support_mode,
                        error_bound=derived_E,
                        sigma_mode=args.error_sigma_mode,
                        explicit_sigma=args.error_sigma,
                    )

                    cached = None if args.force_rerun else load_checkpoint(key)
                    if cached is not None:
                        skipped += 1
                        if cached.get('verified', False):
                            solved += 1
                        print(f"  [{done:>3}/{total}] {key}: RESUMED")
                        sys.stdout.flush()
                        continue

                    # --- Regenerate or load instance ---
                    artifact = load_instance_artifact(key) if not args.no_artifacts else None
                    if artifact is not None:
                        A, b, s_true, e_true = artifact
                        empirical_sigma = float(np.std(e_true))
                    else:
                        seed = args.seed_base + inst
                        if args.error_sigma is not None:
                            A, b, s_true, e_true, empirical_sigma = generate_lwe_instance(
                                n, q, m, sec_type,
                                error_sigma=args.error_sigma,
                                clip_factor=args.clip_factor,
                                seed=seed, return_sigma=True,
                            )
                        else:
                            A, b, s_true, e_true, empirical_sigma = generate_lwe_instance(
                                n, q, m, sec_type,
                                sigma_mode=args.error_sigma_mode,
                                clip_factor=args.clip_factor,
                                seed=seed, return_sigma=True,
                            )
                        if not args.no_artifacts:
                            save_instance_artifact(key, A, b, s_true, e_true, q)

                    stats = instance_statistics(A, s_true, e_true, q)
                    max_abs_error = stats['norm_e_true_inf']
                    planted_feasible = bool(max_abs_error <= derived_E)

                    # --- Optional log capture ---
                    highs_log = None
                    cbc_log = None
                    if args.capture_logs:
                        LOGS_DIR.mkdir(exist_ok=True)
                        highs_log = LOGS_DIR / f'{key}.highs.log'
                        cbc_log = LOGS_DIR / f'{key}.cbc.log'

                    # --- HiGHS ---
                    t0 = time.time()
                    res_h, _, st_h = solve_lwe_milp(
                        A, b, q, sec_type, 'highs',
                        time_limit=tl, highs_path=highs_path,
                        support_mode=args.support_mode,
                        error_bound=derived_E,
                        sigma_mode=args.error_sigma_mode,
                        clip_factor=args.clip_factor,
                        explicit_sigma=args.error_sigma,
                        log_path=highs_log,
                    )
                    t_h = time.time() - t0
                    ok_h = secrets_match(res_h['s_hat'], s_true, q)

                    # --- CBC ---
                    t0 = time.time()
                    res_c, _, st_c = solve_lwe_milp(
                        A, b, q, sec_type, 'cbc',
                        time_limit=tl,
                        support_mode=args.support_mode,
                        error_bound=derived_E,
                        sigma_mode=args.error_sigma_mode,
                        clip_factor=args.clip_factor,
                        explicit_sigma=args.error_sigma,
                        log_path=cbc_log,
                    )
                    t_c = time.time() - t0
                    ok_c = secrets_match(res_c['s_hat'], s_true, q)

                    # --- LLL ---
                    t0 = time.time()
                    res_l, _, _ = solve_lwe_lattice(
                        A, b, q, 'LLL', time_limit=tl,
                        error_bound_for_verification=clip_bound,
                    )
                    t_l = time.time() - t0
                    ok_l = secrets_match(res_l['s_hat'], s_true, q)

                    # --- BKZ ---
                    t0 = time.time()
                    res_b, _, _ = solve_lwe_lattice(
                        A, b, q, 'BKZ', beta=args.bkz_beta,
                        time_limit=tl,
                        error_bound_for_verification=clip_bound,
                    )
                    t_b = time.time() - t0
                    ok_b = secrets_match(res_b['s_hat'], s_true, q)

                    verified = ok_h or ok_c or ok_l or ok_b
                    if verified:
                        solved += 1
                    else:
                        failed += 1

                    # --- Record ---
                    record = {
                        'n': n, 'q': q, 'm': m,
                        'secret_type': sec_type,
                        'instance': inst, 'key': key,
                        'seed': args.seed_base + inst,
                        'git_commit': GIT_COMMIT,
                        'support_mode': args.support_mode,
                        'error_bound': derived_E,
                        'sigma_mode': args.error_sigma_mode,
                        'explicit_sigma': args.error_sigma,
                        'clip_factor': args.clip_factor,
                        'nominal_sigma': float(sigma),
                        'empirical_sigma': empirical_sigma,
                        'clip_bound': int(clip_bound),
                        'max_abs_error': max_abs_error,
                        'planted_feasible': planted_feasible,
                        'rank_A': stats['rank_A'],
                        'norm_s_true_l2': stats['norm_s_true_l2'],
                        'norm_e_true_l2': stats['norm_e_true_l2'],
                        # HiGHS results
                        'highs_success': bool(ok_h),
                        'highs_time': t_h,
                        'highs_status': st_h,
                        'highs_model_build_time': res_h.get('model_build_time'),
                        'highs_residual_l2': res_h.get('residual_l2'),
                        'highs_residual_inf': res_h.get('residual_inf'),
                        'highs_s_hat_b64': (encode_array(res_h['s_hat'], 'int16')
                                            if res_h.get('s_hat') is not None
                                            else None),
                        # CBC results
                        'cbc_success': bool(ok_c),
                        'cbc_time': t_c,
                        'cbc_status': st_c,
                        'cbc_model_build_time': res_c.get('model_build_time'),
                        'cbc_residual_l2': res_c.get('residual_l2'),
                        'cbc_residual_inf': res_c.get('residual_inf'),
                        'cbc_s_hat_b64': (encode_array(res_c['s_hat'], 'int16')
                                          if res_c.get('s_hat') is not None
                                          else None),
                        # LLL results
                        'lll_success': bool(ok_l),
                        'lll_time': t_l,
                        'lll_residual_inf': res_l.get('residual_inf'),
                        'lll_s_hat_b64': (encode_array(res_l['s_hat'], 'int16')
                                          if res_l.get('s_hat') is not None
                                          else None),
                        # BKZ results
                        'bkz_success': bool(ok_b),
                        'bkz_time': t_b,
                        'bkz_beta': args.bkz_beta,
                        'bkz_residual_inf': res_b.get('residual_inf'),
                        'bkz_s_hat_b64': (encode_array(res_b['s_hat'], 'int16')
                                          if res_b.get('s_hat') is not None
                                          else None),
                        # Meta
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

                    if time.time() - last_heartbeat > 60:
                        print(f"  [HEARTBEAT] done={done} skipped={skipped} "
                              f"solved={solved} failed={failed}")
                        sys.stdout.flush()
                        last_heartbeat = time.time()

    # --- Aggregate ---
    print("\n" + "=" * 70)
    print("Rebuilding aggregate results from checkpoints...")
    records = rebuild_aggregate()
    print(f"Aggregate written: {RESULTS_DIR / 'benchmark_results.json'} "
          f"({len(records)} records)")
    print("=" * 70)

    def rate(sub, k):
        return sum(r[k] for r in sub) / len(sub) * 100 if sub else 0.0

    print(f"\n{'q':<6} {'n':<5} {'Secret':<10} {'HiGHS':<9} {'CBC':<9} "
          f"{'LLL':<9} {'BKZ':<9}")
    print("-" * 65)
    for q in q_values:
        for n in args.dimensions:
            for sec_type in args.secret_types:
                sub = [r for r in records
                       if r['n'] == n and r['q'] == q
                       and r['secret_type'] == sec_type
                       and r.get('support_mode') == args.support_mode
                       and r.get('sigma_mode') == args.error_sigma_mode
                       and r.get('explicit_sigma') == args.error_sigma]
                if sub:
                    print(f"{q:<6} {n:<5} {sec_type:<10} "
                          f"{rate(sub, 'highs_success'):>6.1f}%  "
                          f"{rate(sub, 'cbc_success'):>6.1f}%  "
                          f"{rate(sub, 'lll_success'):>6.1f}%  "
                          f"{rate(sub, 'bkz_success'):>6.1f}%")

    print(f"\n  Total: {len(records)} records  |  Skipped: {skipped}  |  "
          f"Solved: {solved}  |  Failed: {failed}")

    return records


# ============================================================
# ARGS
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='LWE Benchmark: MILP vs. Lattice Reduction'
    )
    parser.add_argument('--dimensions', type=int, nargs='+',
                        default=[5, 10, 15])
    parser.add_argument('--secret-types', nargs='+',
                        default=['ternary', 'sparse', 'uniform'])
    parser.add_argument('--instances', type=int, default=10)
    parser.add_argument('--q', type=int, default=97)
    parser.add_argument('--q-list', type=int, nargs='+', default=None)
    parser.add_argument('--sample-ratio', type=int, default=2)
    parser.add_argument('--time-limit', type=int, default=60)
    parser.add_argument('--bkz-beta', type=int, default=15)
    parser.add_argument('--seed-base', type=int, default=1000)
    parser.add_argument('--force-rerun', action='store_true', default=False)
    parser.add_argument('--output-dir', default='results')

    # Extension 2
    parser.add_argument('--support-mode',
                        choices=['weight_cap', 'no_cap'],
                        default='weight_cap')
    parser.add_argument('--error-bound', type=int, default=None)

    # Extension 3
    parser.add_argument('--error-sigma-mode',
                        choices=['fixed_abs', 'fixed_rel'],
                        default='fixed_abs')
    parser.add_argument('--clip-factor', type=float, default=5.0)

    # Extension 4
    parser.add_argument('--error-sigma', type=float, default=None)

    # Reproducibility
    parser.add_argument('--capture-logs', action='store_true', default=False,
                        help='Capture full solver logs (larger artifacts)')
    parser.add_argument('--no-artifacts', action='store_true', default=False,
                        help='Skip raw instance artifact storage')

    return parser.parse_args()


if __name__ == '__main__':
    run_benchmark(parse_args())