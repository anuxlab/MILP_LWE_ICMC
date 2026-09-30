#!/usr/bin/env python3
"""Quick sanity test for HiGHS (Python API) and CBC solvers + LLL."""

import numpy as np
import time
import shutil
import os
from pulp import (
    LpProblem, LpVariable, LpInteger, lpSum, LpMinimize,
    PULP_CBC_CMD, HiGHS_CMD, value,
)
from fpylll import IntegerMatrix, LLL

try:
    from pulp import HiGHS as HiGHS_PY
    HAS_HIGHS_PY = True
except ImportError:
    HAS_HIGHS_PY = False


def find_highs():
    path = shutil.which('highs')
    if path is None:
        for candidate in ['/usr/local/bin/highs', '/usr/bin/highs']:
            if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return path


def build_and_solve(A, b, q, solver_name, time_limit=30, highs_path=None):
    m, n = A.shape
    ERROR_BOUND = 20

    prob = LpProblem(f"LWE_{solver_name}", LpMinimize)
    s_vars = [LpVariable(f"s_{j}", lowBound=-1, upBound=1, cat=LpInteger) for j in range(n)]
    k_vars = [LpVariable(f"k_{i}", lowBound=-100, upBound=100, cat=LpInteger) for i in range(m)]
    e_plus = [LpVariable(f"ep_{i}", lowBound=0, upBound=ERROR_BOUND, cat=LpInteger) for i in range(m)]
    e_minus = [LpVariable(f"em_{i}", lowBound=0, upBound=ERROR_BOUND, cat=LpInteger) for i in range(m)]

    for i in range(m):
        prob += lpSum(A[i, j] * s_vars[j] for j in range(n)) + e_plus[i] - e_minus[i] - q * k_vars[i] == b[i]
    prob += lpSum(e_plus[i] + e_minus[i] for i in range(m))

    start = time.time()
    if solver_name == 'highs':
        if highs_path:
            prob.solve(HiGHS_CMD(path=highs_path, msg=False, timeLimit=time_limit))
        elif HAS_HIGHS_PY:
            print("    [INFO] Using native HiGHS Python API")
            prob.solve(HiGHS_PY(msg=False, timeLimit=time_limit))
        else:
            print("    [WARN] HiGHS unavailable; falling back to CBC")
            prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=2))
    else:
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=2))
    elapsed = time.time() - start

    try:
        vals = [value(v) for v in s_vars]
    except Exception:
        vals = [None] * n

    if all(v is not None for v in vals):
        s_rec = np.array([int(round(v)) for v in vals], dtype=int)
        return s_rec, elapsed, 'SUCCESS'
    return None, elapsed, 'FAIL'


print("=" * 60)
print("Quick Sanity Test — HiGHS + CBC + LLL")
print("=" * 60)

highs_path = find_highs()
print(f"HiGHS CLI path:        {highs_path if highs_path else 'NOT FOUND'}")
print(f"HiGHS Python API:      {'available' if HAS_HIGHS_PY else 'NOT available'}")

n, q, m = 5, 97, 10
np.random.seed(42)

s_true = np.random.choice([-1, 0, 1], size=n)
A = np.random.randint(0, q, size=(m, n))
e_true = np.round(np.random.normal(0, 1, size=m)).astype(int)
b = (A @ s_true + e_true) % q

print(f"True secret: {s_true}")

print("\n[1] MILP — HiGHS solver")
s_h, t_h, st_h = build_and_solve(A, b, q, 'highs', highs_path=highs_path)
ok_h = s_h is not None and np.array_equal(s_h, s_true)
print(f"    Status: {st_h}, Time: {t_h:.3f}s, Match: {ok_h}")

print("\n[2] MILP — CBC solver")
s_c, t_c, st_c = build_and_solve(A, b, q, 'cbc')
ok_c = s_c is not None and np.array_equal(s_c, s_true)
print(f"    Status: {st_c}, Time: {t_c:.3f}s, Match: {ok_c}")

print("\n[3] LLL attack")
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
LLL.reduction(B)
print(f"    LLL reduced {d}-dim basis in {time.time() - start:.4f}s")

print("\n" + "=" * 60)
print("Both MILP solvers OK — ready for full benchmark!" if ok_h and ok_c
      else "[WARN] Some solver failed — check logs above.")
print("=" * 60)