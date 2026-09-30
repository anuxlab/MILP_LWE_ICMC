#!/usr/bin/env python3
"""Quick sanity test for both HiGHS and CBC solvers + LLL."""

import numpy as np
import time
import shutil
from pulp import (
    LpProblem, LpVariable, LpInteger, lpSum, LpMinimize,
    PULP_CBC_CMD, HiGHS_CMD, value,
)
from fpylll import IntegerMatrix, LLL


def find_highs():
    """Locate the HiGHS executable."""
    path = shutil.which('highs')
    if path is None:
        # Check common installation paths
        for candidate in ['/usr/bin/highs', '/usr/local/bin/highs', '/opt/highs/bin/highs']:
            if shutil.which(candidate):
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
        else:
            # Fallback to CBC if HiGHS binary is missing
            print("    [WARN] HiGHS binary not found, falling back to CBC")
            prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=2))
    else:
        prob.solve(PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=2))
    elapsed = time.time() - start

    if prob.status == 1:
        s_rec = np.array([value(s_vars[j]) for j in range(n)], dtype=int)
        return s_rec, elapsed, 'SUCCESS'
    return None, elapsed, f'FAIL(status={prob.status})'


print("=" * 60)
print("Quick Sanity Test — HiGHS + CBC + LLL")
print("=" * 60)

# Detect HiGHS
highs_path = find_highs()
if highs_path:
    print(f"HiGHS executable found at: {highs_path}")
else:
    print("HiGHS executable NOT found — will fall back to CBC")

n, q, m = 5, 97, 10
np.random.seed(42)

s_true = np.random.choice([-1, 0, 1], size=n)
A = np.random.randint(0, q, size=(m, n))
e_true = np.round(np.random.normal(0, 1, size=m)).astype(int)
b = (A @ s_true + e_true) % q

print(f"True secret: {s_true}")

print("\n[1] MILP — HiGHS solver")
s_h, t_h, st_h = build_and_solve(A, b, q, 'highs', highs_path=highs_path)
print(f"    Status: {st_h}, Time: {t_h:.3f}s, Match: {np.array_equal(s_h, s_true) if s_h is not None else False}")

print("\n[2] MILP — CBC solver")
s_c, t_c, st_c = build_and_solve(A, b, q, 'cbc')
print(f"    Status: {st_c}, Time: {t_c:.3f}s, Match: {np.array_equal(s_c, s_true) if s_c is not None else False}")

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
print("All solvers OK — ready for full benchmark!")
print("=" * 60)