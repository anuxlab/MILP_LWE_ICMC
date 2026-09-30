import numpy as np
import time
from pulp import LpProblem, LpVariable, LpInteger, lpSum, LpMinimize, PULP_CBC_CMD, value
from fpylll import IntegerMatrix, LLL

print("=" * 50)
print("Quick Sanity Test (with tight bounds)")
print("=" * 50)

n, q, m = 5, 97, 10
np.random.seed(42)

s_true = np.random.choice([-1, 0, 1], size=n)
A = np.random.randint(0, q, size=(m, n))
e_true = np.round(np.random.normal(0, 1, size=m)).astype(int)
b = (A @ s_true + e_true) % q

print(f"True secret: {s_true}")

# ============================================================
# MILP with TIGHT bounds and a HARD time limit
# ============================================================
print("\n[1] Testing MILP with CBC (tight bounds)...")
ERROR_BOUND = 20  # errors are tiny (sigma=1); 20 is generous

prob = LpProblem("LWE", LpMinimize)
s_vars = [LpVariable(f"s_{j}", lowBound=-1, upBound=1, cat=LpInteger) for j in range(n)]
k_vars = [LpVariable(f"k_{i}", lowBound=-100, upBound=100, cat=LpInteger) for i in range(m)]
e_plus = [LpVariable(f"ep_{i}", lowBound=0, upBound=ERROR_BOUND, cat=LpInteger) for i in range(m)]
e_minus = [LpVariable(f"em_{i}", lowBound=0, upBound=ERROR_BOUND, cat=LpInteger) for i in range(m)]

for i in range(m):
    prob += lpSum(A[i, j] * s_vars[j] for j in range(n)) + e_plus[i] - e_minus[i] - q * k_vars[i] == b[i]
prob += lpSum(e_plus[i] + e_minus[i] for i in range(m))

start = time.time()
prob.solve(PULP_CBC_CMD(msg=True, timeLimit=30, threads=2))
elapsed = time.time() - start

if prob.status == 1:
    s_milp = np.array([value(s_vars[j]) for j in range(n)], dtype=int)
    print(f"MILP match: {np.array_equal(s_milp, s_true)}  (took {elapsed:.2f}s)")
else:
    print(f"MILP status: {prob.status}  (took {elapsed:.2f}s)")

# ============================================================
# LLL
# ============================================================
print("\n[2] Testing LLL with fpylll...")
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
print(f"LLL reduced {d}-dim basis in {time.time() - start:.3f}s")

print("\n" + "=" * 50)
print("Environment OK!")
print("=" * 50)