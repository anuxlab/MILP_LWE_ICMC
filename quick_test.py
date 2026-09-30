import numpy as np
from pulp import LpProblem, LpVariable, LpInteger, lpSum, LpMinimize, PULP_CBC_CMD, value
from fpylll import IntegerMatrix, LLL

print("=" * 50)
print("Quick Sanity Test")
print("=" * 50)

n, q, m = 5, 97, 10
np.random.seed(42)

s_true = np.random.choice([-1, 0, 1], size=n)
A = np.random.randint(0, q, size=(m, n))
e_true = np.round(np.random.normal(0, 1, size=m)).astype(int)
b = (A @ s_true + e_true) % q

print(f"True secret: {s_true}")

print("\n[1] Testing MILP with CBC...")
prob = LpProblem("LWE", LpMinimize)
s_vars = [LpVariable(f"s_{j}", lowBound=-q, upBound=q, cat=LpInteger) for j in range(n)]
k_vars = [LpVariable(f"k_{i}", lowBound=-q, upBound=q, cat=LpInteger) for i in range(m)]
e_plus = [LpVariable(f"ep_{i}", lowBound=0, upBound=q, cat=LpInteger) for i in range(m)]
e_minus = [LpVariable(f"em_{i}", lowBound=0, upBound=q, cat=LpInteger) for i in range(m)]
for i in range(m):
    prob += lpSum(A[i, j] * s_vars[j] for j in range(n)) + e_plus[i] - e_minus[i] - q * k_vars[i] == b[i]
prob += lpSum(e_plus[i] + e_minus[i] for i in range(m))
prob.solve(PULP_CBC_CMD(msg=False))
s_milp = np.array([value(s_vars[j]) for j in range(n)], dtype=int)
print(f"MILP match: {np.array_equal(s_milp, s_true)}")

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
LLL.reduction(B)
print(f"LLL reduced {d}-dim basis successfully")

print("\n" + "=" * 50)
print("Environment OK!")
print("=" * 50)