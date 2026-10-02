#!/usr/bin/env python3
"""
Extension 5: Suggest next cells under the pre-registered adaptive
expansion rule.

Reads existing results (benchmark_results.json or grid_results.json)
and outputs a JSON list of new configurations to run.

The script does NOT run the benchmark. It only plans.
"""

import json
import sys
from pathlib import Path
import numpy as np


ADAPTIVE_RULE = {
    'threshold_pp': 50,      # percent points
    'min_resolution': {
        'n': 1,
        'q_bits': 1,
        'sigma': 0.25,
    },
    'budget_cap': {
        'n': 4,
        'q': 3,
        'sigma': 3,
    },
}


def load_results(paths):
    records = []
    for p in paths:
        if Path(p).exists():
            with open(p) as f:
                records.extend(json.load(f))
    return records


def recovery_rate(records, q, sigma, secret, n, key='highs_success'):
    """Return recovery rate for a fixed (q, sigma, secret, n)."""
    sub = [r for r in records
           if r['q'] == q
           and abs(r.get('nominal_sigma', 1.0) - sigma) < 1e-6
           and r['secret_type'] == secret
           and r['n'] == n
           and r.get('planted_feasible', True)]
    if not sub:
        return None
    return 100.0 * sum(r[key] for r in sub) / len(sub)


def suggest_n_expansion(records, qs, sigmas, secrets, ns):
    """Suggest new n values based on the adaptive rule."""
    suggestions = set()
    for q in qs:
        for sigma in sigmas:
            for sec in secrets:
                ns_sorted = sorted(ns)
                for i in range(len(ns_sorted) - 1):
                    r0 = recovery_rate(records, q, sigma, sec, ns_sorted[i])
                    r1 = recovery_rate(records, q, sigma, sec, ns_sorted[i+1])
                    if r0 is None or r1 is None:
                        continue
                    if abs(r1 - r0) > ADAPTIVE_RULE['threshold_pp']:
                        mid = (ns_sorted[i] + ns_sorted[i+1]) // 2
                        if mid > ns_sorted[i] and mid < ns_sorted[i+1]:
                            suggestions.add(mid)
                        if len(suggestions) >= ADAPTIVE_RULE['budget_cap']['n']:
                            break
    return sorted(suggestions)


def suggest_q_expansion(records, qs, sigmas, secrets, ns):
    """Suggest new modulus values based on log-midpoint rule."""
    suggestions = set()
    qs_sorted = sorted(qs)
    for sigma in sigmas:
        for sec in secrets:
            for n in ns:
                for i in range(len(qs_sorted) - 1):
                    q0, q1 = qs_sorted[i], qs_sorted[i+1]
                    r0 = recovery_rate(records, q0, sigma, sec, n)
                    r1 = recovery_rate(records, q1, sigma, sec, n)
                    if r0 is None or r1 is None:
                        continue
                    if abs(r1 - r0) > ADAPTIVE_RULE['threshold_pp']:
                        mid = int(np.sqrt(q0 * q1))
                        if mid > q0 and mid < q1:
                            suggestions.add(mid)
                        if len(suggestions) >= ADAPTIVE_RULE['budget_cap']['q']:
                            break
    return sorted(suggestions)


def suggest_sigma_expansion(records, qs, sigmas, secrets, ns):
    """Suggest new sigma values based on arithmetic midpoint."""
    suggestions = set()
    sigmas_sorted = sorted(sigmas)
    for q in qs:
        for sec in secrets:
            for n in ns:
                for i in range(len(sigmas_sorted) - 1):
                    s0, s1 = sigmas_sorted[i], sigmas_sorted[i+1]
                    r0 = recovery_rate(records, q, s0, sec, n)
                    r1 = recovery_rate(records, q, s1, sec, n)
                    if r0 is None or r1 is None:
                        continue
                    if abs(r1 - r0) > ADAPTIVE_RULE['threshold_pp']:
                        mid = round((s0 + s1) / 2 * 4) / 4  # 0.25 resolution
                        if mid > s0 and mid < s1:
                            suggestions.add(mid)
                        if len(suggestions) >= ADAPTIVE_RULE['budget_cap']['sigma']:
                            break
    return sorted(suggestions)


def main():
    paths = sys.argv[1:] or [
        'results/benchmark_results.json',
        'results/grid_results.json',
    ]
    records = load_results(paths)

    if not records:
        print("[WARN] No results loaded.")
        sys.exit(0)

    qs = sorted(set(r['q'] for r in records))
    sigmas = sorted(set(r.get('nominal_sigma', 1.0) for r in records))
    secrets = sorted(set(r['secret_type'] for r in records))
    ns = sorted(set(r['n'] for r in records))

    print(f"Loaded {len(records)} records.")
    print(f"  q:      {qs}")
    print(f"  sigma:  {sigmas}")
    print(f"  secrets: {secrets}")
    print(f"  n:      {ns}")

    new_n = suggest_n_expansion(records, qs, sigmas, secrets, ns)
    new_q = suggest_q_expansion(records, qs, sigmas, secrets, ns)
    new_sigma = suggest_sigma_expansion(records, qs, sigmas, secrets, ns)

    print("\nAdaptive expansion suggestions:")
    print(f"  new n values:      {new_n}")
    print(f"  new q values:      {new_q}")
    print(f"  new sigma values:  {new_sigma}")

    plan = {
        'new_dimensions': new_n,
        'new_moduli': new_q,
        'new_sigmas': new_sigma,
        'note': ('Apply the pre-registered rule from ADAPTIVE_RULE.md. '
                 'All suggested cells must be run and reported.'),
    }
    with open('results/expansion_plan.json', 'w') as f:
        json.dump(plan, f, indent=2)
    print("\nWrote results/expansion_plan.json")


if __name__ == '__main__':
    main()