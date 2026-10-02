# Adaptive Expansion Rule (Pre-Registered)

Committed: [date]
Applies to: Extensions 3, 4, and 5 of the MILP-LWE benchmark

## Rule

For each axis (dimension n, modulus q, absolute noise σ), examine
the recovery rate R as a function of the axis value, averaged over
all other axes and all instances.

1. Compute |ΔR| between adjacent tested values.
2. Add midpoints to any interval with |ΔR| > 50 percentage points.
3. Iterate until no such interval remains, or the interval is below
   the minimum resolution.
4. Report all tested cells.

Minimum resolutions:
- n: 1
- q: 1 bit
- σ: 0.25

Budget cap per line: 4 (n), 3 (q), 3 (σ).

## Rationale

Two-value designs cannot establish frontier location if both values
produce identical outcomes. Bracketing by midpoints is the smallest
change that reveals where transitions occur.

## Anti-gaming clause

The rule is fixed. Intermediate points are chosen by the arithmetic
or geometric midpoint, not by any criterion related to outcome. All
tested cells appear in the results regardless of whether they
contribute information.

## Revision policy

If the rule must be revised (e.g., minimum resolution too coarse),
the revision is committed BEFORE the first expansion run, and both
the original and the revision are documented in the appendix.