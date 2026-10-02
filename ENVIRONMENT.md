# Execution Environment Manifest

## Runtime
- Python: 3.11.16
- OS: Ubuntu 24.04.5 LTS
- GitHub Actions runner: ubuntu-latest (image 20260920.314.1)

## Solver versions
- HiGHS: highspy 1.15.1 (bundled HiGHS 1.7.2)
- CBC: bundled with PuLP 3.3.2
- fpylll: 0.6.4 (links to fplll 5.4.5)

## Reproducibility notes
- All random generation uses `np.random.seed` with documented seeds
- Instance generators are in `lwe_benchmark.py` and versioned by git commit
- Each checkpoint records the git commit hash of the code that produced it