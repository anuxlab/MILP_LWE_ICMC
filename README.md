# MILP vs. Lattice Reduction for LWE — Dual-Solver Benchmark

Open-source benchmark comparing **MILP (HiGHS + CBC)** vs. **lattice reduction (LLL/BKZ)** for the Search-LWE problem. Runs on GitHub Actions.

## Outputs

Each run produces (in `results/`):
- `benchmark_results.json` — raw data (all runs)
- `fig_success_rate.pdf` / `.png` — success rate vs. dimension
- `fig_time_to_solution.pdf` / `.png` — timing (log scale)
- `fig_solver_comparison.pdf` / `.png` — HiGHS vs. CBC
- `fig_success_by_secret.pdf` / `.png` — per secret type
- `tables.tex` — LaTeX tables ready to paste into the paper

## Run

### Manual trigger (recommended)
Actions tab → **LWE Benchmark** → Run workflow.

### CLI parameters
| Param | Default | Meaning |
|-------|---------|---------|
| `dimensions` | `5 10 15 20` | Dimensions to test |
| `instances` | `10` | Instances per (n, secret) config |
| `time_limit` | `60` | Max seconds per attack |
| `bkz_beta` | `15` | BKZ block size |

## Local reproduction

```bash
python3.11 -m venv lwe-bench-env
source lwe-bench-env/bin/activate
pip install -r requirements.txt
python quick_test.py
python lwe_benchmark.py --dimensions 5 10 15 --instances 3 --time-limit 30
python generate_figures.py
python generate_latex_tables.py
```

## Solvers

| Solver | License | Notes |
|--------|---------|-------|
| **HiGHS** | MIT / open-source | Fast MILP solver, best-in-class open-source |
| **CBC** | EPL | Classic reference solver, comes bundled with PuLP |
| **LLL** | LGPL | fpylll |
| **BKZ** | LGPL | fpylll |

All solvers are 100% open-source and academic-friendly.

## Expected runtime (GitHub Actions)

~25–35 minutes for `5 10 15 20` × 3 secrets × 10 instances.