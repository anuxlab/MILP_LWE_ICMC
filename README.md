# MILP vs. Lattice Reduction for LWE — Dual-Solver Benchmark

Open-source benchmark comparing **MILP (HiGHS + CBC)** vs. **lattice reduction (LLL/BKZ)** for the Search-LWE problem. Runs on GitHub Actions with **resumable checkpointing**.

## Key Features

- **Resumable**: Each instance's results are written to disk immediately. Interrupted runs pick up where they stopped.
- **Dual-solver**: Compares HiGHS (Python API) vs. CBC — no commercial licenses needed.
- **Robust**: Accepts any feasible MILP solution, not just proven-optimal (fixes a known `pulp.HiGHS` status bug).
- **Per-type time limits**: Uniform secrets (known-hard) get capped budgets.
- **Persistent**: Uses GitHub Actions cache + artifact backup for long-term storage.

## Structure

```
├── .github/workflows/benchmark.yml   # CI/CD pipeline
├── lwe_benchmark.py                  # Resumable benchmark
├── quick_test.py                     # Sanity check
├── generate_figures.py               # 4 figures
├── generate_latex_tables.py          # 4 LaTeX tables
├── requirements.txt
└── results/                          # Created at runtime
    ├── checkpoints/                  # Per-instance JSON (resumable)
    ├── benchmark_results.json        # Aggregated
    ├── run.log                       # Full stdout
    ├── fig_*.pdf/.png                # 4 figures
    └── tables.tex                    # Ready for paper
```

## Usage

### Automatic (recommended)
Push to `main` or trigger manually from Actions tab.

### CLI parameters
| Param | Default | Meaning |
|-------|---------|---------|
| `dimensions` | `5 10 15` | Dimensions to test |
| `instances` | `10` | Instances per (n, secret) config |
| `time_limit` | `60` | Base time limit per attack (uniform capped at 10s) |
| `bkz_beta` | `15` | BKZ block size |
| `force_rerun` | `false` | Ignore checkpoints and rerun all |

### Local run
```bash
python3.11 -m venv lwe-bench-env
source lwe-bench-env/bin/activate
pip install -r requirements.txt
python quick_test.py
python lwe_benchmark.py --dimensions 5 10 --instances 3 --time-limit 30
python generate_figures.py
python generate_latex_tables.py
```

## Checkpoint System

- **Schema versioning**: Cache key uses a manual `schema2` token. Bump to `schema3` only when you change the checkpoint JSON format.
- **Atomic writes**: Each checkpoint writes to a temp file then renames — no partial files.
- **Resume on rerun**: Existing checkpoints are loaded and skipped.
- **Two-level backup**: Cache (fast, auto-restored) + artifact (90 days, survives eviction).

## Invalidation Rules

| Change | Invalidates? |
|--------|-------------|
| Add new dimension or instances | ✅ No |
| Change `time_limit` | ✅ No |
| Fix a code comment | ✅ No |
| Change BKZ β | ⚠️ BKZ results reused; use `--force-rerun` |
| Change MILP formulation | ❌ Yes — bump schema to `schema3` |
| Change checkpoint JSON fields | ❌ Yes — bump schema |

## Cost

Free for public repos. Private: 2000 min/month free tier.
Typical run: 15–25 min per iteration (only new instances execute).

## Solvers

All solvers are 100% open-source and academic-friendly:

| Solver | License |
|--------|---------|
| HiGHS | MIT |
| CBC | EPL |
| LLL / BKZ | LGPL (via fpylll) |