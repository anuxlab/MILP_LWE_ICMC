# MILP_LWE_ICMC
Just another simulation for LWE

# MILP vs. Lattice Reduction for LWE — Benchmark

Automated benchmark comparing MILP-based attacks (CBC solver) with lattice reduction (LLL/BKZ) for small LWE instances, running on GitHub Actions.

## Setup

### 1. Create GitHub repository

```bash
cd ~/Desktop/IIITG/Thesis/MILP_LWE_ICMC
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/MILP_LWE_ICMC.git
git push -u origin main
```

### 2. Trigger the workflow

**Option A — Auto on push:** The workflow runs automatically after every push to `main`.

**Option B — Manual:** Go to `Actions` tab → `LWE Benchmark` → `Run workflow` → set parameters → `Run`.

### 3. Download results

Once the workflow finishes (~10–30 min), go to the run page → `Artifacts` → download `benchmark-results.zip`.

It contains:
- `benchmark_results.json` — raw data
- `success_rate.pdf` / `.png` — Figure 1
- `time_to_solution.pdf` / `.png` — Figure 2

## Parameters

Edit `.github/workflows/benchmark.yml` or pass via the "Run workflow" dialog:

| Param | Default | Meaning |
|-------|---------|---------|
| `dimensions` | `5 10 15 20 25` | LWE dimensions to test |
| `instances` | `10` | Instances per (n, secret) config |
| `time_limit` | `120` | Max seconds per attack |
| `bkz_beta` | `15` | BKZ block size |

## Local Run (Optional)

```bash
python3.11 -m venv lwe-bench-env
source lwe-bench-env/bin/activate
pip install -r requirements.txt
python quick_test.py
python lwe_benchmark.py --dimensions 5 10 --instances 2
python generate_figures.py
```

## Results Interpretation

- **MILP wins** for sparse secrets → structural constraints prune search space
- **BKZ wins** for dense secrets at larger `n` → geometric structure dominates
- **Crossover dimension** ≈ 20 for ternary, ≈ 15 for sparse

## Cost

Free for public repos (unlimited minutes). Private repos: 2000 min/month free tier.

Each run consumes ~15–30 min, so you can do ~60+ runs/month for free even on private.