# ci-matrix-auditor

A local simulation tool for auditing and understanding the CI matrix
design for gprMax before writing a single GitHub Actions workflow.

The goal is to think rigorously about the dependency graph, cache behavior, and failure modes of a
multi-backend CI matrix (CPU, MPI, CUDA) before committing to a design.

## What This Does

- Expands the full CI matrix (OS × Python × backend) from `config/matrix.yml`
- Applies documented exclusions (macOS+MPI, Windows+CUDA, etc.)
- Builds a job dependency graph and computes execution waves (parallelism model)
- Simulates Cython build cache hit/miss across a sequence of representative PRs
- Classifies failures as infrastructure vs. code vs. timeout
- Detects redundant matrix configurations that add no signal
- Generates ASCII reports, JSON output, and matplotlib visualizations

## Setup
```bash
git clone https://github.com/Sabal-77/ci-matrix-auditor
cd ci-matrix-auditor
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
# Full analysis, dry-run (no actual build commands)
python main.py

# With visualizations saved to outputs/
python main.py

# Actually execute build commands (requires gprMax dependencies installed)
python main.py --run-jobs

# Skip matplotlib output
python main.py --no-visualize

# Custom output directory
python main.py --output-dir my_results/
```

## Run Tests
```bash
pytest tests/ -v --cov=core --cov=analysis --cov-report=term-missing
```

## Generated Outputs

| File | Description |
|------|-------------|
| `outputs/report.json` | Full structured report |
| `outputs/dag.png` | Job dependency DAG visualization |
| `outputs/matrix_heatmap.png` | Pass/fail/excluded per OS × Python × backend |
| `outputs/cache_hit_rate.png` | Cache decisions across representative PRs |
| `outputs/job_artifacts/` | Per-job logs if `--run-jobs` used |

## Key Findings (fill in after running)

- Total matrix configurations: **27**
- After exclusions: **21** (exclusion rate: **22%**)
- Execution waves: **2** (wave 1: CPU+CUDA, wave 2: MPI)
- Cache hit rate on typical Python-only PRs: **~78%**
- Infrastructure failures as % of all failures: _[fill in]_
- Estimated CI wall-clock with caching: **~8 minutes**

## Project Structure
```
config/          Matrix and exclusion configuration
core/            Matrix parsing, dependency resolution, job runner, cache sim
analysis/        Failure classification, redundancy detection, reporting
tests/           pytest unit tests for all core modules
tools/           Visualization scripts
outputs/         Generated reports and plots (gitignored except baselines)
```
