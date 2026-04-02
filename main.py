"""
gprMax CI Matrix Lab — main entry point.

Usage:
  python main.py                        # Full analysis, dry-run mode
  python main.py --run-jobs             # Actually execute build commands
  python main.py --no-visualize         # Skip matplotlib output
  python main.py --output-dir results/  # Custom output directory
"""

import argparse
from pathlib import Path

from core.matrix_parser import MatrixParser
from core.dependency_resolver import DependencyResolver
from core.job_runner import JobRunner
from core.cache_simulator import CacheSimulator
from core.artifact_collector import ArtifactCollector
from analysis.report_generator import ReportGenerator


REPRESENTATIVE_PRS = [
    {"pr": 1,  "files": ["gprMax/input_cmds.py", "tests/test_input.py"]},
    {"pr": 2,  "files": ["gprMax/fields_outputs.pyx"]},
    {"pr": 3,  "files": ["README.md", "docs/source/index.rst"]},
    {"pr": 4,  "files": ["gprMax/geometry_primitives.pyx", "setup.py"]},
    {"pr": 5,  "files": ["gprMax/receivers.py"]},
    {"pr": 6,  "files": ["tests/test_receivers.py"]},
    {"pr": 7,  "files": ["gprMax/cython_include.pxd"]},
    {"pr": 8,  "files": ["gprMax/utilities.py", "gprMax/input_cmds.py"]},
    {"pr": 9,  "files": [".github/workflows/ci.yml"]},
    {"pr": 10, "files": ["gprMax/geometry_primitives.pyx"]},
]


def main():
    parser_arg = argparse.ArgumentParser(description="gprMax CI Matrix Lab")
    parser_arg.add_argument("--run-jobs", action="store_true",
                            help="Actually execute build commands (default: dry-run)")
    parser_arg.add_argument("--no-visualize", action="store_true",
                            help="Skip generating matplotlib visualizations")
    parser_arg.add_argument("--output-dir", default="outputs",
                            help="Directory for reports and artifacts")
    parser_arg.add_argument("--config", default="config/matrix.yml",
                            help="Path to matrix.yml")
    args = parser_arg.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config)

    # --- 1. Parse matrix ---
    print("\n[1/5] Parsing matrix configuration...")
    matrix_parser = MatrixParser(config_path)
    active_jobs = matrix_parser.active_jobs()
    print(f"     Active jobs: {len(active_jobs)}")
    print(f"     Excluded jobs: {len(matrix_parser.excluded_jobs())}")

    # --- 2. Build dependency graph ---
    print("\n[2/5] Building job dependency graph...")
    resolver = DependencyResolver(active_jobs)
    resolver.build_gprmax_dependencies()
    dep_summary = resolver.dependency_summary()
    print(f"     Waves: {dep_summary['wave_count']}")
    print(f"     Max parallelism: {dep_summary['max_parallelism']} jobs")
    print(f"     Dependencies: {dep_summary['total_edges']}")

    # --- 3. Simulate cache behavior ---
    print("\n[3/5] Simulating cache behavior across representative PRs...")
    sim = CacheSimulator()
    cache_decisions = sim.simulate_pr_sequence(REPRESENTATIVE_PRS)
    hit_rate = sim.hit_rate(cache_decisions)
    saved = sim.time_saved_minutes(cache_decisions)
    print(f"     Cache hit rate: {hit_rate*100:.0f}%")
    print(f"     Estimated time saved: {saved:.1f} minutes")

    # --- 4. Run jobs (or dry-run) ---
    job_results = []
    if args.run_jobs:
        print(f"\n[4/5] Executing {len(active_jobs)} jobs (this will take a while)...")
        collector = ArtifactCollector(output_dir / "job_artifacts")
        runner = JobRunner(dry_run=False, timeout=600)
        job_results = runner.run_jobs_sequential(active_jobs)
        for result in job_results:
            collector.collect(result)
        manifest_path = collector.save_manifest()
        print(f"     Artifacts saved to: {manifest_path.parent}")
    else:
        print("\n[4/5] Skipping job execution (dry-run mode).")
        print("     Use --run-jobs to execute real build commands.")

    # --- 5. Generate report ---
    print("\n[5/5] Generating report...")
    report = ReportGenerator(
        parser=matrix_parser,
        resolver=resolver,
        job_results=job_results,
        cache_decisions=cache_decisions,
    )
    report.print_full_report()
    report.save_json(output_dir / "report.json")

    # --- Visualizations ---
    if not args.no_visualize:
        print("\n[+] Generating visualizations...")
        try:
            from tools.visualize_dag import plot_dag, plot_matrix_heatmap, plot_cache_hit_rate
            plot_dag(matrix_parser, resolver, output_dir / "dag.png")
            plot_matrix_heatmap(matrix_parser, job_results or None, output_dir / "matrix_heatmap.png")
            plot_cache_hit_rate(cache_decisions, output_dir / "cache_hit_rate.png")
        except ImportError as e:
            print(f"     Visualization skipped: {e}")
        except Exception as e:
            print(f"     Visualization error: {e}")

    print(f"\n✓ Done. All outputs in: {output_dir}/\n")


if __name__ == "__main__":
    main()