"""
Generates human-readable reports from matrix analysis results.
Outputs to terminal (ASCII tables) and optionally to JSON.
"""

import json
from pathlib import Path

from tabulate import tabulate
from colorama import Fore, Style, init as colorama_init

from core.matrix_parser import JobConfig, MatrixParser
from core.dependency_resolver import DependencyResolver
from core.job_runner import JobResult
from core.cache_simulator import CacheDecision
from analysis.failure_classifier import FailureClassifier, FailureCategory
from analysis.redundancy_detector import RedundancyDetector

colorama_init(autoreset=True)


def _color_status(success: bool) -> str:
    if success:
        return Fore.GREEN + "SUCCESS" + Style.RESET_ALL
    return Fore.RED + "FAILED" + Style.RESET_ALL


def _color_decision(decision: str) -> str:
    if decision == "HIT":
        return Fore.GREEN + "HIT" + Style.RESET_ALL
    if decision == "PREFIX_HIT":
        return Fore.YELLOW + "PREFIX_HIT" + Style.RESET_ALL
    return Fore.RED + "MISS" + Style.RESET_ALL


class ReportGenerator:
    def __init__(
        self,
        parser: MatrixParser,
        resolver: DependencyResolver,
        job_results: list[JobResult] | None = None,
        cache_decisions: list[CacheDecision] | None = None,
    ):
        self.parser = parser
        self.resolver = resolver
        self.job_results = job_results or []
        self.cache_decisions = cache_decisions or []

    def print_full_report(self):
        self._print_matrix_summary()
        self._print_exclusion_table()
        self._print_wave_plan()
        self._print_cache_simulation()
        if self.job_results:
            self._print_job_results()
            self._print_failure_analysis()
            self._print_redundancy_analysis()
        self._print_timing_estimate()

    def _print_matrix_summary(self):
        summary = self.parser.exclusion_summary()
        axes = self.parser.get_axes()
        print("\n" + "=" * 65)
        print("  gprMax CI Matrix Audit Report")
        print("=" * 65)
        print(f"\n  OS variants    : {len(axes['os'])} ({', '.join(axes['os'])})")
        print(f"  Python versions: {len(axes['python'])} ({', '.join(axes['python'])})")
        print(f"  Backends       : {len(axes['backend'])} ({', '.join(axes['backend'])})")
        print(f"\n  Total configurations : {summary['total']}")
        print(f"  Active (will run)    : {summary['active']}")
        print(f"  Excluded (documented): {summary['excluded']}")
        print(f"  Exclusion rate       : {summary['exclusion_rate']*100:.1f}%")

    def _print_exclusion_table(self):
        excluded = self.parser.excluded_jobs()
        if not excluded:
            return
        print("\n" + "-" * 65)
        print("  Excluded Configurations (with reasons)")
        print("-" * 65)
        rows = []
        for job in excluded:
            reason_short = job.exclusion_reason[:55].strip()
            if len(job.exclusion_reason) > 55:
                reason_short += "..."
            rows.append([job.job_id, reason_short])
        print(tabulate(rows, headers=["Job ID", "Reason"], tablefmt="simple"))

    def _print_wave_plan(self):
        dep_summary = self.resolver.dependency_summary()
        waves = dep_summary["waves"]

        print("\n" + "-" * 65)
        print("  Execution Wave Plan (GitHub Actions parallel batches)")
        print("-" * 65)
        for i, wave in enumerate(waves):
            print(f"\n  Wave {i+1} ({len(wave)} jobs, run in parallel):")
            for job_id in wave:
                deps = self.resolver.get_dependencies(job_id)
                dep_note = f"  ← depends on: {deps[0]}" if deps else ""
                print(f"    {job_id}{dep_note}")

        print(f"\n  Critical path depth : {dep_summary['wave_count']} wave(s)")
        print(f"  Max parallelism     : {dep_summary['max_parallelism']} jobs")
        print(f"  Total dependencies  : {dep_summary['total_edges']}")

    def _print_cache_simulation(self):
        if not self.cache_decisions:
            return
        print("\n" + "-" * 65)
        print("  Cache Simulation (Cython build cache behavior)")
        print("-" * 65)
        rows = []
        for d in self.cache_decisions:
            files_short = ", ".join(d.files_changed[:2])
            if len(d.files_changed) > 2:
                files_short += f" +{len(d.files_changed)-2} more"
            cython_flag = "YES" if d.is_cython_change else "no"
            rows.append([
                f"PR #{d.pr_number}",
                files_short,
                cython_flag,
                _color_decision(d.decision),
                d.estimated_build_time,
            ])
        print(tabulate(
            rows,
            headers=["PR", "Files Changed", "Cython?", "Cache", "Est. Build Time"],
            tablefmt="simple"
        ))

        from core.cache_simulator import CacheSimulator
        sim = CacheSimulator.__new__(CacheSimulator)
        hit_rate = sum(
            1 for d in self.cache_decisions if d.decision in ("HIT", "PREFIX_HIT")
        ) / len(self.cache_decisions)
        saved = sum(
            1 for d in self.cache_decisions if d.decision in ("HIT", "PREFIX_HIT")
        ) * (4.5 - 0.5)

        print(f"\n  Cache hit rate        : {hit_rate*100:.0f}%")
        print(f"  Estimated time saved  : {saved:.1f} minutes across {len(self.cache_decisions)} PRs")

    def _print_job_results(self):
        print("\n" + "-" * 65)
        print("  Job Execution Results")
        print("-" * 65)
        rows = []
        for r in self.job_results:
            rows.append([
                r.job_id,
                _color_status(r.success),
                f"{r.duration_seconds:.1f}s",
                str(r.exit_code),
                r.phase_failed or "—",
            ])
        print(tabulate(
            rows,
            headers=["Job ID", "Status", "Duration", "Exit", "Failed Phase"],
            tablefmt="simple"
        ))
        passed = sum(1 for r in self.job_results if r.success)
        print(f"\n  {passed}/{len(self.job_results)} jobs passed")

    def _print_failure_analysis(self):
        failures = [r for r in self.job_results if not r.success]
        if not failures:
            print("\n  All jobs passed — no failure analysis needed.")
            return

        classifier = FailureClassifier(self.job_results)
        counts = classifier.category_counts()
        correlation = classifier.axis_correlation()
        infra_rate = classifier.infrastructure_failure_rate()
        actionable = classifier.actionable_failures()

        print("\n" + "-" * 65)
        print("  Failure Analysis")
        print("-" * 65)

        print(f"\n  Total failures : {len(failures)}")
        print(f"  Infrastructure : {counts.get('infrastructure', 0)} "
              f"({infra_rate*100:.0f}% of failures)")
        print(f"  Build errors   : {counts.get('build_error', 0)}")
        print(f"  Test failures  : {counts.get('test_failure', 0)}")
        print(f"  Timeouts       : {counts.get('timeout', 0)}")

        print("\n  Failure correlation by axis:")
        for axis, vals in correlation.items():
            if vals:
                for val, count in sorted(vals.items(), key=lambda x: -x[1]):
                    print(f"    {axis}={val}: {count} failure(s)")

        if actionable:
            print(f"\n  Actionable failures (require code fixes): {len(actionable)}")
            for f in actionable:
                print(f"    {f['job_id']}: {f['explanation']}")

        note = (
            "IMPORTANT: Infrastructure failures do not indicate code bugs.\n"
            "    Fix the runner environment setup (apt-get, pip install order)\n"
            "    before investigating further."
        )
        if infra_rate > 0.5:
            print(f"\n  {Fore.YELLOW}NOTE: {note}{Style.RESET_ALL}")

    def _print_redundancy_analysis(self):
        detector = RedundancyDetector(self.job_results)
        report = detector.redundancy_report()

        print("\n" + "-" * 65)
        print("  Redundancy Analysis")
        print("-" * 65)

        insensitive = report["python_insensitive_combinations"]
        sensitive = report["python_sensitive_combinations"]

        if insensitive:
            print(f"\n  OS+backend combinations where Python version adds no signal:")
            for combo in insensitive:
                print(f"    {combo}  → could reduce to 1 Python version")

        if sensitive:
            print(f"\n  OS+backend combinations where Python version matters:")
            for combo in sensitive:
                print(f"    {combo}  → keep all Python versions")

        uniform = report["uniform_failures"]
        if uniform:
            print(f"\n  Uniformly failing axes (systemic, not code-specific):")
            for axis_val, jobs in uniform.items():
                print(f"    {axis_val}: all {len(jobs)} jobs failed")

    def _print_timing_estimate(self):
        active_jobs = self.parser.active_jobs()
        dep_summary = self.resolver.dependency_summary()
        waves = dep_summary["waves"]

        active_by_id = {j.job_id: j for j in active_jobs}

        print("\n" + "-" * 65)
        print("  Wall-Clock Time Estimate")
        print("-" * 65)

        total_cold = sum(
            j.timing.get("cold_build", 5.0) + j.timing.get("tests", 3.0)
            for j in active_jobs
        )
        # With caching: wave 1 is cold, wave 2+ uses cache
        wave1_cold = sum(
            active_by_id[jid].timing.get("cold_build", 5.0)
            + active_by_id[jid].timing.get("tests", 3.0)
            for jid in waves[0]
            if jid in active_by_id
        )
        wave2_cached = sum(
            active_by_id[jid].timing.get("cached_build", 0.5)
            + active_by_id[jid].timing.get("tests", 3.0)
            for jid in (waves[1] if len(waves) > 1 else [])
            if jid in active_by_id
        )
        with_caching = wave1_cold + wave2_cached

        print(f"\n  Sequential (no parallelism, no cache): {total_cold:.0f} min")
        print(f"  Parallel waves, cold builds           : {wave1_cold:.0f} min")
        print(f"  Parallel waves, with cache            : {with_caching:.0f} min")
        print(f"  Cache savings                         : {total_cold - with_caching:.0f} min")
        print(f"  Estimated real CI wall-clock time     : ~{with_caching:.0f} minutes")

    def save_json(self, output_path: Path):
        summary = self.parser.exclusion_summary()
        dep_summary = self.resolver.dependency_summary()
        report = {
            "matrix_summary": summary,
            "dependency_summary": dep_summary,
            "cache_decisions": [
                {
                    "pr": d.pr_number,
                    "files": d.files_changed,
                    "cython_change": d.is_cython_change,
                    "decision": d.decision,
                    "reason": d.reason,
                    "estimated_build_time": d.estimated_build_time,
                }
                for d in self.cache_decisions
            ],
            "job_results": [
                {
                    "job_id": r.job_id,
                    "success": r.success,
                    "exit_code": r.exit_code,
                    "duration_seconds": round(r.duration_seconds, 2),
                    "phase_failed": r.phase_failed,
                }
                for r in self.job_results
            ],
        }
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\n  Report saved to: {output_path}")