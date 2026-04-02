"""
Reads matrix.yml, expands all axis combinations via Cartesian product,
applies exclusion rules, and returns typed JobConfig objects.
"""

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class JobConfig:
    job_id: str
    os: str
    python: str
    backend: str
    excluded: bool = False
    exclusion_reason: str = ""
    dependencies: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)

    def short_id(self) -> str:
        os_short = self.os.replace("-", "").replace(".", "")
        py_short = self.python.replace(".", "")
        return f"{os_short}-py{py_short}-{self.backend}"

    def __repr__(self):
        status = "EXCLUDED" if self.excluded else "ACTIVE"
        return f"JobConfig({self.short_id()} [{status}])"

    def __hash__(self):
        return hash(self.job_id)

    def __eq__(self, other):
        return isinstance(other, JobConfig) and self.job_id == other.job_id


class MatrixParser:
    def __init__(self, matrix_path: Path):
        self.matrix_path = matrix_path
        with open(matrix_path) as f:
            self.config = yaml.safe_load(f)
        self._validate_schema()

    def _validate_schema(self):
        version = self.config.get("schema_version")
        if version != 1:
            raise ValueError(
                f"Unsupported schema_version: {version}. "
                f"This tool supports schema_version=1 only."
            )
        required = {"axes", "exclusions", "dependencies"}
        missing = required - set(self.config.keys())
        if missing:
            raise ValueError(f"matrix.yml missing required keys: {missing}")

    def expand_matrix(self) -> list[JobConfig]:
        """
        Full Cartesian product of all axes, then apply exclusions.
        Returns all jobs including excluded ones (marked with excluded=True).
        """
        axes = self.config["axes"]
        timing = self.config.get("timing_estimates", {})
        all_combinations = list(itertools.product(
            axes["os"], axes["python"], axes["backend"]
        ))

        jobs = []
        for os_, python, backend in all_combinations:
            job_id = f"{os_}-py{python}-{backend}"
            deps = self._get_dependencies(os_, backend)
            job_timing = {
                "cold_build": timing.get("cold_build_minutes", {}).get(backend, 5.0),
                "cached_build": timing.get("cached_build_minutes", {}).get(backend, 0.5),
                "tests": timing.get("test_minutes", {}).get(backend, 3.0),
            }
            job = JobConfig(
                job_id=job_id,
                os=os_,
                python=python,
                backend=backend,
                dependencies=deps,
                timing=job_timing,
            )
            exclusion = self._find_exclusion(os_, backend)
            if exclusion:
                job.excluded = True
                job.exclusion_reason = exclusion.get("reason", "").strip()

            jobs.append(job)
        return jobs

    def _get_dependencies(self, os_: str, backend: str) -> dict:
        deps = self.config.get("dependencies", {})
        backend_deps = deps.get(backend, {})
        os_specific = backend_deps.get(os_, backend_deps.get("all", {}))
        return os_specific if os_specific else {}

    def _find_exclusion(self, os_: str, backend: str) -> dict | None:
        for excl in self.config.get("exclusions", []):
            if excl["os"] == os_ and excl["backend"] == backend:
                return excl
        return None

    def active_jobs(self) -> list[JobConfig]:
        return [j for j in self.expand_matrix() if not j.excluded]

    def excluded_jobs(self) -> list[JobConfig]:
        return [j for j in self.expand_matrix() if j.excluded]

    def exclusion_summary(self) -> dict:
        all_jobs = self.expand_matrix()
        excluded = [j for j in all_jobs if j.excluded]
        active = [j for j in all_jobs if not j.excluded]
        by_backend = {}
        for j in excluded:
            by_backend.setdefault(j.backend, []).append(j.job_id)
        by_os = {}
        for j in excluded:
            by_os.setdefault(j.os, []).append(j.job_id)
        return {
            "total": len(all_jobs),
            "active": len(active),
            "excluded": len(excluded),
            "exclusion_rate": round(len(excluded) / len(all_jobs), 4),
            "by_backend": by_backend,
            "by_os": by_os,
            "reasons": {j.job_id: j.exclusion_reason for j in excluded},
        }

    def get_axes(self) -> dict:
        return self.config["axes"]