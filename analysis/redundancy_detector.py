"""
Identifies jobs in the matrix that provide redundant failure signal.

Two jobs are "redundant" if they test the same logical configuration
with no additional discriminating power. For example:
- ubuntu-22.04 py3.9 cpu  AND  ubuntu-22.04 py3.10 cpu
  both test the CPU backend on Linux. If one fails, does the other
  provide new information? Only if the failure is Python-version-specific.

This analysis helps justify matrix exclusions quantitatively.
"""

from collections import defaultdict
from core.job_runner import JobResult


class RedundancyDetector:
    def __init__(self, results: list[JobResult]):
        self.results = results
        self._by_id = {r.job_id: r for r in results}

    def group_by_backend(self) -> dict[str, list[JobResult]]:
        groups: dict[str, list[JobResult]] = defaultdict(list)
        for r in self.results:
            groups[r.backend].append(r)
        return dict(groups)

    def group_by_os(self) -> dict[str, list[JobResult]]:
        groups: dict[str, list[JobResult]] = defaultdict(list)
        for r in self.results:
            groups[r.os].append(r)
        return dict(groups)

    def find_uniform_failures(self) -> dict[str, list[str]]:
        """
        Finds axes where ALL jobs with a given value failed.
        e.g., if all MPI jobs failed → backend=mpi is uniformly failing.
        This suggests a systemic issue, not job-specific.
        """
        axis_jobs: dict[str, dict[str, list[bool]]] = {
            "backend": defaultdict(list),
            "os": defaultdict(list),
            "python": defaultdict(list),
        }
        for r in self.results:
            for axis in axis_jobs:
                val = getattr(r, axis)
                axis_jobs[axis][val].append(r.success)

        uniform_failures = {}
        for axis, val_results in axis_jobs.items():
            for val, successes in val_results.items():
                if all(not s for s in successes) and len(successes) > 1:
                    key = f"{axis}={val}"
                    uniform_failures[key] = [
                        r.job_id for r in self.results
                        if getattr(r, axis) == val
                    ]
        return uniform_failures

    def find_python_version_sensitivity(self) -> dict[str, bool]:
        """
        For each OS+backend pair, checks whether failure/success
        varies across Python versions.
        If all Python versions behave the same, Python version
        is not a discriminating axis for that combination.
        """
        groups: dict[str, dict[str, bool]] = defaultdict(dict)
        for r in self.results:
            key = f"{r.os}+{r.backend}"
            groups[key][r.python] = r.success

        sensitivity = {}
        for combination, py_results in groups.items():
            outcomes = list(py_results.values())
            is_sensitive = len(set(outcomes)) > 1  # mixed outcomes
            sensitivity[combination] = is_sensitive

        return sensitivity

    def redundancy_report(self) -> dict:
        uniform = self.find_uniform_failures()
        sensitivity = self.find_python_version_sensitivity()

        insensitive = [k for k, v in sensitivity.items() if not v]
        sensitive = [k for k, v in sensitivity.items() if v]

        return {
            "uniform_failures": uniform,
            "python_sensitive_combinations": sensitive,
            "python_insensitive_combinations": insensitive,
            "interpretation": {
                "uniform_failures": (
                    "These axis values are uniformly failing — "
                    "likely a systemic infrastructure issue, not code."
                ),
                "insensitive": (
                    "These OS+backend combinations show the same result "
                    "across all Python versions — testing all 3 versions "
                    "provides no additional signal for these."
                ),
                "sensitive": (
                    "These combinations show different outcomes across "
                    "Python versions — all versions are worth testing here."
                ),
            }
        }