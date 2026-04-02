"""
Classifies job failures into categories:
- INFRASTRUCTURE: MPI not installed, nvcc not found, missing system libraries
- BUILD_ERROR: Cython compilation failed, import error after build
- TEST_FAILURE: Tests ran and failed (assertion errors, wrong output)
- TIMEOUT: Job exceeded time limit
- SUCCESS: Exit code 0
"""

from enum import Enum

from core.job_runner import JobResult


class FailureCategory(Enum):
    SUCCESS = "success"
    INFRASTRUCTURE = "infrastructure"
    BUILD_ERROR = "build_error"
    TEST_FAILURE = "test_failure"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


# Ordered from most specific to least specific.
# First match wins.
INFRASTRUCTURE_PATTERNS: list[tuple[str, str]] = [
    ("mpi4py requires a working MPI installation", "mpi4py cannot find system MPI runtime"),
    ("nvcc: command not found", "CUDA toolkit (nvcc) not installed on runner"),
    ("nvcc: No such file or directory", "CUDA toolkit (nvcc) not installed on runner"),
    ("fatal error: hdf5.h: No such file", "libhdf5-dev not installed"),
    ("cannot find -lopenmpi", "OpenMPI libraries not found at link time"),
    ("No module named 'mpi4py'", "mpi4py not installed — pip install step failed"),
    ("No module named 'gprMax'", "gprMax installation failed — build step issue"),
    ("Library not loaded: /usr/local/lib/libmpi", "MPI dynamic library path broken (macOS)"),
    ("ERROR 124", "Job timed out"),
    ("TIMEOUT after", "Job timed out in runner"),
    ("error: Microsoft Visual C++ 14.0 or greater is required", "MSVC not available on Windows runner"),
]

BUILD_ERROR_PATTERNS: list[tuple[str, str]] = [
    ("CompileError", "Cython compilation error"),
    ("error: command", "Compiler invocation failed"),
    ("gcc: error:", "GCC reported a build error"),
    ("clang: error:", "Clang reported a build error"),
    ("ModuleNotFoundError: No module named", "Python import failed after build"),
    ("ImportError", "Extension failed to import — likely linking error"),
    ("Cython.Compiler.Errors", "Cython source error"),
]

TEST_FAILURE_PATTERNS: list[tuple[str, str]] = [
    ("FAILED", "pytest test failure"),
    ("AssertionError", "Test assertion failed"),
    ("ERROR", "Test errored"),
    ("short test summary info", "pytest summary with failures"),
]


class FailureClassifier:
    def __init__(self, results: list[JobResult]):
        self.results = results

    def classify(self, result: JobResult) -> tuple[FailureCategory, str]:
        """Returns (category, human-readable explanation)."""
        if result.exit_code == 0:
            return FailureCategory.SUCCESS, "All commands completed successfully"

        if result.exit_code == 124 or "TIMEOUT" in result.stderr.upper():
            return FailureCategory.TIMEOUT, "Job exceeded time limit"

        combined = result.stdout + result.stderr

        for pattern, explanation in INFRASTRUCTURE_PATTERNS:
            if pattern in combined:
                return FailureCategory.INFRASTRUCTURE, explanation

        for pattern, explanation in BUILD_ERROR_PATTERNS:
            if pattern in combined:
                return FailureCategory.BUILD_ERROR, explanation

        for pattern, explanation in TEST_FAILURE_PATTERNS:
            if pattern in combined:
                return FailureCategory.TEST_FAILURE, explanation

        return FailureCategory.UNKNOWN, f"Non-zero exit ({result.exit_code}) — no pattern matched"

    def classify_all(self) -> dict[str, tuple[FailureCategory, str]]:
        return {r.job_id: self.classify(r) for r in self.results}

    def infrastructure_failure_rate(self) -> float:
        classified = self.classify_all()
        failures = [v for v in classified.values() if v[0] != FailureCategory.SUCCESS]
        if not failures:
            return 0.0
        infra = [v for v in failures if v[0] == FailureCategory.INFRASTRUCTURE]
        return len(infra) / len(failures)

    def category_counts(self) -> dict[str, int]:
        classified = self.classify_all()
        counts: dict[str, int] = {}
        for _, (cat, _) in classified.items():
            counts[cat.value] = counts.get(cat.value, 0) + 1
        return counts

    def axis_correlation(self) -> dict[str, dict[str, int]]:
        """
        Finds whether failures cluster on a specific axis value.
        e.g., "all MPI jobs failed" → backend=mpi has count=N
        This distinguishes systemic failures from random noise.
        """
        classified = self.classify_all()
        result_map = {r.job_id: r for r in self.results}
        failed_jobs = [
            result_map[jid]
            for jid, (cat, _) in classified.items()
            if cat != FailureCategory.SUCCESS
        ]

        correlation: dict[str, dict[str, int]] = {
            "backend": {}, "os": {}, "python": {}
        }
        for job in failed_jobs:
            for axis in ("backend", "os", "python"):
                val = getattr(job, axis)
                correlation[axis][val] = correlation[axis].get(val, 0) + 1

        return correlation

    def actionable_failures(self) -> list[dict]:
        """
        Returns only failures that are NOT infrastructure issues.
        These are the failures that represent real problems in the code.
        """
        classified = self.classify_all()
        result_map = {r.job_id: r for r in self.results}
        actionable = []
        for jid, (cat, explanation) in classified.items():
            if cat in (FailureCategory.BUILD_ERROR, FailureCategory.TEST_FAILURE):
                job = result_map[jid]
                actionable.append({
                    "job_id": jid,
                    "category": cat.value,
                    "explanation": explanation,
                    "phase_failed": job.phase_failed,
                    "duration": job.duration_seconds,
                })
        return actionable