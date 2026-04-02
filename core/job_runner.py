"""
Executes real build and test shell commands for each job configuration.
Records timing, stdout/stderr, and exit codes.
In dry-run mode prints commands without executing them.
"""

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from core.matrix_parser import JobConfig


@dataclass
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float


@dataclass
class JobResult:
    job_id: str
    backend: str
    os: str
    python: str
    exit_code: int
    duration_seconds: float
    command_results: list[CommandResult] = field(default_factory=list)
    phase_failed: str = ""   # "install" | "build" | "test" | ""

    @property
    def success(self) -> bool:
        return self.exit_code == 0

    @property
    def stdout(self) -> str:
        return "\n".join(c.stdout for c in self.command_results)

    @property
    def stderr(self) -> str:
        return "\n".join(c.stderr for c in self.command_results)

    def __repr__(self):
        status = "SUCCESS" if self.success else f"FAILED({self.exit_code})"
        return f"JobResult({self.job_id} [{status}] {self.duration_seconds:.1f}s)"


class CommandBuilder:
    """
    Translates a JobConfig into the sequence of shell commands
    that a real CI runner would execute for gprMax.
    """

    def __init__(self, python_bin: str = "python"):
        self.python_bin = python_bin

    def install_commands(self, job: JobConfig) -> list[str]:
        cmds = []
        # Upgrade pip first
        cmds.append(f"{self.python_bin} -m pip install --upgrade pip")

        # System dependencies note (can't install these from Python,
        # but we document them for the report)
        sys_deps = job.dependencies.get("system", [])
        if sys_deps:
            cmds.append(
                f"# SYSTEM: sudo apt-get install -y {' '.join(sys_deps)}"
                if "ubuntu" in job.os
                else f"# SYSTEM: brew install {' '.join(sys_deps)}"
            )

        # Python dependencies
        py_deps = job.dependencies.get("python", [])
        if py_deps:
            cmds.append(f"pip install {' '.join(py_deps)}")

        # MPI-specific
        if job.backend == "mpi":
            cmds.append("pip install mpi4py")

        # Core gprMax deps
        cmds.append("pip install cython numpy h5py matplotlib")
        return cmds

    def build_commands(self, job: JobConfig) -> list[str]:
        if job.backend == "cuda-stub":
            # Only compile CUDA, don't build full gprMax
            return [
                "nvcc --version",
                f"{self.python_bin} setup.py build_ext --inplace --cuda-only 2>&1 | tee cuda_build.log || true",
            ]
        return [
            f"{self.python_bin} -m pip install -e . --no-build-isolation",
        ]

    def test_commands(self, job: JobConfig) -> list[str]:
        if job.backend == "cpu":
            return [
                f"{self.python_bin} -m pytest tests/ -v --timeout=120 -m 'not mpi and not gpu'",
            ]
        if job.backend == "mpi":
            return [
                (
                    f"mpirun --oversubscribe -n 4 "
                    f"{self.python_bin} -m pytest tests/ -v -m mpi --with-mpi --timeout=180"
                ),
            ]
        if job.backend == "cuda-stub":
            return [
                f"{self.python_bin} -m pytest tests/ -v -m 'not gpu and not mpi' --timeout=60",
            ]
        return []

    def all_commands(self, job: JobConfig) -> dict[str, list[str]]:
        return {
            "install": self.install_commands(job),
            "build": self.build_commands(job),
            "test": self.test_commands(job),
        }


class JobRunner:
    def __init__(self, dry_run: bool = False, timeout: int = 600):
        self.dry_run = dry_run
        self.timeout = timeout
        self.builder = CommandBuilder()

    def run_job(self, job: JobConfig) -> JobResult:
        phases = self.builder.all_commands(job)
        all_results: list[CommandResult] = []
        total_start = time.monotonic()
        overall_exit = 0
        failed_phase = ""

        for phase_name, commands in phases.items():
            for cmd in commands:
                result = self._run_command(cmd)
                all_results.append(result)
                if result.exit_code != 0 and not cmd.startswith("#"):
                    overall_exit = result.exit_code
                    failed_phase = phase_name
                    break
            if overall_exit != 0:
                break

        total_duration = time.monotonic() - total_start
        return JobResult(
            job_id=job.job_id,
            backend=job.backend,
            os=job.os,
            python=job.python,
            exit_code=overall_exit,
            duration_seconds=total_duration,
            command_results=all_results,
            phase_failed=failed_phase,
        )

    def _run_command(self, cmd: str) -> CommandResult:
        if self.dry_run or cmd.startswith("#"):
            print(f"  [DRY RUN] {cmd}")
            return CommandResult(
                command=cmd, exit_code=0,
                stdout=f"[dry run]: {cmd}",
                stderr="", duration_seconds=0.0
            )

        print(f"  $ {cmd}")
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True,
                text=True, timeout=self.timeout
            )
            duration = time.monotonic() - start
            if proc.returncode != 0:
                print(f"  [FAILED exit={proc.returncode}]")
            return CommandResult(
                command=cmd,
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return CommandResult(
                command=cmd, exit_code=124,
                stdout="", stderr=f"TIMEOUT after {self.timeout}s",
                duration_seconds=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start
            return CommandResult(
                command=cmd, exit_code=1,
                stdout="", stderr=str(e),
                duration_seconds=duration,
            )

    def run_jobs_sequential(self, jobs: list[JobConfig]) -> list[JobResult]:
        results = []
        for i, job in enumerate(jobs):
            print(f"\n[{i+1}/{len(jobs)}] Running: {job.job_id}")
            result = self.run_job(job)
            results.append(result)
            status = "✓ SUCCESS" if result.success else f"✗ FAILED (phase: {result.phase_failed})"
            print(f"  → {status} ({result.duration_seconds:.1f}s)")
        return results