"""
Topological sort on the CI job dependency graph.

gprMax-specific rules encoded here:
- CPU jobs are independent of each other.
- MPI jobs on Linux depend on the matching CPU job completing first
  (they rely on the Cython build cache produced by the CPU job).
- cuda-stub jobs are fully independent (nvcc only, no gprMax build needed).
- A coverage aggregation job depends on ALL CPU jobs.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from core.matrix_parser import JobConfig


@dataclass
class DependencyEdge:
    source: str   # must complete before target
    target: str
    reason: str


class DependencyResolver:
    def __init__(self, jobs: list[JobConfig]):
        self.jobs = {j.job_id: j for j in jobs}
        self.edges: list[DependencyEdge] = []

    def add_edge(self, source_id: str, target_id: str, reason: str):
        if source_id not in self.jobs:
            raise ValueError(f"Source job not found: {source_id}")
        if target_id not in self.jobs:
            raise ValueError(f"Target job not found: {target_id}")
        self.edges.append(DependencyEdge(source_id, target_id, reason))

    def build_gprmax_dependencies(self):
        """
        Apply gprMax-specific dependency rules across all active jobs.
        Call this after constructing the resolver.
        """
        # Index CPU jobs on Linux by Python version
        linux_cpu_jobs = {
            j.python: j.job_id
            for j in self.jobs.values()
            if j.backend == "cpu" and j.os == "ubuntu-22.04"
        }

        for job in self.jobs.values():
            # MPI jobs reuse the Cython extensions compiled by the CPU job.
            # They must wait for the CPU job to complete and prime the cache.
            if (
                job.backend == "mpi"
                and job.os == "ubuntu-22.04"
                and job.python in linux_cpu_jobs
            ):
                self.add_edge(
                    source_id=linux_cpu_jobs[job.python],
                    target_id=job.job_id,
                    reason=(
                        f"MPI job reuses Cython build cache from CPU job "
                        f"(same Python {job.python})"
                    ),
                )

    def _build_adjacency(self) -> dict[str, list[str]]:
        adj: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            adj[edge.source].append(edge.target)
        return adj

    def _build_in_degree(self, adj: dict) -> dict[str, int]:
        in_degree: dict[str, int] = {job_id: 0 for job_id in self.jobs}
        for source, targets in adj.items():
            for target in targets:
                in_degree[target] += 1
        return in_degree

    def topological_waves(self) -> list[list[str]]:
        """
        Kahn's algorithm. Returns execution waves:
          wave[0] = jobs with no dependencies (run in parallel)
          wave[N] = jobs whose dependencies all completed in previous waves

        This models the real GitHub Actions execution plan.
        Raises ValueError if a cycle is detected.
        """
        adj = self._build_adjacency()
        in_degree = self._build_in_degree(adj)

        waves: list[list[str]] = []
        current_wave = sorted([jid for jid, deg in in_degree.items() if deg == 0])

        while current_wave:
            waves.append(current_wave)
            next_wave = []
            for job_id in current_wave:
                for dependent in adj.get(job_id, []):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_wave.append(dependent)
            current_wave = sorted(next_wave)

        stuck = [jid for jid, deg in in_degree.items() if deg > 0]
        if stuck:
            raise ValueError(
                f"Cycle detected in job dependency graph. "
                f"Stuck jobs: {stuck}"
            )

        return waves

    def detect_cycles(self) -> list[list[str]]:
        """
        DFS-based cycle detection.
        Returns list of cycles found (empty list = no cycles).
        """
        adj = self._build_adjacency()
        visited = set()
        rec_stack = set()
        cycles: list[list[str]] = []

        def dfs(node: str, path: list[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found a cycle — extract it
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
            path.pop()
            rec_stack.discard(node)

        for job_id in self.jobs:
            if job_id not in visited:
                dfs(job_id, [])

        return cycles

    def critical_path_length(self) -> int:
        """Number of sequential waves = minimum wall-clock depth."""
        return len(self.topological_waves())

    def jobs_in_wave(self, wave_index: int) -> list[str]:
        waves = self.topological_waves()
        if wave_index >= len(waves):
            return []
        return waves[wave_index]

    def dependency_summary(self) -> dict:
        waves = self.topological_waves()
        return {
            "total_jobs": len(self.jobs),
            "total_edges": len(self.edges),
            "wave_count": len(waves),
            "jobs_per_wave": [len(w) for w in waves],
            "max_parallelism": max(len(w) for w in waves) if waves else 0,
            "waves": waves,
        }

    def get_dependents(self, job_id: str) -> list[str]:
        """Returns all jobs that depend (directly) on job_id."""
        return [e.target for e in self.edges if e.source == job_id]

    def get_dependencies(self, job_id: str) -> list[str]:
        """Returns all jobs that job_id directly depends on."""
        return [e.source for e in self.edges if e.target == job_id]