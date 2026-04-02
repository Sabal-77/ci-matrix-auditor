import pytest
from core.matrix_parser import MatrixParser, JobConfig
from core.dependency_resolver import DependencyResolver
from pathlib import Path

CONFIG = Path(__file__).parent.parent / "config" / "matrix.yml"


@pytest.fixture
def three_job_graph():
    jobs = [
        JobConfig("job-a", "ubuntu-22.04", "3.11", "cpu"),
        JobConfig("job-b", "ubuntu-22.04", "3.11", "mpi"),
        JobConfig("job-c", "ubuntu-22.04", "3.11", "cuda-stub"),
    ]
    return jobs


@pytest.fixture
def resolver_from_config():
    parser = MatrixParser(CONFIG)
    active = parser.active_jobs()
    resolver = DependencyResolver(active)
    resolver.build_gprmax_dependencies()
    return resolver


def test_topological_sort_respects_order(three_job_graph):
    resolver = DependencyResolver(three_job_graph)
    resolver.add_edge("job-a", "job-b", "test dependency")
    waves = resolver.topological_waves()
    wave_0_ids = waves[0]
    wave_1_ids = waves[1]
    assert "job-a" in wave_0_ids
    assert "job-b" in wave_1_ids


def test_independent_jobs_in_same_wave(three_job_graph):
    resolver = DependencyResolver(three_job_graph)
    waves = resolver.topological_waves()
    assert len(waves) == 1, "No dependencies → all jobs in one wave"
    assert len(waves[0]) == 3


def test_cycle_detection_raises(three_job_graph):
    resolver = DependencyResolver(three_job_graph)
    resolver.add_edge("job-a", "job-b", "a→b")
    resolver.add_edge("job-b", "job-a", "b→a creates cycle")
    with pytest.raises(ValueError, match="Cycle detected"):
        resolver.topological_waves()


def test_detect_cycles_finds_cycle(three_job_graph):
    resolver = DependencyResolver(three_job_graph)
    resolver.add_edge("job-a", "job-b", "a→b")
    resolver.add_edge("job-b", "job-a", "b→a")
    cycles = resolver.detect_cycles()
    assert len(cycles) > 0, "Should detect at least one cycle"


def test_detect_cycles_empty_on_dag(three_job_graph):
    resolver = DependencyResolver(three_job_graph)
    resolver.add_edge("job-a", "job-b", "a→b")
    cycles = resolver.detect_cycles()
    assert cycles == [], "DAG should have no cycles"


def test_mpi_jobs_depend_on_cpu(resolver_from_config):
    edges = resolver_from_config.edges
    mpi_targets = {e.target for e in edges}
    for target_id in mpi_targets:
        assert "mpi" in target_id, f"Unexpected non-MPI job as dependency target: {target_id}"
    sources = {e.source for e in edges}
    for source_id in sources:
        assert "cpu" in source_id, f"Unexpected non-CPU job as dependency source: {source_id}"


def test_wave_count_is_at_least_two(resolver_from_config):
    waves = resolver_from_config.topological_waves()
    assert len(waves) >= 2, (
        "gprMax matrix should have at least 2 waves: "
        "wave 1 (CPU+cuda-stub), wave 2 (MPI depends on CPU)"
    )


def test_every_job_in_exactly_one_wave(resolver_from_config):
    waves = resolver_from_config.topological_waves()
    all_in_waves = [jid for wave in waves for jid in wave]
    assert len(all_in_waves) == len(set(all_in_waves)), (
        "Each job must appear in exactly one wave"
    )


def test_get_dependents(three_job_graph):
    resolver = DependencyResolver(three_job_graph)
    resolver.add_edge("job-a", "job-b", "reason")
    resolver.add_edge("job-a", "job-c", "reason")
    dependents = resolver.get_dependents("job-a")
    assert set(dependents) == {"job-b", "job-c"}


def test_get_dependencies(three_job_graph):
    resolver = DependencyResolver(three_job_graph)
    resolver.add_edge("job-a", "job-b", "reason")
    deps = resolver.get_dependencies("job-b")
    assert deps == ["job-a"]