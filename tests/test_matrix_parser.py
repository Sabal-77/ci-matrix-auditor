import pytest
from pathlib import Path
from core.matrix_parser import MatrixParser

CONFIG = Path(__file__).parent.parent / "config" / "matrix.yml"


@pytest.fixture
def parser():
    return MatrixParser(CONFIG)


def test_total_job_count(parser):
    all_jobs = parser.expand_matrix()
    axes = parser.get_axes()
    expected = len(axes["os"]) * len(axes["python"]) * len(axes["backend"])
    assert len(all_jobs) == expected, (
        f"Expected {expected} total jobs (Cartesian product), got {len(all_jobs)}"
    )


def test_active_plus_excluded_equals_total(parser):
    all_jobs = parser.expand_matrix()
    active = parser.active_jobs()
    excluded = parser.excluded_jobs()
    assert len(active) + len(excluded) == len(all_jobs)


def test_macos_mpi_excluded(parser):
    all_jobs = parser.expand_matrix()
    macos_mpi = [
        j for j in all_jobs
        if j.os == "macos-13" and j.backend == "mpi"
    ]
    assert all(j.excluded for j in macos_mpi), (
        "All macos-13 + mpi jobs should be excluded"
    )
    assert all(j.exclusion_reason for j in macos_mpi), (
        "Excluded jobs must have a documented reason"
    )


def test_ubuntu_cpu_not_excluded(parser):
    active = parser.active_jobs()
    ubuntu_cpu = [
        j for j in active
        if j.os == "ubuntu-22.04" and j.backend == "cpu"
    ]
    assert len(ubuntu_cpu) == 3, (
        "ubuntu-22.04 + cpu should have 3 active jobs (one per Python version)"
    )


def test_no_duplicate_job_ids(parser):
    all_jobs = parser.expand_matrix()
    ids = [j.job_id for j in all_jobs]
    assert len(ids) == len(set(ids)), "All job IDs must be unique"


def test_exclusion_rate_is_nonzero(parser):
    summary = parser.exclusion_summary()
    assert summary["exclusion_rate"] > 0, "There should be at least some exclusions"
    assert summary["exclusion_rate"] < 1.0, "Not all jobs should be excluded"


def test_dependencies_populated_for_mpi(parser):
    active = parser.active_jobs()
    mpi_ubuntu = [
        j for j in active
        if j.backend == "mpi" and j.os == "ubuntu-22.04"
    ]
    assert mpi_ubuntu, "There should be active MPI jobs on ubuntu"
    for job in mpi_ubuntu:
        deps = job.dependencies
        assert "python" in deps or "system" in deps, (
            f"MPI job {job.job_id} should have declared dependencies"
        )


def test_schema_version_required(tmp_path):
    bad_config = tmp_path / "matrix.yml"
    bad_config.write_text("schema_version: 2\naxes: {}\nexclusions: []\ndependencies: {}")
    with pytest.raises(ValueError, match="schema_version"):
        MatrixParser(bad_config)