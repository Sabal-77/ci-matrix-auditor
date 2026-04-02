import pytest
from core.job_runner import JobResult, CommandResult
from analysis.failure_classifier import FailureClassifier, FailureCategory


def make_result(
    job_id: str = "test-job",
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    backend: str = "cpu",
    os: str = "ubuntu-22.04",
    python: str = "3.11",
) -> JobResult:
    return JobResult(
        job_id=job_id,
        backend=backend,
        os=os,
        python=python,
        exit_code=exit_code,
        duration_seconds=10.0,
        command_results=[
            CommandResult(
                command="pytest tests/",
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=10.0,
            )
        ],
        phase_failed="" if exit_code == 0 else "test",
    )


def test_success_classification():
    result = make_result(exit_code=0, stdout="5 passed")
    classifier = FailureClassifier([result])
    cat, _ = classifier.classify(result)
    assert cat == FailureCategory.SUCCESS


def test_mpi_infrastructure_failure():
    result = make_result(
        exit_code=1,
        stderr="mpi4py requires a working MPI installation",
        backend="mpi",
    )
    classifier = FailureClassifier([result])
    cat, explanation = classifier.classify(result)
    assert cat == FailureCategory.INFRASTRUCTURE
    assert "MPI" in explanation or "mpi" in explanation.lower()


def test_nvcc_not_found():
    result = make_result(
        exit_code=127,
        stderr="nvcc: command not found",
        backend="cuda-stub",
    )
    classifier = FailureClassifier([result])
    cat, _ = classifier.classify(result)
    assert cat == FailureCategory.INFRASTRUCTURE


def test_cython_build_error():
    result = make_result(
        exit_code=1,
        stderr="CompileError: gprMax/fields_outputs.pyx:142:5: undeclared name not builtin: np",
    )
    classifier = FailureClassifier([result])
    cat, _ = classifier.classify(result)
    assert cat == FailureCategory.BUILD_ERROR


def test_test_failure():
    result = make_result(
        exit_code=1,
        stdout="FAILED tests/test_geometry.py::test_box_volume - AssertionError",
    )
    classifier = FailureClassifier([result])
    cat, _ = classifier.classify(result)
    assert cat == FailureCategory.TEST_FAILURE


def test_timeout():
    result = make_result(
        exit_code=124,
        stderr="TIMEOUT after 600s",
    )
    classifier = FailureClassifier([result])
    cat, _ = classifier.classify(result)
    assert cat == FailureCategory.TIMEOUT


def test_infrastructure_failure_rate_all_infra():
    results = [
        make_result("job-1", 1, stderr="mpi4py requires a working MPI installation", backend="mpi"),
        make_result("job-2", 1, stderr="mpi4py requires a working MPI installation", backend="mpi"),
    ]
    classifier = FailureClassifier(results)
    rate = classifier.infrastructure_failure_rate()
    assert rate == 1.0


def test_infrastructure_failure_rate_mixed():
    results = [
        make_result("job-1", 1, stderr="mpi4py requires a working MPI installation", backend="mpi"),
        make_result("job-2", 1, stdout="FAILED tests/test_x.py::test_y - AssertionError", backend="cpu"),
    ]
    classifier = FailureClassifier(results)
    rate = classifier.infrastructure_failure_rate()
    assert rate == pytest.approx(0.5, abs=0.01)


def test_axis_correlation_identifies_backend():
    results = [
        make_result("ubuntu-mpi-py39", 1,
                    stderr="mpi4py requires a working MPI installation", backend="mpi"),
        make_result("ubuntu-mpi-py310", 1,
                    stderr="mpi4py requires a working MPI installation", backend="mpi"),
        make_result("ubuntu-cpu-py39", 0, stdout="2 passed", backend="cpu"),
    ]
    classifier = FailureClassifier(results)
    correlation = classifier.axis_correlation()
    assert "mpi" in correlation["backend"]
    assert correlation["backend"]["mpi"] == 2
    assert "cpu" not in correlation["backend"]


def test_actionable_failures_excludes_infrastructure():
    results = [
        make_result("job-infra", 1, stderr="mpi4py requires a working MPI installation", backend="mpi"),
        make_result("job-code", 1, stdout="FAILED tests/test_x.py::test_y", backend="cpu"),
    ]
    classifier = FailureClassifier(results)
    actionable = classifier.actionable_failures()
    ids = [f["job_id"] for f in actionable]
    assert "job-code" in ids
    assert "job-infra" not in ids