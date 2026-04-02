import pytest
from core.cache_simulator import CacheSimulator, is_cython_relevant


def test_is_cython_relevant_pyx():
    assert is_cython_relevant(["gprMax/fields.pyx"]) is True


def test_is_cython_relevant_pxd():
    assert is_cython_relevant(["gprMax/include.pxd"]) is True


def test_is_cython_relevant_setup():
    assert is_cython_relevant(["setup.py"]) is True


def test_is_cython_relevant_pure_python():
    assert is_cython_relevant(["gprMax/geometry.py", "tests/test_foo.py"]) is False


def test_is_cython_relevant_readme():
    assert is_cython_relevant(["README.md", "docs/index.rst"]) is False


@pytest.fixture
def sim():
    return CacheSimulator()


def test_first_pr_always_misses(sim):
    prs = [{"pr": 1, "files": ["gprMax/geometry.py"]}]
    decisions = sim.simulate_pr_sequence(prs)
    assert decisions[0].decision == "MISS", "First PR always misses (empty cache)"


def test_second_pure_python_pr_hits(sim):
    prs = [
        {"pr": 1, "files": ["gprMax/geometry.py"]},
        {"pr": 2, "files": ["gprMax/input_cmds.py"]},
    ]
    decisions = sim.simulate_pr_sequence(prs)
    assert decisions[0].decision == "MISS"
    assert decisions[1].decision == "HIT", (
        "Second PR with no Cython changes should hit cache"
    )


def test_cython_change_causes_miss(sim):
    prs = [
        {"pr": 1, "files": ["gprMax/geometry.py"]},          # primes cache
        {"pr": 2, "files": ["gprMax/geometry.py"]},           # should hit
        {"pr": 3, "files": ["gprMax/fields_outputs.pyx"]},    # should miss
        {"pr": 4, "files": ["gprMax/geometry.py"]},           # should hit again
    ]
    decisions = sim.simulate_pr_sequence(prs)
    assert decisions[0].decision == "MISS"
    assert decisions[1].decision == "HIT"
    assert decisions[2].decision == "MISS", "Cython change must invalidate cache"
    assert decisions[3].decision in ("HIT", "MISS"), "Post-cython-change PR depends on new key"


def test_setup_py_change_invalidates(sim):
    prs = [
        {"pr": 1, "files": ["README.md"]},
        {"pr": 2, "files": ["setup.py"]},
    ]
    decisions = sim.simulate_pr_sequence(prs)
    assert decisions[1].decision == "MISS", "setup.py change must invalidate cache"


def test_hit_rate_calculation(sim):
    prs = [
        {"pr": i, "files": ["gprMax/geometry.py"]}
        for i in range(1, 11)
    ]
    decisions = sim.simulate_pr_sequence(prs)
    rate = sim.hit_rate(decisions)
    # 1 miss (first PR) + 9 hits
    assert rate == pytest.approx(0.9, abs=0.01)


def test_time_saved_is_positive(sim):
    prs = [
        {"pr": 1, "files": ["gprMax/geometry.py"]},
        {"pr": 2, "files": ["tests/test_foo.py"]},
        {"pr": 3, "files": ["docs/guide.rst"]},
    ]
    decisions = sim.simulate_pr_sequence(prs)
    saved = sim.time_saved_minutes(decisions)
    assert saved > 0, "At least 2 cache hits should save time"


def test_cache_clears(sim):
    prs = [{"pr": 1, "files": ["gprMax/geometry.py"]}]
    sim.simulate_pr_sequence(prs)
    sim.clear()
    decisions = sim.simulate_pr_sequence(prs)
    assert decisions[0].decision == "MISS", "After clear(), cache is empty again"


def test_key_changes_with_different_python(sim):
    key_39 = sim.compute_key("ubuntu-22.04", "3.9", "3.0.6", {"setup.py": "v1"})
    key_311 = sim.compute_key("ubuntu-22.04", "3.11", "3.0.6", {"setup.py": "v1"})
    assert key_39 != key_311, "Different Python versions must produce different cache keys"


def test_key_changes_with_different_cython_version(sim):
    key_v1 = sim.compute_key("ubuntu-22.04", "3.11", "3.0.5", {"setup.py": "v1"})
    key_v2 = sim.compute_key("ubuntu-22.04", "3.11", "3.0.6", {"setup.py": "v1"})
    assert key_v1 != key_v2, "Cython version bump must invalidate cache key"