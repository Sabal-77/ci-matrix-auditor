"""
Simulates GitHub Actions cache behavior for gprMax's Cython build step.

The real CI cache key is:
  cython-{os}-py{version}-{hash(setup.py + *.pyx + *.pxd)}-cython{version}

This module lets you model a sequence of PRs and measure
the expected cache hit rate without running real CI.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CacheEntry:
    key: str
    value: str
    os: str
    python: str


@dataclass
class CacheDecision:
    pr_number: int
    job_id: str
    files_changed: list[str]
    is_cython_change: bool
    cache_key: str
    decision: str          # "HIT" | "MISS" | "PREFIX_HIT"
    reason: str
    estimated_build_time: str


# Files that, when changed, require a Cython rebuild
CYTHON_INVALIDATING_FILES = {
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
}
CYTHON_EXTENSIONS = {".pyx", ".pxd", ".pxi"}


def is_cython_relevant(files: list[str]) -> bool:
    for f in files:
        if f in CYTHON_INVALIDATING_FILES:
            return True
        if Path(f).suffix in CYTHON_EXTENSIONS:
            return True
    return False


class CacheSimulator:
    def __init__(self):
        # Simulates the remote cache store: key -> CacheEntry
        self._store: dict[str, CacheEntry] = {}

    def _hash_sources(self, source_snapshot: dict[str, str]) -> str:
        """
        Deterministic hash of source file contents.
        In real CI this would be hashFiles('**/*.pyx', '**/*.pxd', 'setup.py')
        """
        canonical = json.dumps(source_snapshot, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def compute_key(
        self,
        os: str,
        python: str,
        cython_version: str,
        source_snapshot: dict[str, str],
    ) -> str:
        source_hash = self._hash_sources(source_snapshot)
        return f"cython-{os}-py{python}-{source_hash}-cython{cython_version}"

    def compute_prefix(self, os: str, python: str) -> str:
        return f"cython-{os}-py{python}-"

    def get(self, key: str, os: str, python: str) -> tuple[str, str]:
        """
        Returns (decision, reason).
        Checks exact key first, then prefix (restore-keys fallback).
        """
        if key in self._store:
            return "HIT", "Exact key matched — Cython rebuild skipped"

        prefix = self.compute_prefix(os, python)
        for stored_key in self._store:
            if stored_key.startswith(prefix):
                return (
                    "PREFIX_HIT",
                    f"Prefix matched ({prefix}*) — stale cache restored, will overwrite after build"
                )
        return "MISS", "No match — full Cython rebuild required (~4 minutes)"

    def put(self, key: str, os: str, python: str):
        self._store[key] = CacheEntry(
            key=key, value="build_artifact", os=os, python=python
        )

    def simulate_pr_sequence(
        self,
        prs: list[dict],
        os: str = "ubuntu-22.04",
        python: str = "3.11",
        cython_version: str = "3.0.6",
    ) -> list[CacheDecision]:
        """
        Simulate cache behavior across a sequence of PRs.

        Each PR dict:
          {
            "pr": 1,
            "files": ["gprMax/geometry.py", "tests/test_foo.py"],
            "source_snapshot": {"gprMax/fields.pyx": "content_v1", ...}
              # OR omit and it will be derived from files
          }

        source_snapshot represents the content of all Cython-relevant
        files at the time of the PR. If a .pyx file changes between PRs,
        the snapshot value for that file should differ.
        """
        # Start with a base snapshot (simulate the repo's initial state)
        current_snapshot: dict[str, str] = {
            "setup.py": "setup_v1",
            "gprMax/fields_outputs.pyx": "fields_v1",
            "gprMax/geometry_primitives.pyx": "geometry_v1",
            "gprMax/cython_include.pxd": "include_v1",
        }

        decisions: list[CacheDecision] = []

        for pr in prs:
            pr_num = pr.get("pr", len(decisions) + 1)
            files = pr.get("files", [])
            job_id = f"{os}-py{python}-cpu"
            cython_change = is_cython_relevant(files)

            # Update snapshot if this PR changed Cython files
            if "source_snapshot" in pr:
                current_snapshot.update(pr["source_snapshot"])
            elif cython_change:
                # Simulate a change: update affected files in snapshot
                for f in files:
                    if f in current_snapshot or Path(f).suffix in CYTHON_EXTENSIONS:
                        current_snapshot[f] = f"content_pr{pr_num}"

            key = self.compute_key(os, python, cython_version, current_snapshot)
            decision, reason = self.get(key, os, python)

            build_time = "~30s" if decision in ("HIT", "PREFIX_HIT") else "~4min"

            decisions.append(CacheDecision(
                pr_number=pr_num,
                job_id=job_id,
                files_changed=files,
                is_cython_change=cython_change,
                cache_key=key[:40] + "...",
                decision=decision,
                reason=reason,
                estimated_build_time=build_time,
            ))

            # After a MISS or PREFIX_HIT, we always write the new key
            if decision in ("MISS", "PREFIX_HIT"):
                self.put(key, os, python)

        return decisions

    def hit_rate(self, decisions: list[CacheDecision]) -> float:
        hits = sum(1 for d in decisions if d.decision in ("HIT", "PREFIX_HIT"))
        return hits / len(decisions) if decisions else 0.0

    def time_saved_minutes(
        self,
        decisions: list[CacheDecision],
        cold_build_minutes: float = 4.5,
        cached_build_minutes: float = 0.5,
    ) -> float:
        savings = 0.0
        for d in decisions:
            if d.decision in ("HIT", "PREFIX_HIT"):
                savings += cold_build_minutes - cached_build_minutes
        return round(savings, 2)

    def clear(self):
        self._store.clear()