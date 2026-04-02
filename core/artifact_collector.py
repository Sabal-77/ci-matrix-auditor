"""
Collects and persists job results to disk for later analysis.
Writes JSON artifacts per job, plus a combined manifest.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

from core.job_runner import JobResult


class ArtifactCollector:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest: dict = {
            "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "jobs": [],
        }

    def collect(self, result: JobResult):
        job_dir = self.output_dir / result.job_id
        job_dir.mkdir(exist_ok=True)

        # Write stdout
        (job_dir / "stdout.txt").write_text(result.stdout or "")
        (job_dir / "stderr.txt").write_text(result.stderr or "")

        # Write structured result (exclude large stdout/stderr from JSON)
        summary = {
            "job_id": result.job_id,
            "backend": result.backend,
            "os": result.os,
            "python": result.python,
            "exit_code": result.exit_code,
            "success": result.success,
            "duration_seconds": round(result.duration_seconds, 3),
            "phase_failed": result.phase_failed,
            "commands": [
                {
                    "command": c.command,
                    "exit_code": c.exit_code,
                    "duration_seconds": round(c.duration_seconds, 3),
                }
                for c in result.command_results
            ],
        }
        (job_dir / "result.json").write_text(json.dumps(summary, indent=2))

        self.manifest["jobs"].append({
            "job_id": result.job_id,
            "success": result.success,
            "exit_code": result.exit_code,
            "duration_seconds": round(result.duration_seconds, 3),
        })

    def save_manifest(self):
        passed = sum(1 for j in self.manifest["jobs"] if j["success"])
        failed = len(self.manifest["jobs"]) - passed
        self.manifest["summary"] = {
            "total": len(self.manifest["jobs"]),
            "passed": passed,
            "failed": failed,
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest, indent=2))
        return manifest_path

    def load_results(self) -> list[dict]:
        results = []
        for job_dir in sorted(self.output_dir.iterdir()):
            result_file = job_dir / "result.json"
            if result_file.exists():
                results.append(json.loads(result_file.read_text()))
        return results