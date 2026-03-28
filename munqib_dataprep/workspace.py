"""Workspace state for agentic dataprep profiles, recipes, jobs, and artifacts."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import deep_merge, load_structured_file


DEFAULT_PROFILE: Dict[str, Any] = {
    "default_recipe": None,
    "hf_token_env": "HF_TOKEN",
    "output_dir": "outputs",
    "source_defaults": {
        "split": "train",
        "streaming": True,
    },
    "build_defaults": {},
    "curate_defaults": {},
}


def now_iso() -> str:
    """Return an ISO-8601 timestamp in UTC."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sanitize_name(value: str, fallback: str = "item") -> str:
    """Convert user-provided names into filesystem-safe identifiers."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip()).strip("._-").lower()
    return cleaned or fallback


class WorkspaceState:
    """Repo-local persisted state used by the agentic dataprep tools."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.project_root = self.root.parent
        self.profile_path = self.root / "profile.yaml"
        self.recipes_dir = self.root / "recipes"
        self.jobs_dir = self.root / "jobs"
        self.outputs_dir = self.root / "outputs"
        self.artifacts_path = self.root / "artifacts" / "manifest.json"

    def ensure_dirs(self) -> None:
        """Create the workspace directories if they do not exist."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.recipes_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_path.parent.mkdir(parents=True, exist_ok=True)

    def _atomic_write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
            Path(tmp_name).replace(path)
        finally:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass

    def _write_structured(self, path: Path, data: Dict[str, Any]) -> None:
        # JSON is valid YAML and avoids introducing a second serializer dependency.
        self._atomic_write(path, json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    def load_profile(self) -> Dict[str, Any]:
        """Load the repo-local profile merged with defaults."""
        self.ensure_dirs()
        if not self.profile_path.exists():
            return json.loads(json.dumps(DEFAULT_PROFILE))
        return deep_merge(DEFAULT_PROFILE, load_structured_file(str(self.profile_path)))

    def save_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a repo-local profile."""
        self.ensure_dirs()
        merged = deep_merge(DEFAULT_PROFILE, profile)
        self._write_structured(self.profile_path, merged)
        return merged

    def update_profile(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Merge and persist a profile patch."""
        return self.save_profile(deep_merge(self.load_profile(), patch))

    def recipe_path(self, name: str) -> Path:
        """Return the on-disk path for a named recipe."""
        return self.recipes_dir / f"{sanitize_name(name, fallback='recipe')}.yaml"

    def save_recipe(self, name: str, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a recipe file."""
        self.ensure_dirs()
        payload = dict(recipe)
        payload.setdefault("name", sanitize_name(name, fallback="recipe"))
        self._write_structured(self.recipe_path(name), payload)
        return payload

    def load_recipe(self, name: str) -> Dict[str, Any]:
        """Load a saved recipe by name."""
        path = self.recipe_path(name)
        if not path.exists():
            raise FileNotFoundError(f"recipe not found: {name}")
        payload = load_structured_file(str(path))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid recipe contents: {name}")
        payload.setdefault("name", sanitize_name(name, fallback="recipe"))
        return payload

    def list_recipes(self) -> List[Dict[str, Any]]:
        """List saved recipes with a short summary."""
        self.ensure_dirs()
        results: List[Dict[str, Any]] = []
        for path in sorted(self.recipes_dir.glob("*.yaml")):
            try:
                payload = load_structured_file(str(path))
            except Exception as exc:  # pragma: no cover - defensive
                results.append({"name": path.stem, "path": str(path), "error": str(exc)})
                continue
            results.append(
                {
                    "name": payload.get("name", path.stem),
                    "path": str(path),
                    "description": payload.get("description"),
                    "actions": sorted(key for key in payload.keys() if key in {"build", "curate"}),
                }
            )
        return results

    def job_path(self, job_id: str) -> Path:
        """Return the JSON path for a job record."""
        return self.jobs_dir / f"{sanitize_name(job_id, fallback='job')}.json"

    def job_log_path(self, job_id: str) -> Path:
        """Return the log path for a job."""
        return self.jobs_dir / f"{sanitize_name(job_id, fallback='job')}.log"

    def save_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a job record."""
        self.ensure_dirs()
        job = dict(job)
        job["updated_at"] = now_iso()
        self._write_structured(self.job_path(job["job_id"]), job)
        return job

    def load_job(self, job_id: str) -> Dict[str, Any]:
        """Load a saved job record."""
        path = self.job_path(job_id)
        if not path.exists():
            raise FileNotFoundError(f"job not found: {job_id}")
        payload = load_structured_file(str(path))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid job record: {job_id}")
        return payload

    def update_job(self, job_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Merge and persist a job patch."""
        job = self.load_job(job_id)
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(job.get(key), dict):
                job[key] = deep_merge(job[key], value)
            else:
                job[key] = value
        return self.save_job(job)

    def append_job_log(self, job_id: str, line: str) -> None:
        """Append a line to the job log."""
        self.ensure_dirs()
        path = self.job_log_path(job_id)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{now_iso()} {line.rstrip()}\n")

    def read_job_log(self, job_id: str, tail: int = 40) -> Dict[str, Any]:
        """Read the tail of a job log."""
        path = self.job_log_path(job_id)
        if not path.exists():
            return {"job_id": job_id, "lines": [], "path": str(path)}
        lines = path.read_text(encoding="utf-8").splitlines()
        return {"job_id": job_id, "lines": lines[-tail:], "path": str(path)}

    def load_artifact_manifest(self) -> Dict[str, Any]:
        """Load the artifact manifest."""
        self.ensure_dirs()
        if not self.artifacts_path.exists():
            return {"artifacts": {}}
        payload = load_structured_file(str(self.artifacts_path))
        if not isinstance(payload, dict):
            return {"artifacts": {}}
        payload.setdefault("artifacts", {})
        return payload

    def save_artifact_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the artifact manifest."""
        self.ensure_dirs()
        payload = dict(manifest)
        payload.setdefault("artifacts", {})
        self._write_structured(self.artifacts_path, payload)
        return payload

    def register_artifact(
        self,
        ref: str,
        path: str | Path,
        *,
        kind: str,
        job_id: str,
        ready: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create or update an artifact manifest entry."""
        manifest = self.load_artifact_manifest()
        artifacts = manifest.setdefault("artifacts", {})
        entry = artifacts.get(ref, {})
        entry.update(
            {
                "ref": ref,
                "path": str(path),
                "kind": kind,
                "job_id": job_id,
                "ready": ready,
                "updated_at": now_iso(),
                "metadata": metadata or entry.get("metadata", {}),
            }
        )
        artifacts[ref] = entry
        self.save_artifact_manifest(manifest)
        return entry

    def resolve_artifact(self, ref: str) -> Dict[str, Any]:
        """Resolve an artifact manifest entry."""
        manifest = self.load_artifact_manifest()
        entry = manifest.get("artifacts", {}).get(ref)
        if entry is None:
            raise FileNotFoundError(f"artifact not found: {ref}")
        payload = dict(entry)
        payload["exists"] = Path(payload["path"]).exists()
        return payload

