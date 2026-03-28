"""Shared service layer for the CLI, worker, and native agent tools."""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .builders import BuildError, build_to_path, inspect_source_spec
from .config import DEFAULT_CONFIG, ConfigError, deep_merge, load_pipeline_config
from .curate import CurateError, curate_to_outputs
from .workspace import WorkspaceState, now_iso, sanitize_name


DEFAULT_WORKSPACE_DIRNAME = ".munqib"
BACKGROUND_PROCS: list[subprocess.Popen[Any]] = []


class ServiceError(RuntimeError):
    """Raised when the shared dataprep service cannot complete a request."""


def get_workspace(workspace_root: Optional[str] = None) -> WorkspaceState:
    """Return a workspace rooted in the repo-local `.munqib` directory."""
    root = Path(workspace_root) if workspace_root else Path.cwd() / DEFAULT_WORKSPACE_DIRNAME
    return WorkspaceState(root)


def _ensure_mapping(name: str, value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ServiceError(f"{name} must be an object")
    return value


def _resolve_recipe_payload(
    workspace: WorkspaceState,
    action: str,
    recipe_name: Optional[str],
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    profile = workspace.load_profile()
    selected_recipe = recipe_name or profile.get("default_recipe")
    resolved: Dict[str, Any]

    if action == "build":
        resolved = deep_merge(profile.get("source_defaults", {}), profile.get("build_defaults", {}))
        if profile.get("hf_token_env") and "hf_token_env" not in resolved:
            resolved["hf_token_env"] = profile["hf_token_env"]
    else:
        resolved = deep_merge(DEFAULT_CONFIG, profile.get("curate_defaults", {}))

    if selected_recipe:
        recipe = workspace.load_recipe(selected_recipe)
        payload = recipe.get(action) if action in recipe else recipe
        if not isinstance(payload, dict):
            raise ServiceError(f"recipe '{selected_recipe}' does not contain a valid {action} payload")
        resolved = deep_merge(resolved, payload)

    if overrides:
        resolved = deep_merge(resolved, overrides)

    return resolved, selected_recipe


def _derive_artifact_name(action: str, resolved_spec: Dict[str, Any], recipe_name: Optional[str], explicit: Optional[str]) -> str:
    if explicit:
        return sanitize_name(explicit, fallback=action)
    if recipe_name:
        return sanitize_name(recipe_name, fallback=action)
    if action == "build":
        return sanitize_name(str(resolved_spec.get("source") or "build"), fallback="build")
    return sanitize_name(str(resolved_spec.get("name") or "curate"), fallback="curate")


def _build_output_refs(workspace: WorkspaceState, artifact_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    raw_ref = f"artifact:{artifact_name}.raw"
    raw_path = workspace.outputs_dir / f"{artifact_name}.raw.jsonl"
    return {"raw": raw_ref}, {"raw": str(raw_path)}


def _curate_output_refs(workspace: WorkspaceState, artifact_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    audit_ref = f"artifact:{artifact_name}.audit"
    final_ref = f"artifact:{artifact_name}.final"
    audit_path = workspace.outputs_dir / f"{artifact_name}.audit.jsonl"
    final_path = workspace.outputs_dir / f"{artifact_name}.final.jsonl"
    return {"audit": audit_ref, "final": final_ref}, {"audit": str(audit_path), "final": str(final_path)}


def resolve_artifact_path(path_or_ref: str, workspace: WorkspaceState) -> str:
    """Resolve an `artifact:*` reference to a concrete path."""
    if not isinstance(path_or_ref, str):
        return path_or_ref
    if not path_or_ref.startswith("artifact:"):
        return path_or_ref
    return workspace.resolve_artifact(path_or_ref)["path"]


def resolve_curate_config(config: Dict[str, Any], workspace: WorkspaceState) -> Dict[str, Any]:
    """Resolve artifact references inside curation source configs."""
    resolved = json.loads(json.dumps(config))
    for source in resolved.get("sources", []):
        if isinstance(source, dict) and isinstance(source.get("path"), str):
            source["path"] = resolve_artifact_path(source["path"], workspace)
    return resolved


def inspect_source(source_spec: Optional[Dict[str, Any]] = None, *, workspace_root: Optional[str] = None) -> Dict[str, Any]:
    """Inspect a source preset or source spec without launching a job."""
    workspace = get_workspace(workspace_root)
    profile = workspace.load_profile()
    spec = deep_merge(profile.get("source_defaults", {}), _ensure_mapping("source_spec", source_spec))
    if not spec:
        raise ServiceError("source_spec is required")
    return inspect_source_spec(spec, default_hf_token_env=profile.get("hf_token_env"))


def start_job(
    *,
    action: str,
    recipe_name: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    artifact_name: Optional[str] = None,
    background: bool = True,
    workspace_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Create and optionally launch a dataprep job."""
    if action not in {"build", "curate"}:
        raise ServiceError(f"unsupported action: {action}")

    workspace = get_workspace(workspace_root)
    workspace.ensure_dirs()
    resolved_spec, selected_recipe = _resolve_recipe_payload(workspace, action, recipe_name, overrides)
    final_artifact_name = _derive_artifact_name(action, resolved_spec, selected_recipe, artifact_name)
    output_refs, output_paths = (
        _build_output_refs(workspace, final_artifact_name)
        if action == "build"
        else _curate_output_refs(workspace, final_artifact_name)
    )

    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "action": action,
        "status": "queued",
        "recipe_name": selected_recipe,
        "artifact_name": final_artifact_name,
        "spec": resolved_spec,
        "output_refs": output_refs,
        "output_paths": output_paths,
        "log_path": str(workspace.job_log_path(job_id)),
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "pid": None,
        "error": None,
        "result": None,
    }
    workspace.save_job(job)

    for kind, ref in output_refs.items():
        workspace.register_artifact(
            ref,
            output_paths[kind],
            kind=kind,
            job_id=job_id,
            ready=False,
            metadata={"artifact_name": final_artifact_name},
        )

    if background:
        process = subprocess.Popen(
            [sys.executable, "-m", "munqib_dataprep.worker", "--workspace-root", str(workspace.root), "--job-id", job_id],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        BACKGROUND_PROCS.append(process)
        workspace.update_job(job_id, {"pid": process.pid})
    else:
        run_job(job_id, workspace_root=str(workspace.root))

    return get_job_status(job_id, workspace_root=str(workspace.root))


def run_job(job_id: str, *, workspace_root: Optional[str] = None) -> Dict[str, Any]:
    """Execute a queued job to completion."""
    workspace = get_workspace(workspace_root)
    job = workspace.load_job(job_id)
    workspace.update_job(job_id, {"status": "running", "started_at": now_iso()})
    workspace.append_job_log(job_id, f"starting {job['action']} job")

    try:
        if job["action"] == "build":
            written = build_to_path(job["spec"], job["output_paths"]["raw"])
            result = {
                "docs_written": written,
                "raw_ref": job["output_refs"]["raw"],
                "raw_path": job["output_paths"]["raw"],
            }
        else:
            config = resolve_curate_config(_ensure_mapping("curate spec", job["spec"]), workspace)
            summary = curate_to_outputs(config, job["output_paths"]["audit"], job["output_paths"]["final"])
            result = {
                "summary": summary,
                "audit_ref": job["output_refs"]["audit"],
                "final_ref": job["output_refs"]["final"],
                "audit_path": job["output_paths"]["audit"],
                "final_path": job["output_paths"]["final"],
            }

        for kind, ref in job["output_refs"].items():
            workspace.register_artifact(
                ref,
                job["output_paths"][kind],
                kind=kind,
                job_id=job_id,
                ready=True,
                metadata={"artifact_name": job["artifact_name"]},
            )

        workspace.append_job_log(job_id, f"{job['action']} job completed successfully")
        workspace.update_job(
            job_id,
            {
                "status": "succeeded",
                "finished_at": now_iso(),
                "result": result,
                "error": None,
            },
        )
    except (BuildError, CurateError, ConfigError, ServiceError) as exc:
        workspace.append_job_log(job_id, f"{job['action']} job failed: {exc}")
        workspace.append_job_log(job_id, traceback.format_exc())
        workspace.update_job(
            job_id,
            {
                "status": "failed",
                "finished_at": now_iso(),
                "error": str(exc),
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        workspace.append_job_log(job_id, f"{job['action']} job failed unexpectedly: {exc}")
        workspace.append_job_log(job_id, traceback.format_exc())
        workspace.update_job(
            job_id,
            {
                "status": "failed",
                "finished_at": now_iso(),
                "error": f"unexpected error: {exc}",
            },
        )

    return get_job_status(job_id, workspace_root=str(workspace.root))


def get_job_status(job_id: str, *, workspace_root: Optional[str] = None) -> Dict[str, Any]:
    """Return the latest job status summary."""
    workspace = get_workspace(workspace_root)
    job = workspace.load_job(job_id)
    return {
        "job_id": job["job_id"],
        "action": job["action"],
        "status": job["status"],
        "recipe_name": job.get("recipe_name"),
        "artifact_name": job.get("artifact_name"),
        "output_refs": job.get("output_refs", {}),
        "pid": job.get("pid"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "error": job.get("error"),
        "result": job.get("result"),
        "log_path": job.get("log_path"),
    }


def get_job_logs(job_id: str, *, tail: int = 40, workspace_root: Optional[str] = None) -> Dict[str, Any]:
    """Return the tail of a job log."""
    workspace = get_workspace(workspace_root)
    job = workspace.load_job(job_id)
    log_payload = workspace.read_job_log(job_id, tail=tail)
    log_payload["status"] = job["status"]
    return log_payload


def get_job_artifacts(job_id: str, *, workspace_root: Optional[str] = None) -> Dict[str, Any]:
    """Resolve all artifacts produced by a job."""
    workspace = get_workspace(workspace_root)
    job = workspace.load_job(job_id)
    artifacts = {}
    for kind, ref in job.get("output_refs", {}).items():
        artifacts[kind] = workspace.resolve_artifact(ref)
    return {"job_id": job_id, "status": job["status"], "artifacts": artifacts}
