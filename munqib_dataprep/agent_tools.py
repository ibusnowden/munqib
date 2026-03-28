"""Native dataprep tools exposed directly to the Python agent runtime."""

from __future__ import annotations

from typing import Any, Dict

from .service import (
    ServiceError,
    get_job_artifacts,
    get_job_logs,
    get_job_status,
    get_workspace,
    inspect_source,
    start_job,
)


def _workspace_root(args: Dict[str, Any]) -> str | None:
    value = args.get("workspace_root")
    return value if isinstance(value, str) and value.strip() else None


def dataprep_profile_get(args: Dict[str, Any]) -> Dict[str, Any]:
    workspace = get_workspace(_workspace_root(args))
    return {"workspace_root": str(workspace.root), "profile": workspace.load_profile()}


def dataprep_profile_set(args: Dict[str, Any]) -> Dict[str, Any]:
    patch = args.get("patch")
    if not isinstance(patch, dict):
        raise ServiceError("patch must be an object")
    workspace = get_workspace(_workspace_root(args))
    return {"workspace_root": str(workspace.root), "profile": workspace.update_profile(patch)}


def dataprep_recipe_list(args: Dict[str, Any]) -> Dict[str, Any]:
    workspace = get_workspace(_workspace_root(args))
    return {"workspace_root": str(workspace.root), "recipes": workspace.list_recipes()}


def dataprep_recipe_save(args: Dict[str, Any]) -> Dict[str, Any]:
    name = args.get("name")
    recipe = args.get("recipe")
    if not isinstance(name, str) or not name.strip():
        raise ServiceError("name must be a non-empty string")
    if not isinstance(recipe, dict):
        raise ServiceError("recipe must be an object")
    workspace = get_workspace(_workspace_root(args))
    saved = workspace.save_recipe(name, recipe)
    response = {"workspace_root": str(workspace.root), "recipe": saved}
    if args.get("set_default"):
        profile = workspace.update_profile({"default_recipe": saved["name"]})
        response["profile"] = profile
    return response


def dataprep_source_inspect(args: Dict[str, Any]) -> Dict[str, Any]:
    source_spec = args.get("source_spec")
    if not isinstance(source_spec, dict):
        raise ServiceError("source_spec must be an object")
    return inspect_source(source_spec, workspace_root=_workspace_root(args))


def dataprep_job_start(args: Dict[str, Any]) -> Dict[str, Any]:
    action = args.get("action")
    if not isinstance(action, str):
        raise ServiceError("action must be a string")
    recipe_name = args.get("recipe_name")
    if recipe_name is not None and not isinstance(recipe_name, str):
        raise ServiceError("recipe_name must be a string")
    overrides = args.get("spec")
    if overrides is not None and not isinstance(overrides, dict):
        raise ServiceError("spec must be an object")
    artifact_name = args.get("artifact_name")
    if artifact_name is not None and not isinstance(artifact_name, str):
        raise ServiceError("artifact_name must be a string")
    background = args.get("background", True)
    if not isinstance(background, bool):
        raise ServiceError("background must be a boolean")
    return start_job(
        action=action,
        recipe_name=recipe_name,
        overrides=overrides,
        artifact_name=artifact_name,
        background=background,
        workspace_root=_workspace_root(args),
    )


def dataprep_job_status(args: Dict[str, Any]) -> Dict[str, Any]:
    job_id = args.get("job_id")
    if not isinstance(job_id, str):
        raise ServiceError("job_id must be a string")
    return get_job_status(job_id, workspace_root=_workspace_root(args))


def dataprep_job_logs(args: Dict[str, Any]) -> Dict[str, Any]:
    job_id = args.get("job_id")
    if not isinstance(job_id, str):
        raise ServiceError("job_id must be a string")
    tail = args.get("tail", 40)
    if not isinstance(tail, int):
        raise ServiceError("tail must be an integer")
    return get_job_logs(job_id, tail=tail, workspace_root=_workspace_root(args))


def dataprep_job_artifacts(args: Dict[str, Any]) -> Dict[str, Any]:
    job_id = args.get("job_id")
    if not isinstance(job_id, str):
        raise ServiceError("job_id must be a string")
    return get_job_artifacts(job_id, workspace_root=_workspace_root(args))


def get_agent_tools() -> Dict[str, tuple[str, Dict[str, Any], Any]]:
    """Return native dataprep tools in the same shape as the existing core tool registry."""
    workspace_prop = {"type": "string", "description": "Optional override for the .munqib workspace root"}
    object_prop = {"type": "object", "additionalProperties": True}

    return {
        "dataprep_profile_get": (
            "Get the repo-local dataprep workspace profile and defaults",
            {"type": "object", "properties": {"workspace_root": workspace_prop}, "required": []},
            dataprep_profile_get,
        ),
        "dataprep_profile_set": (
            "Update the repo-local dataprep workspace profile with a structured patch",
            {
                "type": "object",
                "properties": {
                    "workspace_root": workspace_prop,
                    "patch": object_prop,
                },
                "required": ["patch"],
            },
            dataprep_profile_set,
        ),
        "dataprep_recipe_list": (
            "List saved dataprep recipes in the repo-local workspace",
            {"type": "object", "properties": {"workspace_root": workspace_prop}, "required": []},
            dataprep_recipe_list,
        ),
        "dataprep_recipe_save": (
            "Save a named dataprep recipe for reuse and optionally set it as the default",
            {
                "type": "object",
                "properties": {
                    "workspace_root": workspace_prop,
                    "name": {"type": "string"},
                    "recipe": object_prop,
                    "set_default": {"type": "boolean"},
                },
                "required": ["name", "recipe"],
            },
            dataprep_recipe_save,
        ),
        "dataprep_source_inspect": (
            "Inspect a source preset or source spec, including shard selectors and dependency requirements",
            {
                "type": "object",
                "properties": {
                    "workspace_root": workspace_prop,
                    "source_spec": object_prop,
                },
                "required": ["source_spec"],
            },
            dataprep_source_inspect,
        ),
        "dataprep_job_start": (
            "Start a build or curate dataprep job, preferably in the background, using profile defaults and saved recipes",
            {
                "type": "object",
                "properties": {
                    "workspace_root": workspace_prop,
                    "action": {"type": "string", "enum": ["build", "curate"]},
                    "recipe_name": {"type": "string"},
                    "artifact_name": {"type": "string"},
                    "background": {"type": "boolean"},
                    "spec": object_prop,
                },
                "required": ["action"],
            },
            dataprep_job_start,
        ),
        "dataprep_job_status": (
            "Check the current status of a dataprep background job",
            {
                "type": "object",
                "properties": {
                    "workspace_root": workspace_prop,
                    "job_id": {"type": "string"},
                },
                "required": ["job_id"],
            },
            dataprep_job_status,
        ),
        "dataprep_job_logs": (
            "Read the tail of a dataprep job log",
            {
                "type": "object",
                "properties": {
                    "workspace_root": workspace_prop,
                    "job_id": {"type": "string"},
                    "tail": {"type": "integer"},
                },
                "required": ["job_id"],
            },
            dataprep_job_logs,
        ),
        "dataprep_job_artifacts": (
            "Resolve artifact refs and file paths produced by a dataprep job",
            {
                "type": "object",
                "properties": {
                    "workspace_root": workspace_prop,
                    "job_id": {"type": "string"},
                },
                "required": ["job_id"],
            },
            dataprep_job_artifacts,
        ),
    }

