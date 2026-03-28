"""Command-line interface for munqib dataprep."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from .builders import run_build_command
from .config import load_pipeline_config, load_structured_file
from .curate import run_curate_command
from .legacy import register_legacy_subcommands
from .service import get_job_artifacts, get_job_logs, get_job_status, get_workspace, start_job


def _parse_json_patch(raw: str) -> Dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("patch must decode to an object")
    return payload


def cmd_build(args: argparse.Namespace) -> None:
    if args.background:
        if not args.source and not args.recipe:
            raise SystemExit("build requires --source or --recipe when using --background")
        spec = {}
        for key in ("source", "split", "subset", "limit", "row_start", "row_count", "hf_token_env"):
            value = getattr(args, key)
            if value is not None:
                spec[key] = value
        if args.files:
            spec["files"] = args.files
        if args.shards:
            spec["shards"] = args.shards
        spec["streaming"] = not args.no_streaming
        result = start_job(
            action="build",
            recipe_name=args.recipe,
            overrides=spec or None,
            artifact_name=args.artifact_name,
            background=True,
            workspace_root=args.workspace_root,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if not args.output:
        raise SystemExit("build requires --output unless --background is used")
    if not args.source:
        raise SystemExit("build requires --source unless --background with --recipe is used")
    run_build_command(args)


def cmd_curate(args: argparse.Namespace) -> None:
    if args.background:
        if not args.config and not args.recipe:
            raise SystemExit("curate requires --config or --recipe when using --background")
        overrides = load_pipeline_config(args.config) if args.config else {}
        result = start_job(
            action="curate",
            recipe_name=args.recipe,
            overrides=overrides or None,
            artifact_name=args.artifact_name,
            background=True,
            workspace_root=args.workspace_root,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if not args.config:
        raise SystemExit("curate requires --config unless --background with --recipe is used")
    if not args.audit_output or not args.final_output:
        raise SystemExit("curate requires --audit-output and --final-output unless --background is used")
    config = load_pipeline_config(args.config)
    run_curate_command(args, config)


def cmd_profile_get(args: argparse.Namespace) -> None:
    workspace = get_workspace(args.workspace_root)
    print(json.dumps(workspace.load_profile(), indent=2, ensure_ascii=False))


def cmd_profile_set(args: argparse.Namespace) -> None:
    workspace = get_workspace(args.workspace_root)
    patch = _parse_json_patch(args.patch)
    profile = workspace.update_profile(patch)
    print(json.dumps(profile, indent=2, ensure_ascii=False))


def cmd_recipe_list(args: argparse.Namespace) -> None:
    workspace = get_workspace(args.workspace_root)
    print(json.dumps(workspace.list_recipes(), indent=2, ensure_ascii=False))


def cmd_recipe_show(args: argparse.Namespace) -> None:
    workspace = get_workspace(args.workspace_root)
    print(json.dumps(workspace.load_recipe(args.name), indent=2, ensure_ascii=False))


def cmd_recipe_save(args: argparse.Namespace) -> None:
    workspace = get_workspace(args.workspace_root)
    recipe = load_structured_file(args.file)
    saved = workspace.save_recipe(args.name, recipe)
    if args.set_default:
        workspace.update_profile({"default_recipe": saved["name"]})
    print(json.dumps(saved, indent=2, ensure_ascii=False))


def cmd_job_status(args: argparse.Namespace) -> None:
    print(json.dumps(get_job_status(args.job_id, workspace_root=args.workspace_root), indent=2, ensure_ascii=False))


def cmd_job_logs(args: argparse.Namespace) -> None:
    print(json.dumps(get_job_logs(args.job_id, tail=args.tail, workspace_root=args.workspace_root), indent=2, ensure_ascii=False))


def cmd_job_artifacts(args: argparse.Namespace) -> None:
    print(json.dumps(get_job_artifacts(args.job_id, workspace_root=args.workspace_root), indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretraining data pipeline: build / curate / profile / recipe / job / filter / dedup / score / sample / stats"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="Download a source dataset and emit canonical JSONL")
    build.add_argument("--source", choices=["fineweb-edu", "nemotron-climbmix"], default=None)
    build.add_argument("--split", default="train")
    build.add_argument("--subset", default=None)
    build.add_argument("--limit", type=int, default=None)
    build.add_argument("--row-start", type=int, default=0)
    build.add_argument("--row-count", type=int, default=None)
    build.add_argument("--files", default=None, help="Comma-separated list of remote data files")
    build.add_argument("--shards", default=None, help="Comma-separated list of remote shard patterns")
    build.add_argument("--hf-token-env", default=None)
    build.add_argument("--no-streaming", action="store_true")
    build.add_argument("--output", default=None)
    build.add_argument("--recipe", default=None)
    build.add_argument("--artifact-name", default=None)
    build.add_argument("--background", action="store_true")
    build.add_argument("--workspace-root", default=None)
    build.set_defaults(func=cmd_build)

    curate = sub.add_parser("curate", help="Run config-driven text curation and export audit/final JSONL")
    curate.add_argument("--config", default=None, help="YAML or JSON pipeline config")
    curate.add_argument("--recipe", default=None)
    curate.add_argument("--audit-output", default=None)
    curate.add_argument("--final-output", default=None)
    curate.add_argument("--artifact-name", default=None)
    curate.add_argument("--background", action="store_true")
    curate.add_argument("--workspace-root", default=None)
    curate.set_defaults(func=cmd_curate)

    profile = sub.add_parser("profile", help="Get or update the repo-local dataprep profile")
    profile_sub = profile.add_subparsers(dest="profile_cmd", required=True)
    profile_get = profile_sub.add_parser("get", help="Show the current profile")
    profile_get.add_argument("--workspace-root", default=None)
    profile_get.set_defaults(func=cmd_profile_get)
    profile_set = profile_sub.add_parser("set", help="Merge a JSON patch into the profile")
    profile_set.add_argument("--patch", required=True, help='JSON object patch, for example {"default_recipe":"fineweb"}')
    profile_set.add_argument("--workspace-root", default=None)
    profile_set.set_defaults(func=cmd_profile_set)

    recipe = sub.add_parser("recipe", help="Save and inspect reusable dataprep recipes")
    recipe_sub = recipe.add_subparsers(dest="recipe_cmd", required=True)
    recipe_list = recipe_sub.add_parser("list", help="List saved recipes")
    recipe_list.add_argument("--workspace-root", default=None)
    recipe_list.set_defaults(func=cmd_recipe_list)
    recipe_show = recipe_sub.add_parser("show", help="Show a saved recipe")
    recipe_show.add_argument("--name", required=True)
    recipe_show.add_argument("--workspace-root", default=None)
    recipe_show.set_defaults(func=cmd_recipe_show)
    recipe_save = recipe_sub.add_parser("save", help="Save a recipe from a YAML or JSON file")
    recipe_save.add_argument("--name", required=True)
    recipe_save.add_argument("--file", required=True)
    recipe_save.add_argument("--set-default", action="store_true")
    recipe_save.add_argument("--workspace-root", default=None)
    recipe_save.set_defaults(func=cmd_recipe_save)

    job = sub.add_parser("job", help="Inspect background dataprep jobs")
    job_sub = job.add_subparsers(dest="job_cmd", required=True)
    job_status = job_sub.add_parser("status", help="Show job status")
    job_status.add_argument("--job-id", required=True)
    job_status.add_argument("--workspace-root", default=None)
    job_status.set_defaults(func=cmd_job_status)
    job_logs = job_sub.add_parser("logs", help="Show job logs")
    job_logs.add_argument("--job-id", required=True)
    job_logs.add_argument("--tail", type=int, default=40)
    job_logs.add_argument("--workspace-root", default=None)
    job_logs.set_defaults(func=cmd_job_logs)
    job_artifacts = job_sub.add_parser("artifacts", help="Show job artifacts")
    job_artifacts.add_argument("--job-id", required=True)
    job_artifacts.add_argument("--workspace-root", default=None)
    job_artifacts.set_defaults(func=cmd_job_artifacts)

    register_legacy_subcommands(sub)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
