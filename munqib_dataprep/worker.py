"""Background worker entrypoint for dataprep jobs."""

from __future__ import annotations

import argparse

from .service import run_job


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a queued munqib dataprep job")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    run_job(args.job_id, workspace_root=args.workspace_root)


if __name__ == "__main__":
    main()
