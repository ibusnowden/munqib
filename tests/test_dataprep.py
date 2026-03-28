from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import core
from munqib_dataprep.builders import inspect_source_spec, map_fineweb_edu_row
from munqib_dataprep.config import load_pipeline_config, load_yaml_without_dependency
from munqib_dataprep.curate import CurateError, apply_pii_redaction, canonicalize_existing_doc, curate_documents, iter_config_sources
from munqib_dataprep.legacy import corpus_stats
from munqib_dataprep.service import get_job_artifacts, get_job_logs, get_job_status, start_job
from munqib_dataprep.workspace import WorkspaceState


ROOT = Path(__file__).resolve().parents[1]


class DataPrepTests(unittest.TestCase):
    def write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def wait_for_job(self, workspace_root: Path, job_id: str, timeout_s: float = 10.0) -> dict:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status = get_job_status(job_id, workspace_root=str(workspace_root))
            if status["status"] in {"succeeded", "failed"}:
                return status
            time.sleep(0.1)
        self.fail(f"job {job_id} did not finish before timeout")

    def test_map_fineweb_row(self) -> None:
        row = {
            "id": "fw-1",
            "text": "A classroom lesson about algebra.",
            "url": "https://example.edu/algebra",
            "language": "en",
            "language_score": 0.99,
            "score": 4.2,
            "token_count": 8,
        }
        doc = map_fineweb_edu_row(row, split="train", subset="sample")
        self.assertEqual(doc["source"], "fineweb-edu")
        self.assertEqual(doc["dataset"], "HuggingFaceFW/fineweb-edu")
        self.assertEqual(doc["source_url"], "https://example.edu/algebra")
        self.assertEqual(doc["metadata"]["upstream_language"], "en")

    def test_source_inspect_exposes_shard_selectors(self) -> None:
        payload = inspect_source_spec({"source": "fineweb-edu", "split": "train"})
        self.assertEqual(payload["dataset_id"], "HuggingFaceFW/fineweb-edu")
        self.assertTrue(payload["supports_shards"])
        self.assertIn("shards", payload["supported_selectors"])

    def test_load_yaml_config_without_pyyaml(self) -> None:
        config = load_yaml_without_dependency(
            textwrap.dedent(
                """
                sources:
                  - path: raw.jsonl
                classifiers:
                  toxicity:
                    enabled: true  # gate toxicity
                    gate: true
                    max_score: 0.3
                """
            ).strip()
            + "\n"
        )
        self.assertEqual(config["sources"][0]["path"], "raw.jsonl")
        self.assertTrue(config["classifiers"]["toxicity"]["gate"])
        self.assertIs(config["classifiers"]["toxicity"]["enabled"], True)

    def test_iter_config_sources_forwards_all_selectors(self) -> None:
        config = {
            "sources": [
                {
                    "source": "fineweb-edu",
                    "split": "train",
                    "subset": "sample-10BT",
                    "files": ["data/shard-0001.parquet"],
                    "shards": ["data/shard-0001.parquet"],
                    "row_start": 25,
                    "row_count": 10,
                    "limit": 5,
                    "streaming": False,
                    "hf_token_env": "ALT_HF_TOKEN",
                }
            ]
        }

        captured: list[dict] = []

        def fake_iter_source_documents(source_spec: dict) -> list[dict]:
            captured.append(dict(source_spec))
            return []

        with patch("munqib_dataprep.curate.iter_source_documents", side_effect=fake_iter_source_documents):
            self.assertEqual(list(iter_config_sources(config)), [])

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["files"], ["data/shard-0001.parquet"])
        self.assertEqual(captured[0]["shards"], ["data/shard-0001.parquet"])
        self.assertEqual(captured[0]["row_start"], 25)
        self.assertEqual(captured[0]["row_count"], 10)
        self.assertEqual(captured[0]["streaming"], False)
        self.assertEqual(captured[0]["hf_token_env"], "ALT_HF_TOKEN")

    def test_build_job_inherits_profile_hf_token_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / ".munqib"
            workspace = WorkspaceState(workspace_root)
            workspace.update_profile({"hf_token_env": "ALT_HF_TOKEN"})

            with patch("munqib_dataprep.service.build_to_path", return_value=0):
                status = start_job(
                    action="build",
                    overrides={"source": "fineweb-edu"},
                    artifact_name="fineweb",
                    background=False,
                    workspace_root=str(workspace_root),
                )

            job = workspace.load_job(status["job_id"])
            self.assertEqual(job["spec"]["hf_token_env"], "ALT_HF_TOKEN")

    def test_apply_pii_redaction_honors_presidio_backend(self) -> None:
        with patch(
            "munqib_dataprep.curate.redact_pii_presidio",
            return_value=("hello {{PERSON}}", {"backend": "presidio", "entities": {"PERSON": 1}, "matches": 1}),
        ) as mocked:
            text, pii = apply_pii_redaction("hello Jane Doe", {"enabled": True, "backend": "presidio"})

        mocked.assert_called_once()
        self.assertEqual(text, "hello {{PERSON}}")
        self.assertEqual(pii["backend"], "presidio")

    def test_apply_pii_redaction_rejects_unknown_backend(self) -> None:
        with self.assertRaises(CurateError):
            apply_pii_redaction("hello Jane Doe", {"enabled": True, "backend": "made-up"})

    def test_curate_outputs_audit_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "raw.jsonl"
            docs = [
                {
                    "id": "keep-1",
                    "text": "This lesson explains fractions in clear English for students. Email me at test@example.com.",
                    "source": "fixture",
                    "dataset": "fixture",
                },
                {
                    "id": "dup-1",
                    "text": "This lesson explains fractions in clear English for students. Email me at test@example.com.",
                    "source": "fixture",
                    "dataset": "fixture",
                },
                {
                    "id": "drop-short",
                    "text": "tiny",
                    "source": "fixture",
                    "dataset": "fixture",
                },
            ]
            self.write_jsonl(raw_path, docs)

            config = load_pipeline_config(str(ROOT / "configs" / "fineweb_climbmix.yaml"))
            config["sources"] = [{"path": str(raw_path), "name": "fixture"}]
            config["heuristics"]["min_chars"] = 40
            config["classifiers"]["toxicity"]["gate"] = False

            audit_docs, final_docs, summary = curate_documents(config)

            self.assertEqual(summary["input_docs"], 3)
            self.assertEqual(len(audit_docs), 3)
            self.assertEqual(len(final_docs), 1)
            self.assertIn("{{EMAIL_ADDRESS}}", final_docs[0]["text"])

            duplicate = next(doc for doc in audit_docs if doc["id"] == "dup-1")
            self.assertIn("dedup:exact", duplicate["drop_reasons"])

            short_doc = next(doc for doc in audit_docs if doc["id"] == "drop-short")
            self.assertTrue(any(reason.startswith("heuristic:") for reason in short_doc["drop_reasons"]))

    def test_workspace_profile_and_recipe_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = WorkspaceState(Path(tmp) / ".munqib")
            profile = workspace.update_profile({"default_recipe": "fractions"})
            saved = workspace.save_recipe(
                "fractions",
                {
                    "description": "small curation recipe",
                    "curate": {"sources": [{"path": "artifact:fractions.raw"}]},
                },
            )
            recipes = workspace.list_recipes()

            self.assertEqual(profile["default_recipe"], "fractions")
            self.assertEqual(saved["name"], "fractions")
            self.assertEqual(recipes[0]["name"], "fractions")

    def test_core_registers_native_dataprep_tools(self) -> None:
        schema_names = {entry["name"] for entry in core.make_schema()}
        self.assertIn("dataprep_job_start", schema_names)
        self.assertIn("dataprep_profile_get", schema_names)

    def test_background_curate_job_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            workspace_root = project_root / ".munqib"
            raw_path = project_root / "raw.jsonl"
            self.write_jsonl(
                raw_path,
                [
                    {
                        "id": "doc-1",
                        "text": "This lesson introduces fractions and decimals in a clear English explanation. Reach me at test@example.com.",
                        "source": "fixture",
                        "dataset": "fixture",
                    }
                ],
            )

            status = start_job(
                action="curate",
                overrides={
                    "sources": [{"path": str(raw_path), "name": "fixture"}],
                    "heuristics": {"min_chars": 40},
                    "classifiers": {"toxicity": {"enabled": True, "gate": False}},
                },
                artifact_name="fixture",
                background=True,
                workspace_root=str(workspace_root),
            )
            final_status = self.wait_for_job(workspace_root, status["job_id"])
            logs = get_job_logs(status["job_id"], workspace_root=str(workspace_root))
            artifacts = get_job_artifacts(status["job_id"], workspace_root=str(workspace_root))

            self.assertEqual(final_status["status"], "succeeded")
            self.assertTrue(any("completed successfully" in line for line in logs["lines"]))
            self.assertTrue(Path(artifacts["artifacts"]["audit"]["path"]).exists())
            self.assertTrue(Path(artifacts["artifacts"]["final"]["path"]).exists())

    def test_canonicalize_existing_doc_preserves_extra_fields(self) -> None:
        doc = canonicalize_existing_doc({"text": "hello world", "foo": "bar"}, source_hint="raw")
        self.assertEqual(doc["metadata"]["extra_fields"]["foo"], "bar")
        self.assertEqual(doc["source"], "raw")

    def test_legacy_stats_still_work(self) -> None:
        report = corpus_stats([{"text": "hello"}, {"text": "world!"}])
        self.assertIn("docs:    2", report)

    def test_cli_legacy_stats_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_path = Path(tmp) / "raw.jsonl"
            self.write_jsonl(raw_path, [{"text": "hello world"}])
            proc = subprocess.run(
                [sys.executable, str(ROOT / "dataprep.py"), "stats", "--input", str(raw_path)],
                cwd=str(ROOT),
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("docs:    1", proc.stdout)

    def test_cli_profile_get_and_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_root = str(Path(tmp) / ".munqib")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "dataprep.py"),
                    "profile",
                    "set",
                    "--workspace-root",
                    workspace_root,
                    "--patch",
                    '{"default_recipe":"algebra"}',
                ],
                cwd=str(ROOT),
                check=True,
                capture_output=True,
                text=True,
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "dataprep.py"),
                    "profile",
                    "get",
                    "--workspace-root",
                    workspace_root,
                ],
                cwd=str(ROOT),
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["default_recipe"], "algebra")


if __name__ == "__main__":
    unittest.main()
