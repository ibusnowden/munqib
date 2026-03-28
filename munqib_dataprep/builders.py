"""Dataset builders for canonical JSONL generation."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import os
from typing import Any, Dict, Iterable, Iterator, Optional

from .legacy import write_jsonl


HF_DATASETS = {
    "fineweb-edu": "HuggingFaceFW/fineweb-edu",
    "nemotron-climbmix": "nvidia/Nemotron-ClimbMix",
}

SOURCE_SELECTOR_FIELDS = [
    "source",
    "subset",
    "split",
    "files",
    "shards",
    "row_start",
    "row_count",
    "limit",
    "streaming",
    "hf_token_env",
]


class BuildError(RuntimeError):
    """Raised when a source builder cannot run."""


def stable_doc_id(
    dataset: str,
    split: str,
    subset: Optional[str],
    upstream_id: Optional[str],
    text: str,
) -> str:
    payload = json.dumps(
        {
            "dataset": dataset,
            "split": split,
            "subset": subset or "default",
            "upstream_id": upstream_id,
            "text_prefix": text[:512],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()


def canonical_document(
    *,
    source: str,
    dataset: str,
    split: str,
    subset: Optional[str],
    text: str,
    upstream_id: Optional[str] = None,
    source_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    text = text or ""
    return {
        "id": stable_doc_id(dataset, split, subset, upstream_id, text),
        "text": text,
        "source": source,
        "dataset": dataset,
        "split": split,
        "subset": subset or "default",
        "upstream_id": upstream_id,
        "source_url": source_url,
        "metadata": metadata or {},
    }


def split_csv(value: Any) -> Optional[list[str]]:
    """Normalize a comma-separated or list input into a list of strings."""
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or None
    return [str(value)]


def normalize_source_spec(
    source_or_spec: str | Dict[str, Any],
    *,
    split: str = "train",
    subset: Optional[str] = None,
    limit: Optional[int] = None,
    row_start: int = 0,
    row_count: Optional[int] = None,
    files: Any = None,
    shards: Any = None,
    streaming: bool = True,
    hf_token_env: Optional[str] = None,
    default_hf_token_env: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize user input into a typed source selector spec."""
    base: Dict[str, Any]
    if isinstance(source_or_spec, dict):
        base = dict(source_or_spec)
    else:
        base = {"source": source_or_spec}

    spec = {
        "source": base.get("source"),
        "subset": base.get("subset", subset),
        "split": base.get("split", split),
        "files": split_csv(base.get("files", files)),
        "shards": split_csv(base.get("shards", shards)),
        "row_start": int(base.get("row_start", row_start or 0) or 0),
        "row_count": base.get("row_count", row_count),
        "limit": base.get("limit", limit),
        "streaming": bool(base.get("streaming", streaming)),
        "hf_token_env": base.get("hf_token_env", hf_token_env or default_hf_token_env),
    }
    if spec["row_count"] is not None:
        spec["row_count"] = int(spec["row_count"])
    if spec["limit"] is not None:
        spec["limit"] = int(spec["limit"])
    if not spec["source"]:
        raise BuildError("source spec must define 'source'")
    if spec["source"] not in HF_DATASETS:
        raise BuildError(f"unsupported source: {spec['source']}")
    return spec


def _require_datasets() -> Any:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise BuildError("build requires the `datasets` package: pip install datasets") from exc
    return load_dataset


def _resolve_hf_token(spec: Dict[str, Any]) -> Optional[str]:
    env_name = spec.get("hf_token_env")
    if not env_name:
        return None
    return os.environ.get(str(env_name))


def _iter_hf_rows(spec: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    load_dataset = _require_datasets()
    kwargs: Dict[str, Any] = {
        "name": spec.get("subset"),
        "split": spec.get("split", "train"),
        "streaming": spec.get("streaming", True),
    }
    data_files = spec.get("files") or spec.get("shards")
    if data_files:
        kwargs["data_files"] = data_files
    token = _resolve_hf_token(spec)
    if token:
        kwargs["token"] = token
    stream = load_dataset(HF_DATASETS[spec["source"]], **kwargs)
    start = max(0, int(spec.get("row_start", 0) or 0))
    stop = None
    if spec.get("row_count") is not None:
        stop = start + max(0, int(spec["row_count"]))
    rows = itertools.islice(stream, start, stop)
    if spec.get("limit") is not None:
        rows = itertools.islice(rows, max(0, int(spec["limit"])))
    return rows


def map_fineweb_edu_row(row: Dict[str, Any], split: str, subset: Optional[str]) -> Dict[str, Any]:
    text = str(row.get("text") or row.get("content") or "")
    upstream_id = row.get("id")
    source_url = row.get("url") or row.get("source_url")
    metadata = {
        "upstream_language": row.get("language"),
        "upstream_language_score": row.get("language_score"),
        "upstream_score": row.get("score"),
        "token_count": row.get("token_count"),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    return canonical_document(
        source="fineweb-edu",
        dataset=HF_DATASETS["fineweb-edu"],
        split=split,
        subset=subset,
        text=text,
        upstream_id=str(upstream_id) if upstream_id is not None else None,
        source_url=str(source_url) if source_url is not None else None,
        metadata=metadata,
    )


def _decode_climbmix_tokens(tokens: Any) -> str:
    try:
        import tiktoken  # type: ignore
    except ImportError as exc:
        raise BuildError("Nemotron-ClimbMix build requires `tiktoken`: pip install tiktoken") from exc
    encoding = tiktoken.get_encoding("gpt2")
    token_list = [int(token) for token in tokens]
    return encoding.decode(token_list)


def map_nemotron_climbmix_row(row: Dict[str, Any], split: str, subset: Optional[str]) -> Dict[str, Any]:
    tokens = row.get("tokens")
    if not isinstance(tokens, list):
        raise BuildError("Nemotron-ClimbMix rows must contain a 'tokens' list")
    text = _decode_climbmix_tokens(tokens)
    upstream_id = row.get("id")
    if upstream_id is None:
        signature = {
            "cluster_id": row.get("cluster_id"),
            "token_count": row.get("token_count"),
            "head": tokens[:64],
        }
        upstream_id = hashlib.md5(
            json.dumps(signature, sort_keys=True).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()
    metadata = {
        "cluster_id": row.get("cluster_id"),
        "token_count": row.get("token_count"),
        "tokenizer": "gpt2",
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    return canonical_document(
        source="nemotron-climbmix",
        dataset=HF_DATASETS["nemotron-climbmix"],
        split=split,
        subset=subset,
        text=text,
        upstream_id=str(upstream_id),
        metadata=metadata,
    )


def iter_source_documents(
    source_or_spec: str | Dict[str, Any],
    *,
    split: str = "train",
    subset: Optional[str] = None,
    limit: Optional[int] = None,
    row_start: int = 0,
    row_count: Optional[int] = None,
    files: Any = None,
    shards: Any = None,
    streaming: bool = True,
    hf_token_env: Optional[str] = None,
    default_hf_token_env: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    spec = normalize_source_spec(
        source_or_spec,
        split=split,
        subset=subset,
        limit=limit,
        row_start=row_start,
        row_count=row_count,
        files=files,
        shards=shards,
        streaming=streaming,
        hf_token_env=hf_token_env,
        default_hf_token_env=default_hf_token_env,
    )
    mapper = map_fineweb_edu_row if spec["source"] == "fineweb-edu" else map_nemotron_climbmix_row
    for row in _iter_hf_rows(spec):
        yield mapper(dict(row), spec["split"], spec["subset"])


def build_to_path(source_spec: Dict[str, Any], output_path: str, *, default_hf_token_env: Optional[str] = None) -> int:
    """Write canonical JSONL for a source spec and return the number of rows written."""
    docs = iter_source_documents(source_spec, default_hf_token_env=default_hf_token_env)
    return write_jsonl(docs, output_path)


def inspect_source_spec(source_spec: Dict[str, Any], *, default_hf_token_env: Optional[str] = None) -> Dict[str, Any]:
    """Return metadata about a source preset and the supported selectors."""
    spec = normalize_source_spec(source_spec, default_hf_token_env=default_hf_token_env)
    return {
        "source": spec["source"],
        "dataset_id": HF_DATASETS[spec["source"]],
        "normalized_spec": spec,
        "supported_selectors": SOURCE_SELECTOR_FIELDS,
        "supports_shards": True,
        "dependencies": {
            "datasets_installed": importlib.util.find_spec("datasets") is not None,
            "tiktoken_installed": importlib.util.find_spec("tiktoken") is not None,
        },
        "notes": [
            "Use files or shards to target specific remote parquet/json files when supported by the dataset builder.",
            "Use row_start, row_count, and limit to target windows without changing the recipe.",
        ],
    }


def run_build_command(args: Any) -> None:
    source_spec = {
        "source": args.source,
        "split": args.split,
        "subset": args.subset,
        "limit": args.limit,
        "row_start": args.row_start,
        "row_count": args.row_count,
        "files": split_csv(args.files),
        "shards": split_csv(args.shards),
        "streaming": not args.no_streaming,
        "hf_token_env": args.hf_token_env,
    }
    written = build_to_path(source_spec, args.output)
    print(
        f"build: wrote {written} canonical docs from {args.source} "
        f"(split={args.split}, subset={args.subset or 'default'})"
    )
