"""Legacy local-first dataprep helpers preserved for backward compatibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


def text_field(doc: Dict[str, Any]) -> str:
    """Return the text content of a JSON doc, checking 'text' then 'content'."""
    value = doc.get("text")
    if value is None:
        value = doc.get("content")
    return value if isinstance(value, str) else ""


def read_jsonl(path: Optional[str]) -> Iterator[Dict[str, Any]]:
    fh = open(path) if path else sys.stdin
    try:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload
    finally:
        if path:
            fh.close()


def write_jsonl(docs: Iterable[Dict[str, Any]], path: Optional[str]) -> int:
    fh = open(path, "w") if path else sys.stdout
    count = 0
    try:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1
    finally:
        if path:
            fh.close()
    return count


def heuristic_filter(
    docs: List[Dict[str, Any]],
    min_len: int = 100,
    max_symbol_ratio: float = 0.1,
    max_rep_ratio: float = 0.3,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Apply simple local quality filters."""
    kept = []
    stats = {
        "total": len(docs),
        "kept": 0,
        "removed_too_short": 0,
        "removed_symbol_ratio": 0,
        "removed_rep_ratio": 0,
    }

    for doc in docs:
        text = text_field(doc)
        size = len(text)

        if size < min_len:
            stats["removed_too_short"] += 1
            continue

        non_alnum = sum(1 for char in text if not char.isalnum() and not char.isspace())
        if non_alnum / size > max_symbol_ratio:
            stats["removed_symbol_ratio"] += 1
            continue

        nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if nonempty_lines:
            rep_ratio = 1.0 - len(set(nonempty_lines)) / len(nonempty_lines)
            if rep_ratio > max_rep_ratio:
                stats["removed_rep_ratio"] += 1
                continue

        kept.append(doc)

    stats["kept"] = len(kept)
    return kept, stats


def minhash_dedup(
    docs: List[Dict[str, Any]],
    threshold: float = 0.85,
    num_hashes: int = 128,
    ngram_size: int = 5,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Local MinHash-based near-deduplication.

    The threshold is accepted for backwards compatibility but is not used in the
    simplified band-collision implementation.
    """
    del threshold

    prime = (1 << 61) - 1
    rng = random.Random(seed)
    a = [rng.randint(1, prime - 1) for _ in range(num_hashes)]
    b = [rng.randint(0, prime - 1) for _ in range(num_hashes)]

    def ngrams(text: str, n: int) -> Iterator[str]:
        for index in range(len(text) - n + 1):
            yield text[index : index + n]

    def stable_hash(gram: str) -> int:
        payload = gram.encode("utf-8", errors="ignore")
        return int(hashlib.md5(payload, usedforsecurity=False).hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF

    def minhash_sig(text: str) -> List[int]:
        sig = [prime] * num_hashes
        for gram in ngrams(text, ngram_size):
            hashed = stable_hash(gram)
            for index in range(num_hashes):
                value = (a[index] * hashed + b[index]) % prime
                if value < sig[index]:
                    sig[index] = value
        return sig

    n_bands = max(1, num_hashes // 8)
    rows_per_band = max(1, num_hashes // n_bands)
    signatures = [minhash_sig(text_field(doc)) for doc in docs]

    bucket_map: Dict[Any, List[int]] = {}
    for idx, sig in enumerate(signatures):
        for band in range(n_bands):
            start = band * rows_per_band
            bucket = (band, tuple(sig[start : start + rows_per_band]))
            bucket_map.setdefault(bucket, []).append(idx)

    parent = list(range(len(docs)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for candidates in bucket_map.values():
        if len(candidates) < 2:
            continue
        root = candidates[0]
        for idx in candidates[1:]:
            union(root, idx)

    kept = []
    seen_roots = set()
    for idx, doc in enumerate(docs):
        root = find(idx)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        kept.append(doc)

    removed = len(docs) - len(kept)
    return kept, removed


def quality_score(
    docs: List[Dict[str, Any]],
    gold_docs: List[Dict[str, Any]],
    min_score: Optional[float] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Train a simple TF-IDF classifier against a gold reference set."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("error: score requires scikit-learn — pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    gold_texts = [text_field(doc) for doc in gold_docs]
    rng = random.Random(seed)
    neg_sample = rng.sample(docs, min(len(gold_texts) * 3, len(docs)))
    neg_texts = [text_field(doc) for doc in neg_sample]

    train_x = gold_texts + neg_texts
    train_y = [1] * len(gold_texts) + [0] * len(neg_texts)

    vectorizer = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(train_x)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(matrix, train_y)

    doc_texts = [text_field(doc) for doc in docs]
    scores = clf.predict_proba(vectorizer.transform(doc_texts))[:, 1]

    result = []
    for doc, score in zip(docs, scores):
        item = dict(doc)
        item["quality_score"] = round(float(score), 4)
        if min_score is None or score >= min_score:
            result.append(item)
    return result


def mixture_sample(
    sources_with_ratios: List[Tuple[str, float]],
    n: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample n docs proportionally from multiple JSONL sources."""
    rng = random.Random(seed)
    total_weight = sum(weight for _, weight in sources_with_ratios)
    result: List[Dict[str, Any]] = []
    for path, weight in sources_with_ratios:
        requested = round(n * weight / total_weight)
        pool = list(read_jsonl(path))
        if len(pool) <= requested:
            result.extend(pool)
        else:
            result.extend(rng.sample(pool, requested))
    rng.shuffle(result)
    return result[:n]


def format_num(value: int) -> str:
    return f"{value:,}"


def corpus_stats(docs: List[Dict[str, Any]], field: str = "text") -> str:
    """Compute JSONL corpus statistics."""
    lengths = [len(doc.get(field) or "") for doc in docs]

    if not lengths:
        return "docs: 0\nchars: 0\ntokens: ~0  (chars / 4)\nlengths: no docs found"

    count = len(lengths)
    total_chars = sum(lengths)
    tokens = total_chars // 4
    sorted_lens = sorted(lengths)
    p50 = sorted_lens[len(sorted_lens) // 2]
    p90 = sorted_lens[len(sorted_lens) * 9 // 10]

    return (
        f"docs:    {format_num(count)}\n"
        f"chars:   {format_num(total_chars)}\n"
        f"tokens:  ~{format_num(tokens)}  (chars / 4)\n"
        f"lengths: min={format_num(min(lengths))}  mean={format_num(total_chars // count)}"
        f"  p50={format_num(p50)}  p90={format_num(p90)}  max={format_num(max(lengths))}\n"
        f"format:  jsonl  field: {field}"
    )


def cmd_filter(args: argparse.Namespace) -> None:
    docs = list(read_jsonl(args.input))
    kept, stats = heuristic_filter(
        docs,
        min_len=args.min_len,
        max_symbol_ratio=args.max_symbol_ratio,
        max_rep_ratio=args.max_rep_ratio,
    )
    write_jsonl(kept, args.output)
    print(
        f"filter: {stats['total']} → {stats['kept']} kept "
        f"(removed: {stats['removed_too_short']} too-short, "
        f"{stats['removed_symbol_ratio']} symbol-ratio, "
        f"{stats['removed_rep_ratio']} rep-ratio)",
        file=sys.stderr,
    )


def cmd_dedup(args: argparse.Namespace) -> None:
    docs = list(read_jsonl(args.input))
    kept, removed = minhash_dedup(
        docs,
        threshold=args.threshold,
        num_hashes=args.num_hashes,
        ngram_size=args.ngram_size,
        seed=args.seed,
    )
    write_jsonl(kept, args.output)
    print(f"dedup: {len(docs)} → {len(kept)} kept ({removed} removed)", file=sys.stderr)


def cmd_score(args: argparse.Namespace) -> None:
    docs = list(read_jsonl(args.input))
    gold_docs = list(read_jsonl(args.gold))
    scored = quality_score(docs, gold_docs, min_score=args.min_score, seed=args.seed)
    write_jsonl(scored, args.output)
    print(f"score: {len(docs)} → {len(scored)} kept (min_score={args.min_score})", file=sys.stderr)


def cmd_sample(args: argparse.Namespace) -> None:
    pairs: List[Tuple[str, float]] = []
    for item in args.sources.split(","):
        entry = item.strip()
        if ":" in entry:
            path, weight_str = entry.rsplit(":", 1)
            pairs.append((path, float(weight_str)))
        else:
            pairs.append((entry, 1.0))
    docs = mixture_sample(pairs, n=args.n, seed=args.seed)
    write_jsonl(docs, args.output)
    print(f"sample: {len(docs)} docs written", file=sys.stderr)


def cmd_stats(args: argparse.Namespace) -> None:
    docs = list(read_jsonl(args.input))
    print(corpus_stats(docs, field=args.field))


def register_legacy_subcommands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("filter", help="Heuristic quality filter")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-len", type=int, default=100)
    parser.add_argument("--max-symbol-ratio", type=float, default=0.1)
    parser.add_argument("--max-rep-ratio", type=float, default=0.3)
    parser.set_defaults(func=cmd_filter)

    parser = subparsers.add_parser("dedup", help="MinHash near-deduplication")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--num-hashes", type=int, default=128)
    parser.add_argument("--ngram-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(func=cmd_dedup)

    parser = subparsers.add_parser("score", help="Quality scoring (TF-IDF + LogReg)")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--gold", required=True, help="JSONL file of high-quality reference docs")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(func=cmd_score)

    parser = subparsers.add_parser("sample", help="Proportional mixture sampling")
    parser.add_argument("--sources", required=True, help="src1.jsonl:0.4,src2.jsonl:0.6")
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(func=cmd_sample)

    parser = subparsers.add_parser("stats", help="Corpus statistics")
    parser.add_argument("--input", default=None)
    parser.add_argument("--field", default="text")
    parser.set_defaults(func=cmd_stats)
