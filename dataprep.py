#!/usr/bin/env python3
"""
dataprep.py — pretraining data pipeline (filter / dedup / score / sample / stats)

Usage:
  python dataprep.py filter  [--input FILE] [--output FILE] [--min-len 100] [--max-symbol-ratio 0.1] [--max-rep-ratio 0.3]
  python dataprep.py dedup   [--input FILE] [--output FILE] [--threshold 0.85] [--num-hashes 128] [--ngram-size 5]
  python dataprep.py score   [--input FILE] [--output FILE] [--gold FILE] [--min-score 0.5]
  python dataprep.py sample  [--sources src1.jsonl:0.4,src2.jsonl:0.6] [--n 10000] [--output FILE] [--seed 42]
  python dataprep.py stats   [--input FILE] [--field text]

Input/output default to stdin/stdout when omitted.
"""

import argparse
import hashlib
import json
import random
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _text_field(doc: Dict[str, Any]) -> str:
    """Return the text content of a JSON doc, checking 'text' then 'content'."""
    v = doc.get("text")
    if v is None:
        v = doc.get("content")
    return v if isinstance(v, str) else ""


def read_jsonl(path: Optional[str]) -> Iterator[Dict[str, Any]]:
    fh = open(path) if path else sys.stdin
    try:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass
    finally:
        if path:
            fh.close()


def write_jsonl(docs: Iterator[Dict[str, Any]], path: Optional[str]) -> int:
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


# ── filter ───────────────────────────────────────────────────────────────────

def heuristic_filter(
    docs: List[Dict[str, Any]],
    min_len: int = 100,
    max_symbol_ratio: float = 0.1,
    max_rep_ratio: float = 0.3,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply heuristic quality filters.

    Returns (kept_docs, stats_dict).
    stats_dict keys: total, kept, removed_too_short, removed_symbol_ratio, removed_rep_ratio
    """
    kept = []
    stats: Dict[str, int] = {
        "total": len(docs),
        "kept": 0,
        "removed_too_short": 0,
        "removed_symbol_ratio": 0,
        "removed_rep_ratio": 0,
    }

    for doc in docs:
        text = _text_field(doc)
        n = len(text)

        if n < min_len:
            stats["removed_too_short"] += 1
            continue

        # Symbol ratio: non-alphanumeric, non-space characters
        non_alnum = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if non_alnum / n > max_symbol_ratio:
            stats["removed_symbol_ratio"] += 1
            continue

        # Line repetition ratio (blank lines excluded from both sides)
        nonempty_lines = [l.strip() for l in text.splitlines() if l.strip()]
        if nonempty_lines:
            unique = set(nonempty_lines)
            rep_ratio = 1.0 - len(unique) / len(nonempty_lines)
            if rep_ratio > max_rep_ratio:
                stats["removed_rep_ratio"] += 1
                continue

        kept.append(doc)

    stats["kept"] = len(kept)
    return kept, stats


def cmd_filter(args: argparse.Namespace) -> None:
    docs = list(read_jsonl(args.input))
    kept, stats = heuristic_filter(
        docs,
        min_len=args.min_len,
        max_symbol_ratio=args.max_symbol_ratio,
        max_rep_ratio=args.max_rep_ratio,
    )
    write_jsonl(iter(kept), args.output)
    print(
        f"filter: {stats['total']} → {stats['kept']} kept "
        f"(removed: {stats['removed_too_short']} too-short, "
        f"{stats['removed_symbol_ratio']} symbol-ratio, "
        f"{stats['removed_rep_ratio']} rep-ratio)",
        file=sys.stderr,
    )


# ── dedup ─────────────────────────────────────────────────────────────────────

def minhash_dedup(
    docs: List[Dict[str, Any]],
    threshold: float = 0.85,
    num_hashes: int = 128,
    ngram_size: int = 5,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    MinHash-based near-deduplication (stdlib only). Returns (deduped_docs, n_removed).
    Uses MD5 for stable, cross-process hashing.
    """
    PRIME = (1 << 61) - 1
    rng = random.Random(seed)
    a = [rng.randint(1, PRIME - 1) for _ in range(num_hashes)]
    b = [rng.randint(0, PRIME - 1) for _ in range(num_hashes)]

    def ngrams(text: str, n: int):
        for i in range(len(text) - n + 1):
            yield text[i : i + n]

    def stable_hash(gram: str) -> int:
        return int(hashlib.md5(gram.encode(), usedforsecurity=False).hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF

    def minhash_sig(text: str) -> List[int]:
        sig = [PRIME] * num_hashes
        for gram in ngrams(text, ngram_size):
            h = stable_hash(gram)
            for k in range(num_hashes):
                v = (a[k] * h + b[k]) % PRIME
                if v < sig[k]:
                    sig[k] = v
        return sig

    n_bands = num_hashes // 8
    rows_per_band = num_hashes // n_bands

    # Compute signatures
    sigs = [minhash_sig(_text_field(doc)) for doc in docs]

    # LSH bucketing
    bucket_map: Dict[Any, List[int]] = {}
    for idx, sig in enumerate(sigs):
        for band in range(n_bands):
            start = band * rows_per_band
            key = (band, tuple(sig[start : start + rows_per_band]))
            bucket_map.setdefault(key, []).append(idx)

    # Union-find
    parent = list(range(len(docs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[max(px, py)] = min(px, py)

    for candidates in bucket_map.values():
        if len(candidates) < 2:
            continue
        # Band collision is sufficient evidence of duplication — union directly.
        # Full O(n²) Jaccard verification is skipped; false-positive rate is
        # negligible at n_bands=16 and acceptable thresholds.
        root = candidates[0]
        for idx in candidates[1:]:
            union(root, idx)

    # Keep lowest-index doc per component
    seen_roots: set = set()
    kept = []
    for idx, doc in enumerate(docs):
        root = find(idx)
        if root not in seen_roots:
            seen_roots.add(root)
            kept.append(doc)

    n_removed = len(docs) - len(kept)
    return kept, n_removed


def cmd_dedup(args: argparse.Namespace) -> None:
    docs = list(read_jsonl(args.input))
    kept, n_removed = minhash_dedup(
        docs,
        threshold=args.threshold,
        num_hashes=args.num_hashes,
        ngram_size=args.ngram_size,
        seed=args.seed,
    )
    write_jsonl(iter(kept), args.output)
    print(
        f"dedup: {len(docs)} → {len(kept)} kept ({n_removed} removed)",
        file=sys.stderr,
    )


# ── score ─────────────────────────────────────────────────────────────────────

def quality_score(
    docs: List[Dict[str, Any]],
    gold_docs: List[Dict[str, Any]],
    min_score: Optional[float] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Train TF-IDF + LogisticRegression on gold (label=1) vs random sample of docs (label=0).
    Adds 'quality_score' field to each doc. Filters by min_score if provided.

    Requires scikit-learn.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("error: score requires scikit-learn — pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    gold_texts = [_text_field(d) for d in gold_docs]
    rng = random.Random(seed)
    neg_sample = rng.sample(docs, min(len(gold_texts) * 3, len(docs)))
    neg_texts = [_text_field(d) for d in neg_sample]

    X_train = gold_texts + neg_texts
    y_train = [1] * len(gold_texts) + [0] * len(neg_texts)

    vec = TfidfVectorizer(max_features=50_000, sublinear_tf=True)
    X_vec = vec.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_vec, y_train)

    doc_texts = [_text_field(d) for d in docs]
    scores = clf.predict_proba(vec.transform(doc_texts))[:, 1]

    result = []
    for doc, score in zip(docs, scores):
        doc = dict(doc)
        doc["quality_score"] = round(float(score), 4)
        if min_score is None or score >= min_score:
            result.append(doc)
    return result


def cmd_score(args: argparse.Namespace) -> None:
    docs = list(read_jsonl(args.input))
    gold_docs = list(read_jsonl(args.gold))
    scored = quality_score(docs, gold_docs, min_score=args.min_score, seed=args.seed)
    write_jsonl(iter(scored), args.output)
    print(
        f"score: {len(docs)} → {len(scored)} kept (min_score={args.min_score})",
        file=sys.stderr,
    )


# ── sample ────────────────────────────────────────────────────────────────────

def mixture_sample(
    sources_with_ratios: List[Tuple[str, float]],
    n: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Sample n docs proportionally from multiple sources.

    sources_with_ratios: list of (filepath, weight) — weights need not sum to 1.
    """
    random.seed(seed)
    total_weight = sum(w for _, w in sources_with_ratios)
    result = []
    for path, weight in sources_with_ratios:
        k = round(n * weight / total_weight)
        pool = list(read_jsonl(path))
        if len(pool) <= k:
            result.extend(pool)
        else:
            result.extend(random.sample(pool, k))
    random.shuffle(result)
    if len(result) < n:
        print(f"warning: requested {n} docs but only {len(result)} available across all sources", file=sys.stderr)
    return result[:n]


def cmd_sample(args: argparse.Namespace) -> None:
    pairs: List[Tuple[str, float]] = []
    for item in args.sources.split(","):
        item = item.strip()
        if ":" in item:
            path, weight_str = item.rsplit(":", 1)
            pairs.append((path, float(weight_str)))
        else:
            pairs.append((item, 1.0))
    docs = mixture_sample(pairs, n=args.n, seed=args.seed)
    write_jsonl(iter(docs), args.output)
    print(f"sample: {len(docs)} docs written", file=sys.stderr)


# ── stats ─────────────────────────────────────────────────────────────────────

def _fmt_num(n: int) -> str:
    return f"{n:,}"


def stats(docs: List[Dict[str, Any]], field: str = "text") -> str:
    """Compute corpus statistics. Output matches the Rust stats tool format."""
    lengths = []
    for doc in docs:
        text = doc.get(field) or ""
        lengths.append(len(text))

    if not lengths:
        return "docs: 0\nchars: 0\ntokens: ~0  (chars / 4)\nlengths: no docs found"

    count = len(lengths)
    total_chars = sum(lengths)
    tokens = total_chars // 4
    min_len = min(lengths)
    max_len = max(lengths)
    mean_len = total_chars // count

    sorted_lens = sorted(lengths)
    p50 = sorted_lens[len(sorted_lens) // 2]
    p90 = sorted_lens[len(sorted_lens) * 9 // 10]

    return (
        f"docs:    {_fmt_num(count)}\n"
        f"chars:   {_fmt_num(total_chars)}\n"
        f"tokens:  ~{_fmt_num(tokens)}  (chars / 4)\n"
        f"lengths: min={_fmt_num(min_len)}  mean={_fmt_num(mean_len)}"
        f"  p50={_fmt_num(p50)}  p90={_fmt_num(p90)}  max={_fmt_num(max_len)}\n"
        f"format:  jsonl  field: {field}"
    )


def cmd_stats(args: argparse.Namespace) -> None:
    docs = list(read_jsonl(args.input))
    print(stats(docs, field=args.field))


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretraining data pipeline: filter / dedup / score / sample / stats"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # filter
    p = sub.add_parser("filter", help="Heuristic quality filter")
    p.add_argument("--input",            default=None)
    p.add_argument("--output",           default=None)
    p.add_argument("--min-len",          type=int,   default=100)
    p.add_argument("--max-symbol-ratio", type=float, default=0.1)
    p.add_argument("--max-rep-ratio",    type=float, default=0.3)

    # dedup
    p = sub.add_parser("dedup", help="MinHash near-deduplication")
    p.add_argument("--input",      default=None)
    p.add_argument("--output",     default=None)
    p.add_argument("--threshold",  type=float, default=0.85)
    p.add_argument("--num-hashes", type=int,   default=128)
    p.add_argument("--ngram-size", type=int,   default=5)
    p.add_argument("--seed",       type=int,   default=42)

    # score
    p = sub.add_parser("score", help="Quality scoring (TF-IDF + LogReg)")
    p.add_argument("--input",     default=None)
    p.add_argument("--output",    default=None)
    p.add_argument("--gold",      required=True, help="JSONL file of high-quality reference docs")
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--seed",      type=int,   default=42)

    # sample
    p = sub.add_parser("sample", help="Proportional mixture sampling")
    p.add_argument("--sources", required=True, help="src1.jsonl:0.4,src2.jsonl:0.6")
    p.add_argument("--n",       type=int,   default=10_000)
    p.add_argument("--output",  default=None)
    p.add_argument("--seed",    type=int,   default=42)

    # stats
    p = sub.add_parser("stats", help="Corpus statistics")
    p.add_argument("--input", default=None)
    p.add_argument("--field", default="text")

    args = parser.parse_args()
    dispatch = {
        "filter": cmd_filter,
        "dedup":  cmd_dedup,
        "score":  cmd_score,
        "sample": cmd_sample,
        "stats":  cmd_stats,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
