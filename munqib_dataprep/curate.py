"""Config-driven curation pipeline."""

from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from .builders import BuildError, canonical_document, iter_source_documents
from .legacy import read_jsonl, write_jsonl


CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
WORD_RE = re.compile(r"[A-Za-z']+")
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]*)?(?:\(?\d{3}\)?[-.\s]*)\d{3}[-.\s]*\d{4}\b")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_RE = re.compile(r"\bhttps?://[^\s<>()]+", re.IGNORECASE)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?|\d{1,2}/\d{1,2}/\d{2,4})\b"
)
ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][a-z0-9]+\s){0,5}"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b\.?",
)
PERSON_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Professor)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")
LOCATION_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*[A-Z]{2}\b")

COMMON_ENGLISH_WORDS = {
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "i",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
    "they",
    "we",
    "say",
    "her",
    "she",
    "or",
    "an",
    "will",
    "my",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "so",
    "up",
    "out",
    "if",
    "about",
    "who",
    "get",
    "which",
    "go",
    "me",
    "when",
    "make",
    "can",
    "like",
    "time",
    "no",
    "just",
    "him",
    "know",
    "take",
    "people",
    "into",
    "year",
    "your",
    "good",
    "some",
    "could",
    "them",
    "see",
    "other",
    "than",
    "then",
    "now",
    "look",
    "only",
    "come",
    "its",
    "over",
    "think",
    "also",
    "back",
    "after",
    "use",
    "two",
    "how",
    "our",
    "work",
    "first",
    "well",
    "way",
    "even",
    "new",
    "want",
}

TOXIC_TERMS = {
    "idiot",
    "stupid",
    "moron",
    "dumb",
    "hate",
    "kill",
    "racist",
    "sexist",
    "trash",
    "worthless",
    "loser",
    "bastard",
    "fool",
}

PII_PATTERNS = {
    "EMAIL_ADDRESS": EMAIL_RE,
    "PHONE_NUMBER": PHONE_RE,
    "ADDRESS": ADDRESS_RE,
    "IP_ADDRESS": IP_RE,
    "URL": URL_RE,
    "US_SSN": SSN_RE,
    "DATE_TIME": DATE_RE,
    "PERSON": PERSON_RE,
    "LOCATION": LOCATION_RE,
}


class CurateError(RuntimeError):
    """Raised when curation fails."""


def canonicalize_existing_doc(doc: Dict[str, Any], source_hint: Optional[str] = None) -> Dict[str, Any]:
    text = str(doc.get("text") or doc.get("content") or "")
    source = str(doc.get("source") or source_hint or "jsonl")
    dataset = str(doc.get("dataset") or source)
    split = str(doc.get("split") or "train")
    subset = doc.get("subset")
    subset_name = str(subset) if subset is not None else "default"
    upstream_id = doc.get("upstream_id")
    source_url = doc.get("source_url")
    metadata = doc.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    reserved = {
        "id",
        "text",
        "content",
        "source",
        "dataset",
        "split",
        "subset",
        "upstream_id",
        "source_url",
        "metadata",
        "lang",
        "lang_score",
        "quality",
        "domain",
        "toxicity",
        "pii",
        "dedup",
        "kept",
        "drop_reasons",
    }
    extras = {key: value for key, value in doc.items() if key not in reserved}
    if extras:
        metadata = dict(metadata)
        metadata["extra_fields"] = extras
    canonical = canonical_document(
        source=source,
        dataset=dataset,
        split=split,
        subset=None if subset_name == "default" else subset_name,
        text=text,
        upstream_id=str(upstream_id) if upstream_id is not None else None,
        source_url=str(source_url) if source_url is not None else None,
        metadata=metadata,
    )
    if doc.get("id"):
        canonical["id"] = str(doc["id"])
    return canonical


def iter_config_sources(config: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    for source_cfg in config["sources"]:
        if "path" in source_cfg:
            path = source_cfg["path"]
            source_hint = source_cfg.get("name") or Path(path).stem
            for doc in read_jsonl(path):
                yield canonicalize_existing_doc(doc, source_hint=source_hint)
            continue
        try:
            yield from iter_source_documents(source_cfg)
        except BuildError as exc:
            raise CurateError(str(exc)) from exc


def clean_text(text: str, config: Dict[str, Any]) -> str:
    cleaned = text
    if config.get("fix_unicode", True):
        cleaned = unicodedata.normalize("NFKC", cleaned)
    if config.get("strip_control_chars", True):
        cleaned = CONTROL_CHARS.sub("", cleaned)
    if config.get("normalize_newlines", True):
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    if config.get("collapse_whitespace", True):
        cleaned = re.sub(r"[ \t\f\v]+", " ", cleaned)
    if config.get("collapse_blank_lines", True):
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def detect_language(text: str, config: Dict[str, Any]) -> Tuple[str, float, str]:
    if not config.get("enabled", True):
        return "unknown", 0.0, "disabled"

    backend = config.get("backend", "fasttext")
    model_path = config.get("model_path")
    if backend == "fasttext" and model_path:
        try:
            import fasttext  # type: ignore
        except ImportError:
            if not config.get("allow_fallback", True):
                raise CurateError("language backend fasttext requested but the fasttext package is not installed")
        else:
            model = fasttext.load_model(model_path)
            labels, scores = model.predict(text.replace("\n", " "), k=1)
            lang = labels[0].replace("__label__", "")
            return lang, float(scores[0]), "fasttext"

    words = [word.lower() for word in WORD_RE.findall(text[:4000])]
    if not words:
        return "unknown", 0.0, "heuristic"
    english_hits = sum(1 for word in words if word in COMMON_ENGLISH_WORDS)
    ascii_ratio = sum(1 for char in text if ord(char) < 128) / max(1, len(text))
    english_ratio = english_hits / max(1, len(words))
    score = min(0.99, (ascii_ratio * 0.55) + (min(1.0, english_ratio * 6.0) * 0.45))
    lang = "en" if score >= 0.65 else "unknown"
    return lang, round(score, 4), "heuristic"


def repeated_line_ratio(text: str) -> float:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return 1.0 - (len(set(lines)) / len(lines))


def repeated_ngram_ratio(text: str, n: int = 3) -> float:
    words = [word.lower() for word in WORD_RE.findall(text)]
    if len(words) < n:
        return 0.0
    grams = [" ".join(words[idx : idx + n]) for idx in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def heuristic_quality(text: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    words = [word.lower() for word in WORD_RE.findall(text)]
    total_chars = len(text)
    alpha_chars = sum(1 for char in text if char.isalpha())
    non_alnum = sum(1 for char in text if not char.isalnum() and not char.isspace())
    english_hits = sum(1 for word in words if word in COMMON_ENGLISH_WORDS)

    metrics = {
        "chars": total_chars,
        "word_count": len(words),
        "alpha_ratio": round(alpha_chars / max(1, total_chars), 4),
        "symbol_ratio": round(non_alnum / max(1, total_chars), 4),
        "line_repetition_ratio": round(repeated_line_ratio(text), 4),
        "ngram_repetition_ratio": round(repeated_ngram_ratio(text), 4),
        "english_word_ratio": round(english_hits / max(1, len(words)), 4),
    }

    failures: List[str] = []
    if total_chars < config["min_chars"]:
        failures.append("heuristic:min_chars")
    if metrics["symbol_ratio"] > config["max_symbol_ratio"]:
        failures.append("heuristic:symbol_ratio")
    if metrics["line_repetition_ratio"] > config["max_line_repetition_ratio"]:
        failures.append("heuristic:line_repetition")
    if metrics["ngram_repetition_ratio"] > config["max_ngram_repetition_ratio"]:
        failures.append("heuristic:ngram_repetition")
    if metrics["alpha_ratio"] < config["min_alpha_ratio"]:
        failures.append("heuristic:alpha_ratio")
    if metrics["english_word_ratio"] < config["min_english_word_ratio"]:
        failures.append("heuristic:english_word_ratio")

    penalty = 0.0
    if total_chars < config["min_chars"]:
        penalty += (config["min_chars"] - total_chars) / max(1, config["min_chars"])
    penalty += max(0.0, metrics["symbol_ratio"] - config["max_symbol_ratio"]) * 2.0
    penalty += max(0.0, metrics["line_repetition_ratio"] - config["max_line_repetition_ratio"]) * 1.5
    penalty += max(0.0, metrics["ngram_repetition_ratio"] - config["max_ngram_repetition_ratio"]) * 1.5
    penalty += max(0.0, config["min_alpha_ratio"] - metrics["alpha_ratio"]) * 1.2
    penalty += max(0.0, config["min_english_word_ratio"] - metrics["english_word_ratio"]) * 8.0
    score = max(0.0, 1.0 - min(1.0, penalty))

    return {
        "backend": "heuristic",
        "score": round(score, 4),
        "label": "high" if score >= 0.8 else "medium" if score >= 0.55 else "low",
        "metrics": metrics,
    }, failures


def quality_classifier(doc: Dict[str, Any]) -> Dict[str, Any]:
    quality = doc.get("quality") or {}
    score = float(quality.get("score", 0.0))
    return {
        "backend": quality.get("backend", "heuristic"),
        "label": "high" if score >= 0.8 else "medium" if score >= 0.55 else "low",
        "score": round(score, 4),
        "metrics": quality.get("metrics"),
    }


def domain_classifier(doc: Dict[str, Any]) -> Dict[str, Any]:
    text = doc["_working_text"]
    lower = text.lower()
    scores = {
        "education": 0.0,
        "documentation": 0.0,
        "code": 0.0,
        "forum": 0.0,
        "qa": 0.0,
        "generic_web": 0.1,
    }

    source_url = doc.get("source_url")
    host = urlparse(source_url).netloc.lower() if source_url else ""
    if any(token in host for token in (".edu", "khanacademy", "coursera", "edx", "wikipedia")):
        scores["education"] += 0.8
    if any(token in host for token in ("github", "gitlab", "readthedocs", "docs.")):
        scores["documentation"] += 0.8
        scores["code"] += 0.2
    if any(token in host for token in ("reddit", "forum", "discuss", "stackexchange", "stackoverflow")):
        scores["forum"] += 0.8
        scores["qa"] += 0.4

    if "```" in text or re.search(r"\b(def|class|import|function|public static)\b", text):
        scores["code"] += 0.7
    if re.search(r"\b(question|answer|q:|a:|faq)\b", lower):
        scores["qa"] += 0.5
    if re.search(r"\b(chapter|lesson|exercise|learning objectives|curriculum|student)\b", lower):
        scores["education"] += 0.5
    if re.search(r"\b(installation|configuration|usage|api|reference)\b", lower):
        scores["documentation"] += 0.4

    label, score = max(scores.items(), key=lambda item: item[1])
    return {"backend": "heuristic", "label": label, "score": round(min(score, 1.0), 4)}


def toxicity_classifier(doc: Dict[str, Any]) -> Dict[str, Any]:
    words = [word.lower() for word in WORD_RE.findall(doc["_working_text"])]
    if not words:
        score = 0.0
    else:
        toxic_hits = sum(1 for word in words if word in TOXIC_TERMS)
        score = min(1.0, toxic_hits / len(words) * 12.0)
    if score >= 0.45:
        label = "toxic"
    elif score >= 0.1:
        label = "possibly_toxic"
    else:
        label = "non_toxic"
    return {"backend": "heuristic", "label": label, "score": round(score, 4)}


def apply_classifier_gate(name: str, result: Dict[str, Any], config: Dict[str, Any], drop_reasons: List[str]) -> None:
    if not config.get("enabled", False) or not config.get("gate", False):
        return
    label = result.get("label")
    score = result.get("score")
    allowed = config.get("allowed_labels")
    blocked = config.get("blocked_labels")
    min_score = config.get("min_score")
    max_score = config.get("max_score")

    if allowed and label not in allowed:
        drop_reasons.append(f"classifier:{name}:label:{label}")
    if blocked and label in blocked:
        drop_reasons.append(f"classifier:{name}:label:{label}")
    if min_score is not None and score is not None and float(score) < float(min_score):
        drop_reasons.append(f"classifier:{name}:min_score")
    if max_score is not None and score is not None and float(score) > float(max_score):
        drop_reasons.append(f"classifier:{name}:max_score")


def redact_pii_regex(text: str, config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    entities = config.get("entities", [])
    matches: List[Tuple[int, int, str]] = []
    for entity in entities:
        pattern = PII_PATTERNS.get(entity)
        if pattern is None:
            continue
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), entity))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    accepted: List[Tuple[int, int, str]] = []
    cursor = -1
    for start, end, entity in matches:
        if start < cursor:
            continue
        accepted.append((start, end, entity))
        cursor = end

    if not accepted:
        return text, {"backend": "regex", "entities": {}, "matches": 0}

    counts: Dict[str, int] = {}
    parts: List[str] = []
    last_end = 0
    for start, end, entity in accepted:
        parts.append(text[last_end:start])
        parts.append(f"{{{{{entity}}}}}" if config.get("replace_with_type", True) else "{{PII}}")
        counts[entity] = counts.get(entity, 0) + 1
        last_end = end
    parts.append(text[last_end:])
    redacted = "".join(parts)
    return redacted, {"backend": "regex", "entities": counts, "matches": sum(counts.values())}


def redact_pii_presidio(text: str, config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore
        from presidio_anonymizer import AnonymizerEngine  # type: ignore
        from presidio_anonymizer.entities import OperatorConfig  # type: ignore
    except ImportError as exc:
        raise CurateError(
            "privacy backend presidio requested but presidio-analyzer/presidio-anonymizer are not installed"
        ) from exc

    analyzer = AnalyzerEngine()
    entities = config.get("entities") or None
    analyzer_results = analyzer.analyze(text=text, entities=entities, language=str(config.get("language", "en")))

    replacement = "{{PII}}"
    operators: Dict[str, Any] = {"DEFAULT": OperatorConfig("replace", {"new_value": replacement})}
    if config.get("replace_with_type", True):
        entity_types = {result.entity_type for result in analyzer_results}
        operators = {
            entity_type: OperatorConfig("replace", {"new_value": f"{{{{{entity_type}}}}}"})
            for entity_type in entity_types
        }
        operators["DEFAULT"] = OperatorConfig("replace", {"new_value": replacement})

    anonymizer = AnonymizerEngine()
    redaction = anonymizer.anonymize(text=text, analyzer_results=analyzer_results, operators=operators)

    counts: Dict[str, int] = {}
    for result in analyzer_results:
        entity_type = str(result.entity_type)
        counts[entity_type] = counts.get(entity_type, 0) + 1

    return redaction.text, {"backend": "presidio", "entities": counts, "matches": sum(counts.values())}


def apply_pii_redaction(text: str, config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if not config.get("enabled", True):
        return text, {"backend": "disabled", "entities": {}, "matches": 0}
    backend = str(config.get("backend", "regex")).lower()
    if backend == "regex":
        return redact_pii_regex(text, config)
    if backend == "presidio":
        return redact_pii_presidio(text, config)
    raise CurateError(f"unsupported privacy backend: {backend}")


def apply_exact_dedup(docs: Sequence[Dict[str, Any]]) -> None:
    seen: Dict[str, str] = {}
    for doc in docs:
        digest = hashlib.md5(doc["_working_text"].encode("utf-8"), usedforsecurity=False).hexdigest()
        exact = {"enabled": True, "hash": digest}
        dedup = doc.setdefault("dedup", {})
        dedup["exact"] = exact
        if digest not in seen:
            seen[digest] = doc["id"]
            exact["canonical_id"] = doc["id"]
            exact["duplicate"] = False
            continue
        exact["canonical_id"] = seen[digest]
        exact["duplicate"] = True
        exact["duplicate_of"] = seen[digest]
        doc["drop_reasons"].append("dedup:exact")


def fuzzy_fingerprint(text: str, num_hashes: int, ngram_size: int, seed: int) -> List[int]:
    primes = (1 << 61) - 1
    import random

    rng = random.Random(seed)
    coeff_a = [rng.randint(1, primes - 1) for _ in range(num_hashes)]
    coeff_b = [rng.randint(0, primes - 1) for _ in range(num_hashes)]
    words = text.split()
    if len(words) < ngram_size:
        return [0] * num_hashes
    grams = [" ".join(words[idx : idx + ngram_size]) for idx in range(len(words) - ngram_size + 1)]
    signature = [primes] * num_hashes
    for gram in grams:
        hashed = int(hashlib.md5(gram.encode("utf-8"), usedforsecurity=False).hexdigest(), 16)
        for idx in range(num_hashes):
            value = (coeff_a[idx] * hashed + coeff_b[idx]) % primes
            if value < signature[idx]:
                signature[idx] = value
    return signature


def apply_fuzzy_dedup(docs: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> None:
    if not config.get("enabled", False):
        return
    if config.get("require_linux", True) and sys.platform != "linux":
        raise CurateError("fuzzy dedup is enabled but requires Linux in this configuration")

    threshold = float(config.get("threshold", 0.9))
    num_hashes = int(config.get("num_hashes", 128))
    ngram_size = int(config.get("ngram_size", 5))
    seed = int(config.get("seed", 42))

    fingerprints = [fuzzy_fingerprint(doc["_working_text"], num_hashes, ngram_size, seed) for doc in docs]
    anchors: Dict[str, int] = {}
    for idx, doc in enumerate(docs):
        if "dedup" not in doc:
            doc["dedup"] = {}
        doc["dedup"]["fuzzy"] = {"enabled": True, "duplicate": False}
        if "dedup:exact" in doc["drop_reasons"]:
            continue
        fp = fingerprints[idx]
        bucket = ":".join(str(item) for item in fp[:16])
        if bucket not in anchors:
            anchors[bucket] = idx
            doc["dedup"]["fuzzy"]["canonical_id"] = doc["id"]
            continue
        anchor_idx = anchors[bucket]
        agreement = sum(1 for left, right in zip(fp, fingerprints[anchor_idx]) if left == right) / max(1, len(fp))
        doc["dedup"]["fuzzy"]["score"] = round(agreement, 4)
        if agreement >= threshold:
            doc["dedup"]["fuzzy"]["duplicate"] = True
            doc["dedup"]["fuzzy"]["duplicate_of"] = docs[anchor_idx]["id"]
            doc["drop_reasons"].append("dedup:fuzzy")
        else:
            anchors[bucket] = idx
            doc["dedup"]["fuzzy"]["canonical_id"] = doc["id"]


def prepare_audit_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": doc["id"],
        "text": doc["text"],
        "source": doc["source"],
        "dataset": doc["dataset"],
        "split": doc["split"],
        "subset": doc["subset"],
        "upstream_id": doc.get("upstream_id"),
        "source_url": doc.get("source_url"),
        "metadata": doc.get("metadata", {}),
        "lang": doc.get("lang"),
        "lang_score": doc.get("lang_score"),
        "quality": doc.get("quality"),
        "domain": doc.get("domain"),
        "toxicity": doc.get("toxicity"),
        "pii": doc.get("pii"),
        "dedup": doc.get("dedup", {}),
        "kept": bool(doc.get("kept")),
        "drop_reasons": list(doc.get("drop_reasons", [])),
    }


def prepare_final_doc(doc: Dict[str, Any], include_metadata: bool = False) -> Dict[str, Any]:
    result = {
        "id": doc["id"],
        "text": doc["text"],
        "source": doc["source"],
        "dataset": doc["dataset"],
        "lang": doc.get("lang"),
    }
    if include_metadata:
        result["metadata"] = doc.get("metadata", {})
    return result


def curate_documents(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    docs = list(iter_config_sources(config))
    for doc in docs:
        doc["drop_reasons"] = []
        cleaned = clean_text(doc["text"], config["cleaning"])
        doc["_working_text"] = cleaned
        doc["metadata"] = dict(doc.get("metadata", {}))
        doc["metadata"]["clean_length"] = len(cleaned)

        language_config = config["language"]
        lang, lang_score, lang_backend = detect_language(cleaned, language_config)
        doc["lang"] = lang
        doc["lang_score"] = lang_score
        doc["metadata"]["language_backend"] = lang_backend
        if language_config.get("enabled", True):
            if lang not in language_config["allowed"]:
                doc["drop_reasons"].append(f"language:label:{lang}")
            if float(lang_score) < float(language_config["min_score"]):
                doc["drop_reasons"].append("language:min_score")

        if config["heuristics"].get("enabled", True):
            quality, heuristic_reasons = heuristic_quality(cleaned, config["heuristics"])
            doc["quality"] = quality
            doc["drop_reasons"].extend(heuristic_reasons)
        else:
            doc["quality"] = {"backend": "disabled", "label": "unknown", "score": 0.0, "metrics": {}}

    if config["dedup"]["exact"].get("enabled", True):
        apply_exact_dedup(docs)
    apply_fuzzy_dedup(docs, config["dedup"]["fuzzy"])

    classifiers = config.get("classifiers", {})
    for doc in docs:
        doc["quality"] = quality_classifier(doc) if classifiers["quality"].get("enabled", False) else doc.get("quality")
        doc["domain"] = domain_classifier(doc) if classifiers["domain"].get("enabled", False) else None
        doc["toxicity"] = toxicity_classifier(doc) if classifiers["toxicity"].get("enabled", False) else None
        if doc.get("quality"):
            apply_classifier_gate("quality", doc["quality"], classifiers["quality"], doc["drop_reasons"])
        if doc.get("domain"):
            apply_classifier_gate("domain", doc["domain"], classifiers["domain"], doc["drop_reasons"])
        if doc.get("toxicity"):
            apply_classifier_gate("toxicity", doc["toxicity"], classifiers["toxicity"], doc["drop_reasons"])

        redacted_text, pii = apply_pii_redaction(doc["_working_text"], config["privacy"])
        doc["pii"] = pii
        doc["text"] = redacted_text
        doc["kept"] = not doc["drop_reasons"]

    audit_docs = [prepare_audit_doc(doc) for doc in docs]
    final_docs = [
        prepare_final_doc(doc, include_metadata=config["outputs"].get("include_final_metadata", False))
        for doc in docs
        if doc.get("kept")
    ]
    summary = {
        "input_docs": len(docs),
        "audit_docs": len(audit_docs),
        "final_docs": len(final_docs),
        "dropped_docs": len(docs) - len(final_docs),
    }
    return audit_docs, final_docs, summary


def run_curate_command(args: Any, config: Dict[str, Any]) -> None:
    summary = curate_to_outputs(config, args.audit_output, args.final_output)
    print(
        f"curate: {summary['input_docs']} input -> {summary['final_docs']} final "
        f"({summary['dropped_docs']} dropped)"
    )


def curate_to_outputs(config: Dict[str, Any], audit_output: str, final_output: str) -> Dict[str, int]:
    """Run curation and write the audit and final corpora to disk."""
    audit_docs, final_docs, summary = curate_documents(config)
    write_jsonl(audit_docs, audit_output)
    write_jsonl(final_docs, final_output)
    return summary
