"""Build a BM25 index from the supplied textbook corpus (corpus.jsonl)"""

import argparse
import hashlib
import json
import pickle
import re
import sys
import time
import threading
from pathlib import Path

import psutil
from rank_bm25 import BM25Okapi


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_no}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, line {line_no}."
                )
            rows.append(row)
    return rows


MATH_SYMBOLS = (
    "+\\-−*/=<>≤≥≠≈∂∆△∇∑∏√∞∈∉⊂⊃⊆⊇∩∪∅"
    "∧∨∀∃→↔ℂℝℤℕ∫∮×·⋅±"
)

WORD_PATTERN = re.compile(r"[^\W]+(?:[_.^][^\W]+)*")
MATH_PATTERN = re.compile(
    r"[^\W]+(?:[_.^][^\W]+)*|[" + re.escape(MATH_SYMBOLS) + r"]"
)

DIGIT_REF = re.compile(r"\d+(?:\.\d+)+")

SECTION_WEIGHT = 2


def get_token_settings(config):
    raw = config.get("tokenization", {})
    settings = {
        "lowercase": raw.get("lowercase", True),
        "include_math_symbols": raw.get("include_math_symbols", True),
        "min_token_length": raw.get("min_token_length", 1),
    }
    if not isinstance(settings["lowercase"], bool):
        raise ValueError("tokenization.lowercase must be true or false.")
    if not isinstance(settings["include_math_symbols"], bool):
        raise ValueError("tokenization.include_math_symbols must be true or false.")
    if (
        not isinstance(settings["min_token_length"], int)
        or isinstance(settings["min_token_length"], bool)
        or settings["min_token_length"] < 1
    ):
        raise ValueError("tokenization.min_token_length must be an integer >= 1.")
    return settings


def tokenize(text, settings):
    """Tokenize preserving digit refs (2.1), identifiers (u_x, x^2), and
    math symbols when enabled.
    """
    digit_refs = DIGIT_REF.findall(text)
    text = DIGIT_REF.sub(" ", text)

    pattern = MATH_PATTERN if settings["include_math_symbols"] else WORD_PATTERN
    tokens = digit_refs + pattern.findall(text)

    if settings["lowercase"]:
        tokens = [token.casefold() for token in tokens]

    minimum = settings["min_token_length"]
    return [token for token in tokens if len(token) >= minimum]


def normalize_documents(rows):
    """Keep one index document per original corpus block and preserve its ID."""
    documents = []
    seen_ids = set()

    required = ("block_id", "page", "section", "text")
    for line_no, row in enumerate(rows, start=1):
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(
                f"Corpus row {line_no} is missing required fields: {missing}"
            )

        block_id = row["block_id"]
        if not isinstance(block_id, str) or not block_id.strip():
            raise ValueError(
                f"Corpus row {line_no}: block_id must be a non-empty string."
            )
        if block_id in seen_ids:
            raise ValueError(f"Duplicate block_id in corpus: {block_id}")
        seen_ids.add(block_id)

        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(
                f"Corpus row {line_no} ({block_id}) has empty text."
            )

        documents.append(
            {
                "block_id": block_id,
                "page": row["page"],
                "section": row["section"],
                "text": row["text"],
            }
        )

    if not documents:
        raise ValueError("The corpus contains no documents.")
    return documents


def index_text(doc):
    """Text fed to the tokenizer. Section is weighted via repetition; the
    stored `text` in documents.jsonl is left clean.
    """
    section = doc.get("section") or ""
    if not section:
        return doc["text"]
    return (section + " ") * SECTION_WEIGHT + doc["text"]


def read_retrieval_config(config):
    retrieval_config = config.get("retrieval", {})

    method = retrieval_config.get("method", "bm25")
    if method != "bm25":
        raise ValueError(f"Unsupported retrieval method: {method}")

    k1 = float(retrieval_config.get("k1", 1.5))
    b = float(retrieval_config.get("b", 0.75))

    if k1 <= 0:
        raise ValueError("retrieval.k1 must be greater than 0.")
    if not 0 <= b <= 1:
        raise ValueError("retrieval.b must be between 0 and 1.")

    return method, k1, b


def main():
    parser = argparse.ArgumentParser(
        description="Build a BM25 index from corpus.jsonl."
    )
    parser.add_argument(
        "--corpus", default="data/corpus.jsonl", help="Input corpus JSONL."
    )
    parser.add_argument(
        "--output", default="index", help="Directory for index files."
    )
    parser.add_argument(
        "--config", default="config.json", help="JSON configuration."
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    config_path = Path(args.config)
    output_dir = Path(args.output)

    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Config validation before touching the corpus — a bad k1/b/method
    # fails immediately rather than after a full tokenize pass.
    config = read_json(config_path)
    token_settings = get_token_settings(config)
    method, k1, b = read_retrieval_config(config)

    documents = normalize_documents(read_jsonl(corpus_path))

    build_start_time = time.perf_counter()

    process = psutil.Process()
    peak_ram_bytes = [process.memory_info().rss]
    stop_monitor = threading.Event()

    def monitor_ram():
        while not stop_monitor.is_set():
            try:
                current_ram = process.memory_info().rss
                peak_ram_bytes[0] = max(peak_ram_bytes[0], current_ram)
            except psutil.Error:
                pass
            stop_monitor.wait(0.01)

    ram_monitor = threading.Thread(target=monitor_ram, daemon=True)
    ram_monitor.start()

    tokenized_documents = [
        tokenize(index_text(document), token_settings) for document in documents
    ]

    empty_ids = [
        doc["block_id"]
        for doc, tokens in zip(documents, tokenized_documents)
        if not tokens
    ]
    if empty_ids:
        raise ValueError(
            "These documents have no tokens under the current tokenization "
            f"settings: {empty_ids[:10]}"
        )

    bm25 = BM25Okapi(tokenized_documents, k1=k1, b=b)

    output_dir.mkdir(parents=True, exist_ok=True)

    documents_path = output_dir / "documents.jsonl"
    bm25_path = output_dir / "bm25.pkl"
    manifest_path = output_dir / "manifest.json"

    # Serialize to bytes first so the recorded sha256 is provably the hash
    # of what is on disk.
    documents_bytes = b"".join(
        json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
        for row in documents
    )
    documents_sha256 = hashlib.sha256(documents_bytes).hexdigest()
    documents_path.write_bytes(documents_bytes)

    bm25_bytes = pickle.dumps(bm25, protocol=pickle.HIGHEST_PROTOCOL)
    bm25_sha256 = hashlib.sha256(bm25_bytes).hexdigest()
    bm25_path.write_bytes(bm25_bytes)

    stop_monitor.set()
    ram_monitor.join()

    build_time_seconds = round(time.perf_counter() - build_start_time, 3)

    peak_ram_mb = round(peak_ram_bytes[0] / (1024 * 1024), 2)

    index_size_bytes = documents_path.stat().st_size + bm25_path.stat().st_size
    index_size_mb = round(index_size_bytes / (1024 * 1024), 2)

    corpus_sha256 = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    manifest = {
        "index_type": "BM25Okapi",
        "document_count": len(documents),
        "corpus_file": corpus_path.name,
        "corpus_sha256": corpus_sha256,
        "retrieval": {
            "method": method,
            "k1": k1,
            "b": b,
            "avgdl": round(bm25.avgdl, 4),
            "idf_terms": len(bm25.idf),
        },
        "tokenization": token_settings,
        "section_weight": SECTION_WEIGHT,
        "tokenized_document_count": len(tokenized_documents),
        "average_tokens_per_document": round(
            sum(map(len, tokenized_documents)) / len(tokenized_documents),
            3,
        ),
        "documents_sha256": documents_sha256,
        "bm25_sha256": bm25_sha256,
        "build_time_seconds": build_time_seconds,
        "peak_ram_mb": peak_ram_mb,
        "index_size_bytes": index_size_bytes,
        "index_size_mb": index_size_mb,
        "files": [
            "documents.jsonl",
            "bm25.pkl",
            "manifest.json",
        ],
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Indexed {len(documents)} source blocks.")
    print(f"Documents: {documents_path}")
    print(f"BM25:      {bm25_path}")
    print(f"Manifest:  {manifest_path}")

    print(f"Build time: {build_time_seconds} seconds")
    print(f"Peak RAM:   {peak_ram_mb} MB")
    print(
        f"Index size: {index_size_mb} MB "
        f"({index_size_bytes} bytes)"
    )
    print(
        f"Retrieval:  {method} k1={k1} b={b} avgdl={manifest['retrieval']['avgdl']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
