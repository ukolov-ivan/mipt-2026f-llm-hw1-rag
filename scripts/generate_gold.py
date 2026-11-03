import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

VALID_ANSWER_IDS = frozenset({"A", "B", "C", "D"})

REQUIRED_FIELDS = frozenset({
    "id",
    "question",
    "options",
    "answer_ids",
    "abstain",
    "reference_answer",
    "reference_pages",
    "source_locator",
    "category",
})

DEFAULT_LOCATOR_QUERY_BOOST = 5


# IO

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", text.lower())


# Validation

def validate_record(record: dict) -> None:
    rid = record.get("id", "<no id>")

    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"{rid}: missing required fields {sorted(missing)}")

    if not isinstance(record["abstain"], bool):
        raise ValueError(f"{rid}: 'abstain' must be bool")

    if record["abstain"]:
        if record["answer_ids"]:
            raise ValueError(
                f"{rid}: abstain=true but answer_ids={record['answer_ids']}"
            )
    else:
        if not record["answer_ids"]:
            raise ValueError(f"{rid}: non-abstain record has empty answer_ids")
        invalid = set(record["answer_ids"]) - VALID_ANSWER_IDS
        if invalid:
            raise ValueError(
                f"{rid}: invalid answer_ids {sorted(invalid)}"
            )
        if not record["reference_pages"]:
            raise ValueError(
                f"{rid}: non-abstain record has empty reference_pages; "
                f"cannot build evidence_sets"
            )

    pages = record["reference_pages"]
    if not isinstance(pages, list) or not all(isinstance(p, int) for p in pages):
        raise ValueError(f"{rid}: reference_pages must be list[int]")

    options = record["options"]
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError(f"{rid}: expected 4 options, got {len(options)}")
    opt_ids = {o.get("id") for o in options}
    if opt_ids != VALID_ANSWER_IDS:
        raise ValueError(
            f"{rid}: option ids must be A/B/C/D, got {sorted(opt_ids)}"
        )


def validate_corpus_has_pages(records: list[dict], corpus_pages: set[int]) -> None:
    missing = [
        (record["id"], page)
        for record in records
        for page in record["reference_pages"]
        if page not in corpus_pages
    ]
    if missing:
        detail = "\n".join(f"  {rid}: page {page}" for rid, page in missing)
        raise ValueError(
            f"Reference pages missing from corpus:\n{detail}\n"
            f"Either the QA page numbers are wrong, or stages 1/2 dropped "
            f"content from those pages."
        )

# evidence_sets construction

def locator_items(locator: str) -> list[str]:
    """Split a source_locator into normalized items.

    'Теорема 2.1; вывод уравнения Лапласа и определение 2.3; теорема 5.3.'
    -> ['теорема 2.1', 'вывод уравнения лапласа и определение 2.3',
        'теорема 5.3']
    """
    return [
        item.strip().lower()
        for item in re.split(r"[;·]", locator)
        if item.strip()
    ]


def build_idf(corpus: list[dict]) -> dict[str, float]:
    df: Counter = Counter()
    for block in corpus:
        df.update(set(tokens(block.get("text", ""))))
    total = max(len(corpus), 1)
    return {t: math.log(1 + total / (1 + n)) for t, n in df.items()}


def build_query_counter(
    record: dict,
    locator_boost: int = DEFAULT_LOCATOR_QUERY_BOOST,
) -> Counter:
    """Query tokens for block selection: title + question + correct options +
    reference_answer, with locator tokens boosted.
    """
    parts = [
        record.get("title", ""),
        record["question"],
        " ".join(
            o["text"] for o in record["options"]
            if o["id"] in record["answer_ids"]
        ),
        record["reference_answer"],
    ]
    counts: Counter = Counter(tokens(" ".join(parts)))
    for tok in tokens(record["source_locator"]):
        counts[tok] += locator_boost
    return counts


def section_matches(block: dict, locator_items_: list[str]) -> int:
    """How many locator items appear verbatim in the block's section."""
    section = (block.get("section") or "").lower()
    if not section:
        return 0
    return sum(1 for item in locator_items_ if item in section)


def choose_block(
    page_blocks: list[dict],
    query_counts: Counter,
    idf: dict[str, float],
    locator_items_: list[str],
) -> tuple[dict, bool]:
    """Select the best block on a page"""
    best_block: dict | None = None
    best_key: tuple[int, float] | None = None

    for block in page_blocks:
        btoks = tokens(block.get("text", ""))
        counts = Counter(btoks)
        tfidf = sum(
            idf.get(t, 0.0) * min(c, counts[t])
            for t, c in query_counts.items()
        )
        size_bonus = min(len(btoks), 500) * 1e-7
        sec = section_matches(block, locator_items_)

        key = (sec, tfidf + size_bonus)
        if best_key is None or key > best_key:
            best_block = block
            best_key = key

    assert best_block is not None
    return best_block, best_key[0] > 0


def build_evidence_sets(
    record: dict,
    by_page: dict[int, list[dict]],
    idf: dict[str, float],
    locator_boost: int = DEFAULT_LOCATOR_QUERY_BOOST,
) -> tuple[list[list[str]], int, int]:
    """Build evidence_sets for a non-abstain record.

    Returns (evidence_sets, grounded_pages, total_pages).
    """
    query_counts = build_query_counter(record, locator_boost=locator_boost)
    loc_items = locator_items(record["source_locator"])

    selected: list[str] = []
    grounded_pages = 0
    for page in record["reference_pages"]:
        page_blocks = by_page.get(page, [])
        if not page_blocks:
            raise ValueError(
                f"{record['id']}: no corpus blocks for PDF page {page}"
            )
        block, grounded = choose_block(page_blocks, query_counts, idf, loc_items)
        selected.append(block["block_id"])
        if grounded:
            grounded_pages += 1

    # Deduplicate while preserving order (a page pair can pick the same block
    # if the locator spans a boundary - rare but harmless to guard).
    deduped = list(dict.fromkeys(selected))
    return [deduped], grounded_pages, len(record["reference_pages"])


def parse_args() -> argparse.Namespace:
    ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Build dev questions and gold evidence sets from QA pairs and corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=ROOT / "data" / "pairs.jsonl",
        help="Path to the input QA pairs JSONL file.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data" / "corpus.jsonl",
        help="Path to the input corpus JSONL file.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "dev_questions.jsonl",
        help="Path to the output questions JSONL file.",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=ROOT / "data" / "dev_gold.jsonl",
        help="Path to the output gold JSONL file.",
    )
    parser.add_argument(
        "--locator-boost",
        type=int,
        default=DEFAULT_LOCATOR_QUERY_BOOST,
        help="Weight boost applied to tokens from source_locator during TF-IDF matching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pairs.exists():
        raise FileNotFoundError(f"Missing {args.pairs}")
    if not args.corpus.exists():
        raise FileNotFoundError(
            f"Missing {args.corpus} - run extract_markdown.py and "
            f"build_corpus.py first"
        )

    records = read_jsonl(args.pairs)
    if not records:
        raise ValueError(f"{args.pairs} is empty")

    seen: set[str] = set()
    for record in records:
        validate_record(record)
        if record["id"] in seen:
            raise ValueError(f"Duplicate id in {args.pairs}: {record['id']}")
        seen.add(record["id"])

    corpus = read_jsonl(args.corpus)
    by_page: dict[int, list[dict]] = {}
    for block in corpus:
        by_page.setdefault(int(block["page"]), []).append(block)
    corpus_pages = set(by_page)

    validate_corpus_has_pages(records, corpus_pages)

    idf = build_idf(corpus)

    dev_questions: list[dict] = []
    dev_gold: list[dict] = []
    fallback_pages: list[tuple[str, int]] = []
    total_pages = 0

    for record in records:
        dev_questions.append(
            {
                "id": record["id"],
                "title": record.get("title", ""),
                "question": record["question"],
                "options": record["options"],
            }
        )

        if record["abstain"]:
            evidence_sets: list[list[str]] = []
        else:
            evidence_sets, grounded, n = build_evidence_sets(
                record,
                by_page,
                idf,
                locator_boost=args.locator_boost,
            )
            total_pages += n
            if grounded < n:
                # Flag pages that fell back to TF-IDF (no section match).
                for page in record["reference_pages"]:
                    page_blocks = by_page.get(page, [])
                    _, g = choose_block(
                        page_blocks,
                        build_query_counter(
                            record, locator_boost=args.locator_boost
                        ),
                        idf,
                        locator_items(record["source_locator"]),
                    )
                    if not g:
                        fallback_pages.append((record["id"], page))

        dev_gold.append(
            {
                "id": record["id"],
                "answer_ids": record["answer_ids"],
                "abstain": record["abstain"],
                "rationale": record["reference_answer"],
                "reference_pages": record["reference_pages"],
                "source_locator": record["source_locator"],
                "evidence_sets": evidence_sets,
                "category": record["category"],
                "difficulty": record.get("difficulty", ""),
                "common_mistake": record.get("common_mistake", ""),
            }
        )

    write_jsonl(args.questions, dev_questions)
    write_jsonl(args.gold, dev_gold)

    print(f"Created {args.questions} ({len(dev_questions)} questions)")
    print(f"Created {args.gold} ({len(dev_gold)} gold records)")
    print(
        f"Citation grounding: "
        f"{total_pages - len(fallback_pages)}/{total_pages} pages matched "
        f"a locator item in the block's section "
        f"({len(fallback_pages)} fallback(s))"
    )
    if fallback_pages:
        print("\nFallback pages (no section match, TF-IDF only):")
        for rid, page in fallback_pages:
            print(f"  {rid}: page {page}")

    cat_counts = Counter(r["category"] for r in dev_gold)
    print("\nCategory distribution:")
    for cat, n in sorted(cat_counts.items()):
        print(f"  {cat:<40} {n}")

    diff_counts = Counter(r["difficulty"] for r in dev_gold if r["difficulty"])
    if diff_counts:
        print("\nDifficulty distribution:")
        for diff, n in sorted(diff_counts.items()):
            print(f"  {diff:<40} {n}")


if __name__ == "__main__":
    main()
