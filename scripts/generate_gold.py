"""
Stage 3: pairs.jsonl -> dev questions + gold.

Input:  data/pairs.jsonl              (source of truth)
Output: data/dev_questions.jsonl
        data/dev_gold.jsonl

Input schema (per record):

    {
        "id": "tfkp_dev_01",
        "title": "Когда условий Коши—Римана достаточно?",
        "category": "conditions",
        "difficulty": "средняя",
        "question": "Пусть $f=u+iv$ ...",
        "options": [{"id": "A", "text": "..."}, ...],
        "answer_ids": ["B", "C"],
        "abstain": false,
        "reference_answer": "Теорема 2.1 требует ...",
        "reference_pages": [11],
        "source_locator": "Теорема 2.1; определение 2.2.",
        "common_mistake": "Подменить вещественную ..."
    }

Output schemas:

  dev_questions.jsonl:
    {"id", "title", "question", "options"}
    -- input presented to the model.

  dev_gold.jsonl:
    {
        "id", "answer_ids", "abstain", "rationale",          # rationale = reference_answer
        "reference_pages", "source_locator",                  # human, source of truth
        "evidence_sets": [[block_id, ...]],                   # derived, heuristic
        "category", "difficulty", "common_mistake"
    }

Both page-level and block-level references are present. Consumers that
depend on block ids should read `evidence_sets`; consumers that prefer
stable, human-authored references should read `reference_pages` and
`source_locator`.

evidence_sets: HOW THEY ARE BUILT AND WHAT THEY ARE WORTH
=========================================================

`evidence_sets` is DERIVED, not human-authored. It is computed from the
human-provided `reference_pages` and `source_locator`, joined against the
current `corpus.jsonl`. It is regenerated on every run, so it always agrees
with the current chunker — but a change to stage 2 chunking WILL change the
block ids in `evidence_sets`, whereas `reference_pages` and `source_locator`
are invariant.

Selection is locator-grounded:

  1. The `source_locator` is split into items ("Теорема 2.1",
     "определение 2.2", ...).
  2. Query tokens = title + question + correct option texts +
     reference_answer, plus locator tokens with a small boost.
  3. For each reference page, prefer blocks whose stage-2 `section` field
     contains at least one locator item verbatim (i.e. the block was
     emitted under exactly the theorem the human cited). Ties broken by
     TF-IDF over the boosted query, then by a tiny size bonus.
  4. If no block on a page has a section match, the top TF-IDF block is
     used and the page is flagged as a "fallback" hit.

This is a heuristic, not a truth claim. It removes the worst form of
circularity (using the retriever's own signal to pick gold) but does not
prove correctness — a page may contain many blocks and the locator may be
paraphrased. Consumers of this gold should therefore:

  * treat `reference_pages` as authoritative and `evidence_sets` as a hint;
  * score RECALL against pages first;
  * score PRECISION against `source_locator` by checking whether a retrieved
    block's text (or its section) matches the locator — substring match on
    theorem numbers, or an LLM judge for paraphrased locators;
  * NEVER treat block_id equality against `evidence_sets` as ground truth
    across runs — it is only valid within the corpus.jsonl it was built from.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS_FILE = ROOT / "data" / "pairs.jsonl"
CORPUS_FILE = ROOT / "data" / "corpus.jsonl"
QUESTIONS_FILE = ROOT / "data" / "dev_questions.jsonl"
GOLD_FILE = ROOT / "data" / "dev_gold.jsonl"

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

# Weight applied to each occurrence of a locator token in the selection
# query. Values of 3-10 are a good range; too high drowns out the question.
LOCATOR_QUERY_BOOST = 5


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-zа-яё0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# evidence_sets construction
# ---------------------------------------------------------------------------

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


def build_query_counter(record: dict) -> Counter:
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
        counts[tok] += LOCATOR_QUERY_BOOST
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
    """Select the best block on a page.

    Returns (block, grounded). `grounded` is True iff the chosen block's
    section contains at least one locator item — i.e. the block was emitted
    under exactly the theorem / definition the human cited, rather than
    being merely topically similar.
    """
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
) -> tuple[list[list[str]], int, int]:
    """Build evidence_sets for a non-abstain record.

    Returns (evidence_sets, grounded_pages, total_pages).

    Shape of evidence_sets matches the previous schema: an outer list to
    allow future multi-hop records (each element is one hop), each element
    being the block ids supporting that hop. Currently every record is
    single-hop, so the outer list has length 1.
    """
    query_counts = build_query_counter(record)
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
    # if the locator spans a boundary — rare but harmless to guard).
    deduped = list(dict.fromkeys(selected))
    return [deduped], grounded_pages, len(record["reference_pages"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not PAIRS_FILE.exists():
        raise FileNotFoundError(f"Missing {PAIRS_FILE}")
    if not CORPUS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {CORPUS_FILE} — run extract_markdown.py and "
            f"extract_text.py first"
        )

    records = read_jsonl(PAIRS_FILE)
    if not records:
        raise ValueError(f"{PAIRS_FILE} is empty")

    seen: set[str] = set()
    for record in records:
        validate_record(record)
        if record["id"] in seen:
            raise ValueError(f"Duplicate id in {PAIRS_FILE}: {record['id']}")
        seen.add(record["id"])

    corpus = read_jsonl(CORPUS_FILE)
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
                record, by_page, idf
            )
            total_pages += n
            if grounded < n:
                # Flag pages that fell back to TF-IDF (no section match).
                for page in record["reference_pages"]:
                    page_blocks = by_page.get(page, [])
                    _, g = choose_block(
                        page_blocks,
                        build_query_counter(record),
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

    write_jsonl(QUESTIONS_FILE, dev_questions)
    write_jsonl(GOLD_FILE, dev_gold)

    print(f"Created {QUESTIONS_FILE} ({len(dev_questions)} questions)")
    print(f"Created {GOLD_FILE} ({len(dev_gold)} gold records)")
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
