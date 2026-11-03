import argparse
import json
import re
from pathlib import Path


DEFAULT_MIN_CHARS = 80
DEFAULT_DISCARD_CHARS = 30
DEFAULT_MAX_CHARS = 1200


def split_long(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?;])\s+", text)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def is_noise(text: str, discard_chars: int = DEFAULT_DISCARD_CHARS) -> bool:
    if len(text) < discard_chars:
        return True
    printable = re.sub(r"[\s.\-–—]", "", text)
    return len(printable) < discard_chars // 2


def merge_and_chunk(
    records: list[dict],
    min_chars: int = DEFAULT_MIN_CHARS,
    discard_chars: int = DEFAULT_DISCARD_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[dict]:
    """
    Merge short records, split long ones, tag each block with the current
    heading, drop noise. Returns a list of {page, section, text}.
    """
    blocks: list[dict] = []
    current_section = ""
    buf_text: str | None = None
    buf_page: int | None = None

    def flush() -> None:
        nonlocal buf_text, buf_page
        if buf_text is None:
            return
        for chunk in split_long(buf_text, max_chars=max_chars):
            chunk = chunk.strip()
            if is_noise(chunk, discard_chars=discard_chars):
                continue
            blocks.append(
                {"page": buf_page, "section": current_section, "text": chunk}
            )
        buf_text = None
        buf_page = None

    for rec in records:
        text = rec["text"]

        if rec["kind"] == "heading":
            flush()
            current_section = text
            buf_text = text
            buf_page = rec["page"]
            continue

        if buf_text is None:
            buf_text = text
            buf_page = rec["page"]
        else:
            buf_text += "\n\n" + text

        if len(buf_text) >= min_chars:
            flush()

    flush()
    return blocks


def build_corpus(
    input_path: Path,
    output_path: Path,
    min_chars: int = DEFAULT_MIN_CHARS,
    discard_chars: int = DEFAULT_DISCARD_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path} - run extract_markdown.py first"
        )

    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    blocks = merge_and_chunk(
        records,
        min_chars=min_chars,
        discard_chars=discard_chars,
        max_chars=max_chars,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for i, block in enumerate(blocks, 1):
            out.write(
                json.dumps(
                    {
                        "block_id": f"b{i:07d}",
                        "page": block["page"],
                        "section": block["section"],
                        "text": block["text"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Corpus        : {output_path}")
    print(f"Input records : {len(records)}")
    print(f"Blocks written: {len(blocks)}")


def parse_args() -> argparse.Namespace:
    ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Merge and chunk extracted paragraphs into a corpus JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=ROOT / "data" / "paragraphs.jsonl",
        help="Path to the input paragraphs JSONL file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "data" / "corpus.jsonl",
        help="Path to the output corpus JSONL file.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="Minimum character count before flushing merged paragraphs.",
    )
    parser.add_argument(
        "--discard-chars",
        type=int,
        default=DEFAULT_DISCARD_CHARS,
        help="Minimum character count to keep a chunk (shorter chunks are discarded).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum character count per chunk before sentence splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_corpus(
        input_path=args.input,
        output_path=args.output,
        min_chars=args.min_chars,
        discard_chars=args.discard_chars,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
