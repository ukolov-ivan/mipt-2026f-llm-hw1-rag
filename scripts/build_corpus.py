import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "paragraphs.jsonl"
OUTPUT_PATH = ROOT / "data" / "corpus.jsonl"

MIN_CHARS = 80
DISCARD_CHARS = 30
MAX_CHARS = 1200


def split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]

    sentences = re.split(r"(?<=[.!?;])\s+", text)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) > MAX_CHARS and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def is_noise(text: str) -> bool:
    if len(text) < DISCARD_CHARS:
        return True
    printable = re.sub(r"[\s.\-–—]", "", text)
    return len(printable) < DISCARD_CHARS // 2


def merge_and_chunk(records: list[dict]) -> list[dict]:
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
        for chunk in split_long(buf_text):
            chunk = chunk.strip()
            if is_noise(chunk):
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

        if len(buf_text) >= MIN_CHARS:
            flush()

    flush()
    return blocks


def build_corpus(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing {input_path} — run extract_markdown.py first"
        )

    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    blocks = merge_and_chunk(records)

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


def main() -> None:
    build_corpus(INPUT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
