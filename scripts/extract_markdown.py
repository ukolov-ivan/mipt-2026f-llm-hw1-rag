import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import pymupdf


LATIN_EXTENDED = re.compile(r"[\u00c0-\u00ff]")
CYRILLIC = re.compile(r"[\u0400-\u052f]")

# Runs of cp1251-as-latin1 mojibake (À-ÿ = А-я, ¨ = Ё, ¸ = ё)
MOJIBAKE_RUN = re.compile(r"[\u00c0-\u00ff\u00a8\u00b8]{3,}")

NEGATION_MAP = {
    "=": "≠",
    "∈": "∉",
    "≡": "≢",
    "<": "≮",
    ">": "≯",
    "⩾": "≱",
    "⩽": "≰",
}
NEGATED_RELATION = re.compile(
    r"\s*\u0338\s*([" + re.escape("".join(NEGATION_MAP)) + r"])\s*"
)

TOC_LINE = re.compile(
    r"(?:\.\s*){3,}\d{1,3}\s*$|" r"…{2,}\s*\d{1,3}\s*$|" r"_{3,}\s*\d{1,3}\s*$"
)

THEOREM_LABEL = re.compile(
    r"^(Теорема|Определение|Лемма|Следствие|Утверждение|"
    r"Замечание|Пример|Задача)\s+\d+(?:\.\d+)*\.?",
    re.IGNORECASE,
)

NUMBERED_HEADING = re.compile(
    r"^\d+(?:\.\d+)*\.?\s+[А-ЯЁA-Z]",
    re.IGNORECASE,
)

CHAPTER_HEADING = re.compile(
    r"^Глава\s+\S+",
    re.IGNORECASE,
)

HEADER_LINE = re.compile(
    r"^\s*(?:\d{1,4}\s+)?(?:"
    r"ЛЕКЦИИ\s+ПО\s+ТФКП"
    r"|В\.\s*В\.\s*ГОРЯЙНОВ\s*,?\s*"
    r"Е\.\s*С\.\s*ПОЛОВИНКИН"
    r")(?:\s+\d{1,4})?\s*$",
    re.IGNORECASE,
)


def _decode_cp1251(text: str) -> str | None:
    try:
        return text.encode("latin-1").decode("cp1251")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def repair_word(word: str) -> str:
    if not LATIN_EXTENDED.search(word) or CYRILLIC.search(word):
        return word

    candidate = _decode_cp1251(word)

    if candidate is not None:
        letters = [char for char in candidate if char.isalpha()]
        if letters:
            russian_count = sum(bool(CYRILLIC.fullmatch(char)) for char in letters)
            if russian_count / len(letters) >= 0.65:
                return candidate

    def replace_run(match: re.Match) -> str:
        repaired = _decode_cp1251(match.group())
        if repaired is None:
            return match.group()

        start = match.start()
        if start > 0 and not CYRILLIC.fullmatch(word[start - 1]):
            return " " + repaired
        return repaired

    return MOJIBAKE_RUN.sub(replace_run, word)


def repair_russian(text: str, log_encoding_repairs: bool = False) -> str:
    def replace_match(match: re.Match) -> str:
        original = match.group()
        repaired = repair_word(original)

        if log_encoding_repairs and repaired != original:
            print(
                f"Encoding repair: {original!r} -> {repaired!r}",
                file=sys.stderr,
            )

        return repaired

    return re.sub(r"\S+", replace_match, text)


def repair_math(text: str) -> str:
    text = NEGATED_RELATION.sub(
        lambda m: f" {NEGATION_MAP[m.group(1)]} ",
        text,
    )
    return unicodedata.normalize("NFC", text)


def heading_candidate(line: str) -> str:
    return re.sub(r"^[^\w]+", "", line).strip()


def is_chapter_one(line: str) -> bool:
    candidate = heading_candidate(line)
    return bool(
        re.match(
            r"^1\.\s*Комплексные\s+числа\b",
            candidate,
            re.IGNORECASE,
        )
    )


def is_heading(line: str) -> bool:
    candidate = heading_candidate(line)

    if len(candidate) > 180:
        return False

    return bool(
        THEOREM_LABEL.match(candidate)
        or NUMBERED_HEADING.match(candidate)
        or CHAPTER_HEADING.match(candidate)
    )


def make_paragraph(lines: list[str]) -> str:
    text = "\n".join(lines)

    text = re.sub(
        r"(?<=[А-Яа-яЁё])-\s*\n\s*(?=[А-Яа-яЁё])",
        "",
        text,
    )

    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def clean_lines(raw_text: str, log_encoding_repairs: bool = False) -> list[str]:
    raw_text = repair_math(repair_russian(raw_text, log_encoding_repairs))
    lines = [line.strip() for line in raw_text.splitlines()]

    cleaned = []
    for line in lines:
        if HEADER_LINE.match(line):
            continue
        if TOC_LINE.search(line):
            continue
        cleaned.append(line)

    return cleaned


def split_inline_headings(lines: list[str]) -> list[str]:
    result = []

    heading_pattern = re.compile(
        r"(?<!\S)"
        r"(?=(?:Теорема|Определение|Лемма|Следствие|Утверждение|"
        r"Замечание|Пример|Задача)\s+\d+(?:\.\d+)*\.?)"
    )

    for line in lines:
        if is_heading(line):
            result.append(line)
            continue

        parts = heading_pattern.split(line)

        if len(parts) == 1:
            result.append(line)
            continue

        result.extend(part.strip() for part in parts if part.strip())

    return result


def extract_page_records(
    raw_text: str,
    log_encoding_repairs: bool = False,
) -> list[dict] | None:
    """
    Return [{"kind": "heading"|"para", "text": ...}, ...] for a page,
    or None if the page is TOC and should be skipped.
    """
    lines = clean_lines(raw_text, log_encoding_repairs)

    chapter_start = next(
        (index for index, line in enumerate(lines) if is_chapter_one(line)),
        None,
    )

    if chapter_start is not None:
        lines = lines[chapter_start:]
    else:
        toc_count = sum(
            bool(TOC_LINE.search(line)) for line in raw_text.splitlines()
        )
        starts_with_contents = any(
            line.casefold() in {"содержание", "оглавление"} for line in lines[:5]
        )
        if toc_count >= 3 or starts_with_contents:
            return None

    lines = split_inline_headings(lines)

    records: list[dict] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            text = make_paragraph(buf)
            if text:
                records.append({"kind": "para", "text": text})
            buf.clear()

    for line in lines:
        if not line:
            flush()
        elif is_heading(line):
            flush()
            records.append({"kind": "heading", "text": line})
        else:
            buf.append(line)

    flush()
    return records


def parse_args() -> argparse.Namespace:
    ROOT = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Extract paragraphs and headings from a PDF textbook.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=ROOT / "data" / "textbook.pdf",
    )
    parser.add_argument(
        "--paragraphs",
        type=Path,
        default=ROOT / "data" / "paragraphs.jsonl",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=ROOT / "data" / "extracted.md",
    )
    parser.add_argument(
        "--log-encoding-repairs",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    args.paragraphs.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    markdown_parts: list[str] = []
    skipped_pages: list[int] = []

    with pymupdf.open(args.pdf) as pdf:
        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            raw_text = page.get_text("text", sort=True)
            records = extract_page_records(
                raw_text,
                log_encoding_repairs=args.log_encoding_repairs,
            )

            if records is None:
                skipped_pages.append(page_number)
                markdown_parts.append(
                    f"<!-- PAGE: {page_number} -->\n<!-- SKIP_TOC_PAGE -->"
                )
                continue

            for rec in records:
                all_records.append(
                    {"page": page_number, "kind": rec["kind"], "text": rec["text"]}
                )

            page_md = "\n\n".join(rec["text"] for rec in records)
            markdown_parts.append(f"<!-- PAGE: {page_number} -->\n{page_md}")

    with args.paragraphs.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    args.markdown.write_text(
        "\n\n".join(markdown_parts) + "\n", encoding="utf-8"
    )

    print(f"Paragraphs     : {args.paragraphs} ({len(all_records)} records)")
    print(f"Debug markdown : {args.markdown}")
    print(f"PDF pages      : {len(markdown_parts)}")
    print(f"Skipped TOC    : {skipped_pages}")


if __name__ == "__main__":
    main()
