"""Mechanical check that every bibliography-manifest row matches its PDF.

The 2026-07-20 reference audit found roughly a third of the manifest rows in
docs/REFERENCES.md carried a wrong title, author, edition, or identifier —
and one PDF was an entirely different paper. Those errors were invisible
because nothing compared the manifest against the files. This tool closes
that gap: it parses every row of the manifest, opens the named PDF, and
requires the row's title to actually appear (word-fraction match) in the
PDF's leading pages.

Exit status is nonzero on any missing PDF, orphan PDF, or title mismatch, so
the check can gate a commit. Rows the manifest marks as pending an
institutional fetch, or as scans with no text layer, are reported but never
fail the run.

Usage:
    python tools/verify_references.py
"""

from __future__ import annotations

import argparse
import re
import sys

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = EXPERIMENT_ROOT / "docs" / "REFERENCES.md"
REFERENCES_DIRECTORY = EXPERIMENT_ROOT / "references"

# Manifest-row annotations that change how a row is verified. These exact
# phrases appear in docs/REFERENCES.md; keep the two files in sync.
PENDING_FETCH_MARKER = "PDF pending institutional fetch"
NO_TEXT_LAYER_MARKER = "no text layer"

# How much of the PDF to search for the title. Scanned journal articles
# sometimes open with front-matter (e.g. the CACM department header ahead of
# Damerau 1964), so one page is not always enough.
LEADING_PAGE_COUNT = 2

# Fraction of the title's significant words that must appear in the leading
# pages. Tolerates OCR noise and ligature damage while still failing hard on
# a wrong paper (distinct titles share far fewer than half their words).
MINIMUM_TITLE_WORD_FRACTION = 0.6
_SIGNIFICANT_WORD_PATTERN = re.compile(r"[a-z]{4,}")

_MANIFEST_ROW_PATTERN = re.compile(r"^\| `(?P<key>[A-Za-z0-9_]+)` \| (?P<reference>.+?) \|")
_TITLE_PATTERN = re.compile(r"\*(?P<title>[^*]+)\*")
_ARXIV_IDENTIFIER_PATTERN = re.compile(r"arXiv:(?P<identifier>\d{4}\.\d{4,5})")


class RowStatus(Enum):
    TITLE_MATCHES = "ok"
    TITLE_MISMATCH = "TITLE MISMATCH"
    MISSING_PDF = "MISSING PDF"
    ORPHAN_PDF = "ORPHAN PDF (no manifest row)"
    PENDING_INSTITUTIONAL_FETCH = "pending institutional fetch"
    NO_TEXT_LAYER = "scan without text layer (skipped)"
    UNREADABLE_PDF = "UNREADABLE PDF"


FAILING_STATUSES = frozenset(
    {RowStatus.TITLE_MISMATCH, RowStatus.MISSING_PDF,
     RowStatus.ORPHAN_PDF, RowStatus.UNREADABLE_PDF})


@dataclass(frozen=True)
class ManifestRow:
    key: str
    title: str
    arxiv_identifier: str | None
    is_pending_institutional_fetch: bool
    has_no_text_layer: bool


@dataclass(frozen=True)
class RowVerification:
    key: str
    status: RowStatus
    detail: str = ""


def parse_manifest_rows(manifest_markdown: str) -> list[ManifestRow]:
    """One ManifestRow per table row whose first column is a backticked key.

    The title is the first *italic* span of the reference column; the arXiv
    identifier (when present) is advisory context for the report, not a
    pass/fail criterion — camera-ready PDFs often lack the arXiv margin stamp.
    """
    rows = []
    for line in manifest_markdown.splitlines():
        row_match = _MANIFEST_ROW_PATTERN.match(line)
        if row_match is None:
            continue
        reference_text = row_match.group("reference")
        title_match = _TITLE_PATTERN.search(reference_text)
        if title_match is None:
            continue
        arxiv_match = _ARXIV_IDENTIFIER_PATTERN.search(reference_text)
        rows.append(ManifestRow(
            key=row_match.group("key"),
            title=title_match.group("title"),
            arxiv_identifier=arxiv_match.group("identifier") if arxiv_match else None,
            is_pending_institutional_fetch=PENDING_FETCH_MARKER in reference_text,
            has_no_text_layer=NO_TEXT_LAYER_MARKER in reference_text,
        ))
    return rows


def extract_leading_page_text(pdf_path: Path, page_count: int = LEADING_PAGE_COUNT) -> str:
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    return " ".join(
        page.extract_text() or "" for page in reader.pages[:page_count])


def _significant_words(text: str) -> set[str]:
    return set(_SIGNIFICANT_WORD_PATTERN.findall(text.lower()))


def title_word_fraction_present(title: str, page_text: str) -> float:
    """Fraction of the title's significant words found in the page text.

    Word containment (not sequence matching) so that all-caps typesetting,
    ligatures, and OCR spacing damage in the PDF do not defeat the check.
    Page-text words are matched by containment ("robustness" also matches a
    hyphen-broken "robust ness" only via its parts, so we additionally match
    against the whitespace-stripped page text).
    """
    title_words = _significant_words(title)
    if not title_words:
        return 1.0
    page_words = _significant_words(page_text)
    squashed_page_text = re.sub(r"\s+", "", page_text.lower())
    present = {
        word for word in title_words
        if word in page_words or word in squashed_page_text}
    return len(present) / len(title_words)


def verify_row(row: ManifestRow) -> RowVerification:
    pdf_path = REFERENCES_DIRECTORY / f"{row.key}.pdf"

    if row.is_pending_institutional_fetch:
        status = (RowStatus.PENDING_INSTITUTIONAL_FETCH if not pdf_path.exists()
                  else RowStatus.TITLE_MATCHES)
        if not pdf_path.exists():
            return RowVerification(row.key, status, "row present, PDF awaited")
    elif not pdf_path.exists():
        return RowVerification(row.key, RowStatus.MISSING_PDF)

    if row.has_no_text_layer:
        return RowVerification(
            row.key, RowStatus.NO_TEXT_LAYER,
            "identity verified visually at fetch time")

    try:
        page_text = extract_leading_page_text(pdf_path)
    except Exception as extraction_error:  # pypdf raises many concrete types
        return RowVerification(row.key, RowStatus.UNREADABLE_PDF, str(extraction_error))

    fraction_present = title_word_fraction_present(row.title, page_text)
    if fraction_present < MINIMUM_TITLE_WORD_FRACTION:
        return RowVerification(
            row.key, RowStatus.TITLE_MISMATCH,
            f"only {fraction_present:.0%} of title words found — "
            f"expected: {row.title!r}")
    return RowVerification(row.key, RowStatus.TITLE_MATCHES,
                           f"{fraction_present:.0%} of title words found")


def find_orphan_pdfs(manifest_rows: list[ManifestRow]) -> list[RowVerification]:
    manifest_keys = {row.key for row in manifest_rows}
    return [
        RowVerification(pdf_path.stem, RowStatus.ORPHAN_PDF)
        for pdf_path in sorted(REFERENCES_DIRECTORY.glob("*.pdf"))
        if pdf_path.stem not in manifest_keys]


def verify_all_references() -> list[RowVerification]:
    manifest_rows = parse_manifest_rows(MANIFEST_PATH.read_text(encoding="utf-8"))
    return ([verify_row(row) for row in manifest_rows]
            + find_orphan_pdfs(manifest_rows))


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    verifications = verify_all_references()
    for verification in verifications:
        detail = f" — {verification.detail}" if verification.detail else ""
        print(f"[{verification.status.value}] {verification.key}{detail}")

    failures = [v for v in verifications if v.status in FAILING_STATUSES]
    passed_count = len(verifications) - len(failures)
    print(f"\n{passed_count}/{len(verifications)} rows verified; "
          f"{len(failures)} failing")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
