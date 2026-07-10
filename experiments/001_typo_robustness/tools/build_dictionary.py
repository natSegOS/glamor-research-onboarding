"""Fetch and pin the English lexicon used for the is_word predicate.

This is a one-time pre-processing step (design/04 §4.7, design/10 §10).
It builds the vocabulary that ``regimes.make_is_word`` uses to decide whether a
candidate perturbation lands on a real English word — which determines whether a
Regime A (nonword) typo accidentally becomes a Regime B (real-word shift) item.
The script records the SCOWL version, dialect, size band, and file SHA-256 in a
provenance sidecar so the is_word boundary is reproducible and auditable.

Lexicon source: SCOWL (Spell-Checker Oriented Word Lists)
----------------------------------------------------------
SCOWL is used exclusively.  It is the standard citable English spell-checker
lexicon (Kevin Atkinson; wordlist.aspell.net), widely used in academic research
precisely because its membership criterion is dictionary-based rather than
corpus-frequency-based.  A frequency-based list (e.g. wordfreq) would introduce
a confound: the is_word boundary would shift with corpus sampling decisions,
making the Regime A / Regime B classification non-reproducible.  SCOWL's
membership is determined by editorial dictionary sources; the boundary is stable,
pre-registrable, and citable.

The recommended SCOWL size band is 60 (``--scowl-max-size 60``), which covers
the full standard English vocabulary without rare, archaic, or technical
vocabulary that falls outside native-speaker competence.  Atkinson (SCOWL
documentation) describes size 60 as "standard dictionary coverage".

Category scoping: ``words`` only (not the full ``mk-list`` bundle)
--------------------------------------------------------------------
SCOWL's own list-building tool, ``mk-list``, packages a dialect's
``abbreviations``/``contractions``/``proper-names``/``upper``/``words``
sub-categories together with the ``special`` category (``hacker`` jargon,
``roman-numerals``). This script deliberately does **not** reproduce that
full bundle: it merges only the ``words`` sub-category, for ``english`` (the
dialect-neutral core) plus the one requested dialect.

The reason is what ``is_word`` is *for*: deciding whether an edited token is
a real word a reader would recognise as a distinct, meaningful word — the
distinction that separates Regime A (nonword typo) from Regime B (context-recoverable
real-word shift). ``abbreviations``/``upper``/``proper-names`` and the
``special`` category exist in SCOWL to keep a *spell-checker* from flagging
"Mr.", "TCP", or "IV" as misspelled — a different question, and one whose
entries (mostly short, lower-cased abbreviations and roman numerals) are not
"real words" in the reader-recognition sense this predicate needs. Measured
effect of including them anyway: at size band 60, the full ``mk-list`` bundle
counts 100% of all single letters, 51.8% of two-letter strings, and 7.3% of
three-letter strings as "real words" — which would (a) inflate false "landed
on a real word" rejections when constructing Regime A items, and (b) let
Regime B accept a substitution into an abbreviation or proper name as if it
were a context-recoverable real-word shift, which it is not. ``contractions``
is dropped too, though its practical effect is nil: entries contain an
apostrophe, which fails the ``str.isalpha()`` filter below and would never
enter the vocabulary regardless.

This still keeps every citability/reproducibility property the SCOWL choice
was made for: SCOWL is still the sole source, ``--scowl-dialect`` (default:
``american``) still selects exactly one dialect category (never merged with
others — see below), and the exact files plus their SHA-256 are still
recorded in ``PROVENANCE.json``. It is simply a narrower, and more construct-
valid, slice of SCOWL than the stock ``mk-list`` bundle.

Dialect scoping (single dialect only)
--------------------------------------
SCOWL's own documentation (``scowl/README.in``) is explicit that files should
be combined as the ``english`` spelling category plus **one** dialect category
("american", "british", "british_z", "canadian", or "australian") — never
several dialects at once. Merging multiple dialects together (e.g. american +
british + canadian + australian, as an earlier version of this script did by
filtering on the numeric size suffix alone, ignoring the file-name prefix) is
not SCOWL's documented usage: it silently pulls in every dialect's spelling
variants (color/colour, realize/realise, ...) into a single is_word boundary
and is not something a reader familiar with SCOWL would recognize as "the
size-60 SCOWL list" — undermining exactly the citability/reproducibility
argument for using SCOWL in the first place.

Download
--------
Download the prebuilt SCOWL release from SourceForge (the same word-list
build to which http://wordlist.aspell.net/ points) and extract it to a local
directory:

    wget -O /tmp/scowl.tar.gz \\
      "https://sourceforge.net/projects/wordlist/files/SCOWL/2020.12.07/scowl-2020.12.07.tar.gz/download"
    tar -xzf /tmp/scowl.tar.gz -C /tmp/

Then pass the extracted ``final/`` directory (or a single word-list file such
as ``english-words.60``) via ``--scowl-path``.

Note: as of this writing, 2020.12.07 is the newest version for which SCOWL
publishes a prebuilt word-list archive in this format.  The upstream project's
newer releases (tracked at github.com/en-wl/wordlist) ship built Hunspell/
Aspell dictionaries only, not the flat SCOWL ``final/`` word lists this script
consumes — so pinning to 2020.12.07 is not staleness, it is the latest
available prebuilt SCOWL release.  Re-check SourceForge
(https://sourceforge.net/projects/wordlist/files/SCOWL/) before a real study
run in case that has changed.

Usage:

    python tools/build_dictionary.py --scowl-path /path/to/scowl-2020.12.07/final/

Outputs (in --output-directory):
    en_us_pinned.txt     one lowercase word per line, alphabetically sorted
    PROVENANCE.json      SCOWL version, dialect, size band, file SHA-256, timestamp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re

from datetime import datetime, timezone
from pathlib import Path

_DIALECT_CHOICES = ("american", "british", "british_z", "canadian", "australian")


def _matching_word_list_files(
        scowl_path: Path,
        maximum_size_band: int,
        dialect: str,
) -> list[Path]:
    """Return the SCOWL ``final/`` ``words`` files for ``dialect`` at size <=
    ``maximum_size_band``.

    Restricted to the ``words`` sub-category only, for ``english`` (the
    dialect-neutral core) and the requested ``dialect`` (e.g. ``american``).

    This is a deliberate departure from SCOWL's own ``mk-list`` default
    packaging, which additionally pulls in ``upper`` (capitalised/proper-noun
    forms), ``proper-names``, ``abbreviations``, ``contractions``, and the
    ``special`` category (``hacker`` jargon, ``roman-numerals``). Those
    sub-categories exist in SCOWL so a *spell-checker* doesn't flag "Mr.",
    "TCP", or "IV" as misspelled — a different question from the one
    ``regimes.make_is_word`` asks, which is whether an edited token is a real
    word a reader would recognise as a distinct, meaningful English word
    (this decides the Regime A/Regime B boundary; design/04 §4.7). Measured
    effect of including them: at SCOWL's size-60 band, the full ``mk-list``
    bundle counts 100% of all single letters, 51.8% of all two-letter
    strings, and 7.3% of all three-letter strings as "real words" (mostly
    lower-cased abbreviations and roman numerals) — which inflates false
    "landed on a real word" rejections in Regime A's construction and would,
    if used the other way, let Regime B accept a substitution into an
    abbreviation or proper name as a "context-recoverable real-word shift,"
    which it is not. ``words`` alone does not have this problem while
    remaining exactly as citable/reproducible (SCOWL is still the sole
    source; the exact files and their SHA-256 are still recorded in
    ``PROVENANCE.json``).

    Other dialect categories (e.g. ``british`` when ``dialect="american"``)
    are excluded so the vocabulary reflects one coherent spelling convention
    rather than a merge of several.
    """
    if scowl_path.is_dir():
        candidates = sorted(scowl_path.iterdir())
    else:
        candidates = [scowl_path]

    prefix_pattern = re.compile(rf"^(english|{re.escape(dialect)})-words$")

    word_list_files = []
    for candidate_file in candidates:
        try:
            size_band_number = int(candidate_file.suffix.lstrip("."))
        except ValueError:
            continue
        if size_band_number > maximum_size_band:
            continue
        if scowl_path.is_dir() and not prefix_pattern.match(candidate_file.stem):
            continue
        word_list_files.append(candidate_file)
    return word_list_files


def _build_scowl_vocabulary(
        scowl_path: Path,
        maximum_size_band: int,
        dialect: str,
) -> tuple[set[str], dict]:
    """Return ``(vocabulary_set, provenance_dict)`` from a SCOWL word-list source.

    SCOWL organises its word lists into size bands numbered 10, 20, 35, 50, 55,
    60, 70, 80, and 95.  Only tokens that are entirely alphabetic and lowercase
    (after stripping) are retained, matching the contract of
    ``regimes.load_wordlist``.

    Parameters
    ----------
    scowl_path :
        Path to a single SCOWL word-list file or to a directory containing
        multiple SCOWL word-list files (SCOWL's ``final/`` directory).  When a
        directory is supplied, files are matched against the ``english``,
        ``special``, and ``dialect`` categories (see ``_matching_word_list_files``).
    maximum_size_band :
        The largest SCOWL size band to include in the vocabulary.
    dialect :
        The single dialect category to include alongside ``english`` and
        ``special`` (one of ``_DIALECT_CHOICES``).
    """
    path = Path(scowl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SCOWL path not found: {path}\n"
            "Download the prebuilt SCOWL release from "
            "https://sourceforge.net/projects/wordlist/files/SCOWL/ "
            "and pass its final/ directory via --scowl-path.")

    word_list_files = _matching_word_list_files(path, maximum_size_band, dialect)
    if not word_list_files:
        if path.is_dir():
            raise FileNotFoundError(
                f"No SCOWL 'words' files for 'english' or '{dialect}' with a "
                f"numeric suffix <= {maximum_size_band} found in {path}.\n"
                "SCOWL files are named e.g. 'english-words.60' or "
                "'american-words.60'.  Check that --scowl-path points to the "
                "extracted SCOWL final/ directory.")
        raise FileNotFoundError(
            f"{path} does not look like a SCOWL word-list file (expected a "
            f"numeric size suffix, e.g. 'english-words.60').")

    vocabulary: set[str] = set()
    for word_list_file in word_list_files:
        for line in word_list_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            word = line.strip().lower()
            if word.isalpha():
                vocabulary.add(word)

    # Compute a single SHA-256 over all source files so the provenance record
    # is tied to the exact bytes on disk.
    combined_hash = hashlib.sha256()
    for word_list_file in word_list_files:
        combined_hash.update(word_list_file.read_bytes())
    sha256_digest = combined_hash.hexdigest()

    provenance = {
        "source": "scowl",
        "scowl_path": str(path),
        "scowl_dialect": dialect,
        "scowl_maximum_size_band": maximum_size_band,
        "word_list_files_merged": sorted(str(file) for file in word_list_files),
        "sha256_of_source_files": sha256_digest,
        "language": "en",
        "vocabulary_size": len(vocabulary),
        "citation": (
            "Atkinson, K. SCOWL (Spell-Checker Oriented Word Lists). "
            "http://wordlist.aspell.net/  "
            "The exact byte content is recorded in sha256_of_source_files "
            "above for reproducibility."
        ),
    }
    return vocabulary, provenance


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scowl-path",
        type=Path,
        required=True,
        help="path to a SCOWL word-list file or to the extracted SCOWL "
             "final/ directory.  Download from "
             "https://sourceforge.net/projects/wordlist/files/SCOWL/",
    )
    parser.add_argument(
        "--scowl-max-size",
        type=int,
        default=60,
        help="maximum SCOWL size band to include (default: 60, which Atkinson "
             "describes as 'standard dictionary coverage'; valid values are "
             "10, 20, 35, 50, 55, 60, 70, 80, 95)",
    )
    parser.add_argument(
        "--scowl-dialect",
        choices=_DIALECT_CHOICES,
        default="american",
        help="the single dialect category to merge alongside 'english' and "
             "'special' (default: american).  SCOWL's own documentation warns "
             "against merging multiple dialects into one word list.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/wordlists"),
        help="directory to write en_us_pinned.txt and PROVENANCE.json "
             "(default: data/wordlists)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output_directory = arguments.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    print(
        f"Building SCOWL vocabulary from {arguments.scowl_path} "
        f"(dialect: {arguments.scowl_dialect}, "
        f"maximum size band: {arguments.scowl_max_size}) ...")

    vocabulary, provenance = _build_scowl_vocabulary(
        arguments.scowl_path, arguments.scowl_max_size, arguments.scowl_dialect)

    word_list_path = output_directory / "en_us_pinned.txt"
    word_list_path.write_text("\n".join(sorted(vocabulary)) + "\n")
    print(f"  Wrote {len(vocabulary):,} words to {word_list_path}")

    full_provenance = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "output_file": str(word_list_path),
        **provenance,
    }
    provenance_path = output_directory / "PROVENANCE.json"
    provenance_path.write_text(json.dumps(full_provenance, indent=2) + "\n")
    print(f"  Provenance written to {provenance_path}")

    print(
        f"\nTo use this word list in the pipeline:\n"
        f"  from regimes import load_wordlist, make_is_word\n"
        f"  is_word = make_is_word(load_wordlist({str(word_list_path)!r}))\n"
        f"\nOr pass it to generation and perturbation tools via:\n"
        f"  --dictionary {word_list_path}"
    )


if __name__ == "__main__":
    main()
