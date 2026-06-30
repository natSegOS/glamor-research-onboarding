"""Fetch and pin the English lexicon used for the is_word predicate.

This is a one-time pre-processing step (design/04 §4.7, design/10 §10).
It builds the vocabulary that ``regimes.make_is_word`` uses to decide whether a
candidate perturbation lands on a real English word — which determines whether a
Regime A (nonword) typo accidentally becomes a Regime B (real-word shift) item.
The script records the SCOWL version, size band, and file SHA-256 in a provenance
sidecar so the is_word boundary is reproducible and auditable.

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

Download
--------
Download a SCOWL release from http://wordlist.aspell.net/dicts/ and extract it
to a local directory.  Then pass the directory (or a single word-list file such
as ``english-words.60``) via ``--scowl-path``.

Usage:

    python tools/build_dictionary.py --scowl-path /path/to/scowl-2020.12.07/

Outputs (in --output-directory):
    en_us_pinned.txt     one lowercase word per line, alphabetically sorted
    PROVENANCE.json      SCOWL version, size band, file SHA-256, timestamp
"""

from __future__ import annotations

import argparse
import hashlib
import json

from datetime import datetime, timezone
from pathlib import Path


def _build_scowl_vocabulary(
        scowl_path: Path,
        maximum_size_band: int,
) -> tuple[set[str], dict]:
    """Return ``(vocabulary_set, provenance_dict)`` from a SCOWL word-list source.

    SCOWL organises its word lists into size bands numbered 10, 20, 35, 50, 55,
    60, 70, 80, and 95.  All files whose numeric suffix is less than or equal to
    ``maximum_size_band`` are merged.  The recommended band is 60 (Atkinson,
    SCOWL documentation: "This is the standard dictionary size").

    Only tokens that are entirely alphabetic and lowercase (after stripping) are
    retained, matching the contract of ``regimes.load_wordlist``.

    Parameters
    ----------
    scowl_path :
        Path to a single SCOWL word-list file or to a directory containing
        multiple SCOWL word-list files.  When a directory is supplied, all
        files whose suffix parses as an integer ≤ ``maximum_size_band`` are
        merged.
    maximum_size_band :
        The largest SCOWL size band to include in the vocabulary.
    """
    path = Path(scowl_path)
    if not path.exists():
        raise FileNotFoundError(
            f"SCOWL path not found: {path}\n"
            "Download a SCOWL release from http://wordlist.aspell.net/dicts/ "
            "and pass the path via --scowl-path.")

    if path.is_dir():
        word_list_files = []
        for candidate_file in sorted(path.iterdir()):
            try:
                size_band_number = int(candidate_file.suffix.lstrip("."))
            except ValueError:
                continue
            if size_band_number <= maximum_size_band:
                word_list_files.append(candidate_file)
        if not word_list_files:
            raise FileNotFoundError(
                f"No SCOWL word-list files with a numeric suffix ≤ "
                f"{maximum_size_band} found in {path}.\n"
                "SCOWL files are named e.g. 'english-words.60'.  Check that "
                "--scowl-path points to the extracted SCOWL release directory.")
    else:
        word_list_files = [path]

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
        "scowl_maximum_size_band": maximum_size_band,
        "word_list_files_merged": [str(file) for file in word_list_files],
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
        help="path to a SCOWL word-list file or to a directory containing "
             "multiple SCOWL word-list files (e.g. the extracted SCOWL release "
             "directory).  Download from http://wordlist.aspell.net/dicts/",
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
        f"(maximum size band: {arguments.scowl_max_size}) ...")

    vocabulary, provenance = _build_scowl_vocabulary(
        arguments.scowl_path, arguments.scowl_max_size)

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
