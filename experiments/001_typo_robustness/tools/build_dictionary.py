"""Fetch and pin the real-study English lexicon for the is_word predicate.

This is a **one-time pre-processing step** (design/04 §4.7, design/10 §10).
It builds the vocabulary that ``regimes.make_is_word`` uses to decide whether a
candidate perturbation lands on a real English word — which determines whether
a Regime A (nonword) typo accidentally becomes a Regime B (real-word shift)
item. The lexicon must be pre-registered before the confirmatory run.

The script records which package and version was used in a provenance sidecar,
so the is_word boundary is reproducible and auditable.

Two sources are supported:

  **wordfreq** (default): ``wordfreq.top_n_list("en", N)`` returns the N most
  frequent English words. The list is deterministic for a fixed (N, package
  version). Use ``--top-n`` to control vocabulary size.

  **hunspell** (``--source hunspell``): loads the ``en_US`` dictionary via the
  ``hunspell`` Python package. Requires the system hunspell library and the
  ``hunspell`` pip package. Output is the union of the dictionary's stem list.

Usage:

    python tools/build_dictionary.py                     # wordfreq, 200 000 words
    python tools/build_dictionary.py --top-n 250000      # wordfreq, 250 000 words
    python tools/build_dictionary.py --source hunspell   # hunspell en_US

Outputs (in --output-directory):
    en_us_pinned.txt     newline-delimited lowercase word list
    PROVENANCE.json      source, version, parameters, timestamp
"""

from __future__ import annotations

import argparse
import json

from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Vocabulary builders.
# ---------------------------------------------------------------------------

def _build_wordfreq_vocabulary(top_n: int) -> tuple[set[str], dict]:
    """Return (vocabulary_set, provenance_dict) using wordfreq."""
    try:
        import wordfreq
        from importlib.metadata import version as pkg_version
    except ImportError as error:
        raise ImportError(
            "building the wordfreq vocabulary requires 'wordfreq' "
            "(pip install wordfreq)") from error

    wordfreq_version = pkg_version("wordfreq")
    vocabulary = set(wordfreq.top_n_list("en", top_n))

    provenance = {
        "source": "wordfreq",
        "wordfreq_version": wordfreq_version,
        "language": "en",
        "top_n": top_n,
        "vocabulary_size": len(vocabulary),
    }
    return vocabulary, provenance


def _build_hunspell_vocabulary() -> tuple[set[str], dict]:
    """Return (vocabulary_set, provenance_dict) using hunspell en_US."""
    try:
        import hunspell
        from importlib.metadata import version as pkg_version
    except ImportError as error:
        raise ImportError(
            "building the hunspell vocabulary requires the 'hunspell' pip "
            "package and the system hunspell library "
            "(pip install hunspell)") from error

    try:
        hunspell_version = pkg_version("hunspell")
    except Exception:
        hunspell_version = "unknown"

    dictionary = hunspell.HunSpell("/usr/share/hunspell/en_US.dic",
                                    "/usr/share/hunspell/en_US.aff")
    vocabulary = {word.lower() for word in dictionary.stem("")
                  if word.isalpha()}

    provenance = {
        "source": "hunspell",
        "hunspell_version": hunspell_version,
        "language": "en_US",
        "vocabulary_size": len(vocabulary),
    }
    return vocabulary, provenance


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["wordfreq", "hunspell"],
        default="wordfreq",
        help="lexicon source (default: wordfreq)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=200_000,
        help="wordfreq only: vocabulary size, i.e. top-N most frequent English "
             "words (default: 200000)",
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

    print(f"building vocabulary from source={arguments.source!r} ...")

    if arguments.source == "wordfreq":
        vocabulary, source_provenance = _build_wordfreq_vocabulary(arguments.top_n)
    else:
        vocabulary, source_provenance = _build_hunspell_vocabulary()

    # Write the word list in the format regimes.load_wordlist already reads:
    # one lowercase word per line, no header.
    word_list_path = output_directory / "en_us_pinned.txt"
    word_list_path.write_text("\n".join(sorted(vocabulary)) + "\n")
    print(f"  wrote {len(vocabulary):,} words to {word_list_path}")

    # Write the provenance sidecar.
    provenance = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "output_file": str(word_list_path),
        **source_provenance,
    }
    provenance_path = output_directory / "PROVENANCE.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"  provenance written to {provenance_path}")

    print("\nTo use this dictionary in the pipeline:")
    print("  from regimes import load_wordlist, make_is_word")
    print(f"  is_word = make_is_word(load_wordlist({str(word_list_path)!r}))")
    print("\nOr pass it to tools/run_generation.py and tools/build_asr_items.py via:")
    print(f"  --dictionary {word_list_path}")


if __name__ == "__main__":
    main()

