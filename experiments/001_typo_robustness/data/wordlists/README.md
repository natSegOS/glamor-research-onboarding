# Word lists

## `demo_wordlist.txt`

A small (~490-word) English word list bundled only so the test suite and offline
smoke runs are hermetic — `regimes.make_is_word()` defaults to it when no
dictionary is supplied.  It is **not** adequate for real runs, where
"is this a real word?" determines whether a Regime A typo accidentally produced
a Regime B item.

## The real-study dictionary

The real-study dictionary is built from **SCOWL (Spell-Checker Oriented Word
Lists)** by Kevin Atkinson (wordlist.aspell.net).  SCOWL is used exclusively
because its membership criterion is editorial (dictionary-based), not
frequency-based.  A frequency-based lexicon would introduce a confound: the
Regime A / Regime B boundary would shift with corpus-sampling decisions,
making the classification non-reproducible and non-citable.  SCOWL's boundary
is stable, pre-registrable, and citable (cite the specific version SHA recorded
in `PROVENANCE.json`).

The vocabulary is scoped to a single dialect (`english` + `american` +
`special`, `build_dictionary.py`'s default) rather than merging every SCOWL
dialect together.  This mirrors SCOWL's own `mk-list` tool, whose documented
usage is "the `english` spelling category ... as well as one of `american`,
`british`, ... `australian`" — never several at once — so the resulting list
is recognizable as a standard single-dialect SCOWL bundle, not a bespoke
merge.

### Build the pinned dictionary (one-time step)

**Step 1** — Download the prebuilt SCOWL release.  On Linux / Colab:

```bash
wget -q -O /tmp/scowl.tar.gz \
  "https://sourceforge.net/projects/wordlist/files/SCOWL/2020.12.07/scowl-2020.12.07.tar.gz/download"
tar -xzf /tmp/scowl.tar.gz -C /tmp/
# Pre-built word lists are in the final/ subdirectory.
```

Note: this is downloaded from SourceForge, not the `en-wl/wordlist` GitHub
repo — that repo's tags are SCOWL *source* (raw ingredients + a Makefile, no
prebuilt `final/` directory), not the built word lists this script consumes.
As of this writing, 2020.12.07 is also the newest version for which a
prebuilt SCOWL archive exists in this format (check
https://sourceforge.net/projects/wordlist/files/SCOWL/ for a newer one before
a real study run) — newer upstream releases ship built Hunspell/Aspell
dictionaries only.

**Step 2** — Build the pinned vocabulary:

```bash
python tools/build_dictionary.py \
    --scowl-path /tmp/scowl-2020.12.07/final/ \
    --scowl-max-size 60 \
    --scowl-dialect american
```

This writes `data/wordlists/en_us_pinned.txt` (one lowercase word per line,
alphabetically sorted) and `data/wordlists/PROVENANCE.json` (SCOWL version,
dialect, size band, SHA-256 of source files, timestamp).

### Using the dictionary at runtime

```python
from typo_robustness.regimes import load_wordlist, make_is_word

is_word = make_is_word(load_wordlist("data/wordlists/en_us_pinned.txt"))
```

Or via the generation tool:

```bash
python tools/run_generation.py \
    --config configs/pilot.yaml \
    --model qwen_1b5_pilot \
    --output-directory results/pilot \
    --dictionary data/wordlists/en_us_pinned.txt
```

### Generated files (not committed)

`en_us_pinned.txt` and `PROVENANCE.json` are written by `build_dictionary.py`
and should be listed in `.gitignore`.  Reproduce them from the SCOWL source
using the commands above; the `PROVENANCE.json` `sha256_of_source_files` field
ties the vocabulary to a specific byte sequence for audit.
