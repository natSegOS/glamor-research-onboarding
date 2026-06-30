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

### Build the pinned dictionary (one-time step)

**Step 1** — Download a SCOWL release.  On Linux / Colab:

```bash
wget -q -O /tmp/scowl.tar.gz \
  https://github.com/en-wl/wordlist/archive/refs/tags/v2020.12.07.tar.gz
tar -xzf /tmp/scowl.tar.gz -C /tmp/
# Pre-built word lists are in the final/ subdirectory.
```

**Step 2** — Build the pinned vocabulary:

```bash
python tools/build_dictionary.py \
    --scowl-path /tmp/wordlist-2020.12.07/final/ \
    --scowl-max-size 60
```

This writes `data/wordlists/en_us_pinned.txt` (one lowercase word per line,
alphabetically sorted) and `data/wordlists/PROVENANCE.json` (SCOWL version,
size band, SHA-256 of source files, timestamp).

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
