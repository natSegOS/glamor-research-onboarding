# Word lists

## `demo_wordlist.txt`

A small (~490-word) English word list bundled ONLY so the test suite and the
offline smoke runs are hermetic — `regimes.make_is_word()` defaults to it when
no dictionary is supplied. It is NOT adequate for the real study, where "is this
a real word?" decides whether a Regime A typo accidentally landed on a real word
(which would make it a Regime B item) and whether a Regime B substitution is
genuinely a valid-word neighbor.

## The real-study dictionary

For the confirmatory run, use `tools/build_dictionary.py` (see below) to fetch
and pin a full English lexicon. The tool records which lexicon version was used
in `data/wordlists/PROVENANCE.json` and writes the vocabulary as a newline-
delimited word list to `data/wordlists/en_us_pinned.txt`.

Once the file exists, pass it to `make_is_word` via `load_wordlist`:

```python
from typo_robustness.regimes import load_wordlist, make_is_word

is_word = make_is_word(load_wordlist("data/wordlists/en_us_pinned.txt"))
```

The choice of lexicon matters for the semantic-regime boundary (it determines
whether a Regime A typo accidentally lands on a real word), so it is a
pre-registered decision, not an implementation detail (design/04 §4.7,
design/10 §10).

### Build the pinned dictionary

```bash
# Default: wordfreq (top N English words, records installed version).
python tools/build_dictionary.py

# Specify the vocabulary size (default 200 000):
python tools/build_dictionary.py --top-n 250000

# Alternative: hunspell en_US (requires the hunspell package).
python tools/build_dictionary.py --source hunspell
```

This writes `data/wordlists/en_us_pinned.txt` and
`data/wordlists/PROVENANCE.json`.

