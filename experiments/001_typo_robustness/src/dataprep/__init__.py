"""Data-preparation stage for Experiment 001.

This package implements Stage 0 of the three-stage pipeline:

    Stage 0  raw task data  →  frozen, annotated input dataset   (this package)
    Stage 1  experiment     →  raw generation outputs            (pipeline/)
    Stage 2  scoring        →  scored generation outputs         (scoring.py)

Stage 0 uses a pinned, deterministic linguistic pipeline (spaCy with a
versioned English model, see data/items/annotation_PROVENANCE.json) to
compute the formally-defined key-term set K_P(x) for every item before the
experiment runs.  The annotations are written into the JSONL item files and
committed as frozen artifacts, so the experiment is a pure consumer and no
key-term computation happens at runtime.

See design/02 §2.x for the formal definition of K_P(x), and
design/04 §4.6 for the literature justification for the spaCy choice and
the key-term feature rule.
"""
