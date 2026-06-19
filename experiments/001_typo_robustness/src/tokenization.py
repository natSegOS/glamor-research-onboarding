"""Tokenization quantities for the primary contribution: the mediation analysis.

Provenance
----------
The primary contribution is a tokenization-fragmentation mediation analysis
(design/01 §1.4, design/06 §6.8). The motivation is Chai et al. 2024
("Tokenization Falling Short", arXiv:2406.11687), which shows a CORRELATION
between subword tokenization and typo brittleness. We move toward a causal
statement with a fragmentation-matched counterfactual: among keyboard-plausible
nonword typos of the SAME word at the SAME edit distance, some fragment the word
into many more subword pieces than others. Comparing model accuracy on
high-fragmentation versus low-fragmentation typos of the same word, holding
meaning, position, and edit count fixed, isolates the tokenization channel.

All token counts are computed with the MODEL'S OWN tokenizer (design/05 §5.8);
raw counts are never compared across tokenizers, only the within-model pattern.
A tokenizer here is any object exposing ``.encode(text) -> list``; tests inject
a deterministic fake.
"""

from __future__ import annotations

import random

from dataclasses import dataclass
from typing import Callable, Optional

from enums import FragmentationStratum, Operation, SelectionPolicy, Scope
from perturbation import PerturbationError, damerau_levenshtein_distance
from regimes import derived_seed, make_regime_a_nonword_typo


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def token_inflation_ratio(tokenizer, clean_text: str, perturbed_text: str) -> float:
    """The token-inflation ratio tau = |Tok(perturbed)| / |Tok(clean)|
    (design/02 §2.5). Values above 1 mean the perturbation made the text
    tokenize into more pieces."""
    clean_token_count = count_tokens(tokenizer, clean_text)
    if clean_token_count == 0:
        raise ValueError("clean text tokenizes to zero tokens")
    return count_tokens(tokenizer, perturbed_text) / clean_token_count


def subword_count_change(tokenizer, word: str, perturbed_word: str,
                         include_leading_space: bool = True) -> int:
    """The change in subword-piece count for a single edited word,
    delta_subwords = |Tok(perturbed_word)| - |Tok(word)| (design/02 §2.5).

    A leading space is included by default because byte-pair-encoding tokenizers
    treat a word differently at the start of a string versus mid-sentence; the
    edited word almost always appears mid-sentence, so the leading-space form is
    the representative one.
    """
    prefix = " " if include_leading_space else ""
    return (count_tokens(tokenizer, prefix + perturbed_word)
            - count_tokens(tokenizer, prefix + word))


def fragmentation_stratum(subword_change: int) -> FragmentationStratum:
    """Bucket a subword-count change into Low (<= 0) or High (>= 1) fragmentation
    (design/02 §2.5)."""
    return FragmentationStratum.HIGH if subword_change >= 1 else FragmentationStratum.LOW


@dataclass
class FragmentationMatchedPair:
    """A matched pair of nonword typos of the same word at the same edit budget,
    one in the Low fragmentation stratum and one in the High stratum. This is
    the unit of the counterfactual: the two variants differ only in their
    tokenization consequence."""
    word: str
    edit_budget: int
    low_fragmentation_variant: str
    high_fragmentation_variant: str
    low_fragmentation_subword_change: int
    high_fragmentation_subword_change: int


def build_fragmentation_matched_pair(
        tokenizer,
        word: str,
        edit_budget: int,
        seed: int,
        is_word: Callable[[str], bool],
        candidate_count: int = 48,
) -> Optional[FragmentationMatchedPair]:
    """Build the fragmentation-matched counterfactual for one word (design/02
    §2.5, design/06 §6.8).

    Enumerates keyboard-plausible nonword (Regime A) variants of ``word`` at the
    given edit budget, partitions them by fragmentation stratum, and returns one
    deterministically-chosen Low/High pair. Returns ``None`` if the word admits
    no such pair under this tokenizer — those words simply drop out of the
    mediation module, which is sound because the contrast is within-word by
    construction (a word with no Low/High contrast contributes no information to
    the counterfactual).
    """
    subword_change_by_variant: dict[str, int] = {}

    for candidate_index in range(candidate_count):
        candidate_seed = derived_seed(seed, "counterfactual", word, edit_budget, candidate_index)

        for operation in (Operation.SUBSTITUTE, Operation.INSERT, Operation.DELETE, Operation.TRANSPOSE):
            try:
                variant, _edits, _metadata = make_regime_a_nonword_typo(
                    word, operation, edit_budget, candidate_seed, is_word,
                    selection_policy=SelectionPolicy.KEYBOARD_NEIGHBOR, scope=Scope.ANYWHERE,
                    max_attempts=8)
            except PerturbationError:
                continue

            stripped_variant = variant.strip()
            variant_is_usable = (
                bool(stripped_variant)
                and stripped_variant.lower() != word.lower()
                and damerau_levenshtein_distance(word, stripped_variant) == edit_budget
            )
            if variant_is_usable:
                subword_change_by_variant.setdefault(
                    stripped_variant,
                    subword_count_change(tokenizer, word, stripped_variant))

    low_fragmentation_variants = sorted(
        variant for variant, change in subword_change_by_variant.items() if change <= 0)
    high_fragmentation_variants = sorted(
        variant for variant, change in subword_change_by_variant.items() if change >= 1)

    if not low_fragmentation_variants or not high_fragmentation_variants:
        return None

    picker = random.Random(derived_seed(seed, "counterfactual_pick", word, edit_budget))
    chosen_low = picker.choice(low_fragmentation_variants)
    chosen_high = picker.choice(high_fragmentation_variants)

    return FragmentationMatchedPair(
        word=word,
        edit_budget=edit_budget,
        low_fragmentation_variant=chosen_low,
        high_fragmentation_variant=chosen_high,
        low_fragmentation_subword_change=subword_change_by_variant[chosen_low],
        high_fragmentation_subword_change=subword_change_by_variant[chosen_high],
    )
