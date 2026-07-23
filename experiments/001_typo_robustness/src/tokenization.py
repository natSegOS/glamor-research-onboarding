"""Tokenization quantities for the primary contribution: the tokenization-
fragmentation mediation analysis (design/01 §1.4, design/06 §6.8).

Chai et al. 2024 ("Tokenization Falling Short", arXiv:2406.11687) shows a
CORRELATION between subword tokenization and typo brittleness. We move
toward a causal statement with a fragmentation-matched counterfactual: among
keyboard-plausible nonword typos of the SAME word at the SAME edit distance,
some fragment the word into many more subword pieces than others. Comparing
model accuracy on high- vs low-fragmentation typos of the same word, holding
meaning, position, and edit count fixed, isolates the tokenization channel.

All token counts use the MODEL'S OWN tokenizer (design/05 §5.8); raw counts
are never compared across tokenizers, only the within-model pattern. A
tokenizer here is any object exposing ``.encode(text) -> list``; tests
inject a deterministic fake.
"""

from __future__ import annotations

import random
import re

from dataclasses import dataclass
from typing import Callable, Optional

from enums import FragmentationStratum, Operation, SelectionPolicy, Scope
from perturbation import PerturbationError, damerau_levenshtein_distance
from regimes import derived_seed, make_regime_a_nonword_typo


_COUNTERFACTUAL_MINIMUM_WORD_LENGTH = 4

# Stop enumerating counterfactual candidates once both strata hold this many
# distinct variants: the choice pool is already diverse, and full enumeration
# of every seed x 4 operations per word per budget dominated request-building
# time on the Colab pilot.
_SUFFICIENT_VARIANTS_PER_STRATUM = 8

# Give up after this many candidates while a stratum is still EMPTY: under a
# 128k vocabulary ~75% of words never yield a High variant, and continuing to
# search for them doubles build time for zero pairs. The wider search (the
# candidate_count default) only continues for words with both strata started.
_GIVE_UP_CANDIDATES_WHILE_STRATUM_EMPTY = 48


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def token_inflation_ratio(tokenizer, clean_text: str, perturbed_text: str) -> float:
    """tau = |Tok(perturbed)| / |Tok(clean)| (design/02 §2.5). Above 1 means
    the perturbation made the text tokenize into more pieces."""
    clean_token_count = count_tokens(tokenizer, clean_text)
    if clean_token_count == 0:
        raise ValueError("clean text tokenizes to zero tokens")
    return count_tokens(tokenizer, perturbed_text) / clean_token_count


def subword_count_change(tokenizer, word: str, perturbed_word: str,
                         include_leading_space: bool = True) -> int:
    """delta_subwords = |Tok(perturbed_word)| - |Tok(word)|.

    A leading space is included by default because byte-pair-encoding
    tokenizers treat a word differently at the start of a string versus
    mid-sentence, and the edited word almost always appears mid-sentence.
    """
    prefix = " " if include_leading_space else ""
    return (count_tokens(tokenizer, prefix + perturbed_word)
            - count_tokens(tokenizer, prefix + word))


def fragmentation_stratum(subword_change: int) -> FragmentationStratum:
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
        # 96 (not 48): under a 128k-vocabulary tokenizer most single typos do
        # not add subword pieces, so High variants are rare. The wider search
        # nearly doubles pair yield, and the early stratum exit keeps the
        # common case cheap.
        candidate_count: int = 96,
) -> Optional[FragmentationMatchedPair]:
    """Build the fragmentation-matched counterfactual for one word.

    Enumerates keyboard-plausible nonword (Regime A) variants of ``word`` at the
    given edit budget, partitions them by fragmentation stratum, and returns one
    deterministically-chosen Low/High pair. Returns ``None`` if the word admits
    no such pair under this tokenizer. Those words drop out of the
    mediation module, which is sound because the contrast is within-word by
    construction (a word with no Low/High contrast contributes no information to
    the counterfactual).
    """
    subword_change_by_variant: dict[str, int] = {}

    def stratum_counts() -> tuple[int, int]:
        changes = subword_change_by_variant.values()
        return (sum(change <= 0 for change in changes),
                sum(change >= 1 for change in changes))

    for candidate_index in range(candidate_count):
        low_count, high_count = stratum_counts()
        if min(low_count, high_count) >= _SUFFICIENT_VARIANTS_PER_STRATUM:
            break
        if (candidate_index >= _GIVE_UP_CANDIDATES_WHILE_STRATUM_EMPTY
                and min(low_count, high_count) == 0):
            break
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


# How many candidate words the pair-aware search tries before giving up on an
# item. Bounds request-building time; under a 128k vocabulary ~75% of words
# yield no High variant, so trying several candidates (rather than committing
# to the single longest word, the pre-2026-07-20 rule) is what lifted Method
# A's item yield: the pilot excluded 607 of 740 items for "no Low/High pair"
# under the single-candidate rule.
MAXIMUM_COUNTERFACTUAL_CANDIDATE_WORDS = 8


def ordered_counterfactual_candidate_words(
        content_text: str, is_word: Callable[[str], bool]) -> list[str]:
    """Candidate target words for the counterfactual, best-first: longest
    words first (richest keyboard-plausible variant space, so the best chance
    both fragmentation strata are populated), first occurrence breaking ties.
    Deterministic (no seed). The caller tries each in order until one admits
    a Low/High pair, capped at MAXIMUM_COUNTERFACTUAL_CANDIDATE_WORDS."""
    eligible_words = []
    for word in re.findall(r"[A-Za-z]+", content_text):
        if (len(word) >= _COUNTERFACTUAL_MINIMUM_WORD_LENGTH
                and is_word(word) and word not in eligible_words):
            eligible_words.append(word)
    by_length_then_first_occurrence = sorted(
        eligible_words,
        key=lambda word: (-len(word), eligible_words.index(word)))
    return by_length_then_first_occurrence[:MAXIMUM_COUNTERFACTUAL_CANDIDATE_WORDS]


def apply_counterfactual_variant(
        content_text: str, word: str, variant: str) -> tuple[str, int]:
    """Replace the first whole-word occurrence of ``word`` with ``variant``.
    Returns the perturbed text and the character index of the replacement.

    Boundaries are letter-based lookarounds, not ``\\b``: the target word comes
    from the ``[A-Za-z]+`` tokenizer above, which extracts "Python" out of
    "Python3", where ``\\bPython\\b`` would NOT match (digits are word
    characters to ``\\b``). The two functions must share one notion of "word"
    (this exact mismatch crashed the first Llama pilot on an MMLU item).
    """
    match = re.search(
        rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", content_text)
    if match is None:
        raise PerturbationError(
            f"counterfactual target word {word!r} not found as a whole word")
    return (content_text[:match.start()] + variant + content_text[match.end():],
            match.start())
