"""The human semantic-validity audit: the gate that keeps the primary endpoint
honest.

Provenance
----------
The audit exists because the whole study rests on the claim that Regime A
perturbations preserve intent and Regime C perturbations change it. That claim
must be checked by humans, not asserted (design/09). LLM-as-judge reliability is
a known concern (Shi et al. "Judging the Judges"; Chen et al. "Humans or LLMs"),
so an LLM may pre-screen but is never the final authority.

Protocol (design/09)
--------------------
- Three or more annotators rate a stratified sample of perturbations.
- Each annotator answers, per item: does the intended meaning stay the same, and
  should the gold answer stay the same?
- Inter-annotator agreement is measured with Fleiss' kappa; kappa >= 0.60 is the
  gate to proceed. Below that, the guidelines are revised and the sample re-rated.
- Items without a majority go to adjudication.
- The exclusion rule: a Regime A item that the majority judges NOT
  intent-preserving is removed from the primary endpoint (it was mislabeled, not
  a robustness failure).

Sample size: 385 items per regime gives a worst-case +/- 5 percentage-point Wald
half-width at 95% confidence (statistics.audit_sample_size(0.05) == 385).
"""

from __future__ import annotations

import random

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from enums import SemanticClass


KAPPA_GATE = 0.60                  # proceed only if Fleiss' kappa >= this (design/09 §9.4)


@dataclass
class AuditRating:
    """One annotator's rating of one perturbed item."""
    item_id: str
    annotator_id: str
    intent_preserved: bool
    gold_answer_unchanged: bool
    regime_label: SemanticClass     # the regime the pipeline assigned (A | B | C)


@dataclass
class ItemAuditOutcome:
    """The resolved audit outcome for one item, after majority vote and
    adjudication."""
    item_id: str
    regime_label: SemanticClass
    majority_intent_preserved: Optional[bool]
    majority_gold_unchanged: Optional[bool]
    was_adjudicated: bool
    rating_count: int
    excluded_from_primary: bool


def fleiss_kappa(ratings_per_item: Sequence[Sequence[int]], category_count: int) -> float:
    """Fleiss' kappa for ``n`` items each rated by the SAME number of annotators
    into ``category_count`` mutually exclusive categories.

    ``ratings_per_item[i]`` is the list of category indices assigned to item i.
    Items not rated by the full panel are excluded by the caller before this is
    called (Fleiss' kappa assumes a fixed number of ratings per item).
    """
    item_count = len(ratings_per_item)
    if item_count == 0:
        raise ValueError("no items to score")

    annotators_per_item = len(ratings_per_item[0])
    if any(len(item_ratings) != annotators_per_item for item_ratings in ratings_per_item):
        raise ValueError("Fleiss' kappa requires the same number of ratings for every item")
    if annotators_per_item < 2:
        raise ValueError("Fleiss' kappa requires at least two annotators per item")

    # category_counts[i][k] = number of annotators who placed item i in category k
    category_counts = [[0] * category_count for _ in range(item_count)]
    for item_index, item_ratings in enumerate(ratings_per_item):
        for category_index in item_ratings:
            category_counts[item_index][category_index] += 1

    # Per-item agreement P_i.
    per_item_agreement = []
    for counts in category_counts:
        agreeing_pairs = sum(count * (count - 1) for count in counts)
        per_item_agreement.append(
            agreeing_pairs / (annotators_per_item * (annotators_per_item - 1)))
    mean_observed_agreement = sum(per_item_agreement) / item_count

    # Category marginals p_k and chance agreement.
    total_ratings = item_count * annotators_per_item
    category_marginals = [
        sum(category_counts[i][k] for i in range(item_count)) / total_ratings
        for k in range(category_count)
    ]
    chance_agreement = sum(marginal ** 2 for marginal in category_marginals)

    if chance_agreement >= 1.0:
        return 1.0
    return (mean_observed_agreement - chance_agreement) / (1.0 - chance_agreement)


def stratified_sample(items: Sequence, stratum_key, per_stratum: int, seed: int = 1729) -> list:
    """Take ``per_stratum`` items from each stratum (defined by ``stratum_key``)
    for the audit, so every regime and severity band is represented (design/09
    §9.3). ``stratum_key`` maps an item to its stratum label."""
    items_by_stratum: dict = defaultdict(list)
    for item in items:
        items_by_stratum[stratum_key(item)].append(item)

    random_generator = random.Random(seed)
    sample: list = []
    for stratum in sorted(items_by_stratum, key=str):
        stratum_items = items_by_stratum[stratum]
        take_count = min(per_stratum, len(stratum_items))
        sample.extend(random_generator.sample(stratum_items, take_count))
    return sample


def _majority_boolean(values: Sequence[bool]) -> Optional[bool]:
    """The majority True/False, or None on a tie."""
    counts = Counter(values)
    if counts[True] > counts[False]:
        return True
    if counts[False] > counts[True]:
        return False
    return None


def resolve_item(
        ratings: Sequence[AuditRating],
        adjudicator_rating: Optional[AuditRating] = None,
) -> ItemAuditOutcome:
    """Resolve one item's ratings into a single outcome by majority vote, using
    the adjudicator's rating to break a tie when supplied (design/09 §9.5).

    The exclusion rule: a Regime A item whose majority says intent was NOT
    preserved is excluded from the primary endpoint.
    """
    item_id = ratings[0].item_id
    regime_label = ratings[0].regime_label

    intent_votes = [rating.intent_preserved for rating in ratings]
    gold_votes = [rating.gold_answer_unchanged for rating in ratings]

    majority_intent = _majority_boolean(intent_votes)
    majority_gold = _majority_boolean(gold_votes)

    was_adjudicated = False
    if (majority_intent is None or majority_gold is None) and adjudicator_rating is not None:
        was_adjudicated = True
        if majority_intent is None:
            majority_intent = adjudicator_rating.intent_preserved
        if majority_gold is None:
            majority_gold = adjudicator_rating.gold_answer_unchanged

    excluded_from_primary = (regime_label == SemanticClass.A and majority_intent is False)

    return ItemAuditOutcome(
        item_id=item_id,
        regime_label=regime_label,
        majority_intent_preserved=majority_intent,
        majority_gold_unchanged=majority_gold,
        was_adjudicated=was_adjudicated,
        rating_count=len(ratings),
        excluded_from_primary=excluded_from_primary,
    )


@dataclass
class AuditReport:
    fleiss_kappa_intent: float
    fleiss_kappa_gold: float
    passes_kappa_gate: bool
    item_outcomes: list = field(default_factory=list)
    intent_preservation_rate_by_regime: dict = field(default_factory=dict)
    excluded_item_ids: list = field(default_factory=list)


def audit_report(
        ratings_by_item: dict,
        adjudications: Optional[dict] = None,
        kappa_gate: float = KAPPA_GATE,
) -> AuditReport:
    """Aggregate the audit: compute Fleiss' kappa on both questions, resolve
    each item, derive the per-regime intent-preservation rates, and apply the
    exclusion rule (design/09 §9.4–9.6).

    ``ratings_by_item`` maps item_id -> list[AuditRating]; only items rated by
    the full panel (the modal panel size) enter the kappa computation, as Fleiss'
    kappa requires a fixed number of ratings per item.
    """
    adjudications = adjudications or {}

    panel_sizes = Counter(len(ratings) for ratings in ratings_by_item.values())
    modal_panel_size = panel_sizes.most_common(1)[0][0]
    fully_rated_items = [
        ratings for ratings in ratings_by_item.values()
        if len(ratings) == modal_panel_size
    ]

    intent_categories = [[int(r.intent_preserved) for r in ratings] for ratings in fully_rated_items]
    gold_categories = [[int(r.gold_answer_unchanged) for r in ratings] for ratings in fully_rated_items]

    kappa_intent = fleiss_kappa(intent_categories, category_count=2)
    kappa_gold = fleiss_kappa(gold_categories, category_count=2)

    item_outcomes = [
        resolve_item(ratings, adjudications.get(item_id))
        for item_id, ratings in ratings_by_item.items()
    ]

    intent_preserved_counts: dict = defaultdict(lambda: [0, 0])  # regime -> [preserved, total]
    for outcome in item_outcomes:
        intent_preserved_counts[outcome.regime_label][1] += 1
        if outcome.majority_intent_preserved:
            intent_preserved_counts[outcome.regime_label][0] += 1

    intent_preservation_rate_by_regime = {
        regime: (preserved / total if total else float("nan"))
        for regime, (preserved, total) in intent_preserved_counts.items()
    }

    excluded_item_ids = [o.item_id for o in item_outcomes if o.excluded_from_primary]

    return AuditReport(
        fleiss_kappa_intent=kappa_intent,
        fleiss_kappa_gold=kappa_gold,
        passes_kappa_gate=(kappa_intent >= kappa_gate and kappa_gold >= kappa_gate),
        item_outcomes=item_outcomes,
        intent_preservation_rate_by_regime=intent_preservation_rate_by_regime,
        excluded_item_ids=excluded_item_ids,
    )

