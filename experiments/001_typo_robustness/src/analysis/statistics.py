"""The pre-registered statistical machinery.

Provenance
----------
Every choice here is justified in design/06 and backed by a reference (see
docs/PROVENANCE.md §6):

  - Paired 2x2 table and McNemar's test (mid-p exact when discordant pairs are
    few, asymptotic chi-square otherwise) for per-cell paired binary comparison.
  - The Connor (1987) McNemar sample-size formula and its planning approximation.
  - BCa item-paired bootstrap confidence intervals at B = 10,000 resamples
    (Bestgen 2022; the 10,000-resample convention in NLP evaluation).
  - Every paired metric defined in design/02 §2.6.

The number of bootstrap resamples is 10,000 exactly, matching the design and the
pre-registration; this is a registered quantity, so it is not approximated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy
from scipy import stats as scipy_stats

from enums import INTERACTIONAL_FAILURE_STATUSES, McNemarTestMethod, SampleSizeMethod


# Exact McNemar when the number of discordant pairs is below this threshold;
# asymptotic chi-square at or above it (design/06 §6.4).
EXACT_TEST_DISCORDANT_THRESHOLD = 25

# BCa bootstrap resamples. 10,000 exactly (design/06 §6.5); a registered value.
DEFAULT_BOOTSTRAP_RESAMPLES = 10000

# Minimum paired observations for a bootstrap confidence interval to be
# meaningful; smaller cells report point estimates and counts but no interval.
MINIMUM_CELL_SIZE_FOR_INTERVAL = 2


# ---------------------------------------------------------------------------
# Paired 2x2 contingency table.
# ---------------------------------------------------------------------------

@dataclass
class PairedContingencyTable:
    """The paired clean-vs-perturbed 2x2 table.

        both_correct       clean correct, perturbed correct   (a)
        broke              clean correct, perturbed wrong      (b)  <- typo-induced failure
        recovered          clean wrong,   perturbed correct    (c)
        both_wrong         clean wrong,   perturbed wrong       (d)
    """
    both_correct: int
    broke: int
    recovered: int
    both_wrong: int

    @property
    def total(self) -> int:
        return self.both_correct + self.broke + self.recovered + self.both_wrong


def build_paired_table(clean_correctness: Sequence[int],
                       perturbed_correctness: Sequence[int]) -> PairedContingencyTable:
    """Build the paired 2x2 table from two matched, equal-length 0/1 arrays
    (same items, same order)."""
    clean = numpy.asarray(clean_correctness, dtype=int)
    perturbed = numpy.asarray(perturbed_correctness, dtype=int)
    if clean.shape != perturbed.shape:
        raise ValueError("clean and perturbed arrays must be matched (same items, same order)")

    return PairedContingencyTable(
        both_correct=int(numpy.sum((clean == 1) & (perturbed == 1))),
        broke=int(numpy.sum((clean == 1) & (perturbed == 0))),
        recovered=int(numpy.sum((clean == 0) & (perturbed == 1))),
        both_wrong=int(numpy.sum((clean == 0) & (perturbed == 0))),
    )


# ---------------------------------------------------------------------------
# McNemar's test (design/06 §6.4).
# ---------------------------------------------------------------------------

@dataclass
class McNemarResult:
    broke: int
    recovered: int
    statistic: Optional[float]
    p_value: float
    method: McNemarTestMethod


def mcnemar_test(broke: int, recovered: int,
                 exact_threshold: int = EXACT_TEST_DISCORDANT_THRESHOLD,
                 use_mid_p: bool = True) -> McNemarResult:
    """McNemar's test on the discordant pairs (broke vs recovered).

    Uses the mid-p exact binomial test when the number of discordant pairs is
    below ``exact_threshold`` and the asymptotic chi-square otherwise. The rule
    is fixed in advance and never chosen after seeing the p-value (design/06
    §6.4).
    """
    discordant_count = broke + recovered

    if discordant_count == 0:
        return McNemarResult(broke, recovered, None, 1.0, McNemarTestMethod.EXACT_MIDP)

    if discordant_count < exact_threshold:
        smaller_count = min(broke, recovered)
        two_sided_p = min(1.0, 2.0 * scipy_stats.binom.cdf(smaller_count, discordant_count, 0.5))
        if use_mid_p:
            two_sided_p = max(0.0, two_sided_p
                              - scipy_stats.binom.pmf(smaller_count, discordant_count, 0.5))
        return McNemarResult(broke, recovered, None, float(two_sided_p), McNemarTestMethod.EXACT_MIDP)

    chi_square_statistic = (broke - recovered) ** 2 / discordant_count
    p_value = float(scipy_stats.chi2.sf(chi_square_statistic, df=1))
    return McNemarResult(broke, recovered, float(chi_square_statistic), p_value, McNemarTestMethod.ASYMPTOTIC)


# ---------------------------------------------------------------------------
# Sample size (design/06 §6.3).
# ---------------------------------------------------------------------------

def mcnemar_sample_size(
        detectable_difference: float,
        discordant_pair_rate: float,
        alpha: float = 0.05,
        power: float = 0.80,
        method: SampleSizeMethod = SampleSizeMethod.CONNOR) -> int:
    """Paired items needed to detect a paired-accuracy difference
    ``detectable_difference`` (= p_broke - p_recovered) at discordant-pair rate
    ``discordant_pair_rate`` (= p_broke + p_recovered).

    method "connor"  Connor 1987 (Biometrics 43:207-211):
        N = [z_{1-a/2} sqrt(p_d) + z_{1-b} sqrt(p_d - delta^2)]^2 / delta^2
    method "simple"  the planning approximation used in the design tables:
        N ~= (z_{1-a/2} + z_{1-b})^2 * p_d / delta^2
    """
    if not (0 < detectable_difference <= discordant_pair_rate <= 1):
        raise ValueError("require 0 < detectable_difference <= discordant_pair_rate <= 1")

    z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
    z_power = scipy_stats.norm.ppf(power)

    if method == SampleSizeMethod.CONNOR:
        sample_size = (
            (z_alpha * numpy.sqrt(discordant_pair_rate)
             + z_power * numpy.sqrt(discordant_pair_rate - detectable_difference ** 2)) ** 2
            / detectable_difference ** 2
        )
    elif method == SampleSizeMethod.SIMPLE:
        sample_size = (
            (z_alpha + z_power) ** 2 * discordant_pair_rate / detectable_difference ** 2
        )
    else:
        raise ValueError(f"unknown method {method!r}")

    return int(numpy.ceil(sample_size))


def audit_sample_size(margin: float, confidence: float = 0.95) -> int:
    """Worst-case (rate = 0.5) Wald sample size for a binary-rate confidence
    interval of half-width ``margin`` (design/09 §9.3):
        n = z^2 * 0.25 / margin^2.
    """
    z = scipy_stats.norm.ppf(0.5 + confidence / 2)
    return int(numpy.ceil(z ** 2 * 0.25 / margin ** 2))


# ---------------------------------------------------------------------------
# BCa item-paired bootstrap confidence intervals (design/06 §6.5).
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceInterval:
    estimate: float
    low: float
    high: float
    method: str
    resamples: int


# ``ConfidenceInterval.method`` when even the percentile bootstrap fails (a
# fully degenerate cell — every item has the same outcome); the interval
# collapses to the point estimate with zero resamples, per the pre-registered
# contingency in bootstrap_confidence_interval_paired's docstring.
DEGENERATE_BOOTSTRAP_METHOD_LABEL = "degenerate"
DEGENERATE_BOOTSTRAP_RESAMPLE_COUNT = 0


_PAIRED_STATISTICS = {
    "delta": lambda clean, perturbed: clean.mean() - perturbed.mean(),
    "retention": lambda clean, perturbed: (perturbed.mean() / clean.mean()
                                           if clean.mean() > 0 else numpy.nan),
    "clean_conditioned_failure": lambda clean, perturbed: (
        ((clean == 1) & (perturbed == 0)).sum() / (clean == 1).sum()
        if (clean == 1).sum() > 0 else numpy.nan),
}


def bootstrap_confidence_interval_paired(
        clean_correctness: Sequence[int],
        perturbed_correctness: Sequence[int],
        statistic: str = "delta",
        resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
        confidence: float = 0.95,
        seed: int = 1729) -> ConfidenceInterval:
    """A BCa bootstrap confidence interval for a paired statistic. Items are
    resampled WITH their clean/perturbed pair kept together (paired=True), so the
    matching structure is preserved (design/06 §6.5).

    Degenerate-distribution guard: if BCa cannot be computed (for example when
    every item has the same outcome), fall back to a percentile interval, and if
    that also fails, return the point estimate as a zero-width interval. This is
    the pre-registered contingency.
    """
    clean = numpy.asarray(clean_correctness, dtype=int)
    perturbed = numpy.asarray(perturbed_correctness, dtype=int)
    if clean.shape != perturbed.shape or clean.size < MINIMUM_CELL_SIZE_FOR_INTERVAL:
        raise ValueError(f"need matched arrays with n >= {MINIMUM_CELL_SIZE_FOR_INTERVAL}")

    def statistic_on_float_arrays(clean_resample, perturbed_resample):
        return _PAIRED_STATISTICS[statistic](
            numpy.asarray(clean_resample, dtype=float),
            numpy.asarray(perturbed_resample, dtype=float))

    point_estimate = float(statistic_on_float_arrays(clean, perturbed))

    for method in ("BCa", "percentile"):
        random_generator = numpy.random.default_rng(seed)
        common_arguments = dict(
            paired=True, vectorized=False, n_resamples=resamples,
            confidence_level=confidence, method=method)
        try:
            import warnings
            with warnings.catch_warnings():
                # A degenerate cell (every item identical) makes BCa undefined;
                # that is exactly the case the percentile fallback handles, so
                # the warning is expected and not informative here.
                warnings.simplefilter("ignore")
                try:
                    result = scipy_stats.bootstrap(
                        (clean, perturbed), statistic_on_float_arrays,
                        rng=random_generator, **common_arguments)
                except TypeError:                   # SciPy < 1.15 used random_state
                    result = scipy_stats.bootstrap(
                        (clean, perturbed), statistic_on_float_arrays,
                        random_state=random_generator, **common_arguments)

            low = float(result.confidence_interval.low)
            high = float(result.confidence_interval.high)
            if numpy.isfinite(low) and numpy.isfinite(high):
                return ConfidenceInterval(point_estimate, low, high, method, resamples)
        except ValueError:
            # scipy raises ValueError when BCa's acceleration/bias-correction
            # is undefined for degenerate data (see the docstring's
            # degenerate-distribution guard) — expected, try the next method.
            continue

    return ConfidenceInterval(
        point_estimate, point_estimate, point_estimate,
        DEGENERATE_BOOTSTRAP_METHOD_LABEL, DEGENERATE_BOOTSTRAP_RESAMPLE_COUNT)


# ---------------------------------------------------------------------------
# The metric set (design/02 §2.6). Each is a pure function of matched arrays.
# ---------------------------------------------------------------------------

def clean_accuracy(clean_correctness: Sequence[int]) -> float:
    return float(numpy.mean(clean_correctness))


def perturbed_accuracy(perturbed_correctness: Sequence[int]) -> float:
    return float(numpy.mean(perturbed_correctness))


def paired_degradation(clean_correctness, perturbed_correctness) -> float:
    """The headline endpoint: absolute paired accuracy drop A0 - A1 (design/02
    §2.6)."""
    return clean_accuracy(clean_correctness) - perturbed_accuracy(perturbed_correctness)


def clean_conditioned_failure(clean_correctness, perturbed_correctness) -> float:
    """Of the items the model got right cleanly, the fraction the perturbation
    broke: b / (a + b)."""
    table = build_paired_table(clean_correctness, perturbed_correctness)
    clean_correct_count = table.both_correct + table.broke
    return table.broke / clean_correct_count if clean_correct_count > 0 else float("nan")


def retention(clean_correctness, perturbed_correctness) -> float:
    """The fraction of clean accuracy that survives perturbation, A1 / A0."""
    baseline = clean_accuracy(clean_correctness)
    return perturbed_accuracy(perturbed_correctness) / baseline if baseline > 0 else float("nan")


def answer_flip_rate(clean_answers: Sequence, perturbed_answers: Sequence) -> float:
    """The fraction of items whose parsed answer changed under perturbation,
    regardless of correctness (behavioral stability)."""
    clean_list, perturbed_list = list(clean_answers), list(perturbed_answers)
    if len(clean_list) != len(perturbed_list) or not clean_list:
        raise ValueError("matched answer lists required")
    return sum(c != p for c, p in zip(clean_list, perturbed_list)) / len(clean_list)


def appropriate_change_rate(perturbed_answers: Sequence, new_gold_answers: Sequence,
                            matches=lambda answer, gold: answer == gold) -> float:
    """For Regime C: the fraction of items where the model correctly updated to
    the new gold answer after the meaning changed (design/02 §2.6)."""
    pairs = list(zip(perturbed_answers, new_gold_answers))
    return sum(matches(answer, gold) for answer, gold in pairs) / len(pairs) if pairs else float("nan")


def over_robustness_rate(perturbed_answers: Sequence, old_gold_answers: Sequence,
                         matches=lambda answer, gold: answer == gold) -> float:
    """For Regime C: the fraction of items where the model clung to the OLD gold
    answer after the meaning changed (over-invariance; design/02 §2.6)."""
    pairs = list(zip(perturbed_answers, old_gold_answers))
    return sum(matches(answer, gold) for answer, gold in pairs) / len(pairs) if pairs else float("nan")


def invalid_or_clarification_rate(parse_statuses: Sequence[str]) -> float:
    """The fraction of items that failed interactionally (unparseable,
    clarification, or refusal) rather than answering (design/02 §2.6)."""
    statuses = list(parse_statuses)
    return (sum(status in INTERACTIONAL_FAILURE_STATUSES for status in statuses) / len(statuses)
            if statuses else float("nan"))


def discordant_rate(clean_correctness, perturbed_correctness) -> float:
    """The discordant-pair rate (broke + recovered) / total — the quantity the
    Stage-2 pilot measures to fix the per-cell sample size (design/06 §6.3)."""
    table = build_paired_table(clean_correctness, perturbed_correctness)
    return (table.broke + table.recovered) / table.total if table.total else float("nan")


def summarize_cell(clean_correctness: Sequence[int], perturbed_correctness: Sequence[int],
                   seed: int = 1729, resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES) -> dict:
    """One reporting block per cell (design/06 §6.10): point estimates, the
    delta confidence interval, the raw 2x2 counts, and the McNemar test —
    everything a reader needs to recompute the cell.

    Cells smaller than the interval minimum report counts and point estimates
    but omit the confidence interval (its bounds equal the estimate) so that an
    underpowered pilot cell never crashes the analysis.
    """
    table = build_paired_table(clean_correctness, perturbed_correctness)
    mcnemar = mcnemar_test(table.broke, table.recovered)

    if table.total >= MINIMUM_CELL_SIZE_FOR_INTERVAL:
        delta_interval = bootstrap_confidence_interval_paired(
            clean_correctness, perturbed_correctness, "delta", resamples, seed=seed)
    else:
        point = paired_degradation(clean_correctness, perturbed_correctness)
        delta_interval = ConfidenceInterval(point, point, point, "insufficient_n", 0)

    return {
        "n": table.total,
        "both_correct": table.both_correct,
        "broke": table.broke,
        "recovered": table.recovered,
        "both_wrong": table.both_wrong,
        "clean_accuracy": clean_accuracy(clean_correctness),
        "perturbed_accuracy": perturbed_accuracy(perturbed_correctness),
        "delta": delta_interval.estimate,
        "delta_ci_low": delta_interval.low,
        "delta_ci_high": delta_interval.high,
        "delta_ci_method": delta_interval.method,
        "clean_conditioned_failure": clean_conditioned_failure(clean_correctness, perturbed_correctness),
        "retention": retention(clean_correctness, perturbed_correctness),
        "discordant_rate": discordant_rate(clean_correctness, perturbed_correctness),
        "mcnemar_p_value": mcnemar.p_value,
        "mcnemar_method": mcnemar.method,
    }
