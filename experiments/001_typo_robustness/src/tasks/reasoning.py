"""Reasoning task items: GSM-Symbolic (primary) and GSM8K (contamination contrast).

Provenance
----------
GSM-Symbolic (Mirzadeh et al., ICLR 2025, arXiv:2410.05229) is the primary
reasoning task. GSM8K (Cobbe et al., 2021, arXiv:2110.14168) is the standard
benchmark used as a contamination-contrast partner.

Items are pre-fetched from HuggingFace once by tools/build_task_items.py using
the ``load_official_*`` functions below, written to pinned JSONL files with SHA
provenance, and loaded during a run by ``load_reasoning_jsonl``. No live HF
loading occurs during a run.

``generate_synthetic_reasoning_items`` is an offline generator used for offline
unit tests and for Regime C operand-swap (which needs the template's answer
function). It is NOT used in the main study or pilot — those use JSONL.

All loaders return ``ReasoningItem`` objects; the pipeline is source-agnostic.
"""

from __future__ import annotations

import json
import random
import re

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

from enums import TaskFamily
from lexicons import load_word_lexicon
from tasks._shared import (
    HASH_DELIMITED_ANSWER_PATTERN,
    build_full_prompt,
    build_instruction_and_content_scope_spans,
)

# Operation words used by the offline synthetic generator to record which
# mathematical-operation terms appear in the generated question text.  These
# are derived from the template vocabulary (the templates embed words like
# "buys", "gives away", "saves") rather than from a free-text heuristic, so
# they remain appropriate for the synthetic context.
#
# Note: this list is NOT used in any runtime path or JSONL loader.  Key terms
# for JSONL items are frozen at data-preparation time by the linguistic
# annotation stage (src/dataprep/annotate.py), which replaces the old
# runtime heuristic that previously called this list on raw question text.
# See design/04 §4.6 and the data-preparation provenance record.
OPERATION_WORDS: frozenset[str] = load_word_lexicon("operation_words.txt")

# Maximum number of times the synthetic generator tries to sample operands
# that satisfy the template's constraint before skipping a template.
_PARAMETER_SAMPLING_MAX_ATTEMPTS = 64

# Names for the synthetic generator; sampled to personalise template questions,
# mirroring Figure 1 of Mirzadeh et al. (2024) GSM-Symbolic.
SYNTHETIC_NAMES = (
    "Ava", "Ben", "Carla", "Dev", "Elena", "Farid", "Grace", "Hiro",
    "Imani", "Jonas", "Keiko", "Liam", "Mara", "Noor", "Omar", "Priya",
)


# One hand-written exemplar demonstrating the '#### <number>' final line. The
# pilot showed small instruct models ignore a bare zero-shot format instruction
# (0/100 clean GSM-Symbolic generations emitted '####'), pushing half of all
# scoring onto the last-number fallback tier. Hand-written — never taken from
# any evaluation dataset — so it cannot leak an item. Frozen at
# pre-registration; part of the fixed prompt scaffold, never perturbed.
REASONING_FORMAT_EXEMPLAR = (
    "Problem: A box holds 4 red pens and 3 blue pens. Tom buys 2 boxes. "
    "How many pens does Tom have?\n"
    "Solution: Each box holds 4 + 3 = 7 pens. Two boxes hold 2 * 7 = 14 pens.\n"
    "#### 14"
)

REASONING_INSTRUCTION = (
    "Solve the following math problem. Reason step by step, then end your "
    "response with the final numeric answer on its own line in exactly the "
    "form '#### <number>'.\n"
    "\n"
    "Here is an example of the required format:\n"
    "\n"
    f"{REASONING_FORMAT_EXEMPLAR}\n"
    "\n"
    "Now solve this problem:"
)

# ---------------------------------------------------------------------------
# GSM-Symbolic template parser (Workstream 7)
# ---------------------------------------------------------------------------

# Canonical English fraction words that appear in GSM-Symbolic {param,value}
# annotations.  Mapped to exact Fraction values so the answer_function can
# evaluate arithmetic expressions that include them.
FRACTION_WORDS: dict[str, Fraction] = {
    # Single-word bareform denominators (e.g. template default "half").
    "half":             Fraction(1, 2),
    "third":            Fraction(1, 3),
    "quarter":          Fraction(1, 4),
    "fifth":            Fraction(1, 5),
    "sixth":            Fraction(1, 6),
    "seventh":          Fraction(1, 7),
    "eighth":           Fraction(1, 8),
    "ninth":            Fraction(1, 9),
    "tenth":            Fraction(1, 10),
    # Hyphenated compound forms (template defaults and question text).
    "one-half":         Fraction(1, 2),
    "one-third":        Fraction(1, 3),
    "one-quarter":      Fraction(1, 4),
    "one-fifth":        Fraction(1, 5),
    "one-sixth":        Fraction(1, 6),
    "one-seventh":      Fraction(1, 7),
    "one-eighth":       Fraction(1, 8),
    "one-ninth":        Fraction(1, 9),
    "one-tenth":        Fraction(1, 10),
    "two-thirds":       Fraction(2, 3),
    "three-quarters":   Fraction(3, 4),
    "two-fifths":       Fraction(2, 5),
    "three-fifths":     Fraction(3, 5),
    "four-fifths":      Fraction(4, 5),
    # Space-separated phrase forms (as they sometimes appear in question text).
    "a half":           Fraction(1, 2),
    "a third":          Fraction(1, 3),
    "a quarter":        Fraction(1, 4),
    "two thirds":       Fraction(2, 3),
    "three quarters":   Fraction(3, 4),
    "two fifths":       Fraction(2, 5),
    "three fifths":     Fraction(3, 5),
    "four fifths":      Fraction(4, 5),
}

# Standard English verbal multipliers appearing in GSM-Symbolic templates.
# Some templates store an integer multiplier as a word in the question text
# (e.g. {mult,twice}) while the answer expression uses it numerically.
# These mappings convert the textual form back to an integer so that
# extract_instance_parameters can validate extracted values against gold_answer.
# Source: Huddleston & Pullum (2002) 'The Cambridge Grammar of the English
# Language', §14 (multiplicative expressions); "thrice" is the standard
# literary form of "three times" (OED, s.v. "thrice").
VERBAL_MULTIPLIER_WORDS: dict[str, int] = {
    # Classical single-word forms.
    "once":        1,
    "twice":       2,
    "double":      2,
    "thrice":      3,
    "triple":      3,
    "quadruple":   4,
    "quintuple":   5,
    "sextuple":    6,
    "septuple":    7,
    "octuple":     8,
    # "<N> times" forms.
    "two times":   2,
    "three times": 3,
    "four times":  4,
    "five times":  5,
    "six times":   6,
    "seven times": 7,
    "eight times": 8,
    "nine times":  9,
    "ten times":   10,
    # Cardinal number words used numerically in arithmetic expressions.
    # These appear as GSM-Symbolic template str params (e.g. {n,seven}) whose
    # answer formula uses them as integers.
    "one":         1,
    "two":         2,
    "three":       3,
    "four":        4,
    "five":        5,
    "six":         6,
    "seven":       7,
    "eight":       8,
    "nine":        9,
    "ten":         10,
    "eleven":      11,
    "twelve":      12,
    "thirteen":    13,
    "fourteen":    14,
    "fifteen":     15,
    "sixteen":     16,
    "seventeen":   17,
    "eighteen":    18,
    "nineteen":    19,
    "twenty":      20,
}

# Regex to locate {param,value} annotations in question_annotated.
_PARAM_VALUE_RE = re.compile(r"\{(\w+),([^}]+)\}")
# Regex to extract the #answer expression.  The Apple repo uses both
# "#answer: expr" and "#answer = expr" (with colon or equals sign).
_ANSWER_SECTION_RE = re.compile(r"#answer\s*[=:]\s*(.+?)(?:\n#|\Z)", re.IGNORECASE | re.DOTALL)

# Safe builtins for eval of #answer: expressions.  Only math operators and
# Fraction/int/float arithmetic are needed; no import or function calls.

# ---------------------------------------------------------------------------
# {param,value} annotation typing — the single source of truth used by
# parse_gsm_symbolic_template, load_reasoning_jsonl, and _fetch_gsm_from_hf.
# ---------------------------------------------------------------------------

def _typed_parameter_value(raw_value: str, *, resolve_verbal_multipliers: bool = False):
    """Classify one {param,value} annotation string into its typed Python
    value: int (digit string) > Fraction (FRACTION_WORDS) >
    [verbal multiplier (VERBAL_MULTIPLIER_WORDS), if enabled] > str."""
    stripped = raw_value.strip()
    if stripped.lstrip("-").isdigit():
        return int(stripped)
    if stripped in FRACTION_WORDS:
        return FRACTION_WORDS[stripped]
    if resolve_verbal_multipliers and stripped.lower() in VERBAL_MULTIPLIER_WORDS:
        return VERBAL_MULTIPLIER_WORDS[stripped.lower()]
    return stripped


def _extract_annotated_parameters(
        question_annotated: str, *, resolve_verbal_multipliers: bool = False) -> dict:
    """Parse every {param,value} annotation in ``question_annotated`` into a
    typed parameter dict, via ``_typed_parameter_value``."""
    return {
        match.group(1).strip(): _typed_parameter_value(
            match.group(2), resolve_verbal_multipliers=resolve_verbal_multipliers)
        for match in _PARAM_VALUE_RE.finditer(question_annotated)
    }


# ---------------------------------------------------------------------------
# Parameter serialisation / deserialisation
# ---------------------------------------------------------------------------
# Fraction values (e.g. Fraction(1, 3) for "third") are not JSON-serializable.
# We use a tagged dict {"__fraction__": [numerator, denominator]} so that
# round-tripping through JSONL is lossless — no float approximation.
# Both halves of the codec live here so the wire format is defined in one place.

_FRACTION_TAG = "__fraction__"


def serialize_parameters(params: dict) -> dict:
    """Encode a parameter dict for JSONL storage.

    ``Fraction`` values are encoded as ``{"__fraction__": [n, d]}``; all other
    value types (int, str) pass through unchanged.  The encoded form is fully
    JSON-serializable.
    """
    return {
        k: {"__fraction__": [v.numerator, v.denominator]} if isinstance(v, Fraction) else v
        for k, v in params.items()
    }


def deserialize_parameters(params: dict) -> dict:
    """Decode a parameter dict loaded from JSONL back to Python types.

    Tagged dicts ``{"__fraction__": [n, d]}`` are restored to exact
    ``Fraction(n, d)`` instances; all other values pass through unchanged.
    """
    out: dict = {}
    for k, v in params.items():
        if isinstance(v, dict) and _FRACTION_TAG in v:
            n, d = v[_FRACTION_TAG]
            out[k] = Fraction(n, d)
        else:
            out[k] = v
    return out
_SAFE_BUILTINS: dict = {"__builtins__": {}, "Fraction": Fraction, "int": int, "float": float}

# Exceptions a template's answer_function(**params) can legitimately raise
# when evaluated against untrusted, per-item, dataset-derived parameter
# values (eval() of a parsed arithmetic expression, e.g. division by a
# parameter that happens to be zero for this item, or a type mismatch
# between a Fraction and a captured string). Expected, and exactly what
# "this item doesn't fit the template, skip it" is meant to absorb —
# anything else (a real bug in this module) still propagates.
TEMPLATE_EVALUATION_FAILURE_EXCEPTIONS = (
    TypeError, ValueError, ArithmeticError, NameError, KeyError, AttributeError)


def _build_param_type_map(question_annotated: str) -> dict[str, str]:
    """Return {param_name: 'int' | 'str'} from the {param,default} pairs in
    question_annotated.

    'int' means the default is a digit string; 'str' covers everything else
    (names, fraction words, verbal multipliers).  The type map drives the
    regex capture-group pattern in extract_instance_parameters: integer params
    use \\d+ patterns; string params use greedy word/phrase patterns that are
    converted through FRACTION_WORDS and VERBAL_MULTIPLIER_WORDS at call time.
    """
    type_map: dict[str, str] = {}
    for m in _PARAM_VALUE_RE.finditer(question_annotated):
        name = m.group(1).strip()
        default = m.group(2).strip()
        type_map[name] = "int" if default.lstrip("-").isdigit() else "str"
    return type_map


def extract_instance_parameters(
        question_annotated: str,
        question_text: str,
        gold_answer: int,
) -> Optional[dict]:
    """Extract the parameter values for a specific GSM-Symbolic question instance.

    The template's question_annotated encodes the STRUCTURE (param names, types,
    answer formula) with DEFAULT parameter values from the template's base
    question.  This function uses the template's format string as a regex pattern
    matched against the actual HF question_text to recover the INSTANCE's true
    parameter values, then validates answer_function(**extracted) == gold_answer.

    Returns a typed parameter dict on success, None on extraction or validation
    failure.  Callers should store a successful result in item.parameters and
    leave it as {} otherwise; empty parameters cause Regime C to fail gracefully.

    Integer params are returned as int; fraction words (FRACTION_WORDS) and
    verbal multipliers (VERBAL_MULTIPLIER_WORDS, e.g. 'twice' → 2) are converted
    to their numeric equivalents; everything else is kept as str.

    The regex uses Python named capture groups (``(?P<name>...)``) so that
    repeated params become named backreferences (``(?P=name)``) without hitting
    the numbered-group limit.  String params use a non-greedy ``[^\\n.!?]+?``
    pattern: the next literal segment in the pattern acts as an anchor, so
    multi-word values like "three times" or currency-prefixed numbers like
    "$200" are captured correctly without an explicit lookahead.
    """
    # Build the template without validating its defaults against gold_answer —
    # the defaults belong to the base question, not this HF instance.
    parsed = parse_gsm_symbolic_template({"question_annotated": question_annotated})
    if parsed is None:
        return None

    param_type_map = _build_param_type_map(question_annotated)
    if not param_type_map:
        return None

    # Split the question_format on {param} tokens.
    # re.split with a capturing group yields alternating (literal, name) pairs:
    # [literal_0, name_0, literal_1, name_1, ..., literal_n]
    parts = re.split(r"\{(\w+)\}", parsed.question_format)
    seen_int: set[str] = set()    # int params with a named group already emitted
    str_counts: dict[str, int] = {}  # str params → occurrence count
    capture_order: list[str] = []
    regex_parts: list[str] = []

    for i, part in enumerate(parts):
        if i % 2 == 0:   # literal text between placeholders
            regex_parts.append(re.escape(part))
        else:             # parameter name
            name = part
            if param_type_map.get(name) == "int":
                if name in seen_int:
                    # Integer params: strict backreference ensures consistent value.
                    regex_parts.append(f"(?P={name})")
                else:
                    seen_int.add(name)
                    capture_order.append(name)
                    regex_parts.append(f"(?P<{name}>-?\\d[\\d,]*)")
            else:
                # String params: each occurrence gets an independent capture group
                # so that article differences ("an accountant" vs "the accountant")
                # don't cause spurious backreference failures.  Only the first
                # occurrence is used in the extracted dict.
                count = str_counts.get(name, 0)
                str_counts[name] = count + 1
                if count == 0:
                    capture_order.append(name)
                    regex_parts.append(f"(?P<{name}>[^\\n.!?]+?)")
                else:
                    # Unique group name avoids duplicate-name error in re module.
                    regex_parts.append(f"(?P<{name}_{count}>[^\\n.!?]+?)")

    pattern = "".join(regex_parts)
    try:
        m = re.match(pattern, question_text, re.DOTALL)
    except re.error:
        return None
    if m is None:
        return None

    extracted: dict = {}
    for name in capture_order:
        raw = m.group(name).strip(" ,.")
        if param_type_map.get(name) == "int":
            try:
                extracted[name] = int(raw.replace(",", ""))
            except ValueError:
                return None
        else:
            lower = raw.lower()
            if lower in FRACTION_WORDS:
                extracted[name] = FRACTION_WORDS[lower]
            elif lower in VERBAL_MULTIPLIER_WORDS:
                extracted[name] = VERBAL_MULTIPLIER_WORDS[lower]
            else:
                extracted[name] = raw

    # Validate: the extracted values must reproduce the HF gold answer.
    try:
        computed = parsed.answer_function(**extracted)
        if int(float(computed)) != int(float(gold_answer)):
            return None
    except TEMPLATE_EVALUATION_FAILURE_EXCEPTIONS:
        return None

    return extracted


def parse_gsm_symbolic_template(record: dict) -> Optional["ReasoningTemplate"]:
    """Parse the ``question_annotated`` field of a GSM-Symbolic record into a
    ``ReasoningTemplate``, or return None if parsing fails.

    The ``question_annotated`` format (Mirzadeh et al., ICLR 2025):

        …fog bank takes {t,10} minutes… every {d,3} miles… city is {y,42} miles
        …cover {frac,half} of the city?

        #init:
        - $t = range(25, 120)
        …

        #answer: (y*frac)//d*t

    This function:
    1. Extracts ``{param,value}`` pairs → ``parameters`` dict.
       Values are typed as: int (digit string), Fraction (FRACTION_WORDS
       lookup), or str (names / other words).
    2. Extracts the ``#answer:`` expression and builds a sandboxed
       ``answer_function(**kw)``.
    3. Validates: calling ``answer_function(**parameters)`` must equal the
       ``gold_answer`` in the record (within int tolerance).  Mismatches are
       logged and the item is skipped (no Regime C row, exclusion sidecar).

    Only records with a non-empty ``question_annotated`` field are processed;
    all others return None and fall back to no Regime C reasoning.
    """
    question_annotated = record.get("question_annotated") or ""
    if not question_annotated.strip():
        return None

    # Step 1: extract {param, value} pairs.
    parameters = _extract_annotated_parameters(question_annotated)
    if not parameters:
        return None

    # Step 2: extract #answer: expression.
    answer_match = _ANSWER_SECTION_RE.search(question_annotated)
    if not answer_match:
        return None
    answer_expr = answer_match.group(1).strip()
    if not answer_expr:
        return None

    # Build sandboxed answer function.  The expression may use only the
    # parameter names and arithmetic operators; Fraction is available so
    # expressions like (y*frac)//d*t work when frac is a Fraction instance.
    def _make_answer_function(expr: str) -> Callable:
        def answer_function(**kw):
            return eval(expr, dict(_SAFE_BUILTINS), kw)  # noqa: S307
        return answer_function

    answer_function = _make_answer_function(answer_expr)

    # Step 3: validate — computed gold must match stored gold.
    gold_answer = record.get("gold_answer")
    if gold_answer is not None:
        try:
            computed = answer_function(**parameters)
            # Accept integer-equivalent results (e.g. Fraction(140,1) == 140).
            if int(float(computed)) != int(float(gold_answer)):
                return None  # annotation mismatch; skip safely
        except TEMPLATE_EVALUATION_FAILURE_EXCEPTIONS:
            return None  # expression not evaluable with these parameters

    # Build question_format: replace {param,value} with {param}.
    question_format = _PARAM_VALUE_RE.sub(
        lambda m: "{" + m.group(1) + "}", question_annotated.split("\n\n#")[0])

    template_id = f"gsm_symbolic_{record.get('id_orig', record.get('task_id', 'unknown'))}"

    return ReasoningTemplate(
        template_id=template_id,
        question_format=question_format,
        answer_function=answer_function,
        operand_ranges={},       # not needed for Regime C swap
        operand_constraint=None,
    )


@dataclass(frozen=True)
class ReasoningTemplate:
    """A symbolic reasoning template: a format string with {name} and numeric
    operand placeholders, an answer function that computes the gold from the
    operands, the inclusive integer ranges each operand is sampled from, and an
    optional constraint the sampled operands must satisfy (for example, that the
    result is a positive integer)."""
    template_id: str
    question_format: str
    answer_function: Callable[..., float]
    operand_ranges: dict
    operand_constraint: Optional[Callable[..., bool]] = None


@dataclass
class ReasoningItem:
    """One reasoning task item, from either the official or synthetic source.

    ``template`` and ``parameters`` are populated only for synthetic items (the
    official items do not expose a Python answer function). Regime C's operand
    swap therefore applies only to synthetic items; the official items are used
    in Regimes A and B and as the contamination-controlled clean baseline. This
    matches design/04 §4.7, which notes the operand-swap control needs the
    template's answer function.
    """
    task_id: str
    task_family: TaskFamily
    source: TaskFamily                # GSM_SYMBOLIC_OFFICIAL | GSM_SYMBOLIC_SYNTHETIC
    question_text: str
    instruction: str
    gold_answer: int
    key_terms: list[str] = field(default_factory=list)

    template: Optional[ReasoningTemplate] = None
    parameters: dict = field(default_factory=dict)
    question_annotated: Optional[str] = None  # raw GSM-Symbolic annotation string; serialized to JSONL
    id_orig: Optional[int] = None             # original_id from apple/GSM-Symbolic; build-time only

    @property
    def full_prompt(self) -> str:
        return build_full_prompt(self.instruction, self.question_text)

    @property
    def scope_spans(self) -> dict:
        """Character spans of the instruction and content regions within the
        full prompt, for scope-restricted perturbation (design/03 §3.2)."""
        return build_instruction_and_content_scope_spans(self.instruction, self.question_text)

    @property
    def supports_regime_c_operand_swap(self) -> bool:
        """Regime C's reasoning operand swap needs the template's answer
        function, which only synthetic items carry."""
        return self.template is not None and bool(self.parameters)


# ---------------------------------------------------------------------------
# Synthetic generator (GSM-Symbolic-style; clearly labeled as synthetic).
# ---------------------------------------------------------------------------

SYNTHETIC_REASONING_TEMPLATES: tuple[ReasoningTemplate, ...] = (
    ReasoningTemplate(
        "buy_each",
        "{name} buys {a} boxes of pencils. Each box contains {b} pencils. "
        "{name} then gives away {c} pencils. How many pencils does {name} have left?",
        lambda a, b, c: a * b - c,
        {"a": (3, 12), "b": (4, 15), "c": (2, 20)},
        lambda a, b, c: a * b - c > 0,
    ),
    ReasoningTemplate(
        "save_weeks",
        "{name} saves {a} dollars every week for {b} weeks, then spends {c} dollars. "
        "How many dollars does {name} have remaining?",
        lambda a, b, c: a * b - c,
        {"a": (5, 20), "b": (3, 10), "c": (5, 40)},
        lambda a, b, c: a * b - c > 0,
    ),
    ReasoningTemplate(
        "read_pages",
        "A book has {a} pages. {name} reads {b} pages each day. "
        "After {c} days, how many pages remain unread?",
        lambda a, b, c: a - b * c,
        {"a": (80, 300), "b": (5, 25), "c": (2, 8)},
        lambda a, b, c: a - b * c > 0,
    ),
    ReasoningTemplate(
        "bake_trays",
        "{name} bakes {a} trays of muffins with {b} muffins per tray, "
        "then bakes {c} more muffins. How many muffins are there altogether?",
        lambda a, b, c: a * b + c,
        {"a": (2, 9), "b": (6, 12), "c": (1, 15)},
    ),
    ReasoningTemplate(
        "split_total",
        "{name} has {a} marbles and divides them equally among {b} friends. "
        "How many marbles does each friend receive?",
        lambda a, b: a // b,
        {"a": (12, 144), "b": (2, 12)},
        lambda a, b: a % b == 0,
    ),
    ReasoningTemplate(
        "twice_plus",
        "{name} scores {a} points in the first game and twice that in the second game. "
        "How many points does {name} score in total?",
        lambda a: a + 2 * a,
        {"a": (4, 40)},
    ),
    ReasoningTemplate(
        "fence_cost",
        "A fence section costs {a} dollars. {name} buys {b} sections and pays "
        "a {c} dollar delivery fee. What is the total cost in dollars?",
        lambda a, b, c: a * b + c,
        {"a": (8, 30), "b": (3, 15), "c": (5, 25)},
    ),
    ReasoningTemplate(
        "apples_left",
        "{name} picks {a} apples. {name} uses {b} apples for a pie and sells {c} apples. "
        "How many apples are left?",
        lambda a, b, c: a - b - c,
        {"a": (20, 90), "b": (3, 15), "c": (2, 20)},
        lambda a, b, c: a - b - c > 0,
    ),
)


def generate_synthetic_reasoning_items(
        item_count: int,
        seed: int,
        task_family: TaskFamily = TaskFamily.GSM_SYMBOLIC_SYNTHETIC,
) -> list[ReasoningItem]:
    """Generate ``item_count`` fresh, contamination-free reasoning items from the
    synthetic templates. Deterministic given the seed. Cycles through the
    templates so the mix is balanced.

    Because the operands are known by construction, the key terms (operand
    digit strings, operation words present in the text, and the name) are known
    exactly — no parsing required.
    """
    random_generator = random.Random(seed)
    items: list[ReasoningItem] = []
    template_index = 0

    while len(items) < item_count:
        template = SYNTHETIC_REASONING_TEMPLATES[template_index % len(SYNTHETIC_REASONING_TEMPLATES)]

        sampled_parameters = None
        for _ in range(_PARAMETER_SAMPLING_MAX_ATTEMPTS):
            candidate_parameters = {
                operand: random_generator.randint(low, high)
                for operand, (low, high) in template.operand_ranges.items()
            }
            constraint_satisfied = (
                template.operand_constraint is None
                or template.operand_constraint(**candidate_parameters)
            )
            if not constraint_satisfied:
                continue
            candidate_gold = template.answer_function(**candidate_parameters)
            if candidate_gold == int(candidate_gold) and candidate_gold >= 0:
                sampled_parameters = candidate_parameters
                break

        if sampled_parameters is None:
            template_index += 1
            continue

        name = random_generator.choice(SYNTHETIC_NAMES)
        question_text = template.question_format.format(name=name, **sampled_parameters)
        gold_answer = int(template.answer_function(**sampled_parameters))

        key_terms = (
            [str(value) for value in sampled_parameters.values()]
            + [word for word in OPERATION_WORDS if f" {word}" in question_text]
            + [name]
        )

        items.append(ReasoningItem(
            task_id=f"{task_family}_{len(items):05d}",
            task_family=task_family,
            source=TaskFamily.GSM_SYMBOLIC_SYNTHETIC,
            question_text=question_text,
            instruction=REASONING_INSTRUCTION,
            gold_answer=gold_answer,
            key_terms=key_terms,
            template=template,
            parameters=sampled_parameters,
        ))
        template_index += 1

    return items


def _retag_legacy_gsm_symbolic(raw_value: str) -> TaskFamily:
    """Re-tag the legacy ``"gsm_symbolic"`` task-family string produced by early
    versions of load_official_gsm_symbolic to the current canonical value
    ``TaskFamily.GSM_SYMBOLIC_OFFICIAL``.

    This shim allows existing pinned JSONL files that pre-date the rename to
    load and score correctly without a re-fetch from HuggingFace.  All new
    exports write ``"gsm_symbolic_official"``.
    """
    if raw_value == TaskFamily.GSM_SYMBOLIC:  # "gsm_symbolic" — the old tag
        return TaskFamily.GSM_SYMBOLIC_OFFICIAL
    return TaskFamily(raw_value)


def load_reasoning_jsonl(
        path: Path,
        task_family: TaskFamily = TaskFamily.GSM_SYMBOLIC_OFFICIAL,
        item_count: Optional[int] = None,
) -> list[ReasoningItem]:
    """Load reasoning items from a JSONL file produced by tools/build_task_items.py.

    The file is a pre-fetched, SHA-pinned subsample; no network access is required
    during a run. The registry's ``call_loader`` passes ``task_family`` from the
    spec so items lacking a per-record field are tagged correctly.

    Parameters
    ----------
    path :
        Path to the JSONL file (one JSON object per line).
    task_family :
        Default ``task_family`` for items that lack a ``task_family`` field in the
        JSONL record. The per-record value takes precedence when present.
    item_count :
        If given, return at most this many items.
    """
    items: list[ReasoningItem] = []

    for line_index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)

        # Backward-compatibility shim: early versions of load_official_gsm_symbolic
        # wrote task_family = "gsm_symbolic" (the historical TaskFamily.GSM_SYMBOLIC
        # value) rather than the current "gsm_symbolic_official".  Re-tag on load so
        # existing JSONL files score correctly without requiring a re-fetch.
        resolved_task_family = _retag_legacy_gsm_symbolic(
            record.get("task_family", str(task_family)))
        resolved_source = _retag_legacy_gsm_symbolic(
            record.get("source", str(task_family)))

        # Attempt to parse a GSM-Symbolic symbolic template when the record
        # carries a question_annotated field (Workstream 7).  Parsed templates
        # enable Regime C operand-swap for official GSM-Symbolic items.  Items
        # without a valid template fall back to template=None (Regime C skipped,
        # logged to exclusion sidecar).
        template: Optional[ReasoningTemplate] = None
        jsonl_parameters: dict = deserialize_parameters(record.get("parameters") or {})
        parameters: dict = {}
        if record.get("question_annotated"):
            # Parse template structure without validating template defaults against
            # gold_answer.  The JSONL's parameters field holds instance values
            # (validated at build time by extract_instance_parameters), which may
            # differ from the template defaults embedded in question_annotated.
            parse_record = {**record, "gold_answer": None}
            parsed_template = parse_gsm_symbolic_template(parse_record)
            if parsed_template is not None:
                template = parsed_template
                if jsonl_parameters:
                    # Instance values extracted and validated at build time — use
                    # them directly; they correspond to this specific HF question.
                    parameters = jsonl_parameters
                else:
                    # Fallback: derive parameters from template defaults.  These
                    # are the base-question values and may not match this instance;
                    # Regime C will validate at operand-swap time and exclude items
                    # where the defaults do not reproduce gold_answer.
                    parameters = _extract_annotated_parameters(
                        record["question_annotated"], resolve_verbal_multipliers=True)

        items.append(ReasoningItem(
            task_id=record.get("task_id", f"{task_family}_{line_index:05d}"),
            task_family=resolved_task_family,
            source=resolved_source,
            question_text=record["question_text"],
            instruction=record.get("instruction", REASONING_INSTRUCTION),
            gold_answer=record["gold_answer"],
            key_terms=record.get("key_terms", []),
            template=template,
            parameters=parameters,
        ))

    if item_count is not None:
        items = items[:item_count]

    return items


# ---------------------------------------------------------------------------
# Official HF fetchers (network; called once by tools/build_task_items.py).
# The ``datasets`` import is lazy so the offline test suite stays network-free.
# ---------------------------------------------------------------------------

def _parse_gsm_answer(answer_text: str) -> Optional[int]:
    """Extract the integer after ``#### `` in a GSM-style answer string.

    Uses the shared HASH_DELIMITED_ANSWER_PATTERN from tasks._shared so the
    gold-side and generation-side parsers both recognise the same surface forms
    (handles optional sign, optional dollar prefix, comma separators, and decimal
    points — the broader pattern is a superset of what GSM gold records contain,
    so matching behaviour is identical for valid gold strings).
    """
    match = HASH_DELIMITED_ANSWER_PATTERN.search(answer_text)
    if not match:
        return None
    try:
        return int(float(match.group(1).replace(",", "").replace("$", "")))
    except ValueError:
        return None


def _to_int(value) -> Optional[int]:
    """Cast a value to int, returning None if it is None or unconvertible."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _fetch_gsm_from_hf(
        hf_repo: str,
        hf_config: str,
        dataset_revision: Optional[str],
        item_count: int,
        seed: int,
        task_family: TaskFamily,
        include_annotated: bool = False,
) -> list[ReasoningItem]:
    """Shared HF-fetch helper for GSM-style datasets (one question + #### answer).

    ``include_annotated``: when True, pass through ``question_annotated`` and
    ``answer_annotated`` fields present in GSM-Symbolic p1/p2 splits.  These
    enable the GSM-Symbolic template parser (Workstream 7) at load time.
    """
    try:
        from datasets import load_dataset as _load_dataset
    except ImportError as error:
        raise ImportError(
            "fetching official datasets requires the 'datasets' package "
            "(pip install -r requirements.txt)") from error

    dataset = _load_dataset(hf_repo, hf_config, revision=dataset_revision, split="test")
    records = list(dataset)
    random.Random(seed).shuffle(records)

    items: list[ReasoningItem] = []

    for record_index, record in enumerate(records):
        if len(items) >= item_count:
            break
        gold = _parse_gsm_answer(record["answer"])
        if gold is None:
            continue

        # For GSM-Symbolic splits that carry symbolic annotations, attempt to
        # parse the template immediately so Regime C is available at run time.
        template: Optional[ReasoningTemplate] = None
        parameters: dict = {}
        if include_annotated and record.get("question_annotated"):
            augmented_record = {
                "question_annotated": record["question_annotated"],
                "answer_annotated": record.get("answer_annotated"),
                "gold_answer": gold,
                "task_id": f"{task_family}_{record_index:05d}",
                "id_orig": record.get("id_orig", record_index),
            }
            parsed = parse_gsm_symbolic_template(augmented_record)
            if parsed is not None:
                template = parsed
                # Re-extract typed parameters from the annotation.
                parameters = _extract_annotated_parameters(record["question_annotated"])

        items.append(ReasoningItem(
            task_id=f"{task_family}_{record_index:05d}",
            task_family=task_family,
            source=task_family,
            question_text=record["question"],
            instruction=REASONING_INSTRUCTION,
            gold_answer=gold,
            key_terms=[],  # frozen by tools/build_annotated_dataset.py before a run
            template=template,
            parameters=parameters,
            question_annotated=(record.get("question_annotated") if include_annotated else None),
            # original_id links this instance back to its Apple template file.
            # The HF apple/GSM-Symbolic dataset exposes this field as a string
            # (e.g. '473'); template files store id_orig as int.  Cast here so
            # _enrich_gsm_items_with_apple_templates can do an exact dict lookup.
            id_orig=_to_int(record.get("original_id") or record.get("id_orig")),
        ))
    return items


def load_official_gsm_symbolic(
        configuration_name: str,
        dataset_revision: Optional[str],
        item_count: int,
        seed: int,
) -> list[ReasoningItem]:
    """Fetch GSM-Symbolic items from HuggingFace (apple/GSM-Symbolic).

    Called once by tools/build_task_items.py; requires network access.
    ``configuration_name`` selects the difficulty variant (``"main"``, ``"p1"``,
    or ``"p2"``). Items are shuffled deterministically with ``seed`` before
    subsampling so the exported JSONL is reproducible.
    """
    # include_annotated=True: p1 and p2 splits carry question_annotated fields
    # that enable Regime C operand-swap (Workstream 7).  The "main" split may
    # not have them; include_annotated is harmless when the field is absent.
    return _fetch_gsm_from_hf(
        "apple/GSM-Symbolic", configuration_name, dataset_revision,
        item_count, seed, TaskFamily.GSM_SYMBOLIC_OFFICIAL,
        include_annotated=True)


def load_official_gsm8k(
        dataset_revision: Optional[str],
        item_count: int,
        seed: int,
) -> list[ReasoningItem]:
    """Fetch GSM8K items from HuggingFace (openai/gsm8k, config ``main``).

    Called once by tools/build_task_items.py; requires network access.
    Used as the contamination-contrast partner for GSM-Symbolic: running the same
    perturbations on familiar memorised items versus fresh items reveals whether
    degradation is driven by content knowledge or surface-form brittleness.
    """
    return _fetch_gsm_from_hf(
        "openai/gsm8k", "main", dataset_revision,
        item_count, seed, TaskFamily.GSM8K)
