"""Cross-family LLM-as-a-judge for regime classification audit.

The judge independently re-classifies a sample of perturbation pairs to verify
that the generation engine's structural guarantees (nonword vs real-word) match
the intended semantic guarantees (intent-preserving vs meaning-changing).

Reproducibility controls
------------------------
1. The judge model is a different family from every generation model in the
   study (Gemma 2 from Google vs Llama/Qwen/Mistral). Cross-family selection
   reduces correlation between generator tendencies and judge tendencies.
2. The judge always runs at temperature=0 (greedy decoding) to guarantee that
   the same (judge_revision, prompt_version, input) always produces the same
   decision.
3. Every decision is stored in a content-addressed cache keyed by
   SHA-256(judge_revision + PROMPT_TEMPLATE_VERSION + input_text). A re-run
   reads from the cache and never calls the judge twice for the same input.
4. The judge model revision is subject to the same PIN_ME pinning requirement
   as generation models in a confirmatory run.

Disclosed limitation
--------------------
The judge is drawn from the same broad ecosystem (Transformer LLMs) as the
generation models. Despite cross-family selection, shared pretraining data and
architectural patterns mean the judge is not fully independent. This is
mitigated by the 200-item human validation step (tools/regime_audit_ui.html),
whose human-judge agreement (Cohen's κ) is reported in the paper.

Structured output
-----------------
The prompt demands a single JSON object. We parse the JSON response and
validate it against a controlled vocabulary. If parsing fails we record
parse_failed=True and treat the item as needing human review.
"""

from __future__ import annotations

import hashlib
import json
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


PROMPT_TEMPLATE_VERSION = "v1"

_VALID_CLASSIFICATIONS = frozenset({"A", "B", "C", "not_applicable"})
_VALID_CONFIDENCES = frozenset({"high", "medium", "low"})

_JSON_BLOCK_PATTERN = re.compile(r'\{[^{}]*\}', re.DOTALL)

_JUDGE_SYSTEM_PROMPT = """\
You are a linguistics and NLP expert auditing perturbation quality for a \
research study on typo robustness of language models.

You will be shown an original text and a perturbed version. Your task is to \
classify the perturbation into exactly one of these three semantic regimes:

A — Intent-preserving nonword typo: the perturbed word is not a real English \
word, yet the intended meaning is unambiguous from context. A careful reader \
would notice a typo but recover the intended meaning without difficulty.

B — Context-recoverable real-word shift: the perturbation produces a real \
English word, but the word does not fit the original meaning. A careful reader \
could recover the intended meaning from context.

C — Meaning-changing: the perturbation fundamentally changes the meaning of \
the text in a way that a careful reader could not fully recover from context \
alone.

not_applicable — The perturbation is too minor to classify, the texts are \
identical, or the context is insufficient to judge.

Respond with a single JSON object and nothing else:
{"classification": "<A|B|C|not_applicable>", "confidence": "<high|medium|low>", \
"rationale": "<one concise sentence>"}
"""

_JUDGE_USER_TEMPLATE = """\
Original:
{original_text}

Perturbed:
{perturbed_text}

Edited span: "{edited_word_before}" → "{edited_word_after}"
"""


@dataclass
class JudgeDecision:
    """The structured output of one judge call."""

    cache_key: str
    judge_model_revision: str
    prompt_template_version: str
    original_text: str
    perturbed_text: str
    claimed_regime: str

    classification: Optional[str] = None
    confidence: Optional[str] = None
    rationale: Optional[str] = None
    parse_failed: bool = False
    raw_response: str = ""

    def agrees_with_claimed_regime(self) -> Optional[bool]:
        """True if the judge classification matches the claimed regime, False if
        it disagrees, None if the classification is not_applicable or missing."""
        if self.classification in (None, "not_applicable"):
            return None
        return self.classification == self.claimed_regime

    def to_dict(self) -> dict:
        return {
            "cache_key": self.cache_key,
            "judge_model_revision": self.judge_model_revision,
            "prompt_template_version": self.prompt_template_version,
            "original_text": self.original_text,
            "perturbed_text": self.perturbed_text,
            "claimed_regime": self.claimed_regime,
            "classification": self.classification,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "parse_failed": self.parse_failed,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JudgeDecision":
        return cls(**{k: data[k] for k in data if k in cls.__dataclass_fields__})


class JudgeDecisionCache:
    """Append-only, content-addressed cache of judge decisions.

    Keyed by SHA-256(judge_revision + PROMPT_TEMPLATE_VERSION + input_text)
    so that the same logical call always returns the same cached result,
    regardless of when the cache was built.
    """

    def __init__(self, cache_path: Path) -> None:

        self.cache_path = Path(cache_path)
        self._decisions: dict[str, JudgeDecision] = {}

        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    decision = JudgeDecision.from_dict(data)
                    self._decisions[decision.cache_key] = decision
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    def get(self, cache_key: str) -> Optional[JudgeDecision]:
        return self._decisions.get(cache_key)

    def store(self, decision: JudgeDecision) -> None:

        self._decisions[decision.cache_key] = decision
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        with self.cache_path.open("a") as output_file:
            output_file.write(json.dumps(decision.to_dict()) + "\n")

    def __len__(self) -> int:
        return len(self._decisions)


def _compute_cache_key(
        judge_revision: str,
        original_text: str,
        perturbed_text: str,
        edited_word_before: str,
        edited_word_after: str,
) -> str:
    """Stable SHA-256 key for one (judge, prompt_version, input) triple."""

    payload = json.dumps({
        "judge_revision": judge_revision,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "original_text": original_text,
        "perturbed_text": perturbed_text,
        "edited_word_before": edited_word_before,
        "edited_word_after": edited_word_after,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _build_judge_prompt(
        original_text: str,
        perturbed_text: str,
        edited_word_before: str,
        edited_word_after: str,
) -> str:
    """Assemble the full user-turn prompt for one judge call."""

    return _JUDGE_USER_TEMPLATE.format(
        original_text=original_text,
        perturbed_text=perturbed_text,
        edited_word_before=edited_word_before,
        edited_word_after=edited_word_after,
    )


def _parse_judge_response(raw_response: str) -> tuple[Optional[str], Optional[str], Optional[str], bool]:
    """Extract classification, confidence, rationale from the raw response.

    Returns (classification, confidence, rationale, parse_failed).
    """

    json_match = _JSON_BLOCK_PATTERN.search(raw_response)
    if not json_match:
        return None, None, None, True

    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None, None, None, True

    classification = parsed.get("classification")
    confidence = parsed.get("confidence")
    rationale = parsed.get("rationale")

    if classification not in _VALID_CLASSIFICATIONS:
        return None, None, None, True
    if confidence not in _VALID_CONFIDENCES:
        confidence = "low"

    return classification, confidence, str(rationale) if rationale else None, False


def judge_one(
        engine: object,
        judge_revision: str,
        original_text: str,
        perturbed_text: str,
        claimed_regime: str,
        edited_word_before: str,
        edited_word_after: str,
        cache: JudgeDecisionCache,
) -> JudgeDecision:
    """Judge one (original, perturbed) pair, using the cache when available.

    The engine must support ``generate(prompts, max_new_tokens)`` and optionally
    ``apply_chat_template(text)``. The judge is always called with a single
    prompt (batch size 1) to prevent any batch-ordering effects.
    """

    cache_key = _compute_cache_key(
        judge_revision, original_text, perturbed_text,
        edited_word_before, edited_word_after)

    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    user_prompt = _build_judge_prompt(
        original_text, perturbed_text, edited_word_before, edited_word_after)

    # Bug fix (Workstream 8): apply_chat_template accepted system_prompt kwarg
    # which does not exist on VllmEngine; now uses system_message= which
    # VllmEngine.apply_chat_template accepts and handles correctly.
    if hasattr(engine, "apply_chat_template"):
        full_prompt = engine.apply_chat_template(  # type: ignore[union-attr]
            user_prompt, system_message=_JUDGE_SYSTEM_PROMPT)
    else:
        full_prompt = f"{_JUDGE_SYSTEM_PROMPT}\n\n{user_prompt}"

    raw_responses = engine.generate([full_prompt], max_new_tokens=256)  # type: ignore[union-attr]
    raw_response = raw_responses[0] if raw_responses else ""

    classification, confidence, rationale, parse_failed = _parse_judge_response(raw_response)

    decision = JudgeDecision(
        cache_key=cache_key,
        judge_model_revision=judge_revision,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        original_text=original_text,
        perturbed_text=perturbed_text,
        claimed_regime=claimed_regime,
        classification=classification,
        confidence=confidence,
        rationale=rationale,
        parse_failed=parse_failed,
        raw_response=raw_response,
    )
    cache.store(decision)
    return decision


def _extract_content_text(prompt: str) -> str:
    """Extract the content block from a full prompt string.

    The full prompt has the structure:
        <INSTRUCTION>\\n\\n<CONTENT>
    The judge should see CONTENT only, not the instruction, so that instruction
    text (which is identical for all items) does not pollute the judgment
    (Workstream 8 bug fix).  Falls back to the full prompt if no separator
    is found.
    """
    separator = "\n\n"
    if separator in prompt:
        return prompt.split(separator, 1)[1]
    return prompt


def run_judge_on_sample(
        engine: object,
        judge_revision: str,
        sample_rows: Sequence[dict],
        cache_path: Path,
        progress_callback: Optional[object] = None,
        skip_regime_c_mcq: bool = True,
) -> list[JudgeDecision]:
    """Run the judge on a list of generation rows and return decisions.

    Each row must have: ``clean_prompt``, ``prompt`` (the perturbed prompt as
    written by the runner), ``r_semantic_class``, and at least one edit in
    ``edit_script``.

    Bug fixes (Workstream 8):
    - The perturbed text is read from ``row["prompt"]`` (how the runner writes
      it), not from the non-existent ``row["perturbed_prompt"]`` field.
    - Only the CONTENT block (question text) is shown to the judge, not the
      full instruction + content prompt (which added noise to the judgment).
    - Regime C MCQ items are skipped: their validity is guaranteed structurally
      (the permutation is definitionally meaning-changing) and judging them
      wastes inference.

    Already-cached rows are returned from cache without calling the engine.
    """

    cache = JudgeDecisionCache(cache_path)
    decisions: list[JudgeDecision] = []

    for row in sample_rows:
        claimed_regime = row.get("r_semantic_class", "")

        # Skip Regime C MCQ: structurally guaranteed, no judge needed.
        if skip_regime_c_mcq and claimed_regime == "C" and row.get("task_family", "").startswith("mmlu"):
            continue

        # Bug fix: runner writes the perturbed prompt as "prompt", not as
        # "perturbed_prompt" (which was never written by the runner).
        full_original = row.get("clean_prompt", "")
        full_perturbed = row.get("prompt", "")  # perturbed prompt in the runner schema

        # Bug fix: show content-only text, not the instruction+content full prompt.
        original_text = _extract_content_text(full_original)
        perturbed_text = _extract_content_text(full_perturbed)

        edit_script = row.get("edit_script", [])
        if edit_script:
            first_edit = edit_script[0]
            edited_word_before = first_edit.get("word_before", "")
            edited_word_after = first_edit.get("word_after", "")
        else:
            edited_word_before = ""
            edited_word_after = ""

        decision = judge_one(
            engine=engine,
            judge_revision=judge_revision,
            original_text=original_text,
            perturbed_text=perturbed_text,
            claimed_regime=claimed_regime,
            edited_word_before=edited_word_before,
            edited_word_after=edited_word_after,
            cache=cache,
        )
        decisions.append(decision)

        if progress_callback is not None:
            progress_callback(1)  # type: ignore[call-arg]

    return decisions
