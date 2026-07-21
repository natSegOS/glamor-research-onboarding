"""Task items and answer scoring, consolidated: the (str, Enum) contract every
JSONL/config consumer depends on, the frozen prompt scaffold, the JSONL
loaders' coercion shims, both extraction-tier priority ladders, the frozen
scoring rules with their dual-accounting guarantee, and the spaCy-free
linguistic parse-status classifier.

Consolidates test_enums.py, test_tasks.py, test_answer_scoring.py, and
test_linguistic_parse_status.py, keeping their strongest invariants.
"""

from __future__ import annotations

import json

from fractions import Fraction

import pytest

import scoring
from enums import (
    ConditionSource,
    Decoding,
    ExtractionTier,
    FragmentationStratum,
    INTERACTIONAL_FAILURE_STATUSES,
    MCQ_FAMILIES,
    Operation,
    ParseStatus,
    Precision,
    REASONING_FAMILIES,
    Scope,
    SelectionPolicy,
    SemanticClass,
    TaskFamily,
    Unit,
)
from tasks import multiple_choice, reasoning
from tasks.reasoning import ReasoningItem, load_reasoning_jsonl

try:
    from tests.conftest import FakeTokenizer
except ImportError:                # the offline shim loads conftest as a top-level module
    from conftest import FakeTokenizer


_EVERY_STRING_ENUM_CLASS = [
    Operation, SelectionPolicy, Scope, Unit, SemanticClass, TaskFamily,
    ParseStatus, FragmentationStratum, ConditionSource, Precision, Decoding,
    ExtractionTier,
]


# ---------------------------------------------------------------------------
# Enum contract.  Breaking it silently corrupts every JSONL row and config
# lookup, because enum members would stop serialising as plain strings.
# ---------------------------------------------------------------------------

class TestStringEnumContract:

    @pytest.mark.parametrize(
        "enum_class", _EVERY_STRING_ENUM_CLASS, ids=lambda cls: cls.__name__)
    def test_every_member_round_trips_through_its_plain_string(self, enum_class):
        """Breaking this means enum members no longer serialise/compare as
        their plain string, corrupting every JSONL row and config lookup."""
        for member in enum_class:
            assert str(member) == member.value
            assert member == member.value
            assert json.dumps(member) == json.dumps(member.value)
            assert enum_class(member.value) is member

    @pytest.mark.parametrize(
        "enum_class", _EVERY_STRING_ENUM_CLASS, ids=lambda cls: cls.__name__)
    def test_an_unrecognised_value_raises_instead_of_minting_a_member(self, enum_class):
        """Breaking this means a typo'd config value would pass silently
        instead of failing loudly at parse time."""
        with pytest.raises((ValueError, KeyError)):
            enum_class("__not_a_real_value__")

    def test_parse_status_is_exactly_the_four_way_taxonomy(self):
        """Breaking this means the pre-registered four-way parse taxonomy
        (design/04 §4.5) has drifted from what analysis assumes."""
        assert set(ParseStatus) == {
            ParseStatus.VALID, ParseStatus.UNPARSEABLE,
            ParseStatus.CLARIFICATION, ParseStatus.REFUSAL}

    def test_interactional_failure_statuses_exclude_only_valid(self):
        """Breaking this either double-counts VALID as a failure or lets an
        interactional failure escape the dual-accounting rule."""
        assert INTERACTIONAL_FAILURE_STATUSES == set(ParseStatus) - {ParseStatus.VALID}

    def test_selection_policy_vocabulary_includes_homophone_and_whitespace(self):
        """Breaking this means the HIVE-crosswalk conditions (homophone,
        missed-space) can no longer be expressed in a config."""
        assert SelectionPolicy.HOMOPHONE in SelectionPolicy
        assert SelectionPolicy.WHITESPACE in SelectionPolicy

    def test_reasoning_and_mcq_families_are_disjoint_and_route_every_scorer(self):
        """Breaking this means some TaskFamily has no scorer, or one family is
        scored by both scorers depending on dict iteration order."""
        assert REASONING_FAMILIES.isdisjoint(MCQ_FAMILIES)
        assert REASONING_FAMILIES == {
            TaskFamily.GSM_SYMBOLIC_OFFICIAL, TaskFamily.GSM_SYMBOLIC_SYNTHETIC,
            TaskFamily.GSM8K}
        assert MCQ_FAMILIES == {TaskFamily.MMLU_PRO, TaskFamily.MMLU, TaskFamily.MCQ_DEMO}


# ---------------------------------------------------------------------------
# Prompt scaffold.  Breaking any of these silently changes what every model
# sees, invalidating the measured 0.99 compliance and the scope-span geometry
# the perturbation engine relies on.
# ---------------------------------------------------------------------------

def _one_synthetic_reasoning_item() -> ReasoningItem:
    return reasoning.generate_synthetic_reasoning_items(1, seed=4)[0]


class TestPromptScaffold:

    def test_every_exemplar_chat_turn_repeats_the_instruction_scaffold(self):
        """Breaking this means exemplar turns and the real turn are scaffolded
        differently, undoing the measured 0.99 compliance of the 8-shot form."""
        assert reasoning.REASONING_CHAT_EXEMPLAR_TURNS
        for user_message, assistant_message in reasoning.REASONING_CHAT_EXEMPLAR_TURNS:
            assert user_message.startswith(reasoning.REASONING_INSTRUCTION)
            assert assistant_message.rstrip().splitlines()[-1].startswith("####")

    def test_reminder_is_appended_after_the_question_content(self):
        """Breaking this loses the +11pp compliance the recency-position
        reminder was measured to buy."""
        item = _one_synthetic_reasoning_item()
        assert item.full_prompt.endswith(reasoning.REASONING_INSTRUCTION_REMINDER)
        assert (item.full_prompt.index(item.question_text)
                < item.full_prompt.rindex(reasoning.REASONING_INSTRUCTION_REMINDER))

    def test_exemplar_problems_never_leak_into_the_items_own_prompt(self):
        """Breaking this would let a hand-written exemplar shadow the real
        question inside a single prompt: a contamination bug."""
        item = _one_synthetic_reasoning_item()
        for exemplar_problem, _solution in reasoning.REASONING_CHAT_EXEMPLARS:
            assert exemplar_problem not in item.full_prompt

    def test_reasoning_scope_spans_slice_exactly_to_instruction_and_content(self):
        """Breaking this misaligns every scope-restricted perturbation: edits
        aimed at 'content' would land on scaffold text or out of bounds."""
        for item in reasoning.generate_synthetic_reasoning_items(5, seed=6):
            spans = item.scope_spans
            instruction_start, instruction_end = spans[str(Scope.INSTRUCTION)]
            content_start, content_end = spans[str(Scope.CONTENT)]
            assert item.full_prompt[instruction_start:instruction_end] == item.instruction
            assert item.full_prompt[content_start:content_end] == item.question_text
            assert instruction_end <= content_start

    def test_mcq_scope_spans_slice_exactly_to_the_content_text(self):
        """Same failure class as above, for the MCQ item type."""
        item = multiple_choice.make_demonstration_multiple_choice_items()[0]
        content_start, content_end = item.scope_spans[str(Scope.CONTENT)]
        assert item.full_prompt[content_start:content_end] == item.content_text


# ---------------------------------------------------------------------------
# JSONL loaders.  Breaking these silently re-tags or truncates the frozen,
# SHA-pinned datasets a run scores against.
# ---------------------------------------------------------------------------

def _write_reasoning_jsonl(tmp_path, records) -> "Path":
    jsonl_path = tmp_path / "reasoning.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return jsonl_path


def _minimal_reasoning_record(task_id: str, task_family: str = "gsm8k") -> dict:
    return {
        "task_id": task_id, "task_family": task_family, "source": task_family,
        "question_text": "Solve 2+2.", "instruction": "Show your work.",
        "gold_answer": 4, "key_terms": [], "parameters": {},
    }


class TestJsonlLoaders:

    @pytest.mark.parametrize("stored_task_family,expected", [
        ("gsm_symbolic", TaskFamily.GSM_SYMBOLIC_OFFICIAL),           # legacy tag, re-tagged
        ("gsm_symbolic_official", TaskFamily.GSM_SYMBOLIC_OFFICIAL),  # current tag, unchanged
        ("gsm8k", TaskFamily.GSM8K),                                  # unrelated tag, unchanged
    ], ids=["legacy_retagged", "current_tag_passthrough", "gsm8k_passthrough"])
    def test_legacy_gsm_symbolic_tag_is_retagged_on_load(
            self, tmp_path, stored_task_family, expected):
        """Breaking this means pre-rename JSONL files load with a family that
        is in neither scorer's routing set, so every row scores as an error."""
        jsonl_path = _write_reasoning_jsonl(
            tmp_path, [_minimal_reasoning_record("t0", stored_task_family)])
        items = load_reasoning_jsonl(jsonl_path)
        assert items[0].task_family == expected
        assert isinstance(items[0].task_family, TaskFamily)

    def test_item_count_truncates_to_exactly_the_requested_prefix(self, tmp_path):
        """Breaking this means a pilot config asking for N items silently runs
        the whole file (cost) or an empty one (no data)."""
        records = [_minimal_reasoning_record(f"t{index}") for index in range(5)]
        jsonl_path = _write_reasoning_jsonl(tmp_path, records)
        loaded = load_reasoning_jsonl(jsonl_path, item_count=2)
        assert [item.task_id for item in loaded] == ["t0", "t1"]

    def test_multiple_choice_loader_preserves_options_and_coerces_the_family(self, tmp_path):
        """Breaking this scrambles option letters or leaves task_family a bare
        string, breaking frozenset scorer routing downstream."""
        record = {
            "question": "Which is the capital of France?",
            "options": {"A": "Paris", "B": "Berlin"},
            "answer": "A",
            "task_family": str(TaskFamily.MMLU_PRO),
            "category": "Geography",
        }
        jsonl_path = tmp_path / "mcq.jsonl"
        jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        items = multiple_choice.load_multiple_choice_jsonl(jsonl_path)
        assert len(items) == 1
        assert items[0].options == {"A": "Paris", "B": "Berlin"}
        assert items[0].gold_letter == "A"
        assert isinstance(items[0].task_family, TaskFamily)

    def test_option_sequence_to_letter_dict_covers_the_full_ten_letter_range(self):
        """Breaking this misassigns option letters, so every MCQ gold answer
        beyond the shifted point scores against the wrong option."""
        option_texts = [f"Option{index}" for index in range(10)]
        letter_dict = multiple_choice._options_sequence_to_letter_dict(option_texts)
        assert list(letter_dict.keys()) == list("ABCDEFGHIJ")
        assert letter_dict["A"] == "Option0"
        assert letter_dict["J"] == "Option9"


# ---------------------------------------------------------------------------
# Extraction-tier priority ladders.  Each row is a frozen, pre-registered
# business rule; a change to any of them un-freezes the scorer.
# ---------------------------------------------------------------------------

REASONING_EXTRACTION_CASES = [
    # (generation_text, expected_value, expected_tier, case_id)
    ("blah 5 blah\n#### 19", 19.0, ExtractionTier.HASH_DELIMITED,
     "hash_delimiter_beats_cot_numbers"),
    ("#### 5\n#### 19", 19.0, ExtractionTier.HASH_DELIMITED,
     "last_hash_delimiter_wins"),
    ("the answer is 7 then 19", 19.0, ExtractionTier.LAST_NUMBER_FALLBACK,
     "falls_back_to_last_number"),
    ("First 3 cats then 7 dogs appear", 7.0, ExtractionTier.LAST_NUMBER_FALLBACK,
     "adversarial_numbers_inside_cot_last_wins"),
    ("Total: $1,234", 1234.0, ExtractionTier.LAST_NUMBER_FALLBACK,
     "currency_and_commas_stripped"),
    ("#### $1,234.50", 1234.50, ExtractionTier.HASH_DELIMITED,
     "hash_with_currency_and_commas"),
    ("the mass is roughly 2e-3 grams", 0.002, ExtractionTier.LAST_NUMBER_FALLBACK,
     "scientific_notation"),
    ("#### -7", -7.0, ExtractionTier.HASH_DELIMITED, "negative_number"),
    ("#### 0", 0.0, ExtractionTier.HASH_DELIMITED, "zero_is_a_real_answer_not_a_miss"),
    ("no digits here", None, ExtractionTier.UNPARSEABLE, "no_number_returns_none"),
    ("", None, ExtractionTier.UNPARSEABLE, "adversarial_empty_string"),
    ("   \n\t  ", None, ExtractionTier.UNPARSEABLE, "adversarial_whitespace_only"),
]

MCQ_EXTRACTION_CASES = [
    # (generation_text, option_count, expected_letter, case_id)
    ("I think A but Answer: C", None, "C", "explicit_marker_beats_mentioned_letter"),
    ("Answer: A ... Answer: C", None, "C", "last_explicit_marker_wins"),
    ("B) is wrong\nAnswer: D", None, "D", "explicit_marker_beats_line_leading"),
    ("B) Photosynthesis", None, "B", "line_leading_letter_tier"),
    ("Option A costs 5 and option B costs 3. So the correct one is D", None, "D",
     "adversarial_letters_in_reasoning_chain_last_sentence_wins"),
    ("ANSWER: B", None, "B", "marker_case_insensitive_upper"),
    ("answer is c", None, "C", "marker_case_insensitive_lower"),
    ("The answer is J", 4, None, "letter_beyond_option_count_invalid"),
    ("Answer: J", 10, "J", "letter_at_max_option_count_valid"),
    ("Answer: K", None, None, "letter_outside_the_alphabet_always_invalid"),
    ("Answer: B", 1, None, "option_count_one_rejects_b"),
    ("no response provided at all", None, None, "no_valid_letter_returns_none"),
    ("", None, None, "adversarial_empty_string"),
]


class TestExtractionTierLadders:

    @pytest.mark.parametrize(
        "generation_text,expected_value,expected_tier",
        [(case[0], case[1], case[2]) for case in REASONING_EXTRACTION_CASES],
        ids=[case[3] for case in REASONING_EXTRACTION_CASES])
    def test_reasoning_ladder(self, generation_text, expected_value, expected_tier):
        """Breaking any row un-freezes a pre-registered extraction rule and
        changes which surface form a scored answer comes from."""
        parsed_value, extraction_tier = scoring.extract_reasoning_answer(generation_text)
        assert parsed_value == expected_value
        assert extraction_tier == expected_tier

    @pytest.mark.parametrize(
        "generation_text,option_count,expected_letter",
        [(case[0], case[1], case[2]) for case in MCQ_EXTRACTION_CASES],
        ids=[case[3] for case in MCQ_EXTRACTION_CASES])
    def test_multiple_choice_ladder(self, generation_text, option_count, expected_letter):
        """Same failure class for the MCQ ladder, including the option-count
        bound that keeps out-of-range letters from scoring."""
        keyword_arguments = {} if option_count is None else {"option_count": option_count}
        extracted_letter, _tier = scoring.extract_multiple_choice_answer(
            generation_text, **keyword_arguments)
        assert extracted_letter == expected_letter


# ---------------------------------------------------------------------------
# Scoring rules: exact match, tolerance, and dual accounting.
# ---------------------------------------------------------------------------

REASONING_SCORING_CASES = [
    # (generation_text, gold_answer, expected_is_correct, case_id)
    ("#### 19", 19, 1, "integer_exact_match"),
    ("#### 18", 19, 0, "integer_mismatch"),
    ("#### 19.5", 19, 0, "integer_gold_rejects_fractional_match"),
    ("#### 3.1415930", 3.1415927, 1, "float_within_tolerance"),
    ("#### 3.142", 3.1415927, 0, "float_outside_tolerance"),
    ("#### -5", -5, 1, "negative_gold_correct"),
    ("#### 5", -5, 0, "negative_gold_sign_matters"),
]


class TestScoringRules:

    @pytest.mark.parametrize(
        "generation_text,gold_answer,expected_is_correct",
        [(case[0], case[1], case[2]) for case in REASONING_SCORING_CASES],
        ids=[case[3] for case in REASONING_SCORING_CASES])
    def test_integer_exact_match_and_float_tolerance(
            self, generation_text, gold_answer, expected_is_correct):
        """Breaking this changes the primary correctness endpoint: either
        near-misses start scoring correct or exact floats stop doing so."""
        assert scoring.score_reasoning(generation_text, gold_answer).is_correct == expected_is_correct

    @pytest.mark.parametrize("generation_text", [
        "Could you clarify what you mean?",
        "I cannot help with that.",
        "no parseable content at all",
    ], ids=["clarification_surface_form", "refusal_surface_form", "plain_noise"])
    def test_every_interactional_failure_scores_zero(self, generation_text):
        """Dual accounting: breaking this lets a non-answer count as correct,
        inflating accuracy exactly where the perturbation caused a failure."""
        result = scoring.score_reasoning(generation_text, 19)
        assert result.parse_status in INTERACTIONAL_FAILURE_STATUSES
        assert result.is_correct == 0
        assert result.parsed_answer is None

    def test_refusal_followed_by_an_answer_is_still_valid(self):
        """Breaking this makes the inline classifier lexical instead of
        structural: hedged-but-answered responses would stop counting."""
        result = scoring.score_reasoning("I won't, but if I had to: #### 19", 19)
        assert result.parse_status == ParseStatus.VALID
        assert result.is_correct == 1

    def test_scoring_dispatch_covers_every_task_family(self):
        """Breaking this means some family's rows are scored by the wrong
        scorer (numbers as letters or vice versa)."""
        for reasoning_family in REASONING_FAMILIES:
            assert scoring.score("#### 7", 7, reasoning_family).is_correct == 1
        for mcq_family in MCQ_FAMILIES:
            assert scoring.score("Answer: B", "B", mcq_family).is_correct == 1
        with pytest.raises(ValueError):
            scoring.score("#### 1", 1, "unknown_family_xyz")


# ---------------------------------------------------------------------------
# Falsy-zero regression.  gold_answer == 0 is falsy; any truthiness check on
# the gold path (`or`-chaining, `if gold:`) silently converts a legitimate
# zero into "missing" and scores the row against None.
# ---------------------------------------------------------------------------

def _reasoning_item_with_gold_zero() -> ReasoningItem:
    return ReasoningItem(
        task_id="gold_zero_item",
        task_family=TaskFamily.GSM_SYMBOLIC_SYNTHETIC,
        source=TaskFamily.GSM_SYMBOLIC_SYNTHETIC,
        question_text="Ava has 3 apples and gives away 3. How many remain?",
        instruction=reasoning.REASONING_INSTRUCTION,
        gold_answer=0,
        key_terms=["3"],
    )


class TestFalsyZeroGoldAnswer:

    def test_gold_zero_scores_a_hash_zero_generation_correct(self):
        """Breaking this scores every zero-gold item wrong, biasing accuracy
        on exactly the items whose answer is 0."""
        result = scoring.score_reasoning("The apples cancel out.\n#### 0", 0)
        assert result.parse_status == ParseStatus.VALID
        assert result.is_correct == 1

    def test_request_builder_preserves_a_zero_gold_answer(self, is_word):
        """Breaking this means the request builder's gold lookup treats 0 as
        missing (falsy), writing gold_answer=None into every zero-gold row."""
        from pipeline.experiment import _build_requests_for_item_slice

        requests, exclusion_records = _build_requests_for_item_slice(
            [_reasoning_item_with_gold_zero()],
            conditions=[], is_word=is_word, tokenizer=FakeTokenizer(), seed=1)

        assert exclusion_records == []
        assert len(requests) == 1
        clean_request = requests[0]
        assert clean_request.is_clean
        assert clean_request.gold_answer == 0
        assert clean_request.gold_answer is not None
        assert scoring.score_reasoning("#### 0", clean_request.gold_answer).is_correct == 1


# ---------------------------------------------------------------------------
# Synthetic-generator self-consistency and the GSM-Symbolic template
# machinery: the guarantees Regime C's gold recomputation stands on.
# ---------------------------------------------------------------------------

def _annotated_question(question_part: str, answer_expression: str) -> str:
    return f"{question_part}\n\n#answer: {answer_expression}"


class TestSyntheticGeneratorSelfConsistency:

    def test_every_generated_item_is_deterministic_recomputable_and_targetable(self):
        """Breaking any clause invalidates Regime C: golds that don't match
        the template make operand-swapped rows score against wrong answers;
        duplicate IDs collide row IDs; operand values missing from key_terms
        break answer-critical targeting; non-determinism breaks resume."""
        generated_items = reasoning.generate_synthetic_reasoning_items(40, seed=7)
        regenerated_items = reasoning.generate_synthetic_reasoning_items(40, seed=7)

        assert ([item.question_text for item in generated_items]
                == [item.question_text for item in regenerated_items])
        assert len({item.task_id for item in generated_items}) == len(generated_items)
        for item in generated_items:
            assert int(item.template.answer_function(**item.parameters)) == item.gold_answer
            assert item.gold_answer >= 0
            assert item.supports_regime_c_operand_swap
            for operand_value in item.parameters.values():
                assert str(operand_value) in item.key_terms


class TestGsmSymbolicTemplateMachinery:

    _MULTIPLY_TEMPLATE = _annotated_question(
        "{name,Alice} buys {a,3} boxes of {b,4} each.", "a * b")

    def test_gold_validation_accepts_matching_and_rejects_mismatching_answers(self):
        """Breaking this either drops every valid template (no Regime C) or
        accepts mis-annotated ones (silently wrong recomputed golds)."""
        matching = reasoning.parse_gsm_symbolic_template(
            {"question_annotated": self._MULTIPLY_TEMPLATE, "gold_answer": 12})
        mismatching = reasoning.parse_gsm_symbolic_template(
            {"question_annotated": self._MULTIPLY_TEMPLATE, "gold_answer": 99})
        assert matching is not None
        assert matching.answer_function(a=3, b=4) == 12
        assert mismatching is None

    def test_sandboxed_answer_eval_blocks_builtins(self):
        """Adversarial: breaking this lets a dataset-supplied #answer
        expression execute arbitrary code (e.g. os.system) at load time."""
        hostile = _annotated_question("{n,3} items.", "__import__('os').system('echo hi')")
        template = reasoning.parse_gsm_symbolic_template(
            {"question_annotated": hostile, "gold_answer": None})
        assert template is not None       # sandbox fires at call time, not parse time
        with pytest.raises((NameError, TypeError)):
            template.answer_function(n=3)

    @pytest.mark.parametrize("annotated,instance_text,gold_answer,expected_parameters", [
        ("{name,Alice} buys {a,3} boxes of {b,4} each.\n\n#answer: a * b",
         "Carla buys 6 boxes of 8 each.", 48, {"name": "Carla", "a": 6, "b": 8}),
        ("{name,Alice} eats {frac,half} of {n,10} cookies.\n\n#answer: int(n * frac)",
         "Bob eats a quarter of 20 cookies.", 5,
         {"name": "Bob", "frac": Fraction(1, 4), "n": 20}),
        ("There are {n,6} girls. There are {mult,twice} as many boys.\n\n"
         "#answer: int(n * mult)",
         "There are 44 girls. There are three times as many boys.", 132,
         {"n": 44, "mult": 3}),
        ("{name,Alice} has {n,5} apples.\n\n#answer: n",
         "Bob has 12 apples.", 99, None),   # gold mismatch must return None
    ], ids=["integer_parameters", "fraction_word_converted",
            "verbal_multiplier_converted", "gold_mismatch_returns_none"])
    def test_instance_parameter_extraction(
            self, annotated, instance_text, gold_answer, expected_parameters):
        """Breaking this mis-extracts an HF instance's true operand values, so
        Regime C swaps operands the question never contained."""
        assert reasoning.extract_instance_parameters(
            annotated, instance_text, gold_answer=gold_answer) == expected_parameters

    @pytest.mark.parametrize("parameters", [
        {"a": 3, "b": 7},
        {"name": "Alice", "item": "apple"},
        {"n": 10, "frac": Fraction(2, 3), "name": "Bob"},
        {},
    ], ids=["ints", "strings", "mixed_int_fraction_string", "empty"])
    def test_parameter_codec_round_trip_is_lossless(self, parameters):
        """Breaking this float-approximates Fractions through JSONL, so
        reloaded items recompute golds off by rounding error."""
        assert reasoning.deserialize_parameters(
            reasoning.serialize_parameters(parameters)) == parameters

    def test_fraction_wire_format_is_the_tagged_pair(self):
        """Breaking this changes the frozen JSONL wire format under every
        already-exported dataset file."""
        wire_form = reasoning.serialize_parameters({"frac": Fraction(1, 3)})
        assert wire_form == {"frac": {"__fraction__": [1, 3]}}


# ---------------------------------------------------------------------------
# Linguistic parse-status classifier: the formal four-way detector, run
# offline against stub objects replicating the spaCy API surface it touches
# (the same mechanism the retired test_linguistic_parse_status.py used).
# ---------------------------------------------------------------------------

class _StubToken:
    def __init__(self, text: str, dep_: str = "", is_punct: bool = False,
                 is_space: bool = False, lemma_: str = ""):
        self.text = text
        self.dep_ = dep_
        self.is_punct = is_punct
        self.is_space = is_space
        self.lemma_ = lemma_ or text


class _StubDocument:
    def __init__(self, sentences):
        self._sentences = sentences

    @property
    def sents(self):
        return iter(self._sentences)


def _pipeline_for(*sentences):
    """A callable stub returning a pre-built document: the entire spaCy
    pipeline API surface the classifier uses."""
    document = _StubDocument([list(sentence) for sentence in sentences])
    return lambda _text: document


_QUESTION_MARK = _StubToken("?", is_punct=True)
_PERIOD = _StubToken(".", is_punct=True)


def _first_person_negated_sentence(*trailing_tokens):
    return [_StubToken("I", dep_="nsubj"), _StubToken("ca"),
            _StubToken("n't", dep_="neg"), _StubToken("help"), *trailing_tokens]


class TestLinguisticParseStatusClassifier:

    def test_a_parsed_answer_is_valid_and_never_invokes_the_pipeline(self):
        """Breaking this either misclassifies answered rows or silently spends
        a dependency parse on every one of them."""
        pipeline_calls = []

        def tracking_pipeline(text):
            pipeline_calls.append(text)
            return _StubDocument([])

        status = scoring.classify_parse_status_with_linguistic_pipeline(
            "The answer is 19.", "19.0", tracking_pipeline)
        assert status == ParseStatus.VALID
        assert pipeline_calls == []

    def test_interrogative_sentence_classifies_as_clarification(self):
        """Breaking this collapses clarifications into UNPARSEABLE and the ICR
        diagnostic loses its clarification component."""
        pipeline = _pipeline_for(
            [_StubToken("Could"), _StubToken("you"), _StubToken("clarify"), _QUESTION_MARK])
        assert scoring.classify_parse_status_with_linguistic_pipeline(
            "Could you clarify?", None, pipeline) == ParseStatus.CLARIFICATION

    def test_first_person_negation_classifies_as_refusal(self):
        """Breaking this collapses refusals into UNPARSEABLE: same ICR
        failure class, refusal component."""
        pipeline = _pipeline_for(_first_person_negated_sentence(_PERIOD))
        assert scoring.classify_parse_status_with_linguistic_pipeline(
            "I can't help.", None, pipeline) == ParseStatus.REFUSAL

    def test_no_structural_marker_and_no_answer_is_unparseable(self):
        """Breaking this over-detects: arbitrary noise would start counting as
        a clarification or refusal instead of the conservative default."""
        pipeline = _pipeline_for([_StubToken("Hmm"), _PERIOD])
        assert scoring.classify_parse_status_with_linguistic_pipeline(
            "Hmm.", None, pipeline) == ParseStatus.UNPARSEABLE

    def test_clarification_takes_precedence_over_refusal(self):
        """Breaking this reorders the taxonomy's tie-break, silently moving
        counts between the two ICR sub-categories."""
        pipeline = _pipeline_for(_first_person_negated_sentence(_QUESTION_MARK))
        assert scoring.classify_parse_status_with_linguistic_pipeline(
            "I can't help?", None, pipeline) == ParseStatus.CLARIFICATION

    def test_refusal_language_with_a_parsed_answer_is_still_valid(self):
        """Breaking this lets hedging language override an actual parsed
        answer, moving answered rows into the failure statuses."""
        pipeline = _pipeline_for(_first_person_negated_sentence())
        assert scoring.classify_parse_status_with_linguistic_pipeline(
            "I can't be sure, but #### 19", "19.0", pipeline) == ParseStatus.VALID
