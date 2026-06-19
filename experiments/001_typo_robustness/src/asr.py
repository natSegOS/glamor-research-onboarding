"""The ASR (automatic speech recognition) arm: clean text becomes audio via TTS,
optionally degraded with noise, then transcribed by Whisper. The difference
between the original text and the transcription is an ASR perturbation.

Provenance and the determinism fix
-----------------------------------
Whisper (Radford et al., 2022, arXiv:2212.04356) is the ASR engine. The CRITICAL
detail (verified against the official implementation, June 2026): Whisper's
``transcribe`` defaults its ``temperature`` to the tuple
(0.0, 0.2, 0.4, 0.6, 0.8, 1.0). With a tuple, the decoder ESCALATES temperature
on any segment that trips the log-probability or gzip-compression-ratio
threshold, which makes the transcription nondeterministic. Passing a SCALAR
``temperature=0.0`` forces argmax (greedy) decoding always and disables that
fallback, which our reproducibility guarantee requires. We also pass
``condition_on_previous_text=False`` so earlier segments cannot influence later
ones. See docs/PROVENANCE.md §3.1 and design/04 §4.7a.

Forcing temperature 0 has a known tradeoff: on a genuinely hard segment Whisper
can emit a repetition/hallucination loop instead of falling back. We therefore
flag degenerate transcriptions (high gzip compression ratio, or a repeated
n-gram run) and exclude them from the dataset rather than letting them enter
silently.

Edit budget is MEASURED, not preset
------------------------------------
ASR errors are item-specific, so the "edit budget" for an ASR item is the
measured Damerau-Levenshtein distance between the original and the
transcription, bucketed into bands {1-2, 3-5, 6+} to stay comparable with the
keyboard severity axis (design/03 §3.5a).

The audio/GPU dependencies (edge-tts, openai-whisper, soundfile) are imported
lazily inside the functions that need them, so this module imports on any
machine and the pure-Python diff/classify/flag logic is fully unit-testable
without audio.
"""

from __future__ import annotations

import gzip
import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from enums import Operation, SelectionPolicy, SemanticClass
from perturbation import damerau_levenshtein_distance


DEFAULT_TTS_VOICE = "en-US-AriaNeural"          # pinned single voice (design/03 §3.5a)
DEFAULT_WHISPER_MODEL = "large-v3"              # pinned; greedy decoding
DEFAULT_NOISE_SIGNAL_TO_NOISE_RATIO_DB = 10.0   # the noisy-ASR stress condition

EDIT_DISTANCE_BANDS = {"1-2": (1, 2), "3-5": (3, 5), "6+": (6, 10 ** 9)}


def edit_distance_band(distance: int) -> Optional[str]:
    """Return the severity band name for a measured edit distance, or None if
    the distance is zero (no error)."""
    for band_name, (low, high) in EDIT_DISTANCE_BANDS.items():
        if low <= distance <= high:
            return band_name
    return None


def normalize_text_for_word_diff(text: str) -> list[str]:
    """Lowercase word stream with punctuation stripped. ASR output formatting
    (capitalization, punctuation) is not the error signal we measure; word
    identity is."""
    return re.findall(r"[a-z0-9']+", text.lower())


@dataclass
class WordDiff:
    operation: Operation    # SUBSTITUTE | DELETE | INSERT
    original_word: str = ""
    hypothesis_word: str = ""


def word_level_diffs(original_text: str, hypothesis_text: str) -> list[WordDiff]:
    """Word-level alignment diffs between the original and the ASR hypothesis,
    computed by dynamic-programming edit alignment (substitution / deletion /
    insertion)."""
    original_words = normalize_text_for_word_diff(original_text)
    hypothesis_words = normalize_text_for_word_diff(hypothesis_text)

    original_length, hypothesis_length = len(original_words), len(hypothesis_words)
    distance = [[0] * (hypothesis_length + 1) for _ in range(original_length + 1)]

    for i in range(original_length + 1):
        distance[i][0] = i
    for j in range(hypothesis_length + 1):
        distance[0][j] = j

    for i in range(1, original_length + 1):
        for j in range(1, hypothesis_length + 1):
            substitution_cost = 0 if original_words[i - 1] == hypothesis_words[j - 1] else 1
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + substitution_cost,
            )

    diffs: list[WordDiff] = []
    i, j = original_length, hypothesis_length
    while i > 0 or j > 0:

        is_diagonal_step = (
            i > 0 and j > 0
            and distance[i][j] == distance[i - 1][j - 1] + (original_words[i - 1] != hypothesis_words[j - 1])
        )
        if is_diagonal_step:
            if original_words[i - 1] != hypothesis_words[j - 1]:
                diffs.append(WordDiff(Operation.SUBSTITUTE, original_words[i - 1], hypothesis_words[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            diffs.append(WordDiff(Operation.DELETE, original_word=original_words[i - 1]))
            i -= 1
        else:
            diffs.append(WordDiff(Operation.INSERT, hypothesis_word=hypothesis_words[j - 1]))
            j -= 1

    diffs.reverse()
    return diffs


def gzip_compression_ratio(text: str) -> float:
    """The gzip compression ratio of the text, used as Whisper's own degeneracy
    signal (a ratio above ~2.4 indicates repetitive/looping output)."""
    encoded = text.encode("utf-8")
    if not encoded:
        return 1.0
    compressed = gzip.compress(encoded)
    return len(encoded) / len(compressed)


def has_repeated_ngram_run(text: str, ngram_size: int = 3, max_repeats: int = 4) -> bool:
    """Detect a degenerate repetition loop: some word phrase repeated more than
    ``max_repeats`` times in immediate succession.

    Whisper loops take the form "<phrase> <phrase> <phrase> ...". We detect them
    by checking, for every phrase length from 1 up to ``ngram_size``, whether a
    phrase of that length repeats back-to-back more than ``max_repeats`` times.
    This catches single-word loops ("the the the ...") and multi-word loops
    ("thank you thank you ...") alike, regardless of stride alignment.
    """
    words = normalize_text_for_word_diff(text)

    for phrase_length in range(1, ngram_size + 1):
        consecutive_repeats = 1
        for start in range(phrase_length, len(words)):
            current_word = words[start]
            previous_word = words[start - phrase_length]
            if current_word == previous_word:
                consecutive_repeats += 1
                # consecutive_repeats counts matched positions; a phrase repeated
                # r times produces (r-1)*phrase_length matched positions.
                if consecutive_repeats >= max_repeats * phrase_length:
                    return True
            else:
                consecutive_repeats = 1

    return False


def flag_degenerate_transcription(transcription: str,
                                  compression_ratio_threshold: float = 2.4) -> bool:
    """Return True if a transcription looks degenerate (a repetition or
    hallucination loop). Such items are excluded from the dataset rather than
    entering it silently — the cost of forcing Whisper's temperature to 0
    (docs/PROVENANCE.md §3.1)."""
    if gzip_compression_ratio(transcription) > compression_ratio_threshold:
        return True
    if has_repeated_ngram_run(transcription):
        return True
    return False


@dataclass
class AsrItem:
    """One ASR perturbation: a clean text and its Whisper transcription, with
    the measured edit distance, severity band, word-level diffs, and a
    provisional regime classification (the human audit makes the final call)."""
    task_id: str
    clean_text: str
    transcription: str
    signal_to_noise_ratio_db: Optional[float]      # None = the quiet condition

    damerau_levenshtein_distance: int = 0
    band: Optional[str] = None
    word_diffs: list = field(default_factory=list)
    regime_candidate: SemanticClass = SemanticClass.B   # most ASR errors are real-word -> B
    is_degenerate: bool = False

    tts_voice: str = DEFAULT_TTS_VOICE
    whisper_model: str = DEFAULT_WHISPER_MODEL

    def classify(self, is_word: Callable[[str], bool]) -> None:
        """Populate the measured fields and the provisional regime candidacy.

        Regime candidacy (design/02 §2.4): if every substituted hypothesis word
        is a real word, the item is a Regime B candidate (context-recoverable
        real-word shift); if any substituted word is a nonword, it is a Regime A
        candidate. The human audit makes the final determination either way
        (design/09).
        """
        self.damerau_levenshtein_distance = damerau_levenshtein_distance(
            self.clean_text.lower(), self.transcription.lower())
        self.band = edit_distance_band(self.damerau_levenshtein_distance)
        self.word_diffs = word_level_diffs(self.clean_text, self.transcription)
        self.is_degenerate = flag_degenerate_transcription(self.transcription)

        substitutions = [diff for diff in self.word_diffs if diff.operation == Operation.SUBSTITUTE]
        any_substitution_is_a_nonword = (
            bool(substitutions)
            and any(not is_word(diff.hypothesis_word) for diff in substitutions)
        )
        self.regime_candidate = SemanticClass.A if any_substitution_is_a_nonword else SemanticClass.B

    @property
    def selection_policy(self) -> SelectionPolicy:
        """The selection-policy tag recorded in the output schema."""
        return SelectionPolicy.ASR_NOISY if self.signal_to_noise_ratio_db is not None else SelectionPolicy.ASR_CLEAN


# ---------------------------------------------------------------------------
# Audio side (guarded imports; runs once as a pre-processing step, design/07).
# ---------------------------------------------------------------------------

def synthesize_speech(text: str, output_wav_path: Path, voice: str = DEFAULT_TTS_VOICE) -> Path:
    """Synthesize ``text`` to a WAV file with edge-tts using a single fixed
    voice (design/03 §3.5a)."""
    import asyncio
    import edge_tts                    # guarded: only needed on the ASR machine

    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    async def _run_synthesis():
        await edge_tts.Communicate(text, voice).save(str(output_wav_path))

    asyncio.run(_run_synthesis())
    return output_wav_path


def add_background_noise(
        input_wav_path: Path,
        output_wav_path: Path,
        signal_to_noise_ratio_db: float = DEFAULT_NOISE_SIGNAL_TO_NOISE_RATIO_DB,
        babble_noise_wav_path: Optional[Path] = None,
        seed: int = 1729,
) -> Path:
    """Mix noise into the speech at a fixed signal-to-noise ratio. A real babble
    noise file is preferred for realism; in its absence we fall back to seeded
    Gaussian noise, which the paper documents as the approximation (design/03
    §3.5a)."""
    import numpy
    import soundfile                   # guarded: only needed on the ASR machine

    signal, sample_rate = soundfile.read(str(input_wav_path))

    if babble_noise_wav_path:
        noise, _ = soundfile.read(str(babble_noise_wav_path))
        repetitions_needed = int(numpy.ceil(len(signal) / len(noise)))
        noise = numpy.tile(noise, repetitions_needed)[:len(signal)]
    else:
        noise = numpy.random.default_rng(seed).standard_normal(len(signal))

    signal_power = float(numpy.mean(signal ** 2)) or 1e-12
    noise_power = float(numpy.mean(noise ** 2)) or 1e-12
    noise_scale = numpy.sqrt(signal_power / (noise_power * 10 ** (signal_to_noise_ratio_db / 10)))

    mixed = signal + noise_scale * noise

    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(output_wav_path), mixed, sample_rate)
    return output_wav_path


_LOADED_WHISPER_MODELS: dict = {}


def transcribe_audio(wav_path: Path, model_name: str = DEFAULT_WHISPER_MODEL) -> str:
    """Transcribe a WAV file with Whisper, DETERMINISTICALLY.

    The scalar ``temperature=0.0`` forces greedy decoding and disables Whisper's
    temperature fallback; ``condition_on_previous_text=False`` removes
    cross-segment drift. Together these make the transcription reproducible
    (docs/PROVENANCE.md §3.1). Models are cached so repeated calls do not reload
    weights.
    """
    import whisper                     # guarded: only needed on the ASR machine

    if model_name not in _LOADED_WHISPER_MODELS:
        _LOADED_WHISPER_MODELS[model_name] = whisper.load_model(model_name)
    model = _LOADED_WHISPER_MODELS[model_name]

    result = model.transcribe(
        str(wav_path),
        language="en",
        temperature=0.0,                       # scalar -> greedy, no fallback escalation
        condition_on_previous_text=False,      # no cross-segment drift
    )
    return result["text"].strip()


def build_asr_items(
        task_items,
        audio_directory: Path,
        is_word: Callable[[str], bool],
        signal_to_noise_ratio_levels=(None, DEFAULT_NOISE_SIGNAL_TO_NOISE_RATIO_DB),
        voice: str = DEFAULT_TTS_VOICE,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        babble_noise_wav_path: Optional[Path] = None,
        keep_degenerate: bool = False,
) -> list[AsrItem]:
    """One-time pre-processing pass: for every task item and every noise level,
    synthesize speech, optionally add noise, transcribe, and build an AsrItem.

    Audio is cached on disk so the step is idempotent and the WAVs can be
    released with the paper (design/04 §4.7a). Items whose transcription equals
    the original (no error) are dropped, and degenerate transcriptions are
    dropped unless ``keep_degenerate`` is set (they are always flagged either
    way).
    """
    audio_directory = Path(audio_directory)
    asr_items: list[AsrItem] = []

    for task_item in task_items:
        text = task_item.content_text if hasattr(task_item, "content_text") else task_item.question_text

        clean_wav_path = audio_directory / f"{task_item.task_id}.wav"
        if not clean_wav_path.exists():
            synthesize_speech(text, clean_wav_path, voice)

        for signal_to_noise_ratio_db in signal_to_noise_ratio_levels:

            wav_path = clean_wav_path
            if signal_to_noise_ratio_db is not None:
                noisy_wav_path = audio_directory / f"{task_item.task_id}_snr{int(signal_to_noise_ratio_db)}.wav"
                if not noisy_wav_path.exists():
                    add_background_noise(clean_wav_path, noisy_wav_path,
                                         signal_to_noise_ratio_db, babble_noise_wav_path)
                wav_path = noisy_wav_path

            transcription = transcribe_audio(wav_path, whisper_model)

            asr_item = AsrItem(
                task_id=task_item.task_id,
                clean_text=text,
                transcription=transcription,
                signal_to_noise_ratio_db=signal_to_noise_ratio_db,
                tts_voice=voice,
                whisper_model=whisper_model,
            )
            asr_item.classify(is_word)

            transcription_has_an_error = asr_item.damerau_levenshtein_distance > 0
            should_keep = transcription_has_an_error and (keep_degenerate or not asr_item.is_degenerate)
            if should_keep:
                asr_items.append(asr_item)

    return asr_items
