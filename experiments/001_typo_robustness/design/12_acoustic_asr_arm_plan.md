# 12 - Acoustic ASR Arm: Proposed Plan

**Status: PROPOSAL ONLY. Not approved, not pre-registered, not implemented.**
The acoustic arm was deferred on 2026-07-09 (design/00 §0.5). This document records the replacement plan proposed by Nathan on 2026-07-21 so work can start immediately if the PI approves it. Nothing here binds the current pre-registration.

## 1. Idea in one paragraph

Treat the audio round trip as one more perturbation source. Take each clean prompt, synthesize it as realistic human-sounding speech across a roster of accents, optionally degrade the audio in reproducible ways (background noise at fixed SNR, telephone codec), transcribe it with a pinned ASR model, and use the transcript as the perturbed prompt. Everything downstream (matched pairs, regimes, human audit, statistics) is the existing machinery unchanged.

## 2. Why this design

| Requirement | How the plan meets it |
|---|---|
| Realism (the reason the TTS+Whisper arm was deferred) | Voice diversity from real accented speakers, not one clean neural voice; degradation from real noise recordings and a real telephony codec |
| Reproducibility | Every component pinned: TTS model revision, seed voices, noise clips and SNRs, codec parameters, ASR revision with deterministic decoding; generated audio released as artifacts |
| Fits the existing study | Produces perturbed text consumed by the same pipeline; errors land mostly in Regime B; the audit protocol already contains ASR examples (design/09 §9.2) |
| Complements HIVE | HIVE's voice arm covers the composition channel (register, disfluency) with LLM verbalization; this covers the acoustic channel (recognition errors) both papers list as future work |

## 3. Pipeline specification

```
clean prompt
  -> TTS (pinned open-weights model, voice cloned from a licensed accented-speaker corpus)
  -> optional degradation (babble noise at fixed SNR | telephone codec 8 kHz mu-law | none)
  -> ASR (Whisper large-v3, pinned revision, scalar temperature=0.0,
          condition_on_previous_text=False; determinism recipe in docs/PROVENANCE.md §3)
  -> transcript = perturbed prompt
```

Design choices, each with its defense:

- **Voices.** 6 to 10 seed speakers spanning L1 and L2 English accents, cloned from a research-licensed corpus (candidates: L2-ARCTIC, VCTK) through a pinned open-weights zero-shot TTS. No API voices: they cannot be pinned and drift across versions.
- **Strongest ASR, degraded channel.** Keep Whisper large-v3 so no reviewer can claim errors were manufactured by a weak transcriber. Errors come from the channel: quiet, noisy (MUSAN babble at fixed SNRs, fixed clip seeds), and telephone-codec conditions.
- **Severity is measured, not preset.** ASR errors are item-specific. The edit budget is the measured Damerau-Levenshtein distance from clean prompt to transcript, bucketed into pre-registered bands, exactly as design/03 §3.5a already specifies.
- **Digit normalization rule (must be pre-registered).** TTS reads "30" aloud and the transcript may return "thirty". This is semantically null (HIVE measured number-word rewriting at -0.6 pp) but inflates edit distance. Normalize digits and number words to one canonical form before computing measured distance, and log the normalization count per item.
- **Audio artifacts are released.** Neural TTS is not guaranteed bit-identical across GPUs. Reproducibility rests on the released 16 kHz mono WAVs plus the regeneration recipe (already planned in design/08).

## 4. Expected behavior and pre-registered framing

- Verbatim transcripts (zero edits) are expected on the quiet condition with a strong ASR. They are reported as the ASR-robust stratum, not discarded.
- Pre-registered prediction: quiet is approximately benign, noise and codec conditions cross into damage, and token-inflation mediates whatever damage appears (the cross-modal mechanism question). A null on quiet is a reportable result under design/06 §6.10.
- Regime assignment is by the existing rules: real-word confusions to B, nonwords to A, meaning changes (dropped negations, mis-transcribed numbers that survive normalization) to C via the human audit.

## 5. Gate before committing

Run a 100-item audio pilot first, same discipline as every other gate in this project:

1. Measure the edit-distance distribution per condition (quiet / each SNR / codec).
2. Confirm nonzero yield in the pre-registered severity bands outside the quiet condition.
3. Lock the speaker roster, SNR levels, normalization rule, and band edges from the pilot readout.
4. Then register the arm as an amendment and run at full N.

## 6. Engineering scope

Small relative to the existing codebase:

- New tool `build_asr_perturbations.py`: TTS -> degrade -> transcribe -> pairs file with measured-k, consumed by the existing pipeline as a new condition source. Schema fields already exist (`asr_source`, `tts_voice_id`, `whisper_model_revision`, `snr_db`); add `speaker_id`, `noise_clip_id`, `codec`.
- New `requirements-asr.txt`: TTS model stack, `openai-whisper`, `soundfile`, audio DSP for SNR mixing and mu-law.
- GPU hours for synthesis and transcription (rough order: a few hours per 1k items per condition).
- Everything else (runner, statistics, audit, report) unchanged.

## 7. Needed from the PI before starting

1. Approval (this reverses the 2026-07-09 deferral; the PI proposed finding an alternative, and this is the candidate).
2. Choice or veto of the TTS model and speaker corpus; licensing check for redistribution of generated audio.
3. Cluster or Colab resources with audio support.
4. How the arm slots into the shared paper (suggested: the acoustic branch both the HIVE limitations section and this design list as future work).
