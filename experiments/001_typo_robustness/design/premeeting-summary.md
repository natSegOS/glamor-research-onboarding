# Experiment 001 — Pre-Meeting Summary
**Typographical Perturbation Robustness in Open Instruction LLMs**
Prepared for: Zizhao Hu · GLAMOR Lab, USC

---

## What we're studying

How instruction-tuned LLMs respond to typographical noise in their input prompts — and crucially, *why* they fail when they do. The goal is a paper that doesn't just show that typos hurt model performance (that's already well-established), but that pinpoints the mechanism and gives the field something actionable.

## Why the original plan needed sharpening

After a thorough literature review, two things became clear:

1. **"LLMs are brittle to typos" is no longer publishable as a finding.** Recent papers (R²ATA 2024, MulTypo 2025, WikiTypos 2025, ArithmAttack 2025) have all established this convincingly. A paper that demonstrates it again would be rejected as a restatement.

2. **The "appropriate invariance" framing — that models should ignore benign noise but respond to real meaning changes — is a concept that already exists.** It was named by Niu & Bansal (2018), formalized in CheckList (Ribeiro et al., ACL 2020 Best Paper), and applied to LLMs explicitly in a 2025 paper. We need to own a *measurement and mechanism* contribution, not the idea itself.

The fix is a sharpening, not a reversal. The theme stays the same. The contribution moves from demonstration to explanation.

## What we're contributing (the two open gaps)

**Primary — Tokenization-fragmentation mediation.** Prior work shows that subword tokenization *correlates* with brittleness under typos (Chai et al., EMNLP 2024). Nobody has done the controlled experiment that asks: holding the word, the edit count, and the position fixed, and varying *only* how badly the typo fragments the word into subword pieces — does more fragmentation actually cause more accuracy loss? We can answer this with a matched counterfactual design, converting a correlation story into a mechanistic one. This is the paper's headline.

**Secondary — Quantization × typo robustness.** The lab already wants to run quantized 7–8B models (repo issue 02). We turn that into a scientific question: does 4-bit quantization change typo robustness, after controlling for the fact that quantized models may already be slightly less accurate? The only adjacent result (Fang et al. 2025, code generation) found quantized models were *more* robust — counterintuitive and untested for reasoning/QA. We test it properly.

Both of these gaps are confirmed open as of this literature review; neither has been closed.

## How the study is designed

**Tasks:** Two objective, deterministically scorable tasks — a math reasoning task (fresh, contamination-controlled instances) and a multiple-choice knowledge task (MMLU-Pro). Chosen because the literature disagrees on which task type is more typo-fragile, so covering both avoids a task-specific overclaim.

**Models:** Five open-weight instruction models across three families (Llama 1B/3B/8B, Qwen2.5-7B, Mistral-7B), covering three scales and three vocabularies. The small-vocabulary Mistral model is itself a cross-check on the mechanism: if fragmentation drives failure, Mistral should show the biggest effect.

**Perturbations:** Character-level edits (insert, delete, substitute, transpose) using a validated keyboard-aware generator (MulTypo, Liu et al. 2025), across three semantic regimes:
- Regime A: intent-preserving nonword typos (the main endpoint)
- Regime B: real-word substitutions that context can recover
- Regime C: edits that genuinely change the question (a control for over-invariance)

**Human audit:** Three annotators confirm that every item used in the primary analysis genuinely preserves intent. This is what pre-empts the most common rejection: "your perturbations changed the meaning." Agreement is quantified (κ ≥ 0.60 required), not assumed.

**Statistics:** Matched-pair design throughout. Every perturbed prompt has a clean twin scored on the same item, so degradation is measured on the same problems, not across populations. Tests are McNemar (the correct test for paired binary outcomes), BCa bootstrap confidence intervals, and a mixed-effects logistic model. Sample sizes are derived from a target minimum detectable effect (5 percentage points), not chosen arbitrarily — this directly addresses the "underpowered NLP experiment" critique.

**Compute:** We use vLLM with continuous batching instead of the current serial generation loop, giving roughly a 10× throughput gain and making the full study feasible. Default hardware is Colab Pro (L4) plus short paid-GPU bursts (< $50 total) for the fp16 comparison arm. A full free-Colab fallback exists that still supports the primary mechanism claim.

## What makes this publishable

Three properties, in the order reviewers weigh them:

1. **It answers a mechanistic question**, not just a benchmark question. The field is moving from "models fail" to "models fail *because*," and this study is on the right side of that.
2. **It's statistically disciplined** — paired tests, confidence intervals, pre-registered analysis plan. Most typo papers aren't; reviewers notice.
3. **It's cheap to reproduce.** Open models, deterministic scoring, released code and configs. This is increasingly a condition for credibility.

Target venues: ACL Rolling Review → EMNLP 2026 main (evaluation and analysis track), or NAACL 2027.

## What I need from this meeting

1. **Alignment on the headline contribution** — mechanism (tokenization mediation) as primary, quantization as secondary. Does this match the lab's priorities?
2. **Sign-off on a small temporary GPU spend** (~$50 of RunPod/Colab Pro) to run the fp16 comparison arm. The study runs on free hardware if needed, but the quantization contribution requires it.
3. **Annotator sourcing** — can labmates serve as the three auditors, or is there another arrangement to prefer?
4. **Timeline** — is there an ARR cycle or EMNLP 2026 deadline I should be building toward?

Full design documentation (13 detailed docs covering every decision with its justification) is ready and available to share; this summary is intentionally the pre-meeting overview only.

