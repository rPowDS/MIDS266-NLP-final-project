# Reducing Hallucinations in Abstractive Summarization via Verifier-Reranking

## Overview
Abstractive summarizers often generate fluent but unsupported statements, limiting real-world utility. This project targets **factuality** in news summarization using a simple, reproducible pipeline:

- **Model**: Fine-tune `facebook/bart-base` on CNN/DailyMail.
- **Decoding**: Generate *K* candidate summaries per article.
- **Reranking**: Score candidates with an automatic factuality verifier and select the top-scoring summary.
- **Verifiers**: Use **FactCC** as the primary verifier; report **QAGS** as a secondary check.
- **Fallback**: If reranking harms ROUGE beyond the threshold (see Success Criteria), test copy-favoring constrained decoding.
- **Stack**: Hugging Face **Transformers**, **Datasets**, and **Evaluate**.

The aim is to reduce hallucinations with minimal engineering and clear, testable metrics.

---

## Dataset
- **Name**: CNN/DailyMail  
- **Hugging Face ID**: `ccdv/cnn_dailymail` (config **3.0.0**)  
- **Splits**: Standard train / validation / test

---

## Evaluation & Success Criteria

**Quality metrics (validation & test):**
- ROUGE-1 / ROUGE-2 / ROUGE-L
- BERTScore

**Factuality metrics (validation):**
- **FactCC** (primary)
- **QAGS** (secondary/diagnostic)

**Human study (validation subset):**
- 50 articles with an error taxonomy (e.g., entity errors, temporal/causal errors, unsupported claims, contradictions).

**Success criterion (validation):**
- **Factuality gain**: `Δ FactCC ≥ +2.0` points over a BART-base baseline  
- **Quality retention**: `|Δ ROUGE-L| ≤ 1.0` absolute drop

If the criterion is not met, enable constrained decoding that biases copying from the source and re-evaluate.

---

## Method

1. **Fine-tuning**
   - Model: `facebook/bart-base`
   - Data: `ccdv/cnn_dailymail:3.0.0`
   - Training details recorded in versioned config files (learning rate, batch size, max length, number of epochs, seed).

2. **Candidate Generation**
   - Produce *K* candidates per article (initially `K ∈ {4, 8}`).
   - Explore beam search vs. nucleus sampling; log decoding params.

3. **Verifier-Reranking**
   - Score each candidate with **FactCC**; pick the top-scoring candidate.
   - Report **QAGS** as an auxiliary factuality check.

4. **Fallback: Constrained Decoding**
   - If reranking materially hurts ROUGE beyond threshold, switch to constrained decoding that promotes source copying (e.g., lexically constrained decoding or penalties that reduce unsupported novel content).

5. **Ablations**
   - No-rerank baseline (best-of-K by log-prob only).
   - Rerank by **FactCC** vs. alternative factuality signals (e.g., AlignScore) for comparison.
   - Beam vs. sampling; varying *K*.
   - Length penalty sensitivity.

---

## Risks & Mitigations

- **Verifier mis-scores paraphrases**  
  *Mitigation*: human spot checks on 50 examples; include an error taxonomy and reviewer agreement.

- **Compute constraints**  
  *Mitigation*: use base-size models, modest *K*, mixed-precision training, gradient accumulation if needed.

- **Reranking underperforms**  
  *Mitigation*: tune decoding parameters, enable constrained decoding, and report ablations transparently.

---

## Why This Matters
Abstractive systems can “sound right” while being wrong. Selecting the most factual candidate at inference time reduces hallucinations without heavy architectural changes, improving reliability for downstream users and decision-makers.

---

## Reproducibility Plan
- **Configs**: All training/decoding/reranking settings in versioned YAMLs.  
- **Seeds**: Fixed random seeds; report ± std over three runs on validation.  
- **Checkpoints**: Save and tag model and tokenizer versions.  
- **Logs**: Persist metrics and artifacts (CSV/JSON).  
- **Release**: Code, configs, and a small sample of preprocessed examples for end-to-end validation.

---

## Repository Structure (planned)
