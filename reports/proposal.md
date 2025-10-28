# Reducing Hallucinations in Abstractive Summarization via Verifier-Reranking

## Abstract
Abstractive summarizers often produce fluent but unsupported statements, limiting practical use. This project targets **factuality** for news summarization with a simple, reproducible approach. I will fine-tune **BART-base** on the public **CNN/DailyMail** dataset. At inference time, I will generate **K** candidate summaries per article and **rerank** them using an automatic factuality score, using **FactCC** as the primary verifier and reporting **QAGS** as a secondary check. If reranking harms ROUGE beyond the preset threshold, I will test **constrained decoding** that favors copying from the source as a fallback. Implementations rely on **Hugging Face Transformers, Datasets, and Evaluate**, and will be cited.

---

## Dataset
- **Dataset**: CNN/DailyMail  
- **Hugging Face ID**: `ccdv/cnn_dailymail` (config **3.0.0**)  
- **Splits**: Standard train / validation / test

---

## Metrics and Success

**Quality (validation & test):**
- ROUGE-1 / ROUGE-2 / ROUGE-L
- BERTScore

**Factuality (validation):**
- **FactCC** (primary)
- **QAGS** (secondary/diagnostic)

**Human check (validation subset):**
- 50 examples with an error taxonomy (e.g., entity errors, temporal/causal errors, unsupported claims, contradictions)

**Success criterion (validation):**
- **Δ FactCC ≥ +2.0** points over the BART-base baseline  
- **ROUGE-L drop ≤ 1.0** absolute point

If the criterion is not met, enable copy-favoring constrained decoding and re-evaluate.

---

## Risks / Challenges
- **Verifier may mis-score paraphrases** → Mitigate with human spot checks and an explicit error taxonomy.  
- **Compute limits** → Use base-size models and modest **K**; keep decoding budgets reasonable.  
- **If reranking underperforms** → Tune decoding parameters, enable constrained decoding, and report ablations.

---

## Why This Matters
Abstractive models often sound good but insert factual mistakes. Selecting the most factual draft at inference time can reduce hallucinations **without heavy engineering**, improving reliability for downstream users.

---

## Reproducibility & Release
This focused design addresses a known failure mode with minimal engineering and clear metrics. It fits the class scope, uses a single public dataset with stable splits, and supports deep analysis. I will **release code, configs, and small data samples** for end-to-end reproducibility.

---

## Annotated Bibliography

| Paper | Role in Project |
| --- | --- |
| **BART (Lewis et al., 2020)** | **Core Baseline Model.** The summarizer to fine-tune. |
| **FactCC (Kryściński et al., 2020)** | **Core Verifier Model.** Scores and reranks summaries. |
| **QAGS (Wang et al., 2020)** | **Core Evaluation Metric.** Secondary factuality check. |
| **Hallucination Survey (Ji et al., 2023)** | **Introduction / Motivation.** Defines the hallucination problem. |
| **AlignScore (Zha et al., 2023)** | **Related Work.** Newer factuality metric for comparison. |
| **Unsupervised Reranking (Ravaut et al., 2023)** | **Related Work.** Contrasts supervised verifier with unsupervised methods. |
| **SelfCheckGPT (Manakul et al., 2023)** | **Related Work.** An internal (LLM-based) verifier. |
| **LoRA (Hu et al., 2021)** | **Future Work.** Efficient fine-tuning option for larger models. |

---

## Core References from Proposal
- **BART**: Lewis et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training…* (ACL Anthology).  
- **FactCC**: Kryściński et al. (2020). *Evaluating the Factual Consistency of Abstractive Text Summarization.* (ACL Anthology).  
- **QAGS**: Wang et al. (2020). *Asking and Answering Questions to Evaluate the Factual Consistency of Summaries.* (ACL Anthology).  
- **PEGASUS**: Zhang et al. (2020). *Pre-training with Extracted Gap-sentences for Abstractive Summarization.* (ICML).  
- **BERT**: Devlin et al. (2019). *Pre-training of Deep Bidirectional Transformers…* (arXiv).
