# Reducing Hallucinations in Summarization via Verifier-Reranking

**Course:** W266 NLP (UC Berkeley MIDS)  
**Student:** Ryan Powers  

This repository contains my W266 final project on reducing factual hallucinations in abstractive summarization using a **generate-then-verify** pipeline. I fine-tune a BART-base summarizer on CNN/DailyMail, generate multiple candidate summaries, and rerank them with factuality verifiers (FactCC and RoBERTa-MNLI). The goal is to improve factual consistency **without** retraining a larger model.

The twist: by automatic metrics, the method looks very successful, but a 50-example human audit shows that those gains **do not translate** into better human-perceived factuality.

---

## Problem Statement & Approach

Large pretrained sequence-to-sequence models like **BART** achieve strong ROUGE scores on CNN/DailyMail but are known to hallucinate—producing fluent text that is not fully grounded in the source article (Ji et al., 2023). This is especially problematic for news summarization, where factual accuracy is critical.

My question:

> **Can I reduce hallucinations at inference time by generating multiple summaries and letting a verifier pick the most factual one?**

I frame this as a **verifier-reranking** problem:

1. **Generate**: Fine-tune `facebook/bart-base` on CNN/DailyMail v3.0.0, then generate **K = 5** candidate summaries per article with beam search.
2. **Verify**: Score each candidate with:
   - **FactCC** (Kryściński et al., 2020) – a summarization-specific factuality classifier.
   - **RoBERTa-MNLI** (Liu et al., 2019) – a general NLI model used as a secondary verifier.
3. **Rerank**: Select the candidate with the highest verifier score as the final summary.

This design is inspired by FactCC and later reranking work (e.g., Ravaut et al., 2023) and matches techniques discussed in class on factuality and evaluation.

---

## Methodology & Architecture

The project follows a **Generate → Verify → Rerank** pipeline:

**(Article)** → **BART Baseline** → **K=5 Candidates** → **FactCC / NLI Scoring** → **Reranker** → **Final Summary**

> **Reranking Objective**

Let \(a\) be the article and \(C = \{c_1, \ldots, c_K\}\) the set of beam candidates.  
The reranker selects:

\[
c^* = \argmax_{c_i \in C} \text{score}_{\text{FactCC}}(a, c_i)
\]

where

\[
\text{score}_{\text{FactCC}}(a, c) = P(\text{Consistent} \mid a, c)
\]

is the probability assigned by the FactCC classifier that summary \(c\) is factually consistent with article \(a\). For the NLI baseline I analogously use:

\[
\text{score}_{\text{NLI}}(a, c) = P(\text{Entailment} \mid a, c)
\]

from RoBERTa-MNLI with \(a\) as premise and \(c\) as hypothesis.  
These are exactly the scores produced by the Hugging Face `manueldeprada/FactCC` and `roberta-large-mnli` models in my code.   

> **Training Objective (Baseline BART)**

I fine-tune BART-base with the standard cross-entropy loss:

\[
\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, x)
\]

where \(x\) is the source article and \(y_t\) are tokens of the gold summary. This is the same objective used in the original BART paper (Lewis et al., 2020).

> **Evaluation Metric (ROUGE-L)**

For summary quality I report ROUGE-L F-score based on the longest common subsequence (LCS):

\[
\text{ROUGE-L} = \frac{(1 + \beta^2) R_{\text{lcs}} P_{\text{lcs}}}{R_{\text{lcs}} + \beta^2 P_{\text{lcs}}}
\]

where

\[
R_{\text{lcs}} = \frac{\text{LCS}(X, Y)}{|X|}, \quad
P_{\text{lcs}} = \frac{\text{LCS}(X, Y)}{|Y|}
\]

for reference summary \(X\) and candidate \(Y\).

> **Statistical Testing**

Following standard NLP evaluation practice (e.g., Efron & Tibshirani, 1994; Dror et al., 2018), I use:

- **Approximate randomization test (N = 10,000 permutations)** for FactCC deltas.
- **Paired bootstrap (N = 1,000 resamples)** for ROUGE-L differences.

I reduced the bootstrap iterations from 10,000 to 1,000 for runtime reasons; 1,000 resamples is commonly used in NLP and provides stable confidence intervals for this scale of test set. The code and configuration explicitly log these choices.   

---

## Key Results

All main numbers are from the **CNN/DailyMail test set (N = 11,490)**.   

### Automatic Metrics (Test Set)

| System         | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | FactCC |
|----------------|---------|---------|---------|--------------|--------|
| LEAD-3         | 39.91   | 17.38   | 24.91   | —            | —      |
| BART Baseline  | 41.12   | 18.56   | 28.09   | 0.882        | 0.427  |
| FactCC Rerank  | 40.74   | 18.36   | 27.92   | 0.882        | **0.661** |
| NLI Rerank     | 40.78   | 18.48   | 28.08   | —            | —      |

- **FactCC improvement:** 0.427 → 0.661 (**+0.234**, +54.8% relative)  
- **ROUGE-L change:** 28.09 → 27.92 (**−0.17**, within my ≤1.0 tolerance)  
- **BERTScore:** 0.882 vs 0.882 on a 1,000‑example sample → semantic similarity preserved.   

**Significance tests:**

- Approximate randomization (FactCC): p < 0.0001 → **highly significant**.  
- Paired bootstrap, N=1,000 (ROUGE-L): 95% CI includes 0 → **no significant change**.

By automatic metrics alone, verifier-reranking “succeeds”: factuality up, ROUGE preserved.

### Human Evaluation (N = 50)

I conducted a 50-example human audit (single annotator, error taxonomy from Ji et al., 2023) on stratified samples (30 “disagreement” cases where reranking changed the summary, 20 “agreement” controls).   

**Overall preference:**

| Preference          | Count | Percentage |
|---------------------|-------|------------|
| Tie                 | 27    | 54%        |
| Baseline preferred  | 14    | 28%        |
| Reranked preferred  | 9     | 18%        |

**Error-level analysis:**

- Cases where reranker **fixed** a hallucination: **1**  
- Cases where reranker **introduced** a new hallucination: **3**  
- Cases where **both** summaries hallucinated: **16** (32%)  

Net effect: the reranker slightly **hurts** human-judged factuality, despite large apparent FactCC gains. This is the central “metrics vs humans” disconnect in the project.   

---

## Repository Structure & Notebooks

The repo is organized as a linear experimental pipeline:

| Notebook                            | Role             | Description |
|------------------------------------|------------------|-------------|
| `01_Setup_&_EDA.ipynb`             | Setup / EDA      | Dependency installation, project config, and CNN/DM length analysis. |
| `02_Baseline_BART.ipynb`           | Training         | Fine-tunes `facebook/bart-base` on 20k CNN/DM examples (3 epochs). |
| `02a_Analyze_Baseline_Run.ipynb`   | Baseline analysis| Loss curves, qualitative baseline inspection. |
| `03_Generate_Candidate.ipynb`      | Candidate generation | Generates K=5 beam candidates for validation / analysis sets. |
| `04_Rerank_&_Score.ipynb`          | Verifier scoring | Scores all candidates with FactCC and RoBERTa-MNLI and performs reranking. |
| `05_Analysis.ipynb`                | Metric analysis  | Produces main result tables, K-ablation, metric correlations and plots. |
| `06_Human_Audit_Prep.ipynb`        | Human eval setup | Prepares 50-example audit CSV and aggregates error labels. |
| `07_Final_Test.ipynb`              | Full test run    | Runs reranking on the full CNN/DM test set (11,490 examples). |
| `08_Evaluation_&_Statistics.ipynb` | Statistics       | Randomization test, bootstrap CIs, and final “money tables.” |
| `Final_Plots.ipynb`                | Visualization    | Generates production-quality figures used in the report. |

Each stage saves intermediate JSONL/CSV files so results are reproducible and traceable.

---

## Dataset & Models

- **Dataset:** CNN/DailyMail summarization corpus via Hugging Face, config `3.0.0` (non-anonymized).  [oai_citation:13‡Hugging Face](https://huggingface.co/datasets/abisee/cnn_dailymail?utm_source=chatgpt.com)  
  - HF dataset card: `ccdv/cnn_dailymail`
- **Summarizer:** `facebook/bart-base` fine-tuned on 20k CNN/DM training examples.
- **Verifiers:**
  - FactCC: `manueldeprada/FactCC` (implementation of Kryściński et al., 2020).  
  - NLI: `roberta-large-mnli` (Liu et al., 2019). Originally, AlignScore (Zha et al., 2023) was cited as a state-of-the-art alignment metric; we compared against NLI instead due to technical constraints. 

---

## Dataset & Models

 **CNN/DailyMail** summarization corpus hosted on Hugging Face. Using version `3.0.0`, which contains the non-anonymized data.

* **Dataset Card:** [`ccdv/cnn_dailymail`](https://huggingface.co/datasets/ccdv/cnn_dailymail)
* **Baseline Model:** [`facebook/bart-base`](https://huggingface.co/facebook/bart-base)
* **FactCC Verifier:** [`manueldeprada/FactCC`](https://huggingface.co/manueldeprada/FactCC) (Port of Kryściński et al. 2020)
* **NLI Verifier:** [`roberta-large-mnli`](https://huggingface.co/roberta-large-mnli)

To download the data within a Python environment:

from datasets import load_dataset

# Download version 3.0.0 
dataset = load_dataset("cnn_dailymail", "3.0.0")

---

##  Experiment Log

This project followed an iterative experimental design:

* **Exp 1 (Baseline):** Fine-tuned BART on 20k examples. Established a strong ROUGE baseline (29.3).
* **Exp 2 (Candidate Gen):** Implemented Beam Search (K=5) to create diverse summary options.
* **Exp 3 (FactCC Rerank):** Applied the FactCC verifier. Achieved primary success criteria (>50% factuality gain).
* **Exp 4 (NLI Rerank):** Applied a standard NLI model (RoBERTa-MNLI) to validate robustness. Result: Confirmed reranking effectiveness.
* **Exp 5 (Human Audit):** Manually reviewed 50 samples to verify automated metrics against human judgment.

---

##  Core References

The following papers provide the foundation for the baseline model, the verifier-reranking methodology, and the evaluation strategy.

| Paper | Role in Project | Status / Notes |
| :--- | :--- | :--- |
| **BART** (Lewis et al., 2020) | **Core Baseline** | **Implemented.** Fine-tuned `bart-base` on CNN/DM (Notebook 02). |
| **FactCC** (Kryściński et al., 2020) | **Core Verifier** | **Implemented.** Used for scoring & reranking (Notebook 04). |
| **RoBERTa** (Liu et al., 2019) | **Secondary Verifier** | **Implemented.** Used `roberta-large-mnli` as a proxy for General Logic verification. |
| **Hallucination Survey** (Ji et al., 2023) | **Problem Def.** | Used to define the taxonomy of hallucination errors. |
| **AlignScore** (Zha et al., 2023) | **Comparison** | **Reference Only.** Cited as a state-of-the-art alignment metric; we compared against NLI instead due to technical constraints. |
| **QAGS** (Wang et al., 2020) | **Conceptual Basis** | **Reference Only.** Establishes the framework for consistency checking, though we opted for NLI over QA. |
| **SelfCheckGPT** (Manakul et al., 2023) | **Related Work** | Cited to contrast our *supervised* pipeline with *zero-shot* methods. |
| **LoRA** (Hu et al., 2021) | **Future Work** | Cited as a potential method for scaling this pipeline to LLMs. |

---
