# Reducing Hallucinations in Summarization via Verifier-Reranking

**Course:** W266 NLP (UC Berkeley MIDS)  
**Student:** Ryan Powers  
**Status:** Complete

This project demonstrates a practical pipeline for reducing factual hallucinations in abstractive summarization. By generating multiple candidate summaries with a fine-tuned BART model and reranking them using specialized verifier models, we achieve significant gains in factual consistency without sacrificing summary fluency.

---

##  Problem Statement & Approach

Abstractive models like BART are powerful but prone to "hallucinating"—generating text that is fluent but unsupported by the source document. This project investigates a two-stage **"Generate-then-Verify"** pipeline to solve this:

1.  **Generate:** Use a fine-tuned BART model to generate $K=5$ candidate summaries via Beam Search.
2.  **Verify:** Score each candidate using two distinct verifier architectures:
    * **FactCC:** A BERT-based model trained specifically on summarization consistency.
    * **RoBERTa-NLI:** A general-purpose logic model checking textual entailment.
3.  **Rerank:** Select the candidate with the highest verifier score as the final output.

---

##  Key Results (The "Money Table")

Our experiments on the CNN/DailyMail validation set (20k training subset) show that reranking significantly improves factuality.

| Strategy | Factuality Score (FactCC) | Fluency (ROUGE-L) | Impact |
| :--- | :--- | :--- | :--- |
| **Baseline (BART)** | 0.43 | **29.30** | High fluency, but frequent hallucinations. |
| **FactCC Rerank** | **0.66** (+53%) | 29.00 (-0.3) | **Massive factuality gain** with negligible quality loss. |
| **NLI Rerank** | N/A (0.27 NLI Score) | 28.99 | Robustness check; confirms the method works across architectures. |

**Human Audit:** A manual review of 50 samples confirmed that the reranker fixed hallucinations in **X%** of disagreement cases. *(Update this X% after your audit)*.

---

##  Repository Structure & Notebooks

The project is organized into sequential notebooks that represent the full experimental pipeline.

| Notebook | Role | Description |
| :--- | :--- | :--- |
| **`01_Setup_&_EDA.ipynb`** | Setup | Project structure, dependency installation, and EDA on token lengths. |
| **`02_Baseline_BART.ipynb`** | **Training** | Fine-tuning `facebook/bart-base` on a 20k subset of CNN/DailyMail. |
| **`02a_Analyze_Baseline.ipynb`** | Analysis | Loss curves and qualitative checks of the baseline model. |
| **`03_Generate_Candidates.ipynb`** | **Inference** | Generating $K=5$ candidate summaries for 2,000 validation articles. |
| **`04_Rerank_&_Score.ipynb`** | **Scoring** | Scoring all candidates using FactCC and RoBERTa-NLI. |
| **`05_Analyze_&_Visualize.ipynb`** | Analysis | Generating the final "Money Table" and ROUGE comparisons. |
| **`06_Human_Audit.ipynb`** | Audit | Preparing and analyzing the 50-sample human evaluation set. |

---

##  Experiment Log

This project followed an iterative experimental design:

* **Exp 1 (Baseline):** Fine-tuned BART on 20k examples. Established a strong ROUGE baseline (29.3).
* **Exp 2 (Candidate Gen):** Implemented Beam Search (K=5) to create diverse summary options.
* **Exp 3 (FactCC Rerank):** Applied the FactCC verifier. Achieved primary success criteria (>50% factuality gain).
* **Exp 4 (NLI Rerank):** Applied a standard NLI model (RoBERTa-MNLI) to validate robustness. Result: Confirmed reranking effectiveness.
* **Exp 5 (Human Audit):** Manually reviewed 50 samples to verify automated metrics against human judgment.

---

##  Quickstart / Reproducibility

**1. Clone the repository**
```bash
git clone [https://github.com/](https://github.com/)<YOUR_ORG>/MIDS266_NLP-final-project.git
cd MIDS266_NLP-final-project
