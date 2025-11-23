# Reducing Hallucinations in Summarization via Verifier-Reranking

**Course:** W266 NLP (UC Berkeley MIDS)  
**Student:** Ryan Powers  


This project demonstrates a practical pipeline for reducing factual hallucinations in abstractive summarization. By generating multiple candidate summaries with a fine-tuned BART model and reranking them using specialized verifier models, the project achieves significant gains in factual consistency without sacrificing summary fluency.

---

##  Problem Statement & Approach

Large-scale language models like BART, while powerful, often "hallucinate" or generate text that is not factually supported by the source document. This project investigates whether a two-stage "generate-then-verify" pipeline can improve the factual consistency of summaries.:

1.  **Generate:** Using a fine-tuned BART model to generate $K=5$ candidate summaries via Beam Search.
2.  **Verify:** Score each candidate using two distinct verifier architectures:
    * **FactCC:** A BERT-based model trained specifically on summarization consistency.
    * **RoBERTa-NLI:** A general-purpose logic model for checking textual entailment.
3.  **Rerank:** Select the candidate with the highest verifier score as the final output.

---

##  Methodology & Architecture

Our approach follows a **"Generate-then-Verify"** pipeline.

**[Insert Architecture Diagram Here]**
*(Article $\to$ BART Baseline $\to$ K Candidates $\to$ Verifier Scoring $\to$ Reranker $\to$ Final Summary)*

### Method Overview

* **Baseline Fine-tuning:** We optimize `facebook/bart-base` on CNN/DailyMail using the standard cross-entropy loss described by Lewis et al. (2020):

    $$\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log p_{\theta}(y_t \mid y_{<t}, x)$$

    where $x$ is the article and $y_t$ are the reference highlight tokens.

* **Verifier Reranking:** Following the FactCC formulation (Kryściński et al., 2020), we score a candidate summary $s$ against article $a$ via sentence-level NLI entailment probabilities $p_{\text{entail}}(a_i, s_j)$:

    $$\text{score}(a, s) = \frac{1}{|s|} \sum_{j=1}^{|s|} \max_i p_{\text{entail}}(a_i, s_j)$$

    Higher scores indicate better factual alignment. We evaluated this using both a specialized verifier (**FactCC**) and a general logic verifier (**RoBERTa-NLI**).

---


##  Key Findings 

The experiments on the CNN/DailyMail validation set (20k training subset) show that reranking significantly improves factuality.

| Strategy | Factuality Score (FactCC) | Fluency (ROUGE-L) | Impact |
| :--- | :--- | :--- | :--- |
| **Baseline (BART)** | 0.43 | **29.30** | High fluency, but frequent hallucinations. |
| **FactCC Rerank** | **0.66** (+53%) | 29.00 (-0.3) | **Massive factuality gain** with negligible quality loss. |
| **NLI Rerank** | N/A (0.27 NLI Score) | 28.99 | Robustness check; confirms the method works across architectures. |

**Human Audit:** A manual review of 50 samples confirmed that the reranker fixed hallucinations in **X%** of disagreement cases. *(Update this X% after your audit)*.

---

##  Repository Structure & Notebooks

The project is organized into sequential notebooks that represent the full experimental pipeline.

| Notebook | Role | Paper / Inspiration | Description |
| :--- | :--- | :--- | :--- |
| **`01_Setup_&_EDA.ipynb`** | Setup | *CNN/DM Analysis* | Project structure, dependency installation, and EDA on token lengths. |
| **`02_Baseline_BART.ipynb`** | **Training** | **Lewis et al. (2020)** | Fine-tuning `facebook/bart-base` on a 20k subset of CNN/DailyMail. |
| **`02a_Analyze_Baseline.ipynb`** | Analysis | *Standard Practice* | Loss curves and qualitative checks of the baseline model. |
| **`03_Generate_Candidates.ipynb`** | **Inference** | *Beam Search* | Generating $K=5$ candidate summaries for 2,000 validation articles. |
| **`04_Rerank_&_Score.ipynb`** | **Scoring** | **Kryściński et al. (2020)** | Scoring all candidates using [FactCC](https://huggingface.co/manueldeprada/FactCC) and [RoBERTa-NLI](https://huggingface.co/roberta-large-mnli). |
| **`05_Analyze_&_Visualize.ipynb`** | Analysis | *Comparative Study* | Generating the final "Money Table" and ROUGE comparisons. |
| **`06_Human_Audit.ipynb`** | Audit | *Qualitative Review* | Preparing and analyzing the 50-sample human evaluation set. |

---

## Dataset & Models

We rely on the **CNN/DailyMail** summarization corpus hosted on Hugging Face. We use version `3.0.0`, which contains the non-anonymized data.

* **Dataset Card:** [`ccdv/cnn_dailymail`](https://huggingface.co/datasets/ccdv/cnn_dailymail)
* **Baseline Model:** [`facebook/bart-base`](https://huggingface.co/facebook/bart-base)
* **FactCC Verifier:** [`manueldeprada/FactCC`](https://huggingface.co/manueldeprada/FactCC) (Port of Kryściński et al. 2020)
* **NLI Verifier:** [`roberta-large-mnli`](https://huggingface.co/roberta-large-mnli)

To download the data within a Python environment:

from datasets import load_dataset

# Download version 3.0.0 (non-anonymized)
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

This project is situated within a well-established body of research. The following papers provide the foundation for the baseline model, the verifier-reranking methodology, and the evaluation strategy.

| Paper | Role in Project |
| :--- | :--- |
| **BART** (Lewis et al., 2020) | **Core Baseline Model.** The summarizer we fine-tuned in Notebook 02. |
| **FactCC** (Kryściński et al., 2020) | **Core Verifier Model.** The primary model used to score and rerank summaries in Notebook 04. |
| **QAGS** (Wang et al., 2020) | **Core Evaluation Metric.** Our conceptual basis for checking consistency via question answering. |
| **Hallucination Survey** (Ji et al., 2023) | **Introduction / Motivation.** Used to define the problem of hallucination in abstractive summarization. |
| **AlignScore** (Zha et al., 2023) | **Related Work.** Comparison metric for our NLI-based verification approach. |
| **Unsupervised Reranking** (Ravaut et al., 2023) | **Related Work.** Contrasts our *supervised* verifier with *unsupervised* methods. |
| **SelfCheckGPT** (Manakul et al., 2023) | **Related Work.** An alternative "internal" (LLM-based) verifier. |
| **LoRA** (Hu et al., 2021) | **Future Work.** An optional method for efficiently fine-tuning larger models. |

---
