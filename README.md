# MIDS266-NLP-final-project

# W266: Reducing Hallucinations via Verifier-Reranking

This repository contains the final project for UC Berkeley's MIDS W266 (NLP). It explores methods for reducing factual hallucinations in abstractive summarization models by reranking generated summaries with verification models.

## Problem Statement

Large-scale language models like BART, while powerful, often "hallucinate" or generate text that is not factually supported by the source document. This project investigates whether a two-stage "generate-then-verify" pipeline can improve the factual consistency of summaries.

We first generate multiple summary candidates with a baseline BART model. Then, we use external verifier models (like FactCC and QAGS) to score each candidate's factuality and rerank them, selecting the one with the highest factual consistency score.

-   **Baseline Model:** Fine-tuned BART on the CNN/DailyMail dataset.
-   **Verifier Models:** FactCC and/or QAGS used for reranking.
-   **Evaluation:** We will compare the baseline's top-1 summary against our reranked summary using ROUGE scores and factuality metrics.

## Method Overview

- **Baseline fine-tuning:** We optimize `facebook/bart-base` on CNN/DailyMail using the cross-entropy loss described by Lewis et al. (2020):
  \[
  \mathcal{L}(\theta) = -\sum_{t=1}^{T} \log p_{\theta}(y_t \mid y_{<t}, x),
  \]
  where \(x\) is the article and \(y_t\) are the reference highlight tokens.
- **Verifier reranking:** Following the FactCC formulation (Kryściński et al., 2020), we score a candidate summary \(s\) against article \(a\) via sentence-level NLI entailment probabilities \(p_{\text{entail}}(a_i, s_j)\):
  \[
  \text{score}(a, s) = \frac{1}{|s|} \sum_{j=1}^{|s|} \max_i p_{\text{entail}}(a_i, s_j).
  \]
  Higher scores indicate better factual alignment; alternate verifiers such as QAGS (Wang et al., 2020) and AlignScore (Zha et al., 2023) are evaluated similarly in notebook `05_verify_and_rerank.ipynb`.

## Dataset

We rely on the CNN/DailyMail summarization corpus hosted on Hugging Face at `ccdv/cnn_dailymail` [dataset card](https://huggingface.co/datasets/ccdv/cnn_dailymail). When using the `datasets` library, the loader script is exposed under the shorter identifier `cnn_dailymail`, and version `3.0.0` ships with the cleaned, non-anonymized splits.

To download the data within a Python environment:

```python
from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0")
```

Because the dataset is already tokenized and cleaned, we typically cache the untouched HF download under `data/raw/` and keep any optional inspection artifacts (e.g., subsets or exploratory exports) in `data/interim/` or `data/processed/`. The `notebooks/00_setup_and_checks.ipynb` notebook will create the folder scaffold, log configuration metadata under `configs/`, and cache a small validation sample under `results/` for quick experiments.

## Quickstart / Reproducibility

1.  **Clone the repository**
    ```bash
    git clone https://github.com/<YOUR_ORG>/MIDS266_NLP-final-project.git
    cd MIDS266_NLP-final-project
    ```

2.  **Create a Python environment and install dependencies**
    ```bash
    conda create -n w266 python=3.10 -y
    conda activate w266
    pip install -U pip
    pip install -r requirements.txt
    ```
    The pinned versions in `requirements.txt` match those logged in `notebooks/00_setup_and_checks.ipynb`. A CUDA-capable GPU (T4/V100 class) is recommended for the fine-tuning notebook; CPU-only runs are possible but substantially slower.

3.  **Run the notebooks in order**
    Launch Jupyter (e.g., `jupyter lab` or `jupyter notebook`) and execute the notebooks sequentially:
    - `00_setup_and_checks.ipynb` — verify the environment, download the CNN/DailyMail dataset, and create baseline config files.
    - `01_data_and_baseline.ipynb` — explore the dataset and establish the baseline summarization metrics.
    - `03_train_baseline_bart.ipynb` — fine-tune `facebook/bart-base` on CNN/DailyMail.
    - `04_generate_k_candidates.ipynb` — generate multiple candidate summaries for each article.
    - `05_verify_and_rerank.ipynb` — score candidates with verifier models (e.g., FactCC) and rerank.

4.  **Review outputs**
    Intermediate artifacts (samples, metrics, model checkpoints) are saved under `results/` and `runs/`. Update the notebooks if you need to change output paths or experiment settings.

## Repository Structure

- `data/` — structured into `raw/`, `interim/`, and `processed/` (with `.gitkeep` placeholders) as optional buckets for the untouched HF download and any derived inspection artifacts the notebooks may create.
- `notebooks/` — end-to-end workflow notebooks (`00`–`05`) plus exploratory analysis under `notebooks/EDA/`.
- `reports/` — proposal and background reading (PDFs, project brief).
- `configs/`, `results/`, `runs/` — created by the notebooks to track experiment settings, cached samples, and model outputs; `.gitkeep` files keep these directories under version control.
- `requirements.txt` — pinned dependency list used for grading and reproduction.
- `README.md` — project overview, dataset details, and usage notes.

## Core References

This project is situated within a well-established body of research. The following papers provide the foundation for the baseline model, the verifier-reranking methodology, and the evaluation strategy.

| Paper | Role in Project |
| :--- | :--- |
| **BART** (Lewis et al., 2020) | **Core Baseline Model.** The summarizer we will fine-tune. |
| **FactCC** (Kryściński et al., 2020) | **Core Verifier Model.** The model we will use to score and rerank summaries. |
| **QAGS** (Wang et al., 2020) | **Core Evaluation Metric.** Our secondary, qualitative check for factuality. |
| **Hallucination Survey** (Ji et al., 2023) | **Introduction / Motivation.** Used to define the problem of hallucination. |
| **AlignScore** (Zha et al., 2023) | **Related Work.** A newer factuality metric to compare against. |
| **Unsupervised Reranking** (Ravaut et al., 2023) | **Related Work.** Contrasts our *supervised* verifier with *unsupervised* methods. |
| **SelfCheckGPT** (Manakul et al., 2023) | **Related Work.** An alternative "internal" (LLM-based) verifier. |
| **LoRA** (Hu et al., 2021) | **Future Work.** An optional method for efficiently fine-tuning larger models. |
