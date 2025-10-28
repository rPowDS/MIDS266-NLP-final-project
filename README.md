# MIDS266-NLP-final-project

# W266: Reducing Hallucinations via Verifier-Reranking

This repository contains the final project for UC Berkeley's MIDS W266 (NLP). It explores methods for reducing factual hallucinations in abstractive summarization models by reranking generated summaries with verification models.

## Problem Statement

Large-scale language models like BART, while powerful, often "hallucinate" or generate text that is not factually supported by the source document. This project investigates whether a two-stage "generate-then-verify" pipeline can improve the factual consistency of summaries.

We first generate multiple summary candidates with a baseline BART model. Then, we use external verifier models (like FactCC and QAGS) to score each candidate's factuality and rerank them, selecting the one with the highest factual consistency score.

-   **Baseline Model:** Fine-tuned BART on the CNN/DailyMail dataset.
-   **Verifier Models:** FactCC and/or QAGS used for reranking.
-   **Evaluation:** We will compare the baseline's top-1 summary against our reranked summary using ROUGE scores and factuality metrics.

## Quickstart / Reproducibility

1.  **Clone Repository:**
    ```bash
    git clone [YOUR_REPO_URL]
    cd [YOUR_REPO_NAME]
    ```

2.  **Create Environment & Install Dependencies:**
    ```bash
    # (Example using conda)
    conda env create -f environment.yml
    conda activate w266-project

    # (Or using pip)
    pip install -r requirements.txt
    ```

3.  **Prepare Data:**
    *(This step will download and process the CNN/DailyMail v3 dataset)*
    ```bash
    python scripts/prepare_data.py
    ```

4.  **Run Pipeline:**
    *(Run the baseline and the full reranking pipeline. Results will be saved to `results/`)*
    ```bash
    # Run the baseline BART model
    python scripts/run_baseline.py --config configs/bart_baseline.yml

    # Run the verifier/reranking model
    python scripts/run_rerank.py --config configs/rerank_factcc.yml
    ```

## Repository Structure


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
