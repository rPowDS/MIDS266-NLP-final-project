# Project Proposal: Reducing Hallucinations via Verifier-Reranking

(Your 200-300 word proposal abstract goes here...)

## Annotated Bibliography

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
