Reducing Hallucinations in Abstractive Summarization via Verifier-Reranking

Proposal Abstract

Abstract summarizers often produce fluent but unsupported statements that limit practical use. This project targets factuality for news summarization with a simple, reproducible approach. I will fine-tune BART-base on the public CNN/DailyMail dataset. At inference, I will generate K candidate summaries per article and rerank them using an automatic factuality score. Using FactCC as the primary verifier and reporting QAGS as a secondary check. If reranking harms ROUGE, I will test constrained decoding that favors copying from the source as a fallback. Implementations rely on Hugging Face Transformers, Datasets, and Evaluate, and will be cited.

Dataset

Dataset: CNN/DailyMail

Hugging Face ID: ccdv/cnn_dailymail, config 3.0.0

Splits: Using standard train/validation/test splits.

Metrics and Success

Quality: ROUGE-1/2/L and BERTScore on validation and test.

Factuality: FactCC (primary) and QAGS on validation.

Human check: 50 examples with an error taxonomy.

Success criterion: $\ge+2.0$ FactCC points over the BART baseline on validation with $\le1.0$ ROUGE-L drop.

Risks / Challenges

Verifier may mis-score paraphrases; mitigate with human spot checks. Compute limits handled by base-size models and modest K. If reranking underperforms, tune decoding and report ablations.

Why this matters

Abstractive models often sound good but insert mistakes. Picking the most factual draft reduces those hallucinations without heavy engineering.

This focused design addresses a known failure mode with minimal engineering and clear metrics. It fits the class scope, uses a single public dataset with stable splits, and supports deep analysis. I will release code, configs, and small data samples for reproducibility.

Annotated Bibliography

This project is situated within a well-established body of research. The following papers provide the foundation for the baseline model, the verifier-reranking methodology, and the evaluation strategy.

Paper

Role in Project

BART (Lewis et al., 2020)

Core Baseline Model. The summarizer we will fine-tune.

FactCC (Kryściński et al., 2020)

Core Verifier Model. The model we will use to score and rerank summaries.

QAGS (Wang et al., 2020)

Core Evaluation Metric. Our secondary, qualitative check for factuality.

Hallucination Survey (Ji et al., 2023)

Introduction / Motivation. Used to define the problem of hallucination.

AlignScore (Zha et al., 2023)

Related Work. A newer factuality metric to compare against.

Unsupervised Reranking (Ravaut et al., 2023)

Related Work. Contrasts our supervised verifier with unsupervised methods.

SelfCheckGPT (Manakul et al., 2023)

Related Work. An alternative "internal" (LLM-based) verifier.

LoRA (Hu et al., 2021)

Future Work. An optional method for efficiently fine-tuning larger models.

Core References from Proposal

BART: Lewis et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training... (ACL Anthology)

FactCC: Kryściński et al. (2020). Evaluating the Factual Consistency of Abstractive Text Summarization. (ACL Anthology)

QAGS: Wang et al. (2020). Asking and Answering Questions to Evaluate the Factual Consistency of Summaries. (ACL Anthology)

PEGASUS: Zhang et al. (2020). PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization. (ICML)

BERT: Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers... (arXiv)