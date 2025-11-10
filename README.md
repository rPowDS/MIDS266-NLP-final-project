# Verifier-Reranking for Factual Abstractive Summarization

**Solo project** implementing a reproducible pipeline:
1) fine-tune `facebook/bart-base` on CNN/DailyMail,
2) generate K candidates per article,
3) rerank with a factuality verifier (FactCC primary; QAGS secondary),
4) evaluate automatic metrics + human audit,
5) statistics (paired bootstrap + permutation tests).

> Success gate: **â‰¥ +2.0** points on FactCC factuality metric with **â‰¤ 1.0** ROUGE-L drop vs baseline; verified on a 50-item human audit. If reranking harms ROUGE, constrained decoding that favors copying from source is used as a fallback.

## Quick start (reproduce in 4 commands)

```bash
# 0) Setup
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

# 1) Train baseline BART (or skip and use a public checkpoint)
python -m align_rerank.train_bart --output_dir runs/bart-baseline

# 2) Generate candidate pools (K=8, top-p sampler on validation split)
python -m align_rerank.generate_candidates --model_dir runs/bart-baseline --split validation --decoder top_p --k 8 --out results/val.candidates.jsonl

# 3) Score + rerank with FactCC (primary verifier; fallbacks to MNLI if FactCC weights unavailable)
python -m align_rerank.score_factcc --in_jsonl results/val.candidates.jsonl --out_csv results/val.factcc.csv
python -m align_rerank.rerank --candidates results/val.candidates.jsonl --scores results/val.factcc.csv --verifier factcc --out results/val.reranked.factcc.jsonl

# 4) Evaluate automatic metrics (ROUGE, BERTScore, FactCC primary, QAGS secondary) + statistics vs baseline
python -m align_rerank.eval_automatic --baseline results/val.baseline.jsonl --system results/val.reranked.factcc.jsonl --out results/val.metrics.csv
python -m align_rerank.stats --baseline_csv results/val.metrics.csv --out results/val.stats.txt
```

> You can replace `--decoder` with `beam`, `diverse_beam`, `top_p`, or `constrained` (favors copying from source). Try different `--k` values (`4, 8, 16`). If ROUGE drops after reranking, use `--decoder constrained` with `--copy_penalty 1.2` to generate more source-grounded candidates. See `python -m align_rerank.<module> --help` for all options.

## Repository layout

```
align-rerank/
â”œâ”€ src/align_rerank/
â”‚  â”œâ”€ __init__.py
â”‚  â”œâ”€ cli.py                         # unified Typer CLI (optional use)
â”‚  â”œâ”€ train_bart.py                  # fine-tune BART on CNN/DM
â”‚  â”œâ”€ generate_candidates.py         # produce K candidates per doc (beam/diverse-beam/top-p/constrained)
â”‚  â”œâ”€ verifiers/
â”‚  â”‚  â”œâ”€ __init__.py
â”‚  â”‚  â”œâ”€ factcc.py                   # FactCC verifier (PRIMARY) with graceful fallback to MNLI
â”‚  â”‚  â”œâ”€ qags.py                     # QAGS verifier (SECONDARY) - Question Answering and Generation Score
â”‚  â”‚  â”œâ”€ alignscore.py               # AlignScore scoring (additional metric; Eq. 3: mean over sentences of max over chunks)
â”‚  â”‚  â””â”€ nli_baseline.py             # generic 3-way NLI using roberta-large-mnli
â”‚  â”œâ”€ score_factcc.py                # batch scoring of (context, candidate) pairs using FactCC (PRIMARY)
â”‚  â”œâ”€ score_alignscore.py            # batch scoring of (context, candidate) pairs using AlignScore
â”‚  â”œâ”€ rerank.py                      # select highest-scoring candidate by chosen verifier
â”‚  â”œâ”€ eval_automatic.py              # ROUGE-1/2/L, BERTScore, FactCC (primary), QAGS (secondary), AlignScore (additional)
â”‚  â”œâ”€ eval_human.py                  # 50-item audit sheet producer + simple instructions
â”‚  â”œâ”€ stats.py                       # paired bootstrap CIs; permutation test
â”‚  â””â”€ utils/
â”‚     â”œâ”€ __init__.py
â”‚     â”œâ”€ data.py                     # dataset loader & preprocessing
â”‚     â”œâ”€ text.py                     # sentence splitting, chunking (~350 tokens, sentence boundaries)
â”‚     â”œâ”€ io.py                       # JSONL/CSV helpers
â”‚     â””â”€ tables.py                   # pretty printing and CSV schema
â”œâ”€ configs/
â”‚  â”œâ”€ train.bart.base.json
â”‚  â”œâ”€ generate.top_p.k8.json
â”‚  â””â”€ eval.metrics.json
â”œâ”€ tests/
â”‚  â”œâ”€ test_chunking.py
â”‚  â”œâ”€ test_alignscore_reduction.py
â”‚  â””â”€ test_io.py
â”œâ”€ paper/
â”‚  â”œâ”€ paper.tex
â”‚  â””â”€ refs.bib
â”œâ”€ scripts/
â”‚  â”œâ”€ run_train.sh
â”‚  â”œâ”€ run_generate.sh
â”‚  â”œâ”€ run_score_rerank.sh
â”‚  â””â”€ run_eval.sh
â”œâ”€ requirements.txt
â”œâ”€ pyproject.toml
â”œâ”€ .pre-commit-config.yaml
â”œâ”€ LICENSE
â””â”€ README.md
```

### Notes

- **FactCC** (PRIMARY) uses 3-way NLI entailment probability to score factuality. If official FactCC weights aren't found, we fallback to `roberta-large-mnli` with a clear warning. This is the main verifier used for reranking.
- **QAGS** (SECONDARY) generates questions from summaries, answers them from source documents, and computes F1 overlap. Uses T5 for question generation and DistilBERT for QA, with graceful fallbacks if models are unavailable.
- **AlignScore** (additional metric) follows the chunkâ€“sentence **mean-of-max** (Eq. 3): for each summary sentence, take the highest **ALIGNED** probability across all ~350-token context chunks; then average across sentences. Reported as an additional metric but not used for primary reranking.
- **Constrained decoding** is available as a fallback if reranking harms ROUGE. Uses `--decoder constrained` with `--copy_penalty` to favor copying tokens from the source document.
- **Stats** includes paired bootstrap CIs for ROUGE and a permutation test for factuality deltas.
- **Human audit** writes a CSV you can fill; it also exports Markdown instructions for raters with error taxonomy (entity, number, date, unsupported, contradiction, omission).

**Dataset**: CNN/DailyMail (Hugging Face ID: `cnn_dailymail`, config `3.0.0`) with standard train/validation/test splits.

See the **paper** folder for a LaTeX skeleton aligned with the pipeline.

## Google Colab Setup

For GPU-accelerated training and scoring, use Google Colab. See `notebooks/COLAB_SETUP.ipynb` for a complete setup notebook.

Quick setup:
```python
from notebooks.colab_setup import setup_colab
result = setup_colab(zip_path='/path/to/align-rerank-updated.zip')
```

Or use the manual setup in `COLAB_SETUP.md`.

