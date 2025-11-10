# Project Status & Completion Guide

## Current State Assessment

### âœ… **COMPLETE (Code Infrastructure)**
You have **100% of the code infrastructure** written and ready to run. This includes:

1. **Training Module** (`train_bart.py`) - Fine-tune BART on CNN/DailyMail
2. **Generation Module** (`generate_candidates.py`) - Generate K candidates with multiple decoders
3. **Scoring Modules**:
   - `score_factcc.py` - Batch FactCC scoring (PRIMARY)
   - `score_alignscore.py` - Batch AlignScore scoring (additional)
4. **Reranking Module** (`rerank.py`) - Select best candidate by verifier score
5. **Evaluation Modules**:
   - `eval_automatic.py` - Compute ROUGE, BERTScore, FactCC, QAGS
   - `eval_human.py` - Generate 50-item human audit sheet
   - `stats.py` - Statistical tests (paired bootstrap, permutation)
6. **Verifiers**:
   - `factcc.py` - PRIMARY verifier
   - `qags.py` - SECONDARY verifier  
   - `alignscore.py` - Additional metric
7. **Utilities** - Data loading, text processing, I/O helpers
8. **Baseline Creation** (`create_baseline.py`) - Extract baseline from candidates

### âŒ **NOT STARTED (Experiments & Results)**
You have **0% of actual experimental results**. This is the critical gap:

- âŒ No trained model
- âŒ No generated candidates
- âŒ No scoring results
- âŒ No reranked outputs
- âŒ No evaluation metrics
- âŒ No statistical analysis
- âŒ No human evaluation
- âŒ No error analysis
- âŒ No visualizations/plots
- âŒ No paper written

---

## What You Need To Do

### Phase 1: Setup & Data Preparation (2-4 hours)

#### 1.1 Environment Setup
```bash
cd /Users/ryanpowers/Downloads/align-rerank
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -U pip
pip install -r requirements.txt
```

#### 1.2 Verify Dataset Access
```bash
python -c "from datasets import load_dataset; ds = load_dataset('cnn_dailymail', '3.0.0'); print(f'Train: {len(ds[\"train\"])}, Val: {len(ds[\"validation\"])}, Test: {len(ds[\"test\"])}')"
```
Expected: Train: 287,113, Val: 13,368, Test: 11,490

#### 1.3 Test Pipeline Components
```bash
# Test each module loads correctly
python -m align_rerank.train_bart --help
python -m align_rerank.generate_candidates --help
python -m align_rerank.score_factcc --help
python -m align_rerank.eval_automatic --help
```

---

### Phase 2: Training Baseline Model (4-8 hours runtime + 2-3 hours monitoring)

#### 2.1 Train BART Model
```bash
python -m align_rerank.train_bart \
    --output_dir runs/bart-baseline \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2
```

**Time Estimate:**
- With GPU (CUDA): **4-6 hours**
- Without GPU (CPU): **20-30 hours** (not recommended)
- **Monitoring**: Check logs every 30-60 minutes

**Checkpoints:**
- Model saves every 2000 steps
- Final model at `runs/bart-baseline/`
- Monitor validation ROUGE in logs

**Alternative:** If training is too slow, you can:
- Use a pre-trained checkpoint (if available)
- Reduce epochs to 1-2 for initial testing
- Use a smaller subset for development

#### 2.2 Verify Training Completed
```bash
ls runs/bart-baseline/
# Should see: config.json, pytorch_model.bin, tokenizer files
```

---

### Phase 3: Generate Candidates (2-4 hours runtime)

#### 3.1 Generate Candidate Pool
```bash
# Validation split (recommended for initial experiments)
python -m align_rerank.generate_candidates \
    --model_dir runs/bart-baseline \
    --split validation \
    --decoder top_p \
    --k 8 \
    --out results/val.candidates.jsonl \
    --batch_size 4

# Test split (for final evaluation)
python -m align_rerank.generate_candidates \
    --model_dir runs/bart-baseline \
    --split test \
    --decoder top_p \
    --k 8 \
    --out results/test.candidates.jsonl \
    --batch_size 4
```

**Time Estimate:**
- Validation (13,368 examples Ã— 8 candidates): **2-3 hours** with GPU
- Test (11,490 examples Ã— 8 candidates): **2-3 hours** with GPU
- **Monitor**: Progress prints every 10 batches

#### 3.2 Create Baseline
```bash
python -m align_rerank.create_baseline \
    --candidates results/val.candidates.jsonl \
    --out results/val.baseline.jsonl
```

---

### Phase 4: Scoring & Reranking (3-6 hours runtime)

#### 4.1 Score with FactCC (PRIMARY)
```bash
python -m align_rerank.score_factcc \
    --in_jsonl results/val.candidates.jsonl \
    --out_csv results/val.factcc.csv
```

**Time Estimate:**
- Validation set: **3-5 hours** (scoring ~107K candidates)
- Each candidate requires NLI model forward pass
- **Monitor**: Progress not currently printed (add if needed)

#### 4.2 Rerank Candidates
```bash
python -m align_rerank.rerank \
    --candidates results/val.candidates.jsonl \
    --scores results/val.factcc.csv \
    --verifier factcc \
    --out results/val.reranked.factcc.jsonl
```

**Time Estimate:** < 5 minutes (just file I/O)

---

### Phase 5: Evaluation & Statistics (2-4 hours runtime)

#### 5.1 Automatic Metrics
```bash
python -m align_rerank.eval_automatic \
    --baseline results/val.baseline.jsonl \
    --system results/val.reranked.factcc.jsonl \
    --out results/val.metrics.csv
```

**Time Estimate:**
- ROUGE: < 1 minute
- BERTScore: 30-60 minutes
- FactCC (re-scoring): 2-3 hours
- QAGS: 3-4 hours (slower, uses QG+QA models)

**Total: 5-8 hours** for full evaluation

#### 5.2 Statistical Analysis
```bash
python -m align_rerank.stats \
    --baseline_jsonl results/val.baseline.jsonl \
    --system_jsonl results/val.reranked.factcc.jsonl \
    --out results/val.stats.txt
```

**Time Estimate:** 1-2 hours (includes per-example FactCC scoring

---

### Phase 6: Human Evaluation (4-6 hours)

#### 6.1 Generate Audit Sheet
```bash
python -m align_rerank.eval_human \
    --system results/val.reranked.factcc.jsonl \
    --out_csv results/human_audit.csv \
    --n 50
```

#### 6.2 Manual Annotation (YOU DO THIS)
- Open `results/human_audit.csv`
- For each of 50 examples:
  - Read article, reference, and summary
  - Mark `is_factual` (0 or 1)
  - Tag error types if not factual
  - Add notes
- **Time Estimate:** 4-6 hours (5-7 minutes per example)

#### 6.3 Analyze Human Results
- Create script or notebook to compute:
  - Factual accuracy (%)
  - Error type distribution
  - Comparison with automatic metrics

---

### Phase 7: Error Analysis & Ablations (8-12 hours)

#### 7.1 Error Analysis
You need to:
1. **Identify failure modes:**
   - Load examples where reranked summary is worse than baseline
   - Categorize errors (entity errors, numbers, dates, contradictions, etc.)
   - Compare with human audit results

2. **Create analysis notebook:**
   - Sample 20-30 error cases
   - Show article, reference, baseline summary, reranked summary
   - Explain why reranking failed/succeeded
   - Identify patterns

**Time Estimate:** 4-6 hours

#### 7.2 Ablation Studies (if time permits)
- Test different K values (4, 8, 16)
- Compare decoders (beam, top_p, constrained)
- Test constrained decoding if ROUGE drops
- Compare FactCC vs QAGS vs AlignScore

**Time Estimate:** 4-6 hours (running experiments)

---

### Phase 8: Visualizations & Plots (2-4 hours)

Create plots for paper:
1. **Metric comparisons:**
   - Bar charts: ROUGE-1/2/L, BERTScore, FactCC, QAGS
   - Baseline vs Reranked

2. **Statistical results:**
   - Confidence intervals for FactCC gains
   - Distribution of per-example FactCC scores

3. **Error analysis:**
   - Pie chart of error types
   - Examples of good/bad reranking

4. **Human evaluation:**
   - Factual accuracy comparison
   - Correlation between automatic and human scores

**Time Estimate:** 2-4 hours

---

### Phase 9: Paper Writing (12-16 hours)

#### 9.1 Structure (4-6 pages)
1. **Abstract** (150-200 words)
   - Problem statement
   - Approach
   - Key results

2. **Introduction** (0.5-1 page)
   - Motivation: hallucinations in abstractive summarization
   - Your approach: verifier-reranking
   - Contributions

3. **Background/Related Work** (1-1.5 pages)
   - Factuality in summarization
   - Verifier methods (FactCC, QAGS, AlignScore)
   - Reranking approaches
   - **4+ references required**

4. **Methods** (1-1.5 pages)
   - Dataset: CNN/DailyMail
   - Training: BART-base fine-tuning
   - Generation: K candidates with various decoders
   - Reranking: FactCC as primary verifier
   - Evaluation: metrics, human audit

5. **Results & Discussion** (1.5-2 pages)
   - Automatic metrics (tables)
   - Statistical significance
   - Human evaluation results
   - Error analysis with examples
   - Ablations (if any)

6. **Conclusion** (0.5 page)
   - Summary of findings
   - Limitations
   - Future work

#### 9.2 Writing Tips
- **Focus on analysis, not just numbers**
- Show example errors and explain why
- Connect results to methodology
- Compare with related work
- Use tables and figures effectively

**Time Estimate:** 12-16 hours

---

### Phase 10: Presentation Prep (4-6 hours)

- 6-minute presentation
- Slides: 8-10 slides
- Practice: 2-3 run-throughs

**Time Estimate:** 4-6 hours

---

## Time Estimates Summary

### Minimum Viable Project (Basic Results)
- Setup: 2-4 hours
- Training: 4-8 hours (runtime)
- Generation: 2-4 hours (runtime)
- Scoring: 3-6 hours (runtime)
- Evaluation: 2-4 hours (runtime)
- Human eval: 4-6 hours (manual)
- Error analysis: 4-6 hours
- Paper writing: 12-16 hours
- **Total: 37-54 hours** (1-2 weeks full-time)

### Full Project (With Ablations & Deep Analysis)
- All above + Ablations: +4-6 hours
- Enhanced visualizations: +2-3 hours
- Presentation: +4-6 hours
- **Total: 47-69 hours** (2-3 weeks full-time)

---

## Critical Path (What to Do First)

1. **Week 1: Get Results**
   - Day 1-2: Setup + Training
   - Day 3-4: Generation + Scoring
   - Day 5-7: Evaluation + Statistics

2. **Week 2: Analysis & Writing**
   - Day 1-2: Human evaluation
   - Day 3-4: Error analysis
   - Day 5-7: Paper writing

3. **Week 3: Polish**
   - Day 1-2: Ablations (if time)
   - Day 3-4: Visualizations
   - Day 5-6: Presentation prep
   - Day 7: Final edits

---

## Success Criteria (From Your Proposal)

âœ… **Must Achieve:**
- â‰¥ +2.0 points on FactCC (validation)
- â‰¤ 1.0 ROUGE-L drop vs baseline
- 50-item human audit completed

**If ROUGE drops too much:**
- Use constrained decoding (`--decoder constrained`)
- Re-run generation and reranking
- Report both results in paper

---

## Potential Issues & Solutions

### Issue 1: Training Takes Too Long
- **Solution:** Use fewer epochs (1-2) for initial testing, or use pre-trained checkpoint if available

### Issue 2: Scoring Takes Too Long
- **Solution:** 
  - Score on subset first (1000 examples)
  - Test on smaller K (K=4 instead of K=8)
  - Use GPU if available

### Issue 3: Results Don't Meet Success Criteria
- **Solution:**
  - Try different decoders (beam, constrained)
  - Try different K values
  - Analyze why reranking isn't working
  - Report honest results with analysis

### Issue 4: QAGS Model Not Available
- **Solution:** 
  - QAGS has fallbacks built in
  - Focus on FactCC as primary
  - Mention QAGS limitations in paper

---

## Code Completeness: 100%
## Results Completeness: 0%
## **Your Next Step: Start Phase 2 (Training)**

Good luck! The infrastructure is solid. Now you need to run experiments and analyze results.

