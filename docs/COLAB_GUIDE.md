# Google Colab Setup Guide

## Overview

This guide explains how to run the align-rerank project on Google Colab for GPU-accelerated training and scoring.

## Why Colab?

- **Free GPU access** (T4, sometimes V100)
- **Faster training**: 4-6 hours vs 20-30 hours on CPU
- **No local resource constraints**
- **Persistent storage** via Google Drive

## Quick Start

### Step 1: Upload Project to Google Drive

1. Zip your project: `zip -r align-rerank-updated.zip .`
2. Upload to Google Drive (e.g., `/MyDrive/W266-NLP/266-NLP_Final_Project/`)

### Step 2: Setup Colab Notebook

Create a new Colab notebook and run:

```python
# Cell 1: Install dependencies
!pip install -q transformers==4.41.2 datasets==2.20.0 evaluate==0.4.2 accelerate==0.31.0
!pip install -q torch>=2.1 numpy==1.26.4 pandas==2.1.4 scipy==1.11.4
!pip install -q tqdm==4.66.4 nltk==3.8.1 typer==0.12.3 rich==13.7.1
!pip install -q bert-score==0.3.13 rouge-score==0.1.2 sentencepiece==0.2.0
!pip install -q sacrebleu==2.4.0 huggingface-hub==0.24.5 orjson==3.10.1

import nltk
nltk.download('punkt', quiet=True)

import torch
print(f"GPU: {torch.cuda.is_available()}")
```

### Step 3: Mount Drive and Extract Project

```python
# Cell 2: Mount Drive and extract
from google.colab import drive
import zipfile
import os
import sys

drive.mount('/content/drive')

# Extract zip (adjust path as needed)
ZIP_PATH = '/content/drive/MyDrive/W266-NLP/266-NLP_Final_Project/align-rerank-updated.zip'

if os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('/content/align-rerank')
    
    # Add to Python path
    sys.path.insert(0, '/content/align-rerank/src')
    
    # Create output directories
    os.makedirs('/content/align-rerank/runs', exist_ok=True)
    os.makedirs('/content/align-rerank/results', exist_ok=True)
    
    print("âœ… Project setup complete!")
else:
    print(f"âŒ File not found: {ZIP_PATH}")
```

### Step 4: Run Pipeline

```python
# Cell 3: Train (4-6 hours)
!cd /content/align-rerank && python -m align_rerank.train_bart \
    --output_dir runs/bart-baseline \
    --num_train_epochs 3

# Cell 4: Generate candidates (2-3 hours)
!cd /content/align-rerank && python -m align_rerank.generate_candidates \
    --model_dir runs/bart-baseline \
    --split validation \
    --decoder top_p \
    --k 8 \
    --out results/val.candidates.jsonl

# Cell 5: Score with FactCC (3-5 hours)
!cd /content/align-rerank && python -m align_rerank.score_factcc \
    --in_jsonl results/val.candidates.jsonl \
    --out_csv results/val.factcc.csv

# Cell 6: Rerank
!cd /content/align-rerank && python -m align_rerank.create_baseline \
    --candidates results/val.candidates.jsonl \
    --out results/val.baseline.jsonl

!cd /content/align-rerank && python -m align_rerank.rerank \
    --candidates results/val.candidates.jsonl \
    --scores results/val.factcc.csv \
    --verifier factcc \
    --out results/val.reranked.factcc.jsonl
```

### Step 5: Download Results

```python
# Cell 7: Download
from google.colab import files

!cd /content/align-rerank && zip -r results_package.zip results/
files.download('/content/align-rerank/results_package.zip')
```

## Tips

1. **Enable GPU**: Runtime â†’ Change runtime type â†’ T4 GPU
2. **Save checkpoints**: Copy to Drive regularly
3. **Session limits**: 12 hours (free tier) - can reconnect
4. **Monitor progress**: Check logs every hour

## Troubleshooting

### Zip extracts to wrong location
If files are in `/content/` instead of `/content/align-rerank/`:
- Use `sys.path.insert(0, '/content/src')` instead
- Update paths accordingly

### Out of memory
- Reduce batch size: `--per_device_train_batch_size 1`
- Use gradient accumulation

### Session disconnects
- Training resumes from last checkpoint automatically
- Restore from Drive if needed

