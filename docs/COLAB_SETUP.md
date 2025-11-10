# Google Colab Setup Guide

## Quick Start: Copy this into a Colab notebook

### Step 1: Setup (Run this first)

```python
# Install dependencies
!pip install -q transformers==4.41.2 datasets==2.20.0 evaluate==0.4.2 accelerate==0.31.0
!pip install -q torch>=2.1 numpy==1.26.4 pandas==2.1.4 scipy==1.11.4
!pip install -q tqdm==4.66.4 nltk==3.8.1 typer==0.12.3 rich==13.7.1
!pip install -q bert-score==0.3.13 rouge-score==0.1.2 sentencepiece==0.2.0
!pip install -q sacrebleu==2.4.0 huggingface-hub==0.24.5 orjson==3.10.1

# Download NLTK data
import nltk
nltk.download('punkt', quiet=True)

# Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Step 2: Upload Project to Colab

**Option A: Upload as ZIP (Recommended for first run)**
```python
# Upload your project folder
from google.colab import files
import zipfile
import os

# You'll need to zip your align-rerank folder locally first
# Then upload it here
uploaded = files.upload()
for fn in uploaded.keys():
    if fn.endswith('.zip'):
        with zipfile.ZipFile(fn, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(fn)
        print(f"Extracted {fn}")

# Add to Python path
import sys
sys.path.insert(0, 'align-rerank/src')
```

**Option B: Clone from GitHub (If you push to GitHub)**
```python
# If you have the project on GitHub:
# !git clone https://github.com/yourusername/align-rerank.git
# sys.path.insert(0, 'align-rerank/src')
```

**Option C: Manual upload (For code changes)**
```python
# Upload individual files if needed
from google.colab import files
files.upload()
```

### Step 3: Create directories
```python
import os
os.makedirs('runs', exist_ok=True)
os.makedirs('results', exist_ok=True)
```

---

## Running Experiments

### Training (4-6 hours - Run this first!)

```python
# IMPORTANT: This will take 4-6 hours
# Run it and let it go overnight or during a long break
# Save checkpoints every 2000 steps automatically

!cd align-rerank && python -m align_rerank.train_bart \
    --output_dir runs/bart-baseline \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2

# After training, download the model
from google.colab import files
!zip -r runs/bart-baseline.zip runs/bart-baseline/
files.download('runs/bart-baseline.zip')
```

### Generation (2-3 hours)
```python
!cd align-rerank && python -m align_rerank.generate_candidates \
    --model_dir runs/bart-baseline \
    --split validation \
    --decoder top_p \
    --k 8 \
    --out results/val.candidates.jsonl \
    --batch_size 4

# Download results
!zip results/val.candidates.jsonl.zip results/val.candidates.jsonl
files.download('results/val.candidates.jsonl.zip')
```

### Scoring (3-5 hours)
```python
!cd align-rerank && python -m align_rerank.score_factcc \
    --in_jsonl results/val.candidates.jsonl \
    --out_csv results/val.factcc.csv

# Download scores
files.download('results/val.factcc.csv')
```

### Reranking (5 minutes)
```python
!cd align-rerank && python -m align_rerank.create_baseline \
    --candidates results/val.candidates.jsonl \
    --out results/val.baseline.jsonl

!cd align-rerank && python -m align_rerank.rerank \
    --candidates results/val.candidates.jsonl \
    --scores results/val.factcc.csv \
    --verifier factcc \
    --out results/val.reranked.factcc.jsonl

# Download reranked results
files.download('results/val.reranked.factcc.jsonl')
files.download('results/val.baseline.jsonl')
```

### Evaluation (5-8 hours - can run locally)
```python
# This can also be done locally since it's slower
# But if running on Colab:
!cd align-rerank && python -m align_rerank.eval_automatic \
    --baseline results/val.baseline.jsonl \
    --system results/val.reranked.factcc.jsonl \
    --out results/val.metrics.csv

files.download('results/val.metrics.csv')
```

---

## Managing Colab Sessions

### Extending Session Time
- Free tier: 12 hours max (can reconnect)
- Click "Runtime" â†’ "Change runtime type" â†’ "High-RAM" if needed
- Pro tip: Add `!nvidia-smi` cell to check GPU usage

### Saving Checkpoints
```python
# Save model checkpoints to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoints to Drive
!cp -r runs/bart-baseline /content/drive/MyDrive/align-rerank-checkpoints/

# Restore from Drive
!cp -r /content/drive/MyDrive/align-rerank-checkpoints/bart-baseline runs/
```

### Reconnecting After Disconnect
```python
# If session disconnects, re-run setup cells
# Then restore checkpoints from Drive or re-download
# Training will resume from last checkpoint if using transformers Trainer
```

---

## Hybrid Approach (Recommended)

**Best strategy for your timeline:**

1. **Colab for:**
   - Training (4-6 hours) - **MUST be on GPU**
   - Generation (2-3 hours) - **MUST be on GPU**
   - Scoring (3-5 hours) - **MUST be on GPU**

2. **Local for:**
   - Evaluation (can run on CPU, slower but acceptable)
   - Error analysis
   - Human evaluation
   - Paper writing
   - Statistical tests

**Workflow:**
1. Train on Colab â†’ Download model
2. Generate on Colab â†’ Download candidates
3. Score on Colab â†’ Download scores
4. Download all results locally
5. Do analysis locally (no GPU needed)

---

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
--per_device_train_batch_size 1

# Or use gradient accumulation
# (Add to train_bart.py if needed)
```

### Session Timeout
- Save checkpoints to Drive every hour
- Use `!cp runs/bart-baseline /content/drive/MyDrive/` regularly
- Training resumes from last checkpoint automatically

### Can't Import Module
```python
# Make sure path is correct
import sys
sys.path.insert(0, 'align-rerank/src')
# Or
import sys
sys.path.insert(0, '/content/align-rerank/src')
```

---

## Quick Checklist

- [ ] Upload project to Colab (zip or GitHub)
- [ ] Install dependencies
- [ ] Check GPU available
- [ ] Start training (let run 4-6 hours)
- [ ] Save model to Drive
- [ ] Generate candidates
- [ ] Score with FactCC
- [ ] Download all results
- [ ] Continue analysis locally

