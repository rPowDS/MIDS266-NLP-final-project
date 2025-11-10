# Setup Guide

## Local Setup

### Prerequisites
- Python 3.10+
- pip
- (Optional) CUDA-capable GPU for faster training

### Installation

```bash
# Clone or download project
cd align-rerank

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True)"
```

### Verify Installation

```bash
# Check GPU (if available)
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Test imports
python -c "from align_rerank import train_bart; print('âœ… Import successful')"
```

## Google Colab Setup

### Method 1: Using Setup Script (Recommended)

1. Upload `align-rerank-updated.zip` to Google Drive
2. Open `notebooks/COLAB_SETUP.ipynb` in Colab
3. Run cells sequentially

### Method 2: Manual Setup

1. Mount Google Drive
2. Extract zip file
3. Install dependencies
4. Run pipeline

See `COLAB_SETUP.md` for detailed instructions.

## Project Structure

After setup, your directory should look like:

```
align-rerank/
â”œâ”€â”€ src/align_rerank/      # Source code
â”œâ”€â”€ configs/               # Configuration files
â”œâ”€â”€ scripts/               # Shell scripts
â”œâ”€â”€ notebooks/              # Colab setup utilities
â”œâ”€â”€ tests/                  # Unit tests
â”œâ”€â”€ paper/                  # LaTeX paper
â”œâ”€â”€ runs/                   # Model checkpoints (created during training)
â””â”€â”€ results/                # Experiment results (created during execution)
```

## Next Steps

After setup, proceed to:
1. Training: `python -m align_rerank.train_bart --output_dir runs/bart-baseline`
2. Generation: `python -m align_rerank.generate_candidates ...`
3. Scoring: `python -m align_rerank.score_factcc ...`
4. Evaluation: `python -m align_rerank.eval_automatic ...`

See `README.md` for complete pipeline instructions.

