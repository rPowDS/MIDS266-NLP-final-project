# Import Update Guide

After reorganizing the project structure, you'll need to update imports in your Python files.

## Old Structure → New Structure

### Main Modules
```python
# OLD
from train_bart import main
from score_factcc import FactCCScorer

# NEW
from align_rerank.train_bart import main
from align_rerank.score_factcc import FactCCScorer
```

### Verifiers
```python
# OLD
from factcc import FactCCScorer
from alignscore import AlignScorer

# NEW
from align_rerank.verifiers.factcc import FactCCScorer
from align_rerank.verifiers.alignscore import AlignScorer
```

### Utilities
```python
# OLD
from io import read_jsonl, write_csv
from text import split_sentences

# NEW
from align_rerank.utils.io import read_jsonl, write_csv
from align_rerank.utils.text import split_sentences
```

## Installation

After reorganizing, install the package in development mode:
```bash
pip install -e .
```

This allows you to import modules as:
```python
from align_rerank import train_bart
from align_rerank.verifiers import factcc
from align_rerank.utils import io
```

## Running Scripts

After reorganization, run scripts using:
```bash
# Using module notation
python -m align_rerank.train_bart --output_dir runs/bart-baseline

# Or if installed
align-rerank train-bart --output_dir runs/bart-baseline
```
