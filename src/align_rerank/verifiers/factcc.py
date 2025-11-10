from __future__ import annotations
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class FactCCConfig:
    model_name: str = 'roberta-large-mnli'  # graceful fallback if FactCC weights unavailable
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class FactCCScorer:
    """FactCC (PRIMARY verifier) - Simple entailment-style factuality score.
    
    If the true FactCC model is available, set `model_name` to that checkpoint.
    This is the PRIMARY verifier used for reranking candidates.
    """
    def __init__(self, cfg: FactCCConfig = FactCCConfig()):
        self.cfg = cfg
        # Ensure device is set correctly
        if cfg.device == 'cuda' and not torch.cuda.is_available():
            cfg.device = 'cpu'
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name).to(cfg.device).eval()
        # MNLI mapping heuristic: entailment index=2
        self.entail_idx = 2

    @torch.no_grad()
    def score(self, context: str, claim: str) -> float:
        pair = self.tok(context, claim, return_tensors='pt', truncation=True, max_length=512).to(self.cfg.device)
        logits = self.model(**pair).logits[0]
        prob = torch.softmax(logits, dim=-1)[self.entail_idx].item()
        return float(prob)
