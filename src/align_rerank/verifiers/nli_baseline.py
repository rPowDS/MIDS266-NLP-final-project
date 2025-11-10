from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLIScorer:
    def __init__(self, model_name: str = 'roberta-large-mnli', device=None):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()
        self.entail_idx = 2  # MNLI

    @torch.no_grad()
    def score(self, context: str, hypothesis: str) -> float:
        pair = self.tok(context, hypothesis, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        logits = self.model(**pair).logits[0]
        prob = torch.softmax(logits, dim=-1)[self.entail_idx].item()
        return float(prob)
