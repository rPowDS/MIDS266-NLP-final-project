from __future__ import annotations
from dataclasses import dataclass
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from align_rerank.utils.text import split_sentences, chunk_text_by_tokens

@dataclass
class AlignScoreConfig:
    model_name: str = 'roberta-large-mnli'  # Fallback if official AlignScore weights are unavailable
    max_chunk_tokens: int = 350
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class AlignScorer:
    """Implements Eq. (3) mean-over-sentences of max-over-chunks of p(ALIGNED).

    If the official AlignScore model weights are installed locally or on HF Hub, set
    `model_name` accordingly (e.g., 'yuh-zha/AlignScore-large'). The code will work
    with any 3-way NLI head that orders labels as [CONTRADICTION, NEUTRAL, ENTAILMENT]
    or [NEUTRAL, CONTRADICTION, ENTAILMENT] (we detect by name contains 'mnli').
    """
    def __init__(self, cfg: AlignScoreConfig = AlignScoreConfig()):
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name).to(cfg.device).eval()

        # Label mapping heuristics
        self.entail_idx = None
        label_map = self.model.config.id2label
        for i, name in label_map.items():
            if 'entail' in name.lower() or name.lower().startswith('aligned'):
                self.entail_idx = i
                break
        if self.entail_idx is None:
            # Default MNLI: 2 is entailment
            self.entail_idx = 2

    @torch.no_grad()
    def sentence_chunk_score(self, context: str, sentence: str) -> float:
        # split context into ~350-token chunks
        chunks = chunk_text_by_tokens(context, tokenizer=self.tok, max_tokens=self.cfg.max_chunk_tokens)
        if not chunks:
            chunks = [context]
        scores = []
        for ch in chunks:
            pair = self.tok(ch, sentence, return_tensors='pt', truncation='only_first', max_length=512).to(self.cfg.device)
            logits = self.model(**pair).logits[0]
            prob = torch.softmax(logits, dim=-1)[self.entail_idx].item()
            scores.append(prob)
        return max(scores)  # max over chunks

    @torch.no_grad()
    def score(self, context: str, claim: str) -> float:
        sentences = split_sentences(claim)
        if not sentences:
            return self.sentence_chunk_score(context, claim)
        vals = [self.sentence_chunk_score(context, s) for s in sentences]
        return float(sum(vals) / len(vals))
