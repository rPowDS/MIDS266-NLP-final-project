from __future__ import annotations
from typing import List, Tuple
import nltk
import re

# Ensure punkt is available for sentence splitting
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text: str) -> List[str]:
    """Split into sentences with NLTK; fall back to a regex if needed."""
    try:
        from nltk.tokenize import sent_tokenize
        sents = sent_tokenize(text.strip())
        return [s.strip() for s in sents if s.strip()]
    except Exception:
        sents = _SENT_SPLIT.split(text.strip())
        return [s.strip() for s in sents if s.strip()]

def chunk_tokens(tokens: List[int], max_len: int) -> List[List[int]]:
    return [tokens[i:i+max_len] for i in range(0, len(tokens), max_len)]

def chunk_text_by_tokens(text: str, tokenizer, max_tokens: int = 350) -> List[str]:
    """Split text into ~max_tokens chunks at sentence boundaries when possible."""
    sentences = split_sentences(text)
    chunks, current = [], []
    current_len = 0
    for sent in sentences:
        sent_ids = tokenizer.encode(sent, add_special_tokens=False)
        if current_len + len(sent_ids) > max_tokens and current:
            chunks.append(' '.join(current))
            current = [sent]
            current_len = len(sent_ids)
        else:
            current.append(sent)
            current_len += len(sent_ids)
    if current:
        chunks.append(' '.join(current))
    # if no sentences were found or tokenization failed, fall back to blunt token chunking
    if not chunks:
        ids = tokenizer.encode(text, add_special_tokens=False)
        for slice_ids in chunk_tokens(ids, max_tokens):
            chunks.append(tokenizer.decode(slice_ids))
    return chunks
