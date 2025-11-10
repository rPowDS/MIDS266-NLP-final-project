from align_rerank.utils.text import chunk_text_by_tokens, split_sentences
from transformers import AutoTokenizer

def test_chunking_basic():
    tok = AutoTokenizer.from_pretrained('roberta-base')
    text = ' '.join(['Sentence.']*100)
    chunks = chunk_text_by_tokens(text, tok, max_tokens=10)
    assert len(chunks) >= 5

def test_sentence_split():
    sents = split_sentences("Hello world. Another one? Yes!")
    assert len(sents) == 3
