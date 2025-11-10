from __future__ import annotations
import math, json
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import typer
from rich import print
from align_rerank.utils.io import write_jsonl

def sequence_logprob(scores: List[torch.Tensor], sequences: torch.LongTensor) -> List[float]:
    # scores: list of [batch*num_return, vocab] tensors at each timestep
    # sequences: [batch*num_return, seq_len]
    # We compute logprob over generated tokens only (ignore BOS)
    logprobs = []
    # Align: scores[t] gives logits for token t+1 (after first generated token)
    # Take softmax at each step, gather probs for the actual token id, then sum log
    for b in range(sequences.shape[0]):
        lp = 0.0
        # Ignore initial context tokens; we only consider the generated segment
        # scores length equals generated length
        for t, step in enumerate(scores):
            token_id = sequences[b, -(len(scores)-t)]
            prob = torch.log_softmax(step[b], dim=-1)[token_id].item()
            lp += prob
        logprobs.append(lp)
    return logprobs

def main(
    model_dir: str = typer.Option(..., help='Path to fine-tuned model directory'),
    split: str = typer.Option('validation', help='Split: validation|test'),
    decoder: str = typer.Option('top_p', help='beam|diverse_beam|top_p|constrained'),
    k: int = typer.Option(8, help='Number of candidates per example'),
    out: str = typer.Option('results/candidates.jsonl', help='Output JSONL file'),
    max_new_tokens: int = 128,
    batch_size: int = 4,
    copy_penalty: float = typer.Option(1.2, help='Copy penalty multiplier for constrained decoding (higher = favor copying)'),
):
    # Use standard dataset name (same as notebook: cnn_dailymail maps to ccdv/cnn_dailymail)
    ds = load_dataset('cnn_dailymail', '3.0.0')
    data = ds[split]
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    rows = []
    for start in range(0, len(data), batch_size):
        batch = data[start:start+batch_size]
        inputs = tok(list(batch['article']), return_tensors='pt', truncation=True, padding=True, max_length=1024).to(device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True)
        if decoder == 'beam':
            gen_kwargs.update(dict(num_beams=max(4, k), num_return_sequences=k, early_stopping=True))
        elif decoder == 'diverse_beam':
            groups = min(4, k)
            gen_kwargs.update(dict(num_beams=max(groups*2, k), num_beam_groups=groups, diversity_penalty=0.3, num_return_sequences=k, early_stopping=True))
        elif decoder == 'top_p':
            gen_kwargs.update(dict(do_sample=True, top_p=0.92, top_k=0, num_return_sequences=k))
        elif decoder == 'constrained':
            # Constrained decoding: favor copying from source by boosting source token probabilities
            # This is a simplified version - in practice, you'd modify logits during generation
            # For now, we use a penalty on non-source tokens via repetition_penalty and top_p
            gen_kwargs.update(dict(
                do_sample=True, 
                top_p=0.95,  # Slightly higher top_p to allow more source tokens
                repetition_penalty=1.0 / copy_penalty,  # Lower penalty = more copying
                num_return_sequences=k
            ))
        else:
            raise ValueError('Unknown decoder')

        with torch.no_grad():
            out_seq = model.generate(**inputs, **gen_kwargs)
        texts = tok.batch_decode(out_seq.sequences, skip_special_tokens=True)
        # compute logprobs
        lps = sequence_logprob(out_seq.scores, out_seq.sequences)

        # group by original example
        per_example = k
        for i, (article, ref) in enumerate(zip(batch['article'], batch['highlights'])):
            cand = []
            for j in range(per_example):
                idx = i*per_example + j
                cand.append({
                    'text': texts[idx],
                    'decoder': decoder,
                    'params': {k2: v for k2, v in gen_kwargs.items() if k2 not in ['return_dict_in_generate','output_scores']},
                    'length': len(tok.encode(texts[idx], add_special_tokens=False)),
                    'logprob': lps[idx],
                })
            rows.append({
                'id': int(start + i),
                'article': article,
                'reference': ref,
                'candidates': cand
            })
        if (start // batch_size) % 10 == 0:
            print(f'Generated up to {start+batch_size} / {len(data)} examples')

    write_jsonl(out, rows)
    print(f'[green]Saved candidates to {out}[/green]')

if __name__ == '__main__':
    typer.run(main)
