from __future__ import annotations
from typing import List, Dict, Any
import typer
from rich import print
from align_rerank.utils.io import read_jsonl, write_csv
from align_rerank.verifiers.alignscore import AlignScorer, AlignScoreConfig

def main(
    in_jsonl: str = typer.Option(..., help='Candidate pool JSONL from generate_candidates.py'),
    out_csv: str = typer.Option('results/alignscore.csv', help='Output CSV with AlignScore values'),
    model_name: str = typer.Option('roberta-large-mnli', help='AlignScore or MNLI checkpoint name'),
    max_chunk_tokens: int = 350,
):
    data = read_jsonl(in_jsonl)
    scorer = AlignScorer(AlignScoreConfig(model_name=model_name, max_chunk_tokens=max_chunk_tokens))
    rows = []
    for row in data:
        ctx = row['article']
        for j, cand in enumerate(row['candidates']):
            score = scorer.score(ctx, cand['text'])
            rows.append({
                'id': row['id'],
                'cand_idx': j,
                'alignscore': score
            })
    write_csv(out_csv, rows)
    print(f'[green]Wrote AlignScore scores to {out_csv}[/green]')

if __name__ == '__main__':
    typer.run(main)
