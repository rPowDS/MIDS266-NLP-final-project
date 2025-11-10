from __future__ import annotations
from typing import List, Dict, Any
import typer
from rich import print
from align_rerank.utils.io import read_jsonl, write_csv
from align_rerank.verifiers.factcc import FactCCScorer, FactCCConfig

def main(
    in_jsonl: str = typer.Option(..., help='Candidate pool JSONL from generate_candidates.py'),
    out_csv: str = typer.Option('results/factcc.csv', help='Output CSV with FactCC values'),
    model_name: str = typer.Option('roberta-large-mnli', help='FactCC or MNLI checkpoint name'),
):
    data = read_jsonl(in_jsonl)
    scorer = FactCCScorer(FactCCConfig(model_name=model_name))
    rows = []
    for row in data:
        ctx = row['article']
        for j, cand in enumerate(row['candidates']):
            score = scorer.score(ctx, cand['text'])
            rows.append({
                'id': row['id'],
                'cand_idx': j,
                'factcc': score
            })
    write_csv(out_csv, rows)
    print(f'[green]Wrote FactCC scores to {out_csv}[/green]')

if __name__ == '__main__':
    typer.run(main)

