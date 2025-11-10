from __future__ import annotations
from typing import Dict, Tuple
import csv
import typer
from rich import print
from align_rerank.utils.io import read_jsonl, write_jsonl

def load_scores(path: str) -> Dict[Tuple[int,int], float]:
    scores = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (int(r['id']), int(r['cand_idx']))
            scores[key] = float(r.get('alignscore') or r.get('factcc') or r.get('score', 0.0))
    return scores

def main(
    candidates: str = typer.Option(..., help='Path to candidates JSONL'),
    scores: str = typer.Option(..., help='CSV from score_alignscore or score_factcc'),
    verifier: str = typer.Option('factcc', help='factcc|alignscore|combo'),
    out: str = typer.Option('results/reranked.jsonl'),
):
    pool = read_jsonl(candidates)
    sc = load_scores(scores)
    out_rows = []
    for row in pool:
        best_idx, best_score = None, -1e9
        for j, cand in enumerate(row['candidates']):
            s = sc.get((row['id'], j), float('-inf'))
            if s > best_score:
                best_idx, best_score = j, s
        out_rows.append({
            'id': row['id'],
            'article': row['article'],
            'reference': row['reference'],
            'chosen': row['candidates'][best_idx],
            'chosen_verifier': verifier,
            'chosen_score': best_score,
            'all_candidates': row['candidates']  # keep for analysis
        })
    write_jsonl(out, out_rows)
    print(f'[green]Saved reranked system outputs to {out}[/green]')

if __name__ == '__main__':
    typer.run(main)
