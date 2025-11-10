from __future__ import annotations
import typer
from rich import print
from align_rerank.utils.io import read_jsonl, write_jsonl

def main(
    candidates: str = typer.Option(..., help='Path to candidates JSONL from generate_candidates.py'),
    out: str = typer.Option('results/baseline.jsonl', help='Output JSONL with first candidate as baseline'),
):
    """Create baseline from candidates file by taking the first candidate for each example."""
    data = read_jsonl(candidates)
    baseline_rows = []
    for row in data:
        # Take first candidate as baseline (typically highest logprob or first in beam)
        baseline_rows.append({
            'id': row['id'],
            'article': row['article'],
            'reference': row['reference'],
            'candidates': [row['candidates'][0]]  # Just first candidate for baseline
        })
    write_jsonl(out, baseline_rows)
    print(f'[green]Created baseline file with {len(baseline_rows)} examples[/green]')
    print(f'[yellow]Baseline uses first candidate from generation (typically highest probability)[/yellow]')

if __name__ == '__main__':
    typer.run(main)

