from __future__ import annotations
import random, csv
import typer
from align_rerank.utils.io import read_jsonl

TAXONOMY = ['entity', 'number', 'date', 'unsupported', 'contradiction', 'omission']

def main(
    system: str = typer.Option(..., help='JSONL reranked outputs'),
    out_csv: str = typer.Option('results/human_audit.csv'),
    n: int = 50,
    seed: int = 2025,
):
    random.seed(seed)
    data = read_jsonl(system)
    sample = random.sample(data, min(n, len(data)))
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id','article','reference','summary','is_factual(0/1)','error_types(comma-separated)','notes'])
        for row in sample:
            writer.writerow([row['id'], row['article'], row['reference'], row['chosen']['text'], '', '', ''])
    instr = out_csv.replace('.csv', '.instructions.md')
    with open(instr, 'w', encoding='utf-8') as f:
        f.write('# Human Audit Instructions\n')
        f.write('Mark `is_factual` as 1 if the summary is fully supported by the article; else 0.\n')
        f.write('If not factual, tag error types from: ' + ', '.join(TAXONOMY) + '.\n')
    print(f'Wrote audit sheet: {out_csv} and {instr}')

if __name__ == '__main__':
    typer.run(main)
