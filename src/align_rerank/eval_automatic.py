from __future__ import annotations
from typing import List, Dict
import typer
from rich import print
import evaluate
from align_rerank.utils.io import read_jsonl, write_csv
from transformers import AutoTokenizer
from align_rerank.verifiers.alignscore import AlignScorer
from align_rerank.verifiers.factcc import FactCCScorer
from align_rerank.verifiers.qags import QAGSScorer

def collect_texts(path: str):
    data = read_jsonl(path)
    preds, refs, arts = [], [], []
    for row in data:
        if 'chosen' in row:
            preds.append(row['chosen']['text'])
        else:
            # baseline file can be produced by saving the first candidate or separate generation
            preds.append(row['candidates'][0]['text'])
        refs.append(row.get('reference', ''))
        arts.append(row.get('article', ''))
    return preds, refs, arts

def main(
    baseline: str = typer.Option(..., help='JSONL baseline (1-best) or candidate file with first candidate used'),
    system: str = typer.Option(..., help='JSONL reranked outputs'),
    out: str = typer.Option('results/metrics.csv'),
):
    base_pred, base_ref, base_art = collect_texts(baseline)
    sys_pred, sys_ref, sys_art = collect_texts(system)

    rouge = evaluate.load('rouge')
    bert = evaluate.load('bertscore')
    # compute ROUGE
    r_base = rouge.compute(predictions=base_pred, references=base_ref, use_stemmer=True)
    r_sys  = rouge.compute(predictions=sys_pred, references=sys_ref, use_stemmer=True)
    # BERTScore
    b_base = bert.compute(predictions=base_pred, references=base_ref, lang='en')
    b_sys  = bert.compute(predictions=sys_pred, references=sys_ref, lang='en')

    # Factuality proxies (FactCC primary, QAGS secondary, AlignScore additional)
    fact = FactCCScorer()
    qags = QAGSScorer()
    align = AlignScorer()
    def avg(scores: List[float]) -> float:
        return float(sum(scores) / max(1, len(scores)))

    f_base = avg([fact.score(a, p) for a, p in zip(base_art, base_pred)])
    f_sys  = avg([fact.score(a, p) for a, p in zip(sys_art,  sys_pred)])
    q_base = avg([qags.score(a, p) for a, p in zip(base_art, base_pred)])
    q_sys  = avg([qags.score(a, p) for a, p in zip(sys_art,  sys_pred)])
    a_base = avg([align.score(a, p) for a, p in zip(base_art, base_pred)])
    a_sys  = avg([align.score(a, p) for a, p in zip(sys_art,  sys_pred)])

    rows = [{
        'system': 'baseline',
        'rouge1': r_base['rouge1'], 'rouge2': r_base['rouge2'], 'rougeL': r_base['rougeL'],
        'bertscore_f1': sum(b_base['f1'])/len(b_base['f1']),
        'factcc': f_base, 'qags': q_base, 'alignscore': a_base
    },{
        'system': 'reranked',
        'rouge1': r_sys['rouge1'], 'rouge2': r_sys['rouge2'], 'rougeL': r_sys['rougeL'],
        'bertscore_f1': sum(b_sys['f1'])/len(b_sys['f1']),
        'factcc': f_sys, 'qags': q_sys, 'alignscore': a_sys
    }]
    write_csv(out, rows)
    print(f'[green]Wrote metrics to {out}[/green]')

if __name__ == '__main__':
    typer.run(main)
