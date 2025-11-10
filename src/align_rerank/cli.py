from typer import Typer
from align_rerank import align_rerank.train_bart as train_bart, generate_candidates, score_alignscore, score_factcc, rerank, eval_automatic, eval_human, stats, create_baseline

app = Typer(help="""Unified CLI for the verifier-reranking pipeline.""")

app.command()(train_bart.main)
app.command()(generate_candidates.main)
app.command()(create_baseline.main)
app.command()(score_alignscore.main)
app.command()(score_factcc.main)
app.command()(rerank.main)
app.command()(eval_automatic.main)
app.command()(eval_human.main)
app.command()(stats.main)

if __name__ == "__main__":
    app()
