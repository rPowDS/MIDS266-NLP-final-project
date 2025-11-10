#!/usr/bin/env python3
"""
Example: Complete pipeline from training to evaluation
"""

from align_rerank import train_bart, generate_candidates, score_factcc, rerank, eval_automatic

def run_pipeline():
    """Run the complete verifier-reranking pipeline."""
    
    # Step 1: Train model (or use existing)
    model_dir = "runs/bart-baseline"
    
    # Step 2: Generate candidates
    generate_candidates.main(
        model_dir=model_dir,
        split="validation",
        decoder="top_p",
        k=8,
        out="results/candidates.jsonl"
    )
    
    # Step 3: Score with FactCC
    score_factcc.main(
        in_jsonl="results/candidates.jsonl",
        out_csv="results/factcc_scores.csv"
    )
    
    # Step 4: Rerank
    rerank.main(
        candidates="results/candidates.jsonl",
        scores="results/factcc_scores.csv",
        verifier="factcc",
        out="results/reranked.jsonl"
    )
    
    # Step 5: Evaluate
    eval_automatic.main(
        baseline="results/baseline.jsonl",
        system="results/reranked.jsonl",
        out="results/metrics.csv"
    )
    
    print("✅ Pipeline complete!")

if __name__ == "__main__":
    run_pipeline()
