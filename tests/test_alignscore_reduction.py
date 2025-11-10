from align_rerank.verifiers.alignscore import AlignScorer
def test_reduction_runs():
    s = AlignScorer()
    score = s.score('Paris is the capital of France.', 'Paris is the capital of France.')
    assert 0.0 <= score <= 1.0
