from align_rerank.utils.io import write_jsonl, read_jsonl
def test_jsonl_roundtrip(tmp_path):
    p = tmp_path/'x.jsonl'
    rows = [{'a':1},{'b':2}]
    write_jsonl(p, rows)
    out = read_jsonl(p)
    assert out == rows
