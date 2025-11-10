from __future__ import annotations
from typing import List, Dict

def to_markdown_table(rows: List[Dict[str, str]], headers: List[str]) -> str:
    if not rows:
        return ''
    header_row = '| ' + ' | '.join(headers) + ' |'
    sep_row = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    body = ['| ' + ' | '.join(str(r.get(h, '')) for h in headers) + ' |' for r in rows]
    return '\n'.join([header_row, sep_row] + body)
