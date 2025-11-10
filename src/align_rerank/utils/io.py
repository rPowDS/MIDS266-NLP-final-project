from __future__ import annotations
import json, orjson
from pathlib import Path
from typing import Iterable, Dict, Any, List

def read_jsonl(path: str | Path) -> List[dict]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                out.append(orjson.loads(line))
            except Exception:
                out.append(json.loads(line))
    return out

def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(orjson.dumps(r).decode('utf-8') + '\n')

def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            f.write('')
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
