import json, os, hashlib, pathlib, typing as t

Path = pathlib.Path

def read_jsonl(path: str) -> t.List[t.Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(records: t.Iterable[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "
")

def source_id(p: str) -> str:
    return hashlib.sha1(p.encode()).hexdigest()[:16]
