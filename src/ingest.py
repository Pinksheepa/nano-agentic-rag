from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

from datasets import load_dataset
from tqdm import tqdm

from config import DATA_RAW_DIR, DOCS_JSONL
from utils import write_jsonl


def load_parquet_dataset(input_dir: Path, cache_dir: Path | None = None):
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in: {input_dir}. Put dataset files in data/raw first."
        )
    data_files = [str(p) for p in parquet_files]
    return load_dataset("parquet", data_files=data_files, split="train", cache_dir=cache_dir)


def normalize_row(row: Dict) -> Dict:
    content = row.get("content") or row.get("text") or ""
    title = row.get("title") or row.get("name") or ""
    url = row.get("url") or row.get("source") or ""
    metadata = {"title": title, "url": url}
    return {"content": content, "metadata": metadata}


def iter_docs(dataset, limit: int | None) -> Iterable[Dict]:
    count = 0
    for row in dataset:
        record = normalize_row(row)
        if not record["content"]:
            continue
        yield record
        count += 1
        if limit is not None and count >= limit:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest parquet dataset into docs.jsonl")
    parser.add_argument("--input_dir", type=Path, default=DATA_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DOCS_JSONL)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset = load_parquet_dataset(args.input_dir, args.cache_dir)

    def records():
        for record in tqdm(iter_docs(dataset, args.limit)):
            yield record

    write_jsonl(args.output, records())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
