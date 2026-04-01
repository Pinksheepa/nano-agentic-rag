from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INDEX_DIR = ROOT_DIR / "indexes" / "faiss"

DOCS_JSONL = DATA_PROCESSED_DIR / "docs.jsonl"
CHUNKS_JSONL = DATA_PROCESSED_DIR / "chunks.jsonl"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "thenlper/gte-small-zh")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))

TOP_K = int(os.getenv("TOP_K", "7"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
