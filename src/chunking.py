from __future__ import annotations

import argparse
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer
from tqdm import tqdm

from config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_JSONL, DOCS_JSONL, EMBEDDING_MODEL
from utils import iter_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk documents with recursive splitting")
    parser.add_argument("--input", type=Path, default=DOCS_JSONL)
    parser.add_argument("--output", type=Path, default=CHUNKS_JSONL)
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in tqdm(iter_jsonl(args.input)):
            doc = Document(page_content=record["content"],
                           metadata=record.get("metadata", {}))
            chunks = splitter.split_documents([doc])
            for chunk in chunks:
                out = {
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                f.write(json_dumps(out) + "\n")

    print(f"Wrote {args.output}")


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


if __name__ == "__main__":
    main()
