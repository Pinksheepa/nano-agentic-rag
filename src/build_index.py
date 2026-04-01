from __future__ import annotations

import argparse
from pathlib import Path

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.documents import Document
from tqdm import tqdm

from config import BATCH_SIZE, CHUNKS_JSONL, EMBEDDING_MODEL, INDEX_DIR
from utils import iter_jsonl


def iter_documents(path: Path):
    for record in iter_jsonl(path):
        yield Document(page_content=record["content"], metadata=record.get("metadata", {}))


def batched(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_embeddings(model_name: str, device: str, batch_size: int):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": batch_size, "normalize_embeddings": True},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from chunks.jsonl")
    parser.add_argument("--input", type=Path, default=CHUNKS_JSONL)
    parser.add_argument("--index_dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--embedding_model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    embedding = build_embeddings(args.embedding_model, args.device, args.batch_size)

    vectorstore = None
    for batch in tqdm(batched(iter_documents(args.input), args.batch_size)):
        if vectorstore is None:
            vectorstore = FAISS.from_documents(
                batch,
                embedding,
                distance_strategy=DistanceStrategy.COSINE,
            )
        else:
            vectorstore.add_documents(batch)

    if vectorstore is None:
        raise RuntimeError("No documents found. Did you run chunking?")

    args.index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(args.index_dir))
    print(f"Saved FAISS index to {args.index_dir}")


if __name__ == "__main__":
    main()
