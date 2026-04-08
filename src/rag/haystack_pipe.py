# src/rag/haystack_pipe.py
"""
Haystack-based hybrid RAG pipeline.

Replaces custom chunker, indexer, and retriever with Haystack 2.x components.
BM25 + dense retrieval with reciprocal rank fusion and cross-encoder reranking.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

from haystack import Document, Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.writers import DocumentWriter
from haystack.utils import ComponentDevice
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

EMBED_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class RetrievalConfig:
    index_device: str = "cuda"
    query_device: str = "cpu"
    rerank_device: str = "cpu"
    dense_top_k: int = 3
    bm25_top_k: int = 3
    rank_top_k: int = 3


def _parse_positive_int(env: dict[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value is None:
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1, got {parsed}")
    return parsed


def get_retrieval_config(env: dict[str, str] | None = None) -> RetrievalConfig:
    """Read runtime retrieval settings from environment variables."""
    source = os.environ if env is None else env
    return RetrievalConfig(
        index_device=source.get("RAG_INDEX_DEVICE", "cuda"),
        query_device=source.get("RAG_QUERY_DEVICE", "cpu"),
        rerank_device=source.get("RAG_RERANK_DEVICE", "cpu"),
        dense_top_k=_parse_positive_int(source, "RAG_DENSE_TOP_K", 3),
        bm25_top_k=_parse_positive_int(source, "RAG_BM25_TOP_K", 3),
        rank_top_k=_parse_positive_int(source, "RAG_RERANK_TOP_K", 3),
    )


def load_scraped_documents(data_dir: Path) -> list[Document]:
    """Load scraped JSON files into Haystack Documents."""
    docs = []
    for path in sorted(data_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        docs.append(Document(
            content=data["text"],
            meta={"url": data.get("url", ""), "title": data.get("title", "")},
        ))
    return docs


def build_index_pipeline(chroma_path: str) -> tuple[Pipeline, ChromaDocumentStore]:
    """Build the document indexing pipeline."""
    config = get_retrieval_config()
    chroma_store = ChromaDocumentStore(
        collection_name="bryant_corpus",
        persist_path=chroma_path,
    )

    pipeline = Pipeline()
    pipeline.add_component("splitter", DocumentSplitter(
        split_by="sentence", split_length=8, split_overlap=2,
    ))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(
        model=EMBED_MODEL,
        batch_size=32,
        device=ComponentDevice.from_str(config.index_device),
    ))
    pipeline.add_component("writer", DocumentWriter(document_store=chroma_store))
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder", "writer")

    return pipeline, chroma_store


def build_retrieval_pipeline(chroma_store: ChromaDocumentStore) -> tuple[Pipeline, InMemoryDocumentStore]:
    """Build the hybrid retrieval pipeline with RRF and cross-encoder reranking."""
    # BM25 needs an InMemory store — copy docs from Chroma
    config = get_retrieval_config()
    bm25_store = InMemoryDocumentStore()

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(
        model=EMBED_MODEL,
        device=ComponentDevice.from_str(config.query_device),
    ))
    pipeline.add_component("dense_retriever", ChromaEmbeddingRetriever(
        document_store=chroma_store,
        top_k=config.dense_top_k,
    ))
    pipeline.add_component("bm25_retriever", InMemoryBM25Retriever(
        document_store=bm25_store,
        top_k=config.bm25_top_k,
    ))
    pipeline.add_component("joiner", DocumentJoiner(
        join_mode="reciprocal_rank_fusion",
    ))
    pipeline.add_component("ranker", TransformersSimilarityRanker(
        model=RERANKER_MODEL,
        top_k=config.rank_top_k,
        device=ComponentDevice.from_str(config.rerank_device),
    ))

    pipeline.connect("text_embedder.embedding", "dense_retriever.query_embedding")
    pipeline.connect("dense_retriever", "joiner")
    pipeline.connect("bm25_retriever", "joiner")
    pipeline.connect("joiner", "ranker")

    return pipeline, bm25_store


def retrieve(pipeline: Pipeline, query: str) -> list[Document]:
    """Run hybrid retrieval for a query."""
    result = pipeline.run({
        "text_embedder": {"text": query},
        "bm25_retriever": {"query": query},
        "ranker": {"query": query},
    })
    return result["ranker"]["documents"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/scraped"))
    parser.add_argument("--chroma", type=str, default="data/chroma")
    args = parser.parse_args()

    print(f"Loading documents from {args.input}...")
    docs = load_scraped_documents(args.input)
    print(f"Loaded {len(docs)} documents")

    print("Building index pipeline...")
    pipeline, _ = build_index_pipeline(args.chroma)

    print("Indexing (embedding + writing to Chroma)...")
    pipeline.run({"splitter": {"documents": docs}})
    print(f"Done. Chroma index written to {args.chroma}")
