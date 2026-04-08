import json
import tempfile
from pathlib import Path

from src.rag.haystack_pipe import RetrievalConfig, get_retrieval_config, load_scraped_documents


def test_load_scraped_documents():
    tmpdir = Path(tempfile.mkdtemp())
    data = {"text": "Bryant University info.", "url": "https://bryant.edu", "title": "Test"}
    with open(tmpdir / "page_0000.json", "w") as f:
        json.dump(data, f)
    docs = load_scraped_documents(tmpdir)
    assert len(docs) == 1
    assert "Bryant" in docs[0].content
    assert docs[0].meta["url"] == "https://bryant.edu"


def test_get_retrieval_config_defaults():
    config = get_retrieval_config({})

    assert config == RetrievalConfig(
        index_device="cuda",
        query_device="cpu",
        rerank_device="cpu",
        dense_top_k=3,
        bm25_top_k=3,
        rank_top_k=3,
    )


def test_get_retrieval_config_reads_env_overrides():
    config = get_retrieval_config(
        {
            "RAG_INDEX_DEVICE": "cpu",
            "RAG_QUERY_DEVICE": "cuda",
            "RAG_RERANK_DEVICE": "cuda",
            "RAG_DENSE_TOP_K": "7",
            "RAG_BM25_TOP_K": "6",
            "RAG_RERANK_TOP_K": "4",
        }
    )

    assert config == RetrievalConfig(
        index_device="cpu",
        query_device="cuda",
        rerank_device="cuda",
        dense_top_k=7,
        bm25_top_k=6,
        rank_top_k=4,
    )
