from pathlib import Path

from rag_service.chunking import load_chunks
from rag_service.retrieval import Retriever, SimpleReranker


def test_hybrid_retrieves_policy_terms():
    root = Path(__file__).resolve().parents[1]
    chunks = load_chunks(root / "data/corpus")
    retriever = Retriever(chunks)
    results = retriever.search("OCR_LOW_CONFIDENCE", mode="hybrid", top_k=3)
    assert any(r.chunk.doc_id == "scanned_pdf_ocr_sample" for r in results)


def test_reranker_returns_requested_top_k():
    root = Path(__file__).resolve().parents[1]
    chunks = load_chunks(root / "data/corpus")
    retriever = Retriever(chunks)
    results = retriever.search("检索模式 vector-only hybrid", mode="hybrid", top_k=5)
    reranked = SimpleReranker().rerank("检索模式 vector-only hybrid", results, top_k=2)
    assert len(reranked) == 2
