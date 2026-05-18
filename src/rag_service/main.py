from __future__ import annotations

from fastapi import FastAPI

from .config import load_config
from .schemas import AskRequest, AskResponse
from .service import answer_question, build_runtime

config = load_config()
runtime = build_runtime(config)

app = FastAPI(title="RAG + Generative AI Service", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "corpus_chunks": len(runtime.retriever.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return answer_question(runtime, request)


@app.get("/metrics")
def metrics() -> dict:
    return runtime.metrics.summary()


@app.post("/admin/report")
def write_report(path: str = "docs/operations_report.csv") -> dict:
    out_path = config.root_dir / path
    runtime.metrics.write_csv_report(out_path)
    return {"written": str(out_path), "summary": runtime.metrics.summary()}
