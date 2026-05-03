import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from inference_hybrid_rag_lora import (
    rag_answer,
    hybrid_retrieve,
    docs,
)

app = FastAPI(title="Java面试 RAG 问答助手")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)


class AskRequest(BaseModel):
    question: str
    k: int = 5
    alpha: float = 0.6


class AskResponse(BaseModel):
    answer: str
    context: list[str]


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    retrieved_ids = hybrid_retrieve(req.question, alpha=req.alpha, top_k=req.k)
    context_chunks = [docs[i] for i in retrieved_ids]
    answer = rag_answer(req.question, k=req.k, alpha=req.alpha)
    return AskResponse(answer=answer, context=context_chunks)


@app.get("/")
def index():
    return FileResponse(os.path.join(static_dir, "index.html"))


app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
