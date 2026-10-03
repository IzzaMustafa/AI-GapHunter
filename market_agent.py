"""GapHunter AI agent module."""
from __future__ import annotations

import json
import re
from typing import Any

from crewai import Agent, Crew, Process, Task, LLM

MODEL = "groq/openai/gpt-oss-120b"


def _llm(api_key: str) -> LLM:
    return LLM(model=MODEL, api_key=api_key, temperature=0.25)


def _parse_json(raw: Any) -> dict:
    text = getattr(raw, "raw", None) or str(raw)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {"raw_output": text, "parse_warning": "Model output was not valid JSON."}


def _run(role: str, goal: str, backstory: str, description: str,
         expected_output: str, api_key: str) -> dict:
    agent = Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=_llm(api_key),
        verbose=False,
        allow_delegation=False,
    )
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return _parse_json(crew.kickoff())

from pathlib import Path
import hashlib
import pickle

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

KB_DIR = Path(__file__).resolve().parent / "knowledge_base"
INDEX_DIR = KB_DIR / ".faiss"
INDEX_FILE = INDEX_DIR / "index.faiss"
META_FILE = INDEX_DIR / "metadata.pkl"
MANIFEST_FILE = INDEX_DIR / "manifest.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _discover_documents() -> list[Path]:
    if not KB_DIR.exists():
        return []
    return sorted(
        p for p in KB_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}
        and ".faiss" not in p.parts
    )


def _fingerprint(paths: list[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        stat = path.stat()
        key = str(path.relative_to(KB_DIR))
        result[key] = hashlib.sha256(
            f"{stat.st_size}:{stat.st_mtime_ns}".encode()
        ).hexdigest()
    return result


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunks(text: str, size: int = 1200, overlap: int = 180) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def ensure_index() -> tuple[Any, list[dict], SentenceTransformer]:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    docs = _discover_documents()
    manifest = _fingerprint(docs)
    model = SentenceTransformer(EMBED_MODEL)

    if INDEX_FILE.exists() and META_FILE.exists() and MANIFEST_FILE.exists():
        try:
            old = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
            if old == manifest:
                return faiss.read_index(str(INDEX_FILE)), pickle.loads(META_FILE.read_bytes()), model
        except Exception:
            pass

    metadata, texts = [], []
    for path in docs:
        category = path.parent.name
        for idx, chunk in enumerate(_chunks(_read_document(path))):
            texts.append(chunk)
            metadata.append({
                "source": str(path.relative_to(KB_DIR)),
                "category": category,
                "chunk": idx,
                "text": chunk,
            })

    if texts:
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
    else:
        index = faiss.IndexFlatIP(384)

    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_bytes(pickle.dumps(metadata))
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return index, metadata, model


def retrieve_context(query: str, top_k: int = 4, categories: list[str] | None = None) -> list[dict]:
    index, metadata, model = ensure_index()
    if index.ntotal == 0:
        return []
    vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    scores, ids = index.search(np.asarray(vector, dtype="float32"), min(max(top_k * 3, top_k), index.ntotal))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        item = dict(metadata[idx])
        if categories and item["category"] not in categories:
            continue
        item["similarity"] = round(float(score), 4)
        results.append(item)
        if len(results) >= top_k:
            break
    return results


def run_market_agent(profile_analysis: dict, problems: dict, rag_context: list[dict], api_key: str) -> dict:
    return _run(
        "Market and Trend Researcher",
        "Analyze relevant market signals without pretending the local knowledge base is complete or current.",
        "You distinguish retrieved evidence from inference and explicitly state research limitations.",
        f"""Profile analysis:
{json.dumps(profile_analysis, ensure_ascii=False)}
Candidate problems:
{json.dumps(problems, ensure_ascii=False)}
Retrieved knowledge-base context:
{json.dumps(rag_context, ensure_ascii=False)}

Return ONLY JSON:
{{"market_signals":[], "emerging_technologies":[], "relevant_markets":[],
"demand_hypotheses":[], "evidence":[], "inferences":[],
"research_limitations":[]}}
Use only retrieved material as evidence. If context is empty, say that no local KB evidence was available.""",
        "Valid JSON market analysis with explicit evidence and limitations.",
        api_key,
    )
