# eval.py
# Proxy evaluation helpers (optional for experimentation).

import numpy as np, math, glob, os
from typing import Dict, Any, List, Iterable
from src.config import openai_client, EMBED_MODEL, EMBED_DIM
from src.resume import parse_resume_content, llm_enrich_resume
from src.comparison import rank_listings_for_resume

def _embed_batch(texts: List[str], model: str = EMBED_MODEL) -> List[np.ndarray]:
    client = openai_client()
    clean = [(t or "").strip() for t in texts]
    if not any(clean):
        return [np.zeros(EMBED_DIM, dtype=np.float32) for _ in texts]
    resp = client.embeddings.create(model=model, input=clean)
    out: List[np.ndarray] = []
    di = 0
    for t in clean:
        if not t:
            out.append(np.zeros(EMBED_DIM, dtype=np.float32))
        else:
            out.append(np.array(resp.data[di].embedding, dtype=np.float32))
            di += 1
    return out

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def _dcg(rels: List[float], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        s += (2**rel - 1.0) / math.log2(i + 1)
    return s

def _ndcg_at_k(y_true: List[float], y_pred_order: List[int], k: int) -> float:
    if not y_true: return 0.0
    rel_pred = [y_true[i] for i in y_pred_order[:k]]
    dcg = _dcg(rel_pred, k)
    ideal_order = sorted(range(len(y_true)), key=lambda i: y_true[i], reverse=True)
    idcg = _dcg([y_true[i] for i in ideal_order[:k]], k)
    return 0.0 if idcg == 0 else dcg / idcg

def _spearmanr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) == 0: return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    sx = rx - rx.mean()
    sy = ry - ry.mean()
    denom = (np.linalg.norm(sx) * np.linalg.norm(sy))
    return 0.0 if denom == 0 else float(np.dot(sx, sy) / denom)

def _collect_resume_paths(resumes_dir: str, limit: int = 10) -> List[str]:
    pdfs  = glob.glob(os.path.join(resumes_dir, "*.pdf"))
    docxs = glob.glob(os.path.join(resumes_dir, "*.docx"))
    return sorted(pdfs + docxs)[:limit]

def build_runs_for_directory(resumes_dir: str, listings: Dict[str, Any], precomputed: Dict[str, Any], top_k: int = 20, limit: int = 10):
    paths = _collect_resume_paths(resumes_dir, limit=limit)
    runs = {}
    for path in paths:
        parsed = parse_resume_content(path)
        enriched = llm_enrich_resume(parsed)
        ranked = rank_listings_for_resume(resume_enriched=enriched, listings=listings, precomputed=precomputed, top_k=top_k)
        rid = os.path.splitext(os.path.basename(path))[0]
        runs[rid] = {"resume_enriched": enriched, "ranked": ranked}
    return runs
