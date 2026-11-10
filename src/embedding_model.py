import hashlib, pickle
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent.parent / "data" / "processed" / "emb_cache.pkl"
_model = None


def _encode(texts, cfg):
    global _model
    from sentence_transformers import SentenceTransformer  # 무거워서 지연 import
    c = cfg["embedding"]
    if _model is None:
        _model = SentenceTransformer(c["model"], device="mps" if __import__("torch").backends.mps.is_available() else "cpu")
        _model.max_seq_length = c["max_seq_length"]  # 입력 길이 제한(초과분은 잘림)
    return _model.encode(texts, batch_size=c["batch_size"], normalize_embeddings=True, show_progress_bar=False)


def embed(texts, cfg):
    """텍스트 해시 기준 캐시. 같은 모델명일 때만 재사용."""
    cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    name = cfg["embedding"]["model"]
    keys = [hashlib.sha1((name + t).encode()).hexdigest() for t in texts]
    todo = sorted({(k, t) for k, t in zip(keys, texts) if k not in cache})
    if todo:
        for (k, _), e in zip(todo, _encode([t for _, t in todo], cfg)):
            cache[k] = e
        CACHE.write_bytes(pickle.dumps(cache))
    return np.array([cache[k] for k in keys])


def similarity(old, new, cfg):
    return embed(old, cfg) @ embed(new, cfg).T  # 정규화된 벡터 -> 내적 = 코사인
