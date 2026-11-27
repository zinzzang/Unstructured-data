"""대표 변화 문장쌍 추출.

한 기업·연도쌍·변화유형의 변화 문장쌍(TF-IDF 결과 CSV) 가운데, 산업 공동 LDA 토픽별로 대표 문장쌍을 고른다.

절차
  1. 변화 CSV에서 기업·연도쌍·유형 행을 읽는다 (source_change_row = CSV 데이터 행 번호, 1부터).
  2. LDA 토픽 확률과 조인한다. NEW·EXPANDED·MODIFIED·REDUCED는 new 문장, REMOVED는 old 문장의 확률을 쓴다
     (REMOVED 확률은 산업 LDA의 기존 vectorizer/model로 transform만 한 값 = 같은 토픽 공간).
     문장의 토픽 = 확률 최대 토픽(동률이면 번호가 작은 토픽).
  3. 문장쌍 벡터(BGE-M3, 정규화): NEW=new, REMOVED=old, 매칭형=[old, new]를 이어 붙여 다시 정규화.
  4. 토픽별 centroid(후보 벡터 평균)와의 코사인으로 순위를 매긴다.
  5. 전체 후보를 centroid 점수 순으로 보며 완전 중복·근접 중복(TF-IDF char 코사인 >= 임계값)을 제거한다.
  6. 토픽 할당량 = K x 토픽 후보 비중(최대 잉여 방식), 후보가 있는 토픽은 최소 1개, 중복 제거 후 후보 수가 상한,
     모자라면 남은 후보가 있는 토픽에 재배분한다. 선택 수 = min(K, 중복 제거 후 후보 수).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CHANGE_TYPES = ("NEW", "EXPANDED", "MODIFIED", "REDUCED", "REMOVED")
MATCHED = frozenset({"EXPANDED", "MODIFIED", "REDUCED"})
INDUSTRIES = {
    "semiconductor": ["SK하이닉스", "삼성전자", "한미반도체"],
    "battery": ["LG에너지솔루션", "삼성SDI", "SK이노베이션"],
    "biopharma": ["삼성바이오로직스", "셀트리온", "유한양행"],
}
DEFAULTS = {
    "source_change_csv": "data/results_v2/tfidf_masked_th0.6.csv",
    "lda_dir": "data/results_industry_lda_corrected",
    "paragraphs_dir": "data/processed/preprocessing_v2",
    "top_k": 10,
    "duplicate_method": "tfidf_char_cosine",
    "duplicate_threshold": 0.92,
    "output_dir": "data/results_representative_changes",
    "embedding_artifact_dir": "data/results_representative_changes/embeddings",
}
Embedder = Callable[[list[str]], np.ndarray]


# ---------------------------------------------------------------- config / io helpers
def load_config(path: Path | None = None) -> dict:
    cfg = yaml.safe_load((path or ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    cfg["representative_changes"] = {**DEFAULTS, **(cfg.get("representative_changes") or {})}
    return cfg


def _path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- 1. change rows
@lru_cache(maxsize=4)
def _read_change_csv(path: str) -> tuple[dict, ...]:
    with open(path, encoding="utf-8-sig") as f:
        return tuple({**r, "source_change_row": i} for i, r in enumerate(csv.DictReader(f), start=1))


def load_change_rows(company: str, year_old: int, year_new: int, path: Path) -> list[dict]:
    return [r for r in _read_change_csv(str(path))
            if r["company"] == company and int(r["year_old"]) == year_old and int(r["year_new"]) == year_new]


def resolve_industry(company: str, industry: str | None = None) -> str:
    found = [ind for ind, cos in INDUSTRIES.items() if company in cos]
    if not found:
        raise ValueError(f"알 수 없는 기업: {company}")
    if industry and industry != found[0]:
        raise ValueError(f"{company}는 {found[0]} 산업입니다 (요청: {industry})")
    return found[0]


# ---------------------------------------------------------------- 2. topics
def load_topic_probabilities(industry: str, lda_dir: Path) -> dict:
    d = lda_dir / industry
    new_csv, old_csv = d / "sentence_topic_probabilities.csv", d / "sentence_topic_probabilities_old.csv"
    meta_file = d / "lda_artifacts_metadata.json"
    for f in (new_csv, old_csv, meta_file, d / "topics.txt"):
        if not f.exists():
            raise FileNotFoundError(f"{f} 없음: scripts/industry_lda_corrected.py {industry} 를 먼저 실행하세요")

    def read(path):
        with open(path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        cols = sorted((c for c in rows[0] if c.startswith("topic_")), key=lambda c: int(c.split("_")[1]))
        return {r["sentence_id"]: np.array([float(r[c]) for c in cols]) for r in rows}, len(cols)

    new, k = read(new_csv)
    old, k_old = read(old_csv)
    if k != k_old:
        raise ValueError(f"토픽 수 불일치: new {k}, old {k_old}")
    keywords = []
    for line in (d / "topics.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            keywords.append([w.strip() for w in line.split(":", 1)[1].split(",")])
    labels_file = d / "topic_labels.json"
    custom = json.loads(labels_file.read_text(encoding="utf-8")) if labels_file.exists() else {}
    labels = [custom.get(str(i + 1), ", ".join(keywords[i][:4])) for i in range(k)]
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    files = {name: f for name, f in [("topic_probabilities_new", new_csv), ("topic_probabilities_old", old_csv),
                                      ("topics", d / "topics.txt"), ("lda_artifact", d / "lda_artifacts.joblib")]}
    if labels_file.exists():
        files["topic_labels"] = labels_file
    return dict(k=k, new=new, old=old, keywords=keywords, labels=labels, labels_custom=bool(custom),
                artifact_version=meta["artifact_version"], metadata=meta, files=files)


@lru_cache(maxsize=4)
def load_sentence_index(paragraphs_jsonl: str) -> dict:
    """sentence_id -> 원문·섹션 경로·XML 위치 (JSONL 역조회)."""
    idx = {}
    with open(paragraphs_jsonl, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            path = " > ".join(p["title_path"])
            for s in p["sentences"]:
                idx[s["sentence_id"]] = dict(text=s["text"], section_path=path, xml_line=p["source_location"]["xml_line"],
                                             source_type=p["source_type"],
                                             paragraph_id=p["paragraph_id"], analysis_area=p["analysis_area"],
                                             company=p["company"], year=int(p["business_year"]))
    return idx


def join_pairs_with_topics(rows: list[dict], change_type: str, topics: dict,
                           sentence_index: dict | None = None) -> tuple[list[dict], list[dict]]:
    """변화 행 -> 토픽이 붙은 후보. 반환: (후보, 조인 실패). 산업 LDA는 문단 문장만 다루므로 표 행은 토픽이 없다."""
    cands, failures = [], []
    for r in rows:
        if r["change_type"] != change_type:
            continue
        old_id, new_id = (r.get("old_id") or "").strip(), (r.get("new_id") or "").strip()
        if change_type in MATCHED and not (old_id and new_id):
            raise ValueError(f"매칭형 변화({change_type})에 old/new 문장 ID가 모두 있어야 합니다: CSV 행 {r['source_change_row']}")
        if change_type == "NEW" and not new_id:
            raise ValueError(f"NEW 행에 new 문장 ID가 없습니다: CSV 행 {r['source_change_row']}")
        if change_type == "REMOVED" and not old_id:
            raise ValueError(f"REMOVED 행에 old 문장 ID가 없습니다: CSV 행 {r['source_change_row']}")
        side, sid = ("old", old_id) if change_type == "REMOVED" else ("new", new_id)
        prob = topics[side].get(sid)
        if prob is None:
            st = (sentence_index or {}).get(sid, {}).get("source_type")
            reason = ("table_row: 산업 LDA는 문단 문장만 학습·추론" if st == "table_row" else
                      "lda_filtered: 산업 범위·위험관리 섹션·각주/짧은 문장·문단 크기 필터 또는 토큰 없음" if st == "paragraph" else
                      "not_in_jsonl")
            failures.append(dict(source_change_row=r["source_change_row"], sentence_id=sid, side=side, reason=reason))
            continue
        t = int(np.argmax(prob))
        cands.append(dict(row=r, old_id=old_id or None, new_id=new_id or None, topic_side=side,
                          topic_id=t + 1, topic_probability=float(prob[t]), probs=prob))
    return cands, failures


def select_change_type(rows: list[dict], topics: dict, change_type: str | None,
                       sentence_index: dict | None = None) -> tuple[str, dict]:
    """유형을 지정하지 않으면 토픽 조인에 성공한 후보가 가장 많은 유형(동률이면 CHANGE_TYPES 순서)."""
    if change_type is not None:
        if change_type not in CHANGE_TYPES:
            raise ValueError(f"change_type은 {CHANGE_TYPES} 중 하나여야 합니다: {change_type}")
        return change_type, {"mode": "requested"}
    counts = {ct: len(join_pairs_with_topics(rows, ct, topics, sentence_index)[0]) for ct in CHANGE_TYPES}
    best = max(CHANGE_TYPES, key=lambda ct: (counts[ct], -CHANGE_TYPES.index(ct)))
    return best, {"mode": "auto", "joined_candidates_by_type": counts}


# ---------------------------------------------------------------- 3. embeddings
def _mask(text: str, cfg: dict) -> str:
    m = cfg["embedding"].get("number_masking", {})
    if m.get("enabled"):
        from .tfidf_model import mask_numbers
        return mask_numbers(text, m.get("token", "NUM"))
    return text


def default_embedder(cfg: dict) -> Embedder:
    state = {}

    def encode(texts: list[str]) -> np.ndarray:
        if "model" not in state:
            import torch
            from sentence_transformers import SentenceTransformer
            e = cfg["embedding"]
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            state["model"] = SentenceTransformer(e["model"], device=device)
            state["model"].max_seq_length = e["max_seq_length"]
        return state["model"].encode(texts, batch_size=cfg["embedding"]["batch_size"],
                                     normalize_embeddings=True, show_progress_bar=False)
    return encode


def _embedding_settings(cfg: dict) -> dict:
    e = cfg["embedding"]
    m = e.get("number_masking", {})
    return dict(model=e["model"], max_seq_length=int(e["max_seq_length"]),
                number_masking=f"{bool(m.get('enabled'))}:{m.get('token', 'NUM')}")


def load_or_create_embeddings(texts: dict[str, str], cfg: dict, artifact_dir: Path,
                              embedder: Embedder | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """sentence_id -> 정규화 벡터. 같은 ID라도 원문 hash·모델 설정이 다르면 다시 만든다. 필요한 문장만 계산해 저장소에 합친다."""
    st = _embedding_settings(cfg)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", st["model"].split("/")[-1]).lower()
    npz, meta_csv, manifest = (artifact_dir / f"{stem}_sentence_embeddings.npz",
                               artifact_dir / f"{stem}_sentence_embeddings_metadata.csv",
                               artifact_dir / f"{stem}_sentence_embeddings_manifest.json")
    store_ids, store_vecs, store_meta = [], None, {}
    if npz.exists() and meta_csv.exists():
        with np.load(npz, allow_pickle=False) as z:  # with: Windows에서 파일 핸들이 남아 덮어쓰기가 막히는 것 방지
            store_ids, store_vecs = [str(x) for x in z["sentence_ids"]], np.array(z["vectors"])
        with open(meta_csv, encoding="utf-8-sig") as f:
            store_meta = {r["sentence_id"]: r for r in csv.DictReader(f)}
    pos = {sid: i for i, sid in enumerate(store_ids)}

    def fresh(sid, h):
        m = store_meta.get(sid)
        return (sid in pos and m is not None and m["text_sha256"] == h and m["model"] == st["model"]
                and int(m["max_seq_length"]) == st["max_seq_length"] and m["number_masking"] == st["number_masking"])

    hashes = {sid: sha256_text(t) for sid, t in texts.items()}
    todo = sorted(sid for sid in texts if not fresh(sid, hashes[sid]))
    out = {sid: store_vecs[pos[sid]] for sid in texts if sid not in todo}
    if todo:
        enc = embedder or default_embedder(cfg)
        vecs = np.asarray(enc([_mask(texts[sid], cfg) for sid in todo]), dtype=np.float32)
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
        for sid, v in zip(todo, vecs):
            out[sid] = v
        if store_vecs is not None and store_vecs.shape[1] != vecs.shape[1]:
            store_ids, store_vecs, store_meta, pos = [], None, {}, {}
        ids = list(store_ids)
        mat = [store_vecs] if store_vecs is not None else []
        new_rows = []
        for sid, v in zip(todo, vecs):
            if sid in pos:
                mat[0][pos[sid]] = v
            else:
                ids.append(sid); new_rows.append(v)
            store_meta[sid] = dict(sentence_id=sid, text_sha256=hashes[sid], **st)
        if new_rows:
            mat.append(np.stack(new_rows))
        allv = np.concatenate(mat).astype(np.float32)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        tmp = npz.with_suffix(".tmp.npz")
        np.savez(tmp, sentence_ids=np.array(ids), vectors=allv)
        tmp.replace(npz)
        with open(meta_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sentence_id", "text_sha256", "model", "max_seq_length", "number_masking"])
            w.writeheader()
            w.writerows(store_meta[s] for s in ids)
        manifest.write_text(json.dumps(dict(
            input_sha256=sha256_text("\n".join(f"{s}\t{store_meta[s]['text_sha256']}" for s in sorted(ids))),
            dimension=int(allv.shape[1]), rows=len(ids), settings=st, batch_size=cfg["embedding"]["batch_size"],
            package_versions=_package_versions()), ensure_ascii=False, indent=2), encoding="utf-8")
    info = dict(requested=len(texts), reused=len(texts) - len(todo), created=len(todo), npz=npz, settings=st)
    return out, info


# ---------------------------------------------------------------- 4. vectors / quotas / ranking / dedup
def build_pair_vectors(cands: list[dict], change_type: str, emb: dict[str, np.ndarray]) -> np.ndarray:
    vecs = []
    for c in cands:
        if change_type == "NEW":
            v = emb[c["new_id"]]
        elif change_type == "REMOVED":
            v = emb[c["old_id"]]
        elif change_type in MATCHED:
            if not (c.get("old_id") and c.get("new_id")):
                raise ValueError(f"매칭형 변화({change_type})의 old/new 문장이 모두 필요합니다")
            v = np.concatenate([emb[c["old_id"]], emb[c["new_id"]]])
        else:
            raise ValueError(f"알 수 없는 변화 유형: {change_type}")
        v = np.asarray(v, dtype=np.float64)
        vecs.append(v / max(np.linalg.norm(v), 1e-12))
    return np.array(vecs)


def allocate_topic_quotas(counts: dict[int, int], caps: dict[int, int], k: int) -> tuple[dict[int, int], dict[int, float]]:
    """counts: 토픽별 후보 수(비중 계산), caps: 토픽별 중복 제거 후 후보 수(상한). 반환: (할당량, 이상적 할당량)."""
    total = sum(counts.get(t, 0) for t in caps if caps[t] > 0)
    live = [t for t in caps if caps[t] > 0]
    target = min(k, sum(caps[t] for t in live))
    if target <= 0:
        return {t: 0 for t in caps}, {t: 0.0 for t in caps}
    ideal = {t: target * counts.get(t, 0) / total if total else target / len(live) for t in live}
    order = sorted(live, key=lambda t: (-counts.get(t, 0), t))
    if target < len(live):  # 토픽이 K보다 많으면 큰 토픽부터 1개씩
        q = {t: (1 if t in order[:target] else 0) for t in live}
    else:
        q = {t: min(caps[t], max(1, math.floor(ideal[t]))) for t in live}
        while sum(q.values()) < target:  # 남는 자리: 잉여(ideal-q)가 큰 토픽부터, 상한 안에서
            t = max((t for t in live if q[t] < caps[t]), key=lambda t: (ideal[t] - q[t], counts.get(t, 0), -t))
            q[t] += 1
        while sum(q.values()) > target:  # 최소 1개 보장으로 넘친 자리: 잉여가 가장 작은 토픽에서 회수
            t = min((t for t in live if q[t] > 1), key=lambda t: (ideal[t] - q[t], counts.get(t, 0), -t))
            q[t] -= 1
    return {t: q.get(t, 0) for t in caps}, {t: ideal.get(t, 0.0) for t in caps}


def rank_by_centroid(vectors: np.ndarray, cands: list[dict]) -> None:
    """토픽별 centroid(후보 벡터 평균, 정규화)와의 코사인 -> centroid_cosine_similarity, rank_in_topic(1부터)."""
    by_topic = defaultdict(list)
    for i, c in enumerate(cands):
        by_topic[c["topic_id"]].append(i)
    for t, idx in by_topic.items():
        cen = vectors[idx].mean(axis=0)
        cen /= max(np.linalg.norm(cen), 1e-12)
        for i in idx:
            cands[i]["centroid_cosine_similarity"] = float(vectors[i] @ cen)
        ranked = sorted(idx, key=lambda i: (-round(cands[i]["centroid_cosine_similarity"], 12), cands[i]["row"]["source_change_row"]))
        for r, i in enumerate(ranked, start=1):
            cands[i]["centroid_rank"] = r


def pair_text(c: dict, texts: dict[str, str]) -> str:
    parts = [texts[s] for s in (c.get("old_id"), c.get("new_id")) if s]
    return " → ".join(parts)


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def select_deduplicated(cands: list[dict], texts: dict[str, str], threshold: float, cfg: dict) -> tuple[list[int], list[dict]]:
    """centroid 점수 순으로 보며 완전 중복·근접 중복 제거. 반환: (남은 후보 index, 제외 기록)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    order = sorted(range(len(cands)), key=lambda i: (-round(cands[i]["centroid_cosine_similarity"], 12),
                                                    cands[i]["row"]["source_change_row"]))
    docs = [_norm_text(pair_text(c, texts)) for c in cands]
    if not docs:
        return [], []
    t = cfg["tfidf"]
    m = t.get("number_masking", {})
    if m.get("enabled"):  # 근접 중복은 프로젝트 TF-IDF와 같은 숫자 마스킹 텍스트로 비교(금액·비율은 보존)
        from .tfidf_model import mask_numbers
        masked = [mask_numbers(d, m.get("token", "NUM")) for d in docs]
    else:
        masked = docs
    X = TfidfVectorizer(analyzer=t["analyzer"], ngram_range=tuple(t["ngram_range"])).fit_transform(masked)
    kept, seen, excluded = [], {}, []
    for i in order:
        if docs[i] in seen:
            excluded.append(dict(source_change_row=cands[i]["row"]["source_change_row"], kind="exact",
                                 duplicate_of=cands[seen[docs[i]]]["row"]["source_change_row"], similarity=1.0))
            continue
        if kept:
            sims = (X[kept] @ X[i].T).toarray().ravel()
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                excluded.append(dict(source_change_row=cands[i]["row"]["source_change_row"], kind="near",
                                     duplicate_of=cands[kept[j]]["row"]["source_change_row"], similarity=round(float(sims[j]), 4)))
                continue
        kept.append(i)
        seen[docs[i]] = i
    return kept, excluded


# ---------------------------------------------------------------- 5. output
def _package_versions() -> dict:
    import importlib.metadata as md
    out = {}
    for p in ("numpy", "scikit-learn", "joblib", "kiwipiepy", "sentence-transformers", "torch", "transformers", "pyyaml"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


CSV_COLUMNS = ["company", "industry", "year_old", "year_new", "change_type",
               "topic_id", "topic_label", "topic_keywords", "topic_probability",
               "topic_candidate_count", "topic_candidate_share", "topic_quota",
               "centroid_cosine_similarity", "rank_in_topic", "global_selection_rank",
               "old_sentence_id", "old_text", "old_section_path", "old_xml_line",
               "new_sentence_id", "new_text", "new_section_path", "new_xml_line",
               "analysis_area", "source_change_row", "embedding_model", "lda_artifact_version"]


def serialize_result(result: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "representatives.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in result["_rows"]:
            w.writerow({c: ("|".join(r[c]) if isinstance(r[c], list) else ("" if r[c] is None else r[c])) for c in CSV_COLUMNS})
    public = {k: v for k, v in result.items() if not k.startswith("_")}
    (out_dir / "representatives.json").write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "audit.json").write_text(json.dumps(result["_audit"], ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": out_dir / "representatives.csv", "json": out_dir / "representatives.json", "audit": out_dir / "audit.json"}


# ---------------------------------------------------------------- API
def extract_representatives(company: str, year_old: int, year_new: int, change_type: str | None = None,
                            top_k: int | None = None, industry: str | None = None, *,
                            config: dict | None = None, embedder: Embedder | None = None, write: bool = True) -> dict:
    cfg = config or load_config()
    rc = cfg["representative_changes"]
    k = int(top_k if top_k is not None else rc["top_k"])
    if k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다")
    year_old, year_new = int(year_old), int(year_new)
    industry = resolve_industry(company, industry)
    src_csv, lda_dir = _path(rc["source_change_csv"]), _path(rc["lda_dir"])
    jsonl = _path(rc["paragraphs_dir"]) / industry / "paragraphs.jsonl"

    rows = load_change_rows(company, year_old, year_new, src_csv)
    if not rows:
        raise ValueError(f"변화 CSV에 {company} {year_old}→{year_new} 행이 없습니다")
    topics = load_topic_probabilities(industry, lda_dir)
    sent = load_sentence_index(str(jsonl))
    change_type, type_choice = select_change_type(rows, topics, change_type, sent)
    rows_of_type = [r for r in rows if r["change_type"] == change_type]
    cands, failures = join_pairs_with_topics(rows, change_type, topics, sent)
    texts, lookup_fail = {}, []
    for c in cands:
        for sid, side_text in ((c["old_id"], c["row"].get("old_text")), (c["new_id"], c["row"].get("new_text"))):
            if not sid:
                continue
            s = sent.get(sid)
            if s is None:
                lookup_fail.append(sid)
                texts[sid] = (side_text or "").strip()
            else:
                texts[sid] = s["text"]
    if lookup_fail:
        raise ValueError(f"JSONL에서 찾지 못한 문장 {len(lookup_fail)}개: {lookup_fail[:3]}")

    # 범위 검증: 후보가 모두 요청한 기업·연도·유형·토픽 안에 있는지
    for c in cands:
        r = c["row"]
        assert r["company"] == company and int(r["year_old"]) == year_old and int(r["year_new"]) == year_new
        assert r["change_type"] == change_type and 1 <= c["topic_id"] <= topics["k"]
        for sid, y in ((c["old_id"], year_old), (c["new_id"], year_new)):
            if sid:
                assert sent[sid]["company"] == company and sent[sid]["year"] == y, (sid, y)

    emb, emb_info = load_or_create_embeddings(texts, cfg, _path(rc["embedding_artifact_dir"]), embedder)
    vectors = build_pair_vectors(cands, change_type, emb) if cands else np.zeros((0, 1))
    rank_by_centroid(vectors, cands)
    kept, dup_excluded = select_deduplicated(cands, texts, float(rc["duplicate_threshold"]), cfg)

    counts = Counter(c["topic_id"] for c in cands)
    caps = Counter(cands[i]["topic_id"] for i in kept)
    all_topics = list(range(1, topics["k"] + 1))
    quotas, ideal = allocate_topic_quotas({t: counts.get(t, 0) for t in all_topics}, {t: caps.get(t, 0) for t in all_topics}, k)

    kept_by_topic = defaultdict(list)
    for i in kept:
        kept_by_topic[cands[i]["topic_id"]].append(i)
    selected = []
    for t in all_topics:
        ranked = sorted(kept_by_topic[t], key=lambda i: (cands[i]["centroid_rank"], cands[i]["row"]["source_change_row"]))
        for r, i in enumerate(ranked[:quotas[t]], start=1):
            selected.append((t, r, i))
    # 전체 순위: 토픽 안 순위 -> 토픽 후보 수(많은 순) -> 토픽 번호
    selected.sort(key=lambda x: (x[1], -counts.get(x[0], 0), x[0]))
    n_total = len(cands)
    emb_model = emb_info["settings"]["model"]

    out_rows, topic_blocks = [], {t: [] for t in all_topics}
    for g, (t, r_in_topic, i) in enumerate(selected, start=1):
        c, row = cands[i], cands[i]["row"]
        o, n = (sent[c["old_id"]] if c["old_id"] else None), (sent[c["new_id"]] if c["new_id"] else None)
        rec = dict(company=company, industry=industry, year_old=year_old, year_new=year_new, change_type=change_type,
                   topic_id=t, topic_label=topics["labels"][t - 1], topic_keywords=topics["keywords"][t - 1],
                   topic_probability=round(c["topic_probability"], 6), topic_candidate_count=counts.get(t, 0),
                   topic_candidate_share=round(counts.get(t, 0) / n_total, 6) if n_total else 0.0, topic_quota=quotas[t],
                   centroid_cosine_similarity=round(c["centroid_cosine_similarity"], 6), rank_in_topic=r_in_topic,
                   global_selection_rank=g,
                   old_sentence_id=c["old_id"], old_text=o["text"] if o else None, old_section_path=o["section_path"] if o else None,
                   old_xml_line=o["xml_line"] if o else None,
                   new_sentence_id=c["new_id"], new_text=n["text"] if n else None, new_section_path=n["section_path"] if n else None,
                   new_xml_line=n["xml_line"] if n else None,
                   analysis_area=row["area"], source_change_row=row["source_change_row"],
                   embedding_model=emb_model, lda_artifact_version=topics["artifact_version"])
        out_rows.append(rec)
        topic_blocks[t].append(rec)

    topic_summary = [dict(topic_id=t, label=topics["labels"][t - 1], keywords=topics["keywords"][t - 1],
                          candidate_count=counts.get(t, 0),
                          candidate_share=round(counts.get(t, 0) / n_total, 6) if n_total else 0.0,
                          deduplicated_count=caps.get(t, 0), quota=quotas[t],
                          representatives=[{kk: v for kk, v in rec.items() if kk not in ("company", "industry", "year_old", "year_new", "change_type",
                                                                                          "topic_id", "topic_label", "topic_keywords")}
                                           for rec in topic_blocks[t]])
                     for t in all_topics if counts.get(t, 0)]
    request = dict(company=company, industry=industry, year_old=year_old, year_new=year_new,
                   change_type=change_type, change_type_selection=type_choice, top_k=k)
    summary = dict(change_rows_for_pair=len(rows), change_rows_of_type=len(rows_of_type), candidates=n_total,
                   join_failures=len(failures), join_failures_by_reason=dict(Counter(f["reason"].split(":")[0] for f in failures)),
                   duplicates_excluded=len(dup_excluded), deduplicated_candidates=len(kept),
                   selected=len(out_rows), expected_selected=min(k, len(kept)),
                   topic_labels_source="topic_labels.json" if topics["labels_custom"] else "keywords",
                   embedding_model=emb_model, lda_artifact_version=topics["artifact_version"])

    inputs = {"source_change_csv": src_csv, "paragraphs_jsonl": jsonl, **topics["files"], "embeddings_npz": emb_info["npz"]}
    audit = dict(
        request=request,
        inputs={name: dict(path=_rel(p), sha256=sha256_file(p) if p.exists() else None) for name, p in inputs.items()},
        lda_metadata=topics["metadata"],
        counts=dict(change_rows_for_pair=len(rows), by_type=dict(Counter(r["change_type"] for r in rows)),
                    change_rows_of_type=len(rows_of_type), joined=n_total, join_failed=len(failures),
                    after_dedup=len(kept), selected=len(out_rows)),
        scope_check=dict(all_candidates_in_scope=True, company=company, year_old=year_old, year_new=year_new,
                         change_type=change_type, topic_range=[1, topics["k"]]),
        topic_allocation=[dict(topic_id=t, candidates=counts.get(t, 0), after_dedup=caps.get(t, 0),
                               ideal=round(ideal.get(t, 0.0), 4), quota=quotas[t],
                               selected=sum(1 for x in selected if x[0] == t)) for t in all_topics],
        join_failures=failures,
        duplicates_excluded=dup_excluded,
        embeddings=dict(requested=emb_info["requested"], reused=emb_info["reused"], created=emb_info["created"],
                        settings=emb_info["settings"]),
        parameters=dict(top_k=k, duplicate_method=rc["duplicate_method"], duplicate_threshold=float(rc["duplicate_threshold"]),
                        pair_vector="NEW=new, REMOVED=old, 매칭형=concat(old,new) 정규화",
                        topic_assignment="argmax topic probability (동률 시 작은 번호)",
                        tfidf=dict(analyzer=cfg["tfidf"]["analyzer"], ngram_range=list(cfg["tfidf"]["ngram_range"]),
                                   number_masking=cfg["tfidf"].get("number_masking", {})),
                        duplicate_text="exact=원문 공백 정규화 일치, near=숫자 마스킹 텍스트의 TF-IDF char 코사인"),
        package_versions=_package_versions(),
    )
    result = dict(request=request, summary=summary, topics=topic_summary, _rows=out_rows, _audit=audit)
    if write:
        out_dir = _path(rc["output_dir"]) / industry / company / f"{year_old}_{year_new}" / change_type
        result["_files"] = serialize_result(result, out_dir)
    return result
