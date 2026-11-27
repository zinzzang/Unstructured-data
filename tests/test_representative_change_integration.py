"""실제 변화 CSV·JSONL·산업 LDA artifact로 검증. 임베딩은 결정적인 가짜 벡터(모델 불필요)."""
import csv
import hashlib
import re
from collections import defaultdict

import numpy as np
import pytest

from src import representative_change as R

LDA = R.ROOT / "data" / "results_industry_lda_corrected" / "battery"
pytestmark = pytest.mark.skipif(not (LDA / "lda_artifacts.joblib").exists(),
                                reason="scripts/industry_lda_corrected.py battery 를 먼저 실행해야 함")


def fake_embedder(texts):
    return np.array([np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).normal(size=16)
                     for t in texts])


@pytest.fixture
def cfg(tmp_path):
    c = R.load_config()
    c["representative_changes"]["output_dir"] = str(tmp_path / "out")
    c["representative_changes"]["embedding_artifact_dir"] = str(tmp_path / "emb")
    return c


def run(cfg, ct, k=10, company="SK이노베이션", y0=2023, y1=2024):
    return R.extract_representatives(company, y0, y1, ct, k, config=cfg, embedder=fake_embedder)


@pytest.mark.parametrize("ct", R.CHANGE_TYPES)
def test_all_types_in_scope_with_source_text(cfg, ct):
    res = run(cfg, ct)
    sent = R.load_sentence_index(str(R.ROOT / "data/processed/preprocessing_v2/battery/paragraphs.jsonl"))
    s = res["summary"]
    assert s["selected"] == min(10, s["deduplicated_candidates"])
    for t in res["topics"]:
        assert 1 <= t["topic_id"] <= 6
        for r in t["representatives"]:
            for side, y in (("old", 2023), ("new", 2024)):
                sid = r[f"{side}_sentence_id"]
                if sid:  # JSONL 역조회: 원문·섹션 경로·XML 위치가 그대로
                    assert sent[sid]["company"] == "SK이노베이션" and sent[sid]["year"] == y
                    assert r[f"{side}_text"] == sent[sid]["text"] and r[f"{side}_section_path"] and r[f"{side}_xml_line"]
            assert (r["old_sentence_id"] is None) == (ct == "NEW")
            assert (r["new_sentence_id"] is None) == (ct == "REMOVED")
    a = res["_audit"]
    assert a["scope_check"]["all_candidates_in_scope"] and a["counts"]["joined"] + a["counts"]["join_failed"] == a["counts"]["change_rows_of_type"]
    assert (R.Path(cfg["representative_changes"]["output_dir"]) / "battery/SK이노베이션/2023_2024" / ct / "audit.json").exists()


def test_removed_uses_old_side_probability(cfg):
    res = run(cfg, "REMOVED")
    with open(LDA / "sentence_topic_probabilities_old.csv", encoding="utf-8-sig") as f:
        old = {r["sentence_id"]: r for r in csv.DictReader(f)}
    reps = [r for t in res["topics"] for r in t["representatives"]]
    assert reps
    for t in res["topics"]:
        for r in t["representatives"]:
            row = old[r["old_sentence_id"]]
            probs = [float(row[f"topic_{i}"]) for i in range(1, 7)]
            assert row["old_side"] == "removed" and int(np.argmax(probs)) + 1 == t["topic_id"]
            assert np.isclose(r["topic_probability"], max(probs), atol=1e-6)


def test_top_k_is_configurable_and_counts_hold(cfg):
    res = run(cfg, "NEW", k=3)
    assert res["summary"]["selected"] == min(3, res["summary"]["deduplicated_candidates"]) == 3
    assert sum(t["quota"] for t in res["topics"]) == 3


def test_same_input_twice_gives_same_result(cfg):
    a, b = run(cfg, "NEW"), run(cfg, "NEW")
    assert a["topics"] == b["topics"] and a["summary"] == b["summary"]
    assert b["_audit"]["embeddings"]["created"] == 0  # 두 번째는 저장된 임베딩 재사용


def test_lda_artifact_reproduces_new_side_probabilities():
    """저장된 vectorizer/model로 new 문서를 다시 transform하면 기존 sentence_topic_probabilities.csv와 같아야 한다."""
    import joblib
    from kiwipiepy import Kiwi
    art = joblib.load(LDA / "lda_artifacts.joblib")
    stop, units, specific = set(art["stopwords"]), set(art["unit_codes"]), set(art["specific_terms"])
    kiwi = Kiwi()

    def tokens(text):
        out = []
        for z in kiwi.tokenize(text):
            w = z.form.lower()
            if z.tag not in ("NNG", "NNP", "SL") or len(w) < 2 or w in stop:
                continue
            if re.fullmatch(r"[ivx]+|p\d+", w) or w in units or w in specific:
                continue
            out.append(w)
        return out

    with open(LDA / "sentence_topic_probabilities.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    docs = defaultdict(list)
    for r in rows:
        docs[r["doc_id"]].append(r)
    k = art["metadata"]["n_topics"]
    checked = 0
    for doc_id in sorted(docs)[:40]:
        rs = sorted(docs[doc_id], key=lambda r: int(r["sentence_id"].rsplit("-S", 1)[1]))
        x = art["vectorizer"].transform([" ".join(tokens(" ".join(r["text"] for r in rs)))])
        if not x.nnz:
            continue
        theta = art["model"].transform(x)[0]
        expected = np.array([float(rs[0][f"topic_{i}"]) for i in range(1, k + 1)])
        assert np.allclose(theta, expected, atol=1e-8), doc_id
        checked += 1
    assert checked >= 20
