import hashlib

import numpy as np
import pytest

from src import representative_change as R


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


EMB = {"o1": unit([1, 0, 0]), "n1": unit([0, 1, 0]), "o2": unit([1, 1, 0]), "n2": unit([0, 0, 1])}


def cand(old=None, new=None, topic=1, row=1):
    return dict(old_id=old, new_id=new, topic_id=topic, row={"source_change_row": row})


# ---------------------------------------------------------------- pair vectors (5 types)
def test_pair_vector_new_uses_new_side():
    v = R.build_pair_vectors([cand(new="n1")], "NEW", EMB)
    assert np.allclose(v[0], EMB["n1"])


def test_pair_vector_removed_uses_old_side():
    v = R.build_pair_vectors([cand(old="o2")], "REMOVED", EMB)
    assert np.allclose(v[0], EMB["o2"])


@pytest.mark.parametrize("ct", ["EXPANDED", "MODIFIED", "REDUCED"])
def test_pair_vector_matched_is_normalized_concat(ct):
    v = R.build_pair_vectors([cand(old="o1", new="n1")], ct, EMB)[0]
    assert v.shape == (6,)
    assert np.allclose(v, np.concatenate([EMB["o1"], EMB["n1"]]) / np.sqrt(2))
    assert np.isclose(np.linalg.norm(v), 1.0)


@pytest.mark.parametrize("ct", ["EXPANDED", "MODIFIED", "REDUCED"])
def test_pair_vector_matched_missing_side_raises(ct):
    with pytest.raises(ValueError):
        R.build_pair_vectors([cand(old=None, new="n1")], ct, EMB)


# ---------------------------------------------------------------- centroid / cosine
def test_centroid_cosine_matches_hand_calculation():
    vecs = np.array([unit([1, 0]), unit([0, 1]), unit([1, 1])])
    cands = [cand(topic=1, row=1), cand(topic=1, row=2), cand(topic=1, row=3)]
    R.rank_by_centroid(vecs, cands)
    cen = unit(vecs.mean(axis=0))            # = (1/√2, 1/√2)
    assert np.isclose(cands[0]["centroid_cosine_similarity"], vecs[0] @ cen)  # 1/√2
    assert np.isclose(cands[2]["centroid_cosine_similarity"], 1.0)
    assert [c["centroid_rank"] for c in cands] == [2, 3, 1]  # 동점(1,2행)은 CSV 행 번호 순


def test_centroid_is_per_topic():
    vecs = np.array([unit([1, 0]), unit([0, 1])])
    cands = [cand(topic=1, row=1), cand(topic=2, row=2)]
    R.rank_by_centroid(vecs, cands)
    assert all(np.isclose(c["centroid_cosine_similarity"], 1.0) for c in cands)


# ---------------------------------------------------------------- quotas
def test_quota_proportional():
    q, _ = R.allocate_topic_quotas({1: 60, 2: 30, 3: 10}, {1: 60, 2: 30, 3: 10}, 10)
    assert q == {1: 6, 2: 3, 3: 1}


def test_quota_minimum_one_per_topic_with_candidates():
    q, _ = R.allocate_topic_quotas({1: 97, 2: 2, 3: 1, 4: 0}, {1: 97, 2: 2, 3: 1, 4: 0}, 10)
    assert q == {1: 8, 2: 1, 3: 1, 4: 0}


def test_quota_capped_by_candidates_and_redistributed():
    # 토픽 1 비중은 크지만 중복 제거 후 2개뿐 -> 나머지를 다른 토픽에 재배분
    q, _ = R.allocate_topic_quotas({1: 80, 2: 10, 3: 10}, {1: 2, 2: 10, 3: 10}, 10)
    assert q[1] == 2 and sum(q.values()) == 10 and q[2] + q[3] == 8


def test_quota_when_fewer_candidates_than_k():
    q, _ = R.allocate_topic_quotas({1: 3, 2: 2}, {1: 3, 2: 2}, 10)
    assert q == {1: 3, 2: 2}


def test_quota_more_topics_than_k_takes_largest_then_lower_id():
    q, _ = R.allocate_topic_quotas({1: 5, 2: 9, 3: 5}, {1: 5, 2: 9, 3: 5}, 2)
    assert q == {1: 1, 2: 1, 3: 0}


def test_quota_tie_break_is_deterministic():
    q, _ = R.allocate_topic_quotas({1: 5, 2: 5}, {1: 5, 2: 5}, 3)
    assert q == {1: 2, 2: 1}


# ---------------------------------------------------------------- dedup
CFG = {"tfidf": {"analyzer": "char", "ngram_range": [2, 5], "number_masking": {"enabled": True, "token": "NUM"}}}


def _dedup(texts_by_row, scores):
    texts, cands = {}, []
    for i, (t, s) in enumerate(zip(texts_by_row, scores), start=1):
        texts[f"n{i}"] = t
        c = cand(new=f"n{i}", row=i)
        c["centroid_cosine_similarity"] = s
        cands.append(c)
    return R.select_deduplicated(cands, texts, 0.92, CFG)


def test_exact_duplicate_removed_keeps_higher_score():
    kept, ex = _dedup(["리튬 가격이 상승했다.", "리튬 가격이 상승했다.", "니켈 가격은 하락했다."], [0.5, 0.9, 0.7])
    assert kept == [1, 2]  # index 1(점수 0.9) 유지, index 0 제외
    assert ex == [dict(source_change_row=1, kind="exact", duplicate_of=2, similarity=1.0)]


def test_near_duplicate_removed():
    a = "당사는 헝가리, 미국, 말레이시아, 중국 등 각 국에 20개의 종속기업을 두고 있습니다."
    b = "당사는 헝가리, 미국, 말레이시아, 중국 등 각 국에 21개의 종속기업을 두고 있습니다."
    kept, ex = _dedup([a, b, "전혀 다른 문장으로 배터리 수주 계약을 체결하였습니다."], [0.9, 0.8, 0.7])
    assert kept == [0, 2] and ex[0]["kind"] == "near" and ex[0]["duplicate_of"] == 1  # 숫자만 다름 -> 마스킹 후 코사인 1.0


def test_near_duplicate_threshold_keeps_different_sentences():
    a = "당사는 헝가리, 미국, 말레이시아, 중국 등 각 국에 종속기업을 두고 있습니다."
    b = "당사는 국내 생산법인과 해외 판매법인을 통해 전지 사업을 영위하고 있습니다."
    kept, ex = _dedup([a, b], [0.9, 0.8])
    assert kept == [0, 1] and not ex


# ---------------------------------------------------------------- join: single side / matched errors
TOPICS = {"k": 2, "new": {"n1": np.array([0.2, 0.8])}, "old": {"o1": np.array([0.7, 0.3])}}


def row(ct, old="", new="", i=1):
    return dict(change_type=ct, old_id=old, new_id=new, source_change_row=i)


def test_join_new_uses_new_probabilities():
    c, f = R.join_pairs_with_topics([row("NEW", new="n1")], "NEW", TOPICS)
    assert c[0]["topic_id"] == 2 and c[0]["topic_side"] == "new" and np.isclose(c[0]["topic_probability"], 0.8) and not f


def test_join_removed_uses_old_probabilities():
    c, _ = R.join_pairs_with_topics([row("REMOVED", old="o1")], "REMOVED", TOPICS)
    assert c[0]["topic_id"] == 1 and c[0]["topic_side"] == "old" and np.isclose(c[0]["topic_probability"], 0.7)


def test_join_matched_missing_id_raises():
    with pytest.raises(ValueError):
        R.join_pairs_with_topics([row("MODIFIED", new="n1")], "MODIFIED", TOPICS)


def test_join_failure_reason_for_table_rows():
    idx = {"t9": {"source_type": "table_row"}}
    c, f = R.join_pairs_with_topics([row("NEW", new="t9")], "NEW", TOPICS, idx)
    assert not c and f[0]["reason"].startswith("table_row")


def test_auto_change_type_picks_most_joined():
    rows = [row("NEW", new="n1", i=1), row("REMOVED", old="o1", i=2), row("REMOVED", old="o1", i=3)]
    ct, info = R.select_change_type(rows, TOPICS, None)
    assert ct == "REMOVED" and info["mode"] == "auto"


# ---------------------------------------------------------------- embedding artifact
def fake_embedder(calls):
    def enc(texts):
        calls.append(list(texts))
        out = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            out.append(np.random.default_rng(seed).normal(size=8))
        return np.array(out)
    return enc


def test_embedding_artifact_reuses_and_regenerates_changed_text(tmp_path):
    cfg = {"embedding": {"model": "fake/model", "max_seq_length": 512, "batch_size": 4, "number_masking": {"enabled": False}}}
    calls = []
    e1, i1 = R.load_or_create_embeddings({"a": "가", "b": "나"}, cfg, tmp_path, fake_embedder(calls))
    assert i1["created"] == 2 and (tmp_path / "model_sentence_embeddings.npz").exists()
    e2, i2 = R.load_or_create_embeddings({"a": "가", "b": "나"}, cfg, tmp_path, fake_embedder(calls))
    assert i2["reused"] == 2 and i2["created"] == 0 and np.allclose(e1["a"], e2["a"])
    _, i3 = R.load_or_create_embeddings({"a": "가", "b": "다르게 바뀐 원문"}, cfg, tmp_path, fake_embedder(calls))
    assert i3["created"] == 1 and calls[-1] == ["다르게 바뀐 원문"]
    z = np.load(tmp_path / "model_sentence_embeddings.npz")
    assert z["vectors"].dtype == np.float32 and list(z["sentence_ids"]) == ["a", "b"]
