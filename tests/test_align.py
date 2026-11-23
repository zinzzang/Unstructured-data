import numpy as np
from src import align, tfidf_model

CFG = {"tfidf": {"analyzer": "char", "ngram_range": [2, 5]}, "align": {"threshold": {"tfidf": 0.3}, "top_k": 2}}


def test_tfidf_reordered_sentences_still_align():
    old = ["생산능력은 연간 100만 개이다.", "당사는 메모리 반도체를 판매한다."]
    new = ["당사는 메모리 반도체를 판매한다.", "생산능력은 연간 150만 개이다."]  # 순서 바뀜 + 수치 변경
    S = tfidf_model.similarity(old, new, CFG)
    assert S.argmax(1).tolist() == [1, 0]
    assert S[0, 1] > 0.3  # 수치만 바뀐 문장은 유사도가 높아도 정렬은 됨 (변화 판정은 Phase 4)


def test_unrelated_sentence_is_low():
    S = tfidf_model.similarity(["유상증자를 결정하였다."], ["DRAM 제품의 시장점유율은 높다."], CFG)
    assert S[0, 0] < 0.3
