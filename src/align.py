"""같은 기업·연속 사업연도·같은 분석영역 안에서 문장 정렬 -> sentence_matches."""
from pathlib import Path

import numpy as np
import yaml

from . import database, tfidf_model

ROOT = Path(__file__).resolve().parent.parent
METHODS = {"tfidf": tfidf_model}


def align_pair(con, company, y0, y1, method, cfg):
    old_all, new_all = database.sentences_of(con, company, y0), database.sentences_of(con, company, y1)
    th, k = cfg["align"]["threshold"], cfg["align"]["top_k"]
    rows = []
    for area in sorted({s["analysis_area"] for s in old_all} | {s["analysis_area"] for s in new_all}):
        old = [s for s in old_all if s["analysis_area"] == area]
        new = [s for s in new_all if s["analysis_area"] == area]
        if not old or not new:  # 한쪽에 영역 자체가 없음: 누락과 실제 종료를 구분 못 하므로 미매칭 후보로만 기록
            rows += [dict(old=s["sentence_id"], new=None, kind="UNMATCHED_OLD", rank=None, similarity=None, is_match=0) for s in old]
            rows += [dict(old=None, new=s["sentence_id"], kind="UNMATCHED_NEW", rank=None, similarity=None, is_match=0) for s in new]
            continue
        S = METHODS[method].similarity([s["text"] for s in old], [s["text"] for s in new], cfg)
        hit_new = set()
        for i, s in enumerate(old):
            top = np.argsort(-S[i])[:k]
            for rank, j in enumerate(top, 1):
                rows.append(dict(old=s["sentence_id"], new=new[j]["sentence_id"], kind="PAIR", rank=rank,
                                 similarity=round(float(S[i, j]), 4), is_match=int(rank == 1 and S[i, j] >= th)))
            if S[i, top[0]] >= th:
                hit_new.add(int(top[0]))
            else:
                rows.append(dict(old=s["sentence_id"], new=None, kind="UNMATCHED_OLD", rank=None, similarity=round(float(S[i].max()), 4), is_match=0))
        for j, s in enumerate(new):  # 어떤 old의 최선 대응도 아니고 임계값 이상 old도 없으면 신규 후보
            if j not in hit_new and S[:, j].max() < th:
                rows.append(dict(old=None, new=s["sentence_id"], kind="UNMATCHED_NEW", rank=None, similarity=round(float(S[:, j].max()), 4), is_match=0))
    return rows, [s["sentence_id"] for s in old_all + new_all]


def run(methods=("tfidf",)):
    cfg = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    ys = cfg["mvp"]["years"]
    con = database.connect()
    for method in methods:
        for c in cfg["mvp"]["companies"]:
            for y0, y1 in zip(ys, ys[1:]):
                rows, scope = align_pair(con, c, y0, y1, method, cfg)
                database.save_matches(con, method, rows, scope)
                print(f"[{method}] {c} {y0}->{y1}: 후보쌍 {sum(r['kind']=='PAIR' for r in rows)}, 매칭 {sum(r['is_match'] for r in rows)}, "
                      f"미매칭(old {sum(r['kind']=='UNMATCHED_OLD' for r in rows)}/new {sum(r['kind']=='UNMATCHED_NEW' for r in rows)})")
