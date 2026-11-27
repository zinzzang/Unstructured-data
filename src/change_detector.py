"""sentence_matches -> detected_changes. 정렬 점수(similarity)와 변화 판정(classifier)은 별개로 기록."""
from pathlib import Path

import yaml

from . import database
from .change_classifier import classify as rule_classify

ROOT = Path(__file__).resolve().parent.parent
MIN_LEN = 15  # '경쟁요소' 같은 제목성 짧은 조각은 신규/삭제 판단에서 제외


def detect(con, company, y0, y1, method, cfg, classifier=rule_classify):
    q = """SELECT m.*, o.text AS otext, n.text AS ntext, COALESCE(o.subsection_title, n.subsection_title) AS sub, COALESCE(o.analysis_area, n.analysis_area) AS area
           FROM sentence_matches m LEFT JOIN sentences o ON o.sentence_id=m.old_sentence_id LEFT JOIN sentences n ON n.sentence_id=m.new_sentence_id
           WHERE m.method=? AND ((o.report_id IN (SELECT report_id FROM reports JOIN companies USING(company_id) WHERE name=? AND business_year=?))
                              OR (n.report_id IN (SELECT report_id FROM reports JOIN companies USING(company_id) WHERE name=? AND business_year=?)))"""
    ms = con.execute(q, (method, company, y0, company, y1)).fetchall()
    other = "embedding" if method == "tfidf" else "tfidf"
    oth = {(r["old_sentence_id"], r["new_sentence_id"]): r["similarity"] for r in con.execute(
        "SELECT old_sentence_id, new_sentence_id, similarity FROM sentence_matches WHERE method=? AND kind='PAIR'", (other,))}
    info = f"{method}, threshold={cfg['align']['threshold'][method]}, rules=v1"
    rows = []
    for m in ms:
        sims = {method: m["similarity"], other: oth.get((m["old_sentence_id"], m["new_sentence_id"]))}
        base = dict(old=m["old_sentence_id"], new=m["new_sentence_id"], topic=m["sub"] or m["area"], method=method, info=info,
                    tfidf=sims["tfidf"], emb=sims["embedding"])
        if m["kind"] == "PAIR" and m["is_match"]:
            c = classifier(m["otext"], m["ntext"])
            rows.append({**base, **c, "review": "auto" if c["change_type"] != "UNCERTAIN" else "needs_review"})
        elif m["kind"] == "UNMATCHED_NEW" and len(m["ntext"]) >= MIN_LEN:
            # 대응 문장 없음 != 실제 신규 사업. 후보로만 기록하고 사람이 확인하도록 표시
            rows.append({**base, "change_type": "NEW", "confidence": 0.4, "reason": "전년도에 대응 문장 없음", "change_summary": None, "review": "needs_review"})
        elif m["kind"] == "UNMATCHED_OLD" and len(m["otext"]) >= MIN_LEN:
            rows.append({**base, "change_type": "REMOVED", "confidence": 0.4, "reason": "당해 연도에 대응 문장 없음", "change_summary": None, "review": "needs_review"})
    return rows


def run(methods=("tfidf", "embedding")):
    cfg = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    ys = cfg["mvp"]["years"]
    con = database.connect()
    for method in methods:
        for c in cfg["mvp"]["companies"]:
            for y0, y1 in zip(ys, ys[1:]):
                rows = detect(con, c, y0, y1, method, cfg)
                database.save_changes(con, method, rows)
                from collections import Counter
                print(f"[{method}] {c} {y0}->{y1}:", dict(Counter(r["change_type"] for r in rows)))
