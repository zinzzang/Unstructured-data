"""대표 변화 문장쌍 추출 (기업별 LDA 토픽 · th=0.3 · 변화 크기 순).

데모 위쪽 차트와 같은 입력(tfidf_masked_th0.3.csv)과 같은 기업별 LDA(scripts/run_lda_analysis.py)를 쓴다.
run_lda_analysis.py는 수정하지 않고 import만 한다. 같은 입력과 random_state=42이므로 토픽 배정이 위 차트와
그대로 같다. 실행할 때 demo.html의 ALL(sbar·topics)과 맞는지 검사하고, 다르면 중단한다.

절차 (기업·연도쌍·변화유형마다)
  1. 변화 행 중 기업별 LDA 대상(RISK 섹션 제외)을 후보로 한다. 문장의 토픽 = 확률 최대 토픽.
  2. 변화 크기 = 1 - TF-IDF 유사도(CSV similarity).
     NEW/REMOVED: 상대 연도에서 가장 비슷한 문장과의 유사도. 낮을수록 완전히 새로 생기거나 사라진 내용.
     EXPANDED/REDUCED/MODIFIED: 전후 문장 유사도. 낮을수록 많이 바뀐 문장.
  3. 토픽별로 변화 크기 순으로 보며 완전 중복·근접 중복(숫자 마스킹 char TF-IDF 코사인 >= 0.92)을 빼고 3개를 고른다.

단건:
    python scripts/extract_representative_changes.py --company SK하이닉스 --year-old 2022 --year-new 2023 --change-type NEW
전체(9개 기업 x 4개 연도쌍 x 5개 유형) 생성 후 analysis/demo.html에 번들:
    python scripts/extract_representative_changes.py --all --update-demo
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_lda_analysis as company_lda  # noqa: E402  (읽기 전용: 기업별 LDA 그대로 재사용)
from src.tfidf_model import mask_numbers  # noqa: E402

INDUSTRIES = {
    "semiconductor": ["SK하이닉스", "삼성전자", "한미반도체"],
    "battery": ["LG에너지솔루션", "삼성SDI", "SK이노베이션"],
    "biopharma": ["삼성바이오로직스", "셀트리온", "유한양행"],
}
# 근접 중복 비교용 TF-IDF (config/settings.yaml tfidf 와 같은 값)
DUP_TFIDF = dict(analyzer="char", ngram_range=(2, 5))
MASK_TOKEN = "NUM"

SOURCE_CSV = ROOT / "data" / "results_v2" / "tfidf_masked_th0.3.csv"
PARAGRAPHS_DIR = ROOT / "data" / "processed" / "preprocessing_v2"
OUT_DIR = ROOT / "data" / "results_representative_changes" / "company_lda_th0.3"
DEMO = ROOT / "analysis" / "demo.html"
YEAR_PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
CHANGE_TYPES = ("NEW", "EXPANDED", "MODIFIED", "REDUCED", "REMOVED")
PER_TOPIC = 3
DUP_THRESHOLD = 0.92

# demo.html topicLabel()과 같은 규칙 (토픽 이름이 위 차트와 같게 보이도록)
KW_NOISE = {'col', 'Col', 'COL', '장부', '유형', '이사', '찬성', '반대', '이내', '이후', '이전', '합계', '소계', '세후', '세전',
            '포괄', '손익', '영향', '자본', '회사', '당사', '관계', '종속'}

DEMO_MARK = re.compile(r"(const REPS = )(\{.*?\})(;\s*\n)", re.S)
DEMO_ALL = re.compile(r"const ALL\s*=\s*")
NOTE_RE = re.compile(r'(<p class="rep-note">)(.*?)(</p>)', re.S)
NOTE_TEXT = "기업별 LDA 토픽 기준(위 차트와 같은 토픽 번호) · 토픽별 변화가 가장 뚜렷한 3개(TF-IDF 유사도 낮은 순) · 중복 제외"
META_OLD = "['centroid',fmt(r.centroid_cosine_similarity,3)]"
META_NEW = "['유사도',fmt(r.similarity,3)]"


def clean_kw(w: str) -> bool:
    return w not in KW_NOISE and not re.fullmatch(r"[a-z]{1,2}", w, re.I)


def topic_label(kws: list[str]) -> str:
    c = [w for w in kws if clean_kw(w)][:2]
    return "/".join(c if c else kws[:2])


def industry_of(company: str) -> str:
    return next(ind for ind, cos in INDUSTRIES.items() if company in cos)


def load_rows() -> dict[str, list[dict]]:
    """run_lda_analysis.analyze()와 같은 키·같은 행 순서."""
    by_key: dict[str, list[dict]] = defaultdict(list)
    with SOURCE_CSV.open(encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            row["source_change_row"] = i
            by_key[f"{row['company']}|{row['year_old'][-2:]}→{row['year_new'][-2:]}"].append(row)
    return by_key


def load_sentence_index(paragraphs_jsonl: Path) -> dict:
    """sentence_id -> 원문·섹션 경로·XML 위치 (JSONL 역조회)."""
    idx = {}
    with paragraphs_jsonl.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            path = " > ".join(p["title_path"])
            for s in p["sentences"]:
                idx[s["sentence_id"]] = dict(text=s["text"], section_path=path, xml_line=p["source_location"]["xml_line"])
    return idx


def load_demo_all() -> dict:
    s = DEMO.read_text(encoding="utf-8")
    m = DEMO_ALL.search(s)
    if not m:
        raise SystemExit("analysis/demo.html에 'const ALL = {...}' 블록이 없습니다")
    obj, _ = json.JSONDecoder().raw_decode(s[m.end():])
    return obj


def company_topics(rows: list[dict]) -> dict | None:
    """run_lda_analysis.analyze()의 LDA 부분을 그대로 재현. 반환: 토픽·대상 행·행별 토픽/확률·sbar."""
    changed = [r for r in rows if r["change_type"] in company_lda.CT_CHANGED and r.get("area") != "RISK"]
    texts = [r["old_text"] if r["change_type"] == "REMOVED" else r["new_text"] for r in changed]
    if len(texts) < 5:
        return None
    n = 2 if len(texts) < 20 else 3 if len(texts) < 50 else 4
    res = company_lda.run_lda(texts, n_topics=n)
    if not res:
        return None
    topics = res["topics"]
    probs = np.array(res["doc_topics"])
    assign = [max(range(len(topics)), key=lambda i: p[i]) for p in res["doc_topics"]]
    sbar = {ct: [0] * len(topics) for ct in company_lda.CT_CHANGED}
    for r, t in zip(changed, assign):
        sbar[r["change_type"]][t] += 1
    return dict(topics=topics, rows=changed, assign=assign, probs=probs, sbar=sbar)


def align_to_demo(key: str, lda: dict | None, demo_all: dict) -> None:
    """다시 계산한 기업별 LDA를 demo.html 위쪽 차트(ALL)에 맞춘다.

    exact : 배정 건수(sbar)가 정확히 같고, 토픽마다 키워드가 8개 중 6개 이상 겹침
            (가중치 동점이면 키워드 순서나 8번째 단어가 바뀔 수 있음).
    approx: 실행 환경(kiwipiepy·scikit-learn 버전) 차이로 배정이 달라진 경우. 다시 계산한 토픽을
            키워드가 가장 많이 겹치는 차트 토픽에 1:1로 짝지어(헝가리안) 번호를 바꾼다.
            토픽 이름은 맞지만 토픽별 후보 수는 차트와 조금 다를 수 있다.
    """
    d = demo_all.get(key)
    if d is None:
        raise SystemExit(f"demo.html ALL에 {key}가 없습니다")
    if lda is None or not d["lda_ok"]:
        if (lda is None) != (not d["lda_ok"]):
            raise SystemExit(f"{key}: LDA 실행 여부가 demo.html 위쪽 차트와 다릅니다")
        return
    if len(lda["topics"]) != len(d["topics"]):
        raise SystemExit(f"{key}: 토픽 수가 demo.html 위쪽 차트와 다릅니다")
    exact = lda["sbar"] == d["sbar"] and all(len(set(a) & set(b)) >= 6 for a, b in zip(lda["topics"], d["topics"]))
    lda["approx"] = not exact
    if not exact:
        from scipy.optimize import linear_sum_assignment
        overlap = np.array([[len(set(a) & set(b)) for b in d["topics"]] for a in lda["topics"]])
        rows, cols = linear_sum_assignment(-overlap)
        perm = dict(zip(rows.tolist(), cols.tolist()))  # 다시 계산한 토픽 -> 차트 토픽
        lda["assign"] = [perm[t] for t in lda["assign"]]
        probs = np.zeros_like(lda["probs"])
        for r, c in perm.items():
            probs[:, c] = lda["probs"][:, r]
        lda["probs"] = probs
        lda["sbar"] = {ct: [0] * len(d["topics"]) for ct in lda["sbar"]}
        for r, t in zip(lda["rows"], lda["assign"]):
            lda["sbar"][r["change_type"]][t] += 1
        print(f"  {key}: 근사 매핑 (키워드 겹침 {[int(overlap[r, perm[r]]) for r in sorted(perm)]}/8)", flush=True)
    lda["topics"] = d["topics"]  # 토픽 이름이 위 차트와 똑같이 보이도록 데모의 키워드 순서를 쓴다
    lda["labels"] = d.get("topic_labels") or []  # lda_results.json에서 직접 붙인 토픽 이름


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def select_type(rows: list[dict], lda: dict | None, ct: str, sent_idx: dict) -> dict:
    rows_of_type = [r for r in rows if r["change_type"] == ct]
    cands = []
    if lda:
        for r, t, p in zip(lda["rows"], lda["assign"], lda["probs"]):
            if r["change_type"] == ct:
                cands.append(dict(row=r, topic=t, prob=float(p[t]), sim=float(r["similarity"])))
    n_fail = len(rows_of_type) - len(cands)
    reason = {"lda_filtered": n_fail} if n_fail and lda else ({"lda_not_run": n_fail} if n_fail else {})

    def pair_text(c):
        return " → ".join(x for x in (c["row"]["old_text"], c["row"]["new_text"]) if x)

    by_topic = defaultdict(list)
    for c in cands:
        by_topic[c["topic"]].append(c)

    from sklearn.feature_extraction.text import TfidfVectorizer
    topics_out, picked, dup_count = [], [], 0
    for ti in sorted(by_topic):
        group = sorted(by_topic[ti], key=lambda c: (c["sim"], c["row"]["source_change_row"]))  # 변화 크기 큰 순
        docs = [_norm(pair_text(c)) for c in group]
        X = TfidfVectorizer(**DUP_TFIDF).fit_transform([mask_numbers(d, MASK_TOKEN) for d in docs])
        kept, seen = [], set()
        for i, c in enumerate(group):
            if len(kept) == PER_TOPIC:
                break
            if docs[i] in seen or (kept and (X[kept] @ X[i].T).toarray().max() >= DUP_THRESHOLD):
                dup_count += 1
                continue
            kept.append(i)
            seen.add(docs[i])
        kws = lda["topics"][ti]
        labels = lda.get("labels") or []
        topics_out.append(dict(topic_id=ti + 1, label=labels[ti] if ti < len(labels) and labels[ti] else topic_label(kws), keywords=[w for w in kws if clean_kw(w)][:6],
                               candidate_count=len(group), candidate_share=round(len(group) / len(cands), 4),
                               quota=PER_TOPIC, reps=[group[i] for i in kept]))
        picked += [group[i] for i in kept]

    rank = {id(c): n for n, c in enumerate(sorted(picked, key=lambda c: (c["sim"], c["row"]["source_change_row"])), 1)}

    def loc(sid, fallback_text, fallback_path):
        if not sid:
            return None, None, None, None
        s = sent_idx.get(sid, {})
        return sid, s.get("text", fallback_text), s.get("section_path", fallback_path), s.get("xml_line")

    for t_out in topics_out:
        recs = []
        for c in t_out["reps"]:
            r = c["row"]
            o_id, o_text, o_path, o_line = loc(r["old_id"], r["old_text"], r["old_path"])
            n_id, n_text, n_path, n_line = loc(r["new_id"], r["new_text"], r["new_path"])
            recs.append(dict(global_selection_rank=rank[id(c)], topic_probability=round(c["prob"], 4),
                             similarity=c["sim"], change_magnitude=round(1 - c["sim"], 4),
                             old_sentence_id=o_id, old_text=o_text, old_section_path=o_path, old_xml_line=o_line,
                             new_sentence_id=n_id, new_text=n_text, new_section_path=n_path, new_xml_line=n_line,
                             source_change_row=r["source_change_row"]))
        t_out["reps"] = recs
    return dict(summary=dict(change_rows_of_type=len(rows_of_type), candidates=len(cands), join_failures=n_fail,
                             join_failures_by_reason=reason, duplicates_excluded=dup_count, selected=len(picked)),
                topics=topics_out, source_csv="data/results_v2/tfidf_masked_th0.3.csv", topic_model="company_lda",
                topic_mapping="approx" if lda and lda.get("approx") else "exact")


def build(keys: list[str], cts: tuple[str, ...]) -> dict:
    rows_by_key = load_rows()
    demo_all = load_demo_all()
    sent_cache: dict[str, dict] = {}
    bundle = {}
    for key in keys:
        rows = rows_by_key.get(key)
        if not rows:
            raise SystemExit(f"{key}: 변화 CSV에 행이 없습니다")
        ind = industry_of(rows[0]["company"])
        lda = company_topics(rows)
        align_to_demo(key, lda, demo_all)
        if ind not in sent_cache:
            sent_cache[ind] = load_sentence_index(PARAGRAPHS_DIR / ind / "paragraphs.jsonl")
        bundle[key] = {ct: select_type(rows, lda, ct, sent_cache[ind]) for ct in cts}
        print(key, {ct: bundle[key][ct]["summary"]["selected"] for ct in cts}, flush=True)
    return bundle


def write_csv(bundle: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "representatives.csv"
    cols = ["key", "change_type", "topic_id", "topic_label", "global_selection_rank", "similarity", "change_magnitude",
            "topic_probability", "source_change_row", "old_sentence_id", "old_text", "new_sentence_id", "new_text"]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for key, entry in bundle.items():
            for ct, d in entry.items():
                for t in d["topics"]:
                    for r in t["reps"]:
                        w.writerow({"key": key, "change_type": ct, "topic_id": t["topic_id"], "topic_label": t["label"],
                                    **{c: r[c] for c in cols[4:]}})
    return out


def update_demo(bundle: dict) -> None:
    s = DEMO.read_text(encoding="utf-8")
    m = DEMO_MARK.search(s)
    if not m:
        raise SystemExit("analysis/demo.html에 'const REPS = {...};' 블록이 없습니다")
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    s = s[:m.start(2)] + payload + s[m.end(2):]
    s = NOTE_RE.sub(lambda mm: mm.group(1) + NOTE_TEXT + mm.group(3), s, count=1)
    s = s.replace(META_OLD, META_NEW)
    DEMO.write_text(s, encoding="utf-8", newline="")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--company")
    ap.add_argument("--year-old", type=int)
    ap.add_argument("--year-new", type=int)
    ap.add_argument("--change-type", choices=CHANGE_TYPES)
    ap.add_argument("--all", action="store_true", help="9개 기업 x 4개 연도쌍 x 5개 유형 모두 생성")
    ap.add_argument("--update-demo", action="store_true", help="--all 결과를 analysis/demo.html에 번들")
    a = ap.parse_args(argv)

    if not a.all:
        if not (a.company and a.year_old and a.year_new):
            ap.error("--company, --year-old, --year-new 가 필요합니다 (또는 --all)")
        key = f"{a.company}|{str(a.year_old)[2:]}→{str(a.year_new)[2:]}"
        cts = (a.change_type,) if a.change_type else CHANGE_TYPES
        for ct, d in build([key], cts)[key].items():
            s = d["summary"]
            print(f"{key} {ct}: 변화 {s['change_rows_of_type']} · 후보 {s['candidates']} "
                  f"(제외 {s['join_failures']}, 중복 {s['duplicates_excluded']}) -> 선택 {s['selected']}")
            for t in d["topics"]:
                print(f"  T{t['topic_id']} {t['label']}: 후보 {t['candidate_count']}")
                for r in t["reps"]:
                    print(f"    #{r['global_selection_rank']} (유사도 {r['similarity']:.3f}) "
                          f"{(r['old_text'] or '대응 문장 없음')[:50]} => {(r['new_text'] or '대응 문장 없음')[:50]}")
        return

    keys = [f"{c}|{str(y0)[2:]}→{str(y1)[2:]}" for cos in INDUSTRIES.values() for c in cos for y0, y1 in YEAR_PAIRS]
    bundle = build(keys, CHANGE_TYPES)
    print("저장:", write_csv(bundle))
    if a.update_demo:
        update_demo(bundle)
        print("analysis/demo.html 갱신")


if __name__ == "__main__":
    main()
