"""preprocessing_v2/<산업>/paragraphs.jsonl -> 인접 연도 쌍(old->new) 문장 정렬 + 변화 분류 -> CSV (DB 없음).

파이프라인 03 TF-IDF matching + 04 Change classification 실행 진입점.
사용: python run_v1.py [--output-dir data/results_v2] [--threshold 0.3] [--company 삼성SDI] [--no-number-masking]
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from src import tfidf_model
from src.change_classifier import classify

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "preprocessing_v2"
OUT = ROOT / "data" / "results_v2"
MIN_LEN = 15  # 제목성 짧은 조각은 신규/삭제 판단 제외


def load():
    """{(company, year, area): [{id, text, path}]}  (duplicate_text 플래그 제외)"""
    g = defaultdict(list)
    for f in sorted(DATA.glob("*/paragraphs.jsonl")):
        for line in f.open(encoding="utf-8"):
            p = json.loads(line)
            if "duplicate_text" in p["quality_flags"]:
                continue
            for s in p["sentences"]:
                g[(p["company"], p["business_year"], p["analysis_area"])].append(
                    dict(id=s["sentence_id"], text=s["text"], path=" > ".join(p["title_path"])))
    return g


def compare(old, new, cfg):
    """같은 기업·영역의 old/new 문장을 TF-IDF 코사인으로 매칭하고 변화 유형을 붙인다."""
    th, k = cfg["align"]["threshold"], cfg["align"]["top_k"]
    S = tfidf_model.similarity([s["text"] for s in old], [s["text"] for s in new], cfg)
    rows, hit = [], set()
    for i, o in enumerate(old):
        j = int(np.argmax(S[i]))
        if S[i, j] >= th:
            hit.add(j)
            c = classify(o["text"], new[j]["text"])
            rows.append(dict(old_id=o["id"], new_id=new[j]["id"], old_path=o["path"], new_path=new[j]["path"],
                             old_text=o["text"], new_text=new[j]["text"], similarity=round(float(S[i, j]), 4),
                             top_k=[round(float(x), 3) for x in np.sort(S[i])[::-1][:k]], **c))
        elif len(o["text"]) >= MIN_LEN:
            rows.append(dict(old_id=o["id"], old_path=o["path"], old_text=o["text"], similarity=round(float(S[i, j]), 4),
                             change_type="REMOVED", confidence=0.4, reason="당해 연도에 대응 문장 없음(후보)"))
    for j, n in enumerate(new):
        if j not in hit and S[:, j].max() < th and len(n["text"]) >= MIN_LEN:
            rows.append(dict(new_id=n["id"], new_path=n["path"], new_text=n["text"], similarity=round(float(S[:, j].max()), 4),
                             change_type="NEW", confidence=0.4, reason="전년도에 대응 문장 없음(후보)"))
    return rows


def run_name(th, number_masking):
    return f"{'tfidf_masked' if number_masking else 'tfidf'}_th{th}"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT, help="result directory")
    parser.add_argument("--threshold", type=float, help="매칭 임계값 (기본: settings.yaml align.threshold)")
    parser.add_argument("--company", action="append", help="process only this company; may be repeated")
    parser.add_argument("--no-number-masking", action="store_true", help="disable TF-IDF number masking for baseline comparison")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    if args.no_number_masking:
        cfg["tfidf"]["number_masking"]["enabled"] = False
    if args.threshold is not None:
        cfg["align"]["threshold"] = args.threshold
    g = load()
    companies = sorted({c for c, _, _ in g})
    if args.company:
        unknown = set(args.company) - set(companies)
        if unknown:
            parser.error(f"unknown company: {', '.join(sorted(unknown))}")
        companies = [c for c in companies if c in args.company]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    name = run_name(cfg["align"]["threshold"], cfg["tfidf"]["number_masking"]["enabled"])
    f = args.output_dir / f"{name}.csv"
    prev = list(csv.DictReader(f.open(encoding="utf-8-sig"))) if f.exists() else []
    done = {(r["company"], int(r["year_old"])) for r in prev}  # 이미 있는 연도 쌍은 건너뛰고 새 쌍만 합산
    out = []
    for c in companies:
        years = sorted({y for cc, y, _ in g if cc == c})
        for y0, y1 in zip(years, years[1:]):  # 인접 연도 쌍만 (2021->2022 ... 2024->2025)
            if (c, y0) in done:
                continue
            for a in sorted({a for cc, y, a in g if cc == c and y in (y0, y1)}):
                old, new = g.get((c, y0, a), []), g.get((c, y1, a), [])
                if not old or not new:  # 한쪽에 영역 없음: 정렬 불가
                    continue
                out += [dict(company=c, year_old=y0, year_new=y1, area=a, method=name, **r) for r in compare(old, new, cfg)]
        print(f"[{name}] {c} {years}", flush=True)
    rows = sorted(prev + out, key=lambda r: int(r["year_old"]))  # 연도 쌍 순, 같은 쌍 안에서는 기존 순서 유지
    cols = list(dict.fromkeys(k for r in rows for k in r))
    tmp = f.with_suffix(f"{f.suffix}.tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(f)
    print(name, f"기존 {len(prev)}행 + 신규 {len(out)}행")
    for k, v in sorted(Counter((r["company"], int(r["year_old"]), r["change_type"]) for r in rows).items()):
        print(name, *k, v)


if __name__ == "__main__":
    main()
