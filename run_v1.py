"""preprocessing_v2/<산업>/paragraphs.jsonl -> 인접 연도 쌍(old->new) 문장 정렬 + 변화 분류 -> CSV (DB 없음).
사용: python run_v1.py [tfidf|bge|KR-SBERT ...]   (기본: 전부)  /  결과: data/results_v1/<방법>_th<임계값>.csv"""
import csv, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from src import tfidf_model, embedding_model
from src.change_classifier import classify

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "preprocessing_v2"  # 2021~2025 (2023~2025는 v1과 동일)
OUT = ROOT / "data" / "results_v1"
METHODS = {"tfidf": tfidf_model, "embedding": embedding_model}
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


def compare(old, new, method, cfg):
    th, k = cfg["align"]["threshold"][method], cfg["align"]["top_k"]
    S = METHODS[method].similarity([s["text"] for s in old], [s["text"] for s in new], cfg)
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


RUNS = [  # (방법, 임베딩 모델, 임계값) -> 결과 파일명 data/results_v1/<name>.csv
    ("tfidf", None, 0.6),
    ("embedding", "BAAI/bge-m3", 0.6),
    ("embedding", "snunlp/KR-SBERT-V40K-klueNLI-augSTS", 0.6),
    ("tfidf", None, 0.3),
]


def run_name(method, model, th):
    return f"tfidf_th{th}" if method == "tfidf" else f"emb_{model.split('/')[-1]}_th{th}"


def main(only):
    base = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())
    g = load()
    companies = sorted({c for c, _, _ in g})
    OUT.mkdir(parents=True, exist_ok=True)
    for method, model, th in RUNS:
        name = run_name(method, model, th)
        if only and not any(o in name for o in only):
            continue
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["align"]["threshold"][method] = th
        if model:
            cfg["embedding"]["model"] = model
            embedding_model._model = None  # 모델 교체 시 이전 모델 인스턴스 폐기
        f = OUT / f"{name}.csv"
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
                    out += [dict(company=c, year_old=y0, year_new=y1, area=a, method=name, **r) for r in compare(old, new, method, cfg)]
            print(f"[{name}] {c} {years}", flush=True)
        rows = sorted(prev + out, key=lambda r: int(r["year_old"]))  # 연도 쌍 순(21-22 -> 24-25), 같은 쌍 안에서는 기존 순서 유지
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(f, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows)
        print(name, f"기존 {len(prev)}행 + 신규 {len(out)}행")
        for k, v in sorted(Counter((r["company"], int(r["year_old"]), r["change_type"]) for r in rows).items()):
            print(name, *k, v)


if __name__ == "__main__":
    main(sys.argv[1:])  # 인자: 이름 일부(예: tfidf, bge, KR-SBERT). 없으면 전부
