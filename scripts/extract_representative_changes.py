"""대표 변화 문장쌍 추출 CLI.

단건:
    python scripts/extract_representative_changes.py --company 삼성SDI --year-old 2023 --year-new 2024 \
        --change-type EXPANDED --top-k 10
    (--change-type 생략 = 토픽 조인 후보가 가장 많은 유형 자동 선택)
전체(9개 기업 x 4개 연도쌍 x 5개 유형) 생성 후 analysis/demo.html에 번들:
    python scripts/extract_representative_changes.py --all --update-demo
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.representative_change import (CHANGE_TYPES, INDUSTRIES, default_embedder, extract_representatives,  # noqa: E402
                                        load_config)

DEMO = ROOT / "analysis" / "demo.html"
YEAR_PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
DEMO_MARK = re.compile(r"(const REPS = )(\{.*?\})(;\s*\n)", re.S)


def demo_record(res: dict) -> dict:
    """데모에 필요한 필드만 (원문·ID·위치·토픽·확률·centroid 점수)."""
    keep = ("topic_probability", "centroid_cosine_similarity", "rank_in_topic", "global_selection_rank",
            "old_sentence_id", "old_text", "old_section_path", "old_xml_line",
            "new_sentence_id", "new_text", "new_section_path", "new_xml_line", "source_change_row")
    return dict(summary={k: res["summary"][k] for k in ("change_rows_of_type", "candidates", "join_failures", "join_failures_by_reason",
                                                         "duplicates_excluded", "selected")},
                topics=[dict(topic_id=t["topic_id"], label=t["label"], keywords=t["keywords"][:6],
                             candidate_count=t["candidate_count"], candidate_share=t["candidate_share"], quota=t["quota"],
                             reps=[{k: r[k] for k in keep} for r in t["representatives"]])
                        for t in res["topics"]],
                lda_artifact_version=res["summary"]["lda_artifact_version"],
                auto=res["request"]["change_type_selection"]["mode"] == "auto")


def update_demo(bundle: dict) -> None:
    s = DEMO.read_text(encoding="utf-8")
    m = DEMO_MARK.search(s)
    if not m:
        raise SystemExit("analysis/demo.html에 'const REPS = {...};' 블록이 없습니다")
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    DEMO.write_text(s[:m.start(2)] + payload + s[m.end(2):], encoding="utf-8", newline="")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--company")
    ap.add_argument("--year-old", type=int)
    ap.add_argument("--year-new", type=int)
    ap.add_argument("--change-type", choices=CHANGE_TYPES)
    ap.add_argument("--top-k", type=int, help="기본값: config/settings.yaml representative_changes.top_k")
    ap.add_argument("--industry", choices=sorted(INDUSTRIES))
    ap.add_argument("--all", action="store_true", help="9개 기업 x 4개 연도쌍 x 5개 유형 + 자동 선택 모두 생성")
    ap.add_argument("--update-demo", action="store_true", help="--all 결과를 analysis/demo.html에 번들")
    a = ap.parse_args(argv)
    cfg = load_config()
    emb = default_embedder(cfg)  # 모델은 처음 필요할 때 한 번만 올림

    if not a.all:
        if not (a.company and a.year_old and a.year_new):
            ap.error("--company, --year-old, --year-new 가 필요합니다 (또는 --all)")
        res = extract_representatives(a.company, a.year_old, a.year_new, a.change_type, a.top_k, a.industry,
                                      config=cfg, embedder=emb)
        s = res["summary"]
        print(f"{a.company} {a.year_old}->{a.year_new} {res['request']['change_type']}: 후보 {s['candidates']} "
              f"(조인 실패 {s['join_failures']}, 중복 제외 {s['duplicates_excluded']}) -> 선택 {s['selected']}")
        for t in res["topics"]:
            print(f"  T{t['topic_id']} {t['label']}: 후보 {t['candidate_count']} · 할당 {t['quota']}")
            for r in t["representatives"]:
                print(f"    #{r['global_selection_rank']} ({r['centroid_cosine_similarity']:.3f}) "
                      f"{(r['old_text'] or '대응 문장 없음')[:60]} => {(r['new_text'] or '대응 문장 없음')[:60]}")
        print("저장:", res["_files"]["csv"].parent)
        return

    bundle = {}
    for industry, companies in INDUSTRIES.items():
        for company in companies:
            for y0, y1 in YEAR_PAIRS:
                key = f"{company}|{str(y0)[2:]}→{str(y1)[2:]}"
                entry = {}
                for ct in CHANGE_TYPES:
                    res = extract_representatives(company, y0, y1, ct, a.top_k, industry, config=cfg, embedder=emb)
                    entry[ct] = demo_record(res)
                auto = extract_representatives(company, y0, y1, None, a.top_k, industry, config=cfg, embedder=emb, write=False)
                entry["auto"] = auto["request"]["change_type"]
                bundle[key] = entry
                print(key, {ct: entry[ct]["summary"]["selected"] for ct in CHANGE_TYPES}, "auto:", entry["auto"], flush=True)
    if a.update_demo:
        update_demo(bundle)
        print("analysis/demo.html 갱신")


if __name__ == "__main__":
    main()
