"""TF-IDF 결과(tfidf_masked_th0.3.csv)에서 기업·연도쌍별 LDA 토픽 분석을 실행한다.

출력: analysis/lda_results.json  (demo.html에 임베딩된 데이터와 동일 포맷)

사용:
    python scripts/run_lda_analysis.py
    python scripts/run_lda_analysis.py --input data/results_v2/tfidf_masked_th0.3.csv
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── 불용어 ────────────────────────────────────────────────────────────────────
STOP = {
    "회사", "당사", "관계", "종속", "손익", "포괄", "세후", "세전", "자본",
    "미만", "초과", "차입금", "판매", "지분",
    "이자", "거래", "상품", "대상", "비중", "매출", "성장", "투자", "부문",
    "매입", "현금", "기준", "변동", "증가", "감소", "결산", "계정", "잔액",
    "원화", "외화", "금액", "장부", "원가", "취득", "처분", "상각", "손상",
    "환율", "통화", "파생", "목적", "위험", "관리", "노출", "측정", "평가",
    "주석", "근거", "정책", "재무", "연결", "별도", "작성", "공시", "주요",
    "고려", "다음", "현재", "이상", "이하", "이내", "이후", "이전",
    "합계", "소계", "총계", "일부", "기타", "관련", "경우", "통해",
    "통해", "포함", "제외", "기반", "활동", "운영", "실행", "수행",
    "분기", "반기", "사업", "기업", "법인", "당사", "우리",
    "시장", "단위", "회피", "누계", "저장", "자산", "수요", "예상", "확대", "축소",
    "기말", "요구", "산출", "가격", "처리", "향상", "적용", "해당", "내용", "효율",
    "상승", "하락", "용도", "시작", "정보", "결과", "솔루션", "직접", "경로",
    "제1기", "제2기", "제3기", "제4기", "흐름", "석사", "평균", "조건", "완료",
    "취소", "한편", "대형", "고객", "경험", "성능", "인식", "보고", "범위",
    "합리", "임원", "인원", "단계", "한국", "제출", "회원권", "구분", "품목",
    "건물", "토지", "구축물", "건설", "기계", "장치", "차량", "운반", "공구", "사용권자산",
    "유형", "회계", "소재지", "비품", "기구", "과제명", "연구과제명", "llc", "kce",
    "and", "the", "with", "on", "in", "of", "co", "ltd", "inc", "company",
    "col", "Col", "COL",  # 표 아티팩트
}

STOP_ = set('''
경쟁력 당사 회사 기업 사업 관련 경우 영향 통해 기준 분야 전망 예상 지속 증가 감소 확대 축소 성장 수치 비중 기록 발생
적용 제공 진행 운영 수행 당기 전기 금액 해당 전년 올해 지난해 다음 전체 부분 상황 정도 시기 수준 현재 미래 측면
내용 위한 대응 확보 추진 강화 개선 위하여 최근 중심 가운데 연도 이후 또한 당시 내외 국내 국외 각종 이번 과거 처음
시간 당사자 우리 모든 각각 상기 아래 관계 전반 발표 대한 대하여 이후 이전 기간 시점 예정 계획 노력 다양 지속적
시장 수요 제품 기술 산업 글로벌 세계 주요 대비 담당 출처 자료 참조 가트너 gartner idc omdia trendforce
억원 백만원 천원 달러 비율 규모 실적 사항 현황 부문 사업부 사업부문 연결 종속 자회사 법인 지배
공시 작성 기준일 서류 포함 이상 이하 가능 변화 활용 기반 바탕 보유 특성 방법 기대 사용 필요 일부 다수 해외 신규
레미케이드 아바스틴 허셉틴 휴미라 엔브렐 루센티스 솔리리스 악템라 스텔라라 프롤리아 아일리아 졸레어 리툭산 키트루다
램시마 트룩시마 허쥬마 유플라이마 베그젤마 짐펜트라 스테키마 렉라자 옴리클로 앱토즈마 에피스클리 오퓨비즈 하드리마
'''.split())

STOP_EXTRA = set('''
등 및 또는 등등 기타 기타사항 참고 비고 별도
동일 상이 차이 여부 관련하여 관련된 관한
따른 따라 따라서는 의한 인한 비롯 비롯한
통한 중심으로 중심적 종합 종합적
일반 일반적 기본 기본적 구체적
적극 적극적 효율적 효과적
목적 목표 방침 방향 관점
상세 세부 별첨 첨부 기재 명시
단위 합계 총계 총액
첫째 둘째 셋째 채용 인재 인력 직원 임직원
'''.split())

STOP = STOP | STOP_ | STOP_EXTRA


def tokenize_ko(text: str) -> list[str]:
    """kiwipiepy로 한국어 명사(NNG·NNP) + 영어 단어(SL) 추출. len>=2 필터."""
    try:
        from kiwipiepy import Kiwi
    except ImportError:
        raise SystemExit("kiwipiepy가 설치되지 않았습니다: pip install kiwipiepy")

    _kiwi = getattr(tokenize_ko, "_kiwi", None)
    if _kiwi is None:
        tokenize_ko._kiwi = Kiwi()
        _kiwi = tokenize_ko._kiwi

    tokens = []
    for token in _kiwi.tokenize(text):
        if token.tag in ("NNG", "NNP", "SL") and len(token.form) >= 2:
            if token.form.lower() not in STOP and token.form not in STOP:
                tokens.append(token.form.lower() if token.tag == "SL" else token.form)
    return tokens


def run_lda(texts: list[str], n_topics: int = 4) -> dict | None:
    """문장 리스트 → {topics: [[kw,...]], doc_topics: [[prob,...]]}"""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    tokenized = [" ".join(tokenize_ko(t)) for t in texts]
    min_df = 1 if len(texts) < 40 else 2

    vectorizer = CountVectorizer(max_df=0.90, min_df=min_df, max_features=300)
    try:
        X = vectorizer.fit_transform(tokenized)
    except ValueError:
        return None

    if X.shape[1] < n_topics:
        return None

    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42, max_iter=50
    )
    doc_topics = lda.fit_transform(X)
    vocab = vectorizer.get_feature_names_out()

    topics = []
    for comp in lda.components_:
        top_idx = comp.argsort()[::-1][:8]
        topics.append([vocab[i] for i in top_idx])

    return {"topics": topics, "doc_topics": doc_topics.tolist()}


CT_CHANGED = ["NEW", "EXPANDED", "REMOVED", "MODIFIED", "REDUCED"]


def analyze(csv_path: Path) -> dict:
    """CSV → {key: {...}} 형태의 결과 딕셔너리."""
    rows_by_key: dict[str, list[dict]] = defaultdict(list)
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = f"{row['company']}|{row['year_old'][-2:]}→{row['year_new'][-2:]}"
            rows_by_key[key].append(row)

    results = {}
    for key, rows in sorted(rows_by_key.items()):
        company = rows[0]["company"]
        yo, yn = key.split("|")[1].split("→")
        bar = Counter(r["change_type"] for r in rows)
        total = len(rows)
        unchanged = bar.get("UNCHANGED", 0)
        uncertain = bar.get("UNCERTAIN", 0)
        changed = total - unchanged - uncertain

        # LDA 대상 문장: REMOVED → old_text, 나머지 → new_text
        def pick_text(r):
            return r["old_text"] if r["change_type"] == "REMOVED" else r["new_text"]

        changed_rows = [r for r in rows if r["change_type"] in CT_CHANGED and r.get("area") != "RISK"]
        texts = [pick_text(r) for r in changed_rows]

        lda_ok = False
        topics: list[list[str]] = []
        sbar: dict[str, list[int]] = {}
        detail: list[dict] = []

        if len(texts) >= 5:
            n_topics = 2 if len(texts) < 20 else 3 if len(texts) < 50 else 4
            lda_res = run_lda(texts, n_topics=n_topics)
            if lda_res:
                lda_ok = True
                topics = lda_res["topics"]
                n_topics = len(topics)
                doc_topics_arr = lda_res["doc_topics"]
                topic_assign = [max(range(n_topics), key=lambda i: p[i]) for p in doc_topics_arr]

                # sbar: {ct: [t0_count, ..., t4_count]}
                for ct in CT_CHANGED:
                    sbar[ct] = [0] * n_topics
                for idx, r in enumerate(changed_rows):
                    ct = r["change_type"]
                    ti = topic_assign[idx]
                    if ct in sbar:
                        sbar[ct][ti] += 1

                # detail: NEW / EXPANDED のみ
                from collections import defaultdict as ddict
                for ct in ["NEW", "EXPANDED"]:
                    by_topic: dict[int, list[str]] = ddict(list)
                    for idx, r in enumerate(changed_rows):
                        if r["change_type"] == ct:
                            by_topic[topic_assign[idx]].append(pick_text(r))
                    for ti, texts_in_topic in by_topic.items():
                        all_tokens = []
                        for t in texts_in_topic:
                            all_tokens.extend(tokenize_ko(t))
                        top_words = [w for w, _ in Counter(all_tokens).most_common(5)]
                        detail.append({"ct": ct, "ti": ti, "n": len(texts_in_topic), "words": top_words})

        pct = round(changed / (total - unchanged) * 100) if (total - unchanged) > 0 else 0

        results[key] = {
            "company": company,
            "yo": yo,
            "yn": yn,
            "total": total,
            "changed": changed,
            "pct": pct,
            "uncertain": uncertain,
            "bar": dict(bar),
            "lda_ok": lda_ok,
            "topics": topics,
            "topic_labels": [""] * len(topics),
            "sbar": sbar,
            "detail": detail,
        }

        print(f"  {key}: total={total}, changed={changed}, pct={pct}%, lda={lda_ok}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "results_v2" / "tfidf_masked_th0.3.csv"),
    )
    parser.add_argument("--output", default=str(ROOT / "analysis" / "lda_results.json"))
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise SystemExit(f"입력 파일 없음: {csv_path}")

    print(f"입력: {csv_path}")
    results = analyze(csv_path)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}  ({len(results)}개 키)")


if __name__ == "__main__":
    main()
