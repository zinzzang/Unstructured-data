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
STOP = set('''
경쟁력 당사 회사 기업 사업 관련 경우 영향 통해 기준 분야 전망 예상 지속 증가 감소 확대 축소 성장 수치 비중 기록 발생
적용 제공 진행 운영 수행 당기 전기 금액 해당 전년 올해 지난해 다음 전체 부분 상황 정도 시기 수준 현재 미래 측면
내용 위한 대응 확보 추진 강화 개선 위하여 최근 중심 가운데 연도 이후 또한 당시 내외 국내 국외 각종 이번 과거 처음
시간 당사자 우리 모든 각각 상기 아래 관계 전반 발표 대한 대하여 이전 기간 시점 예정 계획 노력 다양 지속적
시장 수요 제품 기술 산업 해외 세계 주요 대비 담당 출처 자료 참조 가트너 gartner idc omdia trendforce
억원 백만원 천원 달러 비율 규모 실적 사항 현황 부문 사업부 사업부문 연결 종속 자회사 법인 지배
공시 작성 기준일 서류 포함 이상 이하 가능 변화 활용 기반 바탕 보유 특성 방법 기대 사용 필요 일부 다수 신규
레미케이드 아바스틴 허셉틴 휴미라 엔브렐 루센티스 솔리리스 악템라 스텔라라 프롤리아 아일리아 졸레어 리툭산 키트루다
램시마 트룩시마 허쥬마 유플라이마 베그젤마 짐펜트라 스테키마 렉라자 옴리클로 앱토즈마 에피스클리 오퓨비즈 하드리마
종속 손익 포괄 세후 세전 자본 미만 초과 차입금 판매 지분
이자 거래 상품 대상 매출 투자 매입 현금 변동 결산 계정 잔액
원화 외화 장부 원가 취득 처분 상각 손상
환율 통화 파생 위험 관리 노출 측정 평가
주석 근거 정책 재무 별도 주요 고려
소계 활동 실행 회피 누계 저장 자산
기말 요구 산출 가격 처리 향상 효율 하락 용도 정보 결과 직접 경로
제1기 제2기 제3기 제4기 흐름 평균 조건 완료 취소
한편 대형 고객 경험 성능 인식 보고 범위 합리
임원 인원 단계 한국 제출 회원권 구분 품목
건물 토지 구축물 건설 기계 장치 차량 운반 공구 사용권자산
유형 회계 소재지 비품 기구 과제명 연구과제명 kce col
'''.split())

STOP_LDA = set('''
구체 구체화 명칭 항목 내역 세목 구분 분류 성격
반영 경과 효과 중요 중요성 고도 제고 보완
기존 자체 별개 소정 소위 물론 실제 실질
최초 최종 시작 종료 개시 완료 해당사항
향후 연간 월간 상반기 하반기 분기 반기
중장기 단기 장기 일자 날짜 일수 누적
당기말 전기말 기말 기초 연초 연말
잔여 증감 가액 투입액 시설가액
기재사항 세부사항 참고사항
조사 조사기관 기관 업체 사업자
수취 적기 달성 극복 최소 최대
총괄 구성 업무 경력 명예
학사 석사 박사 팀장 본부
솔루션 application level high
de the and of for
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
첫째 둘째 셋째 직원 임직원
'''.split())

COMPANY_NAMES_EXTRA = set('''
lg에너지솔루션 엘지에너지솔루션
sk이노베이션 에스케이이노베이션
sk하이닉스 에스케이하이닉스
삼성sdi 삼성에스디아이
삼성바이오로직스 삼성바이오에피스
삼성디스플레이 sdc
루브리컨츠 enmove
sk에너지 sk루브리컨츠 sk엔무브
nextstar nextstarenergy
삼성전자 셀트리온 유한양행 한미반도체
samsung hynix hanmi celltrion
'''.split())

UNIT_CODES = set('''
gb tb mb gbps co ltd inc corp mm nm um kg km
kwh mwh gwh twh mw gw kw ppm usd krw eur
tj gj mj kj ton tons g mg ml l
hz khz mhz ghz bps mbps tbps
jpy cny rmb hkd pct
'''.split())

LEGAL_CODES = set('s.a s.l pte limited corporation company incorporated llc'.split())

STOP = {w.lower().strip() for w in STOP | STOP_LDA | STOP_EXTRA | COMPANY_NAMES_EXTRA | UNIT_CODES | LEGAL_CODES}


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
            form = token.form.strip(".,;:()[]") if token.tag == "SL" else token.form
            if not form or len(form) < 2:
                continue
            if form.lower() not in STOP and form not in STOP:
                tokens.append(form.lower() if token.tag == "SL" else form)
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
        n_components=n_topics, random_state=42, max_iter=50,
        topic_word_prior=0.1,   # 낮을수록 토픽 간 단어 분포 더 뚜렷하게 분리
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

        # 토픽별 고유 키워드 (다른 토픽에 없는 단어) → 자동 label 후보
        def make_labels(topics):
            labels = []
            for i, t in enumerate(topics):
                others = {w for j, t2 in enumerate(topics) if j != i for w in t2}
                distinct = [w for w in t if w not in others]
                # 고유 키워드 2개로 label, 없으면 상위 키워드로 대체
                kw = distinct[:2] if len(distinct) >= 2 else (distinct + [w for w in t if w not in distinct])[:2]
                labels.append("·".join(kw))
            # 중복 label 발생 시 번호 붙이기
            seen: dict[str, int] = {}
            result = []
            for lb in labels:
                if lb in seen:
                    seen[lb] += 1
                    result.append(f"{lb}({seen[lb]})")
                else:
                    seen[lb] = 0
                    result.append(lb)
            return result

        topic_labels = make_labels(topics) if topics else []

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
            "topic_labels": topic_labels,
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
