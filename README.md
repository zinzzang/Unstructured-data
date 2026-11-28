# 사업보고서 연도별 변화 자동 탐지 (TF-IDF + LDA)

> OpenDART 국내 기업 사업보고서에서 TF-IDF로 연도별 변화 문장을 탐지하고, LDA 토픽 모델링으로 변화 주제를 해석하는 프로젝트.

**→ 최종 발표자료: [UDA_final_presentation.pdf](UDA_final_presentation.pdf)**  
**→ 상세 설명: [PROJECT.md](PROJECT.md)**  
**→ 인터랙티브 데모: [analysis/demo.html](analysis/demo.html)** (브라우저에서 바로 열기)

---

## 분석 대상

| 산업 | 기업 |
|---|---|
| 반도체 | 삼성전자, SK하이닉스, 한미반도체 |
| 이차전지 | LG에너지솔루션, 삼성SDI, SK이노베이션 |
| 바이오·제약 | 삼성바이오로직스, 셀트리온, 유한양행 |

- **기간:** 2021–2025 사업연도 (5개년, 인접 연도 쌍 4개: 21→22, 22→23, 23→24, 24→25)
- **분석 단위:** 문장 수준. 최종 결과 22,706개 문장 쌍 (`tfidf_masked_th0.3.csv`)

---

## 전체 파이프라인

```
PHASE 1 · 변화 탐지
[01] 데이터 수집        scripts/collect_dart_reports.py
       OpenDART API → 사업보고서 XML → data/annual_reports.db, data/raw/zip/
[02] 전처리             src/preprocess_structure.py + src/report_parser.py
       II장 → 문단 → 문장, 7개 분석 영역 매핑 → data/processed/preprocessing_v2/
[03] TF-IDF 매칭        run_v1.py + src/tfidf_model.py
       숫자 마스킹(NUM) → 문자 2–5gram TF-IDF → 같은 영역 내 코사인, cos ≥ 0.3이면 매칭
[04] 변화 분류          run_v1.py + src/change_classifier.py
       원문 수치 기반 규칙 → 7개 유형 → data/results_v2/tfidf_masked_th0.3.csv

  ↓ 변화 문장만 사용: 5개 유형, 22,706쌍 중 7,262쌍 (UNCHANGED·UNCERTAIN 제외)

PHASE 2 · 변화 해석
[05–08] LDA 토픽 모델   scripts/run_lda_analysis.py
       KiwiPy 명사(NNG·NNP + SL, 2자 이상) → 불용어 제거, RISK 영역 제외
       → CountVectorizer(max_df 0.9, ≤300 단어) → 기업 × 연도쌍별 LDA
       → analysis/lda_results.json
       대표 변화 문장쌍: scripts/extract_representative_changes.py → demo.html

시각화
       analysis/demo.html (기업·연도 선택 → 변화 유형 분포 / 토픽 × 변화 유형)
       scripts/plots/ (발표자료 그림)
```

---

## 파일 구조

```
├── run_v1.py                              # [03–04] TF-IDF 매칭 + 변화 분류 실행
├── requirements.txt
├── UDA_final_presentation.pdf             # 최종 발표자료
│
├── scripts/
│   ├── collect_dart_reports.py            # [01] OpenDART에서 사업보고서 수집
│   ├── run_lda_analysis.py                # [05–08] 기업·연도쌍별 LDA 토픽 분석
│   ├── extract_representative_changes.py  # 토픽별 대표 변화 문장쌍 → demo.html
│   └── plots/                             # 발표자료 그림 (아래 표 참고)
│
├── src/
│   ├── report_parser.py                   # DART XML에서 II장 추출
│   ├── preprocess_structure.py            # [02] 문단·문장 레코드 생성 (JSONL)
│   ├── tfidf_model.py                     # [03] 숫자 마스킹 + TF-IDF 코사인 유사도
│   ├── change_classifier.py               # [04] 규칙 기반 변화 유형 분류 (7가지)
│   └── database.py                        # SQLite 헬퍼 (수집용)
│
├── config/
│   ├── settings.yaml                      # 수집 연도, 매칭 임계값, TF-IDF 설정
│   └── section_mapping.yaml               # 소제목 → 7개 분석 영역 매핑
│
├── sql/schema.sql                         # 수집 DB 스키마
│
├── data/
│   ├── raw/                               # corp_code.json, 원본 ZIP
│   ├── processed/preprocessing_v2/        # 전처리 완료 데이터 (산업별 paragraphs.jsonl)
│   ├── results_v2/tfidf_masked_th0.3.csv  # TF-IDF 매칭 + 변화 분류 결과 (22,706행)
│   └── results_representative_changes/    # 대표 변화 문장쌍 (산업/기업/연도쌍/유형)
│
├── analysis/
│   ├── demo.html                          # 인터랙티브 데모 (서버 불필요)
│   └── lda_results.json                   # LDA 결과 (9개 기업 × 4개 연도쌍 = 36개)
│
├── figures/                               # 발표자료 그림 (png/svg) + 집계 csv
└── tests/
```

> `data/annual_reports.db`(수집한 원문 XML이 든 SQLite)는 용량 문제로 `.gitignore`에서 제외.

### 발표자료 그림

| 슬라이드 | 그림 | 스크립트 (`scripts/plots/`) |
|---|---|---|
| 05 | Term-Document Matrix (SK하이닉스), 기업별 글자 수 | `plot_dataset_overview.py` |
| 06 | 기업별 전처리 문장 수 · TF-IDF 문장 쌍 · 변화 쌍 | `plot_company_sentence_counts.py` |
| 10 | 기업 × 연도쌍 변화율 히트맵 | `plot_change_concentration.py` |
| 10 | 산업별 변화 흐름 + 수출액 (3개) | `plot_industry_change_flow.py` |
| 12 | 유한양행 사례 (변화 유형 · 주가 · 영업이익) | `plot_company_finance.py` |

---

## 실행 방법

```bash
pip install -r requirements.txt

# [01] 사업보고서 수집 (DART_API_KEY 필요: 환경변수 또는 .env)
python scripts/collect_dart_reports.py

# [02] 전처리 결과는 data/processed/preprocessing_v2/에 포함되어 있음
#      (함수: src/preprocess_structure.py build_paragraph_records / write_jsonl)

# [03–04] TF-IDF 매칭 + 변화 분류 → data/results_v2/tfidf_masked_th0.3.csv
#         같은 연도쌍이 이미 있으면 건너뛰므로, 다시 만들려면 기존 CSV를 지우거나 --output-dir 지정
python run_v1.py

# [05–08] LDA 토픽 분석 → analysis/lda_results.json
python scripts/run_lda_analysis.py

# 대표 변화 문장쌍 (단건 / 전체 생성 후 demo.html에 번들)
python scripts/extract_representative_changes.py --company SK하이닉스 --year-old 2022 --year-new 2023 --change-type NEW
python scripts/extract_representative_changes.py --all --update-demo

# 발표자료 그림 → figures/
python scripts/plots/plot_dataset_overview.py      # 다른 그림도 같은 방식

# 테스트
python -m pytest
```

---

## 핵심 기법: 숫자 마스킹

사업보고서에는 "매출액 1,234억 원" 같이 연도마다 수치만 바뀌는 문장이 많다. 수치를 그대로 TF-IDF에 넣으면 같은 내용도 다른 문장으로 판정된다.

→ **금액·비율은 보존**, 그 외 숫자(연도·문단번호 등)는 `NUM` 토큰으로 치환 후 유사도 계산

| 원문 | 마스킹 결과 | 비고 |
|---|---|---|
| `2024년 사업보고서` | `NUM년 사업보고서` | 연도 → 마스킹 |
| `매출액 150억원` | `매출액 150억원` | 금액 → 보존 |
| `200만톤 생산` | `NUM만톤 생산` | 수량 → 마스킹\* |

> \* 마스킹은 **TF-IDF 유사도 계산에만** 적용. `200만톤`→`250만톤` 같은 수치 변화는 원문(`old_text`/`new_text`)을 보는 `change_classifier`가 별도로 EXPANDED/REDUCED로 분류하므로 정보는 손실되지 않는다.

---

## 변화 유형 분류 (7가지)

| 유형 | 의미 | 건수 (22,706쌍 중) |
|---|---|---|
| UNCHANGED | 사실상 동일 | 9,790 |
| NEW | 새로 추가된 문장 | 1,887 |
| EXPANDED | 같은 내용이 증가/확장 | 1,965 |
| REDUCED | 같은 내용이 감소/축소 | 1,048 |
| MODIFIED | 내용 변경 | 712 |
| REMOVED | 삭제된 문장 | 1,650 |
| UNCERTAIN | 규칙으로 판단 불가 | 5,654 |

LDA에는 NEW·EXPANDED·REDUCED·MODIFIED·REMOVED 5개 유형(7,262쌍)만 쓴다. REMOVED는 `old_text`, 나머지는 `new_text`를 입력으로 하며, 토픽 수는 문장 수에 따라 20개 미만 2개, 50개 미만 3개, 그 이상 4개.

---

## 주요 발견

변화율 = 5개 변화 유형 쌍 ÷ 해당 기업·연도쌍 전체 문장 쌍

| 기업 | 연도 | 변화율 | 맥락 |
|---|---|---|---|
| SK하이닉스 | 22→23 | 42% | AI·HBM 수요 |
| 셀트리온 | 23→24 | 42% | 셀트리온헬스케어 흡수합병 |
| SK이노베이션 | 23→24 | 34% | SK E&S 합병 |
