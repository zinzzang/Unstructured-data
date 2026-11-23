# Automated Detection of Year-over-Year Changes in Corporate Annual Reports by Industry

OpenDART 국내 기업 사업보고서를 NLP로 비교해 연도별 실질적 사업 변화를 자동 탐지하는 프로젝트.

## 분석 대상

| 산업 | 기업 |
|---|---|
| 반도체 | 삼성전자, SK하이닉스, 한미반도체 |
| 이차전지 | LG에너지솔루션, 삼성SDI, SK이노베이션 |
| 바이오·제약 | 삼성바이오로직스, 셀트리온, 유한양행 |

- **기간:** 2021–2025 사업연도 (5개년)
- **분석 영역:** 사업의 개요 · 주요 제품 및 서비스 · 재무 · 생산설비 및 투자

## 프로젝트 구조

```
scripts/
  collect_dart_reports.py   # OpenDART에서 사업보고서 수집 → DB 저장
  build_gt_llm.py           # OpenAI API로 문단 단위 Ground Truth 생성

src/
  database.py               # SQLite 스키마 및 헬퍼
  preprocess.py             # XML → 문장/문단 추출
  report_parser.py          # DART XML 파서
  section_classifier.py     # section_mapping.yaml 기반 영역 분류
  tfidf_model.py            # TF-IDF 코사인 유사도 + 숫자 마스킹
  embedding_model.py        # BGE-M3 임베딩 유사도 (캐시 포함)
  align.py                  # 문장 정렬 (top-k 매칭)
  change_classifier.py      # 변화 유형 분류 (LLM 기반)
  change_detector.py        # DB sentence_matches → detected_changes

config/
  settings.yaml             # 연도·임계값·마스킹 설정
  section_mapping.yaml      # 제목 → analysis_area 매핑

data/
  annual_reports.db         # SQLite: 9개 기업 × 5개년 = 45건, 34,952문장
  processed/preprocessing_v2/  # JSONL: 9개 기업 × 2021–2025
  results_v1/               # 비교용 베이스라인 (마스킹 미적용)
  results_v2/               # 현재 결과 (숫자 마스킹 적용)

run_v1.py                   # 메인 실행 엔트리포인트
sql/schema.sql              # DB 스키마 참고용
tests/                      # pytest 테스트
```

## 실행 방법

```bash
# 1. 사업보고서 수집 (DART_API_KEY 필요)
python scripts/collect_dart_reports.py --years 2021 2022 2023 2024 2025

# 2. TF-IDF 결과 생성
python run_v1.py tfidf --output-dir data/results_v2

# 3. 임베딩 결과 생성 (emb_cache.pkl 존재 시 재사용)
python run_v1.py embedding --output-dir data/results_v2
```

## 숫자 마스킹

TF-IDF와 임베딩 모두 `src/tfidf_model.mask_numbers()`를 사용. `config/settings.yaml`의 `number_masking.enabled`로 제어.

- **보존:** 금액(원/억원/조원 등), 비율(%), 복합 금액(30조 1,500억원)
- **마스킹(→NUM):** 연도, 문단 번호, 생산량 등 비금액 숫자

| 원문 | 마스킹 결과 |
|---|---|
| `2024년 사업보고서` | `NUM년 사업보고서` |
| `매출액 150억원` | `매출액 150억원` |
| `30조 1,500억원 투자` | `30조 1,500억원 투자` |
| `200만톤 생산` | `NUM만톤 생산` |

## 현재 결과 파일

| 파일 | 행 수 | 설명 |
|---|---|---|
| `results_v2/tfidf_masked_th0.3.csv` | 35,720 | 마스킹 TF-IDF, 임계값 0.3 |
| `results_v2/tfidf_masked_th0.6.csv` | 38,042 | 마스킹 TF-IDF, 임계값 0.6 |
| `results_v1/tfidf_th0.3.csv` | 36,068 | 베이스라인 (마스킹 미적용) |
| `results_v1/tfidf_th0.6.csv` | 39,480 | 베이스라인 (마스킹 미적용) |
| `results_v1/emb_bge-m3_th0.6.csv` | — | BGE-M3 임베딩, 임계값 0.6 |
| `results_v1/emb_KR-SBERT-V40K-klueNLI-augSTS_th0.6.csv` | — | KR-SBERT, 임계값 0.6 |

## 마스킹 효과 요약 (th0.3 기준)

- mean similarity: 0.498 → 0.541 (+0.043)
- ≥0.9 구간: 26.3% → 36.6% (+10.3%p) — 숫자만 다른 동일 구조 문장이 완전 일치에 가깝게 이동
- 유사도 상승 쌍: 18.8% (Δ>+0.05), 거의 모두 숫자 포함 텍스트
- 기업별 효과: 삼성SDI +0.090, LG에너지솔루션 +0.081 (재무 수치 많은 기업에서 효과 큼)

## Ground Truth 생성

```bash
# GPT 기반 GT 생성 (OPENAI_API_KEY 필요)
python scripts/build_gt_llm.py --companies 삼성전자 --pair 2024-2025
```

출력: `data/gt/{기업명}_gt_pairs.csv`, `_Confirmed_GT_Before_After_Paragraphs.xlsx`
