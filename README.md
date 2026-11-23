# 사업보고서 연도별 변화 자동 탐지 (TF-IDF + LDA)

> OpenDART 국내 기업 사업보고서를 TF-IDF로 연도별 변화 문장을 탐지하고, LDA 토픽 모델링으로 변화 주제를 분류하는 프로젝트.

**→ 상세 설명: [PROJECT.md](PROJECT.md)**  
**→ 인터랙티브 데모: [analysis/demo.html](analysis/demo.html)** (브라우저에서 바로 열기)

---

## 분석 대상

| 산업 | 기업 |
|---|---|
| 반도체 | 삼성전자, SK하이닉스, 한미반도체 |
| 이차전지 | LG에너지솔루션, 삼성SDI, SK이노베이션 |
| 바이오·제약 | 삼성바이오로직스, 셀트리온, 유한양행 |

- **기간:** 2021–2025 사업연도 (5개년, 인접 연도 쌍 4개)
- **분석 단위:** 문장 수준 (약 35,000개 문장 쌍)

---

## 전체 파이프라인

```
[1] DART 수집          scripts/collect_dart_reports.py
      OpenDART API → 사업보고서 XML 다운로드 → data/raw/zip/

[2] 전처리             src/preprocess.py + src/report_parser.py
      XML → 문장/문단 추출 → data/processed/preprocessing_v2/

[3] TF-IDF 분석        run_v1.py
      인접 연도 문장 정렬 → 변화 유형 분류 → data/results_v2/

[4] LDA 분석           scripts/run_lda_analysis.py
      변화 문장 → KiwiPy 형태소 분석 → LDA 5토픽 → analysis/lda_results.json

[5] 시각화             analysis/demo.html
      기업·연도 선택 → 변화 분포 / 토픽별 변화 차트
```

---

## 파일 구조

```
├── run_v1.py                              # [3] TF-IDF 실행 진입점
├── requirements.txt
│
├── scripts/
│   ├── collect_dart_reports.py            # [1] OpenDART에서 사업보고서 수집
│   ├── run_lda_analysis.py                # [4] LDA 토픽 분석
│   └── build_gt_llm.py                    # Ground Truth 생성 (LLM)
│
├── src/
│   ├── tfidf_model.py                     # TF-IDF 코사인 유사도 + 숫자 마스킹
│   ├── change_classifier.py               # 규칙 기반 변화 유형 분류 (7가지)
│   ├── align.py                           # 문장 정렬 (top-k 매칭)
│   ├── change_detector.py                 # 변화 감지 오케스트레이터
│   ├── preprocess.py                      # XML → 문장/문단 추출
│   ├── report_parser.py                   # DART XML 파서
│   ├── section_classifier.py              # 섹션 분류
│   ├── embedding_model.py                 # BGE-M3 임베딩 (대안 방법)
│   └── database.py                        # SQLite 헬퍼
│
├── config/
│   ├── settings.yaml                      # 분석 파라미터
│   └── section_mapping.yaml               # 섹션 분류 매핑
│
├── data/
│   ├── processed/preprocessing_v2/        # 전처리 완료 데이터 (JSONL)
│   │   ├── battery/
│   │   ├── biopharma/
│   │   └── semiconductor/
│   └── results_v2/
│       ├── tfidf_masked_th0.3.csv         # TF-IDF 결과 (임계값 0.3, 35,720행)
│       └── tfidf_masked_th0.6.csv         # TF-IDF 결과 (임계값 0.6)
│
├── analysis/
│   ├── demo.html                          # 인터랙티브 데모 (서버 불필요)
│   └── lda_results.json                   # LDA 분석 결과 (36개 기업·연도쌍)
│
└── tests/
```

> `data/raw/` (원본 ZIP), `data/annual_reports.db` (SQLite)는 용량 문제로 `.gitignore` 제외.

---

## 실행 방법

```bash
pip install -r requirements.txt
pip install kiwipiepy   # 한국어 형태소 분석기

# [1] 사업보고서 수집 (DART_API_KEY 필요, .env에 설정)
python scripts/collect_dart_reports.py

# [3] TF-IDF 분석 (전처리 데이터가 이미 data/processed/에 있음)
python run_v1.py tfidf --output-dir data/results_v2

# [4] LDA 분석
python scripts/run_lda_analysis.py

# [5] 데모 열기
open analysis/demo.html
```

---

## 핵심 기법: 숫자 마스킹

사업보고서에는 "매출액 1,234억 원" 같이 연도마다 수치만 바뀌는 문장이 많다. 수치를 그대로 TF-IDF에 넣으면 같은 내용도 다른 문장으로 판정된다.

→ **금액·비율은 보존**, 그 외 숫자(연도·문단번호 등)는 `NUM` 토큰으로 치환 후 유사도 계산

| 원문 | 마스킹 결과 |
|---|---|
| `2024년 사업보고서` | `NUM년 사업보고서` |
| `매출액 150억원` | `매출액 150억원` ← 보존 |
| `200만톤 생산` | `NUM만톤 생산` |

---

## 변화 유형 분류 (7가지)

| 유형 | 의미 |
|---|---|
| UNCHANGED | 사실상 동일 |
| NEW | 새로 추가된 문장 |
| EXPANDED | 같은 내용이 증가/확장 |
| REDUCED | 같은 내용이 감소/축소 |
| MODIFIED | 내용 변경 |
| REMOVED | 삭제된 문장 |
| UNCERTAIN | 규칙으로 판단 불가 |

---

## 주요 발견

| 기업 | 연도 | 변화율 | 원인 |
|---|---|---|---|
| SK이노베이션 | 23→24 | 84% | SK E&S 합병 |
| 셀트리온 | 23→24 | 86% | 셀트리온헬스케어 흡수합병 |
| SK하이닉스 | 22→23 | 42% | AI·HBM 관련 토픽 급부상 |
