# 사업보고서 텍스트 변화 분석 프로젝트

> **요약**: 국내 주요 기업 9곳의 사업보고서(2021–2025)를 TF-IDF로 연도별 변화 문장을 탐지하고, LDA 토픽 모델링으로 *무엇이* 변했는지를 설명하는 분석 파이프라인 + 인터랙티브 데모

---

## 1. 프로젝트 목적

기업 사업보고서는 매년 갱신되지만 "어디가 얼마나, 어떤 방향으로 바뀌었는지"를 사람이 직접 읽어 파악하기 어렵다.

이 프로젝트는 두 가지를 답하는 것이 목표다:

| 질문 | 방법 |
|------|------|
| **변했나?** (변화 감지) | TF-IDF 코사인 유사도 + 규칙 기반 변화 유형 분류 |
| **무엇이 변했나?** (변화 해석) | LDA 토픽 모델링 (변화 문장만 대상) |

---

## 2. 대상 데이터

| 구분 | 내용 |
|------|------|
| 출처 | 금융감독원 DART 전자공시시스템 |
| 기업 | 9개사 (반도체·배터리·바이오파마 3개 산업) |
| 기간 | 2021 → 2022 → 2023 → 2024 → 2025 (인접 연도 쌍 4개) |
| 분석 단위 | **문장 수준** (전처리 후 약 35,000개 문장 쌍) |

**대상 기업:**

| 산업 | 기업 |
|------|------|
| 반도체 | SK하이닉스, 삼성전자, 한미반도체 |
| 배터리 | LG에너지솔루션, SK이노베이션, 삼성SDI |
| 바이오파마 | 삼성바이오로직스, 셀트리온, 유한양행 |

---

## 3. 전체 분석 흐름

```
DART API
  │
  ▼
[1] 데이터 수집          scripts/collect_dart_reports.py
      └─ 원본 사업보고서 HTML → data/raw/zip/
  │
  ▼
[2] 전처리               src/preprocess.py, src/report_parser.py
      └─ 문단/문장 추출 + 품질 필터링
      └─ 출력: data/processed/preprocessing_v2/{산업}/paragraphs.jsonl
  │
  ▼
[3] 문장 정렬            src/align.py
      └─ 인접 연도(old–new) 문장쌍 생성
      └─ TF-IDF 코사인 유사도로 유사 문장 매칭 (threshold=0.3)
  │
  ▼
[4] 변화 분류            src/change_classifier.py
      └─ 규칙 기반 분류 → 7개 유형
      └─ 출력: data/results_v2/tfidf_masked_th0.3.csv  (35,720행)
  │
  ▼
[5] LDA 토픽 분석        scripts/run_lda_analysis.py
      └─ 변화 문장만 추출 → KiwiPy 형태소 분석 → LDA (5 토픽)
      └─ 출력: analysis/lda_results.json
  │
  ▼
[6] 시각화 데모          analysis/demo.html
      └─ 기업·연도 선택 → ① TF-IDF 분포 / ② 토픽별 변화 / ③ 변화별 토픽
```

---

## 4. 핵심 방법론 설명

### 4-1. Number Masking (숫자 마스킹)

사업보고서에는 "매출액 1,234억 원" 같이 연도마다 수치만 바뀌는 문장이 많다. 수치 자체를 TF-IDF에 넣으면 같은 내용도 "다른 문장"으로 판정된다.

→ 금액·비율 수치는 보존, 그 외 숫자(날짜·수량 등)는 `NUM` 토큰으로 치환 후 유사도 계산  
→ 코드: `src/tfidf_model.py` → `mask_numbers()`

### 4-2. 변화 유형 분류 (7가지)

| 유형 | 의미 |
|------|------|
| UNCHANGED | 사실상 동일 (유사도 ≥ 0.9) |
| NEW | 새로 등장한 문장 (old에 대응 없음) |
| EXPANDED | 같은 내용이 더 길어지거나 수치 증가 |
| REDUCED | 같은 내용이 줄어들거나 수치 감소 |
| MODIFIED | 내용이 달라진 문장 |
| REMOVED | 삭제된 문장 (new에 대응 없음) |
| UNCERTAIN | 규칙으로 판단 불가 (사람/LLM 검토 필요) |

→ 코드: `src/change_classifier.py` → `classify(old, new)`

### 4-3. LDA 토픽 모델링

**왜 LDA인가?**  
TF-IDF는 "얼마나 변했나"를 알려주지만, "어떤 주제가 변했나"는 말해주지 않는다.  
LDA(Latent Dirichlet Allocation)는 변화 문장의 집합에서 잠재 토픽을 추출해 "AI 반도체 관련 내용이 늘었다" 같은 해석을 가능하게 한다.

**적용 방식:**

```
변화 문장 (UNCHANGED·UNCERTAIN 제외)
  → KiwiPy 형태소 분석 (명사 NNG·NNP + 영어 SL, 길이 ≥ 2)
  → 금융/표 불용어 제거 (약 70개)
  → CountVectorizer (max_df=0.9, min_df=2, max_features=300)
  → sklearn LDA (n_topics=5, random_state=42, max_iter=50)
```

**텍스트 선택 규칙:**
- `REMOVED` 유형: `old_text` 사용 (삭제된 내용을 봐야 하므로)
- 그 외 변화 유형: `new_text` 사용

→ 코드: `scripts/run_lda_analysis.py`

---

## 5. 파일 구조

```
Unstructured-data/
├── PROJECT.md                        ← 이 문서
├── README.md                         ← 기본 개요
├── requirements.txt                  ← 의존성
├── run_v1.py                         ← TF-IDF 분석 실행 스크립트
│
├── config/
│   ├── settings.yaml                 ← 분석 파라미터 (임계값, 기업 목록 등)
│   └── section_mapping.yaml          ← 섹션 분류 매핑
│
├── src/                              ← 핵심 분석 모듈
│   ├── tfidf_model.py                ← TF-IDF 유사도 + 숫자 마스킹
│   ├── change_classifier.py          ← 규칙 기반 변화 유형 분류
│   ├── change_detector.py            ← 변화 감지 오케스트레이터
│   ├── align.py                      ← 문장 정렬
│   ├── preprocess.py                 ← 전처리
│   ├── report_parser.py              ← 사업보고서 HTML 파서
│   ├── section_classifier.py         ← 섹션 분류
│   ├── embedding_model.py            ← 임베딩 모델 (대안 방법)
│   └── database.py                   ← SQLite DB 헬퍼
│
├── scripts/
│   ├── collect_dart_reports.py       ← DART API에서 사업보고서 수집
│   ├── build_gt_llm.py               ← LLM 기반 ground truth 생성
│   └── run_lda_analysis.py           ← ★ LDA 분석 실행 스크립트
│
├── analysis/
│   ├── demo.html                     ← ★ 인터랙티브 데모 (브라우저에서 바로 열기)
│   └── lda_results.json              ← ★ LDA 분석 결과 (demo.html에 임베딩됨)
│
├── data/
│   ├── processed/preprocessing_v2/   ← 전처리된 문장 데이터
│   │   ├── battery/
│   │   ├── biopharma/
│   │   └── semiconductor/
│   └── results_v2/
│       ├── tfidf_masked_th0.3.csv    ← ★ TF-IDF 변화 분류 결과 (35,720행)
│       └── tfidf_masked_th0.6.csv    ← 임계값 0.6 버전
│
└── tests/                            ← 단위 테스트
```

★ = 핵심 파일

---

## 6. 데모 사용법

`analysis/demo.html`을 브라우저에서 그냥 열면 된다. 서버 불필요.

```bash
open analysis/demo.html          # macOS
start analysis/demo.html         # Windows
```

**UI 구성:**

| 섹션 | 내용 |
|------|------|
| 상단 선택바 | 기업명 드롭다운 + 연도 버튼 (21→22 / 22→23 / 23→24 / 24→25) |
| KPI 타일 4개 | 비교 문장 수 / 실질 변화 건수 / 변화율 / 주요 변화 유형 |
| ① TF-IDF 분포 | 변화 유형별 문장 수 가로 막대 |
| ② 토픽별 변화 유형 | LDA 토픽 각 행 → 어떤 변화 유형이 많은지 |
| LDA 토픽 설명 | 각 토픽의 대표 키워드 |
| ③ 변화 유형별 토픽 | 변화 유형 각 행 → 어떤 토픽에 속하는지 |

---

## 7. 분석 재실행 방법

### TF-IDF 분석 재실행
```bash
python run_v1.py tfidf --output-dir data/results_v2
```

### LDA 분석 재실행
```bash
python scripts/run_lda_analysis.py
# 또는 입력 파일 지정:
python scripts/run_lda_analysis.py --input data/results_v2/tfidf_masked_th0.3.csv
```

### 의존성 설치
```bash
pip install -r requirements.txt
pip install kiwipiepy      # 한국어 형태소 분석기 (별도 설치)
```

---

## 8. 주요 결과 요약

| 기업 | 연도 | 변화율 | 주요 특이사항 |
|------|------|--------|--------------|
| SK이노베이션 | 23→24 | **84%** | SK E&S 합병으로 대규모 신규 문장 추가 |
| 셀트리온 | 23→24 | **86%** | 셀트리온헬스케어 흡수합병으로 대규모 삭제 |
| SK하이닉스 | 22→23 | 42% | AI·HBM 관련 토픽 급부상 |
| 삼성바이오로직스 | 21→22 | 38% | CDO 사업 확장으로 신규 문장 많음 |
| LG에너지솔루션 | 21→22 | 32% | EV 배터리 시장 관련 내용 확장 |

---

## 9. 한계 및 추후 과제

### 현재 한계

1. **변화 유형 `UNCERTAIN` 비율이 높음**  
   규칙 기반 분류기가 애매한 케이스(표현 변경, 문장 재구성)를 모두 UNCERTAIN으로 분류.  
   → LLM 분류기 도입 시 개선 가능 (`change_classifier.py`의 `classify()` 함수를 교체하면 됨)

2. **TF-IDF의 한계: 의미적 유사도 미반영**  
   "매출 성장"과 "수익 개선"은 의미상 유사하지만 TF-IDF에선 다른 문장으로 처리됨.  
   `src/embedding_model.py`에 BGE-M3 임베딩 기반 대안이 구현되어 있으나,  
   속도·비용 문제로 본 분석은 TF-IDF를 사용함.

3. **LDA 토픽 자동 명명 불가**  
   LDA는 토픽을 "상위 키워드 목록"으로만 출력함.  
   현재 데모는 `개발/연구` 같이 키워드 2개를 이어붙여 임시 명칭을 붙임.

### ⚠️ 추후 필수 과제: LDA 토픽 수동 정제

**현재 상태**: 36개 기업·연도 쌍 × 5개 토픽 = **최대 180개 토픽**이 자동 생성됨.

**문제**: 아래와 같은 노이즈 토픽이 섞여 있어 의미 해석이 어려움.
- 금융 표 아티팩트: `col`, `USD`, `JPY`, `EUR` 등이 토픽을 지배
- 행정 용어: `찬성`, `반대`, `이사회` 등 내용과 무관한 단어
- 범용 경영 용어: `개발`, `연구`, `시장` 등 거의 모든 토픽에 등장

**해야 할 일**:

```
analysis/lda_results.json 파일에서
각 기업·연도쌍의 "topics" 필드를 직접 열어서

1. 각 토픽에 사람이 읽을 수 있는 이름 붙이기
   예) ["ai", "반도체", "hbm", "서버", "메모리"] → "AI·HBM 수요"
   예) ["배터리", "에너지", "ev", "소형", "application"] → "EV 배터리 사업"

2. 의미 없는 토픽(금융 표 아티팩트만 있는 토픽) 제거 또는 병합

3. 정제된 토픽명을 demo.html에 반영
   (현재는 JS에서 kws[0]/kws[1]을 자동 조합해 임시 표시)
```

이 작업은 **각 산업 도메인 지식이 있는 사람이 직접 검토**해야 한다.  
특히 합병이 일어난 케이스(SK이노베이션 23→24, 셀트리온 23→24)는 토픽 해석에 주의 필요.

---

## 10. 참고 사항

- **데이터 크기**: `data/raw/zip/` (약 90개 원본 zip 파일, 각 10–50MB)는 `.gitignore`로 제외됨. 재수집 필요 시 `scripts/collect_dart_reports.py` 실행.
- **SQLite DB**: `data/annual_reports.db`는 중간 저장용으로 `.gitignore`에서 제외됨. `run_v1.py` 실행 시 자동 생성.
- **환경**: `.env`에 DART API 키 필요 (`DART_API_KEY=...`). `.gitignore`로 제외됨.
- **Python 버전**: 3.11 이상 권장.
