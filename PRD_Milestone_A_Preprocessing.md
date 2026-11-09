# PRD — Milestone A: OpenDART 사업보고서 공통 전처리 파이프라인

- **Version:** 1.1
- **Status:** Draft for team review
- **Project:** Automated Detection of Year-over-Year Changes in Corporate Annual Reports by Industry
- **Owner:** 3인 팀 공동; 공통 모듈별 담당자 지정

## 1. Problem

산업과 기업·연도에 따라 사업보고서의 제목, 본문 및 표 구조가 달라서, 일관된 전처리 없이 연도별 텍스트 유사도를 비교하면 파싱 차이가 실제 사업 변화로 잘못 탐지될 수 있다.

## 2. Users

- **주 사용자:** Milestone B에서 TF-IDF와 Sentence Embedding으로 문단 매칭 및 변화 탐지를 구현하는 팀원.
- **운영 사용자:** 산업별 3개 기업의 보고서를 수집·정제·검증하는 Milestone A 담당자 3명.

## 3. Feature

**하나의 설정 기반 공통 파이프라인으로 OpenDART 원본 사업보고서를 코드의 표준 `analysis_area` 값으로 분류한 문단 데이터셋으로 변환한다.**

### 입력

- 산업 3개(반도체, 이차전지, 바이오·제약), 각 3개 기업, 사업연도 2023–2025(총 27건)
- OpenDART 기업 식별자(`corp_code`), 사업보고서 접수번호(`rcept_no`), 원본 XML

### 처리

1. 보고서 식별·수집 및 원본 보존
2. XML의 제목 계층, 본문 문단, 표 정보 파싱
3. 제목 정규화 및 `config/section_mapping.yaml`의 표준 `analysis_area` 값으로 매핑:
   - `BUSINESS`: 사업의 개요
   - `PRODUCT`: 주요 제품
   - `PRODUCTION`: 원재료, 생산설비
   - `SALES`: 매출 및 수주, 매출
   - `RND_CONTRACT`: 주요계약, 연구개발
   - `RISK`: 위험관리, 파생
   - `OTHER`: 기타 참고
   - `UNMAPPED`: 위 키워드와 일치하지 않은 제목
4. 텍스트 정제 및 문단 단위 분리; 숫자·단위와 원문 출처 보존
5. 스키마 검증, 누락·오류·이상치 로깅

`RISK`, `OTHER`, `UNMAPPED`는 원문·추적성 보존을 위해 출력에는 유지한다. Milestone B의 유사도 계산과 산업 집계에 포함할지는 설정으로 결정한다.

### 출력

- `paragraphs.parquet` (필수): 기업·연도·영역·문단 ID·순서·텍스트·원본 위치
- `reports.csv` (필수): 보고서 메타데이터와 수집·처리 상태
- `validation_report.csv` (필수): 보고서/영역별 추출 성공·실패 현황
- `sentences.parquet` (권장): 문장 단위 근거 추출용

## 4. Acceptance Criteria — 3 Checks

### AC-1. 공통 파이프라인의 산업 간 재사용

**Given** 3개 산업에서 선정한 서로 다른 기업의 사업보고서 XML이 준비되었을 때,
**When** 동일한 전처리 실행 명령을 기업별 입력으로 실행하면,
**Then** 산업별 파서 복사본 없이 동일한 필드 구조의 `paragraphs.parquet` 결과를 생성한다.

**검증 방법:** 각 산업에서 최소 1건을 선택해 동일 명령 실행 → 결과 파일 스키마 비교.

### AC-2. 영역 매핑과 원문 추적

**Given** XML에서 분석 대상 제목과 본문을 찾을 수 있을 때,
**When** 전처리를 실행하면,
**Then** 문단은 지정된 `analysis_area` 중 하나로 분류되고, `paragraph_id`, `paragraph_order`, `report_id`, `source_location`이 기록되어 원문을 역추적할 수 있다. 매핑하지 못한 제목은 `UNMAPPED`로 보존한다.

**검증 방법:** 산업별 1건에서 영역별 표본 문단을 골라 원본 XML의 제목·본문·숫자와 대조한다. 실제 없는 영역은 누락으로 기록하고, 매핑되지 않은 제목은 `UNMAPPED` 목록으로 기록한다.

### AC-3. 실패와 누락을 명시적으로 보고

**Given** 원본 XML이 손상되었거나 특정 영역을 매핑할 수 없을 때,
**When** 파이프라인을 실행하면,
**Then** 해당 항목은 `validation_report.csv`에 실패/누락 사유로 기록되고, 정상적으로 추출된 다른 보고서는 유지되며, 실패 항목을 사업의 신규·삭제 변화로 간주하지 않는다.

**검증 방법:** 손상 XML 1건 및 제목 매핑 불가 샘플 1건을 넣고 결과와 오류 로그 확인.

## 5. Non-functional Requirements

- **재현성:** 동일 입력·설정·코드 버전은 동일한 문단 ID와 출력을 생성한다.
- **일관성:** 산업별 동일한 정제·문단 분리·표 처리 원칙을 적용한다.
- **추적성:** 원본 XML과 보고서 식별자, 문단의 원본 위치를 보존한다.
- **관찰 가능성:** 보고서별 성공·누락·오류 수 및 매핑되지 않은 제목을 확인할 수 있다.
- **확장성:** 신규 기업/연도는 설정 추가와 공통 실행으로 처리할 수 있어야 한다.

## 6. Team Responsibilities

| 담당 산업 | 공통 개발 책임 | 산업별 책임 |
|---|---|---|
| 반도체 | XML 파서·텍스트 정제 | 삼성전자, SK하이닉스, 한미반도체 |
| 이차전지 | 제목 정규화·영역 매핑 | LG에너지솔루션, 삼성SDI, SK이노베이션 |
| 바이오·제약 | 출력 스키마·검증 | 삼성바이오로직스, 셀트리온, 유한양행 |

세 명이 인터페이스와 문단 분리 기준을 공동 합의한 후 모듈을 병렬 개발한다. 기업별 특이사항은 가능하면 `company_overrides.yaml`에서 관리한다.

## 7. Out of Scope

- TF-IDF 벡터 생성 및 Sentence Embedding 생성
- 연도 간 문단 유사도 계산·매칭·변화 판정
- Precision/Recall/F1 측정 및 Ground Truth 라벨링 실행
- New / Removed / Expanded / Reduced / Modified 변화 유형 분류
- 산업별 변화 패턴 분석 및 최종 시각화
- 모든 재무표의 회계적 의미 해석 또는 수치 정확성에 대한 외부 감사 수준의 검증

## 8. Dependencies & Risks

| 위험 | 대응 |
|---|---|
| 기업·연도별 제목/태그 구조 차이 | 제목 계층 기반 매핑 + 별칭·예외 설정 + 매핑 실패 기록 |
| 표의 숫자·단위 손실 | 표 구조 보존 정책 및 원문 대조 샘플 검증 |
| 합병·분할·사업부문 변경 | 보고서 메타데이터에 기업구조 변화 주석 기록; 분석 범위 합의 |
| 보고서 누락·수집 실패 | 수집 상태 별도 관리; 데이터가 없는 것을 사업 변화로 간주하지 않음 |
| 산업별 전처리 불일치 | 공통 스키마·테스트·코드 리뷰 강제 |

## 9. Definition of Done

- [ ] 9개 기업 × 3개 사업연도(2023–2025), 총 27건의 수집·처리 상태를 목록화했다.
- [ ] 동일 파이프라인이 세 산업 샘플을 통과했다.
- [ ] 코드의 `analysis_area` 값별 추출 결과와 누락 여부를 검증했다.
- [ ] `UNMAPPED` 제목 목록을 검토하고, 필요한 키워드·우선순위를 `section_mapping.yaml`에 반영했다.
- [ ] 문단 ID가 중복되지 않고 원문 위치를 역추적할 수 있다.
- [ ] 오류·누락 검증 리포트가 생성된다.
- [ ] Milestone B에서 `paragraphs.parquet`를 직접 읽어 모델 입력으로 사용할 수 있다.

---

**Source alignment:** `UDA_Project_Context.md`의 OpenDART 데이터 소스, 분석 대상(3개 산업 × 3개 기업), 2023–2025 사업연도 및 TF-IDF/Sentence Embedding 비교 목표를 기준으로 작성했다. 영역의 실제 코드값과 키워드 우선순위는 `config/section_mapping.yaml`을 단일 기준으로 삼는다. 상세 제품 요구사항과 담당 배분은 팀 구현을 위한 제안 사항이다.
