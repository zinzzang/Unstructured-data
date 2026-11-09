# UDA Term Project — Project Context

> Codex가 프로젝트의 목적과 범위를 이해하기 위한 간단한 문서. 구체적인 구현 지침은 별도 PRD를 따른다.

## 1. 주제와 목표

**Automated Detection of Year-over-Year Changes in Corporate Annual Reports by Industry**

OpenDART의 국내 기업 사업보고서를 자연어처리(NLP)로 비교해, 연도별 **실질적인 사업 변화**를 자동 탐지한다. 단순한 표현 차이와 의미 있는 변화를 구별하고, 기업별 결과를 산업별로 비교한다.

## 2. 분석 대상

- **산업:** 반도체 / 이차전지 / 바이오·제약
- **기간:** 2023–2025 사업연도
- **규모:** 산업별 3개 기업, 총 9개 기업 · 총 27개 사업보고서

| 산업 | 기업 |
|---|---|
| 반도체 | 삼성전자, SK하이닉스, 한미반도체 |
| 이차전지 | LG에너지솔루션, 삼성SDI, SK이노베이션 |
| 바이오·제약 | 삼성바이오로직스, 셀트리온, 유한양행 |

## 3. 데이터와 분석 범위

- **출처:** 금융감독원 OpenDART
- **수집:** 기업 고유번호(`corp_code`) → 사업보고서 접수번호(`rcept_no`) → 원문 공시 XML
- **분석 영역 (4개):**
  1. 사업의 개요 (Business Overview)
  2. 주요 제품 및 서비스 (Key Products and Services)
  3. 재무 정보 (Financial Information)
  4. 생산시설 및 투자 (Production Facilities and Investment)

## 4. 전체 흐름

1. OpenDART에서 사업보고서 수집
2. XML 파싱 → 분석 영역 추출 → 문장·문단 단위 전처리
3. **TF-IDF 코사인 유사도**와 **Sentence Embedding 코사인 유사도** 비교
4. 같은 기업의 연속 사업연도 간 주요 변화 탐지
5. 변화 유형 분류: **New / Removed·Discontinued / Expanded / Reduced / Modified**
6. 기업별·산업별 변화 결과 비교 및 시각화

일부 데이터는 사람이 Ground Truth로 라벨링하고, 두 방법의 변화 탐지 성능을 **Precision / Recall / F1-score**로 비교한다.

## 5. 기대 산출물

- 기업 / 연도 / 분석 영역 / 변화 유형 / 변경 전·후 근거 문장
- 산업별 공통 변화 영역과 변화 유형에 대한 비교 결과

## 6. 현재 작업 맥락

현재는 **Milestone A: 데이터 수집·전처리** 단계다. 세 명이 각각 한 산업의 기업 5개를 담당하되, 가능한 한 **공통 전처리 코드와 동일한 출력 형식**을 사용한다. 이후 Milestone B에서 TF-IDF 및 Sentence Embedding 분석에 활용한다.

> 이 문서는 프로젝트 맥락 요약이며, 파일 구조·구현 방식·검증 기준 등 실제 개발 요구사항은 PRD를 우선 참고한다.
