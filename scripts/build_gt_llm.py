"""사업보고서 연도별 변화 탐지 Ground Truth(GT)를 OpenAI API로 자동 생성한다.

입력: data/processed/preprocessing_v1/<산업>/paragraphs.jsonl
출력: data/gt/{기업명}_Confirmed_GT_Before_After_Paragraphs.xlsx  (+ 판정 전체 기록 jsonl)

흐름
  1. 기업별로 인접 연도쌍(2023→2024, 2024→2025 ...)을 만든다. 없는 연도는 만들지 않는다.
  2. 두 연도의 문단을 같은 목차(title_path) 단위 그룹으로 묶는다. 그룹이 크면 하위 목차, 그래도 크면 순서대로 나눈다.
  3. 그룹마다 모델이 문단 매칭 + 변화 판정을 JSON으로 돌려준다. 모델은 paragraph_id와 판정만 쓰고,
     원문은 코드가 paragraphs.jsonl에서 그대로 채운다 (원문 요약·수정 방지).
  4. 누락·중복 매칭·형식 오류를 점검하고, 확정 GT / 검토 필요 / 처리 현황을 엑셀로 저장한다.

API 키: 환경변수 OPENAI_API_KEY, 또는 프로젝트 .env 의 OPENAI_API_KEY / OPEN_API_KEY
모델 응답은 data/experiments/gt_llm_cache/ 에 캐시되어, 다시 실행하면 API를 다시 부르지 않는다.

사용 예
  python scripts/build_gt_llm.py --companies SK하이닉스 --dry-run   # 그룹·분량만 확인 (API 호출 없음)
  python scripts/build_gt_llm.py --companies SK하이닉스              # 한 기업 실행
  python scripts/build_gt_llm.py                                     # 9개 기업 전체
  python scripts/build_gt_llm.py --model gpt-5 --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed" / "preprocessing_v1"
OUT_DIR = ROOT / "data" / "gt"
CACHE_DIR = ROOT / "data" / "experiments" / "gt_llm_cache"
INDUSTRIES = ["semiconductor", "battery", "biopharma"]

PROMPT_VERSION = "v1"
GROUP_CHAR_LIMIT = 40_000   # 한 번에 보내는 두 연도 문단 합계 글자수 상한
GROUP_PARA_LIMIT = 80       # 한 번에 보내는 두 연도 문단 수 합계 상한 (응답 길이 제한 대비)
MAX_PARA_CHARS = 15_000     # 이보다 긴 문단은 전처리 이상(표 통째 등)으로 보고 모델에 보내지 않음 -> 검토 필요
EXCEL_CELL_LIMIT = 32_000

AREAS = ["사업 내용", "주요 제품 및 서비스", "재무", "생산설비 및 투자", "해당 없음"]
CHANGE_TYPES = ["신규", "삭제/종료", "확대", "축소", "변경", "없음"]
JUDGEMENTS = ["변화", "변화 없음", "검토 필요", "대상 외"]

SYSTEM_PROMPT = """너는 한국 기업 사업보고서의 연도별 변화를 분석하고, 문단 단위 Ground Truth(GT)를 구축하는 전문 데이터 어노테이터다.
같은 기업의 인접 연도 사업보고서에서 같은 목차에 속한 문단 묶음이 주어진다. PREV는 이전 연도, NEXT는 다음 연도다.

## 문단 매칭
1. 동일한 사업 내용, 제품, 재무 항목, 생산설비·투자 내용을 설명하는 문단끼리 매칭한다.
2. 위치나 순서가 바뀌어도 내용이 대응하면 매칭한다.
3. 서로 다른 주제의 문단은 표현이 비슷하다는 이유만으로 매칭하지 않는다.
4. 대응 문단이 없는 문단은 한쪽 id만 채운다. 대응 문단이 없다는 이유만으로 신규 또는 삭제/종료로 판정하지 않는다.
   원문에 신규 착수·출시·취득 또는 종료·중단·철수가 명시된 경우에만 신규/삭제·종료로 판정하고, 아니면 검토 필요로 둔다.
5. source_type이 paragraph인 문단은 서술 내용을 비교한다.
   table_row는 같은 표(label)·같은 항목의 행끼리 대응시키고, 같은 기준일·단위·범위의 값끼리 비교한다.
   표의 열 연도가 한 해씩 밀린 것(예: 2023년 열 → 2024년 열)은 변화가 아니다. 같은 연도 값이 두 보고서에서 다르면 정정 여부를 근거로 적는다.

## 변화 탐지 영역 (이 4개만 GT 대상)
- 사업 내용: 사업 구조, 산업 및 시장 환경, 주요 사업의 시작·종료·확대·축소, 사업 내용과 조직 구조의 실질적 변경
- 주요 제품 및 서비스: 주요 제품·서비스의 추가·종료, 제품군 변화, 주요 제품·서비스 비중 변화
- 재무: 주요 재무 항목 변화, 회계 처리, 재무 위험 및 주요 재무 구조 변화
- 생산설비 및 투자: 신규 생산설비, 생산능력 변화, 투자 규모 및 투자 계획 변경
위 영역 밖의 변화는 judgement를 "대상 외"로 한다.

## 변화 유형 (judgement가 "변화"일 때만, 하나만)
- 신규: 이전 연도에 없던 사업, 제품, 설비, 투자 등이 새롭게 등장했음이 원문에서 확인됨
- 삭제/종료: 기존 사업, 제품, 설비 등이 실제로 종료·중단·철수되었음이 확인됨
- 확대: 규모, 매출, 비중, 생산능력, 투자 등이 증가함
- 축소: 규모, 매출, 비중, 생산능력, 투자 등이 감소함
- 변경: 위 유형에 해당하지 않는 사업 내용, 조건, 구조 또는 회계 관련 변경

## 라벨링 규칙
1. 보고서 원문에 근거한 변화만 "변화"로 확정한다.
2. 전년도 수치와 당해 연도 수치가 동일한 기준일·단위·범위를 사용하는지 확인한다.
3. 단순 문장 표현의 변경은 실제 변화가 아니다 ("변화 없음").
4. 매년 반복되는 연도 표기 변경만으로는 변화가 아니다.
5. 연구개발 실적 목록에서 사라진 제품을 단종으로 추정하지 않는다.
6. 하나의 문단에 독립적인 변화가 여러 개 있으면 변화별로 결과를 나누고, 같은 id를 반복해 쓴다.
7. 확실한 판단이 불가능하면 "검토 필요"로 하고 review_reason에 이유를 적는다. 근거가 부족한 항목을 억지로 변화 유형에 넣지 않는다.
8. 입력된 모든 PREV·NEXT 문단 id가 결과에 최소 한 번씩 나와야 한다. 내용이 같으면 "변화 없음"으로라도 반드시 포함한다.

## 출력 필드
- prev_ids / next_ids: 매칭된 문단 id 목록 (주어진 id만 사용. 없으면 빈 목록)
- judgement: 변화 / 변화 없음 / 검토 필요 / 대상 외
- area: 4개 영역 중 하나. 해당 없으면 "해당 없음"
- change_type: judgement가 "변화"면 5개 유형 중 하나, 아니면 "없음"
- description: 무엇이 어떻게 바뀌었는지 한두 문장 (변화가 아니면 빈 문자열 가능)
- evidence: 판정 근거. 원문 표현이나 수치를 인용해 이전→이후를 구체적으로 적는다
- review_reason: 검토 필요일 때 이유, 아니면 빈 문자열"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "gt_judgements",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["results"],
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["prev_ids", "next_ids", "judgement", "area", "change_type",
                                     "description", "evidence", "review_reason"],
                        "properties": {
                            "prev_ids": {"type": "array", "items": {"type": "string"}},
                            "next_ids": {"type": "array", "items": {"type": "string"}},
                            "judgement": {"type": "string", "enum": JUDGEMENTS},
                            "area": {"type": "string", "enum": AREAS},
                            "change_type": {"type": "string", "enum": CHANGE_TYPES},
                            "description": {"type": "string"},
                            "evidence": {"type": "string"},
                            "review_reason": {"type": "string"},
                        },
                    },
                }
            },
        },
    },
}


# ---------------------------------------------------------------- 입력

def load_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()
    for env_path in (ROOT / ".env", ROOT.parent / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.partition("=")
            name = name.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if sep and name in ("OPENAI_API_KEY", "OPEN_API_KEY") and value:
                return value
    raise SystemExit("OpenAI API 키가 없습니다. .env 에 OPENAI_API_KEY=sk-... (또는 OPEN_API_KEY=...) 를 넣고 저장해 주세요.")


def load_paragraphs(industries):
    rows = []
    for ind in industries:
        path = DATA_DIR / ind / "paragraphs.jsonl"
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    by_company = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_company[r["company"]][int(r["business_year"])].append(r)
    for years in by_company.values():
        for paras in years.values():
            paras.sort(key=lambda r: r["paragraph_order"])
    return by_company


def year_pairs(years):
    ys = sorted(years)
    return [(a, b) for a, b in zip(ys, ys[1:]) if b == a + 1]  # 인접 연도만


# ---------------------------------------------------------------- 그룹 구성

TITLE_PREFIX = re.compile(r"^\s*(\(?\d+\)|\d+\.|[가-힣]\.|[IVX]+\.|[①-⑳])\s*")


def norm_title(t):
    return re.sub(r"\s+", "", TITLE_PREFIX.sub("", t or ""))


def title_key(r, depth):
    path = r.get("title_path") or []
    return tuple(norm_title(t) for t in path[1:depth + 1])  # path[0]은 'II. 사업의 내용'


def chars(rows):
    return sum(len(r["source_text"]) for r in rows)


def build_groups(prev, nxt, depth=1, label=""):
    """목차 단위로 묶고, 크면 하위 목차, 그래도 크면 순서대로 나눈다. [(라벨, prev_rows, next_rows)]"""
    buckets, order = defaultdict(lambda: ([], [])), []
    for side, rows in ((1, nxt), (0, prev)):
        for r in rows:
            k = title_key(r, depth)
            if k not in buckets:
                order.append(k)
            buckets[k][side].append(r)
    groups = []
    for k in order:
        p, n = buckets[k]
        name = " > ".join(x for x in k if x) or "(목차 없음)"
        if chars(p) + chars(n) <= GROUP_CHAR_LIMIT and len(p) + len(n) <= GROUP_PARA_LIMIT:
            groups.append((name, p, n))
            continue
        deeper = {title_key(r, depth + 1) for r in p + n}
        if len(deeper) > 1 and depth < 6:
            groups += build_groups(p, n, depth + 1, name)
        else:  # 더 나눌 목차가 없음 -> 순서 기준으로 같은 비율씩 자름
            k_parts = max(math.ceil((chars(p) + chars(n)) / GROUP_CHAR_LIMIT),
                          math.ceil((len(p) + len(n)) / GROUP_PARA_LIMIT))
            for i in range(k_parts):
                ps = p[len(p) * i // k_parts: len(p) * (i + 1) // k_parts]
                ns = n[len(n) * i // k_parts: len(n) * (i + 1) // k_parts]
                groups.append((f"{name} [{i + 1}/{k_parts}]", ps, ns))
    return groups


# ---------------------------------------------------------------- 모델 호출

def para_payload(r):
    return {
        "id": r["paragraph_id"],
        "source_type": r["source_type"],
        "title_path": r.get("title_path"),
        "text": r["source_text"],
    }


def user_message(company, y1, y2, group_name, prev, nxt):
    return json.dumps({
        "company": company,
        "compare": f"{y1}→{y2}",
        "section": group_name,
        "PREV": {"year": y1, "paragraphs": [para_payload(r) for r in prev]},
        "NEXT": {"year": y2, "paragraphs": [para_payload(r) for r in nxt]},
    }, ensure_ascii=False)


class Judge:
    def __init__(self, model, reasoning_effort=None, max_retries=4):
        from openai import OpenAI
        self.client = OpenAI(api_key=load_api_key(), max_retries=max_retries, timeout=600)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.lock = threading.Lock()
        self.usage = Counter()

    def __call__(self, message, cache_file, force=False):
        if cache_file.exists() and not force:
            return json.loads(cache_file.read_text(encoding="utf-8")), True
        kwargs = {}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": message}],
            response_format=RESPONSE_FORMAT,
            **kwargs,
        )
        choice = resp.choices[0]
        if getattr(choice.message, "refusal", None):
            raise RuntimeError(f"모델 거부: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise RuntimeError("응답이 길이 제한으로 잘림 (그룹을 더 작게: GROUP_CHAR_LIMIT)")
        data = json.loads(choice.message.content)
        with self.lock:
            if resp.usage:
                self.usage["input"] += resp.usage.prompt_tokens or 0
                self.usage["output"] += resp.usage.completion_tokens or 0
            self.usage["calls"] += 1
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return data, False


def cache_path(model, company, y1, y2, message):
    h = hashlib.sha256(f"{PROMPT_VERSION}|{model}|{SYSTEM_PROMPT}|{message}".encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / company / f"{y1}_{y2}" / f"{h}.json"


def missing_ids(results, prev, nxt):
    seen = {i for x in results for i in x["prev_ids"] + x["next_ids"]}
    return [r["paragraph_id"] for r in prev + nxt if r["paragraph_id"] not in seen]


def judge_group(judge, company, y1, y2, group):
    """그룹 하나 판정. 누락 id가 있으면 한 번 더 호출. (results, 정보) 반환"""
    name, prev, nxt = group
    msg = user_message(company, y1, y2, name, prev, nxt)
    cf = cache_path(judge.model, company, y1, y2, msg)
    data, cached = judge(msg, cf)
    results = data.get("results", [])
    if missing_ids(results, prev, nxt) and not cached:
        data2, _ = judge(msg, cf, force=True)
        if len(missing_ids(data2.get("results", []), prev, nxt)) < len(missing_ids(results, prev, nxt)):
            results = data2.get("results", [])
        else:
            cf.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    return results


# ---------------------------------------------------------------- 결과 정리

def join_text(ids, by_id):
    texts = [by_id[i]["source_text"] for i in ids]
    text = "\n\n".join(f"[{i}]\n{t}" for i, t in zip(ids, texts)) if len(ids) > 1 else (texts[0] if texts else "")
    if len(text) > EXCEL_CELL_LIMIT:
        text = text[:EXCEL_CELL_LIMIT] + "\n…(엑셀 셀 한도로 잘림. 전문은 paragraph_id로 확인)"
    return text


def row_of(company, y1, y2, x, by_id):
    return {
        "기업명": company,
        "비교 연도": f"{y1}→{y2}",
        "변화 영역": x["area"],
        "이전 연도 paragraph_id": "\n".join(x["prev_ids"]),
        "이전 연도 원문 전체": join_text(x["prev_ids"], by_id),
        "다음 연도 paragraph_id": "\n".join(x["next_ids"]),
        "다음 연도 원문 전체": join_text(x["next_ids"], by_id),
        "변화 유형": x["change_type"] if x["judgement"] == "변화" else x["judgement"],
        "변화 설명": x["description"],
        "판정 근거": x["evidence"],
    }


def process_pair(judge, company, y1, y2, prev_all, next_all, workers, dry_run):
    by_id = {r["paragraph_id"]: r for r in prev_all + next_all}
    prev_ids_all = {r["paragraph_id"] for r in prev_all}
    next_ids_all = {r["paragraph_id"] for r in next_all}

    oversized = [r for r in prev_all + next_all if len(r["source_text"]) > MAX_PARA_CHARS]
    big = {r["paragraph_id"] for r in oversized}
    prev = [r for r in prev_all if r["paragraph_id"] not in big]
    nxt = [r for r in next_all if r["paragraph_id"] not in big]
    groups = build_groups(prev, nxt)

    stat = {
        "기업명": company, "비교 연도": f"{y1}→{y2}",
        "이전 연도 입력 문단 수": len(prev_all), "다음 연도 입력 문단 수": len(next_all),
        "그룹 수": len(groups), "모델 입력 글자수": chars(prev) + chars(nxt),
        "과대 문단 수(모델 미전송)": len(oversized),
    }
    if dry_run:
        return stat, [], [], []

    all_results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(judge_group, judge, company, y1, y2, g): g for g in groups}
        for fut in as_completed(futs):
            name, gp, gn = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # 그룹 실패 -> 그 그룹 문단 전부 검토 필요
                print(f"  [오류] {company} {y1}→{y2} '{name}': {e}")
                res = [{"prev_ids": [], "next_ids": [], "_unprocessed": [r["paragraph_id"] for r in gp + gn],
                        "_error": str(e)}]
            for x in res:
                x["_group"] = name
            all_results += res
            print(f"  {company} {y1}→{y2} 그룹 완료: {name} (P{len(gp)}/N{len(gn)})")

    confirmed, review, log = [], [], []
    errored = set()
    for x in all_results:
        if "_unprocessed" in x:
            errored |= set(x["_unprocessed"])
            continue
        # 잘못된 id 제거, 방향 오류 보정
        bad = [i for i in x["prev_ids"] + x["next_ids"] if i not in by_id]
        x["prev_ids"] = [i for i in x["prev_ids"] if i in prev_ids_all]
        x["next_ids"] = [i for i in x["next_ids"] if i in next_ids_all]
        if not x["prev_ids"] and not x["next_ids"]:
            continue
        reason = x.get("review_reason", "")
        if bad:
            reason = (reason + f" / 존재하지 않는 id 제거: {', '.join(bad)}").strip(" /")
        if x["judgement"] == "변화" and (x["change_type"] == "없음" or x["area"] == "해당 없음"):
            x["judgement"] = "검토 필요"
            reason = (reason + " / 변화로 판정했으나 유형·영역이 비어 있음").strip(" /")
        if x["judgement"] == "변화" and not x["evidence"].strip():
            x["judgement"] = "검토 필요"
            reason = (reason + " / 판정 근거 없음").strip(" /")
        x["review_reason"] = reason
        log.append({"company": company, "pair": f"{y1}→{y2}", **x})
        row = row_of(company, y1, y2, x, by_id)
        if x["judgement"] == "변화":
            confirmed.append(row)
        elif x["judgement"] == "검토 필요":
            review.append({**row, "검토 사유": reason, "그룹": x["_group"]})

    # 누락·과대·오류 문단 -> 검토 필요
    seen = {i for x in log for i in x["prev_ids"] + x["next_ids"]}
    for r in prev_all + next_all:
        pid = r["paragraph_id"]
        if pid in seen:
            continue
        if pid in big:
            why = f"문단이 너무 김({len(r['source_text']):,}자): 전처리 이상 가능, 모델 미전송"
        elif pid in errored:
            why = "모델 호출 실패"
        else:
            why = "모델이 판정하지 않음(누락)"
        side_prev = pid in prev_ids_all
        x = {"prev_ids": [pid] if side_prev else [], "next_ids": [] if side_prev else [pid],
             "area": "", "judgement": "검토 필요", "change_type": "없음", "description": "", "evidence": ""}
        review.append({**row_of(company, y1, y2, x, by_id), "검토 사유": why, "그룹": ""})

    # 중복 매칭: 한 문단이 서로 다른 상대 문단 묶음과 매칭된 경우
    partners = defaultdict(set)
    for x in log:
        for i in x["prev_ids"]:
            partners[i].add(tuple(sorted(x["next_ids"])))
        for i in x["next_ids"]:
            partners[i].add(tuple(sorted(x["prev_ids"])))
    dup = sorted(i for i, s in partners.items() if len({p for p in s if p}) > 1)

    judged = Counter(x["judgement"] for x in log)
    stat.update({
        "대응 문단쌍 수": sum(1 for x in log if x["prev_ids"] and x["next_ids"]),
        "변화 확정 수": len(confirmed),
        "변화 없음 수": judged["변화 없음"],
        "대상 외 수": judged["대상 외"],
        "검토 필요 수": len(review),
        "누락 문단 수": sum(1 for r in review if r["검토 사유"].startswith("모델이 판정하지 않음")),
        "중복 매칭 문단 수": len(dup),
        "중복 매칭 문단 id": ", ".join(dup)[:EXCEL_CELL_LIMIT],
        "처리된 입력 문단 비율": f"{len(seen & (prev_ids_all | next_ids_all)) / max(1, len(prev_ids_all | next_ids_all)):.1%}",
    })
    return stat, confirmed, review, log


# ---------------------------------------------------------------- 엑셀

def write_excel(path, company, pairs_out):
    cols = ["기업명", "비교 연도", "변화 영역", "이전 연도 paragraph_id", "이전 연도 원문 전체",
            "다음 연도 paragraph_id", "다음 연도 원문 전체", "변화 유형", "변화 설명", "판정 근거"]
    all_conf = [r for _, c, _ in pairs_out for r in c]
    all_rev = [r for _, _, rv in pairs_out for r in rv]
    sheets = {"확정_GT_전체": pd.DataFrame(all_conf, columns=cols)}
    for stat, conf, _ in pairs_out:
        y1, y2 = stat["비교 연도"].split("→")
        sheets[f"{y1}_{y2}"] = pd.DataFrame(conf, columns=cols)
    sheets["검토_필요"] = pd.DataFrame(all_rev, columns=cols + ["검토 사유", "그룹"])
    status = pd.DataFrame([s for s, _, _ in pairs_out])
    sheets["처리_현황"] = status

    widths = {"이전 연도 원문 전체": 70, "다음 연도 원문 전체": 70, "변화 설명": 45, "판정 근거": 60, "검토 사유": 40}
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in sheets.items():
            df = df.map(lambda v: re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", v) if isinstance(v, str) else v)
            df.to_excel(w, sheet_name=name, index=False)
            ws = w.sheets[name]
            ws.freeze_panes = "A2"
            if len(df):
                ws.auto_filter.ref = ws.dimensions
            for i, col in enumerate(df.columns, 1):
                letter = ws.cell(row=1, column=i).column_letter
                ws.column_dimensions[letter].width = widths.get(col, min(max(len(str(col)) + 4, 12), 30))
            if name != "처리_현황":
                from openpyxl.styles import Alignment
                for row in ws.iter_rows(min_row=2):
                    for c in row:
                        c.alignment = Alignment(wrap_text=True, vertical="top")


# ---------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description="OpenAI API로 사업보고서 연도별 변화 GT 생성")
    p.add_argument("--companies", nargs="+", help="기업명 (기본: 전체)")
    p.add_argument("--industries", nargs="+", default=INDUSTRIES, choices=INDUSTRIES)
    p.add_argument("--model", default=os.environ.get("OPENAI_GT_MODEL", "gpt-5"), help="OpenAI 모델 (기본: gpt-5)")
    p.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], help="추론 모델일 때만")
    p.add_argument("--workers", type=int, default=4, help="동시 호출 수")
    p.add_argument("--dry-run", action="store_true", help="API 호출 없이 그룹·분량만 출력")
    p.add_argument("--out", default=str(OUT_DIR), help="엑셀 저장 폴더")
    args = p.parse_args(argv)

    data = load_paragraphs(args.industries)
    companies = args.companies or sorted(data)
    unknown = [c for c in companies if c not in data]
    if unknown:
        p.error(f"데이터에 없는 기업: {', '.join(unknown)} (있는 기업: {', '.join(sorted(data))})")

    judge = None if args.dry_run else Judge(args.model, args.reasoning_effort)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for company in companies:
        years = data[company]
        pairs = year_pairs(years)
        print(f"\n=== {company}: 연도 {sorted(years)} -> 비교 {[f'{a}→{b}' for a, b in pairs]}")
        pairs_out, logs = [], []
        for y1, y2 in pairs:
            stat, conf, rev, log = process_pair(judge, company, y1, y2, years[y1], years[y2], args.workers, args.dry_run)
            pairs_out.append((stat, conf, rev))
            logs += log
            print("  " + ", ".join(f"{k}={v}" for k, v in stat.items() if k not in ("기업명", "중복 매칭 문단 id")))
        if args.dry_run:
            continue
        xlsx = out_dir / f"{company}_Confirmed_GT_Before_After_Paragraphs.xlsx"
        write_excel(xlsx, company, pairs_out)
        with open(out_dir / f"{company}_gt_decisions.jsonl", "w", encoding="utf-8") as f:
            for x in logs:
                f.write(json.dumps({k: v for k, v in x.items() if not k.startswith("_") or k == "_group"},
                                   ensure_ascii=False) + "\n")
        print(f"  저장: {xlsx}")

    if judge:
        u = judge.usage
        print(f"\nAPI 호출 {u['calls']}회, 입력 토큰 {u['input']:,}, 출력 토큰 {u['output']:,} (캐시 사용분 제외)")


if __name__ == "__main__":
    raise SystemExit(main())
