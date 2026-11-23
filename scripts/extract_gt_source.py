"""LLM Ground Truth 생성용 원문 추출: 사업보고서 원본 XML에서 'II. 사업의 내용' 장만 잘라 텍스트로 저장한다.

분석 파이프라인(preprocessing_v2 / run_v1)이 쓰는 범위와 같은 'II. 사업의 내용' 장 전체를 대상으로 하며,
의미를 바꾸는 가공(문장 분리, 정규화, 숫자 마스킹, 표 행 재구성 등)은 하지 않는다. 정리하는 것은 다음뿐이다.

  - 무관한 마크업 제거: 태그·속성, 빈 <P>, <PGBRK>(쪽 나눔), <IMG>/<IMAGE>(이미지 및 캡션)
  - 깨진/보이지 않는 문자 정리: NBSP·전각 공백(U+3000)→공백, 제로폭·BOM·소프트하이픈·제어문자·U+FFFD·사용자 정의 영역 문자 제거,
    DART 전용 개체 &cr;(셀 내 줄바꿈)→공백, XML에 없는 HTML 개체(&nbsp; 등)→해당 문자
  - 공백 정리: 블록(문단/표 행) 안의 연속 공백·줄바꿈을 공백 하나로
  - 중복 제거: 바로 앞 블록과 완전히 같은 블록, 같은 절 안에서 이미 나온 20자 이상 문단의 반복

원문 보존을 위해 다음은 기존 파서(src/report_parser.py)와 다르게 처리한다.
  - '<셀트리온>', '< DRAM 시장점유율 추이 >', '<표1>'처럼 본문에 쓰인 꺾쇠 텍스트는 XML 태그로 오인되어 사라지므로,
    DART 태그 형태(<대문자이름 ...>)가 아닌 '<'는 미리 &lt;로 바꿔 텍스트로 남긴다.
  - 표는 행 단위로 '| 셀 | 셀 |' 형태로 그대로 적는다. ROWSPAN 셀은 행마다 값을 반복해 각 행만 봐도 열 대응이 유지되게 하고,
    COLSPAN의 나머지 칸은 빈 칸으로 둔다.

출력: data/gt_source/<산업>/<기업>_<사업연도>.txt  (+ data/gt_source/manifest.csv)

사용 예
  python scripts/extract_gt_source.py                      # 전체 기업·연도
  python scripts/extract_gt_source.py --years 2023 2024 2025
  python scripts/extract_gt_source.py --companies SK하이닉스
"""
from __future__ import annotations

import argparse
import csv
import gzip
import html.entities
import re
import shutil
import sqlite3
import tempfile
import unicodedata
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "annual_reports.db"
DB_GZ_PATH = ROOT / "data" / "annual_reports.db.gz"
OUT_DIR = ROOT / "data" / "gt_source"
INDUSTRY_DIR = {"반도체": "semiconductor", "2차전지": "battery", "바이오/제약": "biopharma"}

CH2_RE = re.compile(r"^II\.\s*사업의\s*내용")
BLOCK_TAGS = {"TITLE", "P", "TABLE", "SPAN"}
SKIP_TAGS = {"IMG", "IMAGE", "IMG-CAPTION", "PGBRK", "COLGROUP", "COL"}
CELL_TAGS = {"TD", "TH", "TU", "TE"}
DUP_MIN_CHARS = 20

# --------------------------------------------------------------- 원본 XML 로드

BARE_AMP = re.compile(r"&(?!(?:[A-Za-z][A-Za-z0-9]*|#[0-9]+|#x[0-9A-Fa-f]+);)")
NAMED_ENTITY = re.compile(r"&([A-Za-z][A-Za-z0-9]*);")
XML_ENTITIES = {"amp", "lt", "gt", "quot", "apos"}
# DART 태그는 대문자 영문 이름(SECTION-1, TABLE, IMG-CAPTION ...). 그 외 '<'는 본문 텍스트.
TEXT_LT = re.compile(r"<(?!/?[A-Z][A-Z0-9-]*(?:[\s/>])|[!?])")


def _entity(m: re.Match) -> str:
    name = m.group(1)
    if name in XML_ENTITIES:
        return m.group(0)
    if name == "cr":  # DART 전용: 셀 안 줄바꿈
        return "\n"
    ch = html.entities.html5.get(name + ";")
    return ch if ch is not None else "&amp;" + name + ";"  # 모르는 개체는 글자 그대로 남김


def load_xml(raw: bytes) -> etree._Element:
    text = raw.decode("utf-8", errors="replace")
    text = NAMED_ENTITY.sub(_entity, text)
    text = BARE_AMP.sub("&amp;", text)
    head, sep, body = text.partition("?>") if text.lstrip().startswith("<?xml") else ("", "", text)
    body = TEXT_LT.sub("&lt;", body)
    data = (head + sep + body).encode("utf-8")
    return etree.fromstring(data, etree.XMLParser(recover=True, huge_tree=True))


# --------------------------------------------------------------- 문자 정리

SPACE_CHARS = {"\xa0", "　", " ", " ", " ", " ", "\t", "\n", "\r"}
DROP_CHARS = {"​", "‌", "‍", "‎", "‏", "⁠", "﻿", "\xad", "�"}
WS = re.compile(r"[ ]+")


def clean(text: str, stats: dict) -> str:
    out = []
    for ch in text:
        if ch in SPACE_CHARS:
            out.append(" ")
        elif ch in DROP_CHARS or (unicodedata.category(ch) in ("Cc", "Co", "Cs")):
            stats["dropped_chars"] += 1
        else:
            out.append(ch)
    return WS.sub(" ", "".join(out)).strip()


def node_text(el, stats) -> str:
    """요소 안의 텍스트(이미지 등 제외)를 이어 붙인다. 꼬리 텍스트는 포함하지 않는다."""
    parts = [el.text or ""]
    for c in el:
        if c.tag not in SKIP_TAGS:
            parts.append(node_text(c, stats))
        parts.append(c.tail or "")
    return clean("".join(parts), stats) if el is not None else ""


# --------------------------------------------------------------- 표

def _span(c, name):
    try:
        return max(1, int(c.get(name, 1)))
    except ValueError:
        return 1


def render_table(tbl, stats) -> list[str]:
    carry: dict[int, tuple[int, str]] = {}  # col -> (남은 행 수, 값)
    lines = []
    for tr in tbl.iter("TR"):
        row: dict[int, str] = {}
        col = 0
        for c in (x for x in tr if x.tag in CELL_TAGS):
            while col in carry and carry[col][0] > 0:
                row[col] = carry[col][1]
                carry[col] = (carry[col][0] - 1, carry[col][1])
                col += 1
            v = node_text(c, stats).replace("|", "｜")
            cs, rs = _span(c, "COLSPAN"), _span(c, "ROWSPAN")
            for k in range(cs):
                row[col + k] = v if k == 0 else ""
                if rs > 1:
                    carry[col + k] = (rs - 1, row[col + k])
            col += cs
        while col in carry and carry[col][0] > 0:
            row[col] = carry[col][1]
            carry[col] = (carry[col][0] - 1, carry[col][1])
            col += 1
        cells = [row.get(i, "") for i in range(max(row, default=-1) + 1)]
        if any(cells):
            lines.append("| " + " | ".join(cells) + " |")
    return lines


# --------------------------------------------------------------- II장 추출

def chapter2_elements(root):
    """문서 순서로 'II. 사업의 내용' SECTION-1부터 다음 SECTION-1 직전까지의 요소.
    일부 보고서는 XML 중첩이 어긋나 III장 이후가 II장의 자손으로 잡히므로 자손 순회 대신 문서 순서로 자른다."""
    elems = list(root.iter())
    starts = [i for i, e in enumerate(elems) if e.tag == "SECTION-1" and e.find("TITLE") is not None
              and CH2_RE.match(clean("".join(e.find("TITLE").itertext()), {"dropped_chars": 0}))]
    if len(starts) != 1:
        raise ValueError(f"'II. 사업의 내용' SECTION-1 {len(starts)}개 발견(1개여야 함)")
    out = []
    for e in elems[starts[0] + 1:]:
        if e.tag == "SECTION-1":
            break
        out.append(e)
    return out


def extract_blocks(root, stats):
    """[(kind, text_or_lines, section_title)] 원문 순서. kind: heading / para / table"""
    elems = chapter2_elements(root)
    in_range = set(elems)
    blocks = []
    section = ""
    for e in elems:
        if not isinstance(e.tag, str) or e.tag not in BLOCK_TAGS:
            continue
        # 다른 블록(P/TABLE/TITLE/SPAN) 안에 있는 요소는 그 블록에서 함께 처리된다
        if any(a.tag in BLOCK_TAGS or a.tag in CELL_TAGS for a in e.iterancestors() if a in in_range):
            continue
        if e.tag == "TITLE":
            t = node_text(e, stats)
            if not t or e.getparent().tag == "SECTION-1":  # 장 제목은 render()에서 한 번만 쓴다
                continue
            level = {"SECTION-2": 2, "SECTION-3": 3}.get(e.getparent().tag, 3)
            if level == 2:
                section = t
            blocks.append(("heading", "#" * level + " " + t, section))
        elif e.tag == "TABLE":
            lines = render_table(e, stats)
            if lines:
                blocks.append(("table", lines, section))
        else:  # P, 블록 밖 SPAN
            t = node_text(e, stats)
            if t:
                blocks.append(("para", t, section))
    return blocks


def dedupe(blocks, stats):
    out, seen = [], set()
    prev_key = None
    for kind, content, section in blocks:
        key = (kind, content if isinstance(content, str) else "\n".join(content))
        if key == prev_key and kind != "heading":
            stats["dup_adjacent"] += 1
            continue
        prev_key = key
        if kind == "para" and len(content) >= DUP_MIN_CHARS:
            k = (section, content)
            if k in seen:
                stats["dup_in_section"] += 1
                continue
            seen.add(k)
        out.append((kind, content, section))
    return out


def render(blocks, header: str) -> str:
    parts = [header, "", "# II. 사업의 내용"]
    for kind, content, _ in blocks:
        parts.append("")
        parts.append("\n".join(content) if kind == "table" else content)
    return "\n".join(parts).rstrip() + "\n"


# --------------------------------------------------------------- 실행

def open_db():
    if DB_PATH.exists():
        return sqlite3.connect(DB_PATH), None
    tmp = Path(tempfile.mkstemp(suffix=".db")[1])
    with gzip.open(DB_GZ_PATH, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return sqlite3.connect(tmp), tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, help="사업연도 (기본: 전체)")
    ap.add_argument("--companies", nargs="+", help="기업명 (기본: 전체)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    con, tmp = open_db()
    try:
        rows = con.execute(
            """SELECT c.name, c.industry, r.business_year, r.rcept_no, r.rcept_dt, r.report_nm, r.raw_xml
               FROM reports r JOIN companies c USING(company_id)
               WHERE r.status='OK' AND r.raw_xml IS NOT NULL
               ORDER BY c.industry, c.name, r.business_year""").fetchall()
    finally:
        con.close()
        if tmp:
            tmp.unlink(missing_ok=True)

    manifest = []
    for name, industry, year, rcept_no, rcept_dt, report_nm, raw in rows:
        if args.years and year not in args.years or args.companies and name not in args.companies:
            continue
        stats = {"dropped_chars": 0, "dup_adjacent": 0, "dup_in_section": 0}
        rec = {"company": name, "industry": industry, "business_year": year, "rcept_no": rcept_no,
               "rcept_dt": rcept_dt, "report_nm": report_nm}
        try:
            blocks = dedupe(extract_blocks(load_xml(bytes(raw)), stats), stats)
            header = f"[{name} | {industry} | {year} 사업연도 | {report_nm} | 접수번호 {rcept_no}]"
            text = render(blocks, header)
            path = args.out / INDUSTRY_DIR.get(industry, industry) / f"{name}_{year}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            rec.update(status="OK", file=path.relative_to(args.out).as_posix(), chars=len(text),
                       sections=sum(k == "heading" and c.startswith("## ") for k, c, _ in blocks),
                       paragraphs=sum(k == "para" for k, _, _ in blocks),
                       tables=sum(k == "table" for k, _, _ in blocks), **stats, error="")
        except Exception as e:  # 한 보고서 실패가 전체를 막지 않게 기록만 한다
            rec.update(status="FAIL", error=f"{type(e).__name__}: {e}")
        manifest.append(rec)
        print(rec["status"], name, year, rec.get("chars", ""), rec["error"])

    args.out.mkdir(parents=True, exist_ok=True)
    fields = ["company", "industry", "business_year", "rcept_no", "rcept_dt", "report_nm", "status", "file", "chars",
              "sections", "paragraphs", "tables", "dropped_chars", "dup_adjacent", "dup_in_section", "error"]
    with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest)


if __name__ == "__main__":
    main()
