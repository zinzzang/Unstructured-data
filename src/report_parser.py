"""DART 사업보고서 XML에서 'II. 사업의 내용'만 추출. 못 찾으면 예외(임의 본문 반환 금지)."""
import re
from pathlib import Path

from lxml import etree

CH2_RE = re.compile(r"^II\.\s*사업의\s*내용")
SUB_RE = re.compile(r"^(\(\d+\)|\d+\)|[가-힣]\.)\s")  # 가. / (1) / 1)


BARE_AMP = re.compile(rb"&(?!(?:[A-Za-z][A-Za-z0-9]*|#[0-9]+|#x[0-9A-Fa-f]+);)")


class ParseError(Exception):
    pass


def _txt(el):
    return re.sub(r"\s+", " ", "".join(el.itertext()).replace("\xa0", " ")).strip()


def load(source):
    """source: XML bytes 또는 파일 경로. 원문에 이스케이프 안 된 '&'가 있어 엄격 XML 파싱은 실패 -> recover"""
    if not isinstance(source, (bytes, bytearray)):
        source = Path(source).read_bytes()
    source = BARE_AMP.sub(b"&amp;", bytes(source))  # recover만 쓰면 'S&P'가 'S'로 잘려 원문 손실 -> 먼저 이스케이프
    return etree.fromstring(source, etree.XMLParser(recover=True, huge_tree=True))


def find_chapter2(root):
    """본문의 SECTION-1 중 제목이 'II. 사업의 내용'인 것. 목차(<TITLE ATOC="N">, 표 안 텍스트)는 SECTION-1이 아니라 제외됨."""
    hits = [s for s in root.iter("SECTION-1") if CH2_RE.match(_txt(s.find("TITLE")) if s.find("TITLE") is not None else "")]
    if len(hits) != 1:
        raise ParseError(f"II장 SECTION-1 {len(hits)}개 발견(1개여야 함)")
    return hits[0]


def _span(c, name):
    try:
        return max(1, int(c.get(name, 1)))
    except ValueError:
        return 1


def table_rows(tbl):
    """표 -> [(row_label, [(col_header, value)])]. ROWSPAN/COLSPAN을 격자로 펼쳐 헤더·행 제목·값의 대응을 유지."""
    carry, grid = {}, []  # carry: col -> (남은 행수, 값)
    for tr in tbl.iter("TR"):
        head = tr.getparent().tag == "THEAD" or tr.find("TH") is not None
        row, col = {}, 0
        for c in (x for x in tr if x.tag in ("TD", "TH", "TU", "TE")):
            while col in carry and carry[col][0] > 0:
                row[col] = carry[col][1]; carry[col] = (carry[col][0] - 1, carry[col][1]); col += 1
            v, cs, rs = _txt(c), _span(c, "COLSPAN"), _span(c, "ROWSPAN")
            for k in range(cs):
                row[col + k] = v if (head or k == 0) else ""
                if rs > 1:
                    carry[col + k] = (rs - 1, row[col + k])
            col += cs
        while col in carry and carry[col][0] > 0:
            row[col] = carry[col][1]; carry[col] = (carry[col][0] - 1, carry[col][1]); col += 1
        grid.append((head, [row.get(i, "") for i in range(max(row, default=-1) + 1)]))
    if len(grid) > 1 and not any(h for h, _ in grid):
        grid[0] = (True, grid[0][1])  # 헤더 표시가 없으면 첫 행을 헤더로 간주
    heads = [r for h, r in grid if h]
    body = [r for h, r in grid if not h]
    n = max((len(r) for r in heads), default=0)
    header = []
    for i in range(n):
        parts = []
        for r in heads:
            if i < len(r) and r[i] and (not parts or parts[-1] != r[i]):
                parts.append(r[i])
        header.append(" ".join(parts))
    out = []
    for r in body:
        if not any(r):
            continue
        out.append((r[0], [(header[i] if i < len(header) and header[i] else f"col{i}", v) for i, v in enumerate(r) if i > 0 and v]))
    return out


def extract_chapter2(source):
    """반환: [{section_title, section_no, source_line, items:[{kind,subsection,text|rows,line}]}] (원문 순서)."""
    ch = find_chapter2(load(source))
    # II장 끝은 '다음 SECTION-1이 시작되는 곳'. 일부 보고서는 XML 중첩이 어긋나 III장 이후 SECTION-2가 II장의 자손으로 잡히므로,
    # 자손 순회 대신 문서 순서로 따라가다 다음 SECTION-1을 만나면 멈춘다.
    seq = list(ch.getroottree().getroot().iter("SECTION-1", "SECTION-2"))
    s2s = []
    for e in seq[seq.index(ch) + 1:]:
        if e.tag == "SECTION-1":
            break
        s2s.append(e)
    sections = []
    for s2 in s2s:
        t = s2.find("TITLE")
        sec = {"section_title": _txt(t) if t is not None else "", "line": s2.sourceline, "items": []}
        sub = ""
        unit = ""

        def walk(el):
            nonlocal sub, unit
            for c in el:
                if c.tag == "TABLE":
                    rows = table_rows(c)
                    if len(rows) <= 1 and "단위" in _txt(c) and len(_txt(c)) < 80 or not rows or all(not cols for _, cols in rows):
                        tx = _txt(c)  # 값 열 없는 표(단위·기간 표기, 라벨)
                        if "단위" in tx:
                            unit = tx  # 다음 표에 붙임
                        elif tx:
                            sec["items"].append({"kind": "paragraph", "sub": sub, "text": tx, "line": c.sourceline})
                    else:
                        sec["items"].append({"kind": "table", "sub": sub, "rows": rows, "unit": unit, "line": c.sourceline})
                        unit = ""
                elif c.tag == "P":
                    tx = _txt(c)
                    if not tx:
                        continue
                    if len(tx) < 80 and (c.get("USERMARK", "").strip().startswith("B") or SUB_RE.match(tx + " ")):
                        sub = tx
                    else:
                        sec["items"].append({"kind": "paragraph", "sub": sub, "text": tx, "line": c.sourceline})
                elif c.tag not in ("TITLE", "SECTION-1", "SECTION-2"):  # 다음 장/절이 중첩돼 들어와도 내려가지 않음
                    walk(c)

        walk(s2)
        sections.append(sec)
    if not sections:
        raise ParseError("II장은 있으나 하위 SECTION-2 없음")
    return sections
