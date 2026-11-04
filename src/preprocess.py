"""DB의 원본 XML -> II장 소제목·문장 -> DB(sections, sentences). 재실행 시 UPSERT."""
import re

import yaml

from . import database
from .report_parser import extract_chapter2, ParseError
from .section_classifier import classify

SPLIT = re.compile(r"(?<=[.!?])\s+(?=[^\d\s])")  # 소수점(3.5)은 뒤가 숫자라 분리 안 됨
MAXLEN = 400
NOISE = re.compile(r"\(?\d{1,2}[.)]?|[가-하]\.")  # '1.', '(2)', '가.' 같은 번호만 있는 조각



def split_sentences(par):
    out = []
    for s in SPLIT.split(par):
        # 지나치게 긴 문장은 세미콜론 기준으로 추가 분할 (수치/부정 표현은 그대로 유지)
        out += [x.strip() for x in re.split(r";\s+", s)] if len(s) > MAXLEN else [s.strip()]
    return [s for s in out if s and not NOISE.fullmatch(s)]


def table_text(label, cols, unit):
    body = "; ".join(f"{c}: {v}" for c, v in cols)
    return " ".join(filter(None, [unit, f"{label} |" if label else "", body]))


def build_sections(xml_bytes, rcept_no):
    secs, seen = [], set()
    for si, sec in enumerate(extract_chapter2(xml_bytes), 1):
        sid, pi, ti, sents = f"S{si}", 0, 0, []
        for it in sec["items"]:
            if it["kind"] == "paragraph":
                pi += 1
                pk, st, tk = f"{sid}-P{pi:03d}", "paragraph", None
                units = split_sentences(it["text"])
            else:
                ti += 1
                pk, st, tk = f"{sid}-T{ti:03d}", "table", f"{rcept_no}-{sid}-T{ti:03d}"
                units = [table_text(l, c, it["unit"]) for l, c in it["rows"]]
            for k, text in enumerate(units, 1):
                sents.append({"subsection_title": it["sub"], "paragraph_key": pk, "sentence_key": f"{pk}-{k:02d}", "text": text,
                              "source_type": st, "table_key": tk, "xml_line": it["line"], "is_duplicate": (sid, text) in seen})
                seen.add((sid, text))
        secs.append({"section_no": si, "title": sec["section_title"], "analysis_area": classify(sec["section_title"]),
                     "xml_line": sec["line"], "sentences": sents})
    return secs


def run():
    con = database.connect()
    rows = con.execute("SELECT report_id, rcept_no, raw_xml FROM reports WHERE status='OK' AND raw_xml IS NOT NULL").fetchall()
    for r in rows:
        try:
            secs = build_sections(r["raw_xml"], r["rcept_no"])
            database.save_chapter(con, r["report_id"], secs)
            print(r["rcept_no"], sum(len(s["sentences"]) for s in secs), "문장")
        except ParseError as e:  # II장 못 찾으면 임의 본문 대신 실패 기록
            with database.tx(con):
                con.execute("UPDATE reports SET status='FAIL', fail_reason=? WHERE report_id=?", (f"II장 추출 실패: {e}", r["report_id"]))
            print("PARSE FAIL", r["rcept_no"], e)
