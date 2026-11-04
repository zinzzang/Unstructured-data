"""모든 DB 작업은 여기서. foreign_keys ON, 다중 저장은 트랜잭션, 재실행은 UPSERT로 중복 방지."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "annual_reports.db"
SCHEMA = ROOT / "sql" / "schema.sql"


def connect(path=None):
    path = Path(path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")  # 연결마다 켜야 함
    con.executescript(SCHEMA.read_text(encoding="utf-8"))
    return con


@contextmanager
def tx(con):
    """with tx(con): ... 블록 전체가 한 트랜잭션 (예외 시 롤백)."""
    with con:
        yield con


def upsert_company(con, corp_code, name, stock_code, industry):
    con.execute("""INSERT INTO companies (corp_code, name, stock_code, industry) VALUES (?,?,?,?)
                   ON CONFLICT(corp_code) DO UPDATE SET name=excluded.name, stock_code=excluded.stock_code, industry=excluded.industry""",
                (corp_code, name, stock_code, industry))
    return con.execute("SELECT company_id FROM companies WHERE corp_code=?", (corp_code,)).fetchone()[0]


def upsert_report(con, company_id, year, **f):
    cols = ["rcept_no", "rcept_dt", "report_nm", "is_correction", "status", "fail_reason", "zip_path", "raw_xml"]
    vals = [f.get(c) for c in cols]
    con.execute(f"""INSERT INTO reports (company_id, business_year, {",".join(cols)}) VALUES (?,?,{",".join("?"*len(cols))})
                    ON CONFLICT(company_id, business_year) DO UPDATE SET {",".join(f"{c}=excluded.{c}" for c in cols)},
                    collected_at=CURRENT_TIMESTAMP""", [company_id, year, *vals])
    return con.execute("SELECT report_id FROM reports WHERE company_id=? AND business_year=?", (company_id, year)).fetchone()[0]


def get_report(con, company, year):
    return con.execute("""SELECT r.* FROM reports r JOIN companies c USING(company_id) WHERE c.name=? AND r.business_year=?""",
                       (company, year)).fetchone()


def save_chapter(con, report_id, sections):
    """sections: [{section_no,title,analysis_area,xml_line,sentences:[dict]}]. UPSERT 후, 새 결과에 없는 옛 행은 삭제(재실행 시 잔재 방지)."""
    with tx(con):
        keep = []
        for s in sections:
            con.execute("""INSERT INTO sections (report_id, section_no, title, analysis_area, xml_line) VALUES (?,?,?,?,?)
                           ON CONFLICT(report_id, section_no) DO UPDATE SET title=excluded.title, analysis_area=excluded.analysis_area, xml_line=excluded.xml_line""",
                        (report_id, s["section_no"], s["title"], s["analysis_area"], s["xml_line"]))
            sid = con.execute("SELECT section_id FROM sections WHERE report_id=? AND section_no=?", (report_id, s["section_no"])).fetchone()[0]
            for x in s["sentences"]:
                con.execute("""INSERT INTO sentences (report_id, section_id, analysis_area, subsection_title, paragraph_key, sentence_key, text, source_type, table_key, xml_line, is_duplicate)
                               VALUES (?,?,?,?,?,?,?,?,?,?,?)
                               ON CONFLICT(report_id, sentence_key) DO UPDATE SET section_id=excluded.section_id, analysis_area=excluded.analysis_area,
                               subsection_title=excluded.subsection_title, paragraph_key=excluded.paragraph_key, text=excluded.text, source_type=excluded.source_type,
                               table_key=excluded.table_key, xml_line=excluded.xml_line, is_duplicate=excluded.is_duplicate""",
                            (report_id, sid, s["analysis_area"], x["subsection_title"], x["paragraph_key"], x["sentence_key"], x["text"],
                             x["source_type"], x["table_key"], x["xml_line"], int(x["is_duplicate"])))
                keep.append(x["sentence_key"])
        con.execute(f"DELETE FROM sentences WHERE report_id=? AND sentence_key NOT IN ({','.join('?'*len(keep))})", [report_id, *keep])
        con.execute(f"DELETE FROM sections WHERE report_id=? AND section_no NOT IN ({','.join('?'*len(sections))})", [report_id, *[s['section_no'] for s in sections]])


def sentences_of(con, company, year, area=None):
    q = """SELECT s.* FROM sentences s JOIN reports r USING(report_id) JOIN companies c USING(company_id)
           WHERE c.name=? AND r.business_year=? AND s.is_duplicate=0"""
    a = [company, year]
    if area:
        q += " AND s.analysis_area=?"; a.append(area)
    return con.execute(q + " ORDER BY s.sentence_id", a).fetchall()


def save_matches(con, method, rows, scope_sentence_ids):
    """rows: dict(old,new,kind,rank,similarity,is_match). 같은 방법·같은 비교범위의 이전 결과를 지우고 다시 저장(한 트랜잭션)."""
    with tx(con):
        ids = ",".join("?" * len(scope_sentence_ids))
        con.execute(f"DELETE FROM sentence_matches WHERE method=? AND (old_sentence_id IN ({ids}) OR new_sentence_id IN ({ids}))",
                    [method, *scope_sentence_ids, *scope_sentence_ids])
        con.executemany("""INSERT INTO sentence_matches (old_sentence_id, new_sentence_id, method, kind, rank, similarity, is_match)
                           VALUES (:old,:new,:method,:kind,:rank,:similarity,:is_match)
                           ON CONFLICT DO UPDATE SET rank=excluded.rank, similarity=excluded.similarity, is_match=excluded.is_match""",
                        [{**r, "method": method} for r in rows])


def save_changes(con, method, rows):
    """같은 검출방법의 이전 결과를 지우고 다시 저장 (한 트랜잭션, 재실행 안전)."""
    with tx(con):
        ids = [x for r in rows for x in (r["old"], r["new"]) if x]
        if ids:
            q = ",".join("?" * len(ids))
            con.execute(f"DELETE FROM detected_changes WHERE detection_method=? AND (old_sentence_id IN ({q}) OR new_sentence_id IN ({q}))", [method, *ids, *ids])
        con.executemany("""INSERT INTO detected_changes (old_sentence_id, new_sentence_id, change_type, change_topic, change_summary, detection_method,
                           model_info, tfidf_similarity, embedding_similarity, confidence, review_status)
                           VALUES (:old,:new,:change_type,:topic,:change_summary,:method,:info,:tfidf,:emb,:confidence,:review)
                           ON CONFLICT DO UPDATE SET change_type=excluded.change_type, change_summary=excluded.change_summary, confidence=excluded.confidence""", rows)
