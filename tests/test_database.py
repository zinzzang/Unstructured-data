import sqlite3
import pytest
from src import database as db


@pytest.fixture
def con(tmp_path):
    return db.connect(tmp_path / "t.db")


def _sec(texts):
    return [{"section_no": 1, "title": "1. 사업의 개요", "analysis_area": "BUSINESS", "xml_line": 1,
             "sentences": [{"subsection_title": "", "paragraph_key": "S1-P001", "sentence_key": f"S1-P001-{i:02d}", "text": t,
                            "source_type": "paragraph", "table_key": None, "xml_line": 1, "is_duplicate": False} for i, t in enumerate(texts, 1)]}]


def _report(con):
    cid = db.upsert_company(con, "001", "A사", "000001", "반도체")
    return db.upsert_report(con, cid, 2024, rcept_no="R1", status="OK")


def test_rerun_no_duplicates_and_stale_removed(con):
    rid = _report(con)
    _report(con); db.save_chapter(con, rid, _sec(["가.", "나."])); db.save_chapter(con, rid, _sec(["가.", "나."]))
    assert con.execute("SELECT COUNT(*) FROM sentences").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM companies").fetchone()[0] == 1
    db.save_chapter(con, rid, _sec(["가."]))  # 재실행 결과가 줄면 옛 행 삭제
    assert con.execute("SELECT COUNT(*) FROM sentences").fetchone()[0] == 1


def test_foreign_key_enforced(con):
    with pytest.raises(sqlite3.IntegrityError):
        con.execute("INSERT INTO sections (report_id, section_no, title, analysis_area) VALUES (999,1,'x','BUSINESS')")


def test_matches_upsert_and_check(con):
    rid = _report(con); db.save_chapter(con, rid, _sec(["가.", "나."]))
    a, b = [r[0] for r in con.execute("SELECT sentence_id FROM sentences ORDER BY 1")]
    rows = [dict(old=a, new=b, kind="PAIR", rank=1, similarity=.9, is_match=1), dict(old=a, new=None, kind="UNMATCHED_OLD", rank=None, similarity=.1, is_match=0)]
    db.save_matches(con, "tfidf", rows, [a, b]); db.save_matches(con, "tfidf", rows, [a, b])
    assert con.execute("SELECT COUNT(*) FROM sentence_matches").fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError):  # PAIR인데 한쪽이 NULL
        con.execute("INSERT INTO sentence_matches (old_sentence_id,new_sentence_id,method,kind) VALUES (?,NULL,'tfidf','PAIR')", (a,))
