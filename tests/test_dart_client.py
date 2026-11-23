from src import dart_client as d

ROWS = [
    {"report_nm": "사업보고서 (2024.12)", "rcept_no": "20250311000001", "rcept_dt": "20250311"},
    {"report_nm": "[기재정정]사업보고서 (2024.12)", "rcept_no": "20250420000002", "rcept_dt": "20250420"},
    {"report_nm": "반기보고서 (2025.06)", "rcept_no": "20250814000003", "rcept_dt": "20250814"},
    {"report_nm": "사업보고서 (2023.12)", "rcept_no": "20240312000004", "rcept_dt": "20240312"},
]

def test_pick_latest_correction():
    r = d.pick_report(ROWS, 2024)
    assert r["rcept_no"] == "20250420000002" and r["is_correction"]

def test_pick_excludes_other_year_and_half():
    assert d.pick_report(ROWS, 2023)["rcept_no"] == "20240312000004"
    assert d.pick_report(ROWS, 2022) is None

def test_error_response_not_zip(monkeypatch):
    class R:  # DART는 오류 시 ZIP이 아닌 XML을 돌려준다
        content = b"<result><status>013</status></result>"
        text = content.decode()
    monkeypatch.setattr(d, "_get", lambda *a, **k: R())
    try:
        d.fetch_zip("1")
        assert False
    except d.DartError:
        pass
