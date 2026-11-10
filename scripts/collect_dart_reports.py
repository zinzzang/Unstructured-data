"""OpenDART에서 사업보고서 원문 XML을 수집해 SQLite(reports 테이블)에 저장한다.

흐름: corp_code -> 공시검색(list.json)으로 사업보고서 rcept_no 찾기 -> 원문(document.xml) ZIP 다운로드
      -> ZIP은 data/raw/zip/{rcept_no}.zip, 본문 XML은 reports.raw_xml 로 저장.

API 키: 환경변수 DART_API_KEY, 또는 프로젝트 루트/상위 폴더의 .env 파일에 `DART_API_KEY=...`
사용 예:
    python scripts/collect_dart_reports.py                       # 9개 기업 x settings.yaml의 연도
    python scripts/collect_dart_reports.py --companies SK하이닉스 --years 2024 2025
    python scripts/collect_dart_reports.py --force               # 이미 OK인 보고서도 다시 받기
    python scripts/collect_dart_reports.py --db data/test.db     # 다른 DB에 저장
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import database as db

API_BASE = "https://opendart.fss.or.kr/api"
ZIP_DIR = ROOT / "data" / "raw" / "zip"

# 분석 대상 기업 (corp_code는 OpenDART 고유번호)
COMPANIES = [
    {"corp_code": "00164779", "name": "SK하이닉스", "stock_code": "000660", "industry": "반도체"},
    {"corp_code": "00126380", "name": "삼성전자", "stock_code": "005930", "industry": "반도체"},
    {"corp_code": "00161383", "name": "한미반도체", "stock_code": "042700", "industry": "반도체"},
    {"corp_code": "01515323", "name": "LG에너지솔루션", "stock_code": "373220", "industry": "2차전지"},
    {"corp_code": "00126362", "name": "삼성SDI", "stock_code": "006400", "industry": "2차전지"},
    {"corp_code": "00631518", "name": "SK이노베이션", "stock_code": "096770", "industry": "2차전지"},
    {"corp_code": "00877059", "name": "삼성바이오로직스", "stock_code": "207940", "industry": "바이오/제약"},
    {"corp_code": "00413046", "name": "셀트리온", "stock_code": "068270", "industry": "바이오/제약"},
    {"corp_code": "00145109", "name": "유한양행", "stock_code": "000100", "industry": "바이오/제약"},
]


class DartError(Exception):
    pass


def load_settings():
    return yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))


def load_api_key():
    key = os.environ.get("DART_API_KEY")
    if key:
        return key.strip()
    for env_path in (ROOT / ".env", ROOT.parent / ".env"):
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                name, sep, value = line.partition("=")
                if sep and name.strip() == "DART_API_KEY":
                    return value.strip().strip('"').strip("'")
    raise SystemExit("DART_API_KEY가 없습니다. 환경변수나 .env 파일에 DART_API_KEY=발급받은키 를 넣어주세요.")


class DartClient:
    def __init__(self, api_key, interval=0.5, max_retries=3):
        self.api_key = api_key
        self.interval = interval
        self.max_retries = max_retries

    def _get(self, endpoint, **params):
        url = f"{API_BASE}/{endpoint}?" + urllib.parse.urlencode({"crtfc_key": self.api_key, **params})
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            time.sleep(self.interval)  # 호출 간격 유지 (분당 호출 제한 대응)
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    return resp.read()
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(self.interval * attempt * 2)
        raise DartError(f"{endpoint} 요청 실패 ({self.max_retries}회 재시도): {last_err}")

    def list_annual_reports(self, corp_code, year):
        """사업연도 year의 사업보고서 목록. 제출은 보통 다음 해 3월이라 year+1년 전체를 검색."""
        items, page = [], 1
        while True:
            data = json.loads(self._get(
                "list.json", corp_code=corp_code,
                bgn_de=f"{year + 1}0101", end_de=f"{year + 1}1231",
                pblntf_ty="A", pblntf_detail_ty="A001",  # 정기공시 > 사업보고서
                page_no=page, page_count=100,
            ))
            status = data.get("status")
            if status == "013":  # 조회된 데이터 없음
                break
            if status != "000":
                raise DartError(f"list.json status={status} {data.get('message')}")
            items += data.get("list", [])
            if page >= int(data.get("total_page", 1)):
                break
            page += 1
        tag = f"({year}.12)"
        return [x for x in items if "사업보고서" in x["report_nm"] and tag in x["report_nm"]]

    def download_document(self, rcept_no):
        raw = self._get("document.xml", rcept_no=rcept_no)
        if not zipfile.is_zipfile(io.BytesIO(raw)):
            # 오류 시 ZIP 대신 XML/JSON 메시지가 옴
            raise DartError(f"document.xml 응답이 ZIP이 아님: {raw[:200].decode('utf-8', 'replace')}")
        return raw


def pick_latest(reports):
    """정정 공시 정책(latest): 접수일·접수번호가 가장 늦은 최종본."""
    return max(reports, key=lambda x: (x["rcept_dt"], x["rcept_no"]))


def main_xml(zip_bytes, rcept_no):
    """ZIP 안의 본문 XML. 본문은 {rcept_no}.xml, 첨부는 {rcept_no}_00760.xml 같은 이름."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        if not names:
            raise DartError("ZIP 안에 XML 파일이 없음")
        main = f"{rcept_no}.xml"
        name = main if main in names else max(names, key=lambda n: zf.getinfo(n).file_size)
        return to_utf8(zf.read(name))


def to_utf8(xml):
    """2021년 이전 일부 공시는 헤더는 utf-8인데 실제 내용은 EUC-KR(cp949). DB에는 항상 UTF-8로 저장."""
    try:
        xml.decode("utf-8")
        return xml
    except UnicodeDecodeError:
        try:
            return xml.decode("cp949").encode("utf-8")
        except UnicodeDecodeError as e:
            raise DartError(f"XML 인코딩 판별 실패 (utf-8/cp949 모두 아님): {e}")


def collect_one(con, client, company, year, force=False):
    company_id = db.upsert_company(con, company["corp_code"], company["name"], company["stock_code"], company["industry"])
    con.commit()
    label = f"{company['name']} {year}"

    existing = con.execute(
        "SELECT status FROM reports WHERE company_id=? AND business_year=?", (company_id, year)
    ).fetchone()
    if existing and existing["status"] == "OK" and not force:
        print(f"[SKIP] {label}: 이미 수집됨 (--force로 다시 받기)")
        return "SKIP"

    try:
        candidates = client.list_annual_reports(company["corp_code"], year)
        if not candidates:
            raise DartError("사업보고서 공시를 찾지 못함")
        rep = pick_latest(candidates)
        rcept_no = rep["rcept_no"]

        zip_bytes = client.download_document(rcept_no)
        ZIP_DIR.mkdir(parents=True, exist_ok=True)
        zip_file = ZIP_DIR / f"{rcept_no}.zip"
        zip_file.write_bytes(zip_bytes)
        xml = main_xml(zip_bytes, rcept_no)

        with db.tx(con):
            db.upsert_report(
                con, company_id, year,
                rcept_no=rcept_no, rcept_dt=rep["rcept_dt"], report_nm=rep["report_nm"].strip(),
                is_correction=int("정정" in rep["report_nm"]), status="OK", fail_reason=None,
                zip_path=zip_file.relative_to(ROOT).as_posix(), raw_xml=xml,
            )
        print(f"[OK]   {label}: {rcept_no} {rep['report_nm'].strip()} ({len(xml):,} bytes)")
        return "OK"
    except DartError as e:
        if existing and existing["status"] == "OK":
            # 기존 정상 데이터를 실패로 덮어쓰지 않음
            print(f"[FAIL] {label}: {e} (기존 OK 데이터 유지)")
        else:
            with db.tx(con):
                db.upsert_report(con, company_id, year, status="FAIL", fail_reason=str(e))
            print(f"[FAIL] {label}: {e}")
        return "FAIL"


def main(argv=None):
    settings = load_settings()
    p = argparse.ArgumentParser(description="OpenDART 사업보고서 수집")
    p.add_argument("--companies", nargs="+", help="기업명 (기본: 9개 전체)")
    p.add_argument("--years", nargs="+", type=int, default=settings["years"], help="사업연도 (기본: settings.yaml)")
    p.add_argument("--db", help="저장할 SQLite 경로 (기본: data/annual_reports.db)")
    p.add_argument("--force", action="store_true", help="이미 OK인 보고서도 다시 수집")
    args = p.parse_args(argv)

    companies = COMPANIES
    if args.companies:
        unknown = set(args.companies) - {c["name"] for c in COMPANIES}
        if unknown:
            p.error(f"알 수 없는 기업: {', '.join(sorted(unknown))}")
        companies = [c for c in COMPANIES if c["name"] in args.companies]

    client = DartClient(load_api_key(), settings.get("request_interval_sec", 0.5), settings.get("max_retries", 3))
    con = db.connect(args.db)
    counts = {"OK": 0, "FAIL": 0, "SKIP": 0}
    try:
        for company in companies:
            for year in args.years:
                counts[collect_one(con, client, company, year, args.force)] += 1
    finally:
        con.close()
    print(f"\n완료: OK {counts['OK']} / FAIL {counts['FAIL']} / SKIP {counts['SKIP']}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
