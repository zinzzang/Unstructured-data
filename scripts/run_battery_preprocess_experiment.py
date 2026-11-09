"""Run the two preprocessing variants for one industry's nine reports.

The source database is read-only. Generated experiment artifacts are written to
data/experiments/<experiment-name>, which is intentionally gitignored.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess_semantic import build_semantic_records
from src.preprocess_structure import build_paragraph_records, write_jsonl


SOURCE_DB = ROOT / "data" / "annual_reports.db.gz"
DEFAULT_INDUSTRY = "2차전지"
DEFAULT_EXPERIMENT_NAME = "battery_preprocessing_v1"


def read_reports(industry: str) -> list[dict[str, object]]:
    """Read the selected XML documents from the compressed SQLite database."""
    fd, temporary_db = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        with gzip.open(SOURCE_DB, "rb") as compressed, open(temporary_db, "wb") as db_file:
            shutil.copyfileobj(compressed, db_file)
        connection = sqlite3.connect(temporary_db)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT c.name AS company, c.industry, r.business_year, r.rcept_no,
                      r.rcept_dt, r.report_nm, r.status, r.raw_xml
               FROM reports r JOIN companies c USING(company_id)
               WHERE c.industry=? AND r.status='OK' AND r.raw_xml IS NOT NULL
               ORDER BY c.name, r.business_year""",
            (industry,),
        ).fetchall()
        connection.close()
        return [dict(row) for row in rows]
    finally:
        Path(temporary_db).unlink(missing_ok=True)


def enrich(records: list[dict], report: dict[str, object]) -> list[dict]:
    """Add report metadata shared by all paragraph records."""
    for record in records:
        record.update(
            {
                "company": report["company"],
                "industry": report["industry"],
                "business_year": report["business_year"],
                "rcept_no": report["rcept_no"],
            }
        )
    return records


def flatten_sentences(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        for sentence in record["sentences"]:
            rows.append(
                {
                    "report_id": record["report_id"],
                    "paragraph_id": record["paragraph_id"],
                    "analysis_area": record["analysis_area"],
                    "company": record["company"],
                    "business_year": record["business_year"],
                    "rcept_no": record["rcept_no"],
                    "source_type": record["source_type"],
                    "title_path": record["title_path"],
                    **sentence,
                }
            )
    return rows


def summary(records: list[dict]) -> dict[str, object]:
    return {
        "paragraph_count": len(records),
        "sentence_count": sum(len(record["sentences"]) for record in records),
        "table_row_count": sum(record["source_type"] == "table_row" for record in records),
        "duplicate_text_count": sum("duplicate_text" in record["quality_flags"] for record in records),
        "analysis_area_counts": dict(sorted(Counter(record["analysis_area"] for record in records).items())),
        "unmapped_count": sum(record["analysis_area"] == "UNMAPPED" for record in records),
    }


def write_validation(path: Path, rows: list[dict]) -> None:
    fields = [
        "method", "company", "business_year", "rcept_no", "status", "paragraph_count",
        "sentence_count", "table_row_count", "duplicate_text_count", "unmapped_count", "error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(path: Path, structure_records: list[dict], semantic_records: list[dict]) -> None:
    structure_ids = {record["paragraph_id"] for record in structure_records}
    semantic_ids = {record["paragraph_id"] for record in semantic_records}
    rows = [
        {"metric": "paragraph_count", "structure": len(structure_records), "semantic": len(semantic_records)},
        {"metric": "sentence_count", "structure": sum(len(x["sentences"]) for x in structure_records), "semantic": sum(len(x["sentences"]) for x in semantic_records)},
        {"metric": "shared_paragraph_ids", "structure": len(structure_ids & semantic_ids), "semantic": len(structure_ids & semantic_ids)},
        {"metric": "only_structure_ids", "structure": len(structure_ids - semantic_ids), "semantic": 0},
        {"metric": "only_semantic_ids", "structure": 0, "semantic": len(semantic_ids - structure_ids)},
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=["metric", "structure", "semantic"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--industry", default=DEFAULT_INDUSTRY, help="exact companies.industry value")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME, help="subdirectory of data/experiments")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing experiment directory")
    args = parser.parse_args()
    output_root = ROOT / "data" / "experiments" / args.experiment_name

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_root}. Use --overwrite to replace it.")
    if args.overwrite:
        shutil.rmtree(output_root, ignore_errors=True)

    reports = read_reports(args.industry)
    if len(reports) != 9:
        raise SystemExit(f"Expected 9 {args.industry} reports, found {len(reports)}.")

    directories = {name: output_root / name for name in ("structure", "semantic", "comparison")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    input_rows, validation = [], []
    structure_all, semantic_all = [], []
    for report in reports:
        xml_bytes = bytes(report.pop("raw_xml"))
        report_id = str(report["rcept_no"])
        input_rows.append({**report, "xml_sha256": hashlib.sha256(xml_bytes).hexdigest(), "xml_bytes": len(xml_bytes)})

        for method, builder, aggregate in (
            ("structure", build_paragraph_records, structure_all),
            ("semantic", build_semantic_records, semantic_all),
        ):
            try:
                records = enrich(builder(xml_bytes, report_id), report)
                aggregate.extend(records)
                report_summary = summary(records)
                validation.append(
                    {
                        "method": method, "company": report["company"], "business_year": report["business_year"],
                        "rcept_no": report["rcept_no"], "status": "OK", "error": "", **report_summary,
                    }
                )
            except Exception as error:  # Preserve other reports and make the failure visible.
                validation.append(
                    {
                        "method": method, "company": report["company"], "business_year": report["business_year"],
                        "rcept_no": report["rcept_no"], "status": "FAIL", "error": f"{type(error).__name__}: {error}",
                        "paragraph_count": 0, "sentence_count": 0, "table_row_count": 0,
                        "duplicate_text_count": 0, "unmapped_count": 0,
                    }
                )

    with (output_root / "input_reports.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(input_rows[0]))
        writer.writeheader()
        writer.writerows(input_rows)

    for method, records in (("structure", structure_all), ("semantic", semantic_all)):
        directory = directories[method]
        write_jsonl(records, directory / "paragraphs.jsonl")
        write_jsonl(flatten_sentences(records), directory / "sentences.jsonl")
        write_validation(directory / "validation_report.csv", [row for row in validation if row["method"] == method])
        (directory / "summary.json").write_text(json.dumps(summary(records), ensure_ascii=False, indent=2), encoding="utf-8")

    write_comparison(directories["comparison"] / "record_count_comparison.csv", structure_all, semantic_all)
    semantic_by_id = {record["paragraph_id"]: record for record in semantic_all}
    samples = []
    for structure_record in structure_all[:50]:
        semantic_record = semantic_by_id[structure_record["paragraph_id"]]
        samples.append(
            {
                "paragraph_id": structure_record["paragraph_id"], "company": structure_record["company"],
                "business_year": structure_record["business_year"], "analysis_area": structure_record["analysis_area"],
                "normalized_text": structure_record["normalized_text"], "tokens": semantic_record["tokens"],
                "comparison_text": semantic_record["comparison_text"], "tokenization_method": semantic_record["tokenization_method"],
            }
        )
    write_jsonl(samples, directories["comparison"] / "sample_comparison.jsonl")

    manifest = {
        "experiment": output_root.name,
        "created_at_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "source_database": str(SOURCE_DB.relative_to(ROOT)),
        "industry": args.industry,
        "report_count": len(reports),
        "methods": ["structure", "semantic"],
        "semantic_default": "regex_no_lemmatization",
        "code_sha256": {
            "preprocess_structure.py": hashlib.sha256((ROOT / "src" / "preprocess_structure.py").read_bytes()).hexdigest(),
            "preprocess_semantic.py": hashlib.sha256((ROOT / "src" / "preprocess_semantic.py").read_bytes()).hexdigest(),
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_root), "structure": summary(structure_all), "semantic": summary(semantic_all)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
