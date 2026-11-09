"""Promote verified structure-preserving experiment outputs to the shared dataset.

This deliberately publishes only the canonical paragraph-level input for
Milestone B. Full experiment outputs remain under data/experiments and ignored.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "data" / "experiments"
TARGET = ROOT / "data" / "processed" / "preprocessing_v1"
SOURCES = {
    "semiconductor": "semiconductor_preprocessing_v1",
    "battery": "battery_preprocessing_v1",
    "biopharma": "biopharma_preprocessing_v1",
}
FILES = ("paragraphs.jsonl", "summary.json", "validation_report.csv")


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    dataset_manifest = {
        "dataset": "preprocessing_v1",
        "canonical_method": "structure",
        "record_unit": "paragraph or table row",
        "text_field_for_milestone_b": "normalized_text",
        "sentence_field": "sentences",
        "industries": {},
    }

    for industry, experiment_name in SOURCES.items():
        source = EXPERIMENTS / experiment_name
        destination = TARGET / industry
        if not source.is_dir():
            raise FileNotFoundError(f"Missing verified experiment: {source}")
        destination.mkdir(parents=True, exist_ok=True)
        for filename in FILES:
            shutil.copy2(source / "structure" / filename, destination / filename)
        shutil.copy2(source / "input_reports.csv", destination / "input_reports.csv")
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        dataset_manifest["industries"][industry] = {
            "source_experiment": experiment_name,
            "report_count": manifest["report_count"],
            "source_database": manifest["source_database"],
            "code_sha256": manifest["code_sha256"],
        }

    (TARGET / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (TARGET / "README.md").write_text(
        """# Milestone B input dataset — preprocessing_v1

This is the shared, verified input dataset for Milestone B.

- Canonical preprocessing method: **structure-preserving**
- Record unit: one paragraph or one table row
- TF-IDF and embedding input: `normalized_text`
- Sentence-level evidence: nested `sentences` array
- Provenance: `title_path` and `source_location`

Each industry directory contains:

- `paragraphs.jsonl`: canonical model input
- `input_reports.csv`: nine source reports and receipt numbers
- `summary.json`: extraction counts by analysis area
- `validation_report.csv`: report-level processing status

The larger experimental outputs, including tokenization variants and duplicated
flat sentence files, remain reproducible under `data/experiments/` and are not
part of this shared dataset.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
