"""Promote selected structure-preserving experiment outputs to a shared dataset.

This deliberately publishes only the canonical paragraph-level input for
Milestone B. Full experiment outputs remain under data/experiments and ignored.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "data" / "experiments"
FILES = ("paragraphs.jsonl", "summary.json", "validation_report.csv")


def parse_sources(specifications: list[str]) -> dict[str, str]:
    """Parse repeatable ``industry=experiment_directory`` CLI arguments."""
    sources: dict[str, str] = {}
    for specification in specifications:
        industry, separator, experiment_name = specification.partition("=")
        if not separator or not industry or not experiment_name:
            raise ValueError(f"Invalid --source {specification!r}; use INDUSTRY=EXPERIMENT_NAME")
        if Path(experiment_name).name != experiment_name:
            raise ValueError(f"Experiment name must be a directory name: {experiment_name!r}")
        if industry in sources:
            raise ValueError(f"Duplicate industry: {industry}")
        sources[industry] = experiment_name
    return sources


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True, help="directory name under data/processed")
    parser.add_argument(
        "--source", action="append", required=True, metavar="INDUSTRY=EXPERIMENT_NAME",
        help="repeat for each industry to publish",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset directory")
    args = parser.parse_args()
    if Path(args.dataset_name).name != args.dataset_name:
        parser.error("--dataset-name must be a single directory name")
    try:
        sources = parse_sources(args.source)
    except ValueError as error:
        parser.error(str(error))

    target = ROOT / "data" / "processed" / args.dataset_name
    if target.exists() and any(target.iterdir()) and not args.overwrite:
        raise SystemExit(f"Dataset already exists: {target}. Use --overwrite to replace it.")
    if args.overwrite:
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    dataset_manifest = {
        "dataset": args.dataset_name,
        "canonical_method": "structure",
        "record_unit": "paragraph or table row",
        "text_field_for_milestone_b": "normalized_text",
        "sentence_field": "sentences",
        "industries": {},
    }

    for industry, experiment_name in sources.items():
        source = EXPERIMENTS / experiment_name
        destination = target / industry
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

    (target / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    industry_list = "\n".join(f"- `{industry}`: {details['report_count']} reports" for industry, details in dataset_manifest["industries"].items())
    (target / "README.md").write_text(
        f"""# Milestone B input dataset — {args.dataset_name}

This is the shared, verified input dataset for Milestone B.

- Canonical preprocessing method: **structure-preserving**
- Record unit: one paragraph or one table row
- TF-IDF and embedding input: `normalized_text`
- Sentence-level evidence: nested `sentences` array
- Provenance: `title_path` and `source_location`

Each industry directory contains:

- `paragraphs.jsonl`: canonical model input
- `input_reports.csv`: source reports and receipt numbers
- `summary.json`: extraction counts by analysis area
- `validation_report.csv`: report-level processing status

The larger experimental outputs, including tokenization variants and duplicated
flat sentence files, remain reproducible under `data/experiments/` and are not
part of this shared dataset.

## Included industries

{industry_list}
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
