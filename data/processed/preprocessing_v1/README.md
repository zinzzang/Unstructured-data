# Milestone B input dataset — preprocessing_v1

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
