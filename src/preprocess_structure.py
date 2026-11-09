"""구조 보존형 전처리 실험군.

기존 ``preprocess.py``를 바꾸지 않고, 연도 간 문단 매칭에 필요한 JSONL
레코드를 만든다. 원본 XML은 reports.raw_xml에 그대로 보관한다. 이 모듈의
``source_text``는 XML에서 추출한 텍스트이고, ``normalized_text``는 비교용으로
공백·제어문자만 정리한 텍스트다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from .report_parser import extract_chapter2


WHITESPACE = re.compile(r"\s+")
NUMBERING_ONLY = re.compile(r"^(?:\(?\d{1,3}[.)]?|[가-힣][.)])$")
PAGE_NUMBER_ONLY = re.compile(r"^-?\s*\d{1,4}\s*-?$")
# 마침표는 한글·영문·% 뒤일 때만 문장 경계로 간주한다. 따라서 3.5와
# 2025. 3. 19.처럼 숫자 뒤의 점은 보존하면서, "입니다. 2025년"은 나눈다.
SENTENCE_BOUNDARY = re.compile(r"(?:(?<=[가-힣A-Za-z%]\.)|(?<=[!?]))\s+")
TOKENISH_SEPARATOR = re.compile(r";\s+")
SECTION_MAPPING_PATH = Path(__file__).resolve().parent.parent / "config" / "section_mapping.yaml"
_SECTION_MAPPING: dict[str, list[str]] | None = None


def normalize_text(text: str) -> str:
    """의미를 삭제하지 않는 최소 정제. 수치·단위·부정 표현은 보존한다."""
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    return WHITESPACE.sub(" ", text).strip()


def is_noise(text: str) -> bool:
    """분석 단위로 쓸 수 없는 빈 문자열·번호·쪽수 표기만 제거한다."""
    return not text or bool(NUMBERING_ONLY.fullmatch(text)) or bool(PAGE_NUMBER_ONLY.fullmatch(text))


def classify_title(title: str) -> str:
    """현재 section_mapping.yaml을 UTF-8로 읽어 새 파이프라인에 적용한다."""
    global _SECTION_MAPPING
    if _SECTION_MAPPING is None:
        _SECTION_MAPPING = yaml.safe_load(SECTION_MAPPING_PATH.read_text(encoding="utf-8"))
    for area, keywords in _SECTION_MAPPING.items():
        if any(keyword in title for keyword in keywords):
            return area
    return "UNMAPPED"


def split_sentences(text: str, *, long_text_limit: int = 400) -> list[str]:
    """한국어 보고서용 보수적 문장 분리.

    문장 부호 뒤 다음 토큰이 숫자면 분리하지 않아 ``3.5``와 날짜 표기를 보존한다.
    지나치게 긴 단위만 세미콜론을 보조 경계로 사용한다.
    """
    units: list[str] = []
    for sentence in SENTENCE_BOUNDARY.split(normalize_text(text)):
        parts = TOKENISH_SEPARATOR.split(sentence) if len(sentence) > long_text_limit else [sentence]
        units.extend(normalize_text(part) for part in parts)
    return [unit for unit in units if not is_noise(unit)]


def table_row_text(label: str, columns: list[tuple[str, str]], unit: str) -> str:
    values = "; ".join(f"{header}: {value}" for header, value in columns)
    pieces = [normalize_text(unit)]
    if label:
        pieces.append(f"{normalize_text(label)} |")
    if values:
        pieces.append(values)
    return " ".join(piece for piece in pieces if piece)


def _title_path(section_title: str, subsection_title: str) -> list[str]:
    path = ["II. 사업의 내용", section_title]
    if subsection_title:
        path.append(subsection_title)
    return path


def build_paragraph_records(xml_bytes: bytes, report_id: str) -> list[dict[str, Any]]:
    """원본 XML 하나를 추적 가능한 문단/표 행 레코드로 변환한다.

    ``paragraph_id``는 동일 XML·동일 파서 규칙에서 재현 가능하다. XML이 바뀌면
    ``report_id``도 새 접수번호 기반 값으로 바꾸는 것을 전제로 한다.
    """
    records: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    paragraph_order = 0

    for section_order, section in enumerate(extract_chapter2(xml_bytes), start=1):
        area = classify_title(section["section_title"])
        paragraph_item_order = 0
        table_item_order = 0

        for item_order, item in enumerate(section["items"], start=1):
            if item["kind"] == "paragraph":
                paragraph_item_order += 1
                candidates = [(None, item["text"], None)]
                source_type = "paragraph"
                source_unit = f"P{paragraph_item_order:03d}"
            else:
                table_item_order += 1
                candidates = [
                    (row_order, table_row_text(label, columns, item["unit"]), {"label": label, "columns": columns, "unit": item["unit"]})
                    for row_order, (label, columns) in enumerate(item["rows"], start=1)
                ]
                source_type = "table_row"
                source_unit = f"T{table_item_order:03d}"

            for row_order, source_text, table in candidates:
                source_text = normalize_text(source_text)
                if is_noise(source_text):
                    continue
                paragraph_order += 1
                paragraph_id = f"{report_id}-P{paragraph_order:04d}"
                normalized_text = source_text
                is_duplicate = normalized_text in seen_texts
                seen_texts.add(normalized_text)
                records.append(
                    {
                        "report_id": report_id,
                        "paragraph_id": paragraph_id,
                        "paragraph_order": paragraph_order,
                        "analysis_area": area,
                        "title_path": _title_path(section["section_title"], item["sub"]),
                        "source_type": source_type,
                        "source_text": source_text,
                        "normalized_text": normalized_text,
                        "sentences": [
                            {
                                "sentence_id": f"{paragraph_id}-S{sentence_order:02d}",
                                "sentence_order": sentence_order,
                                "text": sentence,
                            }
                            for sentence_order, sentence in enumerate(split_sentences(normalized_text), start=1)
                        ],
                        "source_location": {
                            "xml_line": item["line"],
                            "section_order": section_order,
                            "item_order": item_order,
                            "source_unit": source_unit,
                            "table_row_order": row_order,
                        },
                        "table": table,
                        "quality_flags": ["duplicate_text"] if is_duplicate else [],
                    }
                )
    return records


def write_jsonl(records: Iterable[dict[str, Any]], output_path: str | Path) -> None:
    """중간 산출물 검토용 JSONL을 UTF-8로 쓴다."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
