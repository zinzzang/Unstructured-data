"""의미 정규화형 전처리 실험군.

구조 보존형 레코드에 비교 전용 ``tokens``와 ``comparison_text``를 추가한다.
형태소 분석기는 의존성으로 강제하지 않는다. 분석기를 주입하면 그 분석기의
``morphs(text, stem=True)``를 사용하고, 없으면 원형화하지 않는 정규식 토큰화로
명시적으로 fallback 한다.
"""
from __future__ import annotations

import re
from typing import Any, Protocol

from .preprocess_structure import build_paragraph_records


TOKEN = re.compile(r"[가-힣]+|[A-Za-z]+(?:[&+._-][A-Za-z0-9]+)*|\d+(?:[,.]\d+)*(?:%|원|개|명|톤|t|억)?")


class MorphAnalyzer(Protocol):
    def morphs(self, phrase: str, norm: bool = True, stem: bool = True) -> list[str]: ...


def tokenize(text: str, *, analyzer: MorphAnalyzer | None = None, stopwords: set[str] | None = None) -> tuple[list[str], str]:
    """토큰과 사용 방법을 반환한다. 기본값은 stop-word를 제거하지 않는다."""
    if analyzer is None:
        tokens = TOKEN.findall(text)
        method = "regex_no_lemmatization"
    else:
        tokens = [token for token in analyzer.morphs(text, norm=True, stem=True) if token.strip()]
        method = f"{type(analyzer).__name__}_stem"

    if stopwords:
        tokens = [token for token in tokens if token not in stopwords]
        method += "+custom_stopwords"
    return tokens, method


def build_semantic_records(
    xml_bytes: bytes,
    report_id: str,
    *,
    analyzer: MorphAnalyzer | None = None,
    stopwords: set[str] | None = None,
) -> list[dict[str, Any]]:
    """구조 보존형 결과에 선택적 형태소 토큰·비교 텍스트를 추가한다."""
    records = build_paragraph_records(xml_bytes, report_id)
    for record in records:
        tokens, method = tokenize(record["normalized_text"], analyzer=analyzer, stopwords=stopwords)
        record["tokens"] = tokens
        record["comparison_text"] = " ".join(tokens)
        record["tokenization_method"] = method
    return records
