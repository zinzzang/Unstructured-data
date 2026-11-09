from src.preprocess_semantic import build_semantic_records
from src.preprocess_structure import build_paragraph_records, split_sentences


XML = """<DOCUMENT><SECTION-1><TITLE>II. 사업의 내용</TITLE>
<SECTION-2><TITLE>1. 사업의 개요</TITLE>
<P USERMARK="B">가. 생산능력</P>
<P>생산능력은 3.5배 증가하였습니다. 2025. 3. 19. 기준입니다.</P>
<TABLE><THEAD><TR><TH>구분</TH><TH>2025</TH></TR></THEAD>
<TBODY><TR><TD>DRAM</TD><TD>20</TD></TR></TBODY></TABLE>
</SECTION-2></SECTION-1></DOCUMENT>""".encode()


def test_structure_records_preserve_provenance_and_table_fields():
    records = build_paragraph_records(XML, "demo-2025")
    assert len(records) == 2
    assert records[0]["analysis_area"] == "BUSINESS"
    assert records[0]["paragraph_id"] == "demo-2025-P0001"
    assert records[0]["sentences"][0]["text"] == "생산능력은 3.5배 증가하였습니다."
    assert records[1]["source_type"] == "table_row"
    assert records[1]["table"]["columns"] == [("2025", "20")]


def test_sentence_split_preserves_decimal_and_date():
    assert split_sentences("수치는 3.5입니다. 2025. 3. 19. 기준입니다.") == [
        "수치는 3.5입니다.",
        "2025. 3. 19. 기준입니다.",
    ]


def test_semantic_records_add_comparison_fields_without_changing_source_text():
    records = build_semantic_records(XML, "demo-2025")
    assert records[0]["source_text"] == records[0]["normalized_text"]
    assert records[0]["tokens"]
    assert records[0]["comparison_text"]
    assert records[0]["tokenization_method"] == "regex_no_lemmatization"
