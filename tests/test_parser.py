import pytest
from src.report_parser import extract_chapter2, ParseError
from src.preprocess import split_sentences

XML = """<DOCUMENT><BODY>
<TITLE ATOC="N">목 차</TITLE>
<TABLE><TBODY><TR><TD>II. 사업의 내용</TD></TR></TBODY></TABLE>
<SECTION-1><TITLE>I. 회사의 개요</TITLE><SECTION-2><TITLE>1. 개요</TITLE><P>제외될 문장.</P></SECTION-2></SECTION-1>
<SECTION-1><TITLE>II. 사업의 내용</TITLE><LIBRARY>
<SECTION-2><TITLE>3. 원재료 및 생산설비</TITLE>
<P USERMARK="B">가. 생산능력</P>
<P>생산능력은 연간 100만 개이다. S&P 평가는 3.5배이다.</P>
<TABLE><TBODY><TR><TD>(단위: 개)</TD></TR></TBODY></TABLE>
<TABLE><THEAD><TR><TH>구분</TH><TH>2024</TH><TH>2025</TH></TR></THEAD>
<TBODY><TR><TD ROWSPAN="2">DRAM</TD><TD>10</TD><TD>20</TD></TR><TR><TD>11</TD><TD>21</TD></TR></TBODY></TABLE>
</SECTION-2></LIBRARY></SECTION-1>
<SECTION-1><TITLE>III. 재무에 관한 사항</TITLE><SECTION-2><TITLE>1. 요약</TITLE><P>제외될 재무 문장.</P></SECTION-2></SECTION-1>
</BODY></DOCUMENT>""".encode()


def test_only_chapter2():
    secs = extract_chapter2(XML)
    assert [s["section_title"] for s in secs] == ["3. 원재료 및 생산설비"]
    items = secs[0]["items"]
    assert items[0]["sub"] == "가. 생산능력" and "S&P" in items[0]["text"]
    t = items[1]
    assert t["kind"] == "table" and t["unit"] == "(단위: 개)"
    assert t["rows"][1] == ("DRAM", [("2024", "11"), ("2025", "21")])  # rowspan 유지


def test_missing_chapter2_fails():
    with pytest.raises(ParseError):
        extract_chapter2("<DOCUMENT><SECTION-1><TITLE>I. 회사의 개요</TITLE></SECTION-1></DOCUMENT>".encode())


def test_split_keeps_decimal():
    assert split_sentences("생산능력은 연간 100만 개이다. 비율은 3.5배이다.") == ["생산능력은 연간 100만 개이다.", "비율은 3.5배이다."]


NESTED = """<DOCUMENT><SECTION-1><TITLE>II. 사업의 내용</TITLE>
<SECTION-2><TITLE>1. 사업의 개요</TITLE><P>포함될 문장.</P>
<SECTION-1><TITLE>III. 재무에 관한 사항</TITLE><SECTION-2><TITLE>1. 요약재무</TITLE><P>제외될 재무 문장.</P></SECTION-2></SECTION-1>
</SECTION-2></SECTION-1></DOCUMENT>""".encode()


def test_chapter_end_even_if_next_chapter_is_nested():
    secs = extract_chapter2(NESTED)
    assert [s["section_title"] for s in secs] == ["1. 사업의 개요"]
    assert [i["text"] for i in secs[0]["items"]] == ["포함될 문장."]
