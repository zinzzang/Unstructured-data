from src.tfidf_model import mask_numbers


def test_masks_section_numbers_dates_and_general_quantities():
    assert mask_numbers("(5) 온실가스 감축") == "(NUM) 온실가스 감축"
    assert mask_numbers("2024년 사업보고서") == "NUM년 사업보고서"
    assert mask_numbers("생산량 500대, 공정 7nm") == "생산량 NUM대, 공정 NUMnm"


def test_preserves_money_and_percentages():
    assert mask_numbers("매출액 150억원") == "매출액 150억원"
    assert mask_numbers("투자금 1,200 백만원") == "투자금 1,200 백만원"
    assert mask_numbers("영업이익 -2.5% 증가, 수익률 30 %") == "영업이익 -2.5% 증가, 수익률 30 %"


def test_masks_non_financial_decimal_but_keeps_financial_decimal():
    assert mask_numbers("규격 3.5mm") == "규격 NUMmm"
    assert mask_numbers("비용은 3.5억원입니다") == "비용은 3.5억원입니다"
