from src.change_classifier import classify as c


def t(old, new):
    return c(old, new)["change_type"]


def test_number_increase_is_expanded_even_if_wording_similar():
    r = c("생산능력은 연간 100만 개이다.", "생산능력은 연간 150만 개이다.")
    assert r["change_type"] == "EXPANDED" and "100만개 → 150만개" in r["change_summary"]


def test_decrease_and_unit_mismatch():
    assert t("매출 비중은 30%이다.", "매출 비중은 20%이다.") == "REDUCED"
    assert t("투자 규모는 1조원이다.", "투자 규모는 9,000억원이다.") == "UNCERTAIN"  # 단위 다름 -> 확정 안 함


def test_same_numbers_same_wording_unchanged():
    assert t("당사는 DRAM 사업을 영위한다.", "당사는 DRAM 사업을 영위한다.") == "UNCHANGED"


def test_plan_vs_actual_is_uncertain():
    assert t("2025년에 양산하였다.", "2025년에 양산할 예정이다.") == "UNCERTAIN"


def test_period_labels_shift_compares_first_value_only():
    assert t("반도체 | 제77기: 35,232,737; 제76기: 34,806,295", "반도체 | 제78기: 41,216,512; 제77기: 35,232,737") == "EXPANDED"
