"""규칙 기반 변화 분류. classify(old, new) -> dict. 같은 시그니처의 함수로 LLM 분류기로 교체 가능(change_detector에 인자로 전달).
원칙: 근거 부족하면 UNCERTAIN. 수치 증감은 지표(문장 틀)·단위가 같을 때만 EXPANDED/REDUCED."""
import re
from difflib import SequenceMatcher

UNIT = r"(?:백만|조|억|만|천)?\s?(?:원|개|명|건|배|톤|%p|%|GB|TB|nm|Gbps|MHz|GHz|MW|GWh|kWh|W|달러|곳|사|회)?"
NUM = re.compile(rf"(?<![\w.])(\d{{1,3}}(?:,\d{{3}})+|\d+)(\.\d+)?({UNIT})(?!\s?(?:년|월|일|기\b|분기|차\b|호\b|세대|단계|층))")
PERIOD = re.compile(r"제\s*\d+\s*기|\d{4}\s*년|당기|전기")      # 표의 기간 열: 해마다 한 칸씩 밀리므로 첫 값(당해 값)만 비교
PLAN = re.compile(r"예정|계획|전망|목표|할 것으로|예상")
NEG = re.compile(r"않|없|아니|불가|중단|종료|철수|폐지")


def numbers(text):
    out = [(float(m.group(1).replace(",", "") + (m.group(2) or "")), re.sub(r"\s", "", m.group(3) or "")) for m in NUM.finditer(text)]
    return out[:1] if PERIOD.search(text) and out else out


def masked(text):
    return NUM.sub("#", text)


def classify(old, new):
    no, nn = numbers(old), numbers(new)
    r = SequenceMatcher(None, masked(old), masked(new), autojunk=False).ratio()  # 수치를 가린 문장 틀의 유사도
    out = lambda t, c, why, summary=None: dict(change_type=t, confidence=round(c, 2), reason=why, change_summary=summary, template_sim=round(r, 3))
    if bool(PLAN.search(old)) != bool(PLAN.search(new)) and r < 0.97:
        return out("UNCERTAIN", 0.3, "계획/실적 표현이 한쪽에만 있음")
    if no != nn:
        if len(no) == len(nn) and r >= 0.8 and all(a[1] == b[1] for a, b in zip(no, nn)):
            ups = [b[0] > a[0] for a, b in zip(no, nn) if a[0] != b[0]]
            t = "EXPANDED" if all(ups) else "REDUCED" if not any(ups) else "MODIFIED"
            return out(t, 0.9, "동일 문장 틀·동일 단위에서 수치 변경",
                       "수치 변경: " + ", ".join(f"{a[0]:,.10g}{a[1]} → {b[0]:,.10g}{b[1]}" for a, b in zip(no, nn) if a != b))
        return out("UNCERTAIN", 0.3, "수치 개수·단위·문장 틀이 달라 같은 지표인지 확인 불가")
    if r >= 0.9:
        return out("UNCHANGED", r, "수치 동일, 문장 틀 거의 동일")
    if len(NEG.findall(old)) != len(NEG.findall(new)):
        return out("MODIFIED", 0.5, "부정/중단 표현 개수가 달라짐")
    return out("UNCERTAIN", 0.3, "표현 변경 폭이 커서 규칙으로 판단 불가(LLM/사람 검토 대상)")
