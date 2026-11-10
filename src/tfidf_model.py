import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 금액·비율은 사업 변화의 핵심 근거이므로 TF-IDF 입력에서도 보존한다.
# 나머지 숫자(문단 번호, 연도, 생산 대수 등)는 NUM으로 통일한다.
PROTECTED_NUMBER = re.compile(
    r"(?<![\w])"
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"\s*(?:(?:백만|조|억|만|천)?\s*원|%p?|퍼센트)"
)
GENERAL_NUMBER = re.compile(
    r"(?<![\d가-힣A-Za-z])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?!\d)"
)


def _placeholder(index: int) -> str:
    """숫자를 포함하지 않는 임시 토큰. 이후 일반 숫자 마스킹에서 안전하다."""
    letters = []
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "⟪KEEP" + "".join(reversed(letters)) + "⟫"


def mask_numbers(text: str, token: str = "NUM") -> str:
    """금액·비율을 보존하고 그 외 아라비아 숫자를 ``token``으로 치환한다.

    이 함수는 TF-IDF 벡터화 입력에만 사용한다. 결과 CSV와 변화 유형 분류에는
    원문을 계속 사용해 실제 수치 변화가 사라지지 않게 한다.
    """
    protected: list[str] = []

    def keep(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return _placeholder(len(protected) - 1)

    masked = PROTECTED_NUMBER.sub(keep, text)
    masked = GENERAL_NUMBER.sub(token, masked)
    for index, original in enumerate(protected):
        masked = masked.replace(_placeholder(index), original)
    return masked


def similarity(old, new, cfg):
    """old x new 코사인 유사도. 벡터라이저는 해당 기업·연도쌍의 문장(old+new)으로만 학습 -> gold 라벨 정보 누출 없음."""
    masking = cfg["tfidf"].get("number_masking", {})
    if masking.get("enabled", False):
        token = masking.get("token", "NUM")
        old = [mask_numbers(text, token) for text in old]
        new = [mask_numbers(text, token) for text in new]
    v = TfidfVectorizer(analyzer=cfg["tfidf"]["analyzer"], ngram_range=tuple(cfg["tfidf"]["ngram_range"]))
    v.fit(old + new)
    return cosine_similarity(v.transform(old), v.transform(new))
