from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def similarity(old, new, cfg):
    """old x new 코사인 유사도. 벡터라이저는 해당 기업·연도쌍의 문장(old+new)으로만 학습 -> gold 라벨 정보 누출 없음."""
    v = TfidfVectorizer(analyzer=cfg["tfidf"]["analyzer"], ngram_range=tuple(cfg["tfidf"]["ngram_range"]))
    v.fit(old + new)
    return cosine_similarity(v.transform(old), v.transform(new))
