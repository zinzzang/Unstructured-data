"""데이터셋 개요 그림 2개 (발표 Slide 05).

사용: python scripts/plots/plot_dataset_overview.py
입력: data/processed/preprocessing_v2/{industry}/paragraphs.jsonl  (normalized_text, 2021~2025)
출력: figures/fig_term_document_matrix.png/.svg   SK하이닉스 연도별 주요 단어 등장 횟수 (Term-Document Matrix)
      figures/fig_char_counts.png/.svg            9개 기업 연도별 II장 텍스트 글자 수 (토큰화 전)
      figures/dataset_overview.csv                위 두 그림의 집계값
"""
import csv
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from plot_industry_change_flow import COMPANY_EN, DPI, FIG_DIR, INDUSTRIES, ROOT, setup_style

DATA = ROOT / 'data/processed/preprocessing_v2'
YEARS = [2021, 2022, 2023, 2024, 2025]
TDM_COMPANY = 'SK하이닉스'
# (표시 이름, normalized_text에서 세는 문자열) — 부분 문자열 등장 횟수
TERMS = [('HBM', 'HBM'), ('AI', 'AI'), ('DDR5', 'DDR5'), ('NAND', 'NAND'), ('DRAM', 'D램'),
         ('Memory', '메모리'), ('Customer', '고객'), ('Product', '제품'), ('Server', '서버'),
         ('Production', '생산'), ('Investment', '투자'), ('Sustainability', '친환경')]
LINE_COLORS = {'셀트리온': '#1f3b73', '한미반도체': '#c8553d', 'LG에너지솔루션': '#2e7d4f', 'SK이노베이션': '#8e5bb5',
               'SK하이닉스': '#c9a227', '삼성바이오로직스': '#5b7f95', '삼성전자': '#b5476b', '삼성SDI': '#6b7280',
               '유한양행': '#8a7a2e'}


def load():
    """{(company, year): 문단 normalized_text 목록}"""
    text = defaultdict(list)
    for ind in INDUSTRIES:
        with (DATA / ind / 'paragraphs.jsonl').open(encoding='utf-8') as f:
            for line in f:
                p = json.loads(line)
                text[(p['company'], p['business_year'])].append(p['normalized_text'])
    return text


def save(fig, name):
    for fmt in ('png', 'svg'):
        fig.savefig(FIG_DIR / f'{name}.{fmt}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'[저장] figures/{name}.png, .svg')


def fig_tdm(text):
    m = np.array([[sum(t.count(q) for t in text[(TDM_COMPANY, y)]) for y in YEARS] for _, q in TERMS])
    fig, ax = plt.subplots(figsize=(7, 6.4))
    cmap = LinearSegmentedColormap.from_list('tdm', ['#f4f6fb', '#aebbd8', '#5a6fa8', '#1f3b80'])
    ax.imshow(m, cmap=cmap, vmin=0, vmax=m.max(), aspect='auto')
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(TERMS)):
        for j in range(len(YEARS)):
            v = m[i, j]
            ax.text(j, i, str(v), ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white' if v >= m.max() * 0.6 else '#1f2a44')
    ax.set_xticks(range(len(YEARS)), YEARS, fontsize=12, fontweight='bold')
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(TERMS)), [t for t, _ in TERMS], fontsize=12, fontweight='bold')
    ax.set_xticks(np.arange(-0.5, len(YEARS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(TERMS)), minor=True)
    ax.grid(which='minor', color='white', lw=4)
    ax.tick_params(which='both', length=0)
    save(fig, 'fig_term_document_matrix')
    return m


def fig_chars(text):
    companies = sorted(c for _, cs in INDUSTRIES.values() for c in cs)
    companies.sort(key=lambda c: COMPANY_EN.get(c, c))
    fig, ax = plt.subplots(figsize=(11, 6))
    for c in companies:
        ax.plot(YEARS, [sum(map(len, text[(c, y)])) for y in YEARS], color=LINE_COLORS[c], lw=2, marker='o', ms=5,
                label=COMPANY_EN.get(c, c))
    ax.set_xticks(YEARS)
    ax.set_xlabel('Business year')
    ax.set_ylabel('Characters (before preprocessing)')
    ax.set_ylim(0, 200_000)
    ax.yaxis.set_major_formatter(lambda v, _: f'{v / 1000:.0f}k')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=10)
    save(fig, 'fig_char_counts')
    return {c: [sum(map(len, text[(c, y)])) for y in YEARS] for c in companies}


def main():
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    text = load()
    m = fig_tdm(text)
    chars = fig_chars(text)
    with (FIG_DIR / 'dataset_overview.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['figure', 'row', *YEARS])
        for (label, q), row in zip(TERMS, m):
            w.writerow([f'term_document_matrix ({TDM_COMPANY})', f'{label} ({q})', *row])
        for c, row in chars.items():
            w.writerow(['char_counts', c, *row])


if __name__ == '__main__':
    main()
