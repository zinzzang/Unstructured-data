"""EDA·변화 유형 분포 그림 (발표 Slide 07·08·12용, 정적 이미지).

사용: python scripts/plot_eda.py
입력: data/results_v2/tfidf_masked_th0.3.csv 전체 (행 = 인접 연도 문장 쌍, 섹션 제외 없음)
출력: figures/eda_company_rows.png/.svg   Slide 07 ① 기업별 문장 쌍 수 (산업별 색)
      figures/eda_section_rows.png/.svg   Slide 07 ② 섹션(area)별 문장 쌍 수 (RISK = LDA 제외, 빗금)
      figures/eda_change_types.png/.svg   Slide 08 ① / 12 ① 7개 변화 유형 건수·비율
      figures/eda_counts.csv              위 세 그림의 집계값
"""
import csv
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_industry_change_flow import COLORS, COMPANY_EN, DPI, FIG_DIR, INDUSTRIES, INPUT_CSV, setup_style

FIGSIZE = (8, 5)          # 슬라이드 절반 영역용
INDUSTRY_COLORS = {'semiconductor': '#2a78d6', 'battery': '#3a9a5b', 'biopharma': '#c47a1e'}
INDUSTRY_ORDER = ['semiconductor', 'battery', 'biopharma']
TYPE_ORDER = ['UNCHANGED', 'NEW', 'EXPANDED', 'REDUCED', 'MODIFIED', 'REMOVED', 'UNCERTAIN']
TYPE_COLORS = dict(COLORS, UNCHANGED='#D9D7CF', UNCERTAIN='#6f6e69')
AREA_EN = {'BUSINESS': 'Business Overview', 'PRODUCTION': 'Production & Facilities',
           'SALES': 'Sales Status', 'PRODUCT': 'Products',
           'RND_CONTRACT': 'R&D & Contracts', 'OTHER': 'Other Business', 'RISK': 'Risk Management'}
EXCLUDED_AREAS = {'RISK'}
NOTE = 'tfidf_masked_th0.3.csv · {n:,} rows (1 row = 1 adjacent-year sentence pair)'


def load():
    rows = list(csv.DictReader(INPUT_CSV.open(encoding='utf-8-sig')))
    return (rows, Counter(r['company'] for r in rows), Counter(r['area'] for r in rows),
            Counter(r['change_type'] for r in rows))


def style_hbar(ax):
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color='#000000', alpha=0.07)
    ax.tick_params(axis='y', length=0, pad=6, labelsize=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.invert_yaxis()


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ('png', 'svg'):
        fig.savefig(FIG_DIR / f'{name}.{fmt}', dpi=DPI)
    plt.close(fig)
    print(f'[저장] figures/{name}.png, .svg')


def fig_company(company, n):
    items = [(k, c) for k in INDUSTRY_ORDER for c in INDUSTRIES[k][1]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.24, right=0.93, top=0.86, bottom=0.14)
    vmax = max(company.values())
    for i, (k, c) in enumerate(items):
        ax.barh(i, company[c], 0.68, color=INDUSTRY_COLORS[k], lw=0)
        ax.text(company[c] + vmax * 0.01, i, f'{company[c]:,}', va='center', fontsize=11, color='#222222')
    ax.set_yticks(range(len(items)), [COMPANY_EN.get(c, c) for _, c in items])
    ax.set_xlim(0, vmax * 1.12)
    ax.set_xlabel('Sentence pairs', fontsize=12)
    style_hbar(ax)
    fig.legend(handles=[Patch(color=INDUSTRY_COLORS[k], label=INDUSTRIES[k][0]) for k in INDUSTRY_ORDER],
               loc='upper center', ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.55, 0.985))
    fig.text(0.02, 0.015, NOTE.format(n=n), fontsize=9.5, color='#666666')
    save(fig, 'eda_company_rows')


def fig_section(area, n):
    items = sorted(area, key=lambda a: (a in EXCLUDED_AREAS, -area[a]))   # 제외 섹션은 맨 아래
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.37, right=0.95, top=0.86, bottom=0.14)
    vmax = max(area.values())
    for i, a in enumerate(items):
        ex = a in EXCLUDED_AREAS
        ax.barh(i, area[a], 0.68, color='white' if ex else '#5f8fc9', edgecolor='#999999' if ex else 'none',
                hatch='///' if ex else None, lw=1 if ex else 0)
        ax.text(area[a] + vmax * 0.01, i, f'{area[a]:,}  ({area[a] / n:.1%})' + ('  Excl. from LDA' if ex else ''),
                va='center', fontsize=11, color='#888888' if ex else '#222222')
    ax.set_yticks(range(len(items)), [f'{AREA_EN.get(a, a)}  ({a})' for a in items])
    for lab, a in zip(ax.get_yticklabels(), items):
        if a in EXCLUDED_AREAS:
            lab.set_color('#888888')
    ax.set_xlim(0, vmax * 1.32)
    ax.set_xlabel('Sentence pairs', fontsize=12)
    style_hbar(ax)
    fig.legend(handles=[Patch(color='#5f8fc9', label='Included in LDA'),
                        Patch(facecolor='white', edgecolor='#999999', hatch='///', label='Excl. from LDA (table-heavy)')],
               loc='upper center', ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.6, 0.985))
    fig.text(0.02, 0.015, NOTE.format(n=n), fontsize=9.5, color='#666666')
    save(fig, 'eda_section_rows')


def fig_types(types, n):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.20, right=0.93, top=0.92, bottom=0.14)
    vmax = max(types.values())
    for i, t in enumerate(TYPE_ORDER):
        ax.barh(i, types[t], 0.68, color=TYPE_COLORS[t], lw=0)
        ax.text(types[t] + vmax * 0.01, i, f'{types[t]:,}  ({types[t] / n:.1%})', va='center', fontsize=11,
                color='#222222')
    ax.set_yticks(range(len(TYPE_ORDER)), TYPE_ORDER)
    ax.set_xlim(0, vmax * 1.3)
    ax.set_xlabel('Sentence pairs', fontsize=12)
    style_hbar(ax)
    fig.text(0.02, 0.015, NOTE.format(n=n), fontsize=9.5, color='#666666')
    save(fig, 'eda_change_types')


def main():
    setup_style()
    rows, company, area, types = load()
    n = len(rows)
    with (FIG_DIR / 'eda_counts.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['group', 'key', 'count', 'share'])
        for g, c in (('company', company), ('area', area), ('change_type', types)):
            for k, v in c.most_common():
                w.writerow([g, k, v, round(v / n, 4)])
    print(f'전체 {n:,}행')
    print('유형:', ', '.join(f'{t} {types[t]:,} ({types[t] / n:.1%})' for t in TYPE_ORDER))
    print('섹션:', ', '.join(f'{a} {v:,}' for a, v in area.most_common()))
    fig_company(company, n)
    fig_section(area, n)
    fig_types(types, n)


if __name__ == '__main__':
    main()
