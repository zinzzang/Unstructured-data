"""9개 기업 × 4개 연도쌍 변화율 히트맵 (발표 Slide 10).

사용: python scripts/plots/plot_change_concentration.py
입력: data/results_v2/tfidf_masked_th0.3.csv 전체 (섹션 제외 없음)
변화율 = (NEW+EXPANDED+MODIFIED+REDUCED+REMOVED) / 해당 기업·연도쌍 전체 비교 문장 쌍
출력: figures/fig_change_rate_heatmap.png/.svg
"""
import csv
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from plot_industry_change_flow import (COMPANY_EN, DPI, FIG_DIR, FIGSIZE, INDUSTRIES, INPUT_CSV, PERIODS, TYPES,
                                       setup_style)

INDUSTRY_ORDER = ['semiconductor', 'battery', 'biopharma']
HIGHLIGHT = {('셀트리온', '23→24'), ('SK이노베이션', '23→24')}
N_ROWS = sum(1 for _ in csv.DictReader(INPUT_CSV.open(encoding='utf-8-sig')))
NOTE = f'tfidf_masked_th0.3.csv · {N_ROWS:,} rows · Change rate = 5 types (NEW·EXPANDED·MODIFIED·REDUCED·REMOVED) ÷ total sentence pairs'


def load():
    total, types = Counter(), defaultdict(Counter)
    with INPUT_CSV.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            key = (r['company'], f"{r['year_old'][2:]}→{r['year_new'][2:]}")
            total[key] += 1
            types[key][r['change_type']] += 1
    return total, types


def companies():
    return [c for k in INDUSTRY_ORDER for c in INDUSTRIES[k][1]]


def rate(total, types, key):
    return 100 * sum(types[key][t] for t in TYPES) / total[key] if total[key] else 0


def save(fig, name):
    for fmt in ('png', 'svg'):
        fig.savefig(FIG_DIR / f'{name}.{fmt}', dpi=DPI)
    plt.close(fig)
    print(f'[저장] figures/{name}.png, .svg')


# ---------------------------------------------------------------- 1. 히트맵
def fig_heatmap(total, types):
    cs = companies()
    m = np.array([[rate(total, types, (c, p)) for p in PERIODS] for c in cs])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.20, right=0.90, top=0.90, bottom=0.10)
    cmap = LinearSegmentedColormap.from_list('chg', ['#ffffff', '#f6d9c9', '#e9895e', '#b8341b', '#6e1407'])
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, c in enumerate(cs):
        for j, p in enumerate(PERIODS):
            v = m[i, j]
            ax.text(j, i, f'{v:.0f}%', ha='center', va='center', fontsize=15,
                    fontweight='bold' if (c, p) in HIGHLIGHT else 'normal',
                    color='white' if v >= 55 else '#222222')
            sub = f'{sum(types[(c, p)][t] for t in TYPES):,} / {total[(c, p)]:,}'
            ax.text(j, i + 0.3, sub, ha='center', va='center', fontsize=9.5,
                    color='#f3e7e2' if v >= 55 else '#777777')
            if (c, p) in HIGHLIGHT:
                ax.add_patch(Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, fill=False, ec='#111111', lw=3))
    ax.set_xticks(range(len(PERIODS)), PERIODS, fontsize=14)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(cs)), [COMPANY_EN.get(c, c) for c in cs], fontsize=13)
    ax.tick_params(length=0, pad=8)
    # industry separator lines and labels
    row = 0
    for k in INDUSTRY_ORDER:
        n = len(INDUSTRIES[k][1])
        if row:
            ax.axhline(row - 0.5, color='white', lw=5)
        ax.text(-0.215, row + (n - 1) / 2, INDUSTRIES[k][0], transform=ax.get_yaxis_transform(), ha='center',
                va='center', fontsize=14, fontweight='bold', rotation=90, color='#333333')
        row += n
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('Change Rate (%)', fontsize=12)
    cb.outline.set_visible(False)
    fig.text(0.02, 0.025, NOTE + ' · cell bottom = changed / total · bold border = merger year', fontsize=10,
             color='#666666')
    save(fig, 'fig_change_rate_heatmap')
    return m


def main():
    setup_style()
    fig_heatmap(*load())


if __name__ == '__main__':
    main()
