"""바이오·제약 기업별 그림: (위) 5가지 변화 유형 건수, (가운데) 연말 주가, (아래) 연도별 연결 영업이익.

사용: python scripts/plot_company_finance.py
변화 유형: tfidf_masked_th0.3.csv, 표 성격 섹션 제외 (plot_industry_change_flow.load_counts 와 동일)
주가: plot_biopharma_finance.fetch_stock (네이버 금융 수정주가, 12월 마지막 거래일 종가)
영업이익: 연결 기준, 회사 잠정·확정 공시 보도 (OP_PROFIT, 억원)
출력: figures/fig_company_{slug}.png/.svg (16:9, dpi 200)
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from plot_biopharma_finance import COMPANIES, YEARS, fetch_stock
from plot_industry_change_flow import (COLORS, DPI, FAINT_ALPHA, FAINT_MAX, FIG_DIR, FIGSIZE, NEGATIVE, NET_COLOR,
                                       PERIODS, POSITIVE, TITLE_FS, TYPES, draw_flow, load_counts, setup_style)

SLUG = {'삼성바이오로직스': 'samsung_biologics', '셀트리온': 'celltrion', '유한양행': 'yuhan'}
# 연결 영업이익 (억원)
OP_PROFIT = {
    '삼성바이오로직스': {2021: 5373, 2022: 9836, 2023: 11137, 2024: 13201, 2025: 20692},
    '셀트리온': {2021: 7525, 2022: 6472, 2023: 6510, 2024: 4920, 2025: 11685},
    '유한양행': {2021: 486, 2022: 360, 2023: 568, 2024: 477, 2025: 1044},
}
OP_FLAGS = {('삼성바이오로직스', 2025): '‡'}
OP_NOTES = {
    '삼성바이오로직스': '‡ 2025년: 2025.11 삼성에피스홀딩스 인적분할 후 기준일 가능성 — 앞 연도와 기준이 다를 수 있음. 2025년은 잠정 공시',
    '셀트리온': '2023년 말 셀트리온헬스케어 합병 → 2024년부터 합병 법인 기준. 2021년은 발표 당시 수치(이후 재무표 7,442억). 2025년은 잠정 공시',
    '유한양행': '2024년은 2025.2 잠정 공시 수치(477억). 2025년 보도의 증가율(+90.2%)로 역산하면 2024년 약 549억(재작성 가능성). 2025년은 잠정 공시',
}
TITLES = {'삼성바이오로직스': '', '셀트리온': '', '유한양행': ''}   # 결론 문장 제목 (비우면 제목 없음)
STOCK_COLOR = '#222222'


def plot_company(name, color, period_counts, prices, title=''):
    s = {t: [period_counts.get(p, {}).get(t, 0) for p in PERIODS] for t in TYPES}
    net = [sum(s[t][i] for t in POSITIVE) - sum(s[t][i] for t in NEGATIVE) for i in range(len(PERIODS))]
    total = sum(sum(s[t]) for t in TYPES)
    xs = [y - YEARS[0] - 0.5 for y in YEARS]          # 연도 점 = 시기 막대 사이 경계
    xlim = (-0.8, len(PERIODS) - 0.2)

    fig = plt.figure(figsize=FIGSIZE)
    top = 0.80 if title else 0.87
    gs = fig.add_gridspec(3, 1, height_ratios=[2.6, 1, 1], hspace=0.42, left=0.085, right=0.985, top=top,
                          bottom=0.1)
    ax, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))

    # (위) 변화 유형
    draw_flow(ax, s, net)
    ax.set_ylabel('건수')
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - (hi - lo) * 0.06, hi + (hi - lo) * 0.05)   # 패널이 낮아 합계 숫자가 눈금·머리글과 겹치지 않게
    ax.set_xlim(*xlim)

    # (가운데) 연말 주가
    ps = [prices[name][y] / 10000 for y in YEARS]     # 만 원
    ax1.plot(xs, ps, color=STOCK_COLOR, lw=2.4, marker='o', ms=8, mfc='white', mec=STOCK_COLOR, mew=2.2)
    for x, v in zip(xs, ps):
        ax1.annotate(f'{v:,.1f}', (x, v), xytext=(0, 9), textcoords='offset points', ha='center', va='bottom',
                     fontsize=12, fontweight='bold', color=STOCK_COLOR)
    lo, hi = min(ps), max(ps)
    ax1.set_ylim(lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.6)
    ax1.set_xlim(*xlim)
    ax1.set_xticks(xs, [f'{y}말' for y in YEARS])
    ax1.tick_params(axis='x', length=0, labelsize=11)
    ax1.yaxis.set_major_locator(MaxNLocator(3))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax1.set_ylabel('주가 (만 원)', fontsize=12)

    # (아래) 영업이익
    for x, y in zip(xs, YEARS):
        v = OP_PROFIT[name][y]
        flag = OP_FLAGS.get((name, y))
        ax2.bar(x, v, 0.42, color='white' if flag else color, edgecolor=color, lw=1.6 if flag else 0,
                hatch='//' if flag else None)
        ax2.text(x, v, f'{v:,}' + (flag or ''), ha='center', va='bottom', fontsize=12, fontweight='bold',
                 color='#222222')
    ax2.set_xlim(*xlim)
    ax2.set_ylim(0, max(OP_PROFIT[name].values()) * 1.3)
    ax2.set_xticks(xs, [str(y) for y in YEARS])
    ax2.tick_params(axis='x', length=0, labelsize=11)
    ax2.yaxis.set_major_locator(MaxNLocator(3))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax2.set_ylabel('영업이익 (억 원)', fontsize=12)

    head = ax.text(0, 1.035, name, transform=ax.transAxes, fontsize=17, fontweight='bold', va='bottom')
    head_w = head.get_window_extent(fig.canvas.get_renderer()).width * 72 / fig.dpi
    ax.annotate(f'문장 쌍 {total:,}건 (표 성격 섹션 제외) · 아래 두 칸: 연도 값을 시기 사이 경계에 맞춤',
                xy=(0, 1.035), xycoords='axes fraction', xytext=(head_w + 14, 0), textcoords='offset points',
                fontsize=12, color='#666666', va='bottom')
    handles = [Patch(color=COLORS[t], label=t) for t in TYPES]
    handles += [Line2D([], [], color=NET_COLOR, ls='--', lw=2, marker='o', ms=5, label='순방향 점수'),
                Line2D([], [], color=STOCK_COLOR, lw=2.4, marker='o', ms=7, mfc='white', mec=STOCK_COLOR, mew=2,
                       label='연말 주가 (가운데)'),
                Patch(color=color, label='연결 영업이익 (아래)'),
                Patch(facecolor='#777777', alpha=FAINT_ALPHA, label=f'연한 막대 = {FAINT_MAX}건 이하')]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.925 if title else 0.995), handlelength=1.6, columnspacing=0.9)
    fig.text(0.07, 0.035, '주가: 네이버 금융 수정주가(분할·증자 반영), 12월 마지막 거래일 종가 · '
             '영업이익: 연결 기준, 회사 공시 보도', fontsize=10, color='#666666')
    fig.text(0.07, 0.01, OP_NOTES[name], fontsize=10, color='#666666')
    if title:
        fig.suptitle(title, x=0.03, y=0.965, ha='left', va='top', fontsize=TITLE_FS, fontweight='bold')
    out = f'fig_company_{SLUG[name]}'
    for fmt in ('png', 'svg'):
        fig.savefig(FIG_DIR / f'{out}.{fmt}', dpi=DPI)
    plt.close(fig)
    print(f'[저장] figures/{out}.png, .svg')


def main():
    setup_style()
    prices = fetch_stock()
    _, _, _, by_company = load_counts()
    for name, _, color in COMPANIES:
        print(name, {p: {t: by_company[name][p][t] for t in TYPES} for p in PERIODS})
        plot_company(name, color, by_company[name], prices, title=TITLES[name])


if __name__ == '__main__':
    main()
