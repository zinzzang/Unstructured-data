"""유한양행 사례 그림 (발표 Slide 12): (위) 5가지 변화 유형 건수, (가운데) 연말 주가, (아래) 연도별 연결 영업이익.

사용: python scripts/plots/plot_company_finance.py
변화 유형: tfidf_masked_th0.3.csv, 표 성격 섹션 제외 (plot_industry_change_flow.load_counts 와 동일)
주가: 네이버 금융 수정주가, 12월 마지막 거래일 종가. 내려받기 실패 시 figures/biopharma_stock_yearend.csv 사용
영업이익: 연결 기준, 회사 잠정·확정 공시 보도 (OP_PROFIT, 억원)
출력: figures/fig_company_{slug}.png/.svg (16:9, dpi 200)
"""
import csv
import re

import matplotlib.pyplot as plt
import requests
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from plot_industry_change_flow import (COLORS, DPI, FAINT_ALPHA, FAINT_MAX, FIG_DIR, FIGSIZE, NEGATIVE, NET_COLOR,
                                       PERIODS, POSITIVE, TITLE_FS, TYPES, draw_flow, load_counts, setup_style)

YEARS = [2021, 2022, 2023, 2024, 2025]
COMPANIES = [('유한양행', '000100', '#0f766e')]
SLUG = {'유한양행': 'yuhan'}
# 연결 영업이익 (억원)
OP_PROFIT = {
    '유한양행': {2021: 486, 2022: 360, 2023: 568, 2024: 477, 2025: 1044},
}
OP_FLAGS = {}
OP_NOTES = {
    '유한양행': '2024년은 2025.2 잠정 공시 수치(477억). 2025년 보도의 증가율(+90.2%)로 역산하면 2024년 약 549억(재작성 가능성). 2025년은 잠정 공시',
}
TITLES = {'유한양행': ''}   # 결론 문장 제목 (비우면 제목 없음)
STOCK_COLOR = '#222222'
STOCK_CSV = FIG_DIR / 'biopharma_stock_yearend.csv'


def fetch_stock():
    """{company: {year: 연말 종가}} — 네이버 금융 월봉(수정주가)의 12월 봉 종가."""
    out = {}
    try:
        for name, code, _ in COMPANIES:
            r = requests.get('https://fchart.stock.naver.com/sise.nhn', timeout=15,
                             params=dict(symbol=code, timeframe='month', count=120, requestType=0),
                             headers={'User-Agent': 'Mozilla/5.0'})
            r.raise_for_status()
            rows = re.findall(r'data="(\d{8})\|\d+\|\d+\|\d+\|(\d+)\|\d+"', r.content.decode('euc-kr'))
            out[name] = {int(d[:4]): int(c) for d, c in rows if d[4:6] == '12' and int(d[:4]) in YEARS}
        with STOCK_CSV.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['company', 'year', 'close_adjusted'])
            for name, ys in out.items():
                for y, v in sorted(ys.items()):
                    w.writerow([name, y, v])
        print('[주가] 네이버 금융에서 내려받음')
    except Exception as e:   # 오프라인 등: 저장된 CSV 사용
        print(f'[주가] 내려받기 실패({e}) → {STOCK_CSV.name} 사용')
        out = {}
        with STOCK_CSV.open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                out.setdefault(r['company'], {})[int(r['year'])] = int(r['close_adjusted'])
    return out


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
