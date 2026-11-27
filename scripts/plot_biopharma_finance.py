"""바이오·제약 3개 기업 매출액 그림 + 변화 흐름(기업 동일 가중) 위에 연말 주가 지수 겹친 그림.

사용: python scripts/plot_biopharma_finance.py
매출액: 연결 기준, 언론 보도된 잠정·확정 공시 수치 (REVENUE, 단위 억원). 출처는 REVENUE_SOURCES.
주가: 네이버 금융 월봉 시세(수정주가)의 12월 마지막 거래일 종가. 실행 시 내려받아 figures/biopharma_stock_yearend.csv에 저장하고,
      내려받기 실패 시 그 CSV를 읽는다. 세 회사 주가 수준이 달라 2021년 말 = 100 지수로 겹친다.
출력: figures/fig_bio_revenue.png/.svg, figures/fig0w_biopharma_stock.png/.svg
"""
import csv
import re

import matplotlib.pyplot as plt
import requests
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from plot_industry_change_flow import (DPI, FIG_DIR, FIGSIZE, PERIODS, TITLE_FS, load_counts, plot_industry,
                                       setup_style, weighted_rates)

YEARS = [2021, 2022, 2023, 2024, 2025]
COMPANIES = [('삼성바이오로직스', '207940', '#1f2937'), ('셀트리온', '068270', '#7c3aed'), ('유한양행', '000100', '#0f766e')]
# 연결 매출액 (억원)
REVENUE = {
    '삼성바이오로직스': {2021: 15680, 2022: 30013, 2023: 36946, 2024: 45473, 2025: 45570},
    '셀트리온': {2021: 18908, 2022: 22839, 2023: 21760, 2024: 35573, 2025: 41625},
    '유한양행': {2021: 16878, 2022: 17758, 2023: 18590, 2024: 20678, 2025: 21866},
}
# 기준이 다를 수 있어 그림에 표시할 칸
REVENUE_FLAGS = {('삼성바이오로직스', 2025): '‡'}
REVENUE_NOTE = ('‡ 삼성바이오로직스 2025년: 보도 수치 4조5,570억(전년 대비 +30.3%)은 2024년 공시치(4조5,473억)와 증가율이 맞지 않음. '
                '2025.11 삼성에피스홀딩스 인적분할로 바이오에피스를 뺀 기준일 가능성 — 확인 필요')
REVENUE_SOURCES = '출처: 회사 공시 보도(이데일리·이투데이·더팩트·메디파나·히트뉴스 등), FnGuide'
TITLE_REV = ''
TITLE_STOCK = ''
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


def save(fig, name):
    for fmt in ('png', 'svg'):
        fig.savefig(FIG_DIR / f'{name}.{fmt}', dpi=DPI)
    plt.close(fig)
    print(f'[저장] figures/{name}.png, .svg')


def fig_revenue():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    top = 0.78 if TITLE_REV else 0.86
    fig.subplots_adjust(left=0.07, right=0.985, top=top, bottom=0.15)
    w = 0.26
    for j, (name, _, color) in enumerate(COMPANIES):
        for i, y in enumerate(YEARS):
            v = REVENUE[name][y] / 10000          # 조원
            flag = REVENUE_FLAGS.get((name, y))
            x = i + (j - 1) * w
            ax.bar(x, v, w * 0.9, color='white' if flag else color, edgecolor=color, lw=1.6 if flag else 0,
                   hatch='//' if flag else None)
            ax.text(x, v + 0.06, f'{v:.2f}' + (flag or ''), ha='center', va='bottom', fontsize=11,
                    color='#222222', rotation=0)
    ax.set_xticks(range(len(YEARS)), [str(y) for y in YEARS])
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('연결 매출액 (조 원)')
    ax.set_ylim(0, max(max(v.values()) for v in REVENUE.values()) / 10000 * 1.15)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.text(0, 1.035, '바이오·제약 3개 기업 매출액', transform=ax.transAxes, fontsize=17, fontweight='bold',
            va='bottom')
    fig.legend(handles=[Patch(color=c, label=n) for n, _, c in COMPANIES], loc='upper center', ncol=3,
               frameon=False, fontsize=13, bbox_to_anchor=(0.5, top + 0.12))
    fig.text(0.03, 0.045, REVENUE_NOTE, fontsize=9.5, color='#666666')
    fig.text(0.03, 0.015, REVENUE_SOURCES, fontsize=9.5, color='#666666')
    if TITLE_REV:
        fig.suptitle(TITLE_REV, x=0.03, y=0.965, ha='left', va='top', fontsize=TITLE_FS, fontweight='bold')
    save(fig, 'fig_bio_revenue')


def stock_overlay(prices):
    """연말 주가 지수(2021년 말 = 100)를 오른쪽 보조축 꺾은선으로. 연도 점은 시기 막대 사이 경계에 둔다."""
    def draw(ax):
        tw = ax.twinx()
        ax.set_zorder(tw.get_zorder() + 1)
        ax.patch.set_visible(False)
        tw.grid(False)
        tw.spines['right'].set_visible(True)
        xs = [y - YEARS[0] - 0.5 for y in YEARS]
        idx = {n: [100 * prices[n][y] / prices[n][YEARS[0]] for y in YEARS] for n, _, _ in COMPANIES}
        # 주가 지수는 막대의 0선과 무관하므로 자체 범위로 펼친다 (선이 한 줄로 뭉치지 않게)
        top = max(max(v) for v in idx.values()) * 1.12
        bottom = min(min(v) for v in idx.values()) * 0.55
        tw.set_ylim(bottom, top)
        handles = []
        for n, _, color in COMPANIES:
            tw.plot(xs, idx[n], color=color, lw=2.4, marker='o', ms=7, mfc='white', mec=color, mew=2)
            tw.annotate(f'{idx[n][-1]:.0f}', (xs[-1], idx[n][-1]), xytext=(8, 0), textcoords='offset points',
                        fontsize=11, fontweight='bold', color=color, va='center',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85))
            handles.append(Line2D([], [], color=color, lw=2.4, marker='o', ms=6, mfc='white', mec=color, mew=2,
                                  label=f'{n} 주가'))
        tw.axhline(100, color='#999999', lw=0.8, ls=':', zorder=0)
        for x, y in zip(xs, YEARS):
            tw.annotate(f'{y}말', (x, top), xytext=(0, -4), textcoords='offset points', ha='center', va='top',
                        fontsize=10, color='#888888')
        tw.set_ylabel('연말 주가 지수 (2021년 말 = 100)')
        tw.yaxis.set_major_locator(MaxNLocator(5))
        tw.set_yticks([t for t in tw.get_yticks() if bottom <= t <= top])
        tw.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}'))
        ax.set_xlim(-0.8, len(PERIODS) - 0.2)
        closes = ' · '.join(f'{n} ' + '→'.join(f'{prices[n][y] / 10000:,.1f}' for y in YEARS)
                            for n, _, _ in COMPANIES)
        foot = (f'주가: 네이버 금융 수정주가(분할·증자 반영), 각 연도 12월 마지막 거래일 종가. 오른쪽 축은 2021년 말 = 100 지수, 막대 축과 0선 무관\n'
                f'연말 종가(만 원, 2021→2025)  {closes}')
        return handles, foot
    return draw


def main():
    setup_style()
    fig_revenue()
    prices = fetch_stock()
    for n, _, _ in COMPANIES:
        print(n, {y: prices[n][y] for y in YEARS})
    _, _, _, by_company = load_counts()
    plot_industry('biopharma', weighted_rates('biopharma', by_company), title=TITLE_STOCK, weighted=True,
                  overlay=stock_overlay(prices), name='fig0w_biopharma_stock')


if __name__ == '__main__':
    main()
