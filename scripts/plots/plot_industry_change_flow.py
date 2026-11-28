"""산업별(반도체 / 바이오·제약 / 이차전지) 5가지 변화 유형 × 시기 흐름 + 산업 수출액 (발표 Slide 10).

사용: python scripts/plots/plot_industry_change_flow.py
입력: data/results_v2/tfidf_masked_th0.3.csv (분류 로직·threshold 변경 없음)
      EXCLUDED_SECTIONS(표 성격 섹션)에 속한 문장 쌍은 제외
출력: figures/fig0_{industry}_ext.png / .svg (16:9, dpi 200), figures/industry_counts.csv
제목(결론 문장)은 TITLES_EXT에 넣거나 plot_industry(title=...) 인자로 넘긴다. 해석 문장은 자동 생성하지 않는다.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / 'data/results_v2/tfidf_masked_th0.3.csv'
FIG_DIR = ROOT / 'figures'

INDUSTRIES = {
    'semiconductor': ('Semiconductor', ['삼성전자', 'SK하이닉스', '한미반도체']),
    'biopharma': ('Biopharma', ['삼성바이오로직스', '셀트리온', '유한양행']),
    'battery': ('Battery', ['LG에너지솔루션', '삼성SDI', 'SK이노베이션']),
}
COMPANY_EN = {
    '삼성전자': 'Samsung Electronics', 'SK하이닉스': 'SK hynix', '한미반도체': 'Hanmi Semi.',
    '삼성바이오로직스': 'Samsung Bio.', '셀트리온': 'Celltrion', '유한양행': 'Yuhan',
    'LG에너지솔루션': 'LG Energy Sol.', '삼성SDI': 'Samsung SDI', 'SK이노베이션': 'SK Innovation',
}
# table-like sections excluded (bulk add/remove distorts flow); match against CSV old_path/new_path
EXCLUDED_SECTIONS = ['원재료 및 생산설비', '주요계약 및 연구개발활동']
TITLES_EXT = {'semiconductor': '', 'biopharma': '', 'battery': ''}   # 외부 지표 겹친 그림(_ext)의 제목

# 연도별 외부 지표 (오른쪽 보조축 꺾은선, figures/fig0_{industry}_ext.png). text는 점 위 표시 문자열.
EXT_UNIT = 'USD 100M'
EXTERNAL = {
    'semiconductor': dict(
        label='Semiconductor Exports',
        values={2021: 1280, 2022: 1292, 2023: 986.3, 2024: 1419.2, 2025: 1734.1},
        text={2021: '1,280', 2022: '1,292', 2023: '986.3', 2024: '1,419.2', 2025: '1,734.1'},
        foot='Source: Korea National Indicators (2023–2025)'),
    'battery': dict(
        label='Battery Exports',
        values={2021: 86.72, 2022: 99.81, 2023: 98.26, 2024: 82.02, 2025: 79.39},
        text={2021: '86.72', 2022: '99.81', 2023: '98.26', 2024: '82.02', 2025: '79.39*'},
        foot='* 2025 est. (KIET forecast, not final) · Source: KITA (2021–2023), KIET (2024–2025)'),
    'biopharma': dict(
        label='Pharma Exports',
        values={2021: 70.4, 2022: 80.8, 2023: 75.6, 2024: 92.7, 2025: 104.1},
        text={2021: '70.4', 2022: '80.8', 2023: '75.6', 2024: '92.7', 2025: '104.1'},
        foot='Source: Ministry of Health and Welfare'),
}
EXT_COLOR = '#222222'

PERIODS = ['21→22', '22→23', '23→24', '24→25']
TYPES = ['NEW', 'EXPANDED', 'MODIFIED', 'REDUCED', 'REMOVED']   # UNCERTAIN, UNCHANGED 제외
POSITIVE, NEGATIVE = ('NEW', 'EXPANDED'), ('REDUCED', 'REMOVED')
COLORS = {'NEW': '#2a78d6', 'EXPANDED': '#85B7EB', 'MODIFIED': '#B4B2A9', 'REDUCED': '#F09595', 'REMOVED': '#e34948'}
NET_COLOR = '#888780'
FAINT_ALPHA, FAINT_MAX = 0.3, 2          # 합계 2건 이하 막대는 흐리게
FIGSIZE, DPI = (13.33, 7.5), 200
TITLE_FS, LABEL_FS, TICK_FS = 22, 14, 12
MAIN_W, MOD_W, MAIN_DX, MOD_DX = 0.42, 0.20, -0.12, 0.21


def setup_style():
    names = {f.name for f in fm.fontManager.ttflist}
    font = next((n for n in ('Pretendard', 'NanumGothic', 'Malgun Gothic', 'AppleGothic') if n in names), None)
    plt.rcParams.update({
        'font.family': font or 'sans-serif', 'axes.unicode_minus': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#555555',
        'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': '#000000', 'grid.alpha': 0.07, 'grid.linewidth': 0.8,
        'axes.axisbelow': True, 'axes.labelsize': LABEL_FS, 'xtick.labelsize': TICK_FS, 'ytick.labelsize': TICK_FS,
        'svg.fonttype': 'path',
    })
    return font


def load_counts():
    """{industry: {period: Counter(type)}}, 제외 건수, 기업별 건수,
    기업별 전체 유형 건수 {company: {period: Counter(7개 유형)}}"""
    company_ind = {c: k for k, (_, cs) in INDUSTRIES.items() for c in cs}
    counts = {k: defaultdict(Counter) for k in INDUSTRIES}
    by_company = defaultdict(lambda: defaultdict(Counter))
    excluded, per_company = Counter(), Counter()
    with INPUT_CSV.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            ind = company_ind.get(r['company'])
            if ind is None:
                continue
            if any(s in r['old_path'] or s in r['new_path'] for s in EXCLUDED_SECTIONS):
                if r['change_type'] in TYPES:
                    excluded[ind] += 1
                continue
            period = f"{r['year_old'][2:]}→{r['year_new'][2:]}"
            by_company[r['company']][period][r['change_type']] += 1
            if r['change_type'] not in TYPES:
                continue
            counts[ind][period][r['change_type']] += 1
            per_company[r['company']] += 1
    return counts, excluded, per_company, by_company


def draw_flow(ax, s, net, label_fs=13, turn_fs=14, fmt=lambda v: f'{v}', faint=True):
    """fmt: 막대 숫자 표기 (건수는 정수, 가중 그림은 '%.1f'). faint=False면 2건 이하 흐림 처리를 하지 않는다."""
    hi = max(1, *(s['NEW'][i] + s['EXPANDED'][i] for i in range(4)), *s['MODIFIED'], *net)
    lo = max(1, *(s['REDUCED'][i] + s['REMOVED'][i] for i in range(4)), *(-n for n in net))
    ylim = (-lo * 1.18, hi * 1.22)
    span = ylim[1] - ylim[0]
    pad = span * 0.012
    for i in range(len(PERIODS)):
        pos, neg, mod = s['NEW'][i] + s['EXPANDED'][i], s['REDUCED'][i] + s['REMOVED'][i], s['MODIFIED'][i]
        a_pos, a_neg, a_mod = (FAINT_ALPHA if faint and v <= FAINT_MAX else 1 for v in (pos, neg, mod))
        segs = [('NEW', 0, s['NEW'][i], a_pos), ('EXPANDED', s['NEW'][i], s['EXPANDED'][i], a_pos),
                ('REDUCED', 0, -s['REDUCED'][i], a_neg), ('REMOVED', -s['REDUCED'][i], -s['REMOVED'][i], a_neg)]
        for t, bottom, h, a in segs:
            if h:
                ax.bar(i + MAIN_DX, h, MAIN_W, bottom=bottom, color=COLORS[t], alpha=a, lw=0)
                if abs(h) / span >= 0.045:
                    ax.text(i + MAIN_DX, bottom + h / 2, fmt(abs(h)), ha='center', va='center', fontsize=label_fs,
                            color='white' if t in ('NEW', 'REMOVED') else '#222222', alpha=max(a, 0.6))
        if mod:
            ax.bar(i + MOD_DX, mod, MOD_W, color=COLORS['MODIFIED'], alpha=a_mod, lw=0)
            ax.text(i + MOD_DX, mod + pad, fmt(mod), ha='center', va='bottom', fontsize=label_fs, color='#555555')
        total_box = dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.85)   # 뒤로 지나가는 선 가림
        if pos:
            ax.text(i + MAIN_DX, pos + pad, fmt(pos), ha='center', va='bottom', fontsize=label_fs + 1,
                    fontweight='bold', color='#222222', bbox=total_box)
        if neg:
            ax.text(i + MAIN_DX, -neg - pad, fmt(neg), ha='center', va='top', fontsize=label_fs + 1,
                    fontweight='bold', color='#222222', bbox=total_box)
    ax.axhline(0, color='#333333', lw=1)
    ax.plot(range(len(PERIODS)), net, ls='--', color=NET_COLOR, lw=2, marker='o', ms=5, zorder=5)
    for i in range(len(PERIODS) - 1):
        if net[i] > 0 and net[i + 1] < 0:           # 순방향 점수가 양수 → 음수로 바뀐 구간
            xm = i + net[i] / (net[i] - net[i + 1])
            ax.annotate('Reversal', xy=(xm, 0), xytext=(16, -30), textcoords='offset points', fontsize=turn_fs,
                        color='#444444', fontweight='bold', ha='left', va='center',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9),
                        arrowprops=dict(arrowstyle='->', color='#444444', lw=1.3, shrinkA=2, shrinkB=2), zorder=6)
    ax.set_xticks(range(len(PERIODS)), PERIODS)
    ax.set_xlim(-0.6, len(PERIODS) - 0.4)
    ax.set_ylim(*ylim)
    ax.tick_params(axis='x', length=0, pad=6)
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{abs(int(v))}'))


def draw_external(ax, ext):
    """연도별 외부 지표를 오른쪽 보조축 꺾은선으로. 연도 점은 시기 막대 사이 경계(2021=-0.5, 2022=0.5, ...)에 둔다.
    막대·막대 숫자를 가리지 않도록 선은 막대 뒤에 그리고, 두 축의 0선 높이를 맞춘다. 반환: 범례 핸들"""
    years = sorted(ext['values'])
    first = 2000 + int(PERIODS[0][:2])
    xs = [y - first - 0.5 for y in years]
    vs = [float(ext['values'][y]) for y in years]
    tw = ax.twinx()
    ax.set_zorder(tw.get_zorder() + 1)
    ax.patch.set_visible(False)
    tw.grid(False)
    tw.spines['right'].set_visible(True)
    tw.spines['right'].set_color(EXT_COLOR)
    lo, hi = ax.get_ylim()
    f = -lo / (hi - lo)
    top = max(vs) * 1.3
    tw.set_ylim(-f / (1 - f) * top, top)
    tw.plot(xs, vs, color=EXT_COLOR, lw=2.6, marker='o', ms=9, mfc='white', mec=EXT_COLOR, mew=2.4)
    for x, y, v in zip(xs, years, vs):
        tw.annotate(ext.get('text', {}).get(y, f'{v:,.10g}'), (x, v), xytext=(0, 13), textcoords='offset points',
                    ha='center', va='bottom', fontsize=13, fontweight='bold', color=EXT_COLOR,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
        tw.annotate(str(y), (x, v), xytext=(0, 34), textcoords='offset points', ha='center', va='bottom',
                    fontsize=11, color='#777777')
    tw.set_ylabel(f'{ext["label"]} ({EXT_UNIT})', color=EXT_COLOR)
    tw.yaxis.set_major_locator(MaxNLocator(5))
    tw.set_yticks([t for t in tw.get_yticks() if 0 <= t <= top])
    tw.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    tw.tick_params(axis='y', colors=EXT_COLOR)
    ax.set_xlim(-0.8, len(PERIODS) - 0.2)
    return Line2D([], [], color=EXT_COLOR, lw=2.6, marker='o', ms=8, mfc='white', mec=EXT_COLOR, mew=2.2,
                  label=f'{ext["label"]} (right axis, {EXT_UNIT})')


def plot_industry(key, by_period, title='', ext=None):
    """by_period: {period: Counter(type)} 건수. ext가 있으면 외부 지표를 보조축에 겹치고 파일명에 _ext를 붙인다."""
    label, companies = INDUSTRIES[key]
    s = {t: [by_period.get(p, {}).get(t, 0) for p in PERIODS] for t in TYPES}
    net = [sum(s[t][i] for t in POSITIVE) - sum(s[t][i] for t in NEGATIVE) for i in range(len(PERIODS))]
    total = sum(sum(c.values()) for c in by_period.values())

    fig, ax = plt.subplots(figsize=FIGSIZE)
    top, legend_y = (0.77, 0.905) if title else (0.85, 0.985)
    side = bool(ext)
    fig.subplots_adjust(left=0.07, right=0.93 if side else 0.985, bottom=0.12 if side else 0.10, top=top)
    draw_flow(ax, s, net)
    ax.set_ylabel('Count')
    note = (f'{" · ".join(COMPANY_EN.get(c, c) for c in companies)} combined · '
            f'{total:,} pairs (table-type sections excl.)')
    head = ax.text(0, 1.035, label, transform=ax.transAxes, fontsize=17, fontweight='bold', va='bottom')
    head_w = head.get_window_extent(fig.canvas.get_renderer()).width * 72 / fig.dpi
    ax.annotate(note, xy=(0, 1.035), xycoords='axes fraction', xytext=(head_w + 14, 0),
                textcoords='offset points', fontsize=12, color='#666666', va='bottom')
    handles = [Patch(color=COLORS[t], label=t) for t in TYPES]
    handles.append(Line2D([], [], color=NET_COLOR, ls='--', lw=2, marker='o', ms=5, label='Net score'))
    handles.append(Patch(facecolor='#777777', alpha=FAINT_ALPHA, label=f'Faint bar = ≤{FAINT_MAX} pairs total'))
    if ext:
        handles.insert(5, draw_external(ax, ext))
        if ext.get('foot'):
            fig.text(0.07, 0.015, ext['foot'], fontsize=11, color='#666666', va='bottom')
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               fontsize=11 if side else 12, bbox_to_anchor=(0.5, legend_y), handlelength=1.6,
               columnspacing=0.9 if side else 1.4)
    if title:
        fig.suptitle(title, x=0.03, y=0.965, ha='left', va='top', fontsize=TITLE_FS, fontweight='bold')
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    name = 'fig0_' + key + ('_ext' if ext else '')
    for fmt in ('png', 'svg'):
        fig.savefig(FIG_DIR / f'{name}.{fmt}', dpi=DPI)
    plt.close(fig)
    print(f'[저장] figures/{name}.png, .svg')


def main():
    print(f'[폰트] {setup_style()}')
    counts, excluded, per_company, _ = load_counts()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with (FIG_DIR / 'industry_counts.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['industry', 'period', *TYPES, 'forward_score'])
        for key in INDUSTRIES:
            for per in PERIODS:
                v = [counts[key][per][t] for t in TYPES]
                d = dict(zip(TYPES, v))
                w.writerow([key, per, *v, sum(d[t] for t in POSITIVE) - sum(d[t] for t in NEGATIVE)])
    for key, (label, companies) in INDUSTRIES.items():
        total = sum(sum(c.values()) for c in counts[key].values())
        print(f'{label}: {total:,}건 (' + ' · '.join(f'{c} {per_company[c]:,}' for c in companies)
              + f') / 제외 {excluded[key]:,}건')
        plot_industry(key, counts[key], title=TITLES_EXT.get(key) or '', ext=EXTERNAL[key])


if __name__ == '__main__':
    main()
