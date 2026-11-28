"""TF-IDF 매칭 후 산업별 글자수 변화 그래프 (Slide 01 우측 그림용).

사용: python scripts/plots/plot_tfidf_char_change.py
입력: data/results_v2/tfidf_masked_th0.3.csv
출력: figures/tfidf_char_change.png / .svg
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / 'data/results_v2/tfidf_masked_th0.3.csv'
FIG_DIR = ROOT / 'figures'

INDUSTRIES = {
    'semiconductor': ('반도체', ['삼성전자', 'SK하이닉스', '한미반도체']),
    'biopharma':    ('바이오·제약', ['삼성바이오로직스', '셀트리온', '유한양행']),
    'battery':      ('이차전지', ['LG에너지솔루션', '삼성SDI', 'SK이노베이션']),
}
IND_ORDER = ['semiconductor', 'biopharma', 'battery']
IND_COLORS = {'semiconductor': '#2a78d6', 'biopharma': '#c47a1e', 'battery': '#3a9a5b'}

PERIODS = ['21→22', '22→23', '23→24', '24→25']
FIGSIZE, DPI = (8, 5), 200


def setup_style():
    names = {f.name for f in fm.fontManager.ttflist}
    font = next((n for n in ('Pretendard', 'NanumGothic', 'Malgun Gothic', 'AppleGothic') if n in names), None)
    plt.rcParams.update({
        'font.family': font or 'sans-serif',
        'axes.unicode_minus': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#555555',
        'axes.grid': True, 'axes.grid.axis': 'y',
        'grid.color': '#000000', 'grid.alpha': 0.07, 'grid.linewidth': 0.8,
        'axes.axisbelow': True,
        'svg.fonttype': 'path',
    })


def load():
    df = pd.read_csv(INPUT_CSV)
    df['old_len'] = df['old_text'].str.len().fillna(0)
    df['new_len'] = df['new_text'].str.len().fillna(0)

    comp2ind = {c: k for k, (_, cs) in INDUSTRIES.items() for c in cs}
    df['industry'] = df['company'].map(comp2ind)
    df['year_pair'] = df['year_old'].astype(str).str[-2:] + '→' + df['year_new'].astype(str).str[-2:]

    return df.groupby(['industry', 'year_pair'])[['old_len', 'new_len']].sum().reset_index()


def plot(grp):
    setup_style()

    n_periods = len(PERIODS)
    n_ind = len(IND_ORDER)
    bar_w = 0.18
    group_gap = 0.72  # period 그룹 간격
    ind_gap = 0.02    # 2px surface gap

    # period 중심 x 위치
    group_x = [i * group_gap for i in range(n_periods)]
    # 각 industry 오프셋: 가운데 정렬
    total_w = n_ind * (bar_w + ind_gap) - ind_gap
    offsets = [-(total_w / 2) + j * (bar_w + ind_gap) + bar_w / 2 for j in range(n_ind)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.12)

    # 각 period 그룹 안: old 막대 / 2px gap / new 막대, 산업 순으로 나란히
    # 막대 하나 너비 bar_w, 같은 산업 old-new 쌍 사이 gap = ind_gap
    pair_w = bar_w * 2 + ind_gap        # old + gap + new
    pair_gap = 0.04                     # 산업 간 간격 (surface gap 2px 역할)
    # 6개 막대 중심 범위: leftmost=0, rightmost=(n_ind-1)*(pair_w+pair_gap)+(bar_w+ind_gap)
    span = (n_ind - 1) * (pair_w + pair_gap) + (bar_w + ind_gap)
    pair_offsets = [-(span / 2) + k * (pair_w + pair_gap) for k in range(n_ind)]

    for j, ind_key in enumerate(IND_ORDER):
        ind_label, _ = INDUSTRIES[ind_key]
        color = IND_COLORS[ind_key]

        for i, period in enumerate(PERIODS):
            row = grp[(grp['industry'] == ind_key) & (grp['year_pair'] == period)]
            old_v = row['old_len'].values[0] / 10000 if len(row) else 0
            new_v = row['new_len'].values[0] / 10000 if len(row) else 0

            x0 = group_x[i] + pair_offsets[j]           # old 막대 중심
            x1 = x0 + bar_w + ind_gap                   # new 막대 중심

            kw = dict(linewidth=0, zorder=2)
            ax.bar(x0, old_v, bar_w, color=color, alpha=0.38, **kw)
            ax.bar(x1, new_v, bar_w, color=color, alpha=1.0,  **kw,
                   label=ind_label if i == 0 else None)

    # x축 눈금: period 그룹 중심
    ax.set_xticks(group_x)
    ax.set_xticklabels(PERIODS, fontsize=12)
    ax.set_ylabel('글자수 (만 자)', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}만'))
    ax.tick_params(axis='both', length=0, labelsize=11)

    # 범례: 색 = industry, 명암 = 이전/이후
    from matplotlib.patches import Patch
    handles = [Patch(fc=IND_COLORS[k], label=INDUSTRIES[k][0]) for k in IND_ORDER]
    # 이전/이후 설명용 패치
    before = Patch(fc='#999999', alpha=0.4, label='이전 연도')
    after  = Patch(fc='#999999', alpha=1.0, label='다음 연도')
    ax.legend(handles=handles + [before, after],
              ncol=5, loc='upper left', frameon=False,
              fontsize=10, columnspacing=0.8, handlelength=1.2)

    ax.set_title('TF-IDF 매칭 전후 글자수 변화\n', fontsize=14, fontweight='bold',
                 loc='left', pad=4)
    ax.annotate('연한 막대 = 이전 연도  │  진한 막대 = 다음 연도',
                xy=(0, 1.03), xycoords='axes fraction', fontsize=9, color='#666666')

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ('png', 'svg'):
        fig.savefig(FIG_DIR / f'tfidf_char_change.{fmt}', dpi=DPI)
    plt.close(fig)
    print('[저장] figures/tfidf_char_change.png, .svg')


if __name__ == '__main__':
    plot(load())
