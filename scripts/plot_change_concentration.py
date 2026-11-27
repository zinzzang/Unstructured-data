"""변화율 히트맵 · 섹션별 변화 쏠림 · 케이스 3개 변화율 비교 (발표 Slide 08·12·14용, 정적 이미지).

사용: python scripts/plot_change_concentration.py
입력: data/results_v2/tfidf_masked_th0.3.csv 전체 (섹션 제외 없음)
변화율 = (NEW+EXPANDED+MODIFIED+REDUCED+REMOVED) / 해당 기업·연도쌍 전체 비교 문장 쌍 (README 정의)
출력: figures/fig_change_rate_heatmap.png/.svg     9개 기업 × 4개 연도쌍 변화율
      figures/fig_change_by_section.png/.svg       기업·연도쌍별 변화 문장 쌍의 섹션 구성 (100% 누적)
      figures/fig_case_change_rates.png/.svg       케이스 3개 변화율 (변화 유형별 구성)
      figures/change_concentration.csv             위 그림의 집계값
"""
import csv
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle

from plot_industry_change_flow import (COLORS, DPI, FIG_DIR, FIGSIZE, INDUSTRIES, INPUT_CSV, PERIODS, TYPES,
                                       setup_style)

INDUSTRY_ORDER = ['semiconductor', 'battery', 'biopharma']
HIGHLIGHT = {('셀트리온', '23→24'), ('SK이노베이션', '23→24')}
CASES = [('SK하이닉스', '22→23', 'AI·HBM 수요'), ('SK이노베이션', '23→24', 'SK E&S 합병'),
         ('셀트리온', '23→24', '셀트리온헬스케어 합병')]
SECTIONS = ['1. 사업의 개요', '2. 주요 제품 및 서비스', '3. 원재료 및 생산설비', '4. 매출 및 수주상황',
            '5. 위험관리 및 파생거래', '6. 주요계약 및 연구개발활동', '7. 기타 참고사항']
TABLE_SECTIONS = {'3. 원재료 및 생산설비', '6. 주요계약 및 연구개발활동'}   # 표 성격 섹션 (강조)
SECTION_COLORS = {'1. 사업의 개요': '#cfe0d6', '2. 주요 제품 및 서비스': '#9cc5b0',
                  '3. 원재료 및 생산설비': '#c2410c', '4. 매출 및 수주상황': '#6a9fb5',
                  '5. 위험관리 및 파생거래': '#b9b6ac', '6. 주요계약 및 연구개발활동': '#7c3aed',
                  '7. 기타 참고사항': '#dcd9d0'}
N_ROWS = sum(1 for _ in csv.DictReader(INPUT_CSV.open(encoding='utf-8-sig')))
NOTE = f'tfidf_masked_th0.3.csv 전체 {N_ROWS:,}행 · 변화율 = 5개 변화 유형(NEW·EXPANDED·MODIFIED·REDUCED·REMOVED) ÷ 전체 비교 문장 쌍'


def load():
    total, types, sections = Counter(), defaultdict(Counter), defaultdict(Counter)
    with INPUT_CSV.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            key = (r['company'], f"{r['year_old'][2:]}→{r['year_new'][2:]}")
            total[key] += 1
            types[key][r['change_type']] += 1
            if r['change_type'] in TYPES:
                path = (r['new_path'] or r['old_path']).split(' > ')
                sections[key][path[1] if len(path) > 1 else path[0]] += 1
    return total, types, sections


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
    ax.set_yticks(range(len(cs)), cs, fontsize=13)
    ax.tick_params(length=0, pad=8)
    # 산업 구분선과 라벨
    row = 0
    for k in INDUSTRY_ORDER:
        n = len(INDUSTRIES[k][1])
        if row:
            ax.axhline(row - 0.5, color='white', lw=5)
        ax.text(-0.215, row + (n - 1) / 2, INDUSTRIES[k][0], transform=ax.get_yaxis_transform(), ha='center',
                va='center', fontsize=14, fontweight='bold', rotation=90, color='#333333')
        row += n
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('변화율 (%)', fontsize=12)
    cb.outline.set_visible(False)
    fig.text(0.02, 0.025, NOTE + ' · 칸 아래 = 변화 문장 쌍 / 전체 · 굵은 테두리 = 합병 연도', fontsize=10,
             color='#666666')
    save(fig, 'fig_change_rate_heatmap')
    return m


# ---------------------------------------------------------------- 2. 섹션별 변화 쏠림
def fig_sections(sections):
    rows = [(c, p) for c in companies() for p in PERIODS]
    fig, ax = plt.subplots(figsize=(13.33, 9.5))
    fig.subplots_adjust(left=0.20, right=0.88, top=0.925, bottom=0.095)
    for i, key in enumerate(rows):
        tot = sum(sections[key].values())
        left = 0
        for s in SECTIONS:
            share = 100 * sections[key][s] / tot if tot else 0
            if share:
                ax.barh(i, share, 0.78, left=left, color=SECTION_COLORS[s], lw=0)
                if share >= 12:
                    ax.text(left + share / 2, i, f'{share:.0f}%', ha='center', va='center', fontsize=9,
                            color='white' if s in TABLE_SECTIONS else '#333333')
            left += share
        ax.text(101, i, f'{tot:,}건', va='center', fontsize=9.5, color='#555555')
    ax.set_yticks(range(len(rows)), [f'{c}  {p}' for c, p in rows], fontsize=10)
    for lab, key in zip(ax.get_yticklabels(), rows):
        if key in HIGHLIGHT:
            lab.set_fontweight('bold')
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ['0', '25', '50', '75', '100%'])
    ax.tick_params(axis='y', length=0)
    ax.grid(False)
    ax.spines['left'].set_visible(False)
    for b in range(4, len(rows), 4):
        ax.axhline(b - 0.5, color='white', lw=3)
    for b in (12, 24):
        ax.axhline(b - 0.5, color='#555555', lw=1)
    for k_i, k in enumerate(INDUSTRY_ORDER):
        ax.text(-0.205, k_i * 12 + 5.5, INDUSTRIES[k][0], transform=ax.get_yaxis_transform(), ha='center',
                va='center', fontsize=13, fontweight='bold', rotation=90, color='#333333')
    for key in HIGHLIGHT:
        i = rows.index(key)
        ax.add_patch(Rectangle((0, i - 0.45), 100, 0.9, fill=False, ec='#111111', lw=2, clip_on=False))
    ax.set_xlabel('변화 문장 쌍의 섹션 구성 (%)', fontsize=12)
    handles = [Patch(color=SECTION_COLORS[s], label=s + (' (표 성격)' if s in TABLE_SECTIONS else ''))
               for s in SECTIONS]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=11,
               bbox_to_anchor=(0.54, 0.995), handlelength=1.3, columnspacing=1.2)
    fig.text(0.02, 0.012, 'tfidf_masked_th0.3.csv 전체 · 변화 문장 쌍(5개 유형)만 · 섹션 = 「II. 사업의 내용」 하위 항목 · '
             '오른쪽 = 변화 문장 쌍 수 · 굵은 테두리 = 합병 연도', fontsize=10, color='#666666')
    save(fig, 'fig_change_by_section')


# ---------------------------------------------------------------- 3. 케이스 3개
def fig_cases(total, types):
    fig, ax = plt.subplots(figsize=(9, 5.6))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.86, bottom=0.24)
    order = ['NEW', 'EXPANDED', 'MODIFIED', 'REDUCED', 'REMOVED']
    for i, (c, p, why) in enumerate(CASES):
        key, bottom = (c, p), 0
        for t in order:
            v = 100 * types[key][t] / total[key]
            ax.bar(i, v, 0.55, bottom=bottom, color=COLORS[t], lw=0)
            if v >= 5:
                ax.text(i, bottom + v / 2, f'{t} {v:.0f}%', ha='center', va='center', fontsize=10,
                        color='white' if t in ('NEW', 'REMOVED') else '#222222')
            bottom += v
        ax.text(i, bottom + 2, f'{bottom:.0f}%', ha='center', va='bottom', fontsize=17, fontweight='bold')
    ax.set_xticks(range(len(CASES)), [f'{c}\n{p}' for c, p, _ in CASES], fontsize=13)
    for i, (_, _, why) in enumerate(CASES):
        ax.text(i, -0.19, why, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=11,
                color='#666666')
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, 100)
    ax.set_ylabel('전체 비교 문장 쌍 대비 (%)', fontsize=12)
    ax.set_xlim(-0.6, len(CASES) - 0.4)
    fig.legend(handles=[Patch(color=COLORS[t], label=t) for t in order], loc='upper center', ncol=5,
               frameon=False, fontsize=11, bbox_to_anchor=(0.53, 0.985))
    fig.text(0.02, 0.015, '변화율 = 5개 변화 유형 ÷ 전체 비교 문장 쌍 · 기업·연도쌍 막대 아래 = PROJECT.md의 원인 설명(근거 문장 확인 필요)',
             fontsize=9, color='#666666')
    save(fig, 'fig_case_change_rates')


def main():
    setup_style()
    total, types, sections = load()
    m = fig_heatmap(total, types)
    fig_sections(sections)
    fig_cases(total, types)
    with (FIG_DIR / 'change_concentration.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['company', 'period', 'total_pairs', 'changed_pairs', 'change_rate_pct', *TYPES, *SECTIONS])
        for i, c in enumerate(companies()):
            for j, p in enumerate(PERIODS):
                k = (c, p)
                w.writerow([c, p, total[k], sum(types[k][t] for t in TYPES), round(m[i, j], 1),
                            *(types[k][t] for t in TYPES), *(sections[k][s] for s in SECTIONS)])
    for c, p, _ in CASES:
        k = (c, p)
        top = sections[k].most_common(2)
        print(f'{c} {p}: 변화율 {rate(total, types, k):.0f}% · 상위 섹션 ' +
              ', '.join(f'{s} {n}건({100 * n / sum(sections[k].values()):.0f}%)' for s, n in top))


if __name__ == '__main__':
    main()
