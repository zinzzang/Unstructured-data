"""Per-company sentence counts: preprocessed sentences vs TF-IDF sentence pairs vs changed pairs (static figure, English).

Usage: python scripts/plot_company_sentence_counts.py
Input: data/processed/preprocessing_v2/{industry}/paragraphs.jsonl  (sum of nested sentences, 5 years 2021-2025)
       data/results_v2/tfidf_masked_th0.3.csv                       (one row = one adjacent-year sentence pair)
       Both exclude EXCLUDED_SECTIONS (table-like sections), same rule as plot_industry_change_flow.py.
Output: figures/fig_company_sentence_counts.png / .svg, figures/company_sentence_counts.csv
"""
import csv
import json
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_industry_change_flow import (DPI, EXCLUDED_SECTIONS, FIG_DIR, FIGSIZE, INDUSTRIES, INPUT_CSV, ROOT,
                                       TITLE_FS, TYPES, setup_style)

TITLE = ''   # conclusion-style title (empty = no title)
INDUSTRY_EN = {'semiconductor': 'Semiconductor', 'biopharma': 'Bio-pharma', 'battery': 'Secondary Battery'}
COMPANY_EN = {
    '삼성전자': 'Samsung\nElectronics', 'SK하이닉스': 'SK hynix', '한미반도체': 'Hanmi\nSemiconductor',
    '삼성바이오로직스': 'Samsung\nBiologics', '셀트리온': 'Celltrion', '유한양행': 'Yuhan',
    'LG에너지솔루션': 'LG Energy\nSolution', '삼성SDI': 'Samsung SDI', 'SK이노베이션': 'SK\nInnovation',
}
SECTION_EN = {'원재료 및 생산설비': 'Raw Materials & Production Facilities',
              '주요계약 및 연구개발활동': 'Major Contracts & R&D Activities'}
SERIES = [('pre', 'Preprocessed sentences (5 years)', '#B4B2A9'),
          ('tfidf', 'TF-IDF sentence pairs (4 year-pairs)', '#85B7EB'),
          ('changed', 'Changed pairs (NEW·EXPANDED·MODIFIED·REDUCED·REMOVED)', '#2a78d6')]


def excluded(path_text):
    return any(s in path_text for s in EXCLUDED_SECTIONS)


def load():
    pre, tfidf, changed = Counter(), Counter(), Counter()
    for ind in INDUSTRIES:
        with (ROOT / f'data/processed/preprocessing_v2/{ind}/paragraphs.jsonl').open(encoding='utf8') as f:
            for line in f:
                p = json.loads(line)
                if not excluded(' > '.join(p.get('title_path') or [])):
                    pre[p['company']] += len(p.get('sentences') or [])
    with INPUT_CSV.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if excluded(r['old_path']) or excluded(r['new_path']):
                continue
            tfidf[r['company']] += 1
            if r['change_type'] in TYPES:
                changed[r['company']] += 1
    return {'pre': pre, 'tfidf': tfidf, 'changed': changed}


def main():
    setup_style()
    data = load()
    rows = [(key, c) for key, (_, companies) in INDUSTRIES.items() for c in companies]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with (FIG_DIR / 'company_sentence_counts.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['industry', 'company', 'preprocessed_sentences', 'tfidf_pairs', 'changed_pairs'])
        for key, c in rows:
            w.writerow([INDUSTRY_EN[key], COMPANY_EN[c].replace('\n', ' '), data['pre'][c], data['tfidf'][c],
                        data['changed'][c]])
            print(f'{INDUSTRY_EN[key]:18} {COMPANY_EN[c].replace(chr(10), " "):22} pre {data["pre"][c]:>6,} · '
                  f'tfidf {data["tfidf"][c]:>6,} · changed {data["changed"][c]:>6,}')

    fig, ax = plt.subplots(figsize=FIGSIZE)
    top = 0.80 if TITLE else 0.87
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.22, top=top)
    bar_w, step, gap_group = 0.27, 1.0, 0.55
    x, xs, prev = 0.0, [], None
    for key, _ in rows:
        if prev is not None and key != prev:
            x += gap_group
        xs.append(x)
        prev = key
        x += step
    ymax = max(max(d.values()) for d in data.values())
    for (key, c), xc in zip(rows, xs):
        for j, (k, _, color) in enumerate(SERIES):
            v = data[k][c]
            xx = xc + (j - 1) * bar_w
            ax.bar(xx, v, bar_w * 0.92, color=color, lw=0)
            ax.text(xx, v + ymax * 0.012, f'{v:,}', ha='center', va='bottom', fontsize=10, rotation=90,
                    color='#222222', fontweight='bold' if k == 'changed' else 'normal')
    ax.set_xticks(xs, [COMPANY_EN[c] for _, c in rows], fontsize=12)
    ax.tick_params(axis='x', length=0, pad=6)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylim(0, ymax * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
    ax.set_ylabel('Sentences / sentence pairs')
    # industry labels under company names + separators
    start = 0
    for key, (_, companies) in INDUSTRIES.items():
        seg = xs[start:start + len(companies)]
        ax.text((seg[0] + seg[-1]) / 2, -0.15, INDUSTRY_EN[key], transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=15, fontweight='bold', color='#333333')
        if start:
            ax.axvline(seg[0] - (step + gap_group) / 2, color='#cccccc', lw=0.8)
        start += len(companies)

    fig.legend(handles=[Patch(color=col, label=lab) for _, lab, col in SERIES], loc='upper center', ncol=3,
               frameon=False, fontsize=12, bbox_to_anchor=(0.5, top + 0.11), handlelength=1.4, columnspacing=1.6)
    sections = ', '.join(f'"{SECTION_EN[s]}"' for s in EXCLUDED_SECTIONS)
    fig.text(0.03, 0.045, f'Note: Table-like sections excluded from both counts ({sections}). TF-IDF threshold 0.3.',
             fontsize=11, color='#666666')
    fig.text(0.03, 0.015, 'TF-IDF pairs compare adjacent years (21→22 … 24→25), so mid-year sentences are compared '
             'twice and pair counts can exceed sentence counts.', fontsize=11, color='#666666')
    if TITLE:
        fig.suptitle(TITLE, x=0.03, y=0.965, ha='left', va='top', fontsize=TITLE_FS, fontweight='bold')
    for fmt in ('png', 'svg'):
        fig.savefig(FIG_DIR / f'fig_company_sentence_counts.{fmt}', dpi=DPI)
    plt.close(fig)
    print('[saved] figures/fig_company_sentence_counts.png, .svg')


if __name__ == '__main__':
    main()
