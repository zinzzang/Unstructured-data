"""5개 figure 생성 스크립트 (디자이너 전달용).

출력:
  figures/fig0_semiconductor_ext.png/.svg
  figures/fig0_biopharma_ext.png/.svg
  figures/fig0_battery_ext.png/.svg
  figures/fig_change_rate_heatmap.png/.svg
  figures/eda_change_types.png/.svg

사용: python scripts/plots/export_five_figures.py
입력: data/results_v2/tfidf_masked_th0.3.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_industry_change_flow as flow
import plot_change_concentration as conc
import plot_eda as eda


def main():
    print(f'[폰트] {flow.setup_style()}')

    # fig0_{industry}_ext — 외부 지표 보조축 버전
    counts, _, _, by_company = flow.load_counts()
    for key in flow.INDUSTRIES:
        if key in flow.EXTERNAL:
            flow.plot_industry(key, counts[key],
                               title=flow.TITLES_EXT.get(key) or '',
                               ext=flow.EXTERNAL[key])
            print(f'[완료] fig0_{key}_ext.png')

    # fig_change_rate_heatmap (fig_heatmap이 내부에서 save 호출)
    total, types, *_ = conc.load()
    conc.fig_heatmap(total, types)
    print('[완료] fig_change_rate_heatmap.png')

    # eda_change_types
    rows, _, _, types_count = eda.load()
    eda.fig_types(types_count, len(rows))
    print('[완료] eda_change_types.png')


if __name__ == '__main__':
    main()
