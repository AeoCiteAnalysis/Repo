"""R10 - Where on the page is the matching text?

Question. Do engines prefer pages whose best-matching paragraph is near the top?

best_paragraph_depth (Step A6): position of the paragraph that best matches a prompt,
0 = first paragraph, 1 = last. Pages with fewer than 3 paragraphs are left out (depth is
then mostly 0 or 1 by construction). One row per cited page (url_table_all). Test: median
differs from 0.5 (one-sample Wilcoxon on depth - 0.5), overall, per engine (pages that engine
cited) and per page type. Also: is depth linked with citation count within websites?

Outputs: results/r10_paragraph_depth.csv, figures/r10_depth_hist.png
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import plotstyle
from plotstyle import plt, ENGINE_COLOR, MUTED, INK2
from common import TABLES, RESULTS, FIGURES, ENGINES, ENGINE_LABEL, read_table
from fe import fe_ols

MIN_PARAS = 3


def n_paragraphs():
    n = {}
    with open(TABLES / 'page_text.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            n[r['url']] = len(r['paragraphs'])
    return pd.Series(n, name='n_paragraphs')


def test(depth, group, label):
    depth = depth.dropna()
    if len(depth) < 20:
        return None
    stat = wilcoxon(depth - 0.5)
    return dict(group=group, subgroup=label, n_pages=len(depth), median=depth.median(),
                mean=depth.mean(), share_top_quarter=(depth <= 0.25).mean(),
                share_bottom_quarter=(depth >= 0.75).mean(), wilcoxon_p=stat.pvalue)


def main():
    t = read_table('url_table_all.csv', usecols=lambda c: c in {
        'url', 'domain', 'y', 'page_type', 'best_paragraph_depth_raw', 'citation_count',
        'count_OPENAI', 'count_GEMINI', 'count_PERPLEXITY'})
    t = t.join(n_paragraphs(), on='url')
    t = t[(t.n_paragraphs >= MIN_PARAS) & t.best_paragraph_depth_raw.notna()].copy()
    t['depth'] = t.best_paragraph_depth_raw
    rows = [test(t.depth, 'all', 'all cited pages')]
    for e in ENGINES:
        rows.append(test(t.loc[t[f'count_{e}'] > 0, 'depth'], 'engine', ENGINE_LABEL[e]))
    for pt, g in t.groupby('page_type'):
        rows.append(test(g.depth, 'page type', pt))
    out = pd.DataFrame([r for r in rows if r])
    # within-website link between depth and citations (URLs cited >= 2)
    m = t[t.citation_count >= 2].copy()
    m['depth_sd'] = (m.depth - m.depth.mean()) / m.depth.std()
    res, info = fe_ols(m, 'y', ['depth_sd'], 'domain')
    out = pd.concat([out, pd.DataFrame([dict(
        group='within-website model', subgroup='y on depth (per SD), URLs cited >= 2',
        n_pages=info['n_urls'], mean=100 * (2 ** res.coef.iloc[0] - 1), wilcoxon_p=res.p.iloc[0])])])
    out.to_csv(RESULTS / 'r10_paragraph_depth.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.linspace(0, 1, 21)
    axes[0].hist(t.depth, bins=bins, color='#2a78d6', edgecolor='#fcfcfb', linewidth=1.5)
    axes[0].axvline(t.depth.median(), color=INK2, lw=1.2, ls='--')
    axes[0].text(t.depth.median() + 0.02, axes[0].get_ylim()[1] * 0.92,
                 f'median {t.depth.median():.2f}', fontsize=8, color=INK2)
    axes[0].set_title(f'All cited pages (n = {len(t):,})')
    axes[0].set_xlabel('position of best-matching paragraph (0 = top, 1 = bottom)')
    axes[0].set_ylabel('pages')
    for e in ENGINES:
        dd = t.loc[t[f'count_{e}'] > 0, 'depth']
        h, _ = np.histogram(dd, bins=bins, density=True)
        axes[1].plot((bins[:-1] + bins[1:]) / 2, h, color=ENGINE_COLOR[e],
                     label=f'{ENGINE_LABEL[e]} (median {dd.median():.2f}, n = {len(dd):,})')
    axes[1].axhline(1, color=MUTED, lw=0.8, ls='--')
    axes[1].set_title('By engine (density; dashed = no preference)')
    axes[1].set_xlabel('position of best-matching paragraph')
    axes[1].legend(loc='upper right')
    fig.suptitle('R10: where on the page is the text that matches the question?',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'r10_depth_hist.png')
    plt.close(fig)
    print(out.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
