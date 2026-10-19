"""R12 - Page type and topic effects.

Question. Do some kinds of pages get cited more once other features are accounted for, and do
the yes/no page features work differently by user intent or topic?

1. Page-type effects: read from the R2 model (vs 'article'), all three samples.
2. For each intent (learning / comparing / ready to buy, from prompt_labels.py) and each topic:
   y_g = log2(1 + citations the URL received from prompts of group g), same rows and model
   as R2 (all pages, website indicators, HC1). The yes/no features' effects are shown as a
   heatmap (rows = features, columns = intent or topic). The prompt-match features are
   included for comparison.

Outputs: results/r12_page_types.csv, results/r12_by_group.csv,
         figures/r12_page_types.png, figures/r12_heatmap.png
"""
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import plotstyle
from plotstyle import plt, MUTED, INK2
from common import unit_factors, unit_of, pct_in_unit, TABLES, RESULTS, FIGURES, read_table, drop_excluded
from fe import fe_ols
from r02_main_model import prepare, FEATURES, LABEL

SHOW = ['has_faq', 'has_summary_box', 'has_author', 'has_schema', 'title_match', 'prompt_match_words']
DIVERGING = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])


def group_counts(urls):
    c = drop_excluded(read_table('citations.csv', usecols=[
        'answer_id', 'date', 'engine', 'prompt_id', 'topic', 'url', 'is_redirect', 'repeat_in_answer']))
    c = c[~c.is_redirect & ~c.repeat_in_answer & c.url.isin(urls)]
    lab = pd.read_csv(TABLES / 'prompt_labels.csv', usecols=['prompt_id', 'intent'])
    c = c.merge(lab, on='prompt_id', how='left')
    by_intent = c.pivot_table(index='url', columns='intent', aggfunc='size', fill_value=0)
    by_topic = c.pivot_table(index='url', columns='topic', aggfunc='size', fill_value=0)
    return by_intent.reindex(urls, fill_value=0), by_topic.reindex(urls, fill_value=0)


def main():
    r2 = pd.read_csv(RESULTS / 'r02_main_model.csv')
    pt = r2[r2.kind == 'page type'].copy()
    pt['page_type'] = pt.term.str.removeprefix('type_')
    pt = pt[['sample', 'page_type', 'coef', 'se', 'p', 'pct_change_per_sd', 'ci_pct_low', 'ci_pct_high',
             'n_urls']].rename(columns={'pct_change_per_sd': 'pct_vs_article'})
    pt.to_csv(RESULTS / 'r12_page_types.csv', index=False)

    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    X = FEATURES + flags + types
    bi, bt = group_counts(d.url.tolist())
    rows = []
    fac = unit_factors(d)
    for kind, tab in (('intent', bi), ('topic', bt)):
        for g in tab.columns:
            s = d.assign(y=np.log2(1 + tab[g].to_numpy()))
            s = s[s.groupby('domain').y.transform('size') >= 2]
            res, info = fe_ols(s, 'y', X, 'domain')
            for f in FEATURES:
                rows.append(dict(group_type=kind, group=g, term=f, coef=res.loc[f, 'coef'],
                                 se=res.loc[f, 'se'], p=res.loc[f, 'p'],
                                 pct_change_per_sd=100 * (2 ** res.loc[f, 'coef'] - 1), unit=unit_of(f),
                                 pct_change=pct_in_unit(res.loc[f, 'coef'], f, fac),
                                 n_urls=info['n_urls'], urls_cited_by_group=int((tab[g] > 0).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'r12_by_group.csv', index=False)

    # page-type chart (all pages)
    g = pt[pt['sample'] == 'all pages'].sort_values('pct_vs_article')
    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = np.arange(len(g))
    sig = g.p < 0.05
    ax.hlines(y, g.ci_pct_low, g.ci_pct_high, color='#9ec5f4', lw=2)
    ax.plot(g.pct_vs_article[sig], y[sig.values], 'o', color='#2a78d6', ms=6, mec='#fcfcfb', label='p < 0.05')
    ax.plot(g.pct_vs_article[~sig], y[~sig.values], 'o', color='#fcfcfb', mec='#2a78d6', mew=1.5, ms=6,
            label='not significant')
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, g.page_type)
    ax.set_xlabel('% more citations than an article on the same website')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right')
    ax.set_title('R12: page types vs articles (all pages, other features held fixed)')
    fig.savefig(FIGURES / 'r12_page_types.png')
    plt.close(fig)

    # heatmap
    cols = [('intent', c) for c in ['learning', 'comparing', 'ready to buy']] + \
           [('topic', c) for c in sorted(bt.columns)]
    cols = [(k, c) for k, c in cols if ((out.group_type == k) & (out.group == c)).any()]   # groups too small are skipped
    cell = lambda k, c, f, v: (lambda s: s.iloc[0] if len(s) else np.nan)(
        out[(out.group_type == k) & (out.group == c) & (out.term == f)][v])
    M = np.array([[cell(k, c, f, 'pct_change') for k, c in cols] for f in SHOW])
    P = np.array([[cell(k, c, f, 'p') for k, c in cols] for f in SHOW])
    lim = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(11, 3.9))
    im = ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:+.0f}{"*" if P[i, j] < 0.05 else ""}', ha='center', va='center',
                    fontsize=8, color='#0b0b0b')
    ax.set_xticks(range(len(cols)), [c for _, c in cols], rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(SHOW)), [LABEL[f] for f in SHOW])
    ax.axvline(2.5, color='#fcfcfb', lw=4)
    ax.axhline(3.5, color='#fcfcfb', lw=4)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.set_label('% more citations per +1 SD', fontsize=8)
    ax.set_title('R12: feature effects by user intent (left) and topic (right); * p < 0.05')
    fig.savefig(FIGURES / 'r12_heatmap.png')
    plt.close(fig)

    print(pt[pt['sample'] == 'all pages'][['page_type', 'pct_vs_article', 'p']].round(3).to_string(index=False))
    show = out[out.term.isin(SHOW)].assign(v=lambda x: x['pct_change'].round(1).astype(str)
                                           + np.where(x.p < 0.05, '*', ''))
    print(show.pivot(index='term', columns='group', values='v').to_string())


if __name__ == '__main__':
    main()
