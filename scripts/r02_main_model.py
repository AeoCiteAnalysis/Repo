"""R2 - Main model: what predicts citation count?

Question. Which page features go with more citations when we compare pages on the
same website?

y = log2(1 + citations) for URLs cited >= 2 times (url_table.csv). Regression of y on
the page features (z-scored) + page type (vs 'article') + one indicator per website,
robust HC1 standard errors (fe.py; identical to the guide's `+ C(domain)` formula).
Only URLs with downloaded page features are used, and only websites with >= 2 such URLs.
Missing feature values are filled with the median and get a `<name>_missing` flag
(the guide's rule for features with missing values; page_age_days is ~40 % missing).

Samples: all pages; own + competitor pages (pages brands can change); third-party pages.

Outputs
  results/r02_main_model.csv     sample, term, coef, se, ci, p, pct_change_per_sd, n_urls, n_websites
  results/r02_model_info.csv     sample sizes and within-website R^2
  figures/r02_effects.png        dot-and-interval chart of feature effects
"""
import numpy as np
import pandas as pd

import plotstyle
from plotstyle import plt, INK2, MUTED
from common import unit_factors, unit_of, pct_in_unit, RESULTS, FIGURES, read_table
from fe import fe_ols

FEATURES = ['prompt_match_words', 'title_match', 'intro_match', 'best_paragraph_match',
            'word_count', 'outbound_links', 'n_headings', 'n_tables', 'n_lists',
            'has_faq', 'has_summary_box', 'has_author', 'has_schema', 'page_age_days']
LABEL = {'prompt_match_words': 'word overlap with prompts', 'title_match': 'title matches a prompt',
         'intro_match': 'first paragraph matches', 'best_paragraph_match': 'best paragraph matches',
         'word_count': 'word count', 'outbound_links': 'outbound links',
         'n_headings': 'headings', 'n_tables': 'tables', 'n_lists': 'lists',
         'has_faq': 'has FAQ', 'has_summary_box': 'has summary box', 'has_author': 'has author',
         'has_schema': 'has structured data', 'page_age_days': 'page age'}
SAMPLES = {'all pages': None, 'own + competitor': ['own', 'competitor'],
           'third party': ['third_party']}
REF_TYPE = 'article'


def prepare(t):
    """Rows with page features; median-fill + missing flags; page-type dummies."""
    d = t[t.word_count.notna()].copy()
    flags = []
    for f in FEATURES:
        if d[f].isna().any():
            flag = d[f].isna().astype(float)
            # one flag per distinct missing pattern (intro and best paragraph are missing together)
            if not any(flag.equals(d[g]) for g in flags):
                d[f + '_missing'] = flag
                flags.append(f + '_missing')
            d[f] = d[f].fillna(d[f].median())
    types = sorted(x for x in d.page_type.dropna().unique() if x != REF_TYPE)
    for pt in types:
        d['type_' + pt] = (d.page_type == pt).astype(float)
    return d, flags, ['type_' + pt for pt in types]


def fit_sample(d, roles, flags, types):
    s = d if roles is None else d[d.role.isin(roles)]
    X = FEATURES + flags + types
    res, info = fe_ols(s, 'y', X, 'domain')
    res['pct_change_per_sd'] = 100 * (2 ** res.coef - 1)
    fac = unit_factors(d)                       # yes/no features reported as has vs not
    res['unit'] = [unit_of(t) for t in res.index]
    res['pct_change'] = [pct_in_unit(b, t, fac) for t, b in res.coef.items()]
    res['ci_pct_low'] = [pct_in_unit(b, t, fac) for t, b in res.ci_low.items()]
    res['ci_pct_high'] = [pct_in_unit(b, t, fac) for t, b in res.ci_high.items()]
    res['kind'] = ['feature' if x in FEATURES else 'missing flag' if x in flags else 'page type'
                   for x in res.index]
    return res, info


def plot(res):
    f = res[res.kind == 'feature']
    order = (f[f['sample'] == 'all pages'].sort_values('coef').term.tolist())
    samples = list(SAMPLES)
    fig, axes = plt.subplots(1, len(samples), figsize=(12, 5.2), sharey=True, sharex=True)
    for ax, s in zip(axes, samples):
        g = f[f['sample'] == s].set_index('term').reindex(order)
        y = np.arange(len(order))
        sig = g.p < 0.05
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.hlines(y, g.ci_pct_low, g.ci_pct_high, color='#9ec5f4', lw=2)
        ax.plot(g['pct_change'][sig], y[sig.values], 'o', ms=6, color='#2a78d6',
                mec='#fcfcfb', mew=1.2, label='p < 0.05')
        ax.plot(g['pct_change'][~sig], y[~sig.values], 'o', ms=6, color='#fcfcfb',
                mec='#2a78d6', mew=1.5, label='not significant')
        n = g.n_urls.iloc[0]; w = g.n_websites.iloc[0]
        ax.set_title(f'{s}\n{n:,} URLs, {w:,} websites', fontsize=9.5)
        ax.set_xlabel('% more citations per +1 SD\n(yes/no features: has vs not)')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(np.arange(len(order)), [LABEL[o] for o in order])
    axes[-1].legend(loc='lower right')
    fig.suptitle('R2: page features and citation count, comparing pages on the same website',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'r02_effects.png')
    plt.close(fig)


def main():
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    out, infos = [], []
    for name, roles in SAMPLES.items():
        res, info = fit_sample(d, roles, flags, types)
        res['sample'] = name
        res['n_urls'], res['n_websites'] = info['n_urls'], info['n_websites']
        out.append(res.rename_axis('term').reset_index())
        infos.append(dict(sample=name, **{k: v for k, v in info.items() if k != 'dropped'},
                          dropped_terms=';'.join(info['dropped'])))
    res = pd.concat(out, ignore_index=True)
    cols = ['sample', 'term', 'kind', 'coef', 'se', 'ci_low', 'ci_high', 'p', 'unit', 'pct_change', 'pct_change_per_sd',
            'ci_pct_low', 'ci_pct_high', 'n_urls', 'n_websites']
    res[cols].to_csv(RESULTS / 'r02_main_model.csv', index=False)
    pd.DataFrame(infos).to_csv(RESULTS / 'r02_model_info.csv', index=False)
    plot(res)
    print(pd.DataFrame(infos).to_string(index=False))
    show = res[res.kind == 'feature'].pivot(index='term', columns='sample',
                                            values='pct_change').round(1)
    pv = res[res.kind == 'feature'].pivot(index='term', columns='sample', values='p')
    print('\n% more citations per +1 SD; yes/no features: has vs not (* p<0.05):')
    print(show.astype(str).where(pv >= 0.05, show.astype(str) + '*')[list(SAMPLES)].to_string())
    print('\npage types vs article (% difference, all pages):')
    pt = res[(res.kind == 'page type') & (res['sample'] == 'all pages')]
    print(pt[['term', 'pct_change_per_sd', 'p']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
