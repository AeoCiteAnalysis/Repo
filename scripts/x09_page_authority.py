"""X9 - Page-level authority (DataForSEO) and AI citations.

page_authority.py gives, per page: DataForSEO Rank (0-1,000, backlink-based) and the number of
referring domains (distinct websites linking to the page). Measures used:
  pa_links     log2(1 + referring domains), z-scored
  pa_rank      DataForSEO Rank, z-scored
  pa_any       has at least one referring domain (0/1; reported as has vs not)

1 Citations (R2 design): the R2 model (page features + flags + page types, website fixed effects) plus the
  authority measures, per sample; the same pooled across websites (no website indicators) for contrast;
  per engine; and the share of the within-website variation explained (mixed model) with and without.
2 ChatGPT retrieval and selection (X6 rows): authority z-scored within each result list, added to the
  X6 retrieval model (-log2 rank, list fixed effects) and to the selection model (conditional logit).
Outputs: results[/<ds>]/x09_citations.csv, x09_variance.csv, x09_engines.csv, x09_retrieval_selection.csv
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import TABLES, RESULTS, DATASET, ENGINES, ENGINE_LABEL, read_table
from fe import fe_ols
from r02_main_model import prepare, FEATURES, SAMPLES
from e01_engines_breadth_depth import vc

warnings.filterwarnings('ignore')
PA = ['pa_links', 'pa_rank', 'pa_any']


def add_pa(d):
    pa = pd.read_csv(TABLES / 'page_authority.csv')
    d = d.merge(pa[['url', 'pa_rank', 'pa_referring_domains']], on='url', how='left')
    d['pa_known'] = d.pa_referring_domains.notna()
    d['pa_links'] = np.log2(1 + d.pa_referring_domains.fillna(0))
    d['pa_rank'] = d.pa_rank.fillna(0)
    d['pa_any'] = (d.pa_referring_domains.fillna(0) > 0).astype(float)
    for c in ('pa_links', 'pa_rank'):
        d[c] = (d[c] - d[c].mean()) / d[c].std()
    return d


def pct(b, term):
    return 100 * (2 ** b - 1)


def citations():
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    d = add_pa(d)
    X = FEATURES + flags + types
    rows = []
    for name, roles in SAMPLES.items():
        s = d if roles is None else d[d.role.isin(roles)]
        s = s[s.groupby('domain').y.transform('size') >= 2]
        for pa in (['pa_links'], ['pa_rank'], ['pa_any']):
            r, info = fe_ols(s, 'y', X + pa, 'domain')
            v = pa[0]
            rows.append(dict(sample=name, measure=v, design='within website', coef=r.loc[v, 'coef'], se=r.loc[v, 'se'],
                             p=r.loc[v, 'p'], pct=pct(r.loc[v, 'coef'], v), n_urls=info['n_urls'],
                             title_pct=pct(r.loc['title_match', 'coef'], 't'),
                             best_para_pct=pct(r.loc['best_paragraph_match', 'coef'], 'b')))
            m = sm.OLS(s.y, sm.add_constant(s[X + pa])).fit(cov_type='cluster', cov_kwds={'groups': s.domain.astype('category').cat.codes})
            rows.append(dict(sample=name, measure=v, design='pooled (no website indicators)', coef=m.params[v], se=m.bse[v],
                             p=m.pvalues[v], pct=pct(m.params[v], v), n_urls=len(s)))
    C = pd.DataFrame(rows)
    C['unit'] = np.where(C.measure == 'pa_any', 'has vs not', 'per SD')
    C.to_csv(RESULTS / 'x09_citations.csv', index=False)
    print(C.round(3).to_string(index=False))

    s = d[d.groupby('domain').y.transform('size') >= 2].reset_index(drop=True)
    V = []
    for name, cols in (('page variables', X), ('page variables + page authority', X + ['pa_links', 'pa_rank', 'pa_any'])):
        ws, ex = vc(s, 'y', cols)
        V.append(dict(model=name, within_share=ws, explained_within=ex, n_urls=len(s)))
    V = pd.DataFrame(V)
    V.to_csv(RESULTS / 'x09_variance.csv', index=False)
    print(V.round(4).to_string(index=False))

    E = []
    for e in ENGINES:
        s2 = s.assign(y=np.log2(1 + s[f'count_{e}']))
        r, _ = fe_ols(s2, 'y', X + ['pa_links'], 'domain')
        E.append(dict(engine=ENGINE_LABEL[e], coef=r.loc['pa_links', 'coef'], p=r.loc['pa_links', 'p'],
                      pct=pct(r.loc['pa_links', 'coef'], 'pa_links')))
    E = pd.DataFrame(E)
    E.to_csv(RESULTS / 'x09_engines.csv', index=False)
    print(E.round(3).to_string(index=False))
    print(f"pages with page authority data: {100 * s.pa_known.mean():.0f} %; with >= 1 referring domain: {100 * s.pa_any.mean():.0f} %")


def retrieval_selection():
    from statsmodels.discrete.conditional_models import ConditionalLogit
    p = TABLES / 'x06_rows.csv'
    if not p.exists():
        return
    L = pd.read_csv(p)
    pa = pd.read_csv(TABLES / 'page_authority.csv')
    L = L.merge(pa[['url', 'pa_rank', 'pa_referring_domains']], on='url', how='left')
    L['pa_links_raw'] = np.log2(1 + L.pa_referring_domains.fillna(0))
    L['pa_any'] = (L.pa_referring_domains.fillna(0) > 0).astype(float)
    g = L.groupby('list_id')
    for c, raw in (('pa_links_z', 'pa_links_raw'), ('pa_rank_z', 'pa_rank')):
        L[raw] = L[raw].fillna(0)
        L[c] = ((L[raw] - g[raw].transform('mean')) / g[raw].transform('std').replace(0, np.nan)).fillna(0)
    base = ['text_missing', 'is_reddit', 'emb_best_z', 'emb_title_z', 'bm25_z', 'q_best_z', 'log_words_z', 'log_headings_z',
            'has_faq', 'has_summary_box', 'has_author', 'has_schema', 'type_list', 'type_comparison', 'role_brand', 'authority_z']
    base = [b for b in base if b in L]
    out = []
    y = -L.log_rank
    for name, X in (('retrieval: X6 features', base), ('retrieval: + page authority', base + ['pa_links_z', 'pa_rank_z', 'pa_any'])):
        X = [x for x in X if L[x].std() > 0]
        Xd = L[X] - L.groupby('list_id')[X].transform('mean')
        yd = y - y.groupby(L.list_id).transform('mean')
        f = sm.OLS(yd, Xd).fit(cov_type='cluster', cov_kwds={'groups': L.list_id.astype('category').cat.codes})
        out.append(dict(model=name, fit=f.rsquared, **{f'coef_{v}': f.params[v] for v in X if v.startswith('pa_')},
                        **{f'p_{v}': f.pvalues[v] for v in X if v.startswith('pa_')}))
    d = L[L.groupby('list_id').cited.transform('nunique') == 2]
    for name, X in (('selection: rank + X6 features', ['log_rank', 'multi_call'] + base),
                    ('selection: + page authority', ['log_rank', 'multi_call'] + base + ['pa_links_z', 'pa_rank_z', 'pa_any'])):
        X = [x for x in X if d[x].std() > 0]
        m = ConditionalLogit(d.cited.astype(float), d[X], groups=d.list_id)
        r = m.fit(disp=0, method='bfgs', maxiter=1000)
        out.append(dict(model=name, fit=1 - r.llf / m.loglike(np.zeros(len(X))),
                        **{f'or_{v}': np.exp(r.params[v]) for v in X if v.startswith('pa_')},
                        **{f'p_{v}': r.pvalues[v] for v in X if v.startswith('pa_')}))
    R = pd.DataFrame(out)
    R.to_csv(RESULTS / 'x09_retrieval_selection.csv', index=False)
    print(R.round(4).T.to_string())


if __name__ == '__main__':
    print('#####', DATASET)
    citations()
    retrieval_selection()
