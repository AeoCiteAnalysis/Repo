"""X11 - Brand level: when is a brand named in an AI answer, and what supports the mention?

Unit: one answer x one tracked brand (prompt_labels.BRANDS) that the question itself does NOT name.
Outcome: the answer text names the brand (same patterns as elsewhere; pfetch's own mention table is
reported for comparison).
Support for the brand in the answer's citations:
  own_cited       the answer cites the brand's own website (brand_domains: own/competitor with that brand)
  third_naming    the answer cites another website's page whose downloaded text (title + first 50
                  paragraphs, page_text.jsonl) names the brand
  n_uncertain     cited pages without downloaded text (their content is unknown)
Model prior (X10): the engine's own model names the brand for this question without search
  (ChatGPT -> gpt-5.4-nano prior, Gemini -> Gemini 3.1 Flash Lite prior; Perplexity: none).

1 Decomposition of mentions per engine: own page cited / third-party page naming it cited (no own page) /
  no cited page naming it (split by prior and by whether some cited pages had no text).
2 Logit of mention on own_cited, third_naming, prior, log2(1 + citations in the answer), with brand
  indicators, per engine; standard errors clustered by prompt.
Outputs: results[/<ds>]/x11_decomposition.csv, x11_model.csv, x11_rates.csv
"""
import json
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import TABLES, RESULTS, DATASET, ENGINE_LABEL, read_table, drop_excluded
from prompt_labels import BRANDS

PRIOR = {'OPENAI': 'x10_prior.csv', 'GEMINI': 'x10_prior_gemini.csv'}


def brand_of_label(label):
    if not isinstance(label, str):
        return None
    low = label.lower()
    for k, rx in BRANDS.items():
        if re.search(rx, low) or low.startswith(k.lower().split()[0]):
            return k
    return None


def page_brands():
    out = {}
    with open(TABLES / 'page_text.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            t = (r.get('title') or '') + ' ' + ' '.join((r.get('paragraphs') or [])[:50])
            low = t.lower()
            out[r['url']] = {k for k, rx in BRANDS.items() if re.search(rx, low)}
    return out


def main():
    a = drop_excluded(read_table('answers.csv', usecols=['answer_id', 'engine', 'prompt_id', 'answer_text']))
    lab = pd.read_csv(TABLES / 'prompt_labels.csv', usecols=['prompt_id', 'prompt'])
    q = dict(zip(lab.prompt_id, lab.prompt.str.lower()))
    c = drop_excluded(read_table('citations.csv', usecols=['answer_id', 'url', 'role', 'brand', 'is_redirect', 'repeat_in_answer']))
    c = c[~c.repeat_in_answer]
    c['brand_k'] = c.brand.map(brand_of_label)
    pb = page_brands()
    own = c[c.role.isin(['own', 'competitor']) & c.brand_k.notna()].groupby('answer_id').brand_k.agg(set).to_dict()
    third = c[~(c.role.isin(['own', 'competitor']) & c.brand_k.notna())]
    third_names, unknown = {}, {}
    for aid, u, red in zip(third.answer_id, third.url, third.is_redirect):
        if not red and u in pb:
            third_names.setdefault(aid, set()).update(pb[u])
        else:
            unknown[aid] = unknown.get(aid, 0) + 1
    ncit = c.groupby('answer_id').size().to_dict()
    priors = {}
    for e, f in PRIOR.items():
        p = TABLES / f
        if p.exists():
            d = pd.read_csv(p)
            priors[e] = {r.prompt_id: {k for k, rx in BRANDS.items() if re.search(rx, ' '.join(json.loads(r.prior_brands)).lower())}
                         for r in d.itertuples()}
    m = read_table('mentions.csv', usecols=['answer_id', 'brand'])
    m['brand_k'] = m.brand.map(brand_of_label)
    pf = m.dropna(subset=['brand_k']).groupby('answer_id').brand_k.agg(set).to_dict()

    rows = []
    for r in a.itertuples():
        qt = q.get(r.prompt_id, '')
        text = str(r.answer_text).lower()
        pr = priors.get(r.engine, {}).get(r.prompt_id)
        for k, rx in BRANDS.items():
            if re.search(rx, qt):
                continue
            rows.append(dict(engine=r.engine, prompt_id=r.prompt_id, brand=k,
                             mentioned=bool(re.search(rx, text)), pfetch_mentioned=k in pf.get(r.answer_id, set()),
                             own_cited=k in own.get(r.answer_id, set()), third_naming=k in third_names.get(r.answer_id, set()),
                             uncertain=unknown.get(r.answer_id, 0) > 0, n_cited=ncit.get(r.answer_id, 0),
                             prior=(k in pr) if pr is not None else np.nan))
    d = pd.DataFrame(rows)
    print(f'{DATASET}: {len(d):,} answer x brand rows (brand not named in the question); '
          f'agreement with pfetch mentions: {100 * (d.mentioned == d.pfetch_mentioned).mean():.1f} %')

    dec = []
    for e, g in d.groupby('engine'):
        mm = g[g.mentioned]
        if mm.empty:
            continue
        own_ = mm.own_cited
        third_ = ~mm.own_cited & mm.third_naming
        none_ = ~mm.own_cited & ~mm.third_naming
        r = dict(engine=ENGINE_LABEL[e], mentions=len(mm), pct_of_brand_answer_pairs_mentioned=100 * g.mentioned.mean(),
                 pct_own_page_cited=100 * own_.mean(), pct_third_party_page_naming_it=100 * third_.mean(),
                 pct_no_cited_page_naming_it=100 * none_.mean(),
                 pct_no_cited_page_and_all_pages_known=100 * (none_ & ~mm.uncertain).mean())
        if mm.prior.notna().any():
            r['pct_unsupported_in_prior'] = 100 * mm[none_].prior.mean() if none_.any() else np.nan
            r['pct_mentions_in_prior'] = 100 * mm.prior.mean()
        dec.append(r)
    D = pd.DataFrame(dec)
    D.to_csv(RESULTS / 'x11_decomposition.csv', index=False)
    print(D.round(1).to_string(index=False))

    rates = []
    for e, g in d.groupby('engine'):
        for name, mask in (('own page cited', g.own_cited), ('third-party page naming it cited', ~g.own_cited & g.third_naming),
                           ('no cited page naming it', ~g.own_cited & ~g.third_naming)):
            for pv in ((True, False) if g.prior.notna().any() else (None,)):
                s = g[mask] if pv is None else g[mask & (g.prior == pv)]
                rates.append(dict(engine=ENGINE_LABEL[e], support=name, in_prior=pv, rows=len(s),
                                  pct_mentioned=100 * s.mentioned.mean() if len(s) else np.nan))
    R = pd.DataFrame(rates)
    R.to_csv(RESULTS / 'x11_rates.csv', index=False)
    print(R.round(1).to_string(index=False))

    out = []
    for e, g in d.groupby('engine'):
        g = g.copy()
        g['log_cited'] = np.log2(1 + g.n_cited)
        X = ['own_cited', 'third_naming', 'log_cited'] + (['prior'] if g.prior.notna().all() else [])
        B = pd.get_dummies(g.brand, prefix='b', drop_first=True, dtype=float)
        Xm = sm.add_constant(pd.concat([g[X].astype(float), B], axis=1))
        try:
            f = sm.Logit(g.mentioned.astype(float), Xm).fit(disp=0, cov_type='cluster',
                                                            cov_kwds={'groups': g.prompt_id.astype('category').cat.codes})
        except Exception as ex:
            print(e, 'logit failed:', type(ex).__name__); continue
        for v in X:
            out.append(dict(engine=ENGINE_LABEL[e], variable=v, odds_ratio=np.exp(f.params[v]), p=f.pvalues[v], rows=len(g)))
    M = pd.DataFrame(out)
    M.to_csv(RESULTS / 'x11_model.csv', index=False)
    print(M.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
