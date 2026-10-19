"""N3 - Does being cited go with being mentioned (and liked)?

Question. When a brand's own website is cited, is the brand more likely to be mentioned, mentioned
first, and mentioned positively?

One row per answer x tracked brand (the 8 brands the engines report mentions for; Verizon has no
mention data). Variables:
  brand_cited       the brand's own website (role own/competitor, same brand) is cited
  competitor_cited  another tracked brand's website is cited
  review_cited      a review / comparison site is cited (list from R11)
  prompt_names      the prompt names this brand (the guide's "always control for this")
Outcomes: mentioned (all rows); ranked first and positive (rows where the brand is mentioned).
Rates cited vs not cited, and logistic regressions with engine, topic and brand indicators,
errors clustered by prompt; per engine and pooled.

Outputs: results/n03_rates.csv, results/n03_logit.csv
"""
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import TABLES, RESULTS, ENGINE_LABEL, read_table, drop_excluded
from r11_engines import REVIEW_SITES

BRANDS = ['Google Fiber', 'AT&T Fiber', 'Xfinity', 'Quantum Fiber', 'Spectrum', 'T-Mobile',
          'Metronet', 'Astound']
col = lambda b: 'names_' + re.sub(r'\W+', '_', b.lower()).strip('_')


def build():
    a = drop_excluded(read_table('answers.csv', usecols=['answer_id', 'date', 'engine', 'topic', 'prompt_id']))
    c = drop_excluded(read_table('citations.csv', usecols=['answer_id', 'date', 'engine', 'role', 'brand',
                                                          'reg_domain', 'repeat_in_answer']))
    c = c[~c.repeat_in_answer]
    cited = c[c.role.isin(['own', 'competitor'])].groupby('answer_id').brand.agg(set)
    review = set(c[c.reg_domain.isin(REVIEW_SITES)].answer_id)
    m = drop_excluded(read_table('mentions.csv', usecols=['answer_id', 'date', 'engine', 'brand', 'rank',
                                                         'sentiment']))
    m = m.sort_values('rank').drop_duplicates(['answer_id', 'brand'])
    lab = pd.read_csv(TABLES / 'prompt_labels.csv').set_index('prompt_id')
    rows = []
    for b in BRANDS:
        x = a.copy()
        x['brand'] = b
        cs = x.answer_id.map(cited)
        x['brand_cited'] = cs.map(lambda s: isinstance(s, set) and b in s).astype(int)
        x['competitor_cited'] = cs.map(lambda s: isinstance(s, set) and len(s - {b}) > 0).astype(int)
        x['review_cited'] = x.answer_id.isin(review).astype(int)
        x['prompt_names'] = x.prompt_id.map(lab[col(b)]).fillna(False).astype(int)
        rows.append(x)
    x = pd.concat(rows, ignore_index=True)
    x = x.merge(m[['answer_id', 'brand', 'rank', 'sentiment']], on=['answer_id', 'brand'], how='left')
    x['mentioned'] = x.sentiment.notna().astype(int)
    x['ranked_first'] = (x['rank'] == 1).astype(int)
    x['positive'] = (x.sentiment == 'POSITIVE').astype(int)
    return x


def rates(x):
    out = []
    for eng, g in [('All', x)] + list(x.groupby('engine')):
        for cited, h in g.groupby('brand_cited'):
            mm = h[h.mentioned == 1]
            out.append(dict(engine=ENGINE_LABEL.get(eng, eng), brand_site_cited=bool(cited), rows=len(h),
                            mention_rate=h.mentioned.mean(), first_rate_if_mentioned=mm.ranked_first.mean(),
                            positive_rate_if_mentioned=mm.positive.mean(),
                            mention_rate_prompt_not_naming=h[h.prompt_names == 0].mentioned.mean()))
    return pd.DataFrame(out)


def logits(x):
    f = ('{y} ~ brand_cited + competitor_cited + review_cited + prompt_names'
         ' + C(topic) + C(brand){eng}')
    out = []
    for eng, g in [('All', x)] + list(x.groupby('engine')):
        for y, data in (('mentioned', g), ('ranked_first', g[g.mentioned == 1]),
                        ('positive', g[g.mentioned == 1])):
            if data[y].nunique() < 2 or data.brand_cited.sum() < 30:
                continue
            m = smf.logit(f.format(y=y, eng=' + C(engine)' if eng == 'All' else ''), data=data).fit(
                disp=0, cov_type='cluster', cov_kwds={'groups': data.prompt_id.astype('category').cat.codes})
            for term in ('brand_cited', 'competitor_cited', 'review_cited', 'prompt_names'):
                out.append(dict(engine=ENGINE_LABEL.get(eng, eng), outcome=y, term=term,
                                odds_ratio=np.exp(m.params[term]), p=m.pvalues[term],
                                ci_low=np.exp(m.conf_int().loc[term, 0]),
                                ci_high=np.exp(m.conf_int().loc[term, 1]), rows=len(data)))
    return pd.DataFrame(out)


def main():
    x = build()
    r = rates(x)
    r.to_csv(RESULTS / 'n03_rates.csv', index=False)
    lg = logits(x)
    lg.to_csv(RESULTS / 'n03_logit.csv', index=False)
    print(r.round(3).to_string(index=False))
    print(lg[lg.term == 'brand_cited'].round(3).to_string(index=False))
    print(lg[(lg.engine == 'All')].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
