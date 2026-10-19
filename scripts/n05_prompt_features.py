"""N5 - Which questions lead to which sources?

Question. Do features of the question change how many and what kind of pages are cited?

Prompt features (prompt_labels.py): number of words, names the client (Google Fiber), names a
competitor, comparison wording, price wording, yes/no question, intent (LLM label), topic.
Outcomes per answer: number of different URLs cited; among answers with >= 1 citation, share of
citations to own / competitor sites and to Reddit + YouTube.
One OLS per outcome: prompt features + engine, errors clustered by prompt.
Special check: when a prompt names a competitor, how often is that competitor's site cited vs
the client's?

Outputs: results/n05_models.csv, results/n05_named_competitor.csv
"""
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import TABLES, RESULTS, ENGINE_LABEL, read_table, drop_excluded

COMPETITORS = ['AT&T Fiber', 'Xfinity', 'Quantum Fiber', 'Spectrum', 'T-Mobile', 'Metronet', 'Astound', 'Verizon']
col = lambda b: 'names_' + re.sub(r'\W+', '_', b.lower()).strip('_')


def main():
    a = drop_excluded(read_table('answers.csv', usecols=['answer_id', 'date', 'engine', 'prompt_id',
                                                        'n_unique_citations']))
    c = drop_excluded(read_table('citations.csv', usecols=['answer_id', 'date', 'engine', 'role', 'brand',
                                                          'reg_domain', 'repeat_in_answer']))
    c = c[~c.repeat_in_answer]
    g = c.groupby('answer_id')
    shares = pd.DataFrame({'n': g.size(), 'own': g.role.apply(lambda s: (s == 'own').mean()),
                           'competitor': g.role.apply(lambda s: (s == 'competitor').mean()),
                           'reddit_youtube': g.reg_domain.apply(lambda s: s.isin(['reddit.com', 'youtube.com']).mean())})
    lab = pd.read_csv(TABLES / 'prompt_labels.csv')
    lab['names_client'] = lab[col('Google Fiber')].astype(int)
    lab['names_competitor'] = lab[[col(b) for b in COMPETITORS]].any(axis=1).astype(int)
    for k in ('is_comparison', 'asks_price', 'is_yes_no'):
        lab[k] = lab[k].astype(int)
    lab['words_10'] = lab.n_words / 10
    d = a.merge(lab, on='prompt_id').join(shares, on='answer_id')
    d['pct_own'], d['pct_competitor'], d['pct_reddit_youtube'] = 100 * d.own, 100 * d.competitor, 100 * d.reddit_youtube

    rhs = ('words_10 + names_client + names_competitor + is_comparison + asks_price + is_yes_no'
           ' + C(intent, Treatment("learning")) + C(topic) + C(engine)')
    out = []
    for y, data in (('n_unique_citations', d), ('pct_own', d[d.n > 0]), ('pct_competitor', d[d.n > 0]),
                    ('pct_reddit_youtube', d[d.n > 0])):
        m = smf.ols(f'{y} ~ {rhs}', data=data).fit(cov_type='cluster',
                                                   cov_kwds={'groups': data.prompt_id.astype('category').cat.codes})
        for term in m.params.index:
            if term == 'Intercept' or term.startswith('C(topic)') or term.startswith('C(engine)'):
                continue
            out.append(dict(outcome=y, term=term.replace('C(intent, Treatment("learning"))[T.', 'intent: ').rstrip(']'),
                            coef=m.params[term], se=m.bse[term], p=m.pvalues[term], answers=len(data),
                            outcome_mean=data[y].mean()))
    res = pd.DataFrame(out)
    res.to_csv(RESULTS / 'n05_models.csv', index=False)

    # named competitor: is that competitor's site cited, and the client's?
    cited = c[c.role.isin(['own', 'competitor'])].groupby('answer_id').brand.agg(set)
    d['cited_brands'] = d.answer_id.map(cited)
    rows = []
    for b in COMPETITORS:
        h = d[d[col(b)] == 1]
        for eng, k in h.groupby('engine'):
            cb = k.cited_brands
            rows.append(dict(competitor=b, engine=ENGINE_LABEL[eng], answers=len(k),
                             pct_competitor_site_cited=100 * cb.map(lambda s: isinstance(s, set) and b in s).mean(),
                             pct_client_site_cited=100 * cb.map(lambda s: isinstance(s, set) and 'Google Fiber' in s).mean(),
                             pct_prompt_also_names_client=100 * k.names_client.mean()))
    named = pd.DataFrame(rows)
    named.to_csv(RESULTS / 'n05_named_competitor.csv', index=False)

    show = res.assign(v=res.coef.round(2).astype(str) + np.where(res.p < 0.05, '*', ''))
    print(show.pivot(index='term', columns='outcome', values='v').to_string())
    print('\noutcome means:', res.groupby('outcome').outcome_mean.first().round(2).to_dict())
    print('\nprompt names a competitor -> whose site gets cited (%):')
    print(named.pivot_table(index='competitor', columns='engine',
                            values=['pct_competitor_site_cited', 'pct_client_site_cited']).round(1).to_string())


if __name__ == '__main__':
    main()
