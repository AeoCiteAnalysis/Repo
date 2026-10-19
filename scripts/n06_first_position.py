"""N6 - What gets cited first?

Question. Which pages appear at the top of the source list, and does position matter for
brand mentions?

1. Per URL: average position and share of citations where it is the first source (url_table).
2. The R2 model (website indicators, HC1) with outcome = share cited first, on URLs cited >= 5
   times that have page features.
3. Per answer: is the brand of the first-cited website (own / competitor site) also the brand
   mentioned first (rank 1)? Engines give ranks for the 8 original brands only (not Verizon).

Outputs: results/n06_first_position_model.csv, results/n06_first_source_vs_first_brand.csv
"""
import numpy as np
import pandas as pd

from common import RESULTS, ENGINES, ENGINE_LABEL, read_table, drop_excluded
from fe import fe_ols
from r02_main_model import prepare, FEATURES, LABEL


def main():
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    s = d[d.citation_count >= 5].copy()
    s['y'] = s.share_first                      # fe_ols works on column 'y'
    res, info = fe_ols(s, 'y', FEATURES + flags + types, 'domain')
    res = res.loc[FEATURES].assign(pp_per_sd=lambda x: 100 * x.coef, n_urls=info['n_urls'],
                                   n_websites=info['n_websites'])
    res.rename_axis('term').reset_index().to_csv(RESULTS / 'n06_first_position_model.csv', index=False)

    # first source vs first brand
    c = drop_excluded(read_table('citations.csv', usecols=['answer_id', 'date', 'engine', 'position',
                                                          'role', 'brand', 'repeat_in_answer']))
    first = c[(c.position == 1)].set_index('answer_id')
    m = drop_excluded(read_table('mentions.csv', usecols=['answer_id', 'date', 'engine', 'brand', 'rank']))
    top = m[m['rank'] == 1].drop_duplicates('answer_id').set_index('answer_id').brand.rename('first_brand')
    j = first.join(top, how='inner')
    j = j[j.role.isin(['own', 'competitor']) & (j.brand != 'Verizon')]
    j['agree'] = j.brand == j.first_brand
    agree = j.groupby('engine').agg(answers=('agree', 'size'), agree_rate=('agree', 'mean'))
    agree.loc['All'] = [len(j), j.agree.mean()]
    # baseline: how often would they agree by chance (brand shares independent)?
    base = {}
    for e, g in j.groupby('engine'):
        p1 = g.brand.value_counts(normalize=True)
        p2 = g.first_brand.value_counts(normalize=True)
        base[e] = float((p1 * p2.reindex(p1.index).fillna(0)).sum())
    agree['chance_rate'] = pd.Series(base)
    agree.rename(index=ENGINE_LABEL).to_csv(RESULTS / 'n06_first_source_vs_first_brand.csv')

    print(f"share cited first ~ features (URLs cited >= 5: {info['n_urls']:,}, {info['n_websites']:,} websites)")
    print(res.assign(term=[LABEL[i] for i in res.index])[['term', 'pp_per_sd', 'p']].round(3).to_string(index=False))
    print('\nfirst-cited brand site = first-mentioned brand:')
    print(agree.rename(index=ENGINE_LABEL).round(3).to_string())
    pos = t.groupby('role')[['mean_position', 'share_first']].median()
    print('\nmedian position / share first by role:'); print(pos.round(3).to_string())


if __name__ == '__main__':
    main()
