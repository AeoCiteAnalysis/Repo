"""N2 - Do the engines agree with each other?

Question. For the same question on the same day, do ChatGPT, Gemini and Perplexity cite the same
pages and websites?

1. For each prompt-day answered by both engines of a pair: Jaccard overlap of cited websites and
   of cited URLs. Pairs where neither engine cites anything are skipped.
2. Over the whole period: websites cited by only one engine (engine-specific) vs by all three
   (shared), with their citation counts.
3. Pages cited by all three engines vs by one engine only: compare their page features
   (medians / shares, URLs with page features).

Outputs: results/n02_pairwise_overlap.csv, results/n02_shared_websites.csv,
         results/n02_shared_vs_specific_features.csv
"""
from itertools import combinations

import numpy as np
import pandas as pd

from common import RESULTS, ENGINES, ENGINE_LABEL, read_table, drop_excluded
from r02_main_model import FEATURES, LABEL


def main():
    c = drop_excluded(read_table('citations.csv', usecols=[
        'date', 'engine', 'prompt_id', 'url', 'domain', 'role', 'is_redirect', 'repeat_in_answer']))
    c = c[~c.repeat_in_answer & c.domain.notna()]
    a = drop_excluded(read_table('answers.csv', usecols=['date', 'engine', 'prompt_id']))
    answered = a.groupby(['prompt_id', 'date']).engine.agg(set)
    S = {'website': c.groupby(['engine', 'prompt_id', 'date']).domain.agg(frozenset).to_dict(),
         'url': c[~c.is_redirect].groupby(['engine', 'prompt_id', 'date']).url.agg(frozenset).to_dict()}
    rows = []
    empty = frozenset()
    for e1, e2 in combinations(ENGINES, 2):
        for level, sets in S.items():
            vals = []
            for (pid, d), engs in answered.items():
                if e1 in engs and e2 in engs:
                    x, y = sets.get((e1, pid, d), empty), sets.get((e2, pid, d), empty)
                    u = len(x | y)
                    if u:
                        vals.append(len(x & y) / u)
            rows.append(dict(pair=f'{ENGINE_LABEL[e1]} - {ENGINE_LABEL[e2]}', level=level,
                             mean_overlap=np.mean(vals), share_any_overlap=np.mean(np.array(vals) > 0),
                             n_prompt_days=len(vals)))
    pw = pd.DataFrame(rows)
    pw.to_csv(RESULTS / 'n02_pairwise_overlap.csv', index=False)

    # websites: which engines cite them (restricted to prompt-days all three answered)
    common = {k for k, v in answered.items() if len(v) == 3}
    cc = c[[x in common for x in zip(c.prompt_id, c.date)]]
    w = cc.pivot_table(index='domain', columns='engine', aggfunc='size', fill_value=0)
    w['n_engines'] = (w[ENGINES] > 0).sum(axis=1)
    w['total'] = w[ENGINES].sum(axis=1)
    w['role'] = cc.drop_duplicates('domain').set_index('domain').role.reindex(w.index)
    w['group'] = np.where(w.n_engines == 3, 'shared by all 3',
                          np.where(w.n_engines == 1, 'only ' + w[ENGINES].idxmax(axis=1).map(ENGINE_LABEL),
                                   'two engines'))
    w = w.sort_values('total', ascending=False).rename_axis('website').reset_index()
    w.to_csv(RESULTS / 'n02_shared_websites.csv', index=False)
    summ = w.groupby('group').agg(websites=('website', 'size'), citations=('total', 'sum'))
    summ['pct_citations'] = 100 * summ.citations / summ.citations.sum()

    # page features: URLs cited by all 3 engines vs by exactly one
    u = cc[~cc.is_redirect].pivot_table(index='url', columns='engine', aggfunc='size', fill_value=0)
    u['n_engines'] = (u[ENGINES] > 0).sum(axis=1)
    t = read_table('url_table_all.csv')
    t = t[t.word_count.notna()].set_index('url')
    t = t.join(u.n_engines, how='inner')
    t['group'] = t.n_engines.map({1: 'one engine', 2: 'two engines', 3: 'all three engines'})
    raw = [f + '_raw' for f in FEATURES]
    feat = t.groupby('group')[raw].median().T
    feat.index = [LABEL[i.removesuffix('_raw')] for i in feat.index]
    feat.loc['URLs (n)'] = t.group.value_counts()
    feat.to_csv(RESULTS / 'n02_shared_vs_specific_features.csv')

    print(pw.round(3).to_string(index=False))
    print(summ.round(1).to_string())
    print('\ntop shared websites:', ', '.join(w[w.n_engines == 3].website.head(12)))
    for e in ENGINES:
        print(f'top only-{ENGINE_LABEL[e]}:', ', '.join(w[w.group == 'only ' + ENGINE_LABEL[e]].website.head(8)))
    print(feat.round(3).to_string())


if __name__ == '__main__':
    main()
