"""X6b - Does ChatGPT's retrieval rank follow KEYWORD or MEANING match to its own queries?

X6 found that a page's match to the engine's own search queries (embeddings, q_best) is the content
feature most related to a higher rank in ChatGPT's pooled result list. The queries look like keyword
searches, which suggests lexical (BM25-style) retrieval. Here both kinds of match to the SAME queries
are compared head to head, on the X6 rows with page text (data[/<ds>]/tables/x06_rows.csv):

  q_best      meaning: best cosine of any of the run's queries with the page title or a paragraph
  q_bm25_max  keywords: best BM25 score of the page for any single query of the run
  q_bm25_sum  keywords: sum of BM25 over the run's queries (a page that matches several queries)
BM25 over the consulted corpus of the dataset (title counted twice + first 50 paragraphs; k1 1.5, b 0.75).
All scores z-scored within each result list.

Retrieval: OLS of -log2 rank with list fixed effects (clustered by list), on
  (a) match to the question only, (b) + meaning match to queries, (c) + keyword match to queries,
  (d) both; within-list R2 and coefficients. Also mean within-list Spearman correlation of each score
  with the rank. Selection: conditional logit of cited on rank + each query-match score.
Outputs: results[/<ds>]/x06b_retrieval.csv, x06b_spearman.csv, x06b_selection.csv
"""
import json
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import TABLES, RESULTS, DATASET
from x06_selection import consulted, TOKEN

QUESTION = ['emb_best_z', 'emb_title_z', 'bm25_z']


def bm25_scores(rows, queries):
    from sklearn.feature_extraction.text import CountVectorizer
    pages = {}
    with open(TABLES / 'x06_parsed.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line); pages[r['url']] = r
    urls = sorted(set(rows.url) & set(pages))
    docs = [(pages[u]['title'] + ' ') * 2 + ' '.join(pages[u]['paragraphs']) for u in urls]
    cv = CountVectorizer(token_pattern=TOKEN.pattern, lowercase=True, stop_words='english')
    D = cv.fit_transform(docs).tocsr().astype(float)
    dl = np.asarray(D.sum(axis=1)).ravel(); avg = dl.mean()
    df = np.asarray((D > 0).sum(axis=0)).ravel()
    idf = np.log(1 + (len(urls) - df + 0.5) / (df + 0.5))
    k1, b = 1.5, 0.75
    vocab, ui = cv.vocabulary_, {u: i for i, u in enumerate(urls)}

    def score(i, terms):
        if not terms:
            return 0.0
        row = D[i]
        tf = np.array([row[0, t] for t in terms])
        return float((idf[terms] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[i] / avg))).sum())

    qterms = {}
    mx, sm_ = [], []
    for lid, u in zip(rows.list_id, rows.url):
        pid, run = lid.split('|', 1)
        if (pid, run) not in qterms:
            qterms[(pid, run)] = [[vocab[t] for t in set(TOKEN.findall(q.lower())) if t in vocab]
                                  for q in queries.get((pid, run), [])]
        if u not in ui or not qterms[(pid, run)]:
            mx.append(np.nan); sm_.append(np.nan); continue
        s = [score(ui[u], t) for t in qterms[(pid, run)]]
        mx.append(max(s)); sm_.append(sum(s))
    return np.array(mx), np.array(sm_)


def zwithin(d, c):
    g = d.groupby('list_id')[c]
    return ((d[c] - g.transform('mean')) / g.transform('std').replace(0, np.nan)).fillna(0)


def fe_fit(d, X):
    y = -d.log_rank
    Xd = d[X] - d.groupby('list_id')[X].transform('mean')
    yd = y - y.groupby(d.list_id).transform('mean')
    return sm.OLS(yd, Xd).fit(cov_type='cluster', cov_kwds={'groups': d.list_id.astype('category').cat.codes})


def main():
    from statsmodels.discrete.conditional_models import ConditionalLogit
    rows = pd.read_csv(TABLES / 'x06_rows.csv')
    _, queries = consulted()
    rows = rows[rows.text_missing == 0].copy()
    rows['q_bm25_max'], rows['q_bm25_sum'] = bm25_scores(rows, queries)
    rows = rows.dropna(subset=['q_bm25_max', 'q_best'])
    for c in ('q_bm25_max', 'q_bm25_sum'):
        rows[c + '_z'] = zwithin(rows, c)
    print(f'{DATASET}: {len(rows):,} rows with text in {rows.list_id.nunique()} lists')

    specs = {'(a) match to the question': QUESTION,
             '(b) + meaning match to queries': QUESTION + ['q_best_z'],
             '(c) + keyword match to queries (max)': QUESTION + ['q_bm25_max_z'],
             '(c2) + keyword match to queries (sum)': QUESTION + ['q_bm25_sum_z'],
             '(d) + both': QUESTION + ['q_best_z', 'q_bm25_max_z', 'q_bm25_sum_z']}
    out = []
    for name, X in specs.items():
        f = fe_fit(rows, X)
        for v in X:
            out.append(dict(model=name, variable=v, coef=f.params[v], se=f.bse[v], p=f.pvalues[v],
                            r2_within=f.rsquared, rows=len(rows)))
    R = pd.DataFrame(out)
    R.to_csv(RESULTS / 'x06b_retrieval.csv', index=False)
    print(R.groupby('model', sort=False).r2_within.first().round(4).to_string())
    print(R[R.model == '(d) + both'].round(4).to_string(index=False))

    sp = []
    for c in ('emb_best', 'bm25', 'q_best', 'q_bm25_max', 'q_bm25_sum'):
        r = rows.groupby('list_id').apply(lambda g: g[c].corr(-g.log_rank, method='spearman') if g[c].std() > 0 else np.nan)
        sp.append(dict(score=c, mean_spearman_with_higher_rank=r.mean(), lists=int(r.notna().sum())))
    S = pd.DataFrame(sp)
    S.to_csv(RESULTS / 'x06b_spearman.csv', index=False)
    print(S.round(3).to_string(index=False))

    g = rows[rows.groupby('list_id').cited.transform('nunique') == 2]
    sel = []
    for name, X in {'rank': ['log_rank', 'multi_call'],
                    'rank + meaning': ['log_rank', 'multi_call', 'q_best_z'],
                    'rank + keywords': ['log_rank', 'multi_call', 'q_bm25_max_z', 'q_bm25_sum_z'],
                    'rank + both': ['log_rank', 'multi_call', 'q_best_z', 'q_bm25_max_z', 'q_bm25_sum_z']}.items():
        X = [x for x in X if g[x].std() > 0]
        m = ConditionalLogit(g.cited.astype(float), g[X], groups=g.list_id)
        r = m.fit(disp=0, method='bfgs', maxiter=1000)
        sel.append(dict(model=name, mcfadden_r2=1 - r.llf / m.loglike(np.zeros(len(X))),
                        **{f'or_{v}': np.exp(r.params[v]) for v in X if v.startswith('q_')},
                        **{f'p_{v}': r.pvalues[v] for v in X if v.startswith('q_')}))
    P = pd.DataFrame(sel)
    P.to_csv(RESULTS / 'x06b_selection.csv', index=False)
    print(P.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
