"""E2 - Choice model: on a given day, which of a prompt's candidate pages does the engine pick?

Depth (how often a page is picked for a prompt it is cited for) is almost unexplained by page
features (E1). Here each answer is treated as a choice among the prompt's candidate pages: all R2
pages the engine cited for that prompt at least once in the first 60 days. Outcome days are the
last 30 days (out of time), so the history variables never see the outcome.

Per (engine, prompt, day) with an answer: one row per candidate, cited = in that day's answer.
Variables (z-scored within the candidate set, so they are RELATIVE to the competitors):
  match          page x prompt similarity (X2: embeddings best paragraph/title, BM25)
  page_pull      log2(1 + the page's first-60-day citations for OTHER prompts, same engine)
  pair_history   share of the first 60 days the page was cited for this prompt (same engine)
  yesterday      cited for this prompt by this engine on the previous day
Conditional logit per engine (groups = answers), 300 random prompts (seed 42). Reported: McFadden
pseudo-R² of nested models (match -> + page pull -> + pair history -> + yesterday) and the
coefficients of the full model.

Outputs: results/e02_choice_fit.csv, results/e02_choice_coef.csv
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

from common import TABLES, RESULTS, SEED, ENGINE_LABEL, read_table, drop_excluded

warnings.filterwarnings('ignore')
N_PROMPTS = 300
BLOCKS = [('match', ['emb_best', 'emb_title', 'bm25']), ('+ page pull', ['page_pull']),
          ('+ pair history', ['pair_history']), ('+ yesterday', ['yesterday'])]


def build(engine, c, a, sims, url_ix, prompt_ix, pick):
    c = c[c.engine == engine]
    days = sorted(c.date.unique())
    train, test = set(days[:60]), days[60:]
    ctr = c[c.date.isin(train)]
    pull_all = ctr.groupby('url').size()
    pair = ctr.groupby(['prompt_id', 'url']).size()
    ndays_train = ctr.groupby('prompt_id').date.nunique()
    cited = c.groupby(['date', 'prompt_id']).url.agg(set).to_dict()
    answered = a[(a.engine == engine)].groupby('prompt_id').date.agg(set).to_dict()
    frames = []
    for p in pick:
        cand = [u for u in ctr[ctr.prompt_id == p].url.unique() if u in url_ix]
        if len(cand) < 2 or p not in prompt_ix:
            continue
        pj, ui = prompt_ix[p], [url_ix[u] for u in cand]
        own = np.array([pair.get((p, u), 0) for u in cand], dtype=float)
        base = pd.DataFrame(dict(url=cand, emb_best=sims['emb_best'][ui, pj], emb_title=sims['emb_title'][ui, pj],
                                 bm25=sims['bm25'][ui, pj],
                                 page_pull=np.log2(1 + pull_all.reindex(cand).fillna(0).to_numpy() - own),
                                 pair_history=own / max(ndays_train.get(p, 1), 1)))
        for i, dday in enumerate(test):
            if dday not in answered.get(p, ()):
                continue
            today = cited.get((dday, p), set())
            prev = cited.get((days[days.index(dday) - 1], p), set())
            f = base.copy()
            f['cited'] = f.url.isin(today).astype(float)
            f['yesterday'] = f.url.isin(prev).astype(float)
            f['group'] = f'{p}|{dday}'
            frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    d = d[d.groupby('group').cited.transform('nunique') == 2]
    cols = [x for _, b in BLOCKS for x in b]
    for x in cols:
        g = d.groupby('group')[x]
        d[x] = ((d[x] - g.transform('mean')) / g.transform('std').replace(0, np.nan)).fillna(0)
    return d


def main():
    sims = np.load(TABLES / 'x02_sims.npz', allow_pickle=True)
    lab = pd.read_csv(TABLES / 'prompt_labels.csv', usecols=['prompt_id', 'prompt'])
    pid = dict(zip(lab.prompt, lab.prompt_id))
    prompt_ix = {pid[p]: j for j, p in enumerate(sims['prompts']) if p in pid}
    url_ix = {u: i for i, u in enumerate(sims['urls'])}
    c = drop_excluded(read_table('citations.csv', usecols=['date', 'engine', 'prompt_id', 'url', 'is_redirect',
                                                          'repeat_in_answer']))
    c = c[~c.is_redirect & ~c.repeat_in_answer].drop_duplicates(['date', 'engine', 'prompt_id', 'url'])
    a = drop_excluded(read_table('answers.csv', usecols=['date', 'engine', 'prompt_id']))
    pick = lab.prompt_id.sample(N_PROMPTS, random_state=SEED).tolist()
    fits, coefs = [], []
    for engine in ('OPENAI', 'GEMINI', 'PERPLEXITY'):
        d = build(engine, c, a, sims, url_ix, prompt_ix, pick)
        print(f'{engine}: {len(d)} rows, {d.group.nunique()} answers, {d.cited.mean():.3f} cited')
        cols = []
        m0 = None
        for name, b in BLOCKS:
            cols = cols + b
            m = ConditionalLogit(d.cited, d[cols], groups=d.group)
            r = m.fit(disp=0, method='bfgs', maxiter=500)
            ll0 = m.loglike(np.zeros(len(cols)))
            fits.append(dict(engine=ENGINE_LABEL[engine], model=name, answers=d.group.nunique(), rows=len(d),
                             candidates_per_answer=len(d) / d.group.nunique(), mcfadden_r2=1 - r.llf / ll0))
            print(fits[-1])
        coefs += [dict(engine=ENGINE_LABEL[engine], variable=v, coef=r.params[v], se=r.bse[v], p=r.pvalues[v],
                       odds_ratio_per_sd=np.exp(r.params[v])) for v in cols]
    pd.DataFrame(fits).to_csv(RESULTS / 'e02_choice_fit.csv', index=False)
    pd.DataFrame(coefs).to_csv(RESULTS / 'e02_choice_coef.csv', index=False)
    print(pd.DataFrame(fits).round(3).to_string(index=False))
    print(pd.DataFrame(coefs).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
