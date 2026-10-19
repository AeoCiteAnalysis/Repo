"""X13 - Observational ranking stability: is there a stable brand order in real answers?

pfetch's own rank column is not a usable ordering (most brands in an answer share rank 1), so the order
of a brand in an answer is the position of its FIRST MENTION in the answer text (prompt_labels.BRANDS;
for GFiber incl. Verizon). This measures order of mention, a proxy for prominence, not an explicit
ranking; the controlled experiment asks for explicit rankings.

Unit: pairs of tracked brands both named in the same answer ("A before B"), excluding brands the question
itself names (their order echoes the question). Per engine, study period.
1 consistency   per (question, pair) with >= 10 co-mentions: share of answers with the majority order;
                share of pairs that are contested (majority < 60 %)
2 cycles        per question: majority graph over pairs with >= 10 co-mentions; directed 3-cycles, and
                how many survive 200 bootstrap resamples of answers (>= 90 %)
3 context       does A-before-B change with a third brand C present vs absent? per (question, A, B, C)
                with >= 10 answers each; two-proportion tests, Benjamini-Hochberg; significant reversals
4 drift         A-before-B in the first vs the second half of the days; significant reversals
5 score models  Bradley-Terry scores fitted on a random half of the days, tested on the other half
                (and on the last 30 days after training on the first 60): one score per brand for the
                engine (context-independent), per topic, per question; and, without transitivity, the
                per-question pair majority. Held-out pairwise accuracy and log loss.
Outputs: results[/<ds>]/x13_consistency.csv, x13_cycles.csv, x13_context.csv, x13_drift.csv, x13_models.csv
"""
import itertools
import re

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, TABLES, DATASET, SEED, ENGINE_LABEL, read_table, drop_excluded
from prompt_labels import BRANDS

MIN_N = 10


def pair_rows():
    a = drop_excluded(read_table('answers.csv', usecols=['answer_id', 'date', 'engine', 'topic', 'prompt_id', 'answer_text']))
    lab = pd.read_csv(TABLES / 'prompt_labels.csv',
                      usecols=['prompt_id', 'prompt'])
    q = dict(zip(lab.prompt_id, lab.prompt.str.lower()))
    rx = {b: re.compile(p) for b, p in BRANDS.items()}
    rows, present = [], []
    for r in a.itertuples():
        t = str(r.answer_text).lower()
        qt = q.get(r.prompt_id, '')
        first = {}
        for b, c in rx.items():
            if c.search(qt):                     # brands named in the question: their order echoes the question
                continue
            m = c.search(t)
            if m:
                first[b] = m.start()
        present.append(dict(answer_id=r.answer_id, brands=frozenset(first)))
        for x, y in itertools.combinations(sorted(first), 2):
            rows.append(dict(answer_id=r.answer_id, date=r.date, engine=r.engine, topic=r.topic, prompt_id=r.prompt_id,
                             a=x, b=y, a_first=first[x] < first[y]))
    d = pd.DataFrame(rows)
    pres = pd.DataFrame(present).set_index('answer_id').brands
    return d, pres


def consistency(d):
    g = d.groupby(['engine', 'prompt_id', 'a', 'b']).a_first.agg(['mean', 'size'])
    g = g[g['size'] >= MIN_N]
    g['majority'] = np.maximum(g['mean'], 1 - g['mean'])
    out = g.groupby('engine').agg(pairs=('majority', 'size'), mean_majority=('majority', 'mean'),
                                  pct_contested=('majority', lambda x: 100 * (x < 0.6).mean()),
                                  pct_near_unanimous=('majority', lambda x: 100 * (x >= 0.9).mean())).reset_index()
    return out, g


def cycles(d, rng):
    out = []
    for (e, pid), g in d.groupby(['engine', 'prompt_id']):
        s = g.groupby(['a', 'b']).a_first.agg(['mean', 'size'])
        s = s[s['size'] >= MIN_N]
        if len(s) < 3:
            continue
        brands = sorted(set(s.index.get_level_values(0)) | set(s.index.get_level_values(1)))

        def wins(tab):
            w = {}
            for (x, y), r in tab.iterrows():
                if r['mean'] > 0.5:
                    w[(x, y)] = True
                elif r['mean'] < 0.5:
                    w[(y, x)] = True
            return w

        def tri(w):
            found = set()
            for x, y, z in itertools.permutations(brands, 3):
                if w.get((x, y)) and w.get((y, z)) and w.get((z, x)):
                    found.add(frozenset((x, y, z)))
            return found

        base = tri(wins(s))
        robust = 0
        if base:
            ids = g.answer_id.unique()
            counts = {c: 0 for c in base}
            for _ in range(200):
                pick = rng.choice(ids, len(ids))
                bg = g.set_index('answer_id').loc[pick].reset_index()
                bs = bg.groupby(['a', 'b']).a_first.agg(['mean', 'size'])
                bs = bs[bs['size'] >= MIN_N]
                for c in tri(wins(bs)) & base:
                    counts[c] += 1
            robust = sum(v >= 180 for v in counts.values())
        out.append(dict(engine=e, prompt_id=pid, brands=len(brands), cycles=len(base), robust_cycles=robust))
    c = pd.DataFrame(out)
    return c.groupby('engine').agg(questions=('prompt_id', 'size'), pct_with_cycle=('cycles', lambda x: 100 * (x > 0).mean()),
                                   pct_with_robust_cycle=('robust_cycles', lambda x: 100 * (x > 0).mean()),
                                   cycles=('cycles', 'sum'), robust_cycles=('robust_cycles', 'sum')).reset_index()


def two_prop(p1, n1, p2, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 2 * stats.norm.sf(abs(p1 - p2) / se) if se > 0 else 1.0


def bh(p):
    p = np.asarray(p)
    if not len(p):
        return p
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    q = p * len(p) / r
    q_sorted = np.minimum.accumulate(q[o][::-1])[::-1]
    out = np.empty(len(p)); out[o] = np.minimum(q_sorted, 1)
    return out


def context(d, pres):
    d = d.assign(brands=d.answer_id.map(pres))
    rows = []
    for (e, pid, x, y), g in d.groupby(['engine', 'prompt_id', 'a', 'b']):
        if len(g) < 2 * MIN_N:
            continue
        others = set().union(*g.brands) - {x, y}
        for z in others:
            has = g.brands.map(lambda s: z in s)
            n1, n2 = has.sum(), (~has).sum()
            if n1 >= MIN_N and n2 >= MIN_N:
                p1, p2 = g[has].a_first.mean(), g[~has].a_first.mean()
                rows.append(dict(engine=e, prompt_id=pid, a=x, b=y, c=z, p_with=p1, p_without=p2, n_with=n1, n_without=n2,
                                 p=two_prop(p1, n1, p2, n2), reversal=(p1 - 0.5) * (p2 - 0.5) < 0))
    c = pd.DataFrame(rows)
    if c.empty:
        return c
    c['q'] = c.groupby('engine').p.transform(bh)
    return c.groupby('engine').agg(tests=('p', 'size'), pct_significant=('q', lambda x: 100 * (x < 0.05).mean()),
                                   pct_significant_reversal=('q', lambda x: np.nan)).reset_index().assign(
        pct_significant_reversal=lambda s: [100 * ((c[c.engine == e].q < 0.05) & c[c.engine == e].reversal).mean() for e in s.engine])


def drift(d):
    days = sorted(d.date.unique())
    half = set(days[:len(days) // 2])
    d = d.assign(first_half=d.date.isin(half))
    rows = []
    for (e, pid, x, y), g in d.groupby(['engine', 'prompt_id', 'a', 'b']):
        a1, a2 = g[g.first_half], g[~g.first_half]
        if len(a1) >= MIN_N and len(a2) >= MIN_N:
            p1, p2 = a1.a_first.mean(), a2.a_first.mean()
            rows.append(dict(engine=e, p=two_prop(p1, len(a1), p2, len(a2)), reversal=(p1 - 0.5) * (p2 - 0.5) < 0))
    r = pd.DataFrame(rows)
    if r.empty:
        return r
    r['q'] = r.groupby('engine').p.transform(bh)
    return r.groupby('engine').agg(pairs=('p', 'size'), pct_significant_change=('q', lambda x: 100 * (x < 0.05).mean())).reset_index().assign(
        pct_significant_reversal=lambda s: [100 * ((r[r.engine == e].q < 0.05) & r[r.engine == e].reversal).mean() for e in s.engine])


def bt_fit(tr, keys):
    """Bradley-Terry with a separate score per brand within each key group (L2-regularized logistic)."""
    from sklearn.linear_model import LogisticRegression
    grp = tr[keys].astype(str).agg('|'.join, axis=1) if keys else pd.Series('all', index=tr.index)
    cols = sorted(set(grp + '#' + tr.a) | set(grp + '#' + tr.b))
    idx = {c: i for i, c in enumerate(cols)}
    X = np.zeros((len(tr), len(cols)))
    X[np.arange(len(tr)), (grp + '#' + tr.a).map(idx).to_numpy()] = 1
    X[np.arange(len(tr)), (grp + '#' + tr.b).map(idx).to_numpy()] -= 1
    m = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(X, tr.a_first.astype(int))
    return dict(zip(cols, m.coef_[0]))


def bt_predict(sc, te, keys):
    grp = te[keys].astype(str).agg('|'.join, axis=1) if keys else pd.Series('all', index=te.index)
    sa = (grp + '#' + te.a).map(sc).fillna(0.0)
    sb = (grp + '#' + te.b).map(sc).fillna(0.0)
    return 1 / (1 + np.exp(-(sa - sb)))


def models(d, rng):
    days = sorted(d.date.unique())
    splits = {'random half of days': set(rng.choice(days, len(days) // 2, replace=False)),
              'first 60 -> last 30 days': set(days[:60])}
    rows = []
    for split, train_days in splits.items():
        for e, g in d.groupby('engine'):
            tr, te = g[g.date.isin(train_days)], g[~g.date.isin(train_days)]
            if tr.empty or te.empty:
                continue
            y = te.a_first.astype(int).to_numpy()
            for name, keys in (('brand score (engine-wide)', []), ('brand score per topic', ['topic']),
                               ('brand score per question', ['prompt_id'])):
                p = np.clip(bt_predict(bt_fit(tr, keys), te, keys).to_numpy(), 1e-6, 1 - 1e-6)
                rows.append(dict(split=split, engine=ENGINE_LABEL[e], model=name, test_pairs=len(te),
                                 accuracy=100 * ((p > 0.5) == y).mean(), log_loss=-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))
            maj = tr.groupby(['prompt_id', 'a', 'b']).a_first.mean()
            p = te.set_index(['prompt_id', 'a', 'b']).index.map(maj).to_numpy(dtype=float)
            p = np.where(np.isnan(p), 0.5, p)
            p = np.clip(p * 0.98 + 0.01, 1e-6, 1 - 1e-6)
            rows.append(dict(split=split, engine=ENGINE_LABEL[e], model='pair majority per question (no transitivity)',
                             test_pairs=len(te), accuracy=100 * ((p > 0.5) == y).mean(),
                             log_loss=-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    d, pres = pair_rows()
    print(f'{DATASET}: {len(d):,} brand pairs in {d.answer_id.nunique():,} answers')
    d['engine_label'] = d.engine.map(ENGINE_LABEL)
    c, _ = consistency(d)
    c['engine'] = c.engine.map(ENGINE_LABEL); c.to_csv(RESULTS / 'x13_consistency.csv', index=False)
    print(c.round(1).to_string(index=False))
    cy = cycles(d, rng); cy['engine'] = cy.engine.map(ENGINE_LABEL); cy.to_csv(RESULTS / 'x13_cycles.csv', index=False)
    print(cy.round(1).to_string(index=False))
    cx = context(d, pres)
    if len(cx):
        cx['engine'] = cx.engine.map(ENGINE_LABEL)
    cx.to_csv(RESULTS / 'x13_context.csv', index=False)
    print(cx.round(1).to_string(index=False))
    dr = drift(d)
    if len(dr):
        dr['engine'] = dr.engine.map(ENGINE_LABEL)
    dr.to_csv(RESULTS / 'x13_drift.csv', index=False)
    print(dr.round(1).to_string(index=False))
    m = models(d, rng); m.to_csv(RESULTS / 'x13_models.csv', index=False)
    print(m.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
