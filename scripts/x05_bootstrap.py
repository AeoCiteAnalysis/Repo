"""X5 - Confidence intervals and a shuffled baseline for the replay results (paper, section 6).

Re-run whenever new replay days arrive (x03_daily.py); every number reported in section 6 of the
paper comes from this script. 95 % percentile intervals, B = 1,000 resamples unless stated.

1 rank curve     % of consulted pages cited, by rank bin (ChatGPT runs with per-call source lists).
                 Resampling unit: prompt x run (one ranked source list).
2 selection      conditional logit of cited on log2 rank and 'found by > 1 search call';
                 resampling unit: prompt (B = 200, the model is slower).
3 variance       within-website R² (website fixed effects, in sample) of ChatGPT's 90-day counts
                 (total, breadth, depth) on: page variables; replay retrieval; retrieval + replay
                 citations; all. Difference retrieval - page variables. Resampling unit: website.
   shuffled      the replay variables permuted among pages of the same website (200 permutations):
                 what retrieval would explain if it carried only website-level information.
4 windows        within-website R² of each 14-day collection window on the replay (E6);
                 resampling unit: website.
5 same day       only when replay and collection share dates: per (prompt, day), share of the
                 collected cited URLs the replay consulted / cited, and Jaccard of cited sets;
                 resampling unit: prompt, with days resampled in blocks of 7.

Outputs: results/x05_rank_ci.csv, x05_selection_ci.csv, x05_variance_ci.csv, x05_shuffled.csv,
         x05_windows_ci.csv, x05_same_day.csv
"""
import warnings

import numpy as np
import pandas as pd

from common import RESULTS, SEED, clean_url, read_table
from r02_main_model import prepare, FEATURES
from e01_engines_breadth_depth import outcomes
from x03_analyse import records, ranked, collected

warnings.filterwarnings('ignore')
B = 1000
RNG = np.random.default_rng(SEED)
BINS, LABELS = [0, 1, 2, 3, 5, 10, 20, 999], ['1', '2', '3', '4-5', '6-10', '11-20', '21+']


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def replay_runs(engine='openai'):
    """All (run date, rep) records -> ranked long table with a run key."""
    out = []
    for j in records(engine):
        if j['consulted']:
            out.append(ranked(j).assign(date=j['date'], rep=j['rep'], has_calls=bool(j.get('calls'))))
    r = pd.concat(out, ignore_index=True)
    r['run'] = r.date + '|' + r.rep.astype(str)
    r['list_id'] = r.prompt_id + '|' + r.run
    return r


# 1 ---------------------------------------------------------------------------------------------
def rank_curve(r):
    q = r[r.has_calls].dropna(subset=['rank']).copy()
    q['bin'] = pd.cut(q['rank'], BINS, labels=LABELS)
    g = q.groupby(['list_id', 'bin'], observed=False).cited.agg(['sum', 'size']).unstack(fill_value=0)
    cit, n = g['sum'].to_numpy(float), g['size'].to_numpy(float)
    est = cit.sum(0) / n.sum(0)
    boot = np.empty((B, len(LABELS)))
    for b in range(B):
        i = RNG.integers(0, len(g), len(g))
        boot[b] = cit[i].sum(0) / n[i].sum(0)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    out = pd.DataFrame(dict(rank_bin=LABELS, consulted=n.sum(0).astype(int), pct_cited=100 * est,
                            ci_low=100 * lo, ci_high=100 * hi, source_lists=len(g)))
    out.to_csv(RESULTS / 'x05_rank_ci.csv', index=False)
    print(out.round(2).to_string(index=False))


# 2 ---------------------------------------------------------------------------------------------
def selection(r, reps=200):
    from statsmodels.discrete.conditional_models import ConditionalLogit
    q = r[r.has_calls].dropna(subset=['rank']).copy()
    q['log_rank'], q['multi_call'] = np.log2(q['rank']), (q.n_calls > 1).astype(float)
    q = q[q.groupby('list_id').cited.transform('nunique') == 2]
    X = ['log_rank', 'multi_call']

    def fit(d):
        return ConditionalLogit(d.cited.astype(float), d[X], groups=d.list_id).fit(disp=0).params

    est = fit(q)
    by = {p: g for p, g in q.groupby('prompt_id')}
    keys = list(by)
    boot = []
    for b in range(reps):
        pick = RNG.choice(len(keys), len(keys))
        d = pd.concat([by[keys[k]].assign(list_id=by[keys[k]].list_id + f'#{i}') for i, k in enumerate(pick)])
        try:
            boot.append(fit(d))
        except (ValueError, np.linalg.LinAlgError):          # degenerate resample (few prompts)
            boot.append(pd.Series(np.nan, index=X))
    boot = pd.DataFrame(boot)
    print(f'selection: {boot.isna().any(axis=1).sum()} of {reps} resamples could not be fitted')
    out = pd.DataFrame(dict(variable=X, coef=est[X].to_numpy(), odds_ratio=np.exp(est[X].to_numpy()),
                            or_ci_low=[np.exp(ci(boot[v])[0]) for v in X],
                            or_ci_high=[np.exp(ci(boot[v])[1]) for v in X], prompts=len(keys), resamples=reps))
    out.to_csv(RESULTS / 'x05_selection_ci.csv', index=False)
    print(out.round(3).to_string(index=False))


# 3 ---------------------------------------------------------------------------------------------
def page_table(r):
    c = collected('openai')
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    X = FEATURES + flags + types
    cc = c[~c.is_redirect & c.url.isin(d.url)]
    full = outcomes(cc)
    d = d[d.url.isin(full.index)].copy()
    d = d[d.groupby('domain').url.transform('size') >= 2].reset_index(drop=True)
    for y in ('total', 'breadth', 'depth'):
        d[y] = full[y].reindex(d.url).to_numpy()
    runs = r.run.nunique()
    g = r.groupby('url')
    for name, s in (('n_consulted', g.size()), ('n_top5', r[r['rank'] <= 5].groupby('url').size()),
                    ('n_cited', r[r.cited].groupby('url').size())):
        d['replay_' + name] = np.log2(1 + s.reindex(d.url).fillna(0).to_numpy() / runs)
    return d, X, runs, cc


def demeaned(d, cols, idx):
    """Within-website demeaning of the rows idx (a bootstrap draw of whole websites)."""
    s = d.iloc[idx]
    key = s['_draw'].to_numpy() if '_draw' in s else s.domain.to_numpy()
    M = s[cols].to_numpy(float)
    means = pd.DataFrame(M).groupby(key).transform('mean').to_numpy()
    return M - means


def r2w(Z, y):
    keep = Z.std(0) > 0
    Z = Z[:, keep]
    b = np.linalg.lstsq(Z, y, rcond=None)[0]
    return 1 - ((y - Z @ b) ** 2).sum() / (y ** 2).sum()


MODELS = lambda X: {'page variables': X, 'replay retrieval': ['replay_n_consulted', 'replay_n_top5'],
                    'replay retrieval + cited': ['replay_n_consulted', 'replay_n_top5', 'replay_n_cited'],
                    'page variables + replay retrieval + cited': X + ['replay_n_consulted', 'replay_n_top5', 'replay_n_cited']}


def variance(d, X, reps=B):
    models = MODELS(X)
    sites = d.groupby('domain').indices
    names = list(sites)
    rows = []
    for y in ('total', 'breadth', 'depth'):
        cols = [y] + sorted({c for v in models.values() for c in v})
        allidx = np.arange(len(d))
        Z = pd.DataFrame(demeaned(d, cols, allidx), columns=cols)
        est = {m: r2w(Z[v].to_numpy(), Z[y].to_numpy()) for m, v in models.items()}
        boot = {m: [] for m in models}
        for b in range(reps):
            pick = RNG.integers(0, len(names), len(names))
            idx = np.concatenate([sites[names[k]] for k in pick])
            draw = np.concatenate([np.full(len(sites[names[k]]), i) for i, k in enumerate(pick)])
            s = d.iloc[idx].assign(_draw=draw).reset_index(drop=True)
            Zb = pd.DataFrame(demeaned(s, cols, np.arange(len(s))), columns=cols)
            for m, v in models.items():
                boot[m].append(r2w(Zb[v].to_numpy(), Zb[y].to_numpy()))
        diff = np.array(boot['replay retrieval']) - np.array(boot['page variables'])
        for m in models:
            rows.append(dict(outcome=y, model=m, r2_within=est[m], ci_low=ci(boot[m])[0], ci_high=ci(boot[m])[1]))
        rows.append(dict(outcome=y, model='difference: retrieval - page variables',
                         r2_within=est['replay retrieval'] - est['page variables'], ci_low=ci(diff)[0], ci_high=ci(diff)[1]))
    out = pd.DataFrame(rows).assign(n_urls=len(d), n_websites=len(names), resamples=reps)
    out.to_csv(RESULTS / 'x05_variance_ci.csv', index=False)
    print(out.round(4).to_string(index=False))


def shuffled(d, reps=200):
    R = ['replay_n_consulted', 'replay_n_top5']
    rows = []
    for y in ('total', 'breadth', 'depth'):
        Z = pd.DataFrame(demeaned(d, [y] + R, np.arange(len(d))), columns=[y] + R)
        obs = r2w(Z[R].to_numpy(), Z[y].to_numpy())
        perm = []
        for b in range(reps):
            s = d.copy()
            shuf = s.groupby('domain')[R].transform(lambda v: v.sample(frac=1, random_state=int(RNG.integers(1e9))).to_numpy())
            s[R] = shuf.to_numpy()
            Zp = pd.DataFrame(demeaned(s, [y] + R, np.arange(len(s))), columns=[y] + R)
            perm.append(r2w(Zp[R].to_numpy(), Zp[y].to_numpy()))
        rows.append(dict(outcome=y, r2_within_observed=obs, r2_within_shuffled_mean=np.mean(perm),
                         shuffled_p95=np.percentile(perm, 95), p_value=(1 + np.sum(np.array(perm) >= obs)) / (reps + 1),
                         permutations=reps))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'x05_shuffled.csv', index=False)
    print(out.round(4).to_string(index=False))


# 4 ---------------------------------------------------------------------------------------------
def windows(r, reps=B):
    c = collected('openai')
    c = c[~c.is_redirect]
    days = sorted(c.date.unique())
    wins = [days[i:i + 14] for i in range(0, 84, 14)] + [days[84:]]
    dom = c.drop_duplicates('url').set_index('url').domain
    dom = dom[dom.groupby(dom).transform('size') >= 2]
    urls = dom.index
    runs = r.run.nunique()
    d = pd.DataFrame(dict(url=urls, domain=dom.to_numpy()))
    d['x1'] = np.log2(1 + r.groupby('url').size().reindex(urls).fillna(0).to_numpy() / runs)
    d['x2'] = np.log2(1 + r[r['rank'] <= 5].groupby('url').size().reindex(urls).fillna(0).to_numpy() / runs)
    sites = d.groupby('domain').indices
    names = list(sites)
    rows = []
    for w in wins:
        d['y'] = np.log2(1 + c[c.date.isin(w)].groupby('url').size().reindex(urls).fillna(0).to_numpy())
        Z = demeaned(d, ['y', 'x1', 'x2'], np.arange(len(d)))
        est = r2w(Z[:, 1:], Z[:, 0])
        boot = []
        for b in range(reps):
            pick = RNG.integers(0, len(names), len(names))
            idx = np.concatenate([sites[names[k]] for k in pick])
            draw = np.concatenate([np.full(len(sites[names[k]]), i) for i, k in enumerate(pick)])
            s = d.iloc[idx].assign(_draw=draw).reset_index(drop=True)
            Zb = demeaned(s, ['y', 'x1', 'x2'], np.arange(len(s)))
            boot.append(r2w(Zb[:, 1:], Zb[:, 0]))
        rows.append(dict(window_start=w[0], window_end=w[-1], r2_within=est, ci_low=ci(boot)[0], ci_high=ci(boot)[1],
                         replay_runs=runs))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'x05_windows_ci.csv', index=False)
    print(out.round(4).to_string(index=False))


# 5 ---------------------------------------------------------------------------------------------
def same_day(r, reps=B):
    c = collected('openai', all_days=True)      # days after the study period, matched with the replay
    c = c[~c.is_redirect]
    shared = sorted(set(c.date.astype(str)) & set(r.date))
    if not shared:
        print('same day: no dates shared by replay and collection yet (add new pfetch dumps and rerun load_data.py)')
        pd.DataFrame().to_csv(RESULTS / 'x05_same_day.csv', index=False)
        return
    rows = []
    col = c[c.date.astype(str).isin(shared)].groupby(['date', 'prompt_id']).url.agg(set)
    rep1 = r[r.rep == 1]
    cons = rep1.groupby(['date', 'prompt_id']).url.agg(set)
    cit = rep1[rep1.cited].groupby(['date', 'prompt_id']).url.agg(set)
    for (dd, p), cu in col.items():
        k = (str(dd), p)
        if k not in cons.index:
            continue
        cs, ct = cons[k], cit.get(k, set())
        rows.append(dict(date=str(dd), prompt_id=p, in_consulted=len(cu & cs) / len(cu), in_cited=len(cu & ct) / len(cu),
                         jaccard_cited=len(cu & ct) / len(cu | ct)))
    s = pd.DataFrame(rows)
    prompts, dates = s.prompt_id.unique(), sorted(s.date.unique())
    blocks = [dates[i:i + 7] for i in range(0, len(dates), 7)]
    m = ['in_consulted', 'in_cited', 'jaccard_cited']
    boot = []
    for b in range(reps):
        ps = set(RNG.choice(prompts, len(prompts)))
        ds = [x for k in RNG.integers(0, len(blocks), len(blocks)) for x in blocks[k]]
        sb = s[s.prompt_id.isin(ps) & s.date.isin(ds)]
        boot.append(sb[m].mean())
    boot = pd.DataFrame(boot)
    out = pd.DataFrame(dict(measure=m, mean=s[m].mean().to_numpy(), ci_low=[ci(boot[v])[0] for v in m],
                            ci_high=[ci(boot[v])[1] for v in m], days=len(dates), prompt_days=len(s)))
    out.to_csv(RESULTS / 'x05_same_day.csv', index=False)
    print(out.round(3).to_string(index=False))


def main():
    r = replay_runs()
    print(f'replay: {r.run.nunique()} ChatGPT runs on {r.date.nunique()} day(s)')
    rank_curve(r)
    selection(r)
    d, X, runs, _ = page_table(r)
    variance(d, X)
    shuffled(d)
    windows(r)
    same_day(r)


if __name__ == '__main__':
    main()
