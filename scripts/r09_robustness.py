"""R9 - Robustness checks.

Question. Do the main results hold on different slices of the data and over time?

Re-fits the R2 model (same variables, website indicators, HC1):
  subsets      (a) all URLs cited >= 1 time, (b) cited >= 5 times, (c) top 1 % of counts capped
               — run within each R2 sample; the guide's (d) third-party-only and (e) own +
               competitor-only fits are the R2 samples themselves and are reported too
  time split   same URL list; citations counted in the first 75 % of days (train) and the last
               25 % (hold-out); model fitted on each; train model predicts the hold-out counts
               (train counts rescaled by hold-out days / train days so the scales match)
  leave one website out   8 refits, each without one of the 8 most-cited websites
  per engine   y from ChatGPT-only, Gemini-only, Perplexity-only counts
Stability (the "is it real?" test 5, per sample): the feature keeps the sign of its R2 estimate
in >= 4 of these 5 checks: (a), (b), (c), time-train, time-hold-out.

Outputs: results/r09_robustness.csv (all refits), results/r09_stability.csv (pass/fail),
         results/r09_time_prediction.csv, figures/r09_robustness.png
"""
import numpy as np
import pandas as pd
from scipy import stats

import plotstyle
from plotstyle import plt, MUTED
from common import unit_factors, unit_of, pct_in_unit, RESULTS, FIGURES, ENGINES, ENGINE_LABEL, read_table, drop_excluded
from fe import fe_ols, demean
from r02_main_model import prepare, FEATURES, LABEL, SAMPLES


def fit(s, X, label, sample):
    s = s[s.groupby('domain').y.transform('size') >= 2]
    res, info = fe_ols(s, 'y', X, 'domain')
    res = res.loc[[f for f in FEATURES if f in res.index]]
    return res.assign(check=label, sample=sample, n_urls=info['n_urls'],
                      n_websites=info['n_websites']).rename_axis('term').reset_index()


def period_counts(urls):
    """Citations per URL in the first 75 % and last 25 % of days."""
    c = drop_excluded(read_table('citations.csv', usecols=[
        'answer_id', 'date', 'engine', 'url', 'is_redirect', 'repeat_in_answer']))
    c = c[~c.is_redirect & ~c.repeat_in_answer & c.url.isin(urls)]
    days = sorted(c.date.unique())
    cut = days[int(len(days) * 0.75)]
    c['period'] = np.where(c.date < cut, 'train', 'holdout')
    n = c.groupby(['url', 'period']).size().unstack(fill_value=0)
    ratio = sum(d >= cut for d in days) / sum(d < cut for d in days)
    return n.reindex(urls, fill_value=0), days[0], cut, days[-1], ratio


def predict_holdout(s, X):
    """Train-period FE model -> predicted hold-out y (website level + slopes)."""
    s = s[s.groupby('domain').y_train.transform('size') >= 2].copy()
    res, _ = fe_ols(s.rename(columns={'y_train': 'y_'}), 'y_', X, 'domain')
    b = res.coef.reindex(X).fillna(0)
    xb = s[X].to_numpy() @ b.to_numpy()
    alpha = (s.y_train - xb).groupby(s.domain).transform('mean')
    pred = alpha + xb
    r2 = 1 - ((s.y_holdout - pred) ** 2).sum() / ((s.y_holdout - s.y_holdout.mean()) ** 2).sum()
    return dict(n_urls=len(s), r2_holdout=r2,
                spearman_holdout=stats.spearmanr(pred, s.y_holdout).statistic,
                r2_website_only=1 - ((s.y_holdout - s.y_train.groupby(s.domain).transform('mean')) ** 2).sum()
                / ((s.y_holdout - s.y_holdout.mean()) ** 2).sum())


def main():
    main_t = read_table('url_table.csv')
    all_t = read_table('url_table_all.csv')
    d, flags, types = prepare(main_t)
    da, fa, ta = prepare(all_t)
    X = FEATURES + flags + types
    counts, d0, cut, d1, ratio = period_counts(d.url.tolist())
    # train counts rescaled to the hold-out length, so both periods are on the same scale
    d['y_train'] = np.log2(1 + ratio * counts.train.to_numpy())
    d['y_holdout'] = np.log2(1 + counts.holdout.to_numpy())
    print(f'time split: train {d0}..{cut} (excl.), hold-out {cut}..{d1}')

    out, preds = [], []
    for name, roles in SAMPLES.items():
        pick = (lambda x: x) if roles is None else (lambda x, r=roles: x[x.role.isin(r)])
        s = pick(d)
        out.append(fit(s, X, 'R2 (main)', name))
        out.append(fit(pick(da), [c for c in FEATURES + fa + ta if c in da], '(a) cited >= 1', name))
        out.append(fit(s[s.citation_count >= 5], X, '(b) cited >= 5', name))
        cap = s.y.quantile(0.99)
        out.append(fit(s.assign(y=s.y.clip(upper=cap)), X, '(c) top 1 % capped', name))
        out.append(fit(s.assign(y=s.y_train), X, 'time: first 75 %', name))
        out.append(fit(s.assign(y=s.y_holdout), X, 'time: last 25 %', name))
        preds.append(dict(sample=name, **predict_holdout(s, X)))
        if roles is None:
            top8 = s.groupby('domain').citation_count.sum().nlargest(8).index
            for w in top8:
                out.append(fit(s[s.domain != w], X, f'without {w}', name))
            for e in ENGINES:
                col = f'count_{e}'
                out.append(fit(s.assign(y=np.log2(1 + s[col])), X, f'engine: {ENGINE_LABEL[e]}', name))
    res = pd.concat(out, ignore_index=True)
    res['pct_change_per_sd'] = 100 * (2 ** res.coef - 1)
    fac = unit_factors(d)
    res['unit'] = res.term.map(unit_of)
    res['pct_change'] = [pct_in_unit(b, t, fac) for t, b in zip(res.term, res.coef)]
    res.to_csv(RESULTS / 'r09_robustness.csv', index=False)
    pd.DataFrame(preds).to_csv(RESULTS / 'r09_time_prediction.csv', index=False)

    checks = ['(a) cited >= 1', '(b) cited >= 5', '(c) top 1 % capped',
              'time: first 75 %', 'time: last 25 %']
    main_sign = res[res.check == 'R2 (main)'].set_index(['sample', 'term']).coef.pipe(np.sign)
    st = res[res.check.isin(checks)].copy()
    st['same_sign'] = np.sign(st.coef).to_numpy() == main_sign.reindex(
        pd.MultiIndex.from_frame(st[['sample', 'term']])).to_numpy()
    stab = st.groupby(['sample', 'term']).same_sign.sum().rename('checks_same_sign').reset_index()
    stab['passes'] = stab.checks_same_sign >= 4
    lowo = res[(res['sample'] == 'all pages') & res.check.str.startswith('without')].copy()
    lowo['same'] = np.sign(lowo.coef).to_numpy() == main_sign.loc['all pages'].reindex(lowo.term).to_numpy()
    stab = stab.merge(lowo.groupby('term').same.sum().rename('lowo_same_sign_of_8').reset_index()
                      .assign(sample='all pages'), on=['sample', 'term'], how='left')
    stab.to_csv(RESULTS / 'r09_stability.csv', index=False)

    # figure: all-pages effects across checks
    g = res[res['sample'] == 'all pages']
    order = g[g.check == 'R2 (main)'].sort_values('coef').term.tolist()
    fig, ax = plt.subplots(figsize=(9, 6))
    groups = {'R2 (main)': ('#2a78d6', 'o', 7), 'subset / time': ('#eb6834', 's', 4),
              'leave one website out': ('#c3c2b7', 'o', 3.5), 'per engine': ('#1baf7a', 'D', 4)}
    kind = lambda c: ('R2 (main)' if c == 'R2 (main)' else 'leave one website out'
                      if c.startswith('without') else 'per engine' if c.startswith('engine')
                      else 'subset / time')
    for k, (col, mk, ms) in groups.items():
        h = g[g.check.map(kind) == k]
        y = [order.index(t) for t in h.term]
        ax.plot(h['pct_change'].clip(-60, 80), y, mk, color=col, ms=ms, alpha=0.9,
                label=k, mec='#fcfcfb', mew=0.6, ls='none')
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(order)), [LABEL[o] for o in order])
    ax.set_xlabel('% more citations per +1 SD (clipped to -60..80)')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right')
    ax.set_title('R9: R2 effects (all pages) across robustness checks')
    fig.savefig(FIGURES / 'r09_robustness.png')
    plt.close(fig)

    print(pd.DataFrame(preds).round(3).to_string(index=False))
    print(stab.pivot(index='term', columns='sample', values='checks_same_sign')[list(SAMPLES)].to_string())
    print('\nleave-one-website-out (all pages, same sign of 8):')
    print(stab[stab['sample'] == 'all pages'].set_index('term').lowo_same_sign_of_8.to_string())
    pe = res[res.check.str.startswith('engine')].pivot(index='term', columns='check', values='pct_change')
    print('\nper engine (% per SD):'); print(pe.round(1).to_string())


if __name__ == '__main__':
    main()
