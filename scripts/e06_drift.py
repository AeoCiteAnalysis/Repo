"""E6 - How fast does the set of cited pages drift?

R13's split-half reliability (odd vs even days, 0.90) says the 90-day counts are repeatable, but
both halves span the whole period, so a page that was popular in July and gone in September counts
as 'stable'. A snapshot of page features (crawled in September) cannot explain that kind of change.

Windows: six non-overlapping 14-day windows (days 1-84) of the collection. For each engine, on all
URLs the engine cited (websites with >= 2 such URLs), log2(1 + citations in a window):
  lag curve     within-website R² between two windows, averaged by lag (14, 28, ... 70 days)
  same-window   split-half reliability inside one window (odd vs even days, Spearman-Brown) =
                the lag-0 reference: what is repeatable at one point in time
  replay        (ChatGPT) within-website R² of each window's counts on the replay's retrieval
                variables (x03; pages consulted and top-5 per prompt, 3 runs on 2026-09-27)
Outputs: results/e06_drift_lag.csv, results/e06_replay_by_window.csv, figures/e06_drift.png
"""
import numpy as np
import pandas as pd

import plotstyle
from plotstyle import plt, ENGINE_COLOR, INK2
from common import RESULTS, FIGURES, ENGINES, ENGINE_LABEL, read_table, drop_excluded

WIN = 14


def within(v, dom):
    return v - pd.Series(v).groupby(dom).transform('mean').to_numpy()


def r2w(a, b, dom):
    a, b = within(a, dom), within(b, dom)
    return np.corrcoef(a, b)[0, 1] ** 2


def main():
    c = drop_excluded(read_table('citations.csv', usecols=['date', 'engine', 'prompt_id', 'url', 'domain',
                                                          'is_redirect', 'repeat_in_answer']))
    c = c[~c.is_redirect & ~c.repeat_in_answer]
    days = sorted(c.date.unique())
    wins = [days[i:i + WIN] for i in range(0, 6 * WIN, WIN)]
    rows, rep = [], []
    for e in ENGINES + ['all']:
        ce = c if e == 'all' else c[c.engine == e]
        dom = ce.drop_duplicates('url').set_index('url').domain
        dom = dom[dom.groupby(dom).transform('size') >= 2]
        urls, dv = dom.index, dom.to_numpy()
        cnt = lambda s: np.log2(1 + s.groupby('url').size().reindex(urls).fillna(0).to_numpy())
        W = [cnt(ce[ce.date.isin(w)]) for w in wins]
        # lag 0: odd vs even days inside each window
        sb = []
        for w in wins:
            cw = ce[ce.date.isin(w)]
            odd = cnt(cw[cw.date.isin(w[0::2])]); even = cnt(cw[cw.date.isin(w[1::2])])
            r = np.corrcoef(within(odd, dv), within(even, dv))[0, 1]
            sb.append((2 * r / (1 + r)) ** 2)            # R² between two full windows at lag 0
        rows.append(dict(engine=ENGINE_LABEL.get(e, 'All engines'), lag_days=0, r2_within=np.mean(sb),
                         pairs=len(wins), n_urls=len(urls)))
        for lag in range(1, len(wins)):
            vals = [r2w(W[i], W[i + lag], dv) for i in range(len(wins) - lag)]
            rows.append(dict(engine=ENGINE_LABEL.get(e, 'All engines'), lag_days=lag * WIN, r2_within=np.mean(vals),
                             pairs=len(vals), n_urls=len(urls)))
        if e == 'OPENAI':
            from x03_analyse import records, ranked
            recs = [j for j in records('openai') if j['consulted']]
            runs = len({(j['date'], j['rep']) for j in recs})          # all replay runs, incl. daily ones
            r = pd.concat([ranked(j) for j in recs])
            X = np.column_stack([within(np.log2(1 + r.groupby('url').size().reindex(urls).fillna(0).to_numpy() / runs), dv),
                                 within(np.log2(1 + r[r['rank'] <= 5].groupby('url').size().reindex(urls).fillna(0).to_numpy() / runs), dv)])
            for w in wins + [days[6 * WIN:]]:
                y = within(cnt(ce[ce.date.isin(w)]), dv)
                b = np.linalg.lstsq(X, y, rcond=None)[0]
                rep.append(dict(window_start=w[0], window_end=w[-1], days_before_replay=(pd.Timestamp('2026-09-27') - pd.Timestamp(w[-1])).days,
                                r2_within=1 - ((y - X @ b) ** 2).sum() / (y ** 2).sum(), n_urls=len(urls)))
    lag, rp = pd.DataFrame(rows), pd.DataFrame(rep)
    lag.to_csv(RESULTS / 'e06_drift_lag.csv', index=False)
    rp.to_csv(RESULTS / 'e06_replay_by_window.csv', index=False)
    print(lag.round(3).to_string(index=False))
    print(rp.round(3).to_string(index=False))
    plot(lag, rp)


def plot(lag, rp):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    cols = {'ChatGPT': ENGINE_COLOR['OPENAI'], 'Gemini': ENGINE_COLOR['GEMINI'], 'Perplexity': ENGINE_COLOR['PERPLEXITY'],
            'All engines': '#52514c'}
    for e, g in lag.groupby('engine'):
        ax.plot(g.lag_days, 100 * g.r2_within, marker='o', ms=5, lw=2, color=cols[e], label=e)
    ax.set_xlabel('days between two 14-day windows (0 = same window)')
    ax.set_ylabel('% of within-website variation shared')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5)
    ax.set_title('How much of one period\'s citation pattern is still there later', fontsize=9.5)
    ax = axes[1]
    x = np.arange(len(rp))
    ax.bar(x, 100 * rp.r2_within, color=ENGINE_COLOR['OPENAI'], width=0.6)
    for xi, v in zip(x, rp.r2_within):
        ax.text(xi, 100 * v + 0.8, f'{100 * v:.0f}%', ha='center', fontsize=7.5, color=INK2)
    ax.set_xticks(x, [f'{s[4:6]}/{s[6:]}-\n{t[4:6]}/{t[6:]}' for s, t in zip(rp.window_start, rp.window_end)], fontsize=7.5)
    ax.set_ylabel('% explained by the Sept 27 replay')
    ax.grid(axis='x', visible=False)
    ax.set_title('ChatGPT: replay retrieval vs each collection window', fontsize=9.5)
    fig.suptitle('E6: the cited set drifts over time', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'e06_drift.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
