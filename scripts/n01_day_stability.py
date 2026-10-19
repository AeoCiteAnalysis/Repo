"""N1 - How stable are citations from day to day?

Question. If we ask the same engine the same question on two different days, how many of the
cited pages are the same?

For each engine x prompt x day: the set of cited URLs (redirects dropped) and of cited websites.
For every pair of days d and d + k (k = 1..30, 45, 60) on which the engine answered the prompt:
overlap = |shared| / |all| (Jaccard). Pairs where neither day cites anything are skipped (no
information); a pair where only one day cites something counts as 0.
Averaged by engine and by engine x topic; overlap vs days apart is plotted.
The guide's same-day repeat (5 asks on one day) needs new collection and is not done.

Outputs: results/n01_overlap_by_lag.csv, results/n01_overlap_by_topic.csv, figures/n01_overlap_curve.png
"""
from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

import plotstyle
from plotstyle import plt, ENGINE_COLOR
from common import RESULTS, FIGURES, ENGINES, ENGINE_LABEL, read_table, drop_excluded

LAGS = list(range(1, 31)) + [45, 60]


def sets():
    a = drop_excluded(read_table('answers.csv', usecols=['date', 'engine', 'prompt_id', 'topic']))
    c = drop_excluded(read_table('citations.csv', usecols=[
        'date', 'engine', 'prompt_id', 'url', 'domain', 'is_redirect', 'repeat_in_answer']))
    c = c[~c.repeat_in_answer & c.domain.notna()]
    urls = c[~c.is_redirect].groupby(['engine', 'prompt_id', 'date']).url.agg(frozenset)
    sites = c.groupby(['engine', 'prompt_id', 'date']).domain.agg(frozenset)
    answered = a.groupby(['engine', 'prompt_id']).date.agg(set)
    topic = a.drop_duplicates('prompt_id').set_index('prompt_id').topic
    return urls.to_dict(), sites.to_dict(), answered, topic


def jac(x, y):
    u = len(x | y)
    return len(x & y) / u if u else None


def main():
    urls, sites, answered, topic = sets()
    to_d = lambda s: date(int(s[:4]), int(s[4:6]), int(s[6:]))
    rows = defaultdict(list)                         # (engine, topic, lag, level) -> values
    empty = frozenset()
    for (eng, pid), days in answered.items():
        days_d = {to_d(d): d for d in days}
        for d0, ds in days_d.items():
            for k in LAGS:
                d1 = days_d.get(d0 + timedelta(days=k))
                if d1 is None:
                    continue
                for level, S in (('url', urls), ('website', sites)):
                    v = jac(S.get((eng, pid, ds), empty), S.get((eng, pid, d1), empty))
                    if v is not None:
                        rows[(eng, topic[pid], k, level)].append(v)
    recs = [dict(engine=e, topic=t, lag_days=k, level=l, mean_overlap=np.mean(v), n_pairs=len(v))
            for (e, t, k, l), v in rows.items()]
    df = pd.DataFrame(recs)
    by_lag = (df.assign(w=df.mean_overlap * df.n_pairs).groupby(['engine', 'level', 'lag_days'])
              .agg(w=('w', 'sum'), n_pairs=('n_pairs', 'sum')).reset_index())
    by_lag['mean_overlap'] = by_lag.w / by_lag.n_pairs
    by_lag = by_lag.drop(columns='w')
    by_lag.to_csv(RESULTS / 'n01_overlap_by_lag.csv', index=False)
    top = df[df.lag_days.isin([1, 7, 14])].pivot_table(index=['engine', 'topic'], columns=['level', 'lag_days'],
                                                       values='mean_overlap')
    top.columns = [f'{l}_lag{k}' for l, k in top.columns]
    top.reset_index().to_csv(RESULTS / 'n01_overlap_by_topic.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, level in zip(axes, ['url', 'website']):
        for e in ENGINES:
            g = by_lag[(by_lag.engine == e) & (by_lag.level == level)]
            one = g[g.lag_days == 1].mean_overlap.iloc[0]
            ax.plot(g.lag_days, 100 * g.mean_overlap, 'o-', ms=3, color=ENGINE_COLOR[e],
                    label=f'{ENGINE_LABEL[e]} (next day {100 * one:.0f}%)')
        ax.set_title(f'same cited {level}s', fontsize=9.5)
        ax.set_xlabel('days apart')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')
    axes[0].set_ylabel('% overlap (Jaccard)')
    fig.suptitle('N1: how much the cited sources change for the same question over time',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'n01_overlap_curve.png')
    plt.close(fig)
    print(by_lag[by_lag.lag_days.isin([1, 7, 14, 30, 60])].pivot_table(
        index=['engine', 'level'], columns='lag_days', values='mean_overlap').mul(100).round(1).to_string())


if __name__ == '__main__':
    main()
