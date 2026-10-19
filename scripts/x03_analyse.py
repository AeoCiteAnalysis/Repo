"""X3 (E6, E10) - What the engine replay shows about retrieval and selection.

Uses the replay files from x03_replay.py and the collected citations (90 days).

1 faithful   Does the replay behave like the collection? (a) prompts where the replay did not
             search vs the collected share of answers with no citations; (b) share of the replay's
             cited URLs that the collection also cited for the prompt in its last 14 days, compared
             with the same share for the collection's own last day (the day-to-day baseline).
2 coverage   ChatGPT: share of the collected (prompt, URL) citations whose URL was among the pages
             the replay CONSULTED (retrieved) or CITED - weighted by collected citations.
3 selection  ChatGPT: chance that a consulted page is cited, by its rank in the search results
             (best rank over the search calls; the order of the source list for records without
             per-call lists); conditional logit per prompt on log rank and 'found by more than one
             search call'.
4 depth      ChatGPT: do collected 90-day counts of consulted (prompt, URL) pairs follow the replay's
             retrieval rank? R² of log2(1 + collected count) on log rank.
5 pages      ChatGPT, R2 pages: page-level retrieval variables from the replay (number of prompts the
             page was consulted for, number where it ranked top 5, both log2(1 + n)) added to the R2
             page variables; share of within-website variation explained (mixed model), for total,
             breadth and depth (E1 baseline: page variables alone).
6 repeat     Same-day repeats (rep 1 vs rep 2): overlap of search queries, consulted and cited URLs.

Outputs: results/x03_*.csv, figures/x03_retrieval.png
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import plotstyle
from plotstyle import plt, ENGINE_COLOR, INK2, MUTED
from common import ROOT, TABLES, RESULTS, FIGURES, ENGINE_LABEL, clean_url, read_table, drop_excluded
from r02_main_model import prepare, FEATURES
from e01_engines_breadth_depth import outcomes, vc, rel_within

warnings.filterwarnings('ignore')
REPLAY = TABLES.parent / 'replay'
ENG = {'openai': 'OPENAI', 'gemini': 'GEMINI'}


def records(engine, rep=None):
    out = []
    for f in sorted((REPLAY / engine).glob('*/*.json')):
        j = json.loads(f.read_text(encoding='utf-8'))
        if rep is None or j['rep'] == rep:
            out.append(j)
    return out


def ranked(j):
    """Consulted URLs with best rank over search calls and number of calls that found them."""
    calls = j.get('calls') or [dict(sources=j['consulted'])]
    best, found = {}, {}
    for c in calls:
        for i, u in enumerate(dict.fromkeys(clean_url(x) for x in c['sources']), 1):
            best[u] = min(best.get(u, 999), i)
            found[u] = found.get(u, 0) + 1
    cited = {clean_url(u) for u in j['cited']}
    for u in cited - set(best):                  # cited without being in a source list (rare)
        best[u], found[u] = np.nan, 0
    return pd.DataFrame(dict(url=list(best), rank=list(best.values()), n_calls=[found[u] for u in best],
                             cited=[u in cited for u in best])).assign(prompt_id=j['prompt_id'])


def collected(engine, all_days=False):
    c = drop_excluded(read_table('citations.csv', all_days=all_days, usecols=['date', 'engine', 'prompt_id', 'url', 'domain',
                                                          'is_redirect', 'repeat_in_answer']))
    return c[(c.engine == ENG[engine]) & ~c.repeat_in_answer]


def faithful(engine, recs, c):
    a = read_table('answers.csv', usecols=['date', 'engine', 'prompt_id', 'n_citations'])
    a = a[a.engine == ENG[engine]]
    zero = (a.n_citations == 0).groupby(a.prompt_id).mean()
    searched = {j['prompt_id'] for j in recs if j['queries'] or j['cited']}
    days = sorted(c.date.unique())
    last14 = c[c.date.isin(days[-14:])]
    ref = set(zip(last14.prompt_id, last14.url))
    prev13 = c[c.date.isin(days[-14:-1])]
    ref13 = set(zip(prev13.prompt_id, prev13.url))
    lastday = c[(c.date == days[-1]) & ~c.is_redirect].drop_duplicates(['prompt_id', 'url'])
    rep_cited = [(j['prompt_id'], clean_url(u)) for j in recs for u in set(j['cited'])]
    return dict(engine=ENGINE_LABEL[ENG[engine]], prompts=len(recs), searched=len(searched),
                collected_zero_share_when_replay_searched=zero[zero.index.isin(searched)].mean(),
                collected_zero_share_when_replay_did_not=zero[~zero.index.isin(searched)].mean(),
                replay_cited_in_collected_last14=np.mean([k in ref for k in rep_cited]),
                collected_lastday_in_previous13=np.mean([k in ref13 for k in zip(lastday.prompt_id, lastday.url)]))


def main():
    rows = []
    cache = {}
    for engine in ('openai', 'gemini'):
        recs = records(engine, rep=1)
        # sections 1-2 describe ONE full replay run: the first day's run 1 (daily runs are also 'run 1',
        # on later dates; grouping by question would add them up). Same-day comparisons are in x05.
        first = min((j['date'] for j in recs), default=None)
        recs = [j for j in recs if j['date'] == first]
        if recs:
            cache[engine] = (recs, collected(engine))
            rows.append(faithful(engine, *cache[engine]))
    f = pd.DataFrame(rows)
    f.to_csv(RESULTS / 'x03_faithful.csv', index=False)
    print(f.round(3).T.to_string())

    recs, c = cache['openai']
    r = pd.concat([ranked(j) for j in recs if j['consulted']], ignore_index=True)
    hist = c.groupby(['prompt_id', 'url']).size()
    r['collected'] = [hist.get(k, 0) for k in zip(r.prompt_id, r.url)]

    # 2 coverage
    srch = set(r.prompt_id)
    pairs = c[c.prompt_id.isin(srch) & ~c.is_redirect].groupby(['prompt_id', 'url']).size()
    cons, cit = set(zip(r.prompt_id, r.url)), set(zip(r[r.cited].prompt_id, r[r.cited].url))
    inc = np.array([k in cons for k in pairs.index]); inr = np.array([k in cit for k in pairs.index])
    cov = pd.DataFrame([dict(pairs=len(pairs), pct_pairs_consulted=100 * inc.mean(),
                             pct_citations_consulted=100 * pairs[inc].sum() / pairs.sum(),
                             pct_pairs_cited=100 * inr.mean(), pct_citations_cited=100 * pairs[inr].sum() / pairs.sum(),
                             consulted_per_prompt=r.groupby('prompt_id').size().median(),
                             cited_per_prompt=r.groupby('prompt_id').cited.sum().median(),
                             pct_consulted_cited=100 * r.cited.mean(),
                             pct_consulted_ever_cited_in_collection=100 * (r.collected > 0).mean())])
    cov.to_csv(RESULTS / 'x03_coverage.csv', index=False)
    print(cov.round(1).T.to_string())

    # 3 selection by rank + conditional logit: repeats with per-call source lists
    allr = {k: pd.concat([ranked(j).assign(rep=k) for j in records('openai', k) if j['consulted']], ignore_index=True)
            for k in (1, 2, 3)}
    for k in allr:
        allr[k]['collected'] = [hist.get(x, 0) for x in zip(allr[k].prompt_id, allr[k].url)]
    r = pd.concat([allr[2], allr[3]], ignore_index=True)
    r['prompt_id_rep'] = r.prompt_id + '|' + r.rep.astype(str)
    bins = [0, 1, 2, 3, 5, 10, 20, 999]
    labels = ['1', '2', '3', '4-5', '6-10', '11-20', '21+']
    r['rank_bin'] = pd.cut(r['rank'], bins, labels=labels)
    sel = r.groupby('rank_bin', observed=True).agg(consulted=('cited', 'size'), pct_cited_replay=('cited', 'mean'),
                                                   pct_ever_cited_collected=('collected', lambda x: (x > 0).mean()),
                                                   mean_collected_citations=('collected', 'mean')).reset_index()
    sel[['pct_cited_replay', 'pct_ever_cited_collected']] *= 100
    sel.to_csv(RESULTS / 'x03_selection_by_rank.csv', index=False)
    print(sel.round(2).to_string(index=False))
    from statsmodels.discrete.conditional_models import ConditionalLogit
    q = r.dropna(subset=['rank']).copy()
    q['log_rank'], q['multi_call'] = np.log2(q['rank']), (q.n_calls > 1).astype(float)
    q = q[q.groupby('prompt_id_rep').cited.transform('nunique') == 2]
    Q = ['log_rank'] + (['multi_call'] if q.multi_call.std() > 0 else [])   # needs per-call lists
    m = ConditionalLogit(q.cited.astype(float), q[Q], groups=q.prompt_id_rep).fit(disp=0)
    cl = pd.DataFrame(dict(coef=m.params, se=m.bse, p=m.pvalues, odds_ratio=np.exp(m.params)))
    cl.to_csv(RESULTS / 'x03_selection_logit.csv')
    print(cl.round(3).to_string())

    # 4 depth: collected counts vs replay rank among consulted pairs
    q['y'] = np.log2(1 + q.collected)
    dm = sm.OLS(q.y, sm.add_constant(q[Q])).fit()
    dmp = sm.OLS(q.y - q.groupby('prompt_id_rep').y.transform('mean'),
                 q[Q] - q.groupby('prompt_id_rep')[Q].transform('mean')).fit()
    dep = pd.DataFrame([dict(pairs=len(q), r2=dm.rsquared, r2_within_prompt=dmp.rsquared,
                             **{f'coef_{v}': dm.params[v] for v in Q})])
    dep.to_csv(RESULTS / 'x03_depth_rank.csv', index=False)
    print(dep.round(3).to_string(index=False))

    # 5 page level
    pages = page_level(allr, c)
    pages.to_csv(RESULTS / 'x03_page_level.csv', index=False)
    print(pages.round(3).to_string(index=False))

    rep = repeats()
    plot(sel, pages, rep)


def replay_vars(allr, k):
    """Page-level replay variables averaged over repeats 1..k."""
    r = pd.concat([allr[i] for i in range(1, k + 1)], ignore_index=True)
    return r.groupby('url').agg(n_consulted=('prompt_id', 'size'), n_top5=('rank', lambda x: (x <= 5).sum()),
                                n_cited=('cited', 'sum')) / k


def page_level(allr, c):
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    X = FEATURES + flags + types
    cc = c[~c.is_redirect & c.url.isin(d.url)]
    order = {x: i for i, x in enumerate(sorted(cc.date.unique()))}
    cc = cc.assign(half=cc.date.map(order) % 2)
    full = outcomes(cc)
    d = d[d.url.isin(full.index)].copy()
    d = d[d.groupby('domain').url.transform('size') >= 2].reset_index(drop=True)
    halves = [outcomes(cc[cc.half == h]).reindex(d.url, fill_value=0).reset_index(drop=True) for h in (0, 1)]
    R, C = ['replay_n_consulted', 'replay_n_top5'], ['replay_n_cited']
    rows = []
    for k in (1, 2, 3):
        pg = replay_vars(allr, k)
        for v in ('n_consulted', 'n_top5', 'n_cited'):
            d['replay_' + v] = np.log2(1 + pg[v].reindex(d.url).fillna(0).to_numpy())
        models = {'page variables (E1)': X, 'replay cited only': C, 'replay retrieval only': R,
                  'replay retrieval + cited': R + C, 'page variables + replay retrieval': X + R,
                  'page variables + replay retrieval + cited': X + R + C}
        if k < 3:
            models = {m: v for m, v in models.items() if m in ('replay retrieval only', 'replay retrieval + cited')}
        for y in ('total', 'breadth', 'depth'):
            d[y] = full[y].reindex(d.url).to_numpy()
            rel = rel_within(halves[0][y], halves[1][y], d.domain)
            for name, cols in models.items():
                ws, ex = vc(d, y, cols)
                rows.append(dict(replays=k, outcome=y, model=name, n_urls=len(d), within_share=ws, explained_within=ex,
                                 reliability_within=rel, stable_unexplained=ws * (rel - ex),
                                 pct_pages_consulted=100 * (d.replay_n_consulted > 0).mean()))
    return pd.DataFrame(rows)


def repeats():
    r1 = {j['prompt_id']: j for j in records('openai', 1)}
    rows = []
    for j in records('openai', 2):
        a = r1.get(j['prompt_id'])
        if not a:
            continue
        jac = lambda x, y: len(x & y) / len(x | y) if x | y else np.nan
        U = lambda xs: {clean_url(u) for u in xs}
        rows.append(dict(prompt_id=j['prompt_id'], both_searched=bool(a['queries']) and bool(j['queries']),
                         either_searched=bool(a['queries']) or bool(j['queries']),
                         queries=jac({q.lower() for q in a['queries']}, {q.lower() for q in j['queries']}),
                         consulted=jac(U(a['consulted']), U(j['consulted'])), cited=jac(U(a['cited']), U(j['cited']))))
    rp = pd.DataFrame(rows)
    s = rp[rp.both_searched]
    out = pd.DataFrame([dict(prompts=len(rp), pct_search_agree=100 * (rp.both_searched == rp.either_searched).mean(),
                             jaccard_queries=s.queries.mean(), jaccard_consulted=s.consulted.mean(),
                             jaccard_cited=s.cited.mean())])
    out.to_csv(RESULTS / 'x03_same_day_repeat.csv', index=False)
    print(out.round(3).to_string(index=False))
    return out


def plot(sel, pages, rep):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    x = np.arange(len(sel))
    ax.bar(x, sel.pct_cited_replay, color=ENGINE_COLOR['OPENAI'], width=0.6)
    for xi, v, m in zip(x, sel.pct_cited_replay, sel.mean_collected_citations):
        ax.text(xi, v + 1, f'{v:.0f}%', ha='center', fontsize=7.5, color=INK2)
    ax.set_xticks(x, sel.rank_bin)
    ax.set_xlabel("rank in ChatGPT's own search results")
    ax.set_ylabel('% of consulted pages cited')
    ax.grid(axis='x', visible=False)
    ax.set_title('Selection: a consulted page is cited mostly when it ranks high', fontsize=9.5)
    ax = axes[1]
    s = pages[pages.replays == pages.replays.max()]
    outs = ['total', 'breadth', 'depth']
    for i, (m, col) in enumerate((('page variables (E1)', '#9ec5f4'), ('page variables + replay retrieval', ENGINE_COLOR['OPENAI']))):
        v = [100 * s[(s.outcome == o) & (s.model == m)].explained_within.iloc[0] for o in outs]
        ax.bar(np.arange(3) + (i - 0.5) * 0.36, v, 0.34, color=col, label=m)
        for xi, vi in zip(np.arange(3) + (i - 0.5) * 0.36, v):
            ax.text(xi, vi + 0.8, f'{vi:.0f}', ha='center', fontsize=7.5, color=INK2)
    rel = [100 * s[(s.outcome == o)].reliability_within.iloc[0] for o in outs]
    ax.scatter(np.arange(3), rel, marker='_', s=900, color='#52514c', label='repeatable ceiling', zorder=3)
    ax.set_xticks(np.arange(3), outs)
    ax.set_ylim(0, 100)
    ax.set_ylabel('% of within-website variation explained')
    ax.legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.grid(axis='x', visible=False)
    ax.set_title('ChatGPT pages: adding three replay runs of retrieval data', fontsize=9.5)
    fig.suptitle('X3: replaying ChatGPT with its search results visible', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'x03_retrieval.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
