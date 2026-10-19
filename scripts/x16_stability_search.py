"""X16 - X15 with web search: is the ranking of brands still described by one score per brand when each
answer finds its own evidence?

Same 20 sets (question + 6 brands) and the same two models as X15, but instead of fixed fact sheets the
model searches the web (ChatGPT: gpt-5.4-nano with the web_search tool; Gemini: gemini-3.1-flash-lite with
Google Search grounding). The prompt names the brands to compare, in a given order. Per set:

  pair   15 pairs x both orders x 2 repeats              60
  full   6 cyclic orders x 2 repeats                     12
  drop   each brand removed (5 left)                      6
  add    a 7th brand added (3 different ones)             3
  para   3 rewordings x 2 orders (cyclic shift 0 and 3)   6

Every answer's search queries, consulted and cited sources are stored, so instability can be split into
"the evidence changed" and "the model changed its mind on the same evidence".

Two arms with identical wording apart from the search directive:
  search   "Search the web: <question> ... Briefly explain, citing sources, and finish with one line of JSON"
           (web search on; this wording is needed - with a JSON-only prompt Gemini did not search at all)
  memory   the same prompt without "Search the web:" and "citing sources", search off, no fact sheets:
           the model's own knowledge only
Together with X15 (fixed fact sheets) this gives three evidence regimes: fixed, none, searched.

  python scripts/x16_stability_search.py run        # answers.jsonl (resumable; X16_LIMIT=n for a pilot)
  python scripts/x16_stability_search.py cost       # calls, searches and tokens so far
  python scripts/x16_stability_search.py analyse    # the X15 measures side by side + evidence decomposition
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import x14_fact_sheets as fs  # noqa: E402
import x15_stability as x15  # noqa: E402

OUT = ROOT / 'data' / 'x16'
RESULTS = ROOT / 'results'
SEED = 42
PAIR_REPS, FULL_REPS = 2, 2

ARMS = ('search', 'memory', 'facts')
LEAD = {'search': 'Search the web: {q}\n', 'memory': '{q}\n', 'facts': '{q}\n'}
CITE = {'search': ', citing sources', 'memory': '', 'facts': ''}
# facts: the X15 fact sheets of the compared products, in the order they are named, with the X16 wording;
# separates the answer format (X15 vs X16 facts) from the evidence (X16 facts vs memory vs search)


def profiles(ds, items):
    sheets = profiles.sheets
    m = fs.MARKETS[ds]
    out = ['Product profiles:']
    for b in items:
        out += ['', f'[{b}]', fs.render_sheet(sheets[b]['sheet'], m['fields'], b)]
    return '\n'.join(out) + '\n\n'
RANK = ('Compare {items} and rank all {k}, best first. Briefly explain{cite}, and finish with one line of JSON: '
        '{{"ranking": [...]}} naming each product exactly as written above.')
PAIR = ('Compare {a} and {b}: which is the better choice? You must choose one. Briefly explain{cite}, and finish '
        'with one line of JSON: {{"choice": "<product>"}} naming the product exactly as written above.')


def jobs_all():
    d = json.loads((ROOT / 'data' / 'x15' / 'sets.json').read_text(encoding='utf-8'))
    profiles.sheets = json.loads((ROOT / 'data' / 'x14' / 'sheets.json').read_text(encoding='utf-8'))
    jobs = []
    for s, arm in itertools.product(d['sets'], ARMS):
        ds, q, B = s['dataset'], s['question'], s['brands']
        lead = LEAD[arm].format(q=q)
        base = dict(sid=s['sid'], dataset=ds, arm=arm)

        def ev(items, arm=arm, ds=ds):
            return profiles(ds, items) if arm == 'facts' else ''

        def rank(items, question=q, arm=arm):
            return (LEAD[arm].format(q=question) + ev(items)
                    + RANK.format(k=len(items), items=', '.join(items), cite=CITE[arm]))

        for a, b in itertools.combinations(B, 2):
            for first, second in ((a, b), (b, a)):
                for rep in range(PAIR_REPS):
                    jobs.append(base | dict(part='pair', items=[first, second], rep=rep, qv=0,
                                            text=lead + ev([first, second]) + PAIR.format(a=first, b=second, cite=CITE[arm])))
        for r in range(len(B)):
            for rep in range(FULL_REPS):
                jobs.append(base | dict(part='full', items=x15.rot(B, r), rot=r, rep=rep, qv=0,
                                        text=rank(x15.rot(B, r))))
        for i, drop in enumerate(B):
            items = x15.rot([b for b in B if b != drop], i % (len(B) - 1))
            jobs.append(base | dict(part='drop', dropped=drop, filler=False, items=items, rep=0, qv=0,
                                    text=rank(items)))
        for j, add in enumerate(s['extra']):
            items = B[:]
            items.insert((2 * j + 1) % (len(B) + 1), add)
            jobs.append(base | dict(part='add', added=add, items=items, rep=0, qv=0, text=rank(items)))
        for v, pq in enumerate(s['paraphrases'], start=1):
            for r in (0, 3):
                jobs.append(base | dict(part='para', items=x15.rot(B, r), rot=r, rep=0, qv=v,
                                        text=rank(x15.rot(B, r), pq)))
    return d, jobs


def ask(engine, text, arm):
    if arm in ('memory', 'facts'):                          # same models, no search tool
        ins = 'Answer the user.' + (' Use only the information provided in the message.' if arm == 'facts' else '')
        return dict(answer=fs.ask(engine, ins, text), queries=[], consulted=[], cited=[], model=x15.MODELS[engine])
    import x03_replay as rp
    return rp.ask_openai(text) if engine == 'openai' else rp.ask_gemini(text)


def run():
    import common  # noqa: F401
    _, jobs = jobs_all()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'answers.jsonl'
    done = set()
    if path.exists():
        done = {json.loads(l)['key'] for l in open(path, encoding='utf-8')}
    todo = []
    for j in jobs:
        for eng, model in x15.MODELS.items():
            k = hashlib.sha1(f'{j["arm"]}|{model}|{j["text"]}|{j["rep"]}'.encode()).hexdigest()
            if k not in done:
                todo.append(j | dict(engine=eng, key=k))
    if os.environ.get('X16_ARMS'):                          # e.g. X16_ARMS=facts (search quota exhausted)
        todo = [j for j in todo if j['arm'] in os.environ['X16_ARMS'].split(',')]
    if int(os.environ.get('X16_LIMIT', 0)):
        todo = random.Random(1).sample(todo, int(os.environ['X16_LIMIT']))
    print(f'{len(todo)} calls to make', flush=True)
    lock, n = threading.Lock(), [0]

    def one(j):
        try:
            r = ask(j['engine'], j['text'], j['arm'])
        except Exception as e:
            print(f'  {j["engine"]} {j["part"]} s{j["sid"]}: {type(e).__name__}: {str(e)[:120]}', flush=True)
            return
        rec = {k: v for k, v in j.items() if k != 'text'} | dict(
            answer=r['answer'], queries=r.get('queries') or [], consulted=r.get('consulted') or [],
            cited=r.get('cited') or [], usage=r.get('usage'), model=r.get('model'))
        with lock:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
            n[0] += 1
            if n[0] % 250 == 0:
                print(f'  {n[0]}/{len(todo)}', flush=True)

    for arm, eng, workers in (('facts', 'openai', 8), ('facts', 'gemini', 6), ('memory', 'openai', 8),
                              ('memory', 'gemini', 6), ('search', 'openai', 6), ('search', 'gemini', 4)):
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, [j for j in todo if j['engine'] == eng and j['arm'] == arm]))
    print('done', flush=True)


def cost():
    rows = [json.loads(l) for l in open(OUT / 'answers.jsonl', encoding='utf-8')]
    d = pd.DataFrame(rows)
    d['n_queries'] = d.queries.map(len)
    d['n_consulted'] = d.consulted.map(len)
    d['n_cited'] = d.cited.map(len)
    d['searched'] = d.n_queries > 0

    def tok(u, *keys):
        if not isinstance(u, dict):
            return 0
        return sum(u.get(k) or 0 for k in keys)
    d['in_tok'] = [tok(u, 'input_tokens', 'promptTokenCount') for u in d.usage]
    d['out_tok'] = [tok(u, 'output_tokens', 'candidatesTokenCount', 'thoughtsTokenCount') for u in d.usage]
    s = d.groupby(['arm', 'engine']).agg(calls=('key', 'size'), searched=('searched', 'mean'), queries=('n_queries', 'sum'),
                                queries_per_call=('n_queries', 'mean'), consulted_per_call=('n_consulted', 'mean'),
                                cited_per_call=('n_cited', 'mean'), input_tokens=('in_tok', 'sum'),
                                output_tokens=('out_tok', 'sum'))
    print(s.round(2).to_string())
    return s


# ---------------------------------------------------------------- analyse

def parse_last(a):
    import re
    for m in reversed(re.findall(r'\{[^{}]*\}', a['answer'] or '')):
        try:
            return json.loads(m)
        except Exception:
            continue
    return None


def load(arm):
    rows = [json.loads(l) for l in open(OUT / 'answers.jsonl', encoding='utf-8')]
    rows = [a for a in rows if a['arm'] == arm]
    pairs, ranks, bad, meta = [], [], 0, []
    for a in rows:
        js = parse_last(a)
        ev = dict(queries=a['queries'], consulted=[norm(u) for u in a['consulted']], cited=[norm(u) for u in a['cited']])
        if a['part'] == 'pair':
            c = x15.match(js.get('choice'), a['items']) if js else None
            if c is None:
                bad += 1
                continue
            x, y = sorted(a['items'])
            pairs.append(dict(engine=a['engine'], sid=a['sid'], dataset=a['dataset'], a=x, b=y, first=a['items'][0],
                              rep=a['rep'], a_wins=c == x, first_wins=c == a['items'][0]) | ev)
        else:
            r = [x15.match(v, a['items']) for v in (js or {}).get('ranking', [])] if js else []
            r = [v for v in dict.fromkeys(r) if v]
            if len(r) != len(a['items']):
                bad += 1
                continue
            ranks.append({k: a.get(k) for k in ('engine', 'sid', 'dataset', 'part', 'rep', 'rot', 'qv', 'dropped',
                                                  'filler', 'added')} | dict(items=a['items'], ranking=r) | ev)
    print(f'{arm}: {len(rows)} answers, {bad} unparsed or incomplete')
    return pd.DataFrame(pairs), pd.DataFrame(ranks)


def norm(u):
    p = urlparse(u or '')
    return (p.netloc.lower().removeprefix('www.') + p.path.rstrip('/')).lower()


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else np.nan


def x15_measures(P, R):
    """Run the X15 analysis unchanged on X16 answers; returns its results table (X15's own file is kept)."""
    import shutil
    orig_csv = RESULTS / 'x15_stability.csv'
    backup = orig_csv.read_bytes()
    x15_out, orig_load = x15.OUT, x15.load
    shutil.copy(ROOT / 'data' / 'x15' / 'sets.json', OUT / 'sets.json')
    x15.OUT, x15.load = OUT, (lambda: (P, R))
    try:
        x15.analyse()
        return pd.read_csv(orig_csv)
    finally:
        orig_csv.write_bytes(backup)
        x15.OUT, x15.load = x15_out, orig_load


def analyse():
    x15_res = pd.read_csv(RESULTS / 'x15_stability.csv')
    res = {'x15 fixed facts': x15_res}
    loaded = {}
    for arm in ARMS:
        P, R = load(arm)
        loaded[arm] = (P, R)
        res[f'x16 {arm}'] = x15_measures(P, R)
        res[f'x16 {arm}'].to_csv(RESULTS / f'x16_stability_{arm}.csv', index=False)
    both = None
    for k, v in res.items():
        v = v[['engine', 'part', 'measure', 'value']].rename(columns={'value': k})
        both = v if both is None else both.merge(v, on=['engine', 'part', 'measure'], how='outer')
    both.to_csv(RESULTS / 'x16_vs_x15.csv', index=False)
    P, R = loaded['search']
    # evidence decomposition: identical repeats (same prompt, same order) - how much of the ranking change
    # goes with a change of the sources?
    ev = []
    full = R[R.part == 'full']
    for (eng, sid, rt), g in full.groupby(['engine', 'sid', 'rot']):
        if len(g) != 2:
            continue
        a, b = g.iloc[0], g.iloc[1]
        ev.append(dict(engine=eng, sid=sid, tau=x15.kendall(a.ranking, b.ranking),
                       j_consulted=jaccard(a.consulted, b.consulted), j_cited=jaccard(a.cited, b.cited),
                       j_queries=jaccard([q.lower() for q in a.queries], [q.lower() for q in b.queries])))
    ev = pd.DataFrame(ev)
    ev.to_csv(RESULTS / 'x16_evidence_pairs.csv', index=False)
    rows = []
    import statsmodels.formula.api as smf
    for eng, g in ev.groupby('engine'):
        g = g.dropna(subset=['j_consulted'])
        rows.append(dict(engine=eng, measure='identical repeats: mean Kendall tau', value=g.tau.mean()))
        rows.append(dict(engine=eng, measure='identical repeats: consulted-source overlap (Jaccard)', value=g.j_consulted.mean()))
        rows.append(dict(engine=eng, measure='identical repeats: search-query overlap (Jaccard)', value=g.j_queries.mean()))
        if len(g) > 10 and g.j_consulted.std() > 0:
            m = smf.ols('tau ~ j_consulted', g).fit(cov_type='cluster', cov_kwds={'groups': g.sid})
            pred1 = m.params['Intercept'] + m.params['j_consulted']
            rows.append(dict(engine=eng, measure='tau per +1 source overlap (slope)', value=m.params['j_consulted'],
                             ci_low=m.conf_int().loc['j_consulted', 0], ci_high=m.conf_int().loc['j_consulted', 1],
                             p=m.pvalues['j_consulted']))
            rows.append(dict(engine=eng, measure='predicted tau with identical sources (overlap = 1)', value=pred1))
        # brands the answers look up: share of answers whose queries name each compared brand
    evs = pd.DataFrame(rows)
    evs.to_csv(RESULTS / 'x16_evidence.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 80)
    print('\n=== X15 (fixed facts) vs X16 (memory, search)')
    print(both.round(3).to_string(index=False))
    print('\n=== evidence decomposition (identical repeats)')
    print(evs.round(3).to_string(index=False))
    cost()


def by_market():
    """Per arm, engine and market: agreement of identical repeats, of different orders, and Bradley-Terry fit."""
    sets = {s_['sid']: s_ for s_ in json.loads((ROOT / 'data' / 'x15' / 'sets.json').read_text(encoding='utf-8'))['sets']}
    rows = []
    loaded = {arm: load(arm) for arm in ARMS}
    loaded['x15'] = x15.load()
    for arm, (P, R) in loaded.items():
        for eng in x15.MODELS:
            for ds in ('gfiber', 'workorb', 'all'):
                r = R[(R.engine == eng) & ((R.dataset == ds) | (ds == 'all'))]
                p = P[(P.engine == eng) & ((P.dataset == ds) | (ds == 'all'))]
                full = r[r.part == 'full']
                same = [x15.kendall(g.ranking.iloc[0], g.ranking.iloc[1]) for _, g in full.groupby(['sid', 'rot'])
                        if len(g) == 2]
                bt = []
                for sid, gs in p.groupby('sid'):
                    sc = x15.bt_fit(gs, sets[sid]['brands'])
                    bt += [x15.agree_pairs(rk, lambda u, v, sc=sc: sc[u] - sc[v]) for rk in full[full.sid == sid].ranking]
                first = np.mean([rk.index(it[0]) + 1 for rk, it in zip(full.ranking, full['items'])])
                last = np.mean([rk.index(it[-1]) + 1 for rk, it in zip(full.ranking, full['items'])])
                rows.append(dict(arm=arm, engine=eng, dataset=ds, tau_repeat=np.nanmean(same), bt_full=100 * np.nanmean(bt),
                                 rank_listed_first=first, rank_listed_last=last, answers=len(r) + len(p)))
    x = pd.DataFrame(rows)
    x.to_csv(RESULTS / 'x16_by_market.csv', index=False)
    print(x.round(2).to_string(index=False))
    return x


def plot():
    import common
    import plotstyle  # noqa: F401
    from plotstyle import plt, ENGINE_COLOR, INK2
    v = pd.read_csv(RESULTS / 'x16_vs_x15.csv')
    cols = [('x15 fixed facts', 'fixed facts\n(X15, JSON only)'), ('x16 facts', 'fixed facts\n(X16 wording)'),
            ('x16 search', 'web search'), ('x16 memory', 'memory only\n(no evidence)')]
    col = {'openai': ENGINE_COLOR['OPENAI'], 'gemini': ENGINE_COLOR['GEMINI']}
    name = {'openai': 'ChatGPT (gpt-5.4-nano)', 'gemini': 'Gemini (3.1-flash-lite)'}
    panels = [('Kendall tau, identical repeat (same order)', 'same prompt twice: ranking agreement (Kendall tau)', (0, 1.05)),
              ('full ranking: pairs ordered as Bradley-Terry predicts (%)', 'full-ranking pairs predicted by\none score per brand (%)', (50, 100))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (measure, ylab, ylim) in zip(axes, panels):
        for k, eng in enumerate(('openai', 'gemini')):
            row = v[(v.engine == eng) & (v.measure == measure)].iloc[0]
            vals = [row[c] for c, _ in cols]
            xs = np.arange(len(cols)) + (k - 0.5) * 0.36
            ax.bar(xs, vals, 0.34, color=col[eng], label=name[eng])
            for xi, vv in zip(xs, vals):
                ax.text(xi, vv + (0.015 if ylim[1] < 2 else 0.8), f'{vv:.2f}' if ylim[1] < 2 else f'{vv:.0f}',
                        ha='center', fontsize=8, color=INK2)
        ax.set_xticks(range(len(cols)), [b for _, b in cols], fontsize=8.5)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
        ax.grid(axis='x', visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc='upper right', ncol=2, frameon=False)
    axes[0].set_title('Consistency depends on the evidence', fontsize=9.5)
    axes[1].set_title('One score per brand still describes the ranking', fontsize=9.5)
    fig.suptitle('X16: ranking stability with fixed facts, web search, or memory only', x=0.01, ha='left',
                 fontweight='bold')
    fig.tight_layout()
    for d_ in (common.ROOT / 'figures', common.ROOT / 'figures' / 'workorb'):
        fig.savefig(d_ / 'x16_search.png')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'cost', 'analyse', 'by_market', 'plot'])
    {'run': run, 'cost': cost, 'analyse': analyse, 'by_market': by_market, 'plot': plot}[ap.parse_args().cmd]()
