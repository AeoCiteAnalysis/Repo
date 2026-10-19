"""X1 - Does search rank explain which pages get cited?

R13 left ~72 % of the variation in citation counts stable but unexplained. All three engines
search the web before answering (Gemini's citation links in our data are Google "grounding"
redirects, i.e. Google Search results), so a page's search rank for each prompt is the prime
candidate.

  python scripts/x01_search_rank.py fetch [--engine google|bing] [--limit N]
  python scripts/x01_search_rank.py analyse [--engine google]

fetch    One SerpAPI search per prompt (651 per engine), US results in English, top 10 organic
         results. Needs SERPAPI_KEY in .env. Checks the account's remaining searches first and
         stops if there are not enough. Every response is cached in data/serp/<engine>/<prompt_id>.json
         and never fetched twice. Snapshot date is recorded (results are from today, citations
         are from 24 Jun - 21 Sep 2026; the analysis also uses only the last 14 days of citations).
analyse  1. Overlap: share of each engine's cited URLs / websites for a prompt that are in the search
            top 10 for that prompt.
         2. Rank curve: for pages in the top 10, share cited for that prompt, by rank position.
         3. Page level: number of prompts a page ranks top 10 for (and a rank-weighted version) added
            to the R2 model (website indicators) - gain in within-website R², and within-website
            variance explained in a mixed model (R13: 3.6 % with the Step-A6 features).
         4. Search-results pages never cited: how many top-10 results are never cited at all.

Outputs: data/serp/, data/tables/serp_results.csv, results/x01_*.csv, figures/x01_search_rank.png
"""
import argparse
import json
import os
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

import plotstyle
from plotstyle import plt, ENGINE_COLOR, INK2, MUTED
from common import ROOT, TABLES, RESULTS, FIGURES, ENGINES, ENGINE_LABEL, clean_url, domain_of, read_table, drop_excluded
from fe import fe_ols
from r02_main_model import prepare, FEATURES

warnings.filterwarnings('ignore')
SERP = ROOT / 'data' / 'serp'
API = 'https://serpapi.com/search.json'
ACCOUNT = 'https://serpapi.com/account.json'
PARAMS = {'google': dict(engine='google', gl='us', hl='en', num=10, google_domain='google.com'),
          'bing': dict(engine='bing', cc='US', count=10)}


def key():
    k = os.environ.get('SERPAPI_KEY')
    if not k:
        raise SystemExit('set SERPAPI_KEY in .env (see .env.example)')
    return k


def searches_left(k):
    r = requests.get(ACCOUNT, params={'api_key': k}, timeout=30)
    r.raise_for_status()
    j = r.json()
    return j.get('total_searches_left', j.get('plan_searches_left')), j


def fetch(engine, limit):
    k = key()
    lab = pd.read_csv(TABLES / 'prompt_labels.csv', usecols=['prompt_id', 'prompt'])
    out = SERP / engine
    out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in lab.itertuples() if not (out / f'{r.prompt_id}.json').exists()]
    if limit:
        todo = todo[:limit]
    left, acct = searches_left(k)
    print(f'{engine}: {len(todo)} searches needed; account has {left} searches left '
          f'(plan: {acct.get("plan_name", "?")})')
    if left is not None and left < len(todo):
        raise SystemExit('not enough searches left - use --limit or top up the plan')
    for i, r in enumerate(todo, 1):
        for attempt in range(3):
            try:
                resp = requests.get(API, params=dict(PARAMS[engine], q=r.prompt, api_key=k), timeout=90)
                if resp.status_code == 200:
                    j = resp.json()
                    j['_snapshot_date'] = f'{date.today():%Y-%m-%d}'
                    (out / f'{r.prompt_id}.json').write_text(json.dumps(j), encoding='utf-8')
                    break
                print(f'  {r.prompt_id}: http {resp.status_code}')
            except requests.RequestException as e:
                print(f'  {r.prompt_id}: {type(e).__name__}')
            time.sleep(5 * (attempt + 1))
        if i % 50 == 0:
            print(f'  {i}/{len(todo)}')


def parse(engine):
    rows = []
    for f in sorted((SERP / engine).glob('*.json')):
        j = json.loads(f.read_text(encoding='utf-8'))
        for o in j.get('organic_results') or []:
            link = o.get('link')
            if link:
                u = clean_url(link)
                rows.append(dict(engine=engine, prompt_id=f.stem, rank=o.get('position'), url=u,
                                 domain=domain_of(u), snapshot=j.get('_snapshot_date')))
    s = pd.DataFrame(rows)
    s = s[s['rank'] <= 10]
    s.to_csv(TABLES / f'serp_results_{engine}.csv', index=False)
    return s


def analyse(engine):
    s = parse(engine)
    if s.empty:
        raise SystemExit(f'no cached {engine} results - run fetch first')
    n_prompts = s.prompt_id.nunique()
    print(f'{engine}: {n_prompts} prompts, {len(s):,} top-10 results, snapshot {s.snapshot.iloc[0]}')
    c = drop_excluded(read_table('citations.csv', usecols=['date', 'engine', 'prompt_id', 'url', 'domain',
                                                          'is_redirect', 'repeat_in_answer']))
    c = c[~c.repeat_in_answer & c.prompt_id.isin(s.prompt_id)]
    last = sorted(c.date.unique())[-14:]
    serp_url = s.groupby('prompt_id').url.agg(set).to_dict()
    serp_dom = s.groupby('prompt_id').domain.agg(set).to_dict()

    # 1. overlap per engine
    rows = []
    for period, cc in (('all 90 days', c), ('last 14 days', c[c.date.isin(last)])):
        for e in ENGINES:
            g = cc[cc.engine == e]
            if g.empty:
                continue
            u = g[~g.is_redirect].drop_duplicates(['prompt_id', 'url'])
            dd = g.drop_duplicates(['prompt_id', 'domain'])
            rows.append(dict(period=period, engine=ENGINE_LABEL[e],
                             pct_cited_urls_in_top10=100 * np.mean([x in serp_url.get(p, ()) for p, x in zip(u.prompt_id, u.url)]),
                             pct_cited_websites_in_top10=100 * np.mean([x in serp_dom.get(p, ()) for p, x in zip(dd.prompt_id, dd.domain)]),
                             cited_prompt_url_pairs=len(u)))
    ov = pd.DataFrame(rows)
    ov.to_csv(RESULTS / f'x01_overlap_{engine}.csv', index=False)

    # 2. rank curve: top-10 results cited for that prompt?
    cited_pairs = {e: set(zip(c[c.engine == e].prompt_id, c[c.engine == e].url)) for e in ENGINES}
    cited_any = set(zip(c.prompt_id, c.url))
    for e in ENGINES:
        s[f'cited_{e}'] = [(p, u) in cited_pairs[e] for p, u in zip(s.prompt_id, s.url)]
    s['cited_any'] = [(p, u) in cited_any for p, u in zip(s.prompt_id, s.url)]
    curve = s.groupby('rank')[[f'cited_{e}' for e in ENGINES] + ['cited_any']].mean().mul(100)
    curve.to_csv(RESULTS / f'x01_rank_curve_{engine}.csv')
    ever = set(c.url)
    never = dict(top10_results=len(s), pct_never_cited_by_any_engine=100 * (~s.url.isin(ever)).mean(),
                 pct_not_cited_for_this_prompt=100 * (~s.cited_any).mean())

    # 3. page level: search breadth in the R2 model
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    d = d[d.groupby('domain').y.transform('size') >= 2].reset_index(drop=True)
    cnt = s.groupby('url').prompt_id.nunique()
    rr = s.assign(rr=1 / s['rank']).groupby('url').rr.sum()
    d['serp_prompts_top10'] = d.url.map(cnt).fillna(0)
    d['serp_rank_weighted'] = d.url.map(rr).fillna(0)
    d['serp_any'] = (d.serp_prompts_top10 > 0).astype(float)
    for f in ('serp_prompts_top10', 'serp_rank_weighted'):
        v = np.log1p(d[f])
        d[f] = (v - v.mean()) / v.std()
    X0 = FEATURES + flags + types
    _, i0 = fe_ols(d, 'y', X0, 'domain')
    prow = [dict(model='R2 features', r2_within=i0['r2_within'])]
    for extra in (['serp_prompts_top10'], ['serp_rank_weighted'], ['serp_any']):
        res, info = fe_ols(d, 'y', X0 + extra, 'domain')
        e = extra[0]
        prow.append(dict(model=f'R2 features + {e}', r2_within=info['r2_within'],
                         effect_pct=100 * (2 ** res.loc[e, 'coef'] - 1), p=res.loc[e, 'p']))
    null = sm.MixedLM(d.y, np.ones((len(d), 1)), groups=d.domain).fit(reml=True).scale
    for label, X in (('Step-A6 features', X0), ('+ search breadth', X0 + ['serp_rank_weighted'])):
        sc = sm.MixedLM(d.y, sm.add_constant(d[X]), groups=d.domain).fit(reml=True).scale
        prow.append(dict(model=f'mixed model, within-website variance explained ({label})', r2_within=1 - sc / null))
    pl = pd.DataFrame(prow)
    pl['pages_in_any_top10'] = int(d.serp_any.sum())
    pl.to_csv(RESULTS / f'x01_page_level_{engine}.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    g = ov[ov.period == 'all 90 days']
    x = np.arange(len(g))
    axes[0].bar(x - 0.18, g.pct_cited_urls_in_top10, width=0.34, color=[ENGINE_COLOR[k] for k in ENGINES][:len(g)], label='cited URL in top 10')
    axes[0].bar(x + 0.18, g.pct_cited_websites_in_top10, width=0.34, color='#c3c2b7', label='cited website in top 10')
    for xi, v in zip(x - 0.18, g.pct_cited_urls_in_top10):
        axes[0].text(xi, v + 0.5, f'{v:.0f}%', ha='center', fontsize=7.5, color=INK2)
    for xi, v in zip(x + 0.18, g.pct_cited_websites_in_top10):
        axes[0].text(xi, v + 0.5, f'{v:.0f}%', ha='center', fontsize=7.5, color=INK2)
    axes[0].set_xticks(x, g.engine); axes[0].set_ylabel('% of cited (prompt, source) pairs')
    axes[0].set_title(f'Cited sources that are in the {engine.title()} top 10 for the same prompt', fontsize=9.5)
    axes[0].legend(fontsize=7.5); axes[0].grid(axis='x', visible=False)
    for e in ENGINES:
        axes[1].plot(curve.index, curve[f'cited_{e}'], 'o-', color=ENGINE_COLOR[e], ms=4, label=ENGINE_LABEL[e])
    axes[1].set_xlabel(f'{engine.title()} rank'); axes[1].set_ylabel('% cited for that prompt')
    axes[1].set_xticks(range(1, 11)); axes[1].legend(fontsize=7.5)
    axes[1].set_title('Chance a top-10 result is cited, by rank', fontsize=9.5)
    fig.suptitle(f'X1: search rank and AI citations ({engine.title()}, {n_prompts} prompts)', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(); fig.savefig(FIGURES / f'x01_search_rank_{engine}.png'); plt.close(fig)

    print(ov.round(1).to_string(index=False))
    print(curve.round(1).to_string())
    print(never)
    print(pl.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fetch', 'analyse'])
    ap.add_argument('--engine', default='google', choices=list(PARAMS))
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    fetch(a.engine, a.limit) if a.cmd == 'fetch' else analyse(a.engine)


if __name__ == '__main__':
    main()
