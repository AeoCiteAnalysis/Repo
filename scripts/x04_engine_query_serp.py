"""X4 (E7) - Does the engines' retrieval follow Bing's or Google's ranking for THEIR OWN queries?

X1 searched Google for the question as typed and found almost none of the engines' sources in the
top 10. The replay (x03) shows the engines' real search queries. Here those queries are sent to
Bing and Google through SerpAPI (US, English, top 10 organic):
  ChatGPT  120 random searched prompts (seed 42) from replay repeat 2 (which keeps per-call source
           lists); the first search call's first two queries -> Bing and Google (480 searches).
           Compared with that call's source list (what ChatGPT retrieved) and the record's cited URLs.
  Gemini   120 random searched prompts from replay repeat 1; its first two Google queries -> Google
           (240 searches). Compared with Gemini's grounding sources for that answer.
Measures, per engine and search provider: share of retrieved / cited URLs and websites that are in
the union of the two queries' top 10; for ChatGPT also the rank correlation between its source-list
order and the provider's order for the shared URLs.

  python scripts/x04_engine_query_serp.py fetch --engine openai
  python scripts/x04_engine_query_serp.py fetch --engine gemini
  python scripts/x04_engine_query_serp.py analyse
Cache: data/serp/engine_queries/<provider>/<sha1(query)>.json. Output: results/x04_engine_query_serp.csv
"""
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

from common import ROOT, RESULTS, SEED, domain_of
from common import clean_url as _clean

TRACKING = ('msockid', 'srsltid', 'gclid', 'fbclid', 'hl', 'rd', 'src', 'visit_id')


def clean_url(u):
    """clean_url plus removal of search-engine tracking parameters (Bing msockid, Google srsltid)."""
    p = urlparse(_clean(u))
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in TRACKING]
    return urlunparse(p._replace(query=urlencode(q)))
from x01_search_rank import API, key, searches_left
from x03_analyse import records

OUT = ROOT / 'data' / 'serp' / 'engine_queries'
N_PROMPTS, N_QUERIES = 120, 2
PROVIDERS = {'openai': ('bing', 'google'), 'gemini': ('google',)}
PARAMS = {'google': dict(engine='google', gl='us', hl='en', location='United States', google_domain='google.com'),
          'bing': dict(engine='bing', cc='US', mkt='en-US')}


def sample(engine):
    recs = [j for j in records(engine, rep=2 if engine == 'openai' else 1) if j['queries']]
    if engine == 'openai':
        recs = [j for j in recs if j.get('calls')]
    recs = sorted(recs, key=lambda j: j['prompt_id'])
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(recs), size=min(N_PROMPTS, len(recs)), replace=False)
    out = []
    for i in sorted(pick):
        j = recs[i]
        if engine == 'openai':
            call = j['calls'][0]
            out.append(dict(prompt_id=j['prompt_id'], queries=call['queries'][:N_QUERIES],
                            retrieved=call['sources'], cited=j['cited']))
        else:
            out.append(dict(prompt_id=j['prompt_id'], queries=j['queries'][:N_QUERIES],
                            retrieved=j['consulted'], cited=j['cited']))
    return out


def path(provider, q):
    return OUT / provider / (hashlib.sha1(q.lower().encode()).hexdigest()[:16] + '.json')


def fetch(engine):
    k = key()
    s = sample(engine)
    todo = [(p, q) for p in PROVIDERS[engine] for x in s for q in x['queries'] if not path(p, q).exists()]
    left, _ = searches_left(k)
    print(f'{engine}: {len(s)} prompts, {len(todo)} searches needed, {left} left')
    if left is not None and left < len(todo):
        raise SystemExit('not enough searches left')
    def one(item):
        p, q = item
        path(p, q).parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                r = requests.get(API, params=dict(PARAMS[p], q=q, api_key=k), timeout=90)
                if r.status_code == 200:
                    j = r.json()
                    path(p, q).write_text(json.dumps(dict(query=q, organic=[
                        dict(position=o.get('position'), link=o.get('link')) for o in j.get('organic_results') or []
                    ])), encoding='utf-8')
                    break
                print(f'  http {r.status_code}')
            except requests.RequestException as e:
                print(f'  {type(e).__name__}')
            time.sleep(5 * (attempt + 1))

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 50 == 0:
                print(f'  {i}/{len(todo)}', flush=True)


def top10(p, q):
    f = path(p, q)
    if not f.exists():
        return None
    org = json.loads(f.read_text(encoding='utf-8'))['organic']
    return [clean_url(o['link']) for o in sorted(org, key=lambda o: o['position'] or 99) if o['link']][:10]


def analyse():
    rows = []
    for engine in PROVIDERS:
        for x in sample(engine):
            ret = list(dict.fromkeys(clean_url(u) for u in x['retrieved']))
            cit = {clean_url(u) for u in x['cited']}
            for p in PROVIDERS[engine]:
                lists = [top10(p, q) for q in x['queries']]
                if any(l is None for l in lists):
                    continue
                top = set().union(*lists)
                tdom = {domain_of(u) for u in top}
                order = {u: i for i, u in enumerate(lists[0])}
                shared = [u for u in ret if u in order]
                rho = spearmanr([ret.index(u) for u in shared], [order[u] for u in shared])[0] if len(shared) >= 4 else np.nan
                rows.append(dict(engine=engine, provider=p, prompt_id=x['prompt_id'],
                                 retrieved=len(ret), cited=len(cit),
                                 pct_retrieved_in_top10=100 * np.mean([u in top for u in ret]) if ret else np.nan,
                                 pct_retrieved_sites_in_top10=100 * np.mean([domain_of(u) in tdom for u in ret]) if ret else np.nan,
                                 pct_cited_in_top10=100 * np.mean([u in top for u in cit]) if cit else np.nan,
                                 pct_cited_sites_in_top10=100 * np.mean([domain_of(u) in tdom for u in cit]) if cit else np.nan,
                                 pct_top10_retrieved=100 * np.mean([u in set(ret) for u in top]) if top else np.nan,
                                 rank_corr=rho))
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / 'x04_engine_query_serp_by_prompt.csv', index=False)
    out = d.groupby(['engine', 'provider']).agg(prompts=('prompt_id', 'size'),
                                                **{c: (c, 'mean') for c in d.columns if c.startswith('pct_') or c == 'rank_corr'}).reset_index()
    out.to_csv(RESULTS / 'x04_engine_query_serp.csv', index=False)
    print(out.round(1).T.to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fetch', 'analyse'])
    ap.add_argument('--engine', choices=list(PROVIDERS), default='openai')
    a = ap.parse_args()
    fetch(a.engine) if a.cmd == 'fetch' else analyse()
