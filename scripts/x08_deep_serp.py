"""X8 - Are the engines' sources deeper in Google/Bing, or not there at all?

X4 found that only 0.5-0.7 % of ChatGPT's retrieved pages are in Bing's or Google's TOP 10 for its own
queries. Two explanations: the engines take pages from further down the same search results, or from
an index of their own. Here every query of a sampled answer is searched DEEP:
  ChatGPT  60 answers (50 GFiber + 10 Workorb, random, seed 42) from replay runs with per-call source
           lists; ALL queries of the answer -> Google top 50 (5 pages) and Bing top 30 (3 pages)
  Gemini   50 answers (40 + 10); all its queries -> Google top 50
Google and Bing return ~10 results per request (Google dropped num=100 in 2025), so depth is paged.
For each retrieved (ChatGPT) or cited page: its best position over the union of the answer's queries'
results -> top 10 / 11-30 / 31-50 / not found; the same at website level.

  AEO_DATASET=<ds> python scripts/x08_deep_serp.py fetch --n 50
  AEO_DATASET=<ds> python scripts/x08_deep_serp.py analyse --n 50|10   (one dataset per process)
Cache: data/serp/deep/<provider>/<sha1(query)>_<start>.json. Output: results/x08_deep_serp*.csv
"""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from common import ROOT, RESULTS, SEED, clean_url, domain_of
from x01_search_rank import API, key, searches_left

OUT = ROOT / 'data' / 'serp' / 'deep'
SAMPLE_MAX_DATE = '20260928'      # replays that existed when the searches were fetched; later daily runs do not change the sample
DEPTH = {('openai', 'google'): 50, ('openai', 'bing'): 30, ('gemini', 'google'): 50}
PARAMS = {'google': dict(engine='google', gl='us', hl='en', location='United States', google_domain='google.com'),
          'bing': dict(engine='bing', cc='US', mkt='en-US')}
TRACKING = ('msockid', 'srsltid', 'gclid', 'fbclid')


def norm(u):
    from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
    p = urlparse(clean_url(u))
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in TRACKING]
    return urlunparse(p._replace(query=urlencode(q)))


def sample(engine, n, dataset):
    os.environ['AEO_DATASET'] = dataset
    from x03_analyse import records
    recs = [j for j in records(engine) if j['queries'] and j['consulted'] and j['date'] <= SAMPLE_MAX_DATE]
    if engine == 'openai':
        recs = [j for j in recs if j.get('calls')]
    recs = sorted(recs, key=lambda j: (j['prompt_id'], j['date'], j['rep']))
    seen, pool = set(), []
    for j in recs:                                   # one answer per prompt
        if j['prompt_id'] not in seen:
            seen.add(j['prompt_id']); pool.append(j)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    out = []
    for i in sorted(pick):
        j = pool[i]
        out.append(dict(dataset=dataset, engine=engine, prompt_id=j['prompt_id'], queries=j['queries'],
                        retrieved=list(dict.fromkeys(norm(u) for u in j['consulted'])),
                        cited=list(dict.fromkeys(norm(u) for u in j['cited']))))
    return out


def path(provider, q, start):
    return OUT / provider / f"{hashlib.sha1(q.lower().encode()).hexdigest()[:16]}_{start}.json"


def fetch_page(provider, q, start, k):
    f = path(provider, q, start)
    if f.exists():
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    p = dict(PARAMS[provider], q=q, api_key=k)
    if start:
        p['start' if provider == 'google' else 'first'] = start if provider == 'google' else start + 1
    for attempt in range(3):
        try:
            r = requests.get(API, params=p, timeout=120)
            if r.status_code == 200:
                j = r.json()
                f.write_text(json.dumps(dict(query=q, start=start, organic=[
                    dict(position=o.get('position'), link=o.get('link')) for o in j.get('organic_results') or []])),
                    encoding='utf-8')
                return
            print(f'  http {r.status_code}', flush=True)
        except requests.RequestException as e:
            print(f'  {type(e).__name__}', flush=True)
        time.sleep(5 * (attempt + 1))


def fetch(n):
    k = key()
    ds = os.environ.get('AEO_DATASET', 'gfiber')
    jobs = []
    for engine, nn in (('openai', n), ('gemini', max(1, int(n * 0.8)))):
        for x in sample(engine, nn, ds):
            for (e, prov), depth in DEPTH.items():
                if e == engine:
                    jobs += [(prov, q, s) for q in x['queries'] for s in range(0, depth, 10)]
    jobs = list(dict.fromkeys(jobs))
    todo = [j for j in jobs if not path(*j).exists()]
    left, _ = searches_left(k)
    print(f'{ds}: {len(todo)} searches needed, {left} left', flush=True)
    if left is not None and left < len(todo):
        raise SystemExit('not enough searches left')
    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(lambda j: fetch_page(*j, k), todo), 1):
            if i % 100 == 0:
                print(f'  {i}/{len(todo)}', flush=True)


def positions(provider, queries, depth):
    """url -> best position over the queries (position = start + rank on the page)."""
    best = {}
    for q in queries:
        for s in range(0, depth, 10):
            f = path(provider, q, s)
            if not f.exists():
                continue
            org = json.loads(f.read_text(encoding='utf-8'))['organic']
            for i, o in enumerate(o for o in org if o.get('link')):
                u = norm(o['link'])
                best[u] = min(best.get(u, 999), s + i + 1)
    return best


def band(p):
    return 'top 10' if p <= 10 else '11-30' if p <= 30 else '31-50' if p <= 50 else 'not found'


def analyse(n):
    # one dataset per process: x03_analyse/common read their paths at import time
    rows = []
    ds = os.environ.get('AEO_DATASET', 'gfiber')
    for ds, nn in ((ds, n),):
        for engine, m in (('openai', nn), ('gemini', max(1, int(nn * 0.8)))):
            for x in sample(engine, m, ds):
                for (e, prov), depth in DEPTH.items():
                    if e != engine:
                        continue
                    pos = positions(prov, x['queries'], depth)
                    dpos = {}
                    for u, p in pos.items():
                        dpos[domain_of(u)] = min(dpos.get(domain_of(u), 999), p)
                    kinds = [('retrieved', x['retrieved'])] if engine == 'openai' else []
                    kinds.append(('cited', x['cited']))
                    for kind, urls in kinds:
                        for u in urls:
                            rows.append(dict(dataset=ds, engine=engine, provider=prov, depth=depth, prompt_id=x['prompt_id'],
                                             kind=kind, url=u, pos=pos.get(u, 999), site_pos=dpos.get(domain_of(u), 999)))
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / 'x08_deep_serp_rows.csv', index=False)
    d['page'] = d.pos.map(band)
    d['site'] = d.site_pos.map(band)
    out = []
    for key_, g in d.groupby(['engine', 'provider', 'kind', 'dataset']):
        r = dict(zip(['engine', 'provider', 'kind', 'dataset'], key_), pages=len(g), answers=g.prompt_id.nunique(),
                 depth=int(g.depth.iloc[0]))
        for b in ('top 10', '11-30', '31-50', 'not found'):
            r[f'page_{b}'] = 100 * (g.page == b).mean()
            r[f'site_{b}'] = 100 * (g.site == b).mean()
        out.append(r)
    S = pd.DataFrame(out)
    S.to_csv(RESULTS / 'x08_deep_serp.csv', index=False)
    pd.set_option('display.width', 250)
    print(S.round(1).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fetch', 'analyse'])
    ap.add_argument('--n', type=int, default=50)
    a = ap.parse_args()
    fetch(a.n) if a.cmd == 'fetch' else analyse(a.n)
