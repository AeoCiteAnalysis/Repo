"""X1c - Google AI Overview sources vs the engines' sources.

X1 / X1b: Google's organic top 10 contains only ~5 % of the websites the engines cite. In the 7 X1
responses that included Google's AI Overview source list, 27-35 % of the engines' cited websites were
among them - AI answer engines may share sources with each other more than with classic search.

SerpAPI returns the AI Overview sources either inline or behind a page_token that expires quickly,
so each prompt needs a fresh Google search (location "United States") followed at once by an
engine=google_ai_overview call: up to 2 searches per prompt. Responses are cached in
data/serp/ai_overview/<prompt_id>.json.

  python scripts/x01c_ai_overview.py fetch --n 3        test / fetch N prompts (random, seed 42)
  python scripts/x01c_ai_overview.py analyse

Outputs: data/tables/ai_overview_refs.csv, results/x01c_ai_overview.csv
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import requests

from common import ROOT, TABLES, RESULTS, SEED, ENGINES, ENGINE_LABEL, clean_url, domain_of, read_table, drop_excluded
from x01_search_rank import API, key, searches_left

OUT = ROOT / 'data' / 'serp' / 'ai_overview'


def refs_of(ao):
    return [r.get('link') for r in (ao.get('references') or []) if r.get('link')]


def fetch_one(prompt, k):
    """Fresh search, then the AI Overview call if the sources are behind a token."""
    s = requests.get(API, params=dict(engine='google', q=prompt, gl='us', hl='en', location='United States',
                                      google_domain='google.com', api_key=k), timeout=90).json()
    ao = s.get('ai_overview') or {}
    out = dict(query=prompt, has_ai_overview=bool(ao), searches=1, source='none', references=refs_of(ao))
    if out['references']:
        out['source'] = 'inline'
    elif ao.get('page_token'):
        a = requests.get(API, params=dict(engine='google_ai_overview', page_token=ao['page_token'], api_key=k),
                         timeout=90).json()
        out['searches'] = 2
        out['references'] = refs_of(a.get('ai_overview') or {})
        out['source'] = 'page_token' if out['references'] else f'page_token_empty: {a.get("error", "")}'
    return out


def fetch(n):
    k = key()
    OUT.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(TABLES / 'prompt_labels.csv', usecols=['prompt_id', 'prompt'])
    pick = lab.sample(len(lab), random_state=SEED).head(n)
    todo = [r for r in pick.itertuples() if not (OUT / f'{r.prompt_id}.json').exists()]
    left, _ = searches_left(k)
    print(f'{len(todo)} prompts to fetch (up to {2 * len(todo)} searches); {left} searches left')
    if left is not None and left < 2 * len(todo):
        raise SystemExit('not enough searches left')
    for i, r in enumerate(todo, 1):
        try:
            res = fetch_one(r.prompt, k)
        except requests.RequestException as e:
            print(f'  {r.prompt_id}: {type(e).__name__}'); continue
        (OUT / f'{r.prompt_id}.json').write_text(json.dumps(res), encoding='utf-8')
        print(f'  {r.prompt_id}: AI Overview={res["has_ai_overview"]}, sources={len(res["references"])} ({res["source"]})')
        time.sleep(1)


def analyse():
    rows = []
    for f in sorted(OUT.glob('*.json')):
        j = json.loads(f.read_text(encoding='utf-8'))
        for link in j['references']:
            u = clean_url(link)
            rows.append(dict(prompt_id=f.stem, url=u, domain=domain_of(u)))
    a = pd.DataFrame(rows)
    a.to_csv(TABLES / 'ai_overview_refs.csv', index=False)
    fetched = [f.stem for f in OUT.glob('*.json')]
    print(f'{len(fetched)} prompts fetched, {a.prompt_id.nunique() if len(a) else 0} with sources, {len(a)} sources')
    if a.empty:
        return
    c = drop_excluded(read_table('citations.csv', usecols=['date', 'engine', 'prompt_id', 'url', 'domain',
                                                          'is_redirect', 'repeat_in_answer']))
    c = c[~c.repeat_in_answer & c.prompt_id.isin(a.prompt_id)]
    last = sorted(c.date.unique())[-14:]
    ru, rd = a.groupby('prompt_id').url.agg(set).to_dict(), a.groupby('prompt_id').domain.agg(set).to_dict()
    out = []
    for period, cc in (('all 90 days', c), ('last 14 days', c[c.date.isin(last)])):
        for e in ENGINES + ['all']:
            g = cc if e == 'all' else cc[cc.engine == e]
            u = g[~g.is_redirect].drop_duplicates(['prompt_id', 'url'])
            d = g.drop_duplicates(['prompt_id', 'domain'])
            if len(d):
                out.append(dict(period=period, engine=ENGINE_LABEL.get(e, e), prompts=a.prompt_id.nunique(),
                                pct_cited_urls_in_ai_overview=100 * np.mean([x in ru.get(p, ()) for p, x in zip(u.prompt_id, u.url)]),
                                pct_cited_websites_in_ai_overview=100 * np.mean([x in rd.get(p, ()) for p, x in zip(d.prompt_id, d.domain)])))
    cited_any = set(zip(c.prompt_id, c.domain))
    res = pd.DataFrame(out)
    res['pct_ai_overview_websites_cited_by_engines'] = 100 * np.mean([(p, x) in cited_any for p, x in zip(a.prompt_id, a.domain)])
    res.to_csv(RESULTS / 'x01c_ai_overview.csv', index=False)
    print(res.round(1).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fetch', 'analyse'])
    ap.add_argument('--n', type=int, default=3)
    a = ap.parse_args()
    fetch(a.n) if a.cmd == 'fetch' else analyse()


if __name__ == '__main__':
    main()
