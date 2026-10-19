"""X1b - Pilot: do engine-style search queries match the engines' sources better?

X1 found that Google's top 10 for the literal question text contains almost none of the pages the
engines cite (0.1-0.8 % of URLs). Two explanations: (1) the literal search was not set to a national
location, so Google returned local pages; (2) the engines rewrite the question into their own search
queries. This pilot separates the two on 50 random prompts (seed 42), comparing three searches:
  literal          the question as typed, gl=us, no location        (X1, already cached)
  literal_us       the question as typed, location "United States"   (50 searches)
  rewrite_us       an LLM-written search query, location "United States" (50 searches)
The LLM (llm.py, cached) is asked to write the single short query an AI assistant's search step
would send; queries are saved to data/tables/search_queries.csv.
Measure: share of each engine's cited (prompt, URL) and (prompt, website) pairs that are in the
top 10, over all 90 days and the last 14 days; and how often a top-10 result is cited for its prompt.

  python scripts/x01b_rewrite_pilot.py [--n 50]

Outputs: data/serp/google_literal_us/, data/serp/google_rewrite_us/, data/tables/search_queries.csv,
         results/x01b_pilot.csv
"""
import argparse
import json
import os
import time
from datetime import date

import numpy as np
import pandas as pd
import requests

from common import ROOT, TABLES, RESULTS, SEED, ENGINES, ENGINE_LABEL, clean_url, domain_of, read_table, drop_excluded
from x01_search_rank import API, key, searches_left

SERP = ROOT / 'data' / 'serp'
BASE = dict(engine='google', gl='us', hl='en', num=10, google_domain='google.com')
INSTRUCTIONS = (
    'You are the search step of an AI assistant that answers questions about home internet service '
    'in the United States. Given the user\'s question, write the single web search query you would send '
    'to Google to find pages to cite in the answer. Use 4 to 10 words. Keep brand and product names. '
    'Drop filler words and politeness. Add context only when the question implies it (for example '
    '"fiber internet"). Add the year 2026 only for questions about prices, plans, deals or promotions. '
    'Do not add a city or state, and do not stack synonyms. Output only the query.')


def rewrite(prompts):
    import llm
    qs = llm.ask_many(INSTRUCTIONS, prompts)
    return [q.strip().strip('"').strip() for q in qs]


def fetch(variant, rows, k):
    out = SERP / variant
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        f = out / f'{r["prompt_id"]}.json'
        if f.exists():
            continue
        params = dict(BASE, q=r['query'], location='United States', api_key=k)
        for attempt in range(3):
            try:
                resp = requests.get(API, params=params, timeout=90)
                if resp.status_code == 200:
                    j = resp.json()
                    j['_snapshot_date'], j['_query'] = f'{date.today():%Y-%m-%d}', r['query']
                    f.write_text(json.dumps(j), encoding='utf-8')
                    break
                print(f'  {r["prompt_id"]}: http {resp.status_code}')
            except requests.RequestException as e:
                print(f'  {r["prompt_id"]}: {type(e).__name__}')
            time.sleep(5 * (attempt + 1))


def top10(variant, pids):
    rows = []
    for p in pids:
        f = SERP / variant / f'{p}.json'
        if not f.exists():
            continue
        for o in json.loads(f.read_text(encoding='utf-8')).get('organic_results') or []:
            if o.get('link') and (o.get('position') or 99) <= 10:
                u = clean_url(o['link'])
                rows.append(dict(prompt_id=p, rank=o['position'], url=u, domain=domain_of(u)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    a = ap.parse_args()
    lab = pd.read_csv(TABLES / 'prompt_labels.csv', usecols=['prompt_id', 'prompt', 'topic'])
    pilot = lab.sample(a.n, random_state=SEED).reset_index(drop=True)
    qpath = TABLES / 'search_queries.csv'
    old = pd.read_csv(qpath) if qpath.exists() else pd.DataFrame(columns=['prompt_id', 'query'])
    need = pilot[~pilot.prompt_id.isin(old.prompt_id)]
    if len(need):
        new = pd.DataFrame({'prompt_id': need.prompt_id, 'query': rewrite(need.prompt.tolist())})
        old = pd.concat([old, new], ignore_index=True)
        old.to_csv(qpath, index=False)
    pilot = pilot.merge(old, on='prompt_id')

    k = key()
    todo = sum(not (SERP / v / f'{p}.json').exists() for v in ('google_literal_us', 'google_rewrite_us')
               for p in pilot.prompt_id)
    left, _ = searches_left(k)
    print(f'pilot: {len(pilot)} prompts, {todo} searches needed, {left} left')
    if left is not None and left < todo:
        raise SystemExit('not enough searches left')
    fetch('google_literal_us', [dict(prompt_id=r.prompt_id, query=r.prompt) for r in pilot.itertuples()], k)
    fetch('google_rewrite_us', [dict(prompt_id=r.prompt_id, query=r.query) for r in pilot.itertuples()], k)

    c = drop_excluded(read_table('citations.csv', usecols=['date', 'engine', 'prompt_id', 'url', 'domain',
                                                          'is_redirect', 'repeat_in_answer']))
    c = c[~c.repeat_in_answer & c.prompt_id.isin(pilot.prompt_id)]
    last = sorted(c.date.unique())[-14:]
    rows = []
    for variant in ('google', 'google_literal_us', 'google_rewrite_us'):
        s = top10(variant, pilot.prompt_id)
        su = s.groupby('prompt_id').url.agg(set).to_dict()
        sd = s.groupby('prompt_id').domain.agg(set).to_dict()
        cited_any = set(zip(c.prompt_id, c.url))
        rank1 = s[s['rank'] == 1]
        for period, cc in (('all 90 days', c), ('last 14 days', c[c.date.isin(last)])):
            for e in ENGINES + ['all']:
                g = cc if e == 'all' else cc[cc.engine == e]
                u = g[~g.is_redirect].drop_duplicates(['prompt_id', 'url'])
                d = g.drop_duplicates(['prompt_id', 'domain'])
                rows.append(dict(variant=variant, period=period, engine=ENGINE_LABEL.get(e, e),
                                 pct_cited_urls_in_top10=100 * np.mean([x in su.get(p, ()) for p, x in zip(u.prompt_id, u.url)]) if len(u) else np.nan,
                                 pct_cited_websites_in_top10=100 * np.mean([x in sd.get(p, ()) for p, x in zip(d.prompt_id, d.domain)]) if len(d) else np.nan,
                                 pct_top10_results_cited=100 * np.mean([(p, x) in cited_any for p, x in zip(s.prompt_id, s.url)]),
                                 pct_rank1_cited=100 * np.mean([(p, x) in cited_any for p, x in zip(rank1.prompt_id, rank1.url)]),
                                 prompts=s.prompt_id.nunique()))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'x01b_pilot.csv', index=False)
    show = out[out.period == 'all 90 days'].pivot(index='engine', columns='variant',
                                                  values=['pct_cited_urls_in_top10', 'pct_cited_websites_in_top10'])
    print(show.round(1).to_string())
    print(out[(out.engine == 'all') & (out.period == 'all 90 days')][['variant', 'pct_top10_results_cited', 'pct_rank1_cited']].round(1).to_string(index=False))
    print(out[out.period == 'last 14 days'].pivot(index='engine', columns='variant', values='pct_cited_websites_in_top10').round(1).to_string())
    print('\nexample queries:')
    print(pilot[['prompt', 'query']].head(8).to_string(index=False))


if __name__ == '__main__':
    main()
