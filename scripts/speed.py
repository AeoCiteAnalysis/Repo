"""Step A6 (optional, needed for R8) - page speed.

  python scripts/speed.py crux [--limit N]     real-user speed, Chrome UX Report API (fast)
  python scripts/speed.py psi  [--limit N]     Lighthouse performance score, PageSpeed Insights (slow)

Both need PSI_API_KEY in .env: one Google API key with "Chrome UX Report API" and
"PageSpeed Insights API" enabled.

crux  75th-percentile LCP (ms), INP (ms), CLS for the exact URL (mobile, last 28 days). Only URLs
      with enough Chrome traffic have data; for the rest the website's ORIGIN is queried and
      flagged `crux_level = origin`. ~0.1 s per call, limit 150 calls / minute.
psi   Lighthouse mobile performance score 0-1 (runs a lab test: 10-30 s per URL); use for a subset.

Writes data/tables/page_speed.csv (url, lcp, inp, cls, crux_level, perf_score); resumes where it
stopped. build_url_table.py joins it on url; R8 combines the speed metrics into one score.
Default URL list: URLs cited >= 2 times that have page features (the R2 rows).
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import pandas as pd
import requests

from common import TABLES, read_table

CRUX = 'https://chromeuxreport.googleapis.com/v1/records:queryRecord'
PSI = 'https://www.googleapis.com/pagespeedonline/v5/runPagespeed'
OUT = TABLES / 'page_speed.csv'
COLS = ['url', 'lcp', 'inp', 'cls', 'crux_level', 'perf_score', 'error']


def key():
    k = os.environ.get('PSI_API_KEY')
    if not k:
        raise SystemExit('set PSI_API_KEY in .env (see .env.example)')
    return k


def crux_query(body, k):
    for attempt in range(4):
        r = requests.post(CRUX, headers={'x-goog-api-key': k}, json=dict(body, formFactor='PHONE'), timeout=30)
        if r.status_code == 429:                     # over 150/min: back off
            time.sleep(15 * (attempt + 1))
            continue
        return r
    return r


def crux_one(url, k, origin_cache):
    out = dict(url=url, crux_level='', error='')
    r = crux_query({'url': url}, k)
    level = 'url'
    if r.status_code == 404:                         # not enough traffic for this URL
        p = urlparse(url)
        origin = f'{p.scheme}://www.{p.netloc}' if not p.netloc.startswith('www.') else f'{p.scheme}://{p.netloc}'
        if origin not in origin_cache:
            rr = crux_query({'origin': origin}, k)
            if rr.status_code == 404:
                rr = crux_query({'origin': f'{p.scheme}://{p.netloc}'}, k)
            origin_cache[origin] = rr
        r, level = origin_cache[origin], 'origin'
    if r.status_code != 200:
        out['error'] = f'http_{r.status_code}'
        return out
    m = r.json().get('record', {}).get('metrics', {})
    p75 = lambda name: (m.get(name, {}).get('percentiles') or {}).get('p75')
    out.update(lcp=p75('largest_contentful_paint'), inp=p75('interaction_to_next_paint'),
               cls=p75('cumulative_layout_shift'), crux_level=level)
    return out


def psi_one(url, k):
    try:
        r = requests.get(PSI, params={'url': url, 'key': k, 'strategy': 'mobile',
                                      'category': 'performance'}, timeout=120)
        if r.status_code != 200:
            return dict(url=url, error=f'psi_http_{r.status_code}')
        score = r.json().get('lighthouseResult', {}).get('categories', {}).get('performance', {}).get('score')
        return dict(url=url, perf_score=score, error='')
    except requests.RequestException as e:
        return dict(url=url, error=f'psi_{type(e).__name__}')


def target_urls(limit):
    t = read_table('url_table.csv', usecols=['url', 'citation_count', 'word_count'])
    t = t[t.word_count.notna()].sort_values('citation_count', ascending=False)
    return t.url.tolist()[:limit] if limit else t.url.tolist()


def save(done):
    done = done.drop_duplicates('url', keep='last')
    for c in ('lcp', 'inp', 'cls', 'perf_score'):     # CrUX returns CLS as text
        if c in done:
            done[c] = pd.to_numeric(done[c], errors='coerce')
    for c in COLS:
        if c not in done:
            done[c] = None
    done[COLS].to_csv(OUT, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', choices=['crux', 'psi'])
    ap.add_argument('--limit', type=int)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    k = key()
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=COLS)
    urls = target_urls(a.limit)
    have = done if a.source == 'crux' else done[done.perf_score.notna()]
    todo = [u for u in urls if u not in set(have.url)]
    print(f'{a.source}: {len(todo)} URLs to measure')
    origin_cache = {}
    for i in range(0, len(todo), 200):
        batch = todo[i:i + 200]
        if a.source == 'crux':
            with ThreadPoolExecutor(a.workers) as ex:
                rows = list(ex.map(lambda u: crux_one(u, k, origin_cache), batch))
        else:
            with ThreadPoolExecutor(max(a.workers, 8)) as ex:
                rows = list(ex.map(lambda u: psi_one(u, k), batch))
        new = pd.DataFrame(rows).set_index('url')
        done = done.set_index('url')
        # crux: new values replace old ones; psi: only fill the missing perf_score
        done = new.combine_first(done) if a.source == 'crux' else done.combine_first(new)
        done = done.reset_index()
        save(done)
        print(f'  {min(i + 200, len(todo))}/{len(todo)}')


if __name__ == '__main__':
    main()
