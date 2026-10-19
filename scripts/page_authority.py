"""Page-level authority from the DataForSEO Backlinks API (bulk endpoints, up to 1,000 pages per request).

Pages: every URL in url_table.csv (cited >= 2 times) and every page ChatGPT consulted in the replays
(x06_rows.csv), for the current dataset (AEO_DATASET).
  rank                DataForSEO Rank of the page (0-1,000, based on its backlink profile)
  referring_domains   distinct websites linking to the page
  backlinks           links to the page
Our URLs are normalised (no www., no trailing slash); DataForSEO knows a page under its exact address
(test: https://att.com/internet/fiber 2 referring domains, https://www.att.com/internet/fiber/ 478). Each
page is therefore sent in up to four forms - the address the crawler actually fetched (final_url), and
the normalised address with and without www. and with and without a trailing slash - and the form with
the most referring domains is kept (ties: higher rank).
Cache: data[/<ds>]/tables/page_authority_raw.jsonl (one line per target and endpoint; resumable).
Output: data[/<ds>]/tables/page_authority.csv (url, pa_rank, pa_referring_domains, pa_backlinks, pa_form)

  python scripts/page_authority.py fetch        needs DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD in .env
  python scripts/page_authority.py table
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import pandas as pd
import requests

from common import TABLES, DATASET

API = 'https://api.dataforseo.com/v3/backlinks/{}/live'
ENDPOINTS = {'bulk_referring_domains': 'referring_domains', 'bulk_ranks': 'rank'}   # backlink counts skipped (cost)
RAW = TABLES / 'page_authority_raw.jsonl'
BATCH = 1000


def pages():
    urls = set(pd.read_csv(TABLES / 'url_table.csv', usecols=['url']).url)
    x6 = TABLES / 'x06_rows.csv'
    if x6.exists():
        urls |= set(pd.read_csv(x6, usecols=['url']).url)
    return sorted(u for u in urls if isinstance(u, str) and u.startswith('http'))


def forms(urls):
    """url -> list of target forms (exact fetched address first, then normalised)."""
    final = {}
    for f in ('pages_index.csv', 'x06_pages_index.csv', 'n07_pages_index.csv'):
        p = TABLES / f
        if p.exists():
            x = pd.read_csv(p, dtype=str, usecols=['url', 'status', 'final_url'])
            for r in x[(x.status == 'ok') & x.final_url.notna()].itertuples():
                final.setdefault(r.url, r.final_url)
    out = {}
    for u in urls:
        f = []
        g = final.get(u)
        if g and urlparse(g).netloc.removeprefix('www.') == urlparse(u).netloc.removeprefix('www.'):
            f.append(g)                          # the address as the site actually serves it
        p = urlparse(u)
        hosts = [p.netloc] + ([] if p.netloc.startswith('www.') or p.netloc.count('.') > 1 else ['www.' + p.netloc])
        paths = [p.path] + ([p.path + '/'] if p.path and not p.path.endswith('/') and not p.query else [])
        if not p.path:
            paths = ['/']
        for h in hosts:
            for pa in paths:
                f.append(p._replace(netloc=h, path=pa).geturl())
        out[u] = list(dict.fromkeys(f))[:4]
    return out


def done_targets():
    got = set()
    if RAW.exists():
        with open(RAW, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line); got.add((r['endpoint'], r['target']))
    return got


def call(endpoint, targets, auth):
    import time
    body = [{'targets': targets, 'rank_scale': 'one_thousand'}] if endpoint == 'bulk_ranks' else [{'targets': targets}]
    for attempt in range(8):                     # 40104 'verify your account' came back intermittently
        try:                                     # right after activation, and 5xx: retry with backoff
            j = requests.post(API.format(endpoint), auth=auth, json=body, timeout=300).json()
        except (requests.RequestException, ValueError):
            j = {'status_code': 50000, 'status_message': 'request failed'}
        if j.get('status_code') == 20000:
            break
        print(f"  {endpoint}: {j.get('status_code')} {str(j.get('status_message'))[:60]} - retry {attempt + 1}", flush=True)
        time.sleep(5 * (attempt + 1))
    if j.get('status_code') != 20000:
        raise SystemExit(f"{endpoint}: {j.get('status_code')} {j.get('status_message')}")
    t = j['tasks'][0]
    if t['status_code'] == 40501 and len(targets) > 1:          # an address DataForSEO rejects: bisect
        h = len(targets) // 2
        a, ca = call(endpoint, targets[:h], auth)
        b, cb = call(endpoint, targets[h:], auth)
        return a + b, ca + cb
    if t['status_code'] == 40501:
        print(f'  rejected target: {targets[0][:120]}', flush=True)
        return [], 0.0
    if t['status_code'] != 20000:
        raise SystemExit(f"{endpoint} task: {t['status_code']} {t['status_message']}")
    items = (t.get('result') or [{}])[0].get('items') or []
    return items, t.get('cost', 0)


def fetch():
    auth = (os.environ['DATAFORSEO_LOGIN'], os.environ['DATAFORSEO_PASSWORD'])
    fm = forms(pages())
    targets = sorted({t for f in fm.values() for t in f})
    got = done_targets()
    total_cost = 0.0
    for ep, field in ENDPOINTS.items():
        todo = [t for t in targets if (ep, t) not in got]
        batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
        print(f'{DATASET} {ep}: {len(todo)} targets in {len(batches)} requests', flush=True)

        def one(b):
            return call(ep, b, auth)

        with ThreadPoolExecutor(1) as ex, open(RAW, 'a', encoding='utf-8') as out:
            for b, (items, cost) in zip(batches, ex.map(one, batches)):
                total_cost += cost or 0
                seen = set()
                for it in items:
                    out.write(json.dumps(dict(endpoint=ep, target=it.get('target'), value=it.get(field),
                                              extra={k: it.get(k) for k in ('referring_main_domains', 'referring_ips')
                                                     if k in it})) + '\n')
                    seen.add(it.get('target'))
                for t in set(b) - seen:          # not in DataForSEO's index
                    out.write(json.dumps(dict(endpoint=ep, target=t, value=None)) + '\n')
                out.flush()
    print(f'cost of this run: ${total_cost:.2f}')
    table()


def table():
    raw = {}
    with open(RAW, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            raw[(r['endpoint'], r['target'])] = r['value']
    fm = forms(pages())
    rows = []
    for u, fs in fm.items():
        best = None
        for t in fs:
            v = (raw.get(('bulk_referring_domains', t)) or 0, raw.get(('bulk_ranks', t)) or 0)
            if best is None or v > best[0]:
                best = (v, t)
        t = best[1]
        rows.append(dict(url=u, pa_rank=raw.get(('bulk_ranks', t)), pa_referring_domains=raw.get(('bulk_referring_domains', t)),
                         pa_form='normalised' if t == u else 'other form'))
    d = pd.DataFrame(rows)
    d.to_csv(TABLES / 'page_authority.csv', index=False)
    print(f'{len(d)} pages; with any referring domain: {100 * (d.pa_referring_domains.fillna(0) > 0).mean():.0f} %; '
          f'median rank {d.pa_rank.median()}; another address form used for {100 * (d.pa_form != "normalised").mean():.0f} %')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fetch', 'table'])
    fetch() if ap.parse_args().cmd == 'fetch' else table()
