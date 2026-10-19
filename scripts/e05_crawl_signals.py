"""E5 - Are pages that crawlers visit often cited more?

Search indexes revisit popular and often-linked pages more often. The Internet Archive's capture
history is a free, page-level proxy for that: for a random 3,000 of the R2 pages (seed 42), the
Wayback CDX API gives
  wb_captures   number of captures (collapsed to one per day), log2(1 + n)
  wb_months     number of distinct months with a capture in the last 24 months
  wb_first_age  days since the first capture (how long the URL has existed), log2
  wb_archived   URL has any capture at all
Added to the R2 page variables; within-website effects per SD (fe_ols) and the share of the
within-website variation explained (mixed model), for total, breadth and depth.

  python scripts/e05_crawl_signals.py fetch     (cached in data/wayback/)
  python scripts/e05_crawl_signals.py analyse
Outputs: data/tables/wayback.csv, results/e05_crawl_signals.csv, results/e05_crawl_effects.csv
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests

from common import ROOT, TABLES, RESULTS, SEED, read_table, drop_excluded

OUT = ROOT / 'data' / 'wayback'
N = 3000
CDX = 'https://web.archive.org/cdx/search/cdx'
SIGNALS = ['wb_captures', 'wb_months', 'wb_first_age', 'wb_archived']


def sample():
    t = read_table('url_table.csv', usecols=['url', 'word_count'])
    return t[t.word_count.notna()].url.sample(N, random_state=SEED).tolist()


def path(u):
    return OUT / (hashlib.sha1(u.encode()).hexdigest()[:16] + '.json')


def get(u):
    f = path(u)
    if f.exists():
        return
    bare = u.split('://', 1)[-1]
    for attempt in range(4):
        try:
            r = requests.get(CDX, params=dict(url=bare, output='json', fl='timestamp', collapse='timestamp:8',
                                              filter='statuscode:200', limit=5000), timeout=60)
            if r.status_code == 200:
                rows = r.json() if r.text.strip() else []
                f.write_text(json.dumps(dict(url=u, ts=[x[0] for x in rows[1:]])), encoding='utf-8')
                return
            if r.status_code in (429, 503):
                time.sleep(20 * (attempt + 1))
                continue
            f.write_text(json.dumps(dict(url=u, ts=None, status=r.status_code)), encoding='utf-8')
            return
        except (requests.RequestException, ValueError):
            time.sleep(10 * (attempt + 1))


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [u for u in sample() if not path(u).exists()]
    print(f'{len(todo)} URLs to query')
    with ThreadPoolExecutor(3) as ex:
        for i, _ in enumerate(ex.map(get, todo), 1):
            if i % 200 == 0:
                print(f'  {i}/{len(todo)}', flush=True)


def table():
    rows, today = [], date(2026, 9, 21)
    for u in sample():
        f = path(u)
        if not f.exists():
            continue
        j = json.loads(f.read_text(encoding='utf-8'))
        if j['ts'] is None:
            continue
        ts = [datetime.strptime(t[:8], '%Y%m%d').date() for t in j['ts']]
        recent = {(t.year, t.month) for t in ts if (today - t).days <= 730}
        rows.append(dict(url=u, wb_captures=np.log2(1 + len(ts)), wb_months=len(recent),
                         wb_first_age=np.log2(1 + (today - min(ts)).days) if ts else 0.0,
                         wb_archived=float(bool(ts))))
    w = pd.DataFrame(rows)
    w.to_csv(TABLES / 'wayback.csv', index=False)
    return w


def analyse():
    from fe import fe_ols
    from r02_main_model import prepare, FEATURES
    from e01_engines_breadth_depth import outcomes, vc
    w = table()
    print(f'{len(w)} URLs with a Wayback answer; {100 * w.wb_archived.mean():.0f} % archived')
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    X = FEATURES + flags + types
    d = d.merge(w, on='url')
    for s in SIGNALS:
        d[s] = (d[s] - d[s].mean()) / d[s].std()
    c = drop_excluded(read_table('citations.csv', usecols=['date', 'engine', 'prompt_id', 'url', 'is_redirect',
                                                          'repeat_in_answer']))
    full = outcomes(c[~c.is_redirect & ~c.repeat_in_answer & c.url.isin(d.url)])
    d = d[d.url.isin(full.index)]
    d = d[d.groupby('domain').url.transform('size') >= 2].reset_index(drop=True)
    rows, eff = [], []
    for y in ('total', 'breadth', 'depth'):
        d[y] = full[y].reindex(d.url).to_numpy()
        for name, cols in (('page variables', X), ('page variables + crawl signals', X + SIGNALS)):
            ws, ex = vc(d, y, cols)
            rows.append(dict(outcome=y, model=name, n_urls=len(d), n_websites=d.domain.nunique(),
                             within_share=ws, explained_within=ex))
        r, _ = fe_ols(d, y, X + SIGNALS, 'domain')
        for s in SIGNALS:
            if s in r.index:
                eff.append(dict(outcome=y, feature=s, coef=r.loc[s, 'coef'], se=r.loc[s, 'se'], p=r.loc[s, 'p'],
                                pct_change_per_sd=100 * (2 ** r.loc[s, 'coef'] - 1)))
    out, eff = pd.DataFrame(rows), pd.DataFrame(eff)
    out.to_csv(RESULTS / 'e05_crawl_signals.csv', index=False)
    eff.to_csv(RESULTS / 'e05_crawl_effects.csv', index=False)
    print(out.round(3).to_string(index=False))
    print(eff.round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fetch', 'analyse'])
    fetch() if ap.parse_args().cmd == 'fetch' else analyse()
