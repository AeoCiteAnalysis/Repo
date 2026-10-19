"""E4 - Do websites that block an engine's crawler get cited less by that engine?

Each engine fetches pages with its own crawler, and a website's robots.txt can shut any of them
out. For every website cited at least once (url_table_all), robots.txt is downloaded and each
cited URL is checked against these user agents:
  OAI-SearchBot   OpenAI's search index (ChatGPT search results)
  ChatGPT-User    OpenAI fetches made while answering
  GPTBot          OpenAI model training
  Googlebot       Google Search (Gemini is grounded on Google Search)
  Google-Extended Google's AI-use control (Google says it does not affect Search)
  PerplexityBot   Perplexity's index
Also: pages whose downloaded HTML has almost no text (< 100 words without running JavaScript),
which a crawler that does not render JavaScript cannot read.

Measures: (1) share of cited URLs blocked for each agent, per engine that cited them - an engine
citing pages its crawler may not fetch shows it does not depend on that crawler;
(2) per website, each engine's share of the website's citations, blocked vs not blocked;
(3) in the R2 page table, the within-website effect of 'no text without JavaScript' per engine.

  python scripts/e04_ai_crawlers.py fetch      download robots.txt (cached in data/robots/)
  python scripts/e04_ai_crawlers.py analyse
Outputs: data/tables/robots_access.csv, results/e04_ai_crawlers.csv, results/e04_js_text.csv
"""
import argparse
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib import robotparser
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from common import ROOT, TABLES, RESULTS, ENGINES, ENGINE_LABEL

ROBOTS = ROOT / 'data' / 'robots'
AGENTS = ['OAI-SearchBot', 'ChatGPT-User', 'GPTBot', 'Googlebot', 'Google-Extended', 'PerplexityBot']
ENGINE_AGENT = {'OPENAI': 'OAI-SearchBot', 'GEMINI': 'Googlebot', 'PERPLEXITY': 'PerplexityBot'}
UA = 'Mozilla/5.0 (compatible; AeoCiteAnalysis research; robots.txt check)'


def path_of(domain):
    return ROBOTS / (hashlib.sha1(domain.encode()).hexdigest()[:16] + '.txt')


def get(domain):
    f = path_of(domain)
    if f.exists():
        return
    status, text = 'error', ''
    for scheme in ('https', 'http'):
        try:
            r = requests.get(f'{scheme}://{domain}/robots.txt', headers={'User-Agent': UA}, timeout=15)
            status, text = str(r.status_code), r.text if r.status_code == 200 else ''
            break
        except requests.RequestException as e:
            status = type(e).__name__
    f.write_text(f'#STATUS {status}\n{text}', encoding='utf-8')


def fetch():
    ROBOTS.mkdir(parents=True, exist_ok=True)
    doms = pd.read_csv(TABLES / 'url_table_all.csv', usecols=['domain']).domain.dropna().unique()
    todo = [d for d in doms if not path_of(d).exists()]
    print(f'{len(todo)} websites to fetch')
    with ThreadPoolExecutor(24) as ex:
        for i, _ in enumerate(ex.map(get, todo), 1):
            if i % 500 == 0:
                print(f'  {i}/{len(todo)}'); sys.stdout.flush()


def access_table():
    u = pd.read_csv(TABLES / 'url_table_all.csv',
                    usecols=['url', 'domain', 'citation_count'] + [f'count_{e}' for e in ENGINES])
    rows = []
    for dom, g in u.dropna(subset=['domain']).groupby('domain'):
        raw = path_of(dom).read_text(encoding='utf-8') if path_of(dom).exists() else '#STATUS missing\n'
        status, text = raw.split('\n', 1)
        status = status.removeprefix('#STATUS ').strip()
        rp = robotparser.RobotFileParser()
        # 4xx: no robots file = everything allowed; 5xx / errors: treated as unknown
        rp.parse(text.splitlines() if status == '200' else [])
        known = status == '200' or status.startswith('4')
        for r in g.itertuples():
            path = urlparse(r.url)._replace(scheme='', netloc='').geturl() or '/'
            row = dict(url=r.url, domain=dom, robots_status=status, robots_known=known,
                       **{f'count_{e}': getattr(r, f'count_{e}') for e in ENGINES})
            for a in AGENTS:
                row[f'blocked_{a}'] = (not rp.can_fetch(a, 'https://' + dom + path)) if known else np.nan
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(TABLES / 'robots_access.csv', index=False)
    return t


def analyse():
    t = access_table()
    k = t[t.robots_known]
    print(f'{len(t)} cited URLs on {t.domain.nunique()} websites; robots.txt known for '
          f'{k.domain.nunique()} websites ({100 * len(k) / len(t):.0f} % of URLs)')
    out = []
    for e in ENGINES:
        cited = k[k[f'count_{e}'] > 0]
        for a in AGENTS:
            b = cited[f'blocked_{a}'].astype(bool)
            out.append(dict(measure='share of the engine\'s cited URLs blocked for agent', engine=ENGINE_LABEL[e],
                            agent=a, urls=len(cited), pct=100 * b.mean(),
                            pct_of_citations=100 * cited.loc[b, f'count_{e}'].sum() / cited[f'count_{e}'].sum()))
    # website level: each engine's share of the website's citations, own agent blocked vs not
    w = k.groupby('domain').agg(**{f'count_{e}': (f'count_{e}', 'sum') for e in ENGINES},
                                **{f'blocked_{a}': (f'blocked_{a}', 'mean') for a in AGENTS})
    w['total'] = w[[f'count_{e}' for e in ENGINES]].sum(axis=1)
    w = w[w.total >= 5]
    for e in ENGINES:
        a = ENGINE_AGENT[e]
        for blocked in (False, True):
            g = w[(w[f'blocked_{a}'] > 0.5) == blocked]
            out.append(dict(measure='website share of citations from engine (websites with >= 5 citations)',
                            engine=ENGINE_LABEL[e], agent=a + (' blocked' if blocked else ' allowed'),
                            urls=len(g), pct=100 * (g[f'count_{e}'] / g.total).mean(),
                            pct_of_citations=100 * (g[f'count_{e}'] > 0).mean()))
    res = pd.DataFrame(out)
    res.to_csv(RESULTS / 'e04_ai_crawlers.csv', index=False)
    print(res.round(1).to_string(index=False))
    js_text()


def js_text():
    """Within-website effect of 'no readable text without JavaScript' on each engine's count."""
    from fe import fe_ols
    t = pd.read_csv(TABLES / 'url_table.csv', usecols=['url', 'domain', 'word_count_raw'] +
                    [f'count_{e}' for e in ENGINES])
    t = t[t.word_count_raw.notna()]
    t['no_text'] = (t.word_count_raw < 100).astype(float)
    t = t[t.groupby('domain').url.transform('size') >= 2]
    rows = []
    for e in ENGINES:
        t['y'] = np.log2(1 + t[f'count_{e}'])
        r, _ = fe_ols(t, 'y', ['no_text'], 'domain')
        rows.append(dict(engine=ENGINE_LABEL[e], pages=len(t), pct_no_text=100 * t.no_text.mean(),
                         coef=r.loc['no_text', 'coef'], se=r.loc['no_text', 'se'], p=r.loc['no_text', 'p'],
                         pct_change=100 * (2 ** r.loc['no_text', 'coef'] - 1)))
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / 'e04_js_text.csv', index=False)
    print(res.round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fetch', 'analyse'])
    fetch() if ap.parse_args().cmd == 'fetch' else analyse()
