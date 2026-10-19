"""Step A6b - outside authority of every cited website (one value per registered domain).

  python scripts/authority.py cc --release cc-main-2026-jun-jul-aug   Common Crawl web graph (links)
  python scripts/authority.py majestic                                Majestic Million (linking networks)
  python scripts/authority.py tranco [--list-id XXXX]                 Tranco top-1M (traffic)
  python scripts/authority.py crux [--month 202608]                   Chrome UX Report rank (traffic)
  python scripts/authority.py opr                                     Open PageRank (optional, needs OPR_API_KEY)
  python scripts/authority.py merge                                   -> data/tables/domain_authority.csv

All sources except opr are free downloads that need no key or account.

Each source writes a dated snapshot to data/tables/authority/<source>_<YYYYMMDD>.csv
and never overwrites old ones (R14 needs monthly history); `merge` uses the
latest snapshot of each source. Domains missing from a source get NaN plus a
<source>_found = 0 flag - never zero.
"""
import argparse
import gzip
import io
import os
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests

from common import TABLES, read_table, registered_domain

SNAP = TABLES / 'authority'
CC_URL = 'https://data.commoncrawl.org/projects/hyperlinkgraph/{r}/domain/{r}-domain-ranks.txt.gz'
# Classic Open PageRank endpoint; the guide's newer host (openpagerank.keywordseverywhere.com,
# /v1/domains/bulk) can be passed with --endpoint - check its docs for the response format.
OPR_URL = 'https://openpagerank.com/api/v1.0/getPageRank'
TRANCO_ID_URL = 'https://tranco-list.eu/top-1m-id'
TRANCO_URL = 'https://tranco-list.eu/download/{id}/1000000'
MAJESTIC_URL = 'https://downloads.majestic.com/majestic_million.csv'
# Monthly CrUX top lists (origin, rank bucket 1000 / 10000 / ... / 50000000), mirrored on GitHub
CRUX_URL = 'https://raw.githubusercontent.com/zakird/crux-top-lists/main/data/global/{m}.csv.gz'


def study_domains():
    """Registered domains of every cited website (redirect rows use their title-derived host)."""
    c = read_table('citations.csv', usecols=['domain'])
    hosts = c.domain.dropna().unique()
    return sorted({registered_domain(h) for h in hosts})


def save(df, source):
    SNAP.mkdir(parents=True, exist_ok=True)
    path = SNAP / f'{source}_{date.today():%Y%m%d}.csv'
    df.to_csv(path, index=False)
    print(f'{len(df)} rows -> {path}')


def cmd_cc(release):
    """Stream the multi-GB ranks file and keep only our domains (stored reversed: com.google)."""
    want = {'.'.join(reversed(d.split('.'))): d for d in study_domains()}
    url = CC_URL.format(r=release)
    print(f'streaming {url} for {len(want)} domains ...')
    rows = []
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw) as gz:
            for i, line in enumerate(io.TextIOWrapper(gz, encoding='utf-8')):
                if line.startswith('#'):
                    continue
                hc_pos, hc_val, pr_pos, pr_val, host_rev, *_ = line.rstrip('\n').split('\t')
                if host_rev in want:
                    rows.append(dict(domain=want[host_rev], cc_harmonic=float(hc_val),
                                     cc_harmonic_rank=int(hc_pos), cc_pagerank=float(pr_val),
                                     cc_pagerank_rank=int(pr_pos), cc_release=release))
                if i % 10_000_000 == 0 and i:
                    print(f'  {i:,} lines, {len(rows)} matched')
    save(pd.DataFrame(rows), 'cc')


def cmd_opr(endpoint):
    key = os.environ.get('OPR_API_KEY')
    if not key:
        raise SystemExit('set OPR_API_KEY (free key, see guide Step A6b)')
    doms, rows = study_domains(), []
    for i in range(0, len(doms), 100):                       # 100 domains per request
        batch = doms[i:i + 100]
        r = requests.get(endpoint, params=[('domains[]', d) for d in batch],
                         headers={'API-OPR': key}, timeout=60)
        r.raise_for_status()
        for x in r.json().get('response', []):
            ok = str(x.get('status_code')) == '200'
            rows.append(dict(domain=x.get('domain'),
                             opr_score=x.get('page_rank_decimal') if ok else np.nan,
                             opr_rank=pd.to_numeric(x.get('rank'), errors='coerce') if ok else np.nan,
                             opr_ref_domains=x.get('referring_domains', np.nan),
                             opr_date=f'{date.today():%Y-%m-%d}'))
        print(f'  {min(i + 100, len(doms))}/{len(doms)}')
    save(pd.DataFrame(rows), 'opr')


def cmd_tranco(list_id):
    list_id = list_id or requests.get(TRANCO_ID_URL, timeout=30).text.strip()
    r = requests.get(TRANCO_URL.format(id=list_id), timeout=120)
    r.raise_for_status()
    if r.content[:2] == b'PK':
        r_text = zipfile.ZipFile(io.BytesIO(r.content)).read('top-1m.csv').decode()
    else:
        r_text = r.text
    t = pd.read_csv(io.StringIO(r_text), names=['tranco_rank', 'domain'])
    t = t.groupby('domain', as_index=False).tranco_rank.min()
    doms = pd.DataFrame({'domain': study_domains()})
    out = doms.merge(t, on='domain', how='left')
    out['tranco_list_id'] = list_id
    save(out, 'tranco')


def cmd_majestic():
    """Top 1M domains ranked by the number of different IP subnets linking to them."""
    m = pd.read_csv(MAJESTIC_URL, usecols=['GlobalRank', 'Domain', 'RefSubNets', 'RefIPs'])
    m = m.rename(columns={'Domain': 'domain', 'GlobalRank': 'majestic_rank',
                          'RefSubNets': 'majestic_ref_subnets', 'RefIPs': 'majestic_ref_ips'})
    m['domain'] = m.domain.str.lower().str.removeprefix('www.')
    m = m.groupby('domain', as_index=False).agg(
        majestic_rank=('majestic_rank', 'min'), majestic_ref_subnets=('majestic_ref_subnets', 'max'),
        majestic_ref_ips=('majestic_ref_ips', 'max'))
    out = pd.DataFrame({'domain': study_domains()}).merge(m, on='domain', how='left')
    out['majestic_date'] = f'{date.today():%Y-%m-%d}'
    save(out, 'majestic')


def cmd_crux(month):
    """Best (smallest) CrUX rank bucket over all origins of each registered domain."""
    month = month or 'current'
    c = pd.read_csv(CRUX_URL.format(m=month), compression='gzip')
    c['domain'] = c.origin.str.replace(r'^https?://', '', regex=True).str.split('/').str[0]         .str.split(':').str[0].str.lower().str.removeprefix('www.').map(registered_domain)
    c = c.groupby('domain', as_index=False)['rank'].min().rename(columns={'rank': 'crux_rank'})
    out = pd.DataFrame({'domain': study_domains()}).merge(c, on='domain', how='left')
    out['crux_month'] = month
    save(out, 'crux')


def latest(source):
    files = sorted(SNAP.glob(f'{source}_*.csv'))
    return pd.read_csv(files[-1]) if files else None


def cmd_merge():
    out = pd.DataFrame({'domain': study_domains()})
    for src in ('cc', 'majestic', 'tranco', 'crux', 'opr', 'paid'):
        df = latest(src)
        if df is None:
            print(f'no {src} snapshot - skipped')
            continue
        out = out.merge(df.drop_duplicates('domain'), on='domain', how='left')
    if 'tranco_rank' in out:
        out['tranco_found'] = out.tranco_rank.notna().astype(int)
        out['tranco_rank'] = out.tranco_rank.fillna(1_000_001)       # guide: not listed = 1,000,001
    if 'cc_harmonic' in out:
        out['cc_found'] = out.cc_harmonic.notna().astype(int)
    if 'opr_score' in out:
        out['opr_found'] = out.opr_score.notna().astype(int)
    if 'majestic_rank' in out:
        out['majestic_found'] = out.majestic_rank.notna().astype(int)
    if 'crux_rank' in out:
        out['crux_found'] = out.crux_rank.notna().astype(int)
    # ranks and link counts are very skewed: keep log versions for the models
    for col in ('cc_harmonic_rank', 'cc_pagerank_rank', 'tranco_rank', 'opr_rank', 'opr_ref_domains',
                'majestic_rank', 'majestic_ref_subnets', 'majestic_ref_ips', 'crux_rank'):
        if col in out:
            out[f'log_{col}'] = np.log10(pd.to_numeric(out[col], errors='coerce') + 1)
    if 'cc_pagerank' in out:                 # PageRank values are tiny and very skewed: log them too
        out['log_cc_pagerank'] = np.log10(out.cc_pagerank)
    out.to_csv(TABLES / 'domain_authority.csv', index=False)
    print(f'{len(out)} domains -> data/tables/domain_authority.csv')
    found = [c for c in out if c.endswith('_found')]
    if found:
        print(out[found].mean().round(3).to_string())
    for col in ('cc_harmonic', 'majestic_ref_subnets', 'opr_score'):
        if col in out:                     # sanity check: big platforms should be near the top
            print(f'\ntop 20 by {col}:', ', '.join(out.nlargest(20, col).domain))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('cc'); p.add_argument('--release', required=True)
    p = sub.add_parser('opr'); p.add_argument('--endpoint', default=OPR_URL)
    p = sub.add_parser('tranco'); p.add_argument('--list-id')
    sub.add_parser('majestic')
    p = sub.add_parser('crux'); p.add_argument('--month', help='YYYYMM (default: current)')
    sub.add_parser('merge')
    a = ap.parse_args()
    {'cc': lambda: cmd_cc(a.release), 'opr': lambda: cmd_opr(a.endpoint),
     'tranco': lambda: cmd_tranco(a.list_id), 'majestic': cmd_majestic,
     'crux': lambda: cmd_crux(a.month), 'merge': cmd_merge}[a.cmd]()


if __name__ == '__main__':
    main()
