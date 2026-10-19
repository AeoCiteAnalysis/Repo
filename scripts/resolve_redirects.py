"""Resolve Gemini redirect links (vertexaisearch.cloud.google.com) to real pages.

Writes data/tables/redirect_map.csv (redirect_url, final_url, status). Re-run
load_data.py afterwards so citations use the real URLs. Links that cannot be
resolved stay flagged is_redirect = True and are dropped (and counted) later.
Already-resolved links are skipped, so the script can be re-run safely.

Redirect tokens expire: links from the last few weeks resolve, older ones
return 404. Run this soon after each new day file arrives.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from common import TABLES, read_table

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/128.0 Safari/537.36')


def resolve(u):
    try:
        r = requests.get(u, headers={'User-Agent': UA}, timeout=20, allow_redirects=False)
        loc = r.headers.get('Location')
        return u, loc, r.status_code
    except requests.RequestException as e:
        return u, None, type(e).__name__
    finally:
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()

    c = read_table('citations.csv', all_days=True, usecols=['date', 'url', 'is_redirect'])
    # newest first: redirect tokens expire after some weeks (old ones return 404)
    todo = pd.Series(c[c.is_redirect].sort_values('date', ascending=False).url.unique())
    path = TABLES / 'redirect_map.csv'
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(
        columns=['redirect_url', 'final_url', 'status'])
    todo = todo[~todo.isin(old.dropna(subset=['final_url']).redirect_url)]
    if a.limit:
        todo = todo.head(a.limit)
    print(f'{len(todo)} redirect links to resolve')
    with ThreadPoolExecutor(a.workers) as ex:
        new = pd.DataFrame(list(ex.map(resolve, todo)),
                           columns=['redirect_url', 'final_url', 'status'])
    out = pd.concat([old[~old.redirect_url.isin(new.redirect_url)], new], ignore_index=True)
    out.to_csv(path, index=False)
    print(f'resolved {new.final_url.notna().sum()} / {len(new)}; '
          f'total mapped {out.final_url.notna().sum()}  -> {path}')


if __name__ == '__main__':
    main()
