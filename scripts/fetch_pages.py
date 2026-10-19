"""Step A5 - download every cited page once per round.

  python scripts/fetch_pages.py --round 2026-09-26 [--min-citations 2] [--limit N]

Saves HTML to data/pages/<round>/<id>.html (id = short hash of the URL) and
appends one row per URL to data/tables/pages_index.csv:
  url, id, round, status, final_url, content_type, bytes, fetched_at, error
Pages that fail (errors, robots.txt, PDFs, video) stay in the index with a status.
The script resumes: URLs already in the index for this round are skipped.

Politeness: browser user-agent, 20 s timeout, >= 1 s between requests to the
same host, robots.txt respected. Re-run every 2 weeks with a new --round (N8).
"""
import argparse
import csv
import ssl
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib import robotparser
from urllib.parse import urlparse

import pandas as pd
import certifi
import requests
from requests.adapters import HTTPAdapter

from common import TABLES, PAGES, read_table, url_id, domain_of

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/128.0 Safari/537.36')
TIMEOUT = 20
HOST_DELAY = 1.0
INDEX_COLS = ['url', 'id', 'round', 'status', 'final_url', 'content_type', 'bytes',
              'fetched_at', 'error']
SKIP_HOSTS = {'youtube.com': 'video', 'm.youtube.com': 'video', 'youtu.be': 'video'}


class SharedSSLAdapter(HTTPAdapter):
    """Reuse one SSL context for every host.

    By default requests makes urllib3 reload the whole CA bundle for each new
    connection (~140 ms CPU); with thousands of hosts that made the crawl CPU-bound.
    Certificates are still verified against the same bundle.
    """
    CTX = ssl.create_default_context(cafile=certifi.where())

    def init_poolmanager(self, *a, **kw):
        kw['ssl_context'] = self.CTX
        return super().init_poolmanager(*a, **kw)

    def cert_verify(self, conn, url, verify, cert):
        super().cert_verify(conn, url, verify, cert)
        if verify is True:
            conn.ca_certs = conn.ca_cert_dir = None      # CTX already holds the CA bundle


def make_session(workers):
    s = requests.Session()
    ad = SharedSSLAdapter(pool_connections=256, pool_maxsize=workers)
    s.mount('https://', ad)
    s.mount('http://', HTTPAdapter(pool_connections=256, pool_maxsize=workers))
    return s


class HostGate:
    """One lock + last-request time per host, so each host sees >= HOST_DELAY spacing."""

    def __init__(self):
        self.locks = defaultdict(threading.Lock)
        self.last = defaultdict(float)
        self.robots = {}
        self.meta = threading.Lock()

    def wait(self, host):
        with self.meta:
            lock = self.locks[host]
        lock.acquire()
        gap = time.monotonic() - self.last[host]
        if gap < HOST_DELAY:
            time.sleep(HOST_DELAY - gap)
        return lock

    def done(self, host, lock):
        self.last[host] = time.monotonic()
        lock.release()

    def allowed(self, url, session):
        p = urlparse(url)
        key = f'{p.scheme}://{p.netloc}'
        if key not in self.robots:
            rp = robotparser.RobotFileParser()
            try:
                r = session.get(key + '/robots.txt', headers={'User-Agent': UA}, timeout=10)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except requests.RequestException:
                rp.parse([])
            self.robots[key] = rp
        return self.robots[key].can_fetch(UA, url)

    def known_disallowed(self, url):
        """True if robots.txt was already read for this host and blocks the URL (no request needed)."""
        p = urlparse(url)
        rp = self.robots.get(f'{p.scheme}://{p.netloc}')
        return rp is not None and not rp.can_fetch(UA, url)


def fetch_one(url, rnd, out_dir, gate, session):
    row = dict(url=url, id=url_id(url), round=rnd, status='', final_url='', content_type='',
               bytes=0, fetched_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
               error='')
    host = domain_of(url)
    if host in SKIP_HOSTS:
        row['status'] = SKIP_HOSTS[host]
        return row
    if gate.known_disallowed(url):               # no request is sent, so no need to wait
        row['status'] = 'robots_disallowed'
        return row
    lock = gate.wait(host)
    try:
        if not gate.allowed(url, session):
            row['status'] = 'robots_disallowed'
            return row
        hdrs = {'User-Agent': UA, 'Accept-Language': 'en-US,en;q=0.8'}
        try:
            r = session.get(url, headers=hdrs, timeout=TIMEOUT, allow_redirects=True)
        except requests.ConnectionError:
            # clean_url() dropped 'www.'; some sites only answer on the www host
            p = urlparse(url)
            r = session.get(p._replace(netloc='www.' + p.netloc).geturl(), headers=hdrs,
                            timeout=TIMEOUT, allow_redirects=True)
        ctype = r.headers.get('Content-Type', '').split(';')[0].strip().lower()
        row.update(final_url=r.url, content_type=ctype, bytes=len(r.content))
        if r.status_code != 200:
            row['status'] = f'http_{r.status_code}'
        elif 'pdf' in ctype:
            row['status'] = 'pdf'
        elif 'html' not in ctype and ctype:
            row['status'] = 'not_html'
        else:
            (out_dir / f"{row['id']}.html").write_bytes(r.content)
            row['status'] = 'ok'
    except requests.RequestException as e:
        row.update(status='error', error=type(e).__name__)
    finally:
        gate.done(host, lock)
    return row


def interleave_by_host(urls):
    """Round-robin over hosts so workers do not all queue on one busy host."""
    by_host = defaultdict(list)
    for u in urls:
        by_host[domain_of(u)].append(u)
    queues = list(by_host.values())
    out = []
    for i in range(max(map(len, queues), default=0)):
        out += [q[i] for q in queues if i < len(q)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', required=True, help='download round label, e.g. 2026-09-26')
    ap.add_argument('--min-citations', type=int, default=1)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()

    c = read_table('citations.csv', usecols=['answer_id', 'url', 'is_redirect', 'repeat_in_answer'])
    c = c[~c.is_redirect & ~c.repeat_in_answer]
    counts = c.url.value_counts()
    urls = counts[counts >= a.min_citations].index.tolist()   # most-cited first

    idx_path = TABLES / 'pages_index.csv'
    if idx_path.exists():
        idx = pd.read_csv(idx_path, dtype=str)
        urls = [u for u in urls if u not in set(idx.loc[idx['round'] == a.round, 'url'])]
    if a.limit:
        urls = urls[:a.limit]
    urls = interleave_by_host(urls)
    out_dir = PAGES / a.round
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'{len(urls)} URLs to fetch into {out_dir}')

    gate, session = HostGate(), make_session(a.workers)
    new_file = not idx_path.exists()
    with open(idx_path, 'a', newline='', encoding='utf-8') as f, ThreadPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(f, fieldnames=INDEX_COLS)
        if new_file:
            w.writeheader()
        futs = [ex.submit(fetch_one, u, a.round, out_dir, gate, session) for u in urls]
        stats = defaultdict(int)
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            w.writerow(row)
            stats[row['status']] += 1
            if i % 200 == 0:
                f.flush()
                print(f'{i}/{len(urls)} {dict(stats)}')
    print('done', dict(stats))


if __name__ == '__main__':
    main()
