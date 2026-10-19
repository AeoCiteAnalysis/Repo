"""Step A4 - label every website as own, competitor, third_party (or mixed).

  python scripts/brand_domains.py list     write results/top_domains_for_labelling.csv
                                           (200 most-cited hosts, to decide by hand)
  python scripts/brand_domains.py apply    add role/brand/reg_domain columns to citations.csv

data/tables/brand_domains.csv (brand,domain,role,confidence,note) is kept by hand.
`domain` is a host, optionally followed by a path prefix (support.google.com/fiber).
Play Store rows can name one app: play.google.com/store/apps/details?id=<app id>.
A citation matches a row if its host equals the row's host or ends with '.' + host,
and (for path rows) its path starts with the prefix. The longest matching row wins,
so support.google.com/fiber is 'own' while the rest of support.google.com is 'mixed'.
Unlisted sites are third_party with confidence 'default'.

`brand` names the brand a site is about. Rows with role third_party and a brand
are dealer / reseller sites: not controlled by the brand, kept for reference.
"""
import argparse
from urllib.parse import urlparse, parse_qs

import pandas as pd

from common import DATASET, TABLES, RESULTS, read_table, registered_domain

# Starting lists per dataset (AEO_DATASET). gfiber: client assumed to be Google Fiber.
SEEDS = {}
SEEDS['workorb'] = """brand,domain,role
Workorb,workorb.com,own
Loopio,loopio.com,competitor
Unanet,unanet.com,competitor
OpenAsset,openasset.com,competitor
Joist AI,joist.ai,competitor
"""
SEEDS['gfiber'] = """brand,domain,role
Google Fiber,gfiber.com,own
Google Fiber,fiber.google.com,own
Google Fiber,fiber.googleblog.com,own
AT&T Fiber,att.com,competitor
Xfinity,xfinity.com,competitor
Quantum Fiber,quantumfiber.com,competitor
Spectrum,spectrum.com,competitor
Spectrum,spectrum.net,competitor
T-Mobile,t-mobile.com,competitor
Metronet,metronet.com,competitor
Astound,astound.com,competitor
,support.google.com,mixed
"""
SEED = SEEDS[DATASET]


def load_brand_domains():
    path = TABLES / 'brand_domains.csv'
    if not path.exists():
        path.write_text(SEED, encoding='utf-8')
        print(f'wrote starting list {path} - review it by hand')
    bd = pd.read_csv(path)
    bd['domain'] = bd.domain.str.lower().str.removeprefix('www.')
    return bd


def _rules(bd):
    rules = []
    for r in bd.itertuples():
        host, _, path = r.domain.partition('/')
        rules.append((len(r.domain), host, '/' + path if path else '', r.role,
                      r.brand if isinstance(r.brand, str) else None,
                      getattr(r, 'confidence', 'sure') if isinstance(getattr(r, 'confidence', None), str)
                      else 'sure'))
    return sorted(rules, key=lambda x: -x[0])


def label_sites(keys, bd):
    """keys are 'host' or 'host/path'. Return DataFrame(site_key, role, brand, label_confidence)."""
    rules = _rules(bd)
    out = []
    for k in keys:
        host, _, path = k.partition('/')
        path = '/' + path if path else ''
        for _, rh, rp, role, brand, conf in rules:
            if (host == rh or host.endswith('.' + rh)) and                     (not rp or path == rp or path.startswith((rp + '/', rp + '?'))):
                out.append((k, role, brand, conf))
                break
        else:
            out.append((k, 'third_party', None, 'default'))
    return pd.DataFrame(out, columns=['site_key', 'role', 'brand', 'label_confidence'])


def label_hosts(hosts, bd):
    """Host-level labels (path rules ignored unless the host itself is listed)."""
    return label_sites(hosts, bd).rename(columns={'site_key': 'domain'})


def site_key(domain, url, redirect):
    """host + lower-case path; redirect rows only know their host."""
    if not isinstance(domain, str) or not domain:
        return None
    if redirect or not isinstance(url, str):
        return domain
    p = urlparse(url)
    key = domain + p.path.lower()
    if domain == 'play.google.com':            # app pages: the owner is in ?id=<app id>
        app = parse_qs(p.query).get('id')
        if app:
            key += '?id=' + app[0].lower()
    return key


def cmd_list(n):
    c = read_table('citations.csv', usecols=['domain', 'engine', 'is_redirect', 'repeat_in_answer'])
    c = c[~c.is_redirect & ~c.repeat_in_answer]
    top = (c.groupby('domain').size().rename('citations').to_frame()
           .join(c.pivot_table(index='domain', columns='engine', aggfunc='size', fill_value=0))
           .sort_values('citations', ascending=False).head(n).reset_index())
    lab = label_hosts(top.domain, load_brand_domains())
    top = top.merge(lab, on='domain')
    top['reg_domain'] = top.domain.map(registered_domain)
    RESULTS.mkdir(exist_ok=True)
    top.to_csv(RESULTS / 'top_domains_for_labelling.csv', index=False)
    print(top.head(40).to_string(index=False))
    print(f'\nfull list -> results/top_domains_for_labelling.csv; '
          f'add brand sites you find to data/tables/brand_domains.csv')


def cmd_apply():
    c = read_table('citations.csv', all_days=True)       # rewrites the file: keep every day
    c = c.drop(columns=[x for x in ('role', 'brand', 'reg_domain', 'label_confidence', 'site_key')
                        if x in c.columns])
    c['site_key'] = [site_key(d, u, r) for d, u, r in zip(c.domain, c.url, c.is_redirect)]
    lab = label_sites(c.site_key.dropna().unique(), load_brand_domains())
    c = c.merge(lab, on='site_key', how='left').drop(columns='site_key')
    c['role'] = c.role.fillna('third_party')
    c['label_confidence'] = c.label_confidence.fillna('default')
    regs = {d: registered_domain(d) for d in c.domain.dropna().unique()}
    c['reg_domain'] = c.domain.map(regs)
    c.to_csv(TABLES / 'citations.csv', index=False)
    (c.groupby(['domain', 'role', 'brand', 'label_confidence'], dropna=False).size()
      .rename('citations').reset_index().sort_values('citations', ascending=False)
      .to_csv(TABLES / 'domain_roles.csv', index=False))
    u = c[~c.repeat_in_answer]
    share = u.groupby(['engine', 'role']).size().unstack(fill_value=0)
    print((100 * share.div(share.sum(axis=1), axis=0)).round(1).to_string())
    print(u[u.role.isin(['own', 'competitor'])].groupby(['brand', 'label_confidence'])
          .size().unstack(fill_value=0).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['list', 'apply'])
    ap.add_argument('-n', type=int, default=200)
    a = ap.parse_args()
    cmd_list(a.n) if a.cmd == 'list' else cmd_apply()


if __name__ == '__main__':
    main()
