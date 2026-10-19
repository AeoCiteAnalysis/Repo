"""N7 - Add pages that were never cited (brand-website half).

Question. Are the features that predict HOW OFTEN a page is cited the same as the ones that decide
WHETHER it is cited at all?

Comparison group: pages on the tracked brands' own websites that were never cited. The guide's
other group (top-10 search results per prompt) needs a paid search API and is not done yet.
  1. collect  For each brand website with >= 200 citations (dealer sites and path rules excluded):
              read its sitemaps (robots.txt 'Sitemap:' lines, else /sitemap.xml; sitemap indexes
              followed), keep HTML-looking URLs on that host, drop every URL ever cited, and draw
              a random sample of up to 600 never-cited pages per website (seed 42).
  2. fetch    Download them exactly like Step A5 (same user-agent, robots.txt, 1 s per host).
  3. measure  Step A6 features (same code, same prompts, same embedding model).
  4. model    Logistic regression: cited (yes/no) on the z-scored page features + page type +
              one indicator per website; cited = pages of the same websites cited >= 1 time.
              Compared side by side with R2 (how often, own + competitor pages).
Re-running skips finished steps.

Outputs: data/tables/n07_candidates.csv, data/tables/n07_pages_index.csv,
         data/tables/n07_page_features.csv, results/n07_cited_vs_not.csv
"""
import csv
import gzip
import io
import re
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from urllib.parse import urlparse
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

from common import BINARY_FEATURES, unit_of, TABLES, PAGES, RESULTS, SEED, clean_url, domain_of, url_id, read_table
from fetch_pages import UA, HostGate, fetch_one, make_session, interleave_by_host, INDEX_COLS
from r02_main_model import FEATURES, LABEL

ROUND = 'n07-2026-09-27'
PER_SITE = 600
MIN_CITATIONS = 200
MAX_SITEMAP_URLS = 50_000
SKIP_EXT = re.compile(r'\.(pdf|jpe?g|png|gif|svg|webp|mp4|mp3|zip|xml|json|css|js|ico|doc|docx|xls|xlsx)$', re.I)


def brand_sites():
    bd = pd.read_csv(TABLES / 'brand_domains.csv')
    bd = bd[bd.role.isin(['own', 'competitor']) & ~bd.domain.str.contains('/')
            & ~bd.note.fillna('').str.contains('dealer', case=False)]
    c = read_table('citations.csv', usecols=['domain', 'url', 'is_redirect', 'repeat_in_answer', 'role', 'brand'])
    c = c[~c.is_redirect & ~c.repeat_in_answer & c.role.isin(['own', 'competitor'])]
    counts = c.groupby('domain').agg(citations=('url', 'size'), brand=('brand', 'first'))
    sites = counts[(counts.citations >= MIN_CITATIONS) & counts.index.isin(bd.domain)]
    return sites.sort_values('citations', ascending=False), set(c.url)


def get(url, session):
    try:
        r = session.get(url, headers={'User-Agent': UA}, timeout=30)
        if r.status_code != 200:
            return None
        data = r.content
        if url.endswith('.gz') or data[:2] == b'\x1f\x8b':
            data = gzip.decompress(data)
        return data
    except Exception:
        return None


def sitemap_urls(host, session):
    """All page URLs listed in the host's sitemaps (capped)."""
    roots = []
    for base in (f'https://www.{host}', f'https://{host}'):
        txt = get(base + '/robots.txt', session)
        if txt:
            roots += re.findall(rb'(?im)^\s*sitemap:\s*(\S+)', txt)
        if roots:
            break
    roots = [r.decode() for r in roots] or [f'https://www.{host}/sitemap.xml', f'https://{host}/sitemap.xml']
    seen, queue, pages = set(), list(roots), []
    while queue and len(pages) < MAX_SITEMAP_URLS and len(seen) < 300:
        sm_url = queue.pop(0)
        if sm_url in seen:
            continue
        seen.add(sm_url)
        data = get(sm_url, session)
        if not data:
            continue
        try:
            root = ElementTree.fromstring(data)
        except ElementTree.ParseError:
            continue
        locs = [e.text.strip() for e in root.iter() if e.tag.endswith('loc') and e.text]
        if root.tag.endswith('sitemapindex'):
            queue += locs
        else:
            pages += locs
    return pages


def collect(sites, cited):
    path = TABLES / 'n07_candidates.csv'
    if path.exists():
        return pd.read_csv(path)
    rng = np.random.default_rng(SEED)
    session = make_session(8)
    rows = []
    for host, r in sites.iterrows():
        urls = sitemap_urls(host, session)
        clean = {clean_url(u) for u in urls if not SKIP_EXT.search(urlparse(u).path)}
        clean = [u for u in clean if domain_of(u) == host and u not in cited]
        take = rng.choice(len(clean), size=min(PER_SITE, len(clean)), replace=False) if clean else []
        sample = [clean[i] for i in take]
        rows += [dict(url=u, domain=host, brand=r.brand) for u in sample]
        print(f'  {host}: {len(urls):,} sitemap URLs, {len(clean):,} never cited, sampled {len(sample)}')
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def fetch(cand):
    idx_path = TABLES / 'n07_pages_index.csv'
    done = set(pd.read_csv(idx_path).url) if idx_path.exists() else set()
    todo = interleave_by_host([u for u in cand.url if u not in done])     # spread over websites
    out_dir = PAGES / ROUND
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'fetch: {len(todo)} pages')
    gate, session = HostGate(), make_session(16)
    new = not idx_path.exists()
    with open(idx_path, 'a', newline='', encoding='utf-8') as f, ThreadPoolExecutor(16) as ex:
        w = csv.DictWriter(f, fieldnames=INDEX_COLS)
        if new:
            w.writeheader()
        futs = [ex.submit(fetch_one, u, ROUND, out_dir, gate, session) for u in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            w.writerow(fu.result())
            if i % 500 == 0:
                f.flush(); print(f'  {i}/{len(todo)}')
    return pd.read_csv(idx_path)


def measure(idx):
    path = TABLES / 'n07_page_features.csv'
    if path.exists():
        return pd.read_csv(path)
    from features import _parse, embed_matches, terms, jaccard
    ok = idx[idx.status == 'ok']
    jobs = [(PAGES / ROUND / f'{r.id}.html', r.url) for r in ok.itertuples()]
    parsed = {}
    with ProcessPoolExecutor(8) as ex:
        for url, p, err in ex.map(_parse, jobs, chunksize=16):
            if p is not None:
                parsed[url] = p
    prompts = sorted(read_table('answers.csv', usecols=['prompt']).prompt.dropna().unique())
    prompt_terms = set().union(*(terms(q) for q in prompts))
    f = pd.DataFrame.from_dict({u: {k: v for k, v in p.items() if k != 'paragraphs'} for u, p in parsed.items()},
                               orient='index')
    f['prompt_match_words'] = [jaccard(terms(p['title'] + '\n' + '\n'.join(p['paragraphs'])), prompt_terms)
                               for p in parsed.values()]
    f = f.join(embed_matches(parsed, prompts))
    f.index.name = 'url'
    f = f.reset_index()
    f.loc[f.page_type == 'forum', 'has_author'] = 0
    f.to_csv(path, index=False)
    return f


def model(nf, sites):
    pf = pd.read_csv(TABLES / 'page_features.csv')
    ua = read_table('url_table_all.csv', usecols=['url', 'domain', 'citation_count'])
    cited = pf.merge(ua, on='url')
    cited = cited[cited.domain.isin(sites.index)].assign(cited=1)
    nf = nf.assign(domain=nf.url.map(domain_of), cited=0, citation_count=0)
    d = pd.concat([cited, nf], ignore_index=True)
    # keep websites with both groups (>= 20 each): a website with only cited or only never-cited
    # pages is predicted perfectly by its indicator and breaks the model (separation)
    k = d.groupby(['domain', 'cited']).size().unstack(fill_value=0)
    keep = k[(k[0] >= 20) & (k[1] >= 20)].index
    print('websites used:', ', '.join(keep), '| dropped:', ', '.join(k.index.difference(keep)))
    d = d[d.domain.isin(keep)].reset_index(drop=True)
    d['page_age_days'] = (pd.Timestamp('2026-09-06') - pd.to_datetime(d.page_date, errors='coerce')).dt.days
    d.loc[d.page_age_days < 0, 'page_age_days'] = np.nan
    X = []
    counts = {'word_count', 'outbound_links', 'n_headings', 'n_tables', 'n_lists', 'page_age_days'}
    sd = {}
    for f_ in FEATURES:
        v = d[f_].astype(float)
        if f_ in counts:                     # very skewed (up to 51k words, 692 headings): log for stability
            v = np.log1p(v.clip(lower=0))
        if v.isna().any():
            d[f_ + '_missing'] = v.isna().astype(float); X.append(f_ + '_missing')
            v = v.fillna(v.median())
        sd[f_] = v.std()
        d[f_] = (v - v.mean()) / v.std()
        X.append(f_)
    # page types with < 20 pages in either group would separate perfectly: fold them into "other"
    pt = pd.crosstab(d.page_type, d.cited)
    rare = pt[(pt.min(axis=1) < 20)].index
    d['page_type'] = d.page_type.where(~d.page_type.isin(rare), 'other')
    types = pd.get_dummies(d.page_type, prefix='type', drop_first=True, dtype=float)
    sites_d = pd.get_dummies(d.domain, prefix='site', drop_first=True, dtype=float)
    Xm = sm.add_constant(pd.concat([d[X], types, sites_d], axis=1))
    Xm = Xm.loc[:, Xm.std() > 0].assign(const=1.0)
    m = sm.Logit(d.cited, Xm).fit(disp=0, cov_type='HC1', maxiter=500)
    print('converged:', m.mle_retvals['converged'])
    r2 = pd.read_csv(RESULTS / 'r02_main_model.csv')
    r2 = r2[(r2['sample'] == 'own + competitor') & (r2.kind == 'feature')].set_index('term')
    out = pd.DataFrame({'feature': [LABEL[f_] for f_ in FEATURES],
                        'odds_ratio_cited_at_all': [np.exp(m.params[f_]) for f_ in FEATURES],
                        # yes/no features: has vs not (0 -> 1 = 1 / SD in z units); others per SD
                        'unit': [unit_of(f_) for f_ in FEATURES],
                        'odds_ratio_in_unit': [np.exp(m.params[f_] / (sd[f_] if f_ in BINARY_FEATURES else 1.0))
                                               for f_ in FEATURES],
                        'p_cited_at_all': [m.pvalues[f_] for f_ in FEATURES],
                        'r2_pct_more_citations': [r2.loc[f_, 'pct_change'] for f_ in FEATURES],
                        'r2_p': [r2.loc[f_, 'p'] for f_ in FEATURES]})
    out['n_cited'], out['n_never_cited'], out['n_websites'] = int(d.cited.sum()), int((d.cited == 0).sum()), d.domain.nunique()
    out.to_csv(RESULTS / 'n07_cited_vs_not.csv', index=False)
    print(out.round(3).to_string(index=False))


def main():
    sites, cited = brand_sites()
    print(f'{len(sites)} brand websites:', ', '.join(sites.index))
    cand = collect(sites, cited)
    print(f'candidates: {len(cand):,}')
    idx = fetch(cand)
    print(idx.status.value_counts().to_dict())
    nf = measure(idx)
    print(f'measured never-cited pages: {len(nf):,}')
    model(nf, sites)


if __name__ == '__main__':
    main()
