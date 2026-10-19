"""X6 - Selection at equal rank: does content matter once a page is retrieved?

X3 showed that ChatGPT cites mostly from the top of its own search results. Two roles for page
content are possible: (a) it gets the page retrieved and ranked high (retrieval), and/or (b) among the
retrieved pages, better-matching pages are preferred at the same rank (selection). X6 separates them.

Unit: every page ChatGPT consulted in a replay search with per-call source lists (x03_replay), i.e. one
row per (prompt, run, page) with its best rank over the run's search calls, the number of calls that
returned it, and whether the answer cited it. A "list" = one prompt in one run.

  fetch    download consulted pages that have no HTML yet (data[/<ds>]/pages/replay_consulted/,
           index x06_pages_index.csv; same polite crawler as Step A5)
  extras   text for pages the crawl could not use: PDFs (text extracted with pypdf, first 30 pages)
           and Wikipedia articles (403 for browser user-agents; fetched through Wikipedia's REST API,
           which its robots.txt allows). Reddit forbids all crawlers in robots.txt and stays missing.
  analyse  parse all consulted pages, compute features, fit the models

Pages without text (robots, 403, errors) are KEPT in the models: their text features are set to the
list mean (z = 0) and a text_missing flag (and a Reddit flag) absorbs their difference, so cited and
uncited pages are compared on the same rows. A text-only version is reported as a robustness check.

Features per row:
  match to the prompt   emb_best (best cosine of the prompt with the title or any of the first 50
                        paragraphs, all-MiniLM-L6-v2 as in X2), emb_title, bm25 (BM25 of the page
                        for the prompt over the consulted corpus)
  match to the engine's own queries   q_best: best cosine of any search query of that run with the
                        title or paragraphs
  page                  log word count, headings, has FAQ / summary box / author / structured data
                        (0/1: has vs not), page type list/comparison/review, role own/competitor,
                        outside authority (Common Crawl harmonic centrality of the website)
Continuous features are z-scored WITHIN each list (relative to the other consulted pages).

Selection models (conditional logit, groups = lists with >= 1 cited and >= 1 uncited page with text):
  S0 rank only (log2 rank, found by > 1 call)       S1 + match to the prompt
  S2 + match to the engine's queries                S3 + page features
McFadden pseudo-R2 of each and the odds ratios of S3. Equal-rank table: % cited by match tercile
(within list) inside rank bands 1-3, 4-10, 11+.
Retrieval model: within each list, how rank relates to the same features (OLS of -log2 rank on the
within-list z-scores, list fixed effects; positive = ranked higher).

Outputs: results[/<ds>]/x06_selection_models.csv, x06_selection_coef.csv, x06_equal_rank.csv,
         x06_retrieval_coef.csv, x06_coverage.csv, figures[/<ds>]/x06_selection.png
"""
import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from common import TABLES, PAGES, RESULTS, FIGURES, DATASET, url_id
from x03_analyse import records, ranked

ROUND = 'replay_consulted'
IDX = TABLES / 'x06_pages_index.csv'
PARSED = TABLES / 'x06_parsed.jsonl'
MAX_PARAS = 50
TOKEN = re.compile(r'[a-z0-9][a-z0-9&\-]+')


def consulted():
    """Long table of consulted pages per list (runs with per-call source lists only)."""
    rows, queries = [], {}
    for j in records('openai'):
        if not j.get('calls') or not j['consulted']:
            continue
        run = f"{j['date']}|{j['rep']}"
        rows.append(ranked(j).assign(run=run))
        queries[(j['prompt_id'], run)] = [q for c in j['calls'] for q in c['queries']]
    L = pd.concat(rows, ignore_index=True).dropna(subset=['rank'])
    L['list_id'] = L.prompt_id + '|' + L.run
    return L, queries


def html_paths():
    """url -> saved HTML path, from every crawl index of this dataset."""
    out = {}
    for f in ('pages_index.csv', 'n07_pages_index.csv', 'x06_pages_index.csv'):
        p = TABLES / f
        if not p.exists():
            continue
        x = pd.read_csv(p, dtype=str)
        for r in x[x.status == 'ok'].itertuples():
            path = PAGES / r.round / f'{r.id}.html'
            if path.exists():
                out.setdefault(r.url, path)
    return out


def fetch():
    from fetch_pages import fetch_one, HostGate, make_session, interleave_by_host, INDEX_COLS
    L, _ = consulted()
    have = html_paths()
    done = set(pd.read_csv(IDX, dtype=str).url) if IDX.exists() else set()
    todo = interleave_by_host([u for u in L.url.unique() if u not in have and u not in done])
    out_dir = PAGES / ROUND
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'{DATASET}: {len(todo)} consulted pages to download', flush=True)
    gate, session = HostGate(), make_session(16)
    new = not IDX.exists()
    with open(IDX, 'a', newline='', encoding='utf-8') as f, ThreadPoolExecutor(16) as ex:
        w = csv.DictWriter(f, fieldnames=INDEX_COLS)
        if new:
            w.writeheader()
        futs = [ex.submit(fetch_one, u, ROUND, out_dir, gate, session) for u in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            w.writerow(fu.result())
            if i % 500 == 0:
                f.flush(); print(f'  {i}/{len(todo)}', flush=True)


def _pdf_html(content):
    import io
    from pypdf import PdfReader
    r = PdfReader(io.BytesIO(content))
    meta_title = (r.metadata.title if r.metadata and r.metadata.title else '') or ''
    text = '\n'.join((pg.extract_text() or '') for pg in r.pages[:30])
    paras = [re.sub(r'\s+', ' ', x).strip() for x in re.split(r'\n\s*\n|(?<=[.!?])\n', text)]
    paras = [x for x in paras if len(x) > 40]
    esc = lambda t: t.replace('&', '&amp;').replace('<', '&lt;')
    return ('<html><head><title>' + esc(meta_title) + '</title></head><body><main>' +
            ''.join('<p>' + esc(x) + '</p>' for x in paras) + '</main></body></html>')


def extras():
    """PDF text and Wikipedia REST API for consulted pages the crawl could not use."""
    import requests
    from datetime import datetime, timezone
    from fetch_pages import UA, INDEX_COLS, HostGate, interleave_by_host
    from common import domain_of
    from urllib.parse import urlparse, unquote, quote
    L, _ = consulted()
    have = html_paths()
    x = pd.read_csv(IDX, dtype=str) if IDX.exists() else pd.DataFrame(columns=INDEX_COLS)
    last = x.drop_duplicates('url', keep='last').set_index('url').status
    ever_pdf = set(x[x.status == 'pdf'].url)
    retry = ('http_403', 'http_429', 'error', None)            # rate limits and timeouts are retried
    want = [u for u in L.url.unique() if u not in have and
            ((u in ever_pdf and last.get(u) in ('pdf',) + retry) or
             ('wikipedia.org/wiki/' in u and last.get(u) in retry))]
    want = interleave_by_host(want)
    out_dir = PAGES / ROUND
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'{DATASET}: {len(want)} PDF / Wikipedia pages', flush=True)
    gate = HostGate()
    wiki_ua = 'AeoCiteAnalysis/1.0 (academic study of citations in AI answers)'

    def one(u):
        row = dict(url=u, id=url_id(u), round=ROUND, status='', final_url='', content_type='', bytes=0,
                   fetched_at=datetime.now(timezone.utc).isoformat(timespec='seconds'), error='')
        host = domain_of(u)
        lock = gate.wait(host)
        try:
            if 'wikipedia.org/wiki/' in u:
                pu = urlparse(u)
                title = unquote(pu.path.split('/wiki/', 1)[1])
                api = f'https://{pu.netloc}/api/rest_v1/page/html/{quote(title, safe="")}'
                r = requests.get(api, headers={'User-Agent': wiki_ua}, timeout=30)
                if r.status_code == 200:
                    html = r.text.replace('<head>', f'<head><title>{title.replace("_", " ")}</title>', 1)
                    (out_dir / f"{row['id']}.html").write_text(html, encoding='utf-8')
                    row.update(status='ok', final_url=api, content_type='text/html; wikipedia-api', bytes=len(r.content))
                else:
                    row['status'] = f'http_{r.status_code}'
            else:
                r = requests.get(u, headers={'User-Agent': UA}, timeout=60, stream=True)
                content = r.raw.read(25_000_000, decode_content=True)
                if r.status_code != 200:
                    row['status'] = f'http_{r.status_code}'
                else:
                    (out_dir / f"{row['id']}.html").write_text(_pdf_html(content), encoding='utf-8')
                    row.update(status='ok', final_url=r.url, content_type='application/pdf; text-extracted',
                               bytes=len(content))
        except Exception as e:
            row.update(status='error', error=type(e).__name__)
        finally:
            gate.done(host, lock)
        return row

    with open(IDX, 'a', newline='', encoding='utf-8') as f, ThreadPoolExecutor(8) as ex:
        w = csv.DictWriter(f, fieldnames=INDEX_COLS)
        stats = {}
        for i, row in enumerate(ex.map(one, want), 1):
            w.writerow(row)
            stats[row['status']] = stats.get(row['status'], 0) + 1
            if i % 200 == 0:
                f.flush(); print(f'  {i}/{len(want)} {stats}', flush=True)
    print(stats)


def parse_all(urls):
    """Parse HTML of the given URLs once (cached in x06_parsed.jsonl)."""
    from features import _parse
    cache = {}
    if PARSED.exists():
        with open(PARSED, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line); cache[r['url']] = r
    paths = html_paths()
    jobs = [(paths[u], u) for u in urls if u in paths and u not in cache]
    print(f'parsing {len(jobs)} pages ({len(cache)} cached)', flush=True)
    with open(PARSED, 'a', encoding='utf-8') as out, ProcessPoolExecutor(8) as ex:
        for url, p, err in ex.map(_parse, jobs, chunksize=16):
            if p is None:
                continue
            r = dict(url=url, title=p.get('title') or '', paragraphs=(p.get('paragraphs') or [])[:MAX_PARAS],
                     **{k: p.get(k) for k in ('word_count', 'n_headings', 'has_faq', 'has_summary_box',
                                              'has_author', 'has_schema', 'page_type')})
            cache[url] = r
            out.write(json.dumps(r, ensure_ascii=False) + '\n')
    return cache


def similarities(L, queries, pages, prompts):
    """emb_best, emb_title, q_best, bm25 for every row of L with page text."""
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    torch.set_num_threads(8)
    m = SentenceTransformer('all-MiniLM-L6-v2')
    enc = lambda xs: m.encode(xs, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    urls = sorted(set(L.url) & set(pages))
    units, owner, is_title = [], [], []
    for i, u in enumerate(urls):
        p = pages[u]
        for k, t in enumerate([p['title']] + p['paragraphs']):
            if t and t.strip():
                units.append(t[:1000]); owner.append(i); is_title.append(k == 0)
    print(f'embedding {len(units):,} title/paragraph units of {len(urls):,} pages', flush=True)
    U = enc(units)
    owner, is_title = np.array(owner), np.array(is_title)
    pids = sorted(L.prompt_id.unique())
    P = enc([prompts[p] for p in pids])
    S = U @ P.T                                            # units x prompts
    best = np.full((len(urls), len(pids)), np.nan, dtype=np.float32)
    title = np.full((len(urls), len(pids)), np.nan, dtype=np.float32)
    np.fmax.at(best, owner, S)
    np.fmax.at(title, owner[is_title], S[is_title])
    ui, pi = {u: i for i, u in enumerate(urls)}, {p: i for i, p in enumerate(pids)}
    L_all = L
    L = L[L.url.isin(ui)].copy()
    r_u, r_p = L.url.map(ui).to_numpy(), L.prompt_id.map(pi).to_numpy()
    L['emb_best'], L['emb_title'] = best[r_u, r_p], title[r_u, r_p]
    # match to the engine's own search queries of that run (best over queries and units)
    qtexts = sorted({q for v in queries.values() for q in v})
    Q = enc(qtexts) if qtexts else np.zeros((0, U.shape[1]))
    qi = {q: i for i, q in enumerate(qtexts)}
    qb = []
    for (lid, grp) in L.groupby('list_id'):
        pid, run = lid.split('|', 1)
        qs = [qi[q] for q in queries.get((pid, run), []) if q in qi]
        if not qs:
            qb += [(i, np.nan) for i in grp.index]; continue
        for i, u in zip(grp.index, grp.url):
            mask = owner == ui[u]
            qb.append((i, float((U[mask] @ Q[qs].T).max()) if mask.any() else np.nan))
    L['q_best'] = pd.Series(dict(qb))
    # BM25 of each page for its prompt over the consulted corpus
    docs = [(pages[u]['title'] + ' ') * 2 + ' '.join(pages[u]['paragraphs']) for u in urls]
    cv = CountVectorizer(token_pattern=TOKEN.pattern, lowercase=True, stop_words='english')
    D = cv.fit_transform(docs).tocsc().astype(float)
    dl = np.asarray(D.sum(axis=1)).ravel(); avg = dl.mean()
    df = np.asarray((D > 0).sum(axis=0)).ravel()
    idf = np.log(1 + (len(urls) - df + 0.5) / (df + 0.5))
    k1, b = 1.5, 0.75
    D = D.tocsr()
    vocab = cv.vocabulary_
    bm = np.zeros(len(L))
    for n, (u, p) in enumerate(zip(L.url, L.prompt_id)):
        terms = [vocab[t] for t in set(TOKEN.findall(prompts[p].lower())) if t in vocab]
        if not terms:
            continue
        row = D[ui[u]]
        tf = np.array([row[0, t] for t in terms])
        bm[n] = float((idf[terms] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[ui[u]] / avg))).sum())
    L['bm25'] = bm
    return L_all.join(L[['emb_best', 'emb_title', 'q_best', 'bm25']])


def page_features(L, pages):
    from common import read_table
    f = pd.DataFrame([{k: v for k, v in pages[u].items() if k not in ('paragraphs', 'title')} for u in L.url.unique()
                      if u in pages])
    L = L.merge(f, on='url', how='left')
    L['log_words'] = np.log1p(L.word_count.fillna(0))
    L['log_headings'] = np.log1p(L.n_headings.fillna(0))
    for c in ('has_faq', 'has_summary_box', 'has_author', 'has_schema'):
        L[c] = L[c].fillna(0).astype(float)
    L['type_list'] = L.page_type.isin(['list', 'review list']).astype(float)
    L['type_comparison'] = (L.page_type == 'comparison').astype(float)
    from common import domain_of
    from brand_domains import load_brand_domains, label_hosts
    L['domain'] = L.url.map(domain_of)
    lab = label_hosts(L.domain.unique(), load_brand_domains()).set_index('domain').role
    L['role_brand'] = L.domain.map(lab).isin(['own', 'competitor']).astype(float)
    auth = TABLES / 'domain_authority.csv'
    if auth.exists():
        a = pd.read_csv(auth)
        from common import registered_domain
        key = 'reg_domain' if 'reg_domain' in a else a.columns[0]
        m = a.set_index(key)['cc_harmonic'] if 'cc_harmonic' in a else None
        L['authority'] = L.domain.map(registered_domain).map(m) if m is not None else np.nan
    else:
        L['authority'] = np.nan
    L['text_missing'] = L.emb_best.isna().astype(float)
    L['is_reddit'] = (L.domain == 'reddit.com').astype(float)
    L['authority_missing'] = L.authority.isna().astype(float)
    L['authority'] = L.authority.fillna(L.authority.median())
    return L


MATCH = ['emb_best', 'emb_title', 'bm25']
QUERY = ['q_best']
PAGE = ['log_words', 'log_headings', 'has_faq', 'has_summary_box', 'has_author', 'has_schema',
        'type_list', 'type_comparison', 'role_brand', 'authority']
FLAGS = ['text_missing', 'is_reddit']
ZCOLS = MATCH + QUERY + ['log_words', 'log_headings', 'authority']


def zwithin(L, cols):
    g = L.groupby('list_id')                    # NaN (no text) is skipped by mean/std and set to 0 = list mean
    for c in cols:
        L[c + '_z'] = ((L[c] - g[c].transform('mean')) / g[c].transform('std').replace(0, np.nan)).fillna(0)
    return L


def analyse():
    from statsmodels.discrete.conditional_models import ConditionalLogit
    import statsmodels.api as sm
    L, queries = consulted()
    lab = pd.read_csv(TABLES / 'prompt_labels.csv', usecols=['prompt_id', 'prompt'])
    prompts = dict(zip(lab.prompt_id, lab.prompt))
    pages = parse_all(sorted(L.url.unique()))
    cov = dict(rows=len(L), lists=L.list_id.nunique(), pages=L.url.nunique(),
               pct_rows_with_text=100 * L.url.isin(pages).mean(),
               pct_cited_rows_with_text=100 * L[L.cited].url.isin(pages).mean(),
               pct_uncited_rows_with_text=100 * L[~L.cited].url.isin(pages).mean())
    print(cov, flush=True)
    L = similarities(L, queries, pages, prompts)
    L = page_features(L, pages)
    L['log_rank'] = np.log2(L['rank'])
    L['multi_call'] = (L.n_calls > 1).astype(float)
    L = zwithin(L, ZCOLS)
    L.to_csv(TABLES / 'x06_rows.csv', index=False)
    g = L[L.groupby('list_id').cited.transform('nunique') == 2].copy()
    cov.update(lists_in_model=g.list_id.nunique(), rows_in_model=len(g))
    pd.DataFrame([cov]).to_csv(RESULTS / 'x06_coverage.csv', index=False)

    Z = lambda cs: [c + '_z' if c in ZCOLS else c for c in cs]
    specs = {'S0 rank only': ['log_rank', 'multi_call'],
             'S0b rank + text-missing flags': ['log_rank', 'multi_call'] + FLAGS,
             'S1 + match to the prompt': ['log_rank', 'multi_call'] + FLAGS + Z(MATCH),
             'S2 + match to the engine queries': ['log_rank', 'multi_call'] + FLAGS + Z(MATCH + QUERY),
             'S3 + page features': ['log_rank', 'multi_call'] + FLAGS + Z(MATCH + QUERY + PAGE),
             'M only (no rank): match + queries + page': FLAGS + Z(MATCH + QUERY + PAGE),
             'S3 text-only rows (robustness)': ['log_rank', 'multi_call'] + Z(MATCH + QUERY + PAGE)}
    fits, coefs = [], []
    for name, X in specs.items():
        d = g[g.text_missing == 0] if 'text-only' in name else g
        d = d[d.groupby('list_id').cited.transform('nunique') == 2]
        X = [x for x in X if d[x].std() > 0]
        mdl = ConditionalLogit(d.cited.astype(float), d[X], groups=d.list_id)
        r = mdl.fit(disp=0, method='bfgs', maxiter=1000)
        ll0 = mdl.loglike(np.zeros(len(X)))
        fits.append(dict(model=name, lists=d.list_id.nunique(), rows=len(d), mcfadden_r2=1 - r.llf / ll0))
        if name.startswith('S3'):
            coefs += [dict(model=name, variable=v, coef=r.params[v], se=r.bse[v], p=r.pvalues[v],
                          odds_ratio=np.exp(r.params[v]),
                          unit='has vs not' if v.startswith(('has_', 'type_', 'role_', 'text_', 'is_')) or v == 'multi_call'
                          else 'per doubling of rank' if v == 'log_rank' else 'per SD within list')
                     for v in X]
    fits, coefs = pd.DataFrame(fits), pd.DataFrame(coefs)
    fits.to_csv(RESULTS / 'x06_selection_models.csv', index=False)
    coefs.to_csv(RESULTS / 'x06_selection_coef.csv', index=False)
    print(fits.round(4).to_string(index=False)); print(coefs.round(3).to_string(index=False))

    # equal rank: % cited by within-list match tercile, inside rank bands
    g = g[g.text_missing == 0].copy()
    g['band'] = pd.cut(g['rank'], [0, 3, 10, 999], labels=['rank 1-3', 'rank 4-10', 'rank 11+'])
    g['match_tercile'] = g.groupby('list_id').emb_best_z.transform(
        lambda s: pd.qcut(s.rank(method='first'), 3, labels=['low', 'mid', 'high']) if len(s) >= 3 else 'mid')
    eq = (g.groupby(['band', 'match_tercile'], observed=True).cited.agg(['mean', 'size'])
          .rename(columns={'mean': 'pct_cited', 'size': 'rows'}).reset_index())
    eq['pct_cited'] *= 100
    eq.to_csv(RESULTS / 'x06_equal_rank.csv', index=False)
    print(eq.round(1).to_string(index=False))

    # retrieval: which features go with a higher rank inside a list (list fixed effects)
    R = FLAGS + Z(MATCH + QUERY + PAGE)
    R = [x for x in R if L[x].std() > 0]
    y = -L.log_rank
    Xd = L[R] - L.groupby('list_id')[R].transform('mean')
    yd = y - y.groupby(L.list_id).transform('mean')
    ok = Xd.notna().all(axis=1)
    fit = sm.OLS(yd[ok], Xd[ok]).fit(cov_type='cluster', cov_kwds={'groups': L.loc[ok, 'list_id'].astype('category').cat.codes})
    rc = pd.DataFrame(dict(variable=R, coef=fit.params[R].to_numpy(), se=fit.bse[R].to_numpy(),
                           p=fit.pvalues[R].to_numpy())).assign(r2_within=fit.rsquared, rows=int(ok.sum()))
    rc.to_csv(RESULTS / 'x06_retrieval_coef.csv', index=False)
    print(rc.round(4).to_string(index=False))
    plot(fits, eq)


def plot(fits, eq):
    import plotstyle
    from plotstyle import plt, ENGINE_COLOR, INK2
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    f = fits[fits.model.isin(['S0b rank + text-missing flags', 'S1 + match to the prompt',
                              'S2 + match to the engine queries', 'S3 + page features'])]
    x = np.arange(len(f))
    ax.bar(x, 100 * f.mcfadden_r2, color=['#9ec5f4', ENGINE_COLOR['OPENAI'], ENGINE_COLOR['OPENAI'], ENGINE_COLOR['OPENAI']], width=0.6)
    for xi, v in zip(x, f.mcfadden_r2):
        ax.text(xi, 100 * v + 0.5, f'{100 * v:.1f}', ha='center', fontsize=8, color=INK2)
    ax.set_xticks(x, ['rank (+ text-\nmissing flags)', '+ match to\nquestion', "+ match to engine's\nqueries", '+ page\nfeatures'], fontsize=8)
    ax.set_ylabel('selection explained (McFadden R², %)')
    ax.grid(axis='x', visible=False)
    ax.set_title('Which consulted pages are cited', fontsize=9.5)
    ax = axes[1]
    bands = list(dict.fromkeys(eq.band))
    cols = {'low': '#c3c2b7', 'mid': '#9ec5f4', 'high': ENGINE_COLOR['OPENAI']}
    for k, t in enumerate(['low', 'mid', 'high']):
        v = [eq[(eq.band == b) & (eq.match_tercile == t)].pct_cited.sum() for b in bands]
        ax.bar(np.arange(len(bands)) + (k - 1) * 0.27, v, 0.25, color=cols[t], label=f'{t} match to the question')
    ax.set_xticks(range(len(bands)), bands)
    ax.set_ylabel('% of consulted pages cited')
    ax.legend(fontsize=7.5)
    ax.grid(axis='x', visible=False)
    ax.set_title('At equal rank: does a better-matching page win?', fontsize=9.5)
    name = {'gfiber': 'home internet', 'workorb': 'AEC software'}.get(DATASET, DATASET)
    fig.suptitle(f'X6: selection among retrieved pages (ChatGPT, {name})', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'x06_selection.png')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fetch', 'extras', 'analyse', 'plot'])
    replot = lambda: plot(pd.read_csv(RESULTS / 'x06_selection_models.csv'), pd.read_csv(RESULTS / 'x06_equal_rank.csv'))
    {'fetch': fetch, 'extras': extras, 'analyse': analyse, 'plot': replot}[ap.parse_args().cmd]()
