"""X10 - The model's prior: does what the model already "knows" shape what it searches for and cites?

Probe: every prompt is sent to the model ChatGPT's search uses (gpt-5.4-nano) WITHOUT web search, asking
which brands/products it would mention and which websites it would cite (JSON, up to 10 each; cached in
llm_cache). The answer is the model's prior from pretraining.

Tests (per dataset):
1 queries    tracked brands (prompt_labels.BRANDS) that the question does not name: is a brand added to
             ChatGPT's search queries more often when it is in the prior? Same for Gemini's queries (a
             different model, a different prior) as a comparison. Also: share of queries containing any
             prior brand string not in the question, ChatGPT vs Gemini.
2 X6 rows    prior_site = the page's registered domain is among the prior websites of that prompt.
             Retrieval: -log2 rank on prior_site + X6 features (list fixed effects).
             Selection: conditional logit, rank + X6 features + prior_site.
3 collected  share of each engine's cited (prompt, website) pairs (90 days, weighted by citations) whose
             website is in the prompt's prior, against a shuffled baseline (prior of another prompt).
Gemini: `--model gemini` asks Gemini 3.1 Flash Lite (the model Gemini's grounded search uses) the same
question with no grounding tool; tests 1 and 3 are repeated with Gemini's prior (test 2 needs the
consulted set, which only ChatGPT's API returns). Answers cached in x10_gemini_cache.jsonl.
Outputs: data[/<ds>]/tables/x10_prior[_gemini].csv; results[/<ds>]/x10_queries[_gemini].csv,
         x10_rank_selection.csv, x10_collected[_gemini].csv

  python scripts/x10_prior_probe.py [--model openai|gemini]
"""
import json
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

import llm
from common import TABLES, RESULTS, DATASET, INDUSTRY, SEED, ENGINES, ENGINE_LABEL, read_table, drop_excluded, registered_domain
from prompt_labels import BRANDS

MODEL = 'gpt-5.4-nano'
INSTRUCTIONS = (f'You answer questions about {INDUSTRY} from your own knowledge only; you cannot search the web. '
                'For the user\'s question, list (1) the brands, companies or products you would mention in your '
                'answer, most relevant first, and (2) the websites (domain names such as example.com) you would '
                'cite as sources, most relevant first; up to 10 each. Reply with JSON only: '
                '{"brands": [...], "websites": [...]}')


def ask_gemini_many(prompts):
    import hashlib
    import os
    import threading
    import time
    import requests
    from concurrent.futures import ThreadPoolExecutor
    cache_path = TABLES / 'x10_gemini_cache.jsonl'
    cache, lock = {}, threading.Lock()
    if cache_path.exists():
        with open(cache_path, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line); cache[r['key']] = r['answer']
    model = os.environ.get('GEMINI_MODEL', 'gemini-3.1-flash-lite')

    def one(q):
        k = hashlib.sha1(f'{model}\x00{INSTRUCTIONS}\x00{q}'.encode()).hexdigest()
        if k in cache:
            return cache[k]
        body = {'systemInstruction': {'parts': [{'text': INSTRUCTIONS}]},
                'contents': [{'role': 'user', 'parts': [{'text': q}]}]}          # no tools: no grounding
        for attempt in range(5):
            r = requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
                              headers={'x-goog-api-key': os.environ['GEMINI_API_KEY']}, json=body, timeout=120)
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        cand = (r.json().get('candidates') or [{}])[0]
        a = ''.join(x.get('text', '') for x in (cand.get('content') or {}).get('parts') or [])
        with lock:
            cache[k] = a
            with open(cache_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(dict(key=k, answer=a), ensure_ascii=False) + '\n')
        return a

    with ThreadPoolExecutor(6) as ex:
        return list(ex.map(one, prompts))


def probe(model='openai'):
    lab = pd.read_csv(TABLES / 'prompt_labels.csv', usecols=['prompt_id', 'prompt'])
    if model == 'gemini':
        answers = ask_gemini_many(lab.prompt.tolist())
    else:
        answers = llm.ask_many(INSTRUCTIONS, lab.prompt.tolist(), model=MODEL)
    rows = []
    for pid, p, a in zip(lab.prompt_id, lab.prompt, answers):
        try:
            j = json.loads(a[a.index('{'):a.rindex('}') + 1])
        except ValueError:
            j = {}
        brands = [str(b).strip() for b in j.get('brands') or [] if str(b).strip()]
        sites = []
        for w in j.get('websites') or []:
            w = re.sub(r'^https?://', '', str(w).strip().lower()).split('/')[0].removeprefix('www.')
            if '.' in w:
                sites.append(registered_domain(w))
        rows.append(dict(prompt_id=pid, prompt=p, prior_brands=json.dumps(brands), prior_sites=json.dumps(list(dict.fromkeys(sites)))))
    d = pd.DataFrame(rows)
    d.to_csv(TABLES / f'x10_prior{SUFFIX[model]}.csv', index=False)
    print(f'{DATASET} ({model}): prior for {len(d)} prompts; median brands {d.prior_brands.map(lambda x: len(json.loads(x))).median()}, '
          f'websites {d.prior_sites.map(lambda x: len(json.loads(x))).median()}')
    return d


SUFFIX = {'openai': '', 'gemini': '_gemini'}
SUF = ''


def tracked(text):
    low = text.lower()
    return {b for b, rx in BRANDS.items() if re.search(rx, low)}


def test_queries(prior):
    from x03_analyse import records
    pb = {r.prompt_id: json.loads(r.prior_brands) for r in prior.itertuples()}
    pq = dict(zip(prior.prompt_id, prior.prompt))
    out, share = [], []
    for engine in ('openai', 'gemini'):
        added = {}
        qtexts = {}
        for j in records(engine):
            if j['queries']:
                added.setdefault(j['prompt_id'], set()).update(tracked(' '.join(j['queries'])))
                qtexts.setdefault(j['prompt_id'], []).extend(j['queries'])
        rows = []
        for pid, a in added.items():
            q = pq.get(pid, '')
            in_q = tracked(q)
            prior_t = tracked(' '.join(pb.get(pid, [])))
            for b in set(BRANDS) - in_q:
                rows.append(dict(in_prior=b in prior_t, added=b in a))
        r = pd.DataFrame(rows)
        if len(r):
            p1 = r[r.in_prior].added.mean() if r.in_prior.any() else np.nan
            p0 = r[~r.in_prior].added.mean()
            out.append(dict(engine=ENGINE_LABEL[engine.upper()], pairs=len(r), pct_pairs_in_prior=100 * r.in_prior.mean(),
                            pct_added_if_in_prior=100 * p1, pct_added_if_not_in_prior=100 * p0,
                            ratio=p1 / p0 if p0 else np.nan))
        # any prior brand string (not in the question) appearing in the engine's queries
        hits = []
        for pid, qs in qtexts.items():
            ql, q = ' '.join(qs).lower(), pq.get(pid, '').lower()
            names = [b.lower() for b in pb.get(pid, []) if len(b) > 2 and b.lower() not in q]
            hits.append(np.mean([n in ql for n in names]) if names else np.nan)
        share.append(dict(engine=ENGINE_LABEL[engine.upper()], prompts=len(hits),
                          pct_prior_brands_not_in_question_found_in_queries=100 * np.nanmean(hits)))
    O = pd.DataFrame(out).merge(pd.DataFrame(share), on='engine', how='outer')
    O.to_csv(RESULTS / f'x10_queries{SUF}.csv', index=False)
    print(O.round(2).to_string(index=False))


def test_rank_selection(prior):
    from statsmodels.discrete.conditional_models import ConditionalLogit
    p = TABLES / 'x06_rows.csv'
    if not p.exists():
        return
    L = pd.read_csv(p)
    ps = {r.prompt_id: set(json.loads(r.prior_sites)) for r in prior.itertuples()}
    L['reg'] = L.domain.map(registered_domain)
    L['prior_site'] = [float(rg in ps.get(pid, set())) for pid, rg in zip(L.prompt_id, L.reg)]
    base = ['text_missing', 'is_reddit', 'emb_best_z', 'emb_title_z', 'bm25_z', 'q_best_z', 'log_words_z', 'log_headings_z',
            'has_faq', 'has_summary_box', 'has_author', 'has_schema', 'type_list', 'type_comparison', 'role_brand', 'authority_z']
    base = [b for b in base if b in L and L[b].std() > 0]
    out = [dict(model='share of consulted pages from a prior website', value=100 * L.prior_site.mean(),
                cited=100 * L[L.cited].prior_site.mean(), not_cited=100 * L[~L.cited].prior_site.mean())]
    for name, X in (('retrieval: prior site only', ['prior_site']), ('retrieval: + X6 features', ['prior_site'] + base)):
        y = -L.log_rank
        Xd = L[X] - L.groupby('list_id')[X].transform('mean')
        yd = y - y.groupby(L.list_id).transform('mean')
        f = sm.OLS(yd, Xd).fit(cov_type='cluster', cov_kwds={'groups': L.list_id.astype('category').cat.codes})
        out.append(dict(model=name, coef=f.params['prior_site'], p=f.pvalues['prior_site'],
                        rank_factor=2 ** f.params['prior_site']))
    d = L[L.groupby('list_id').cited.transform('nunique') == 2]
    for name, X in (('selection: rank + prior site', ['log_rank', 'multi_call', 'prior_site']),
                    ('selection: rank + X6 features + prior site', ['log_rank', 'multi_call'] + base + ['prior_site'])):
        m = ConditionalLogit(d.cited.astype(float), d[X], groups=d.list_id)
        r = m.fit(disp=0, method='bfgs', maxiter=1000)
        out.append(dict(model=name, odds_ratio=np.exp(r.params['prior_site']), p=r.pvalues['prior_site'],
                        mcfadden_r2=1 - r.llf / m.loglike(np.zeros(len(X)))))
    O = pd.DataFrame(out)
    O.to_csv(RESULTS / 'x10_rank_selection.csv', index=False)
    print(O.round(4).to_string(index=False))


def test_collected(prior):
    c = drop_excluded(read_table('citations.csv', usecols=['date', 'engine', 'prompt_id', 'domain', 'repeat_in_answer']))
    c = c[~c.repeat_in_answer & c.domain.notna()]
    c['reg'] = c.domain.map(registered_domain)
    ps = {r.prompt_id: set(json.loads(r.prior_sites)) for r in prior.itertuples()}
    pids = sorted(ps)
    rng = np.random.default_rng(SEED)
    shuffled = dict(zip(pids, rng.permutation(pids)))       # prior of another prompt
    out = []
    for e in ENGINES:
        g = c[c.engine == e]
        if g.empty:
            continue
        own = np.array([rg in ps.get(pid, set()) for pid, rg in zip(g.prompt_id, g.reg)])
        other = np.array([rg in ps.get(shuffled.get(pid), set()) for pid, rg in zip(g.prompt_id, g.reg)])
        pairs = g.drop_duplicates(['prompt_id', 'reg'])
        own_p = np.array([rg in ps.get(pid, set()) for pid, rg in zip(pairs.prompt_id, pairs.reg)])
        out.append(dict(engine=ENGINE_LABEL[e], citations=len(g), pct_citations_from_prior_site=100 * own.mean(),
                        pct_with_other_prompts_prior=100 * other.mean(), ratio=own.mean() / other.mean() if other.mean() else np.nan,
                        pct_websites_in_prior=100 * own_p.mean()))
    O = pd.DataFrame(out)
    O.to_csv(RESULTS / f'x10_collected{SUF}.csv', index=False)
    print(O.round(2).to_string(index=False))


def main():
    import argparse
    global SUF
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['openai', 'gemini'], default='openai')
    m = ap.parse_args().model
    SUF = SUFFIX[m]
    prior = probe(m)
    ex = prior.head(3)
    for r in ex.itertuples():
        print(f'  Q: {r.prompt}\n     brands: {json.loads(r.prior_brands)[:6]}\n     websites: {json.loads(r.prior_sites)[:6]}')
    test_queries(prior)
    if m == 'openai':
        test_rank_selection(prior)
    test_collected(prior)


if __name__ == '__main__':
    main()
