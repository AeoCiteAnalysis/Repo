"""X3 (E6) - Replay the collection through the engines' APIs, logging retrieval.

The pfetch collection called the engines' APIs with web search on (no system prompt, default
temperature): ChatGPT = gpt-5.4-nano with the web_search tool, Gemini = Gemini 3.1 Flash Lite
with Google Search grounding. It stored only the answer text. This replay sends the same 651
questions the same way and keeps what the collection discarded:

  OpenAI   the search queries it issued and EVERY source it consulted
           (include=web_search_call.action.sources), plus the cited URLs (annotations)
  Gemini   its Google search queries (groundingMetadata.webSearchQueries), the grounding chunks
           (redirect links, resolved to real URLs) and which chunks support the answer

One file per engine / day / prompt / repeat: data/replay/<engine>/<YYYYMMDD>/<prompt_id>_r<rep>.json
with a common layout: queries, consulted, cited, answer, usage. Already-fetched files are skipped,
so a run can be resumed.

  python scripts/x03_replay.py fetch --engine openai --n 5          pilot (random prompts, seed 42)
  python scripts/x03_replay.py fetch --engine openai                all 651 prompts, repeat 1, today
  python scripts/x03_replay.py fetch --engine gemini --rep 2 --n 100   same-day repeat (E10)
  python scripts/x03_replay.py models                                  list Gemini models
Keys: OPENAI_API_KEY, GEMINI_API_KEY (and GEMINI_MODEL) in .env.
"""
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import requests

from common import ROOT, TABLES, SEED, clean_url, is_redirect

OUT = TABLES.parent / 'replay'            # data/replay (gfiber) or data/<dataset>/replay
OPENAI_MODEL = 'gpt-5.4-nano'
GEMINI_API = 'https://generativelanguage.googleapis.com/v1beta'
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36'
_redirects, _rlock = {}, threading.Lock()


def gemini_model():
    return os.environ.get('GEMINI_MODEL', 'gemini-3.1-flash-lite')


def resolve(u):
    """Follow one Vertex AI Search redirect link to the real URL (cached per run)."""
    if not is_redirect(u):
        return u
    with _rlock:
        if u in _redirects:
            return _redirects[u]
    real = None
    for _ in range(2):
        try:
            r = requests.get(u, headers={'User-Agent': UA}, timeout=20, allow_redirects=False)
            real = r.headers.get('Location')
            break
        except requests.RequestException:
            time.sleep(2)
    with _rlock:
        _redirects[u] = real or u
    return real or u


def ask_openai(prompt):
    from openai import OpenAI
    r = OpenAI().responses.create(model=OPENAI_MODEL, input=prompt, tools=[{'type': 'web_search'}],
                                  include=['web_search_call.action.sources'])
    d = r.model_dump()
    queries, consulted, opened, cited, calls, answer = [], [], [], [], [], ''
    for o in d['output']:
        if o['type'] == 'web_search_call':
            a = o.get('action') or {}
            if a.get('type') == 'search':
                q = a.get('queries') or ([a['query']] if a.get('query') else [])
                src = [s['url'] for s in a.get('sources') or [] if s.get('url')]
                queries += q
                consulted += src
                calls.append(dict(queries=q, sources=src))     # each call's own ranked source list
            elif a.get('url'):                      # open_page / find_in_page
                opened.append(a['url'])
        elif o['type'] == 'message':
            for c in o['content']:
                answer += c.get('text') or ''
                cited += [x['url'] for x in c.get('annotations') or [] if x.get('type') == 'url_citation']
    return dict(model=OPENAI_MODEL, queries=queries, consulted=consulted, opened=opened, cited=cited,
                calls=calls, answer=answer, usage=d.get('usage'))


def ask_gemini(prompt):
    k = os.environ['GEMINI_API_KEY']
    body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}], 'tools': [{'google_search': {}}]}
    for attempt in range(4):
        r = requests.post(f'{GEMINI_API}/models/{gemini_model()}:generateContent', headers={'x-goog-api-key': k},
                          json=body, timeout=120)
        if r.status_code not in (429, 500, 503):
            break
        time.sleep(10 * (attempt + 1))
    r.raise_for_status()
    d = r.json()
    cand = (d.get('candidates') or [{}])[0]
    g = cand.get('groundingMetadata') or {}
    chunks = [(c.get('web') or {}) for c in g.get('groundingChunks') or []]
    urls = [resolve(c.get('uri', '')) for c in chunks]
    used = sorted({i for s in g.get('groundingSupports') or [] for i in s.get('groundingChunkIndices') or []})
    answer = ''.join(p.get('text', '') for p in (cand.get('content') or {}).get('parts') or [])
    return dict(model=gemini_model(), queries=g.get('webSearchQueries') or [], consulted=urls,
                consulted_titles=[c.get('title') for c in chunks], opened=[],
                cited=[urls[i] for i in used if i < len(urls)], answer=answer, usage=d.get('usageMetadata'))


ASK = {'openai': ask_openai, 'gemini': ask_gemini}


def fetch(engine, n, rep, workers):
    day = f'{date.today():%Y%m%d}'
    out = OUT / engine / day
    out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(TABLES / 'prompt_labels.csv', usecols=['prompt_id', 'prompt'])
    pick = lab.sample(len(lab), random_state=SEED).head(n or len(lab))
    todo = [r for r in pick.itertuples() if not (out / f'{r.prompt_id}_r{rep}.json').exists()]
    print(f'{engine}: {len(todo)} prompts to fetch into {out}')
    done, fails, lock = [0], [0], threading.Lock()

    def one(r):
        t0 = time.time()
        try:
            res = ASK[engine](r.prompt)
        except Exception as e:
            with lock:
                fails[0] += 1
            print(f'  {r.prompt_id}: {type(e).__name__}: {str(e)[:150]}')
            return
        res.update(prompt_id=r.prompt_id, prompt=r.prompt, engine=engine, date=day, rep=rep,
                   seconds=round(time.time() - t0, 1))
        (out / f'{r.prompt_id}_r{rep}.json').write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
        with lock:
            done[0] += 1
            if done[0] % 25 == 0 or n and n <= 10:
                print(f'  {done[0]}/{len(todo)}  {r.prompt_id}: {len(res["queries"])} queries, '
                      f'{len(set(res["consulted"]))} consulted, {len(set(res["cited"]))} cited')

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print(f'done {done[0]}, failed {fails[0]}')


def load(engine=None):
    """All replay records as a long table: one row per (record, url, stage)."""
    rows, recs = [], []
    for f in sorted(OUT.glob(f'{engine or "*"}/*/*.json')):
        j = json.loads(f.read_text(encoding='utf-8'))
        rec = dict(engine=j['engine'], date=j['date'], rep=j['rep'], prompt_id=j['prompt_id'])
        recs.append(dict(rec, n_queries=len(j['queries']), queries=j['queries']))
        cited = {clean_url(u) for u in j['cited']}
        for u in dict.fromkeys(clean_url(x) for x in j['consulted'] + j.get('opened', []) + j['cited']):
            rows.append(dict(rec, url=u, cited=u in cited))
    return pd.DataFrame(recs), pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fetch', 'models'])
    ap.add_argument('--engine', choices=list(ASK), default='openai')
    ap.add_argument('--n', type=int, default=0, help='number of prompts (0 = all 651)')
    ap.add_argument('--rep', type=int, default=1)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    if a.cmd == 'models':
        r = requests.get(f'{GEMINI_API}/models', params={'pageSize': 200}, headers={'x-goog-api-key': os.environ['GEMINI_API_KEY']}).json()
        print('\n'.join(m['name'] for m in r.get('models', []) if 'flash' in m['name'] or 'pro' in m['name']))
    else:
        fetch(a.engine, a.n, a.rep, a.workers)


if __name__ == '__main__':
    main()
