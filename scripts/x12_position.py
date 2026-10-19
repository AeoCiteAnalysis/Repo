"""X12 - Position vs evidence: does a source get cited because it is better or because it comes first?

X6 found a steep association between a page's position in ChatGPT's result list and being cited
(47 % at rank 1, 3 % at ranks 11-20). Naturally occurring positions may carry better content. Here the
evidence set is held fixed and only its order changes.

build   For each of 80 GFiber + 20 Workorb questions (seed 42): one replay result list (X6 rows) with at
        least 8 consulted pages that have text; the 8 highest-ranked of them become the sources. Excerpt
        per source = title + the 3 paragraphs most similar to the question (all-MiniLM-L6-v2), in page
        order, each cut to 90 words. Every source gets a fixed random 3-letter ID (no order information).
        Stored in data/x12/packets.json (identical in every call).
run     For every question: 8 cyclic rotations (every source in every position exactly once) x 3 label
        conditions x 2 repeats x 2 models (gpt-5.4-nano, gemini-3.1-flash-lite; no web search):
          none         no rank labels
          aligned      "Search rank: k" equal to the position
          independent  rank labels shuffled independently of the position (seeded per call)
        The model is asked to answer from the sources only and to cite them by ID. Answers cached in
        data/x12/answers.jsonl (resumable).
analyse Citation of each source in each call; brands named in each answer (prompt_labels.BRANDS) and their
        order of first mention.
          position curve         P(cited | position) per model (condition none)
          position effect        per source: P(cited | first) - P(cited | last), mean over sources,
                                 bootstrap CI over questions
          net of content         linear probability model with source fixed effects: cited ~ position
                                 (+ label in 'independent'), clustered by question
          label vs placement     'independent' condition: position and label effects together
          original rank          does the replay's original rank predict citation once position is
                                 randomized (a proxy for content)
          fidelity               condition none, original order: share of the replay's cited sources
                                 (among the 8) cited in the simulation vs the other sources
          recommendation         per question and model: stability of the named-brand set across
                                 rotations; does a brand move up when a source naming it moves up
Outputs: results/x12_*.csv, figures/x12_position.png

  python scripts/x12_position.py build | run | analyse
"""
import argparse
import hashlib
import json
import os
import random
import re
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'x12'
PACKETS = OUT / 'packets.json'
ANSWERS = OUT / 'answers.jsonl'
RESULTS = ROOT / 'results'
N_SOURCES, N_Q = 8, {'gfiber': 80, 'workorb': 20}
CONDITIONS = ('none', 'aligned', 'independent')
REPEATS = 2
MODELS = {'openai': 'gpt-5.4-nano', 'gemini': 'gemini-3.1-flash-lite'}
INSTRUCTIONS = ('Answer the user\'s question using only the sources provided below. Cite every source you '
                'rely on with its ID in square brackets, for example [ABC]. Do not use any other knowledge.')
SEED = 42


def dataset_paths(ds):
    t = ROOT / 'data' / ('tables' if ds == 'gfiber' else 'workorb/tables')
    return t / 'x06_rows.csv', t / 'x06_parsed.jsonl', t / 'prompt_labels.csv'


def build():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('all-MiniLM-L6-v2')
    enc = lambda xs: m.encode(xs, normalize_embeddings=True, show_progress_bar=False)
    rng = random.Random(SEED)
    used_ids, packets = set(), []
    for ds, n in N_Q.items():
        rows_p, parsed_p, lab_p = dataset_paths(ds)
        rows = pd.read_csv(rows_p)
        pages = {}
        with open(parsed_p, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line); pages[r['url']] = r
        prompts = dict(pd.read_csv(lab_p, usecols=['prompt_id', 'prompt']).values)
        rows = rows[(rows.text_missing == 0) & rows.url.isin(pages)]
        rows = rows.sort_values(['list_id', 'rank'])
        ok = rows.groupby('list_id').size()
        lists = ok[ok >= N_SOURCES].index.tolist()
        # one list per question
        per_q = {}
        for lid in lists:
            per_q.setdefault(lid.split('|')[0], lid)
        qs = sorted(per_q)
        pick = random.Random(SEED).sample(qs, min(n, len(qs)))
        for pid in sorted(pick):
            lid = per_q[pid]
            g = rows[rows.list_id == lid].head(N_SOURCES)
            q = prompts[pid]
            qv = enc([q])[0]
            sources = []
            for r in g.itertuples():
                p = pages[r.url]
                paras = [x for x in p['paragraphs'] if len(x.split()) >= 8][:50] or p['paragraphs'][:3]
                sims = enc(paras) @ qv if paras else np.array([])
                top = sorted(np.argsort(-sims)[:3]) if len(paras) else []
                excerpt = ' '.join(' '.join(paras[i].split()[:90]) for i in top)
                while True:
                    sid = ''.join(rng.choice(string.ascii_uppercase) for _ in range(3))
                    if sid not in used_ids:
                        used_ids.add(sid); break
                sources.append(dict(id=sid, url=r.url, domain=r.domain, title=p.get('title') or '',
                                    excerpt=excerpt, orig_rank=int(r.rank), replay_cited=bool(r.cited)))
            packets.append(dict(dataset=ds, prompt_id=pid, list_id=lid, question=q, sources=sources))
        print(f'{ds}: {len(pick)} questions from {len(qs)} with >= {N_SOURCES} sources with text')
    OUT.mkdir(parents=True, exist_ok=True)
    PACKETS.write_text(json.dumps(packets, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{len(packets)} packets -> {PACKETS}')


def order_and_labels(k, rot, cond, rep):
    order = [(i + rot) % k for i in range(k)]                     # position p shows source order[p]
    if cond == 'none':
        labels = None
    elif cond == 'aligned':
        labels = list(range(1, k + 1))
    else:
        r = random.Random(f'{rot}-{rep}-labels')
        labels = list(range(1, k + 1)); r.shuffle(labels)
    return order, labels


def render(pk, order, labels):
    parts = [f'Question: {pk["question"]}', '', 'Sources:']
    for p, si in enumerate(order):
        s = pk['sources'][si]
        head = f'[{s["id"]}] {s["domain"]} - {s["title"]}'
        if labels:
            head += f' (search rank: {labels[p]})'
        parts += [head, s['excerpt'], '']
    return '\n'.join(parts)


def key(model, text, rep):
    return hashlib.sha1(f'{model}\x00{INSTRUCTIONS}\x00{text}\x00{rep}'.encode()).hexdigest()


def ask_openai(text):
    from openai import OpenAI
    r = OpenAI().responses.create(model=MODELS['openai'], instructions=INSTRUCTIONS, input=text)
    return r.output_text


def ask_gemini(text):
    import requests
    body = {'systemInstruction': {'parts': [{'text': INSTRUCTIONS}]},
            'contents': [{'role': 'user', 'parts': [{'text': text}]}]}
    for attempt in range(5):
        r = requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/{MODELS["gemini"]}:generateContent',
                          headers={'x-goog-api-key': os.environ['GEMINI_API_KEY']}, json=body, timeout=120)
        if r.status_code == 200:
            break
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()
    cand = (r.json().get('candidates') or [{}])[0]
    return ''.join(x.get('text', '') for x in (cand.get('content') or {}).get('parts') or [])


def run():
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    import common                                                  # loads .env
    packets = json.loads(PACKETS.read_text(encoding='utf-8'))
    done = set()
    if ANSWERS.exists():
        with open(ANSWERS, encoding='utf-8') as f:
            for line in f:
                done.add(json.loads(line)['key'])
    jobs = []
    for qi, pk in enumerate(packets):
        k = len(pk['sources'])
        for rot in range(k):
            for cond in CONDITIONS:
                for rep in range(REPEATS):
                    order, labels = order_and_labels(k, rot, cond, rep)
                    text = render(pk, order, labels)
                    for eng, model in MODELS.items():
                        kk = key(model, text, rep)
                        if kk not in done:
                            jobs.append(dict(key=kk, engine=eng, qi=qi, rot=rot, cond=cond, rep=rep,
                                             order=order, labels=labels, text=text))
    import os
    if int(os.environ.get('X12_LIMIT', 0)):                        # pilot: first N calls only
        jobs = jobs[:int(os.environ['X12_LIMIT'])]
    print(f'{len(jobs)} calls to make', flush=True)
    lock, n = threading.Lock(), [0]

    def one(j):
        try:
            a = (ask_openai if j['engine'] == 'openai' else ask_gemini)(j['text'])
        except Exception as e:
            print(f'  {j["engine"]} q{j["qi"]}: {type(e).__name__}: {str(e)[:100]}', flush=True)
            return
        rec = {k: v for k, v in j.items() if k != 'text'} | dict(answer=a)
        with lock:
            with open(ANSWERS, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n[0] += 1
            if n[0] % 250 == 0:
                print(f'  {n[0]}/{len(jobs)}', flush=True)

    for eng, workers in (('openai', 8), ('gemini', 6)):
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, [j for j in jobs if j['engine'] == eng]))
    print('done', flush=True)


def long_table():
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    packets = json.loads(PACKETS.read_text(encoding='utf-8'))
    rows = []
    with open(ANSWERS, encoding='utf-8') as f:
        for line in f:
            a = json.loads(line)
            pk = packets[a['qi']]
            cited = {i for g in re.findall(r'\[([A-Z]{3}(?:\s*,\s*[A-Z]{3})*)\]', a['answer'])   # also [ABC, DEF]
                     for i in re.findall(r'[A-Z]{3}', g)}
            for p, si in enumerate(a['order']):
                s = pk['sources'][si]
                rows.append(dict(engine=a['engine'], dataset=pk['dataset'], qi=a['qi'], rot=a['rot'], cond=a['cond'],
                                 rep=a['rep'], call=a['key'], source=f"{a['qi']}:{si}", position=p + 1,
                                 label=(a['labels'][p] if a['labels'] else np.nan), orig_rank=s['orig_rank'],
                                 orig_order=si + 1, replay_cited=s['replay_cited'], cited=s['id'] in cited))
    return pd.DataFrame(rows)


def fe_lpm(d, X):
    """Linear probability model with source fixed effects (demeaned), SE clustered by question."""
    import statsmodels.api as sm
    Y = d.cited.astype(float)
    Z = pd.get_dummies(d[X].astype(str) if len(X) == 1 else d[X].astype(str), drop_first=True, dtype=float)
    Zd = Z - Z.groupby(d.source.to_numpy()).transform('mean')
    yd = Y - Y.groupby(d.source.to_numpy()).transform('mean')
    return sm.OLS(yd, Zd).fit(cov_type='cluster', cov_kwds={'groups': d.qi.to_numpy()})


def analyse():
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    d = long_table()
    print(f'{d.call.nunique():,} calls, {len(d):,} source rows')
    out = []
    rng = np.random.default_rng(SEED)
    for eng, g in d.groupby('engine'):
        n0 = g[g.cond == 'none']
        curve = n0.groupby('position').cited.mean()
        for p, v in curve.items():
            out.append(dict(engine=eng, measure='position curve (no labels)', key=p, value=100 * v))
        # per-source first vs last
        per = n0.groupby(['source', 'qi', 'position']).cited.mean().unstack('position')
        eff = (per[1] - per[N_SOURCES]).dropna()
        qs = eff.index.get_level_values('qi').unique().to_numpy()
        byq = eff.groupby(level='qi').mean()
        boot = [byq.reindex(rng.choice(qs, len(qs))).mean() for _ in range(1000)]
        out.append(dict(engine=eng, measure='position effect first - last (pp)', key='mean over sources',
                        value=100 * eff.mean(), ci_low=100 * np.percentile(boot, 2.5), ci_high=100 * np.percentile(boot, 97.5)))
        f = fe_lpm(n0, ['position'])
        for k_, v in f.params.items():
            out.append(dict(engine=eng, measure='net of source (LPM, vs position 1, pp)', key=k_, value=100 * v,
                            ci_low=100 * (v - 1.96 * f.bse[k_]), ci_high=100 * (v + 1.96 * f.bse[k_])))
        ind = g[g.cond == 'independent'].copy()
        ind['early'] = (ind.position <= 2).astype(float)
        ind['top_label'] = (ind.label <= 2).astype(float)
        import statsmodels.api as sm
        Z = ind[['early', 'top_label']]
        Zd = Z - Z.groupby(ind.source.to_numpy()).transform('mean')
        yd = ind.cited.astype(float) - ind.cited.astype(float).groupby(ind.source.to_numpy()).transform('mean')
        fl = sm.OLS(yd, Zd).fit(cov_type='cluster', cov_kwds={'groups': ind.qi.to_numpy()})
        for v in ('early', 'top_label'):
            out.append(dict(engine=eng, measure='independent labels: position 1-2 vs label 1-2 (pp)', key=v,
                            value=100 * fl.params[v], ci_low=100 * (fl.params[v] - 1.96 * fl.bse[v]),
                            ci_high=100 * (fl.params[v] + 1.96 * fl.bse[v]), p=fl.pvalues[v]))
        al = g[g.cond == 'aligned'].groupby('position').cited.mean()
        out.append(dict(engine=eng, measure='aligned labels: first - last (pp)', key='', value=100 * (al[1] - al[N_SOURCES])))
        # original replay rank, once position is randomized
        rk = n0.groupby('orig_order').cited.mean()
        for k_, v in rk.items():
            out.append(dict(engine=eng, measure='cited by original replay order (position randomized)', key=k_, value=100 * v))
        # fidelity: original order (rotation 0)
        o = n0[n0.rot == 0]
        out.append(dict(engine=eng, measure='fidelity: cited rate, replay-cited sources', key='', value=100 * o[o.replay_cited].cited.mean()))
        out.append(dict(engine=eng, measure='fidelity: cited rate, other sources', key='', value=100 * o[~o.replay_cited].cited.mean()))
    R = pd.DataFrame(out)
    R.to_csv(RESULTS / 'x12_position.csv', index=False)
    pd.set_option('display.width', 200)
    print(R.round(2).to_string(index=False))
    recommendation(d)


def recommendation(d):
    from prompt_labels import BRANDS_BY_DATASET
    packets = json.loads(PACKETS.read_text(encoding='utf-8'))
    ans = {}
    with open(ANSWERS, encoding='utf-8') as f:
        for line in f:
            a = json.loads(line); ans[a['key']] = a
    rows, move = [], []
    for k_, a in ans.items():
        pk = packets[a['qi']]
        brands = BRANDS_BY_DATASET[pk['dataset']]
        text = a['answer'].lower()
        first = {b: m.start() for b, rx in brands.items() for m in [re.search(rx, text)] if m}
        order = sorted(first, key=first.get)
        rows.append(dict(engine=a['engine'], qi=a['qi'], cond=a['cond'], rot=a['rot'], rep=a['rep'],
                         brands=frozenset(order), first_brand=order[0] if order else None))
        # brand rank vs the position of the sources that name it
        for b in order:
            pos = [p + 1 for p, si in enumerate(a['order'])
                   if re.search(brands[b], (pk['sources'][si]['title'] + ' ' + pk['sources'][si]['excerpt']).lower())]
            if pos:
                move.append(dict(engine=a['engine'], qi=a['qi'], brand=b, cond=a['cond'], brand_rank=order.index(b) + 1,
                                 n_brands=len(order), best_source_position=min(pos)))
    r = pd.DataFrame(rows)
    out = []
    for (eng, qi), g in r[r.cond == 'none'].groupby(['engine', 'qi']):
        sets = g.brands.tolist()
        jac = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:] if a | b]
        firsts = g.first_brand.dropna()
        out.append(dict(engine=eng, qi=qi, brand_set_jaccard=np.mean(jac) if jac else np.nan,
                        same_first_brand=(firsts.value_counts().iloc[0] / len(firsts)) if len(firsts) else np.nan))
    S = pd.DataFrame(out).groupby('engine')[['brand_set_jaccard', 'same_first_brand']].mean().reset_index()
    m = pd.DataFrame(move)
    if len(m):
        m = m[(m.cond == 'none') & (m.n_brands > 1)]
        m['rank_share'] = (m.brand_rank - 1) / (m.n_brands - 1)          # 0 = named first
        c = m.groupby('engine').apply(lambda x: x.rank_share.corr(x.best_source_position, method='spearman'))
        S['brand_rank_vs_source_position_spearman'] = S.engine.map(c)
    S.to_csv(RESULTS / 'x12_recommendation.csv', index=False)
    print(S.round(3).to_string(index=False))


def plot():
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    from common import FIGURES
    import plotstyle  # noqa: F401
    from plotstyle import plt, ENGINE_COLOR, INK2
    d = long_table()
    n0 = d[d.cond == 'none']
    by = n0.groupby(['dataset', 'engine', 'position']).cited.mean().unstack('position') * 100
    by['first_minus_last'] = by[1] - by[8]
    by.round(1).to_csv(RESULTS / 'x12_position_by_market.csv')
    print(by.round(1).to_string())
    name = {'openai': 'ChatGPT (gpt-5.4-nano)', 'gemini': 'Gemini (3.1-flash-lite)'}
    col = {'openai': ENGINE_COLOR['OPENAI'], 'gemini': ENGINE_COLOR['GEMINI']}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (x, title, xl) in zip(axes, [('position', 'Same sources, shown in a different slot',
                                          'slot in the list shown to the model'),
                                         ('orig_order', "Engine's original ranking, slot randomised",
                                          "rank in the engine's replay list (1 = top)")]):
        for eng in ('openai', 'gemini'):
            c = n0[n0.engine == eng].groupby(x).cited.mean() * 100
            ax.plot(c.index, c.values, marker='o', lw=2, ms=8, color=col[eng], label=name[eng])
            ax.text(8.15, c.values[-1], f'{c.values[-1]:.0f}%', va='center', fontsize=8, color=INK2)
            ax.text(0.85, c.values[0], f'{c.values[0]:.0f}%', va='center', ha='right', fontsize=8, color=INK2)
        ax.set_xticks(range(1, 9))
        ax.set_xlim(0.4, 8.6)
        ax.set_xlabel(xl)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('% of answers citing the source')
    axes[0].set_ylim(35, 72)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('X12: does the order of the sources change what is cited?', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'x12_position.png')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'run', 'analyse', 'plot'])
    {'build': build, 'run': run, 'analyse': analyse, 'plot': plot}[ap.parse_args().cmd]()
