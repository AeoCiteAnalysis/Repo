"""X15 - controlled ranking stability: can a model's choices be represented by one stable score per brand?

Evidence is fixed: one question and 6 brands, each described by its fixed fact sheet (x14_fact_sheets.py),
shown under the real brand names. For every set:

  pairs      all 15 pairs, both orders, 3 repeats                       -> consistency, order effect, cycles
  full       ranking of all 6, 6 cyclic orders, 2 repeats               -> reference ranking, repeat noise
  drop       each brand removed (5 left), with and without a neutral     -> context dependence; length control
             filler of the same length in its place, 2 repeats
  add        a 7th brand added (3 different ones), 2 repeats            -> context dependence
  para       3 equivalent rewordings of the question, 6 cyclic orders   -> paraphrase vs identical-repeat change

A Bradley-Terry model fitted on the pairwise answers is tested on held-out pairs and on the full, drop and
add rankings; a pair-by-pair majority (no single score) and the repeat-to-repeat agreement are the references.

  python scripts/x15_stability.py build     # sets, paraphrases (review data/x15/paraphrases.md), fillers
  python scripts/x15_stability.py run       # answers.jsonl (resumable; X15_LIMIT=n for a pilot)
  python scripts/x15_stability.py analyse
"""
import argparse
import hashlib
import itertools
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import x14_fact_sheets as fs  # noqa: E402

OUT = ROOT / 'data' / 'x15'
RESULTS = ROOT / 'results'
MODELS = fs.MODELS
SEED = 42
N_SETS = {'gfiber': 12, 'workorb': 8}
K = 6
PAIR_REPS, FULL_REPS, DROP_REPS, ADD_REPS = 3, 2, 2, 2

INSTRUCTIONS = 'You help a user choose a {kind}. Use only the information provided in the message.'
ASK_RANK = ('Rank all {k} products for the user\'s question, best first. Answer with JSON only: '
            '{{"ranking": ["<product>", ...]}}. Write each product exactly as it is named above.')
ASK_PAIR = ('Which of the two products is the better choice for the user\'s question? You must choose one; if neither is clearly better, pick the more suitable one. '
            'Answer with JSON only: {"choice": "<product>"}. Write the product exactly as it is named above.')

PARA = """Rewrite the question below in {n} different ways that ask exactly the same thing: same requirement,
same scope, no added or dropped conditions, natural wording a real user might type. Return JSON only:
{{"paraphrases": ["...", ...]}}"""
FILLER = """Write a short neutral note titled "General buying considerations" for someone choosing a {kind}:
seven bullet points of generic advice, {lo}-{hi} words in total. Name no company, product, price or number,
and do not recommend any provider. Return the bullets only, one per line, each starting with "- "."""


# ---------------------------------------------------------------- build

def build():
    import common  # noqa: F401
    import llm
    OUT.mkdir(parents=True, exist_ok=True)
    qs = json.loads((ROOT / 'data' / 'x14' / 'questions.json').read_text(encoding='utf-8'))
    sheets = json.loads((ROOT / 'data' / 'x14' / 'sheets.json').read_text(encoding='utf-8'))
    rng = random.Random(SEED)
    sets = []
    for ds, m in fs.MARKETS.items():
        dq = [q for q in qs if q['dataset'] == ds]
        rng.shuffle(dq)
        appear = {b: 0 for b in m['brands']}
        for q in dq[:N_SETS[ds]]:
            chosen = sorted(m['brands'], key=lambda b: (appear[b], rng.random()))[:K]
            for b in chosen:
                appear[b] += 1
            rng.shuffle(chosen)
            extra = [b for b in m['brands'] if b not in chosen]
            rng.shuffle(extra)
            raw = llm.ask(PARA.format(n=3), q['question'], model=fs.EXTRACT_MODEL)
            paras = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))['paraphrases'][:3]
            sets.append(dict(sid=len(sets), dataset=ds, question=q['question'], paraphrases=paras,
                             brands=chosen, extra=extra[:3]))
    fillers = {}
    for ds, m in fs.MARKETS.items():
        mean_words = int(np.mean([sheets[b]['words'] for b in m['brands']]))
        fillers[ds] = llm.ask(FILLER.format(kind=m['kind'], lo=mean_words - 10, hi=mean_words + 10), 'go',
                              model=fs.EXTRACT_MODEL).strip()
        print(f'{ds} filler: {len(fillers[ds].split())} words (sheets average {mean_words})')
    (OUT / 'sets.json').write_text(json.dumps(dict(sets=sets, fillers=fillers), indent=1, ensure_ascii=False),
                                   encoding='utf-8')
    with open(OUT / 'paraphrases.md', 'w', encoding='utf-8') as f:
        for s in sets:
            f.write(f'**{s["sid"]}** {s["question"]}\n' + ''.join(f'- {p}\n' for p in s['paraphrases']) + '\n')
        for ds, t in fillers.items():
            f.write(f'## filler {ds}\n{t}\n\n')
    print(f'{len(sets)} sets -> {OUT / "sets.json"}; review {OUT / "paraphrases.md"}')


# ---------------------------------------------------------------- prompts

def render(ds, question, blocks, ask):
    """blocks: list of brand names or the string '<filler>'."""
    sheets = render.sheets
    m = fs.MARKETS[ds]
    lines = [f'Question: {question}', '', 'Product profiles:']
    for b in blocks:
        if b == '<filler>':
            lines += ['', '[General buying considerations - not a product]', render.fillers[ds]]
        else:
            lines += ['', f'[{b}]', fs.render_sheet(sheets[b]['sheet'], m['fields'], b)]
    lines += ['', ask]
    return '\n'.join(lines)


def rot(xs, r):
    return xs[r:] + xs[:r]


def jobs_all():
    d = json.loads((OUT / 'sets.json').read_text(encoding='utf-8'))
    render.sheets = json.loads((ROOT / 'data' / 'x14' / 'sheets.json').read_text(encoding='utf-8'))
    render.fillers = d['fillers']
    jobs = []
    for s in d['sets']:
        ds, q, B = s['dataset'], s['question'], s['brands']
        ins = INSTRUCTIONS.format(kind=fs.MARKETS[ds]['kind'])
        base = dict(sid=s['sid'], dataset=ds, ins=ins)
        for a, b in itertools.combinations(B, 2):
            for first, second in ((a, b), (b, a)):
                for rep in range(PAIR_REPS):
                    jobs.append(base | dict(part='pair', items=[first, second], rep=rep, qv=0,
                                            text=render(ds, q, [first, second], ASK_PAIR)))
        for r in range(K):
            for rep in range(FULL_REPS):
                jobs.append(base | dict(part='full', items=rot(B, r), rot=r, rep=rep, qv=0,
                                        text=render(ds, q, rot(B, r), ASK_RANK.format(k=K))))
        for i, drop in enumerate(B):
            for filler in (False, True):
                blocks = [b if b != drop else '<filler>' for b in B] if filler else [b for b in B if b != drop]
                blocks = rot(blocks, i % len(blocks))       # vary the order across the drop-one sets
                items = [b for b in blocks if b != '<filler>']
                for rep in range(DROP_REPS):
                    jobs.append(base | dict(part='drop', dropped=drop, filler=filler, items=items, rep=rep, qv=0,
                                            text=render(ds, q, blocks, ASK_RANK.format(k=K - 1))))
        for j, add in enumerate(s['extra']):
            blocks = B[:]
            blocks.insert((2 * j + 1) % (K + 1), add)
            for rep in range(ADD_REPS):
                jobs.append(base | dict(part='add', added=add, items=blocks, rep=rep, qv=0,
                                        text=render(ds, q, blocks, ASK_RANK.format(k=K + 1))))
        for v, pq in enumerate(s['paraphrases'], start=1):
            for r in range(K):
                jobs.append(base | dict(part='para', items=rot(B, r), rot=r, rep=0, qv=v,
                                        text=render(ds, pq, rot(B, r), ASK_RANK.format(k=K))))
    return d, jobs


# ---------------------------------------------------------------- run

def run():
    import common  # noqa: F401
    _, jobs = jobs_all()
    path = OUT / 'answers.jsonl'
    done = set()
    if path.exists():
        done = {json.loads(l)['key'] for l in open(path, encoding='utf-8')}
    todo = []
    for j in jobs:
        for eng, model in MODELS.items():
            k = hashlib.sha1(f'{model}|{j["ins"]}|{j["text"]}|{j["rep"]}'.encode()).hexdigest()
            if k not in done:
                todo.append(j | dict(engine=eng, key=k))
    if int(os.environ.get('X15_LIMIT', 0)):
        todo = random.Random(1).sample(todo, int(os.environ['X15_LIMIT']))
    print(f'{len(todo)} calls to make', flush=True)
    lock, n = threading.Lock(), [0]

    def one(j):
        try:
            a = fs.ask(j['engine'], j['ins'], j['text'])
        except Exception as e:
            print(f'  {j["engine"]} {j["part"]} s{j["sid"]}: {type(e).__name__}: {str(e)[:100]}', flush=True)
            return
        rec = {k: v for k, v in j.items() if k not in ('ins', 'text')} | dict(answer=a)
        with lock:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n[0] += 1
            if n[0] % 500 == 0:
                print(f'  {n[0]}/{len(todo)}', flush=True)

    for eng, workers in (('openai', 8), ('gemini', 6)):
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, [j for j in todo if j['engine'] == eng]))
    print('done', flush=True)


# ---------------------------------------------------------------- analyse

def parse(a):
    try:
        return json.loads(re.search(r'\{.*\}', a['answer'], re.S).group(0))
    except Exception:
        return None


def match(name, items):
    name = str(name).strip().strip('[]').lower()
    hit = [b for b in items if b.lower() == name] or [b for b in items if b.lower() in name or name in b.lower()]
    return hit[0] if len(hit) == 1 else None


def load():
    rows = [json.loads(l) for l in open(OUT / 'answers.jsonl', encoding='utf-8')]
    pairs, ranks, bad = [], [], 0
    for a in rows:
        js = parse(a)
        if a['part'] == 'pair':
            c = match(js.get('choice'), a['items']) if js else None
            if c is None:
                bad += 1
                continue
            x, y = sorted(a['items'])
            pairs.append(dict(engine=a['engine'], sid=a['sid'], dataset=a['dataset'], a=x, b=y,
                              first=a['items'][0], rep=a['rep'], a_wins=c == x, first_wins=c == a['items'][0]))
        else:
            r = [match(v, a['items']) for v in (js or {}).get('ranking', [])] if js else []
            r = [v for v in dict.fromkeys(r) if v]
            if len(r) != len(a['items']):
                bad += 1
                continue
            ranks.append({k: a.get(k) for k in ('engine', 'sid', 'dataset', 'part', 'rep', 'rot', 'qv', 'dropped',
                                                  'filler', 'added')} | dict(items=a['items'], ranking=r))
    print(f'{len(rows)} answers, {bad} unparsed or incomplete')
    return pd.DataFrame(pairs), pd.DataFrame(ranks)


def pair_order(ranking, x, y):
    return ranking.index(x) < ranking.index(y)


def bt_fit(p, brands, l2=0.1):
    """Bradley-Terry scores from pairwise wins (L2-regularised logistic)."""
    from sklearn.linear_model import LogisticRegression
    idx = {b: i for i, b in enumerate(brands)}
    X = np.zeros((len(p), len(brands)))
    for k, r in enumerate(p.itertuples()):
        X[k, idx[r.a]] = 1
        X[k, idx[r.b]] = -1
    y = p.a_wins.astype(int).to_numpy()
    if len(set(y)) < 2:
        return {b: 0.0 for b in brands}
    m = LogisticRegression(C=1 / l2, fit_intercept=False).fit(X, y)
    return dict(zip(brands, m.coef_[0]))


def kendall(r1, r2):
    from scipy.stats import kendalltau
    common_ = [b for b in r1 if b in r2]
    return kendalltau([r1.index(b) for b in common_], [r2.index(b) for b in common_])[0]


def agree_pairs(ranking, pred):
    """Share of pairs ordered the same way in a ranking and by a score / majority dict."""
    ok = n = 0
    for x, y in itertools.combinations(ranking, 2):
        s = pred(x, y)
        if s is None:
            continue
        n += 1
        ok += (s > 0) == pair_order(ranking, x, y)
    return ok / n if n else np.nan


def analyse():
    P, R = load()
    sets = {s['sid']: s for s in json.loads((OUT / 'sets.json').read_text(encoding='utf-8'))['sets']}
    out = []

    def add(eng, part, measure, value, lo=np.nan, hi=np.nan, note=''):
        out.append(dict(engine=eng, part=part, measure=measure, value=value, ci_low=lo, ci_high=hi, note=note))

    rng = np.random.default_rng(SEED)
    for eng in MODELS:
        p, r = P[P.engine == eng], R[R.engine == eng]
        # 1. pairwise consistency and order effect
        g = p.groupby(['sid', 'a', 'b']).a_wins.mean()
        maj = np.maximum(g, 1 - g)
        add(eng, 'pairs', 'mean majority share (6 answers per pair)', 100 * maj.mean())
        add(eng, 'pairs', '% of pairs unanimous', 100 * (maj == 1).mean())
        add(eng, 'pairs', '% of pairs where order decides (first shown always wins)',
            100 * p.groupby(['sid', 'a', 'b']).first_wins.mean().eq(1).mean())
        add(eng, 'pairs', 'first-shown product wins (%)', 100 * p.first_wins.mean(),
            *np.percentile([p.first_wins.iloc[rng.integers(0, len(p), len(p))].mean() * 100 for _ in range(500)],
                           [2.5, 97.5]))
        # 2. cycles in the majority tournament, and whether they survive resampling of the answers
        n_cyc = n_robust = n_trip = 0
        for sid, gs in p.groupby('sid'):
            B = sets[sid]['brands']
            w = gs.groupby(['a', 'b']).a_wins.agg(['sum', 'count'])

            def beats(x, y, wtab):
                a_, b_ = sorted([x, y])
                if (a_, b_) not in wtab.index:              # every answer on this pair abstained
                    return False, False
                s_, c_ = wtab.loc[(a_, b_)]
                share = s_ / c_ if a_ == x else 1 - s_ / c_
                return share > 0.5, share < 0.5

            for tri in itertools.combinations(B, 3):
                n_trip += 1
                x, y, z = tri

                def cyc(wtab):
                    e = [beats(x, y, wtab), beats(y, z, wtab), beats(z, x, wtab)]
                    return all(v[0] for v in e) or all(v[1] for v in e)

                if cyc(w):
                    n_cyc += 1
                    sub = gs[gs.a.isin(tri) & gs.b.isin(tri)]
                    hits = 0
                    for _ in range(200):
                        bs = sub.groupby(['a', 'b'], group_keys=False).apply(
                            lambda t: t.sample(len(t), replace=True, random_state=int(rng.integers(1e9))))
                        hits += cyc(bs.groupby(['a', 'b']).a_wins.agg(['sum', 'count']))
                    n_robust += hits / 200 >= 0.9
        add(eng, 'cycles', '% of brand triples with a majority cycle', 100 * n_cyc / n_trip, note=f'{n_cyc} of {n_trip}')
        add(eng, 'cycles', 'cycles that survive resampling (>=90% of bootstraps)', n_robust)
        # 3. references from repeats: agreement between two identical full rankings (same order)
        full = r[r.part == 'full']
        rep_tau = [kendall(g_.ranking.iloc[0], g_.ranking.iloc[1]) for _, g_ in full.groupby(['sid', 'rot']) if len(g_) == 2]
        rep_pair = [agree_pairs(g_.ranking.iloc[0], lambda x, y, rr=g_.ranking.iloc[1]: 1 if pair_order(rr, x, y) else -1)
                    for _, g_ in full.groupby(['sid', 'rot']) if len(g_) == 2]
        add(eng, 'full', 'Kendall tau, identical repeat (same order)', np.nanmean(rep_tau))
        add(eng, 'full', 'pair agreement, identical repeat (%)', 100 * np.nanmean(rep_pair))
        rot_tau = [kendall(a_, b_) for _, g_ in full[full.rep == 0].groupby('sid')
                   for a_, b_ in itertools.combinations(g_.ranking.tolist(), 2)]
        add(eng, 'full', 'Kendall tau, different presentation order', np.nanmean(rot_tau))
        # 4. predicting rankings from pairwise answers: Bradley-Terry vs pair majority
        res = {k: [] for k in ('bt_full', 'maj_full', 'bt_drop', 'bt_add', 'bt_heldout', 'maj_heldout',
                               'bt_para', 'reversal')}
        for sid, gs in p.groupby('sid'):
            B = sets[sid]['brands']
            sc = bt_fit(gs, B)
            share = gs.groupby(['a', 'b']).a_wins.mean()

            def maj_pred(x, y, share=share):
                a_, b_ = sorted([x, y])
                v = share.get((a_, b_), 0.5)
                v = v if a_ == x else 1 - v
                return None if v == 0.5 else v - 0.5

            def bt_pred(x, y, sc=sc):
                return sc.get(x, 0) - sc.get(y, 0)

            rs = r[r.sid == sid]
            for part, key in (('full', 'bt_full'), ('drop', 'bt_drop'), ('add', 'bt_add'), ('para', 'bt_para')):
                for rk in rs[rs.part == part].ranking:
                    res[key].append(agree_pairs(rk, bt_pred))
                    if part == 'full':
                        res['maj_full'].append(agree_pairs(rk, maj_pred))
            # held-out pairs: refit without the pair, predict its majority
            for (a_, b_), gp in gs.groupby(['a', 'b']):
                m_ = gp.a_wins.mean()
                if m_ == 0.5:
                    continue
                sc_ = bt_fit(gs[~((gs.a == a_) & (gs.b == b_))], B)
                res['bt_heldout'].append((sc_[a_] > sc_[b_]) == (m_ > 0.5))
            # pairwise reversal: strong pairwise majority (>= 5 of 6) reversed in the full-ranking majority
            fr = rs[rs.part == 'full'].ranking.tolist()
            for (a_, b_), m_ in share.items():
                if m_ >= 5 / 6 or m_ <= 1 / 6:
                    in_full = np.mean([pair_order(rk, a_, b_) for rk in fr])
                    res['reversal'].append((in_full < 0.5) if m_ > 0.5 else (in_full > 0.5))
        add(eng, 'prediction', 'full ranking: pairs ordered as Bradley-Terry predicts (%)', 100 * np.nanmean(res['bt_full']))
        add(eng, 'prediction', 'full ranking: pairs ordered as pair majority predicts (%)', 100 * np.nanmean(res['maj_full']))
        add(eng, 'prediction', 'drop-one ranking: pairs as Bradley-Terry predicts (%)', 100 * np.nanmean(res['bt_drop']))
        add(eng, 'prediction', 'add-one ranking: pairs as Bradley-Terry predicts (%)', 100 * np.nanmean(res['bt_add']))
        add(eng, 'prediction', 'paraphrased question: pairs as Bradley-Terry predicts (%)', 100 * np.nanmean(res['bt_para']))
        add(eng, 'prediction', 'held-out pair majority predicted by Bradley-Terry (%)', 100 * np.nanmean(res['bt_heldout']))
        add(eng, 'reversal', '% of strong pairwise preferences (>=5 of 6) reversed in the full ranking',
            100 * np.nanmean(res['reversal']), note=f'{len(res["reversal"])} strong pairs')
        # 5. context: drop / add one, relative order of the remaining pairs vs the full ranking
        ctx = {'drop, no filler': [], 'drop, filler': [], 'add': [], 'repeat': []}
        for sid, rs in r.groupby('sid'):
            fr = rs[rs.part == 'full'].ranking.tolist()
            ref = {pr: np.mean([pair_order(rk, *pr) for rk in fr])
                   for pr in itertools.permutations(sets[sid]['brands'], 2)}
            for rk, part, filler in zip(rs.ranking, rs.part, rs.filler):
                if part not in ('drop', 'add'):
                    continue
                lab = 'add' if part == 'add' else ('drop, filler' if filler else 'drop, no filler')
                for x, y in itertools.combinations([b for b in rk if b in sets[sid]['brands']], 2):
                    if ref[(x, y)] in (0.0, 1.0):              # only pairs unanimous across the 12 full rankings
                        ctx[lab].append(pair_order(rk, x, y) != (ref[(x, y)] == 1.0))
            # repeat baseline: a full ranking vs the other 11 (leave one out), same restriction
            for i, rk in enumerate(fr):
                others = fr[:i] + fr[i + 1:]
                for x, y in itertools.combinations(rk, 2):
                    o = np.mean([pair_order(q, x, y) for q in others])
                    if o in (0.0, 1.0):
                        ctx['repeat'].append(pair_order(rk, x, y) != (o == 1.0))
        for lab, v in ctx.items():
            add(eng, 'context', f'flips of otherwise unanimous pairs: {lab} (%)', 100 * np.mean(v), note=f'n={len(v)}')
        # 6. paraphrase vs identical repeat (same presentation order)
        para = r[r.part == 'para']
        pt = []
        for (sid, rt), g_ in para.groupby(['sid', 'rot']):
            base_ = full[(full.sid == sid) & (full.rot == rt) & (full.rep == 0)].ranking.tolist()
            if base_:
                pt += [kendall(base_[0], rk) for rk in g_.ranking]
        add(eng, 'paraphrase', 'Kendall tau, paraphrase vs original (same order)', np.nanmean(pt))
        # bootstrap the difference paraphrase - repeat by set
        dd = []
        per_set = {}
        for (sid, rt), g_ in para.groupby(['sid', 'rot']):
            base_ = full[(full.sid == sid) & (full.rot == rt)].sort_values('rep').ranking.tolist()
            if len(base_) == 2:
                per_set.setdefault(sid, []).append(
                    np.mean([kendall(base_[0], rk) for rk in g_.ranking]) - kendall(base_[0], base_[1]))
        sids = list(per_set)
        for _ in range(2000):
            take = rng.choice(sids, len(sids))
            dd.append(np.nanmean(np.concatenate([per_set[s_] for s_ in take])))
        diff = np.nanmean(np.concatenate([per_set[s_] for s_ in sids]))
        add(eng, 'paraphrase', 'tau difference: paraphrase minus identical repeat', diff, *np.percentile(dd, [2.5, 97.5]))
    res_ = pd.DataFrame(out)
    res_.to_csv(RESULTS / 'x15_stability.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 90)
    print(res_.round(3).to_string(index=False))


def extra_by_market():
    """Per engine and market: pair agreement between full rankings in the same / any order, and Bradley-Terry."""
    P, R = load()
    sets = {s_['sid']: s_ for s_ in json.loads((OUT / 'sets.json').read_text(encoding='utf-8'))['sets']}
    rows = []
    for eng in MODELS:
        for ds in list(fs.MARKETS) + ['all']:
            r = R[(R.engine == eng) & ((R.dataset == ds) | (ds == 'all'))]
            p = P[(P.engine == eng) & ((P.dataset == ds) | (ds == 'all'))]
            full = r[r.part == 'full']
            same = [agree_pairs(g.ranking.iloc[0], lambda u, v, b=g.ranking.iloc[1]: 1 if pair_order(b, u, v) else -1)
                    for _, g in full.groupby(['sid', 'rot']) if len(g) == 2]
            anyo = [agree_pairs(a, lambda u, v, b=b: 1 if pair_order(b, u, v) else -1)
                    for _, g in full.groupby('sid') for a, b in itertools.permutations(g.ranking.tolist(), 2)]
            bt = []
            for sid, gs in p.groupby('sid'):
                sc = bt_fit(gs, sets[sid]['brands'])
                bt += [agree_pairs(rk, lambda u, v, sc=sc: sc[u] - sc[v]) for rk in full[full.sid == sid].ranking]
            first = np.mean([rk.index(it[0]) + 1 for rk, it in zip(full.ranking, full['items'])])
            last = np.mean([rk.index(it[-1]) + 1 for rk, it in zip(full.ranking, full['items'])])
            g = p.groupby(['sid', 'a', 'b']).a_wins.mean()
            maj = np.maximum(g, 1 - g)
            rows.append(dict(engine=eng, dataset=ds, pair_majority=100 * maj.mean(), unanimous=100 * (maj == 1).mean(),
                             agree_same_order=100 * np.mean(same), agree_any_order=100 * np.mean(anyo),
                             bt_full=100 * np.mean(bt), rank_listed_first=first, rank_listed_last=last))
    x = pd.DataFrame(rows)
    x.to_csv(RESULTS / 'x15_by_market.csv', index=False)
    print(x.round(2).to_string(index=False))
    return x


def plot():
    import common
    import plotstyle  # noqa: F401
    from plotstyle import plt, ENGINE_COLOR, INK2
    col = {'openai': ENGINE_COLOR['OPENAI'], 'gemini': ENGINE_COLOR['GEMINI'], 'ChatGPT': ENGINE_COLOR['OPENAI'],
           'Gemini': ENGINE_COLOR['GEMINI'], 'Perplexity': ENGINE_COLOR['PERPLEXITY']}
    name = {'openai': 'ChatGPT (gpt-5.4-nano)', 'gemini': 'Gemini (3.1-flash-lite)'}
    # --- X13 (observational): held-out accuracy by granularity of the brand score, per market
    for ds, res_dir, fig_dir in (('gfiber', RESULTS, common.ROOT / 'figures'),
                                 ('workorb', RESULTS / 'workorb', common.ROOT / 'figures' / 'workorb')):
        m = pd.read_csv(res_dir / 'x13_models.csv')
        m = m[m.split == 'random half of days']
        models = ['brand score (engine-wide)', 'brand score per topic', 'brand score per question',
                  'pair majority per question (no transitivity)']
        labels = ['one score\nper brand', 'one score per\nbrand and topic', 'one score per\nbrand and question',
                  'pair by pair\n(no single score)']
        fig, ax = plt.subplots(figsize=(10, 3.8))
        engines = [e for e in ('ChatGPT', 'Gemini', 'Perplexity') if e in set(m.engine)]
        w = 0.8 / len(engines)
        for k, e in enumerate(engines):
            g = m[m.engine == e].set_index('model').reindex(models)
            xs = np.arange(len(models)) + (k - (len(engines) - 1) / 2) * w
            ax.bar(xs, g.accuracy, w * 0.92, color=col[e], label=f'{e} ({int(g.test_pairs.iloc[0]):,} held-out pairs)')
            for xi, v in zip(xs, g.accuracy):
                ax.text(xi, v + 0.8, f'{v:.0f}', ha='center', fontsize=7.5, color=INK2)
        ax.set_xticks(range(len(models)), labels, fontsize=8.5)
        ax.set_ylim(40, 90)
        ax.axhline(50, color=INK2, lw=1, ls='--')
        ax.set_ylabel('held-out pairs predicted (%)')
        ax.grid(axis='x', visible=False)
        ax.legend(fontsize=8, loc='upper left')
        mk = {'gfiber': 'home internet', 'workorb': 'AEC software'}[ds]
        fig.suptitle(f'X13: brand order in real answers ({mk}); which brand is named first in a pair, other half of the days',
                     x=0.01, ha='left', fontweight='bold', fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / 'x13_rank_models.png')
        plt.close(fig)
    # --- X15 (controlled)
    res = pd.read_csv(RESULTS / 'x15_stability.csv')
    bym = extra_by_market()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    items = [('identical repeat', 'Kendall tau, identical repeat (same order)'),
             ('question reworded', 'Kendall tau, paraphrase vs original (same order)'),
             ('products listed in\na different order', 'Kendall tau, different presentation order')]
    for k, eng in enumerate(MODELS):
        v = [res[(res.engine == eng) & (res.measure == mm)].value.iloc[0] for _, mm in items]
        xs = np.arange(3) + (k - 0.5) * 0.36
        ax.bar(xs, v, 0.34, color=col[eng], label=name[eng])
        for xi, vv in zip(xs, v):
            ax.text(xi, vv + 0.015, f'{vv:.2f}', ha='center', fontsize=8, color=INK2)
    ax.set_xticks(range(3), [a for a, _ in items], fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('agreement of two rankings (Kendall tau)')
    ax.set_title('What changes the ranking of the same 6 products', fontsize=9.5)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='x', visible=False)
    ax = axes[1]
    items = [('one score per brand\n(Bradley-Terry, pairs only)', None, 'bt_full'),
             ('pair by pair\n(no single score)', 'full ranking: pairs ordered as pair majority predicts (%)', None),
             ('another full ranking,\nany order (reference)', None, 'agree_any_order')]
    for k, eng in enumerate(MODELS):
        allrow = bym[(bym.engine == eng) & (bym.dataset == 'all')].iloc[0]
        v = [allrow[c] if c else res[(res.engine == eng) & (res.measure == mm)].value.iloc[0] for _, mm, c in items]
        xs = np.arange(3) + (k - 0.5) * 0.36
        ax.bar(xs, v, 0.34, color=col[eng], label=name[eng])
        for xi, vv in zip(xs, v):
            ax.text(xi, vv + 0.8, f'{vv:.0f}', ha='center', fontsize=8, color=INK2)
    ax.set_xticks(range(3), [a for a, _, _ in items], fontsize=8.5)
    ax.set_ylim(50, 100)
    ax.set_ylabel('pairs in the full rankings predicted (%)')
    ax.set_title('Predicting the full 6-product rankings', fontsize=9.5)
    ax.grid(axis='x', visible=False)
    fig.suptitle('X15: controlled ranking stability (20 questions, fixed facts)', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    for d_ in (common.ROOT / 'figures', common.ROOT / 'figures' / 'workorb'):
        fig.savefig(d_ / 'x15_stability.png')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'run', 'analyse', 'plot'])
    {'build': build, 'run': run, 'analyse': analyse, 'plot': plot}[ap.parse_args().cmd]()
