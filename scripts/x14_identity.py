"""X14 - brand vs publisher identity: how much of a model's preference comes from facts, brand names
and the names of the websites supporting them?

Packets: one real question, 5 competing products (the fixed fact sheets of x14_fact_sheets.py) and one
third-party excerpt per product (a paragraph about that product from a page of another website).
Brand identity and publisher identity are masked or revealed independently (2x2); everything else -
facts, wording, order, length - is identical across the four conditions.

  A  brands masked  / publishers masked       C  brands masked  / publishers shown
  B  brands shown   / publishers masked       D  brands shown   / publishers shown

Masked brands appear as "Vendor K" (the same letter for a product in every mention: sheet and excerpts);
masked publishers as "Publisher: not shown", and the publisher's own name inside an excerpt as "this site".
Every packet is shown in two orders (random, reversed) because order matters for ChatGPT (X12).

Hypothetical version (no excerpts; the prompt states that the profiles are hypothetical and the names were
assigned at random and carry no information):
  permuted   the packet's 5 real sheets, brand names rotated over the sheets (5 rotations x 2 directions)
  identical  5 copies of one sheet under the 5 brand names (5 rotations), so only the name differs

  python scripts/x14_identity.py questions   # pick questions (LLM screen), data/x14/questions.json
  python scripts/x14_identity.py publishers  # publisher display names + familiarity
  python scripts/x14_identity.py build       # packets.json + a sample for review
  python scripts/x14_identity.py run         # answers.jsonl (resumable; X14_LIMIT=n for a pilot)
  python scripts/x14_identity.py analyse
"""
import argparse
import itertools
import json
import os
import random
import re
import string
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import x14_fact_sheets as fs  # noqa: E402

OUT = ROOT / 'data' / 'x14'
RESULTS = ROOT / 'results'
MODELS = fs.MODELS
SEED = 42
N_QUESTIONS = {'gfiber': 40, 'workorb': 20}
K = 5                                                   # products per packet
EXCERPT_WORDS = (35, 140)
FAMOUS_TRANCO = 20_000                                  # publisher "well known" if Tranco rank <= this
OBSCURE_TRANCO = 200_000                                # "little known" if > this or not ranked
CONDITIONS = {'A': (False, False), 'B': (True, False), 'C': (False, True), 'D': (True, True)}  # brand, publisher


def rx_word(names):
    return re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                      + r')(?![A-Za-z0-9])', re.I)


# ---------------------------------------------------------------- questions

SCREEN = """A user asks an AI assistant the question below about {kind}s. Is it a question that can be
answered by comparing competing {kind}s and recommending some of them (rather than a how-to, a definition
or a question about one specific company)? Answer "yes" or "no" only."""


def questions():
    import common  # noqa: F401
    import llm
    out = []
    for ds, m in fs.MARKETS.items():
        p = pd.read_csv(m['tables'] / 'prompt_labels.csv')
        p = p[p.n_brands_named == 0].drop_duplicates('prompt')
        al, rx = fs.load_aliases()
        p = p[~p.prompt.map(lambda q: any(r.search(q) for b, r in rx.items() if al[b]['dataset'] == ds))]
        ans = llm.ask_many(SCREEN.format(kind=m['kind']), list(p.prompt), model=fs.EXTRACT_MODEL)
        p['ok'] = [a.strip().lower().startswith('yes') for a in ans]
        ok = p[p.ok]
        # comparison questions first, then the rest, sampled evenly over topics
        ok = ok.assign(pri=(ok.intent != 'comparing').astype(int),
                       r=np.random.default_rng(SEED).random(len(ok))).sort_values(['pri', 'r'])
        take = (ok.groupby('topic', group_keys=False).apply(lambda g: g.assign(k=range(len(g))))
                .sort_values(['pri', 'k', 'r']).head(N_QUESTIONS[ds]))
        print(f'{ds}: {len(p)} brand-free questions, {len(ok)} pass the screen, {len(take)} taken '
              f'({(take.intent == "comparing").sum()} comparing)')
        out += [dict(dataset=ds, prompt_id=r.prompt_id, question=r.prompt, topic=r.topic, intent=r.intent)
                for r in take.itertuples()]
    (OUT / 'questions.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')


# ---------------------------------------------------------------- excerpts + publishers

def third_party_paragraphs(ds):
    """Paragraphs of third-party pages that name exactly one of the market's sheet brands and no other vendor."""
    m = fs.MARKETS[ds]
    al, rx = fs.load_aliases()
    roles = pd.read_csv(m['tables'] / 'domain_roles.csv')
    tp = set(roles[roles.role == 'third_party'].domain)
    bd = pd.read_csv(m['tables'] / 'brand_domains.csv')
    others = [b for b in bd.brand.dropna().unique() if b not in m['brands'] and len(b) > 3]
    orx = rx_word(others) if others else None
    rows, seen = [], set()
    for f in ('page_text.jsonl', 'x06_parsed.jsonl'):
        with open(m['tables'] / f, encoding='utf-8') as fh:
            for line in fh:
                r = json.loads(line)
                u = r.get('url') or ''
                if u in seen:
                    continue
                seen.add(u)
                h = urlparse(u).netloc.lower().removeprefix('www.')
                reg = '.'.join(h.split('.')[-2:])
                site = h if h in tp else reg if reg in tp else None
                if not site or 'reddit' in h or 'youtube' in h:
                    continue
                for p in r.get('paragraphs') or []:
                    n = len(p.split())
                    if not EXCERPT_WORDS[0] <= n <= EXCERPT_WORDS[1]:
                        continue
                    hit = [b for b in m['brands'] if rx[b].search(p)]
                    if len(hit) == 1 and not (orx and orx.search(p)):
                        rows.append(dict(dataset=ds, brand=hit[0], site=site, url=u, text=p.strip(), n=n))
    return pd.DataFrame(rows).drop_duplicates(['brand', 'text'])


PUB_NAMES = """For each website domain below, give the name under which the publication or company is known
to readers (e.g. "cnet.com" -> "CNET", "highspeedinternet.com" -> "HighSpeedInternet.com").
Return JSON only: {"domain": "name", ...}."""


def publishers():
    import common  # noqa: F401
    import llm
    exc = pd.concat([third_party_paragraphs(ds) for ds in fs.MARKETS])
    exc.to_csv(OUT / 'excerpt_pool.csv', index=False)
    sites = sorted(exc.site.unique())
    names = {}
    for i in range(0, len(sites), 60):
        raw = llm.ask(PUB_NAMES, '\n'.join(sites[i:i + 60]), model=fs.EXTRACT_MODEL)
        try:
            names |= json.loads(re.search(r'\{.*\}', raw, re.S).group(0))
        except Exception:
            print('name batch failed', i)
    auth = pd.concat([pd.read_csv(m['tables'] / 'domain_authority.csv', usecols=['domain', 'tranco_rank'])
                      for m in fs.MARKETS.values()]).drop_duplicates('domain').set_index('domain').tranco_rank
    pub = pd.DataFrame({'site': sites})
    pub['name'] = pub.site.map(lambda s: names.get(s, s))
    pub['tranco_rank'] = pub.site.map(lambda s: auth.get(s, auth.get('.'.join(s.split('.')[-2:]), np.nan)))
    pub['familiarity'] = np.where(pub.tranco_rank <= FAMOUS_TRANCO, 'well known',
                                  np.where(pub.tranco_rank.isna() | (pub.tranco_rank > OBSCURE_TRANCO),
                                           'little known', 'middle'))
    pub.to_csv(OUT / 'publishers.csv', index=False)
    x = exc.merge(pub, on='site')
    print(x.groupby(['dataset', 'familiarity']).agg(paragraphs=('text', 'size'), sites=('site', 'nunique')))
    print(x.groupby(['dataset', 'brand', 'familiarity']).site.nunique().unstack().fillna(0).astype(int))


# ---------------------------------------------------------------- packets

COMMON_WORD_PUBS = {'Money', 'Wired', 'Time', 'Fortune', 'Zoom', 'Eero', 'Insider', 'Axios', 'Vox',
                    'Reviews', 'Motherboard', 'Popular', 'Quartz'}


def publisher_regex(site, name):
    stem = site.split('.')[0]
    terms = {site, name, name.replace('.com', '')}
    if len(stem) > 4:
        terms.add(stem)
    return rx_word([t for t in terms if len(t) > 2])


def build():
    from sentence_transformers import SentenceTransformer
    al, rx = fs.load_aliases()
    sheets = json.loads((OUT / 'sheets.json').read_text(encoding='utf-8'))
    qs = json.loads((OUT / 'questions.json').read_text(encoding='utf-8'))
    pub = pd.read_csv(OUT / 'publishers.csv').set_index('site')
    pool = pd.read_csv(OUT / 'excerpt_pool.csv')
    pool = pool[pool.site.isin(pub.index)]
    pool['familiarity'] = pool.site.map(pub.familiarity)
    pool = pool[pool.familiarity != 'middle'].reset_index(drop=True)
    known = pub[pub.familiarity != 'little known']
    names = [n for n in set(known.name) | set(known.index)
             if (len(n) > 3 and n[0].isupper() or '.' in n) and n not in COMMON_WORD_PUBS]
    other_pub = re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True))
                           + r')(?![A-Za-z0-9])')                  # case-sensitive: "Wired" the magazine, not "wired"
    enc = SentenceTransformer('all-MiniLM-L6-v2')
    pe = enc.encode(pool.text.tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    rng = random.Random(SEED)
    used = {}                                         # excerpt reuse counter
    packets = []
    for ds, m in fs.MARKETS.items():
        dq = [q for q in qs if q['dataset'] == ds]
        qe = enc.encode([q['question'] for q in dq], normalize_embeddings=True)
        brands = m['brands'][:]
        appear = {b: 0 for b in brands}
        for qi, q in enumerate(dq):
            # brands: the least used so far, ties at random (balanced appearance)
            chosen = sorted(brands, key=lambda b: (appear[b], rng.random()))[:K]
            for b in chosen:
                appear[b] += 1
            # publishers: alternate well known / little known within the packet, flipped between packets
            fam = ['well known', 'little known'] * 3
            if qi % 2:
                fam = fam[::-1]
            rng.shuffle(chosen)
            letters = rng.sample([c for c in string.ascii_uppercase if c not in 'IOSQ'], K)
            products, sources = [], []
            for j, b in enumerate(chosen):
                cand = pool[(pool.dataset == ds) & (pool.brand == b)]
                want = cand[cand.familiarity == fam[j]]
                if want.empty:
                    want = cand
                sims = pe[want.index] @ qe[qi] - 0.15 * np.array([used.get(i, 0) for i in want.index])
                # one excerpt per publisher site within the packet
                taken = {s['site'] for s in sources}
                order = [want.index[k] for k in np.argsort(-sims) if pool.site[want.index[k]] not in taken]
                ix = order[0]
                used[ix] = used.get(ix, 0) + 1
                e = pool.loc[ix]
                pname = pub.loc[e.site, 'name']
                text = rx[b].sub('{B}', e.text)
                text = publisher_regex(e.site, pname).sub('{P}', text)
                text = other_pub.sub(lambda mm: '{Q:' + mm.group(0) + '}', text)   # other publications named
                products.append(dict(brand=b, letter=letters[j], sheet=sheets[b]['sheet']))
                sources.append(dict(brand=b, site=e.site, publisher=pname, familiarity=pub.loc[e.site, 'familiarity'],
                                    tranco_rank=None if pd.isna(pub.loc[e.site, 'tranco_rank']) else int(pub.loc[e.site, 'tranco_rank']),
                                    url=e.url, text=text, sim=float(pe[ix] @ qe[qi])))
            src_order = list(range(K))
            rng.shuffle(src_order)
            for sid, k in enumerate(src_order):
                sources[k]['sid'] = f'S{sid + 1}'
            packets.append(dict(pid=len(packets), dataset=ds, question=q['question'], prompt_id=q['prompt_id'],
                                products=products, sources=[sources[k] for k in src_order]))
    (OUT / 'packets.json').write_text(json.dumps(packets, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'{len(packets)} packets; excerpts reused: {sum(v > 1 for v in used.values())}')
    with open(OUT / 'packet_sample.md', 'w', encoding='utf-8') as f:
        for pk in packets[:2] + packets[-2:]:
            for c in 'AD':
                f.write(f'# packet {pk["pid"]} condition {c}\n\n```\n{render(pk, c, 0)}\n```\n\n')


# ---------------------------------------------------------------- prompts

MAIN_INSTRUCTIONS = ('You help a user choose a {kind}. Use only the information provided in the message: '
                     'the product profiles and the excerpts from third-party websites.')
HYPO_INSTRUCTIONS = ('You help a user choose a {kind}. The product profiles in the message are hypothetical but '
                     'complete: they contain everything that is known about each product. The product names were '
                     'assigned to the profiles at random and carry no information.')
ASK = ('Rank all {k} products for the user\'s question, best first. Then list the IDs of the excerpts that support '
       'your top three choices. Answer with JSON only: {{"ranking": ["<product>", ...], "sources": ["S1", ...], '
       '"reason": "<one sentence>"}}. Write each product exactly as it is named above.')


def name_of(p, show_brand):
    return p['brand'] if show_brand else f'Vendor {p["letter"]}'


def render(pk, cond, order):
    show_brand, show_pub = CONDITIONS[cond]
    m = fs.MARKETS[pk['dataset']]
    prods = pk['products'] if order == 0 else pk['products'][::-1]
    srcs = pk['sources'] if order == 0 else pk['sources'][::-1]
    names = {p['brand']: name_of(p, show_brand) for p in pk['products']}
    lines = [f'Question: {pk["question"]}', '', 'Product profiles:']
    for p in prods:
        lines += ['', f'[{names[p["brand"]]}]', fs.render_sheet(p['sheet'], m['fields'], names[p['brand']])]
    lines += ['', 'Excerpts from third-party websites:']
    for s in srcs:
        publisher = f'{s["publisher"]} ({s["site"]})' if show_pub else 'not shown'
        text = s['text'].replace('{B}', names[s['brand']]).replace('{P}', s['publisher'] if show_pub else 'this site')
        text = re.sub(r'\{Q:([^}]*)\}', lambda mm: mm.group(1) if show_pub else 'another site', text)
        lines += ['', f'[{s["sid"]}] Publisher: {publisher}', f'"{text}"']
    lines += ['', ASK.format(k=K)]
    return '\n'.join(lines)


def render_hypo(pk, names_for_sheet, sheets_used):
    m = fs.MARKETS[pk['dataset']]
    lines = [f'Question: {pk["question"]}', '', 'Product profiles:']
    for name, sheet in zip(names_for_sheet, sheets_used):
        lines += ['', f'[{name}]', fs.render_sheet(sheet, m['fields'], name)]
    lines += ['', ASK.format(k=K).replace(' Then list the IDs of the excerpts that support your top three choices.', '')
              .replace('"sources": ["S1", ...], ', '')]
    return '\n'.join(lines)


def jobs_all():
    packets = json.loads((OUT / 'packets.json').read_text(encoding='utf-8'))
    jobs = []
    for pk in packets:
        kind = fs.MARKETS[pk['dataset']]['kind']
        for cond in CONDITIONS:
            for order in (0, 1):
                for rep in (0, 1):
                    jobs.append(dict(exp='main', pid=pk['pid'], cond=cond, order=order, rep=rep,
                                     ins=MAIN_INSTRUCTIONS.format(kind=kind), text=render(pk, cond, order)))
        brands = [p['brand'] for p in pk['products']]
        sheets = [p['sheet'] for p in pk['products']]
        for rot in range(K):                          # permuted: brand name i+rot on sheet i, both directions
            for direction in (1, -1):
                names = [brands[(i * direction + rot) % K] for i in range(K)]
                jobs.append(dict(exp='permuted', pid=pk['pid'], rot=rot, direction=direction, rep=0, names=names,
                                 ins=HYPO_INSTRUCTIONS.format(kind=kind), text=render_hypo(pk, names, sheets)))
        base = sheets[pk['pid'] % K]                  # identical: one sheet, 5 names in rotated order
        for rot in range(K):
            names = [brands[(i + rot) % K] for i in range(K)]
            for rep in (0, 1):
                jobs.append(dict(exp='identical', pid=pk['pid'], rot=rot, rep=rep, names=names,
                                 base_brand=brands[pk['pid'] % K],
                                 ins=HYPO_INSTRUCTIONS.format(kind=kind), text=render_hypo(pk, names, [base] * K)))
    return packets, jobs


# ---------------------------------------------------------------- run

def run():
    import common  # noqa: F401
    import hashlib
    _, jobs = jobs_all()
    ans_path = OUT / 'answers.jsonl'
    done = set()
    if ans_path.exists():
        done = {json.loads(l)['key'] for l in open(ans_path, encoding='utf-8')}
    todo = []
    for j in jobs:
        for eng, model in MODELS.items():
            k = hashlib.sha1(f'{model}|{j["ins"]}|{j["text"]}|{j["rep"]}'.encode()).hexdigest()
            if k not in done:
                todo.append(j | dict(engine=eng, key=k))
    if int(os.environ.get('X14_LIMIT', 0)):
        rng = random.Random(1)
        todo = rng.sample(todo, int(os.environ['X14_LIMIT']))
    print(f'{len(todo)} calls to make', flush=True)
    lock, n = threading.Lock(), [0]

    def one(j):
        try:
            a = fs.ask(j['engine'], j['ins'], j['text'])
        except Exception as e:
            print(f'  {j["engine"]} {j["exp"]} p{j["pid"]}: {type(e).__name__}: {str(e)[:100]}', flush=True)
            return
        rec = {k: v for k, v in j.items() if k not in ('ins', 'text')} | dict(answer=a)
        with lock:
            with open(ans_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n[0] += 1
            if n[0] % 250 == 0:
                print(f'  {n[0]}/{len(todo)}', flush=True)

    for eng, workers in (('openai', 8), ('gemini', 6)):
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, [j for j in todo if j['engine'] == eng]))
    print('done', flush=True)


# ---------------------------------------------------------------- analyse

def parse(answer):
    try:
        return json.loads(re.search(r'\{.*\}', answer, re.S).group(0))
    except Exception:
        return None


def long_main(packets, rows):
    out, bad = [], 0
    for a in rows:
        pk = packets[a['pid']]
        show_brand, show_pub = CONDITIONS[a['cond']]
        js = parse(a['answer'])
        if not js or not isinstance(js.get('ranking'), list):
            bad += 1
            continue
        rank = [str(x).strip().strip('[]').lower() for x in js['ranking']]
        cited = {str(s).strip().strip('[]').upper() for s in js.get('sources') or []}
        src_of = {s['brand']: s for s in pk['sources']}
        for p in pk['products']:
            nm = name_of(p, show_brand).lower()
            pos = next((i + 1 for i, r in enumerate(rank) if r == nm or nm in r), None)
            s = src_of[p['brand']]
            out.append(dict(engine=a['engine'], pid=a['pid'], dataset=pk['dataset'], cond=a['cond'], order=a['order'],
                            rep=a['rep'], brand=p['brand'], show_brand=show_brand, show_pub=show_pub,
                            rank=pos, top3=pos is not None and pos <= 3, top1=pos == 1,
                            source_cited=s['sid'] in cited, publisher=s['publisher'], familiarity=s['familiarity'],
                            famous_pub=s['familiarity'] == 'well known'))
    return pd.DataFrame(out), bad


def fe_ols(d, y, X, fe, cluster):
    import statsmodels.api as sm
    Y = d[y].astype(float)
    Z = d[X].astype(float)
    g = d[fe].astype(str).agg('|'.join, axis=1) if isinstance(fe, list) else d[fe].astype(str)
    Zd = Z - Z.groupby(g.to_numpy()).transform('mean')
    Yd = Y - Y.groupby(g.to_numpy()).transform('mean')
    return sm.OLS(Yd, Zd).fit(cov_type='cluster', cov_kwds={'groups': d[cluster].to_numpy()})


def rms(x):
    return float(np.sqrt(np.nanmean(np.square(x))))


def cell_split(g, col, y, a=True, b=False):
    """Per product-in-packet: mean outcome with col == a minus with col == b (pp)."""
    m = g.groupby(['cell', col])[y].mean().unstack(col)
    return 100 * (m[a] - m[b])


def boot_ratio(eff, plc, n=2000):
    pid = pd.Series(eff.index.str.split(':').str[0], index=eff.index)
    packs = pid.unique()
    rng = np.random.default_rng(SEED)
    r = []
    for _ in range(n):
        take = rng.choice(packs, len(packs))
        idx = np.concatenate([pid.index[pid == p_] for p_ in take])
        r.append(rms(eff.loc[idx]) / rms(plc.loc[idx]))
    return np.percentile(r, [2.5, 97.5])


def rankings(packets, rows):
    out = []
    for a in rows:
        js = parse(a['answer'])
        pk = packets[a['pid']]
        show_brand = CONDITIONS[a['cond']][0]
        rank = [str(x).strip().strip('[]').lower() for x in js['ranking']]
        pos = {}
        for p in pk['products']:
            nm = name_of(p, show_brand).lower()
            pos[p['brand']] = next((i + 1 for i, r in enumerate(rank) if r == nm or nm in r), 6)
        out.append(dict(engine=a['engine'], pid=a['pid'], cond=a['cond'], order=a['order'], rep=a['rep'],
                        show_brand=show_brand, show_pub=CONDITIONS[a['cond']][1],
                        ranks=[pos[p['brand']] for p in pk['products']]))
    return pd.DataFrame(out)


def kendall_pairs(r, col):
    """Mean Kendall tau between answers that differ only in col, and between repeats of the same prompt."""
    from scipy.stats import kendalltau
    other = 'show_pub' if col == 'show_brand' else 'show_brand'
    rev, rep = [], []
    for _, g in r.groupby(['pid', 'order', 'rep', other]):
        if g[col].nunique() == 2:
            rev.append(kendalltau(g[g[col]].ranks.iloc[0], g[~g[col]].ranks.iloc[0])[0])
    for _, g in r.groupby(['pid', 'cond', 'order']):
        if len(g) == 2:
            rep.append(kendalltau(g.ranks.iloc[0], g.ranks.iloc[1])[0])
    return float(np.nanmean(rev)), float(np.nanmean(rep))


def brand_reveal(d, n=2000):
    rows = []
    rng = np.random.default_rng(SEED)
    for (eng, ds, b), g in d.groupby(['engine', 'dataset', 'brand']):
        m = g.groupby(['pid', 'show_brand']).top3.mean().unstack('show_brand')
        diff = 100 * (m[True] - m[False])
        bs = [diff.iloc[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
        rows.append(dict(engine=eng, dataset=ds, brand=b, packets=len(diff),
                         top3_masked=100 * m[False].mean(), top3_shown=100 * m[True].mean(),
                         reveal_pp=diff.mean(), ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5)))
    return pd.DataFrame(rows).set_index(['engine', 'dataset', 'brand'])


def name_premium(h):
    """Hypothetical (permuted) version: how much a name lifts top-3 over what its sheet gets in the same packet."""
    h = h.copy()
    h['top3'] = h.top3.astype(float)
    base = h.groupby(['engine', 'pid', 'sheet_brand']).top3.transform('mean')
    h['lift'] = 100 * (h.top3 - base)
    return h.groupby(['engine', 'name']).lift.mean().rename('name_premium_pp').rename_axis(['engine', 'brand'])


def analyse():
    packets = json.loads((OUT / 'packets.json').read_text(encoding='utf-8'))
    rows = [json.loads(l) for l in open(OUT / 'answers.jsonl', encoding='utf-8')]
    leak = pd.read_csv(RESULTS / 'x14_leak_calls.csv')
    leak_rate = leak.groupby(['engine', 'brand']).correct.mean()
    out = []
    main = [r for r in rows if r['exp'] == 'main']
    d, bad = long_main(packets, main)
    print(f'main: {len(main)} answers, {bad} unparsed; {d["rank"].isna().mean():.1%} product-rows unranked')
    d['leak'] = [leak_rate.get((e, b), np.nan) for e, b in zip(d.engine, d.brand)]
    d['identifiable'] = d.leak >= 0.5
    d['sb'] = d.show_brand.astype(float)
    d['sp'] = d.show_pub.astype(float)
    d['sbxsp'] = d.sb * d.sp
    d['cell'] = d.pid.astype(str) + ':' + d.brand
    d.to_csv(RESULTS / 'x14_main_rows.csv', index=False)
    # Every answer puts exactly 3 of 5 products in its top 3, so the average effect of revealing brand names is 0
    # by construction; revealing can only reshuffle. We therefore measure (1) how much it reshuffles compared with
    # repeating the same prompt, (2) which brands gain or lose, and (3) whether the gains follow the name effect
    # the model shows in the hypothetical version.
    rk = rankings(packets, main)
    for eng, g in d.groupby('engine'):
        for what, col in (('brand names', 'show_brand'), ('publisher names', 'show_pub')):
            eff = cell_split(g, col, 'top3')
            plc = cell_split(g, 'rep', 'top3', a=0, b=1)
            lo, hi = boot_ratio(eff, plc)
            out.append(dict(engine=eng, part=f'reveal {what}: RMS change in top-3 per product (pp)', outcome='top3',
                            term='reveal', value=rms(eff), ci_low=np.nan, ci_high=np.nan, p=np.nan))
            out.append(dict(engine=eng, part='repeat same prompt: RMS change in top-3 per product (pp)', outcome='top3',
                            term='placebo', value=rms(plc), ci_low=np.nan, ci_high=np.nan, p=np.nan))
            out.append(dict(engine=eng, part=f'reveal {what}: RMS ratio to repeat', outcome='top3', term='ratio',
                            value=rms(eff) / rms(plc), ci_low=lo, ci_high=hi, p=np.nan))
            tau_rev, tau_rep = kendall_pairs(rk[rk.engine == eng], col)
            out.append(dict(engine=eng, part=f'reveal {what}: Kendall tau masked vs shown', outcome='ranking',
                            term='tau', value=tau_rev, ci_low=np.nan, ci_high=np.nan, p=np.nan))
            out.append(dict(engine=eng, part='repeat same prompt: Kendall tau', outcome='ranking', term='tau',
                            value=tau_rep, ci_low=np.nan, ci_high=np.nan, p=np.nan))
        # publisher reveal: reliance on the excerpt, by publisher familiarity (not constrained to sum to zero)
        g = g.assign(famous=(g.familiarity == 'well known').astype(float))
        g['spxfam'] = g.sp * g.famous
        for y in ('source_cited', 'top3'):
            r = fe_ols(g, y, ['sb', 'sp', 'spxfam'], 'cell', 'pid')
            for v, lab in (('sp', 'publisher shown (little-known publisher)'), ('spxfam', 'extra if publisher well known')):
                lo, hi = r.conf_int().loc[v]
                out.append(dict(engine=eng, part=lab, outcome=y, term=v, value=100 * r.params[v],
                                ci_low=100 * lo, ci_high=100 * hi, p=r.pvalues[v]))
        cell_means = g.groupby(['cond', 'identifiable', 'familiarity']).top3.mean().unstack('cond') * 100
        cell_means.to_csv(RESULTS / f'x14_cells_{eng}.csv')
    # brand-level reveal effects with packet-bootstrap CIs
    bl = brand_reveal(d)
    bl = bl.join(leak_rate.rename('leak').mul(100), on=['engine', 'brand'])
    # hypothetical versions: name effect net of the sheet
    for exp in ('permuted', 'identical'):
        hx = []
        for a in (r for r in rows if r['exp'] == exp):
            js = parse(a['answer'])
            if not js or not isinstance(js.get('ranking'), list):
                continue
            rank = [str(x).strip().strip('[]').lower() for x in js['ranking']]
            for slot, nm in enumerate(a['names']):
                pos = next((i + 1 for i, r in enumerate(rank) if r == nm.lower() or nm.lower() in r), None)
                hx.append(dict(engine=a['engine'], pid=a['pid'], dataset=packets[a['pid']]['dataset'], name=nm,
                               slot=slot + 1, sheet_brand=packets[a['pid']]['products'][slot]['brand'],
                               top3=pos is not None and pos <= 3, rank=pos))
        h = pd.DataFrame(hx)
        h.to_csv(RESULTS / f'x14_{exp}_rows.csv', index=False)
        for (eng, ds), g in h.groupby(['engine', 'dataset']):
            # share of top-3 variance explained by the name, beyond sheet (permuted) or slot (identical)
            import statsmodels.formula.api as smf
            base = 'C(sheet_brand)' if exp == 'permuted' else 'C(slot)'
            f0 = smf.ols(f'top3 ~ {base} + C(pid)', g.assign(top3=g.top3.astype(float))).fit()
            f1 = smf.ols(f'top3 ~ {base} + C(pid) + C(name)', g.assign(top3=g.top3.astype(float))).fit()
            from statsmodels.stats.anova import anova_lm
            p = anova_lm(f0, f1)['Pr(>F)'].iloc[1]
            by = (g.groupby('name').top3.mean() * 100).round(1)
            out.append(dict(engine=eng, part=f'{exp}: name effect', outcome='top3', term=ds,
                            value=by.max() - by.min(), ci_low=np.nan, ci_high=np.nan, p=p,
                            detail='; '.join(f'{k} {v}' for k, v in by.sort_values(ascending=False).items())))
    prem = name_premium(pd.read_csv(RESULTS / 'x14_permuted_rows.csv'))
    bl = bl.join(prem, on=['engine', 'brand'])
    bl.round(1).to_csv(RESULTS / 'x14_brand_reveal.csv')
    print(bl.round(1).to_string())
    from scipy.stats import spearmanr, pearsonr
    for eng, g in bl.reset_index().groupby('engine'):
        for x in ('name_premium_pp', 'leak'):
            ok = g[['reveal_pp', x]].dropna()
            rs, ps = spearmanr(ok.reveal_pp, ok[x])
            rp, pp = pearsonr(ok.reveal_pp, ok[x])
            out.append(dict(engine=eng, part=f'brand reveal effect vs {x} (19 brands)', outcome='reveal_pp',
                            term='spearman', value=rs, ci_low=np.nan, ci_high=np.nan, p=ps,
                            detail=f'pearson {rp:.2f} (p={pp:.3f})'))
    res = pd.DataFrame(out)
    res.to_csv(RESULTS / 'x14_identity.csv', index=False)
    pd.set_option('display.width', 250)
    print(res.drop(columns=['detail'], errors='ignore').round(3).to_string(index=False))
    for r in res[res.part.str.contains('name effect')].itertuples():
        print(r.engine, r.part, r.term, '|', r.detail)


def plot(market=None):
    """Left: change in top-3 rate when the brand name is shown vs the name effect in the hypothetical version.
    Right: how much revealing brand / publisher names changes the ranking, relative to repeating the prompt."""
    import common
    import plotstyle  # noqa: F401
    from plotstyle import plt, ENGINE_COLOR, INK2
    bl = pd.read_csv(RESULTS / 'x14_brand_reveal.csv')
    res = pd.read_csv(RESULTS / 'x14_identity.csv')
    col = {'openai': ENGINE_COLOR['OPENAI'], 'gemini': ENGINE_COLOR['GEMINI']}
    name = {'openai': 'ChatGPT (gpt-5.4-nano)', 'gemini': 'Gemini (3.1-flash-lite)'}
    for mk, figdir in (('all', common.ROOT / 'figures'), ('workorb', common.ROOT / 'figures' / 'workorb')):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={'width_ratios': [1.5, 1]})
        ax = axes[0]
        for eng in ('openai', 'gemini'):
            g = bl[bl.engine == eng]
            hi = g if mk == 'all' else g[g.dataset == 'workorb']
            lo = g.iloc[0:0] if mk == 'all' else g[g.dataset != 'workorb']
            ax.scatter(lo.name_premium_pp, lo.reveal_pp, s=30, color=col[eng], alpha=0.25, edgecolor='none')
            ax.scatter(hi.name_premium_pp, hi.reveal_pp, s=60, color=col[eng], label=name[eng],
                       edgecolor='white', linewidth=1.5)
            for r in hi.itertuples():
                if abs(r.reveal_pp) >= 12 or (mk == 'workorb' and abs(r.reveal_pp) >= 5) or r.brand == 'Google Fiber':
                    dy = 4 if eng == 'openai' else -11              # ChatGPT labels above, Gemini below
                    ax.annotate(r.brand, (r.name_premium_pp, r.reveal_pp), xytext=(5, dy), textcoords='offset points',
                                fontsize=7.5, color=col[eng])
        ax.axhline(0, color='#c3c2b7', lw=1)
        ax.axvline(0, color='#c3c2b7', lw=1)
        ax.set_xlabel('name effect in the hypothetical version (pp of top-3, same facts)')
        ax.set_ylabel('change in top-3 rate when the name is shown (pp)')
        rho = res[res.part.str.contains('vs name_premium')].set_index('engine').value
        ax.set_title(f'Gains follow the name effect (Spearman {rho["openai"]:.2f} ChatGPT, {rho["gemini"]:.2f} Gemini)'
                     , fontsize=9)
        ax.legend(fontsize=8, loc='lower right')
        ax = axes[1]
        rr = res[res.term == 'ratio'].copy()
        rr['what'] = np.where(rr.part.str.contains('brand'), 'brand names', 'publisher names')
        x = np.arange(2)
        for k, eng in enumerate(('openai', 'gemini')):
            g = rr[rr.engine == eng].set_index('what').loc[['brand names', 'publisher names']]
            ax.bar(x + (k - 0.5) * 0.36, g.value, 0.34, color=col[eng], label=name[eng])
            ax.errorbar(x + (k - 0.5) * 0.36, g.value, yerr=[g.value - g.ci_low, g.ci_high - g.value],
                        fmt='none', ecolor=INK2, capsize=3, lw=1)
            for xi, v in zip(x + (k - 0.5) * 0.36, g.value):
                ax.text(xi, 0.3, f'{v:.1f}x', ha='center', fontsize=8.5, color='white', fontweight='bold')
        ax.axhline(1, color=INK2, lw=1, ls='--')
        ax.set_xticks(x, ['brand names shown', 'publisher names shown'])
        ax.set_ylabel('change in top-3 rates\n(1 = same as repeating the prompt)')
        ax.set_title('How much the ranking changes', fontsize=9)
        ax.grid(axis='x', visible=False)
        if mk == 'all':                           # paper version: no embedded title
            fig.tight_layout()
            (common.ROOT / 'figures' / 'paper').mkdir(exist_ok=True)
            fig.savefig(common.ROOT / 'figures' / 'paper' / 'x14_identity.png')
        fig.suptitle('X14: brand and publisher identity', x=0.01, ha='left', fontweight='bold')
        fig.tight_layout()
        figdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figdir / 'x14_identity.png')
        plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['questions', 'publishers', 'build', 'run', 'analyse', 'plot'])
    {'questions': questions, 'publishers': publishers, 'build': build, 'run': run,
     'analyse': analyse, 'plot': plot}[ap.parse_args().cmd]()
