"""X14 step 1 - fixed product fact sheets, masking alias lists and a leak test.

Shared input for the brand/publisher identity experiment (X14) and the controlled
ranking-stability experiment (X15). One fact sheet per brand, written from the brand's own
downloaded pages into a fixed template of fixed length, in which the company is only ever
called {BRAND}: the same sheet is shown masked ("Vendor K") or named, and nothing else changes.

  python scripts/x14_fact_sheets.py aliases   # LLM-proposed alias lists (+ our patterns/domains) -> review
  python scripts/x14_fact_sheets.py sheets    # evidence selection + extraction + checks
  python scripts/x14_fact_sheets.py leak      # can the experiment's models name the masked brand?

Outputs in data/x14/: aliases.json, evidence/<brand>.txt, sheets.json, leak.jsonl;
results/x14_leak.csv.
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'data' / 'x14'
RESULTS = ROOT / 'results'
EXTRACT_MODEL = 'gpt-5.4-mini'
MODELS = {'openai': 'gpt-5.4-nano', 'gemini': 'gemini-3.1-flash-lite'}   # same as X12 and the experiment
SEED = 42

MARKETS = {
    'gfiber': dict(tables=ROOT / 'data' / 'tables', kind='home internet service provider',
                   brands=['Google Fiber', 'AT&T Fiber', 'Xfinity', 'Quantum Fiber', 'Spectrum', 'T-Mobile',
                           'Metronet', 'Astound', 'Verizon'],
                   topic_words=r'internet|fiber|wi-?fi|broadband|speed|mbps|gig',
                   fields=[('service', 'type of service and network technology'),
                           ('plans', 'plan tiers and monthly prices (numbers only, no plan names)'),
                           ('speeds', 'download and upload speeds'),
                           ('terms', 'contract, fees, data caps and price guarantees'),
                           ('equipment', 'router, Wi-Fi equipment and installation'),
                           ('support', 'reliability, customer support and satisfaction claims'),
                           ('extras', 'other notable features, bundles or perks')]),
    'workorb': dict(tables=ROOT / 'data' / 'workorb' / 'tables', kind='proposal and RFP software vendor',
                    brands=['Workorb', 'Loopio', 'Unanet', 'OpenAsset', 'Joist AI', 'Deltek', 'Responsive',
                            'AutoGenAI', 'Inventive AI', 'Arphie'],
                    topic_words=r'proposal|rfp|bid|response|content|ai|integrat|pricing|customer|aec|architect|engineer',
                    fields=[('product', 'what the product is and who it is for'),
                            ('features', 'core proposal and RFP features'),
                            ('ai', 'AI and automation capabilities'),
                            ('integrations', 'integrations with other software (generic categories, e.g. "CRM")'),
                            ('aec_fit', 'fit for architecture, engineering and construction firms'),
                            ('pricing', 'pricing and plans'),
                            ('support', 'implementation, onboarding, support and security')]),
}
SHEET_WORDS = (150, 195)
PLACEHOLDER = '{BRAND}'


def ds_brand_patterns(ds):
    """Our existing name patterns (prompt_labels.BRANDS_BY_DATASET) without importing the dataset switch."""
    src = (ROOT / 'scripts' / 'prompt_labels.py').read_text(encoding='utf-8')
    ns = {}
    block = re.search(r"BRANDS_BY_DATASET = \{\}\n(.*?)\nBRANDS = ", src, re.S).group(1)
    exec('BRANDS_BY_DATASET = {}\n' + block, ns)
    return ns['BRANDS_BY_DATASET'][ds]


def brand_domains(ds):
    b = pd.read_csv(MARKETS[ds]['tables'] / 'brand_domains.csv')
    b = b[b.role.isin(['own', 'competitor', 'client', 'mixed'])]
    return b


# ---------------------------------------------------------------- aliases

ALIAS_INSTRUCTIONS = """You help anonymize product descriptions. For the company named by the user (a {kind}),
list every string that would reveal its identity if it appeared in a text: company names and
abbreviations, former names, parent and subsidiary companies, product and plan names, trademarked
feature names, slogans, website domains, app names, and names of well-known executives.
Return JSON only: {{"aliases": ["...", ...]}}. Include spelling variants. Do not include generic words."""


def aliases():
    import llm
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for ds, m in MARKETS.items():
        pats = ds_brand_patterns(ds)
        doms = brand_domains(ds)
        for br in m['brands']:
            raw = llm.ask(ALIAS_INSTRUCTIONS.format(kind=m['kind']), br, model=EXTRACT_MODEL)
            try:
                llm_al = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))['aliases']
            except Exception:
                llm_al = []
            d = sorted(doms[doms.brand == br].domain)
            out[br] = dict(dataset=ds, llm=sorted(set(a.strip() for a in llm_al if len(a.strip()) > 1)),
                           domains=d, pattern=pats.get(br), manual_add=[], manual_remove=[])
            print(f'{br}: {len(llm_al)} LLM aliases, {len(d)} domains', flush=True)
    p = OUT / 'aliases.json'
    if p.exists():                                             # keep hand edits
        old = json.loads(p.read_text(encoding='utf-8'))
        for br, v in old.items():
            if br in out:
                out[br]['manual_add'], out[br]['manual_remove'] = v.get('manual_add', []), v.get('manual_remove', [])
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'-> {p}  (review: manual_add / manual_remove)')


def alias_regex(entry):
    """One case-insensitive regex for all identity strings of a brand."""
    terms = set(entry['llm']) | set(entry['manual_add'])
    terms -= set(entry['manual_remove'])
    terms |= set(entry['domains'])                            # full domains only (stems such as "play" are generic)
    alts = [re.escape(t).replace(r'\ ', r'[\s-]?') for t in sorted(terms, key=len, reverse=True) if len(t) > 1]
    if entry.get('pattern'):
        alts.append(entry['pattern'])
    return re.compile(r'(?<![A-Za-z0-9])(?:' + '|'.join(alts) + r')(?![A-Za-z0-9])', re.I)


def load_aliases():
    a = json.loads((OUT / 'aliases.json').read_text(encoding='utf-8'))
    return a, {br: alias_regex(v) for br, v in a.items()}


# ---------------------------------------------------------------- evidence + sheets

def brand_pages(ds, brand):
    """Paragraphs of the brand's own downloaded pages (page_text + X6 extras), deduplicated."""
    doms = set(brand_domains(ds).query('brand == @brand').domain)
    t = MARKETS[ds]['tables']
    seen, paras = set(), []
    for f in ('page_text.jsonl', 'x06_parsed.jsonl'):
        if not (t / f).exists():
            continue
        with open(t / f, encoding='utf-8') as fh:
            for line in fh:
                r = json.loads(line)
                u = r.get('url') or ''
                h = urlparse(u).netloc.lower().removeprefix('www.')
                if u in seen or not any(h == d or h.endswith('.' + d) for d in doms):
                    continue
                seen.add(u)
                for p in r.get('paragraphs') or []:
                    paras.append((u, (r.get('title') or '').strip(), p.strip()))
    return paras


COMPARISON_URL = re.compile(r'[-/_](?:vs|versus|alternatives?|compare|comparison|competitors?)[-/_.]|/best-', re.I)
# names of third-party products that the sheets must describe generically ("a major CRM")
THIRD_PARTY = re.compile(r'(?<![A-Za-z])(?:salesforce|hubspot|slack|sharepoint|microsoft|google|egnyte|indesign|'
                         r'confluence|notion|seismic|highspot|eero|netflix|hulu|disney|peacock|apple|amazon|'
                         r'excel|outlook|microsoft teams|zoom|dropbox|box\.com|procore|autodesk|revit)(?![A-Za-z])', re.I)
META = re.compile(r'\bevidence (?:also )?(?:says|states|describes|mentions|shows|lists|gives)|'
                  r'\b(?:one|another|a) sources? (?:says|states|lists|mentions)|not stated\.\s*\S', re.I)


def select_evidence(ds, brand, budget=3500):
    """Keyword-scored paragraphs, spread over the template fields, within a word budget."""
    m = MARKETS[ds]
    topic = re.compile(m['topic_words'], re.I)
    field_words = {f: set(re.findall(r'[a-z]{4,}', desc.lower())) for f, desc in m['fields']}
    al, rx = load_aliases()
    others = [r for b, r in rx.items() if b != brand and al[b]['dataset'] == ds]
    seen, cands = set(), []
    for u, title, p in brand_pages(ds, brand):
        n = len(p.split())
        key = p[:120].lower()
        if n < 15 or n > 250 or key in seen or not topic.search(p):
            continue
        if COMPARISON_URL.search(u) or any(r.search(p) for r in others):
            continue                                         # the brand's own comparison posts describe competitors
        seen.add(key)
        low = p.lower()
        nums = len(re.findall(r'\$\s?\d|\d+\s?(?:mbps|gbps|gig|%|/mo)', low))
        cands.append(dict(url=u, title=title, text=p, n=n, nums=nums,
                          score={f: sum(w in low for w in ws) for f, ws in field_words.items()}))
    chosen, used = [], 0
    for rnd in range(40):                                      # round robin over fields
        added = False
        for f, _ in m['fields']:
            pool = [c for c in cands if c not in chosen and c['score'][f] > 0]
            if not pool:
                continue
            c = max(pool, key=lambda c: (c['score'][f] + 0.5 * min(c['nums'], 4)) / (1 + c['n'] / 150))
            if used + c['n'] > budget:
                continue
            chosen.append(c)
            used += c['n']
            added = True
        if not added:
            break
    return chosen


SHEET_INSTRUCTIONS = """You write a neutral, factual product profile of one {kind} for a comparison study.
Use ONLY facts stated in the evidence below. Write in plain third person.

Strict rules:
- Refer to the company ONLY as {{BRAND}} (literally, with the braces). Never write the company's name,
  former names, parent company, product names, plan names, trademarked feature names, website, app
  names, people or slogans. Describe products generically ("its fastest plan", "its AI assistant").
- Do not name ANY other company or product either (partners, integrations, competitors, parent
  companies); describe them generically ("a major CRM", "a streaming service", "a leading AEC ERP system").
- Do not name specific cities, states or regions; say "selected U.S. markets" or "nationwide" if stated.
- Keep numbers (prices, speeds, fees, counts) exactly as stated in the evidence.
- No marketing adjectives ("best-in-class", "industry-leading") unless attributed as a claim ("{{BRAND}} says ...").
- If the evidence says nothing on a field, write exactly "Not stated." and nothing else. Never mention the
  evidence, sources or what is missing ("evidence says", "one source lists").
Return JSON only, with exactly these keys. Each value is one or two sentences of AT MOST 27 words;
the values together must have {lo}-{hi} words (count them):
{fields}"""


def render_sheet(sheet, fields, name):
    return '\n'.join(f'- {desc.split(" (")[0].capitalize()}: {sheet[f].replace(PLACEHOLDER, name)}'
                     for f, desc in fields)


def sheet_words(sheet):
    return sum(len(v.split()) for v in sheet.values())


FIELD_MAX = 30
SHORTEN = (f'Shorten this sentence to at most {FIELD_MAX - 5} words. Keep every number and the placeholder '
           '{BRAND} exactly; drop the least important details. Return the shortened text only.')


def check_sheet(sheet, ds, al, rx):
    problems = []
    for other, r in rx.items():                             # any brand of the same market, not only its own
        if al[other]['dataset'] != ds:
            continue
        hits = {h.group(0) for v in sheet.values() for h in r.finditer(v.replace(PLACEHOLDER, ''))}
        if hits:
            problems.append(f'identity strings of {other} present: {sorted(hits)[:5]}')
    tp = sorted({h.group(0) for v in sheet.values() for h in THIRD_PARTY.finditer(v)})
    if tp:
        problems.append(f'names other companies or products {tp}: describe them generically')
    meta = [f for f, v in sheet.items() if META.search(v)]
    if meta:
        problems.append(f'fields {meta} mention evidence/sources or add text after "Not stated."')
    long = {f: len(v.split()) for f, v in sheet.items() if len(v.split()) > FIELD_MAX}
    if long:
        problems.append('fields over 27 words: ' + ', '.join(f'"{f}" has {n}' for f, n in long.items())
                        + ' - shorten them, keep the numbers')
    w = sheet_words(sheet)
    if not SHEET_WORDS[0] - 10 <= w <= SHEET_WORDS[1] + 10:
        problems.append(f'{w} words in total, need {SHEET_WORDS[0]}-{SHEET_WORDS[1]}')
    return problems


def sheets():
    import common  # noqa: F401  loads .env
    import llm
    al, rx = load_aliases()
    (OUT / 'evidence').mkdir(parents=True, exist_ok=True)
    path = OUT / 'sheets.json'
    done = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for ds, m in MARKETS.items():
        fields_txt = '\n'.join(f'"{f}": {desc}' for f, desc in m['fields'])
        ins = SHEET_INSTRUCTIONS.format(kind=m['kind'], lo=SHEET_WORDS[0], hi=SHEET_WORDS[1], fields=fields_txt)
        for br in m['brands']:
            if br in done and done[br].get('ok'):
                continue
            ev = select_evidence(ds, br)
            ev_txt = '\n\n'.join(f'[{c["title"]}] {c["text"]}' for c in ev)
            (OUT / 'evidence' / f'{re.sub(r"[^A-Za-z0-9]+", "_", br)}.txt').write_text(
                '\n\n'.join(f'{c["url"]}\n{c["text"]}' for c in ev), encoding='utf-8')
            sheet, problems = None, ['no answer']
            for attempt in range(6):
                text = f'Company: {br}\n\nEvidence:\n{ev_txt}'
                if attempt and sheet:                     # revise the previous draft instead of starting over
                    text += (f'\n\nYour previous JSON:\n{json.dumps(sheet, ensure_ascii=False)}\n\n'
                             f'Return the full corrected JSON, fixing only these problems: {"; ".join(problems)}')
                raw = llm.ask(ins, text, model=EXTRACT_MODEL)
                try:
                    sheet = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))
                    sheet = {f: str(sheet.get(f, 'Not stated.')).strip() for f, _ in m['fields']}
                except Exception:
                    problems = ['answer was not valid JSON']
                    continue
                problems = check_sheet(sheet, ds, al, rx)
                if not problems:
                    break
            if sheet and problems and all(p.startswith('fields over') or ' words in total' in p for p in problems):
                for f, v in sheet.items():                # length only: shorten the long fields one at a time
                    for _ in range(3):
                        if len(v.split()) <= FIELD_MAX:
                            break
                        v = llm.ask(SHORTEN, v, model=EXTRACT_MODEL).strip().strip('"')
                    sheet[f] = v
                problems = check_sheet(sheet, ds, al, rx)
            done[br] = dict(dataset=ds, sheet=sheet, words=sheet_words(sheet) if sheet else 0,
                            evidence_paragraphs=len(ev), evidence_words=sum(c['n'] for c in ev),
                            problems=problems, ok=not problems)
            print(f'{br:14s} {done[br]["words"]:4d} words from {len(ev)} paragraphs'
                  f'{"" if not problems else "  PROBLEMS: " + "; ".join(problems)}', flush=True)
            path.write_text(json.dumps(done, indent=1, ensure_ascii=False), encoding='utf-8')
    review = OUT / 'sheets_review.md'
    with open(review, 'w', encoding='utf-8') as f:
        for br, v in done.items():
            if v.get('sheet'):
                f.write(f'## {br} ({v["dataset"]}, {v["words"]} words)\n\n'
                        f'{render_sheet(v["sheet"], MARKETS[v["dataset"]]["fields"], "Vendor K")}\n\n')
    print(f'-> {path}, {review}')


# ---------------------------------------------------------------- leak test

LEAK_OPEN = ('Below is an anonymized profile of a real {kind} in the United States, called "Vendor K". '
             'Which company is it? Answer with the company name only; if you cannot tell, give your best guess.')
LEAK_CLOSED = ('Below is an anonymized profile of a real {kind}, called "Vendor K". It is one of: {options}. '
               'Which one? Answer with one name from the list only.')


def ask(engine, instructions, text):
    if engine == 'openai':
        from openai import OpenAI
        return OpenAI().responses.create(model=MODELS['openai'], instructions=instructions, input=text).output_text
    import requests
    body = {'systemInstruction': {'parts': [{'text': instructions}]},
            'contents': [{'role': 'user', 'parts': [{'text': text}]}]}
    for attempt in range(5):
        r = requests.post(f'https://generativelanguage.googleapis.com/v1beta/models/{MODELS["gemini"]}:generateContent',
                          headers={'x-goog-api-key': os.environ['GEMINI_API_KEY']}, json=body, timeout=120)
        if r.status_code == 200:
            break
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()
    cand = (r.json().get('candidates') or [{}])[0]
    return ''.join(p.get('text', '') for p in cand.get('content', {}).get('parts', []))


def leak(reps=3):
    import common  # noqa: F401
    al, rx = load_aliases()
    sh = json.loads((OUT / 'sheets.json').read_text(encoding='utf-8'))
    log = OUT / 'leak.jsonl'
    done = set()
    if log.exists():
        done = {json.loads(l)['key'] for l in open(log, encoding='utf-8')}
    rng = random.Random(SEED)
    jobs = []
    for br, v in sh.items():
        if not v.get('sheet'):
            continue
        m = MARKETS[v['dataset']]
        text = render_sheet(v['sheet'], m['fields'], 'Vendor K')
        for eng in MODELS:
            for rep in range(reps):
                opts = m['brands'][:]
                rng.shuffle(opts)
                for mode, ins in (('open', LEAK_OPEN.format(kind=m['kind'])),
                                  ('closed', LEAK_CLOSED.format(kind=m['kind'], options=', '.join(opts)))):
                    k = f'{br}|{eng}|{mode}|{rep}'
                    if k not in done:
                        jobs.append(dict(key=k, brand=br, dataset=v['dataset'], engine=eng, mode=mode, rep=rep,
                                         ins=ins, text=text, k_options=len(opts)))
    print(f'{len(jobs)} leak calls', flush=True)
    lock = threading.Lock()

    def one(j):
        try:
            a = ask(j['engine'], j['ins'], j['text']).strip()
        except Exception as e:
            print(f'  {j["key"]}: {type(e).__name__}', flush=True)
            return
        rec = {k: v for k, v in j.items() if k not in ('ins', 'text')} | dict(answer=a)
        with lock, open(log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, jobs))
    rows = [json.loads(l) for l in open(log, encoding='utf-8')]
    d = pd.DataFrame(rows)
    d['correct'] = [bool(rx[b].search(a)) for b, a in zip(d.brand, d.answer)]
    s = (d.groupby(['dataset', 'brand', 'engine', 'mode']).correct.mean().unstack(['engine', 'mode']) * 100).round(0)
    print(s.to_string())
    t = d.groupby(['dataset', 'engine', 'mode']).agg(pct_correct=('correct', 'mean'), n=('correct', 'size'),
                                                      k=('k_options', 'first')).reset_index()
    t['pct_correct'] *= 100
    t['chance_closed'] = 100 / t.k
    print(t.round(1).to_string(index=False))
    d.drop(columns=[]).to_csv(RESULTS / 'x14_leak_calls.csv', index=False)
    t.to_csv(RESULTS / 'x14_leak.csv', index=False)
    s.to_csv(RESULTS / 'x14_leak_by_brand.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['aliases', 'sheets', 'leak'])
    {'aliases': aliases, 'sheets': sheets, 'leak': leak}[ap.parse_args().cmd]()
