"""Step A2 - turn the daily JSON files into three tables.

  data/tables/answers.csv    one row per answer
  data/tables/citations.csv  one row per cited URL in an answer (in order)
  data/tables/mentions.csv   one row per brand mentioned in an answer

If data/tables/redirect_map.csv exists (made by resolve_redirects.py), Gemini
redirect links are replaced by the real page URL; unresolved ones keep
is_redirect = True so URL-level steps can drop and count them. For those rows
`domain` comes from the citation title, where Gemini writes the site name, so
website-level counts can still include them.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from common import (RAW_DIR, TABLES, LINK, clean_url, domain_of, is_redirect, prompt_id)


def load_redirect_map():
    path = TABLES / 'redirect_map.csv'
    if not path.exists():
        return {}
    m = pd.read_csv(path).dropna(subset=['final_url'])
    return {r.redirect_url: clean_url(r.final_url) for r in m.itertuples()}


def parse_day(day, redirects):
    answers, cites, mentions = [], [], []
    date = str(day['date'])
    for item in day['items']:
        pf = item['pfetch']
        resp = pf.get('chatConvResponse') or {}
        text = resp.get('assistantResponse') or ''
        analysis = pf.get('analysis') or {}
        sw = resp.get('stopwatch') or {}
        aid = item['facetId']
        engine = pf['searchVendorEnum']
        common = dict(answer_id=aid, date=date, engine=engine, topic=pf.get('topicName'))

        clist = [(c.get('url') or c.get('origUrl'), c.get('title'))
                 for c in analysis.get('citations') or []]
        clist = [(u, t) for u, t in clist if u]
        source = 'citations_field'
        if not clist:                                   # fall back to links written in the text
            clist, source = [(u, None) for u in LINK.findall(text)], 'answer_text'
        seen = set()
        for pos, (u, title) in enumerate(clist, start=1):
            cu = clean_url(u)
            cu = redirects.get(cu, cu)
            redirect = is_redirect(cu)
            domain = domain_of(cu)
            if redirect and title and '.' in title and ' ' not in title.strip():
                domain = title.strip().lower().removeprefix('www.')   # Gemini puts the site in title
            cites.append(dict(common, prompt_id=prompt_id(pf['prompt']), position=pos,
                              url=cu, domain=domain, is_redirect=redirect,
                              repeat_in_answer=cu in seen, source=source))
            seen.add(cu)

        brand_rows = analysis.get('brandMentions') or []
        for m in brand_rows:
            rank = m.get('rank')
            mentions.append(dict(common, brand=m.get('brandName'), rank=rank,
                                 rank_missing=rank is None, sentiment=m.get('sentiment'),
                                 mention_text=m.get('mentionText')))

        start, stop = sw.get('startTime'), sw.get('stopTime')
        answers.append(dict(common, model=resp.get('chatModelEnum'),
                            prompt_id=prompt_id(pf['prompt']), prompt=pf['prompt'],
                            answer_chars=len(text), n_citations=len(clist),
                            n_unique_citations=len(seen), citation_source=source,
                            n_mentions=len(brand_rows),
                            latency_ms=(stop - start) if start and stop else None,
                            answer_text=text))
    return answers, cites, mentions


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--raw', default=str(RAW_DIR))
    ap.add_argument('--days', nargs='*', help='only these YYYYMMDD days (default: all)')
    args = ap.parse_args()

    files = sorted(p for p in Path(args.raw).glob('pfetch-*.json'))
    if args.days:
        files = [p for p in files if any(d in p.name for d in args.days)]
    redirects = load_redirect_map()
    A, C, M = [], [], []
    for path in files:
        with open(path, encoding='utf-8') as f:
            day = json.load(f)
        a, c, m = parse_day(day, redirects)
        A += a; C += c; M += m
        print(f'{path.name}: {len(a):5d} answers {len(c):6d} citations {len(m):5d} mentions')

    TABLES.mkdir(parents=True, exist_ok=True)
    answers = pd.DataFrame(A)
    dup = answers.answer_id.duplicated().sum()
    if dup:
        print(f'WARNING: {dup} duplicate answer_id values')
    answers.to_csv(TABLES / 'answers.csv', index=False)
    pd.DataFrame(C).to_csv(TABLES / 'citations.csv', index=False)
    pd.DataFrame(M).to_csv(TABLES / 'mentions.csv', index=False)
    print(f'total: {len(A)} answers, {len(C)} citation rows, {len(M)} brand mentions '
          f'from {len(files)} days; {len(redirects)} redirects mapped')


if __name__ == '__main__':
    main()
