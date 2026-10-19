"""Step A6 - measure every downloaded page.

  python scripts/features.py [--round 2026-09-26] [--workers 6] [--no-embed]

Reads data/pages/<round>/*.html (pages with status 'ok' in pages_index.csv) and writes
  data/tables/page_features.csv      one row per URL (features of Step A6)
  data/tables/page_text.jsonl        url, title, paragraphs  (reused by N10, N13)
  results/page_check_sample.csv      50 random pages to hand-check (written once, never overwritten)

Match features use ALL prompts of the client, never only the prompts that
cited the page, so the score is not built from the outcome we try to predict.
page_age_days is added later in build_url_table.py, because it needs the
citation dates.
"""
import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from common import TABLES, PAGES, RESULTS, SEED, read_table

EMBED_MODEL = 'all-MiniLM-L6-v2'
MAX_PARAS = 300
TOKEN = re.compile(r"[a-z0-9][a-z0-9'\-]*")


def terms(text):
    """Set of lower-case words and two-word pairs, common words removed."""
    w = [t for t in TOKEN.findall((text or '').lower()) if t not in ENGLISH_STOP_WORDS]
    return set(w) | {f'{a} {b}' for a, b in zip(w, w[1:])}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else np.nan


def _parse(args):
    path, url = args
    from page_parse import parse_html
    try:
        html = path.read_text(encoding='utf-8', errors='replace')
        return url, parse_html(html, url), None
    except Exception as e:                       # keep going; report at the end
        return url, None, f'{type(e).__name__}: {e}'


def embed_matches(pages, prompts):
    """title_match, intro_match, best_paragraph_match, best_paragraph_depth per page."""
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)                    # faster than all 16 on this CPU (measured)
    model = SentenceTransformer(EMBED_MODEL)
    enc = lambda xs: model.encode(xs, batch_size=128, normalize_embeddings=True,
                                  show_progress_bar=False)
    P = enc(prompts)                                              # (n_prompts, d)
    items = list(pages.items())
    out = {}
    for start in range(0, len(items), 200):                      # encode 200 pages at a time
        chunk = items[start:start + 200]
        texts, spans = [], []
        for url, p in chunk:
            paras = p['paragraphs'][:MAX_PARAS]
            spans.append((len(texts), len(paras)))
            texts += [p['title'] or ''] + paras
        sims = (enc(texts) @ P.T).max(axis=1)                     # best prompt per text
        for (url, p), (i, n) in zip(chunk, spans):
            para_s = sims[i + 1:i + 1 + n]
            out[url] = dict(
                title_match=float(sims[i]) if p['title'] else np.nan,
                intro_match=float(para_s[0]) if n else np.nan,
                best_paragraph_match=float(para_s.max()) if n else np.nan,
                best_paragraph_depth=(float(para_s.argmax() / (n - 1)) if n > 1
                                      else (0.0 if n else np.nan)))
        print(f'  embedded {min(start + 200, len(items))}/{len(items)} pages')
    return pd.DataFrame.from_dict(out, orient='index')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', help='download round (default: latest in pages_index)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--no-embed', action='store_true', help='skip sentence-embedding features')
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()

    idx = pd.read_csv(TABLES / 'pages_index.csv', dtype=str)
    rnd = a.round or sorted(idx['round'].unique())[-1]
    ok = idx[(idx['round'] == rnd) & (idx.status == 'ok')].drop_duplicates('url', keep='last')
    if a.limit:
        ok = ok.head(a.limit)
    jobs = [(PAGES / rnd / f'{r.id}.html', r.url) for r in ok.itertuples()]
    print(f'round {rnd}: parsing {len(jobs)} pages')

    parsed, errors = {}, []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (url, p, err) in enumerate(ex.map(_parse, jobs, chunksize=16), 1):
            if p is None:
                errors.append((url, err))
            else:
                parsed[url] = p
            if i % 1000 == 0:
                print(f'  parsed {i}/{len(jobs)}')
    print(f'parsed {len(parsed)}, failed {len(errors)}')

    with open(TABLES / 'page_text.jsonl', 'w', encoding='utf-8') as f:
        for url, p in parsed.items():
            f.write(json.dumps(dict(url=url, round=rnd, title=p['title'],
                                    paragraphs=p['paragraphs']), ensure_ascii=False) + '\n')

    prompts = sorted(read_table('answers.csv', usecols=['prompt']).prompt.dropna().unique())
    prompt_terms = set().union(*(terms(q) for q in prompts))
    feats = pd.DataFrame.from_dict(
        {u: {k: v for k, v in p.items() if k != 'paragraphs'} for u, p in parsed.items()},
        orient='index')
    feats['prompt_match_words'] = [
        jaccard(terms(p['title'] + '\n' + '\n'.join(p['paragraphs'])), prompt_terms)
        for p in parsed.values()]
    if not a.no_embed:
        feats = feats.join(embed_matches(parsed, prompts))
    feats.index.name = 'url'
    feats['round'] = rnd
    feats = feats.reset_index()
    feats.to_csv(TABLES / 'page_features.csv', index=False)
    print(f'wrote {len(feats)} rows -> data/tables/page_features.csv')
    print(feats.page_type.value_counts().to_string())

    # never overwrite: the sheet holds the hand checks reported in Appendix A
    check = RESULTS / 'page_check_sample.csv'
    if not check.exists():
        sample = feats.sample(min(50, len(feats)), random_state=SEED)[
            ['url', 'title', 'page_type', 'page_date', 'word_count']]
        sample.assign(type_ok='', date_ok='', text_ok='', markers_ok='', role_ok='').to_csv(
            check, index=False)
    if errors:
        pd.DataFrame(errors, columns=['url', 'error']).to_csv(
            RESULTS / 'page_parse_errors.csv', index=False)


if __name__ == '__main__':
    main()
