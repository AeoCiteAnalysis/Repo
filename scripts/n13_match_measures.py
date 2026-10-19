"""N13 - Which way of measuring "matches the question" works best?

Question. Is simple word overlap the best measure, or do search-engine (BM25) or meaning-based
(embedding) scores predict citations better?

Rows: the R2 rows (URLs cited >= 2 with page text, websites with >= 2 such URLs).
Page text = title + main-text paragraphs (page_text.jsonl). Measures (each vs ALL client prompts):
  overlap_unigram       Jaccard of word sets (common words removed)
  overlap_uni_bigram    Jaccard of words + word pairs (Step A6 prompt_match_words)
  bm25_max              best BM25 score of the page for any prompt (k1 = 1.5, b = 0.75)
  minilm_title / minilm_best_paragraph   Step A6 embeddings (all-MiniLM-L6-v2)
  bge_title / bge_best_paragraph         second embedding model BAAI/bge-small-en-v1.5
                                         (first 50 paragraphs per page to save time)
Each measure is z-scored and swapped into the R2 model in place of the four Step-A6 match
features (all other R2 variables kept): effect, gain in within-website R², and prediction of the
hold-out period (R9 time split, website level + slopes).
Topic level: for each topic, word overlap and BM25 against that topic's prompts only vs against
all prompts, predicting citations from that topic's prompts.

Outputs: data/tables/n13_match_scores.csv, results/n13_measures.csv, results/n13_topic_match.csv,
         figures/n13_measures.png
"""
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

import plotstyle
from plotstyle import plt, INK2
from common import TABLES, RESULTS, FIGURES, read_table
from fe import fe_ols
from features import TOKEN
from r02_main_model import prepare, FEATURES, SAMPLES
from r09_robustness import period_counts, predict_holdout
from r12_page_type_topic import group_counts

MATCH = ['prompt_match_words', 'title_match', 'intro_match', 'best_paragraph_match']
BGE = 'BAAI/bge-small-en-v1.5'
BGE_QUERY = 'Represent this sentence for searching relevant passages: '
MAX_PARAS = 50


def tokens(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def load_text(urls):
    keep, out = set(urls), {}
    with open(TABLES / 'page_text.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['url'] in keep:
                out[r['url']] = r
    return out


def bm25_matrix(docs, k1=1.5, b=0.75):
    cv = CountVectorizer(tokenizer=tokens, lowercase=False, token_pattern=None)
    D = cv.fit_transform(docs).tocsr().astype(float)
    N = D.shape[0]
    df = np.bincount(D.indices, minlength=D.shape[1])
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
    dl = np.asarray(D.sum(1)).ravel()
    W = D.copy()
    rows = np.repeat(np.arange(N), np.diff(W.indptr))
    tf = W.data
    W.data = idf[W.indices] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[rows] / dl.mean()))
    return cv, W, (D > 0).astype(float)


def query_matrix(cv, prompts):
    Q = cv.transform(prompts)
    Q.data[:] = 1
    return Q.astype(float)


def jaccard_vs(Dbin, qvec):
    inter = Dbin @ qvec
    size_p = np.asarray(Dbin.sum(1)).ravel()
    return inter / (size_p + qvec.sum() - inter)


def bge_scores(pages, prompts):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    m = SentenceTransformer(BGE)
    enc = lambda xs: m.encode(xs, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    P = enc([BGE_QUERY + p for p in prompts])
    title, best = {}, {}
    items = list(pages.items())
    for s in range(0, len(items), 200):
        chunk = items[s:s + 200]
        texts, spans = [], []
        for url, r in chunk:
            paras = r['paragraphs'][:MAX_PARAS]
            spans.append((len(texts), len(paras)))
            texts += [r['title'] or ''] + paras
        sims = (enc(texts) @ P.T).max(1)
        for (url, r), (i, n) in zip(chunk, spans):
            title[url] = sims[i] if r['title'] else np.nan
            best[url] = sims[i + 1:i + 1 + n].max() if n else np.nan
        if (s // 200) % 10 == 0:
            print(f'  bge {min(s + 200, len(items))}/{len(items)}')
    return pd.Series(title), pd.Series(best)


def main():
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    d = d[d.groupby('domain').y.transform('size') >= 2].reset_index(drop=True)
    pages = load_text(d.url)
    d = d[d.url.isin(pages)].reset_index(drop=True)
    labels = pd.read_csv(TABLES / 'prompt_labels.csv')
    prompts = labels.prompt.tolist()
    docs = [(pages[u]['title'] or '') + '\n' + '\n'.join(pages[u]['paragraphs']) for u in d.url]

    cv, W, Dbin = bm25_matrix(docs)
    Q = query_matrix(cv, prompts)
    S = (W @ Q.T).toarray()                                     # pages x prompts
    uni_q = np.asarray((Q.sum(0) > 0)).ravel().astype(float)
    scores = pd.DataFrame({'url': d.url})
    scores['overlap_unigram'] = jaccard_vs(Dbin, uni_q)
    scores['overlap_uni_bigram'] = d.prompt_match_words_raw.to_numpy()
    scores['bm25_max'] = S.max(1)
    scores['minilm_title'] = d.title_match_raw.to_numpy()
    scores['minilm_best_paragraph'] = d.best_paragraph_match_raw.to_numpy()
    cache = TABLES / 'n13_match_scores.csv'
    old = pd.read_csv(cache) if cache.exists() else None
    if old is not None and {'bge_title', 'bge_best_paragraph'} <= set(old) and set(d.url) <= set(old.url):
        o = old.set_index('url')
        scores['bge_title'] = scores.url.map(o.bge_title)
        scores['bge_best_paragraph'] = scores.url.map(o.bge_best_paragraph)
    else:
        bt, bb = bge_scores({u: pages[u] for u in d.url}, prompts)
        scores['bge_title'] = scores.url.map(bt)
        scores['bge_best_paragraph'] = scores.url.map(bb)
    scores.to_csv(cache, index=False)

    # swap each measure into the R2 model
    counts, *_, ratio = period_counts(d.url.tolist())
    d['y_train'] = np.log2(1 + ratio * counts.train.to_numpy())
    d['y_holdout'] = np.log2(1 + counts.holdout.to_numpy())
    base = [f for f in FEATURES if f not in MATCH] + flags + types
    _, info0 = fe_ols(d, 'y', base, 'domain')
    hold0 = predict_holdout(d, base)
    rows = [dict(measure='(none: no match feature)', coef=np.nan, p=np.nan, r2_within=info0['r2_within'],
                 r2_gain=0, holdout_r2=hold0['r2_holdout'], holdout_spearman=hold0['spearman_holdout'])]
    measures = [c for c in scores if c != 'url']
    for m in measures + ['all four Step-A6 measures']:
        dd = d.copy()
        if m in scores:
            v = scores[m].to_numpy()
            v = np.where(np.isnan(v), np.nanmedian(v), v)
            dd['match'] = (v - v.mean()) / v.std()
            X = base + ['match']
        else:
            X = base + MATCH
        res, info = fe_ols(dd, 'y', X, 'domain')
        hold = predict_holdout(dd, X)
        rows.append(dict(measure=m, coef=res.coef.get('match', np.nan), p=res.p.get('match', np.nan),
                         pct_per_sd=100 * (2 ** res.coef.get('match', np.nan) - 1),
                         r2_within=info['r2_within'], r2_gain=info['r2_within'] - info0['r2_within'],
                         holdout_r2=hold['r2_holdout'], holdout_spearman=hold['spearman_holdout']))
    res = pd.DataFrame(rows).sort_values('r2_gain', ascending=False)
    res['n_urls'] = len(d)
    res.to_csv(RESULTS / 'n13_measures.csv', index=False)

    # topic-level match
    _, bt = group_counts(d.url.tolist())
    trows = []
    for topic in sorted(labels.topic.unique()):
        idx = np.where(labels.topic == topic)[0]
        qv = np.asarray((Q[idx].sum(0) > 0)).ravel().astype(float)
        feats = {'overlap_topic': jaccard_vs(Dbin, qv), 'overlap_all': scores.overlap_unigram.to_numpy(),
                 'bm25_topic': S[:, idx].max(1), 'bm25_all': S.max(1)}
        dd = d.assign(y=np.log2(1 + bt[topic].to_numpy()))
        for kind in ('overlap', 'bm25'):
            for k in (f'{kind}_topic', f'{kind}_all'):
                v = feats[k]
                dd[k] = (v - v.mean()) / v.std()
            r, _ = fe_ols(dd, 'y', base + [f'{kind}_topic', f'{kind}_all'], 'domain')
            trows.append(dict(topic=topic, measure=kind,
                              topic_prompts_coef=r.coef[f'{kind}_topic'], topic_prompts_p=r.p[f'{kind}_topic'],
                              all_prompts_coef=r.coef[f'{kind}_all'], all_prompts_p=r.p[f'{kind}_all']))
    tp = pd.DataFrame(trows)
    tp.to_csv(RESULTS / 'n13_topic_match.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    g = res[res.measure != '(none: no match feature)'].iloc[::-1]
    ax.barh(g.measure, 100 * g.r2_gain, color=['#eb6834' if m.startswith('all') else '#2a78d6' for m in g.measure],
            height=0.65)
    for y, v in enumerate(100 * g.r2_gain):
        ax.text(v + 0.02, y, f'+{v:.2f}', va='center', fontsize=7.5, color=INK2)
    ax.set_xlabel('gain in within-website R² (percentage points) over a model without match features')
    ax.grid(axis='y', visible=False)
    ax.set_title(f'N13: which "matches the question" measure explains citations best ({len(d):,} URLs)')
    fig.savefig(FIGURES / 'n13_measures.png'); plt.close(fig)

    print(res.round(4).to_string(index=False))
    print(tp.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
