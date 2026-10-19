"""X2 - Prompt-level matching (follow-up to R13 / N13).

R13 left ~72 % of the variation in citation counts stable but unexplained. The Step-A6 match
features compare a page with ALL 651 prompts at once; a page written for a few specific questions
scores average on them. Here matching is measured per page x prompt.

Unit: each R2 page (URLs cited >= 2 with page text) x each of the 651 prompts.
Match scores per pair:
  emb_best    best cosine similarity of the prompt with the page title or any of its first 50
              paragraphs (all-MiniLM-L6-v2, as in Step A6)
  emb_title   similarity of the prompt with the page title
  bm25        BM25 score of the page for the prompt (as in N13)
Outcome per pair: was the page cited for that prompt (any engine, any day)?

A. Within a page: is a page cited for the prompts it matches best?
   - AUC per page: chance that a prompt the page IS cited for has a higher match than one it is
     not cited for (0.5 = no link); averaged over pages, overall and per engine.
   - Conditional logit grouped by page (each page compared only with itself), on all cited pairs
     plus 30 random uncited prompts per page, controlling for how many sources each prompt draws.
B. Across pages: does the NUMBER of prompts a page matches well explain its total citations?
   Breadth features (no outcome information used): number of prompts with emb_best in the top 5 %
   of all pairs, and the mean of the page's 10 best prompt similarities. Added to the R2 model
   (website indicators); gain in within-website R², and share of the within-website variance
   explained in a mixed model (R13: 3.6 % with the Step-A6 features).

Outputs: data/tables/x02_sims.npz (cache), results/x02_within_page.csv,
         results/x02_conditional_logit.csv, results/x02_breadth.csv, figures/x02_prompt_match.png
"""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

import plotstyle
from plotstyle import plt, ENGINE_COLOR, INK2, MUTED
from common import TABLES, RESULTS, FIGURES, SEED, ENGINES, ENGINE_LABEL, read_table, drop_excluded
from fe import fe_ols
from r02_main_model import prepare, FEATURES
from n13_match_measures import load_text, bm25_matrix, query_matrix

warnings.filterwarnings('ignore')
MAX_PARAS = 50
CACHE = TABLES / 'x02_sims.npz'
NEG_PER_PAGE = 30


def similarities(urls, pages, prompts):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z['urls']) == list(urls) and len(z['prompts']) == len(prompts):
            return z['emb_best'], z['emb_title'], z['bm25']
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    m = SentenceTransformer('all-MiniLM-L6-v2')
    enc = lambda xs: m.encode(xs, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    P = enc(list(prompts))
    best = np.zeros((len(urls), len(prompts)), dtype=np.float32)
    title = np.full((len(urls), len(prompts)), np.nan, dtype=np.float32)
    for s in range(0, len(urls), 200):
        chunk = urls[s:s + 200]
        texts, spans = [], []
        for u in chunk:
            r = pages[u]
            paras = r['paragraphs'][:MAX_PARAS]
            spans.append((len(texts), len(paras), bool(r['title'])))
            texts += [r['title'] or ''] + paras
        S = enc(texts) @ P.T                                   # texts x prompts
        for j, (i, n, has_title) in enumerate(spans):
            block = S[i:i + 1 + n] if has_title else S[i + 1:i + 1 + n]
            best[s + j] = block.max(0) if len(block) else np.nan
            if has_title:
                title[s + j] = S[i]
        if (s // 200) % 10 == 0:
            print(f'  embedded {min(s + 200, len(urls))}/{len(urls)} pages')
    docs = [(pages[u]['title'] or '') + '\n' + '\n'.join(pages[u]['paragraphs']) for u in urls]
    cv, W, _ = bm25_matrix(docs)
    bm = (W @ query_matrix(cv, list(prompts)).T).toarray().astype(np.float32)
    np.savez_compressed(CACHE, urls=np.array(urls), prompts=np.array(list(prompts)),
                        emb_best=best, emb_title=title, bm25=bm)
    return best, title, bm


def cited_matrix(urls, pids):
    c = drop_excluded(read_table('citations.csv', usecols=['date', 'engine', 'prompt_id', 'url',
                                                          'is_redirect', 'repeat_in_answer']))
    c = c[~c.is_redirect & ~c.repeat_in_answer & c.url.isin(set(urls))]
    ui = {u: i for i, u in enumerate(urls)}
    pi = {p: j for j, p in enumerate(pids)}
    out = {}
    for eng in ['all'] + ENGINES:
        g = c if eng == 'all' else c[c.engine == eng]
        g = g.groupby(['url', 'prompt_id']).size().reset_index(name='n')
        M = np.zeros((len(urls), len(pids)), dtype=np.int32)
        M[g.url.map(ui).to_numpy(), g.prompt_id.map(pi).to_numpy()] = g.n.to_numpy()
        out[eng] = M
    return out


def within_page_auc(score, C):
    aucs = []
    for i in range(C.shape[0]):
        y = C[i] > 0
        if 0 < y.sum() < len(y):
            s = score[i]
            ok = ~np.isnan(s)
            if y[ok].sum() and (~y[ok]).sum():
                aucs.append(roc_auc_score(y[ok], s[ok]))
    return np.array(aucs)


def conditional_logit(Z, C, load, rng):
    from statsmodels.discrete.conditional_models import ConditionalLogit
    rows = []
    for i in range(C.shape[0]):
        pos = np.where(C[i] > 0)[0]
        neg = np.where(C[i] == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        neg = rng.choice(neg, size=min(NEG_PER_PAGE, len(neg)), replace=False)
        for j in np.concatenate([pos, neg]):
            rows.append((i, j, int(C[i, j] > 0)))
    r = np.array(rows)
    X = pd.DataFrame({k: v[r[:, 0], r[:, 1]] for k, v in Z.items()})
    X['log_prompt_sources'] = np.log(load[r[:, 1]])
    X = X.fillna(X.median())
    m = ConditionalLogit(r[:, 2], X, groups=r[:, 0]).fit(disp=0)
    ci = m.conf_int()
    return pd.DataFrame({'term': m.params.index, 'odds_ratio_per_sd': np.exp(m.params.values),
                         'ci_low': np.exp(ci[0].values), 'ci_high': np.exp(ci[1].values),
                         'p': m.pvalues.values}).assign(n_pairs=len(r), n_pages=len(np.unique(r[:, 0])))


def main():
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    d = d[d.groupby('domain').y.transform('size') >= 2]
    pages = load_text(d.url)
    d = d[d.url.isin(pages)].reset_index(drop=True)
    urls = d.url.tolist()
    lab = pd.read_csv(TABLES / 'prompt_labels.csv')
    pids, prompts = lab.prompt_id.tolist(), lab.prompt.tolist()
    best, title, bm = similarities(urls, pages, prompts)
    C = cited_matrix(urls, pids)
    zs = lambda M: (M - np.nanmean(M)) / np.nanstd(M)
    Z = {'emb_best': zs(best), 'emb_title': zs(title), 'bm25': zs(bm)}

    # A. within page
    rows = []
    for eng in ['all'] + ENGINES:
        for name, M in (('emb_best', best), ('emb_title', title), ('bm25', bm)):
            a = within_page_auc(M, C[eng])
            if len(a):
                rows.append(dict(engine=ENGINE_LABEL.get(eng, eng), score=name, pages=len(a),
                                 mean_auc=a.mean(), median_auc=np.median(a),
                                 share_pages_auc_above_0_5=(a > 0.5).mean()))
    wp = pd.DataFrame(rows)
    wp.to_csv(RESULTS / 'x02_within_page.csv', index=False)
    load = (C['all'] > 0).sum(0) + 1                      # how many R2 pages each prompt cites
    cl = conditional_logit(Z, C['all'], load, np.random.default_rng(SEED))
    cl.to_csv(RESULTS / 'x02_conditional_logit.csv', index=False)

    # B. breadth features in the R2 model
    thr = np.nanpercentile(best, 95)
    d['breadth_strong'] = (best >= thr).sum(1)
    d['top10_mean_sim'] = np.sort(np.nan_to_num(best, nan=-1), axis=1)[:, -10:].mean(1)
    for f in ('breadth_strong', 'top10_mean_sim'):
        d[f] = (d[f] - d[f].mean()) / d[f].std()
    X0 = FEATURES + flags + types
    res0, i0 = fe_ols(d, 'y', X0, 'domain')
    brows = [dict(model='R2 features (Step-A6 match features)', r2_within=i0['r2_within'])]
    for extra in (['breadth_strong'], ['top10_mean_sim'], ['breadth_strong', 'top10_mean_sim']):
        res, info = fe_ols(d, 'y', X0 + extra, 'domain')
        brows.append(dict(model='R2 features + ' + ' + '.join(extra), r2_within=info['r2_within'],
                          **{f'{e}_pct_per_sd': 100 * (2 ** res.loc[e, 'coef'] - 1) for e in extra},
                          **{f'{e}_p': res.loc[e, 'p'] for e in extra}))
    # mixed-model share of within-website variance explained
    null = sm.MixedLM(d.y, np.ones((len(d), 1)), groups=d.domain).fit(reml=True).scale
    for label, X in (('Step-A6 features', X0), ('+ breadth', X0 + ['breadth_strong', 'top10_mean_sim'])):
        sc = sm.MixedLM(d.y, sm.add_constant(d[X]), groups=d.domain).fit(reml=True).scale
        brows.append(dict(model=f'mixed model, within-website variance explained ({label})',
                          r2_within=1 - sc / null))
    br = pd.DataFrame(brows)
    br.to_csv(RESULTS / 'x02_breadth.csv', index=False)

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    g = wp[wp.score == 'emb_best']
    cols = ['#898781'] + [ENGINE_COLOR[e] for e in ENGINES]
    axes[0].bar(g.engine, g.mean_auc, color=cols[:len(g)], width=0.6)
    for x, v, n in zip(range(len(g)), g.mean_auc, g.pages):
        axes[0].text(x, v + 0.005, f'{v:.2f}\n({n:,} pages)', ha='center', fontsize=7.5, color=INK2)
    axes[0].axhline(0.5, color=MUTED, ls='--', lw=0.8)
    axes[0].set_ylim(0.4, 1); axes[0].set_ylabel('mean AUC within page')
    axes[0].set_title('Is a page cited for the prompts it matches best?\n(0.5 = no link)', fontsize=9.5)
    axes[0].grid(axis='x', visible=False)
    ax = axes[1]
    for name, col in (('emb_best', '#2a78d6'), ('bm25', '#eb6834')):
        pos = np.concatenate([M[C['all'][i] > 0] for i, M in enumerate(Z[name])])
        neg = np.concatenate([M[C['all'][i] == 0] for i, M in enumerate(Z[name])])
        ax.hist(neg[~np.isnan(neg)], bins=60, density=True, histtype='step', color=col, ls='--', lw=1.2,
                label=f'{name}: prompts not cited for')
        ax.hist(pos[~np.isnan(pos)], bins=60, density=True, histtype='step', color=col, lw=2,
                label=f'{name}: prompts cited for')
    ax.set_xlim(-3, 5); ax.set_xlabel('match score (z)'); ax.legend(fontsize=7)
    ax.set_title('Match of page and prompt: cited vs not cited pairs', fontsize=9.5)
    fig.suptitle('X2: prompt-level matching', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(); fig.savefig(FIGURES / 'x02_prompt_match.png'); plt.close(fig)

    print(wp.round(3).to_string(index=False))
    print(cl.round(4).to_string(index=False))
    print(br.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
