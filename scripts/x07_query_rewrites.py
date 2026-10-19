"""X7 - How do the engines reword a question into search queries?

Uses the search queries saved by the replay (x03_replay: ChatGPT web_search calls, Gemini grounding
webSearchQueries). Per engine:
  volume       queries per answer, search calls per answer (ChatGPT), words per query vs the question
  overlap      share of the question's content words kept in the queries; share of query words that
               are new (not in the question); embedding similarity query-question (all-MiniLM-L6-v2)
  additions    most frequent added words; how often queries add a year, a price/plan word, a brand the
               question did not name, a comparison ("vs"), a site name or search operator (site:, quotes)
  stability    for the same question in two runs: share of identical queries and word-set overlap
Examples: the queries of a few questions across the three runs.

Outputs: results[/<ds>]/x07_query_summary.csv, x07_added_words.csv, x07_examples.csv
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

from common import TABLES, RESULTS, DATASET
from x03_analyse import records
from prompt_labels import BRANDS

WORD = re.compile(r"[a-z0-9][a-z0-9&'\-\.]*")
STOP = set('a an the and or of to in for on with is are do does can i my me what which how who why when where '
           'it its be vs than that this from at by as if any best better more most get should there their you your '
           'we our they them about have has will would not no into out up so just also some such each only very '
           'much many there here was were been being'.split())
YEAR = re.compile(r'\b20(2[4-9]|3\d)\b')
PRICE = re.compile(r'\b(price|prices|pricing|cost|costs|fee|fees|plan|plans|deal|deals|promo|\$\d+)')
OPER = re.compile(r'(\bsite:|"|\bintitle:|\bfiletype:)')
SITE = re.compile(r'\b(reddit|youtube|wikipedia|g2|capterra|fcc|consumer reports|pcmag|cnet)\b')


def content(t):
    return {w for w in WORD.findall(t.lower()) if w not in STOP and len(w) > 1}


def brand_hits(t):
    low = t.lower()
    return {b for b, rx in BRANDS.items() if re.search(rx, low)}


def main():
    lab = pd.read_csv(TABLES / 'prompt_labels.csv', usecols=['prompt_id', 'prompt'])
    prompts = dict(zip(lab.prompt_id, lab.prompt))
    rows, per_answer = [], []
    for engine in ('openai', 'gemini'):
        for j in records(engine):
            qs = [q.strip() for q in j['queries'] if q and q.strip()]
            per_answer.append(dict(engine=engine, prompt_id=j['prompt_id'], run=f"{j['date']}|{j['rep']}",
                                   n_queries=len(qs), n_calls=len(j.get('calls') or []), queries=qs))
            p = prompts.get(j['prompt_id'], '')
            pc, pb = content(p), brand_hits(p)
            for q in qs:
                qc = content(q)
                rows.append(dict(engine=engine, prompt_id=j['prompt_id'], run=f"{j['date']}|{j['rep']}",
                                 query=q, prompt=p, q_words=len(q.split()), p_words=len(p.split()),
                                 kept=len(qc & pc) / len(pc) if pc else np.nan,
                                 new=len(qc - pc) / len(qc) if qc else np.nan,
                                 added=' '.join(sorted(qc - pc)),
                                 year=bool(YEAR.search(q)) and not YEAR.search(p),
                                 price=bool(PRICE.search(q.lower())) and not PRICE.search(p.lower()),
                                 new_brand=bool(brand_hits(q) - pb), vs=' vs ' in f' {q.lower()} ' and ' vs' not in p.lower(),
                                 site=bool(SITE.search(q.lower())), operator=bool(OPER.search(q))))
    Q, A = pd.DataFrame(rows), pd.DataFrame(per_answer)
    # embedding similarity query vs question
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('all-MiniLM-L6-v2')
    enc = lambda xs: m.encode(xs, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    if len(Q):
        Q['similarity'] = np.einsum('ij,ij->i', enc(Q['query'].tolist()), enc(Q['prompt'].tolist()))
    out = []
    for e, g in Q.groupby('engine'):
        a = A[(A.engine == e) & (A.n_queries > 0)]
        # stability: same prompt, two different runs
        st = []
        for pid, h in a.groupby('prompt_id'):
            runs = h.queries.tolist()
            for i in range(len(runs)):
                for k in range(i + 1, len(runs)):
                    s1, s2 = {q.lower() for q in runs[i]}, {q.lower() for q in runs[k]}
                    w1, w2 = set().union(*map(content, runs[i])), set().union(*map(content, runs[k]))
                    st.append((len(s1 & s2) / len(s1 | s2), len(w1 & w2) / len(w1 | w2) if w1 | w2 else np.nan))
        st = np.array(st) if st else np.full((1, 2), np.nan)
        out.append(dict(engine={'openai': 'ChatGPT', 'gemini': 'Gemini'}[e], answers_with_search=len(a),
                        share_answers_with_search=len(a) / max(len(A[A.engine == e]), 1),
                        queries_per_answer=a.n_queries.median(), mean_queries_per_answer=a.n_queries.mean(),
                        calls_per_answer=a.n_calls.median() if e == 'openai' else np.nan,
                        words_per_query=g.q_words.median(), words_per_question=g.p_words.median(),
                        pct_question_words_kept=100 * g.kept.mean(), pct_query_words_new=100 * g.new.mean(),
                        similarity_to_question=g.similarity.mean(),
                        pct_add_year=100 * g.year.mean(), pct_add_price_plan=100 * g.price.mean(),
                        pct_add_brand=100 * g.new_brand.mean(), pct_add_vs=100 * g.vs.mean(),
                        pct_name_a_site=100 * g.site.mean(), pct_operator=100 * g.operator.mean(),
                        pct_identical_queries_two_runs=100 * np.nanmean(st[:, 0]),
                        pct_word_overlap_two_runs=100 * np.nanmean(st[:, 1])))
    S = pd.DataFrame(out)
    S.to_csv(RESULTS / 'x07_query_summary.csv', index=False)
    print(S.round(2).T.to_string())
    added = []
    for e, g in Q.groupby('engine'):
        c = Counter(w for s in g.added for w in s.split())
        n = len(g)
        added += [dict(engine=e, word=w, pct_of_queries=100 * k / n) for w, k in c.most_common(30)]
    pd.DataFrame(added).to_csv(RESULTS / 'x07_added_words.csv', index=False)
    for e in ('openai', 'gemini'):
        print(e, ', '.join(f"{r.word} {r.pct_of_queries:.0f}%" for r in pd.DataFrame(added).query('engine == @e').head(20).itertuples()))
    # examples: 4 random prompts searched in >= 2 runs by ChatGPT
    a = A[(A.engine == 'openai') & (A.n_queries > 0)]
    multi = a.groupby('prompt_id').size()
    pick = multi[multi >= 2].sample(min(4, (multi >= 2).sum()), random_state=42).index
    ex = []
    for pid in pick:
        for r in a[a.prompt_id == pid].itertuples():
            ex.append(dict(prompt=prompts[pid], engine='ChatGPT', run=r.run, queries=' | '.join(r.queries)))
        g = A[(A.engine == 'gemini') & (A.prompt_id == pid) & (A.n_queries > 0)]
        for r in g.itertuples():
            ex.append(dict(prompt=prompts[pid], engine='Gemini', run=r.run, queries=' | '.join(r.queries)))
    ex = pd.DataFrame(ex)
    ex.to_csv(RESULTS / 'x07_examples.csv', index=False)
    for p, g in ex.groupby('prompt', sort=False):
        print('\nQUESTION:', p)
        for r in g.itertuples():
            print(f'  {r.engine:8s} {r.run}: {r.queries}')


if __name__ == '__main__':
    main()
