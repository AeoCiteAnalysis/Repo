"""Write paper/numbers.tex: every result number the paper quotes that can change, as LaTeX macros.

The replay results change with every daily replay and every weekly pfetch upload; the paper therefore
never types them by hand. Run this after the analyses (weekly_refresh.py does it) and recompile.
Macro names: \\gf... = main dataset (GFiber), \\wo... = second dataset (Workorb); letters only
(LaTeX macro names cannot contain digits). A number whose result file is missing is written as a bold
'??' so the gap is visible in the PDF.

  .venv/Scripts/python.exe scripts/paper_numbers.py
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIRS = {'gf': ROOT / 'results', 'wo': ROOT / 'results' / 'workorb'}
REPLAY = {'gf': ROOT / 'data' / 'replay', 'wo': ROOT / 'data' / 'workorb' / 'replay'}
OUT = ROOT / 'paper' / 'numbers.tex'

macros, missing = {}, []


def put(name, value, fmt='{:.1f}'):
    """name without backslash; value a number or string."""
    if value is None or (isinstance(value, float) and value != value):
        macros[name] = r'\textbf{??}'
        missing.append(name)
    else:
        text = fmt.format(value) if isinstance(value, (int, float)) else str(value)
        if isinstance(value, (int, float)) and text.startswith('-'):
            text = r'\textminus' + text[1:]          # typographic minus, not a hyphen
        macros[name] = text


def read(pfx, f):
    p = DIRS[pfx] / f
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 5 else None


def safe(fn):
    def wrap(pfx):
        try:
            fn(pfx)
        except Exception as e:                   # a result file changed shape: mark, do not stop
            missing.append(f'{pfx}:{fn.__name__} ({type(e).__name__}: {e})')
    return wrap


@safe
def replay_scope(pfx):
    runs = set()
    for f in (REPLAY[pfx] / 'openai').glob('*/*.json'):
        runs.add((f.parent.name, f.stem.rsplit('_r', 1)[1]))
    days = sorted({d for d, _ in runs})
    put(f'{pfx}ReplayRuns', len(runs), '{}')
    put(f'{pfx}ReplayDays', len(days), '{}')
    if days:
        d = datetime.strptime(days[-1], '%Y%m%d')
        put(f'{pfx}ReplayLastDay', f'{d.day} {d:%B %Y}')
    else:
        put(f'{pfx}ReplayLastDay', None)


@safe
def coverage(pfx):
    c = read(pfx, 'x03_coverage.csv')
    if c is not None:
        r = c.iloc[0]
        put(f'{pfx}ConsultedPerQ', r.consulted_per_prompt, '{:.0f}')
        put(f'{pfx}CitedPerQ', r.cited_per_prompt, '{:.0f}')
        put(f'{pfx}PctConsultedCited', r.pct_consulted_cited)
    f = read(pfx, 'x03_faithful.csv')
    if f is not None:
        g = f[f.engine == 'ChatGPT'].iloc[0]
        put(f'{pfx}SearchedQ', g.searched, '{:.0f}')
        put(f'{pfx}ZeroIfSearched', 100 * g.collected_zero_share_when_replay_searched)
        put(f'{pfx}ZeroIfNotSearched', 100 * g.collected_zero_share_when_replay_did_not)
        put(f'{pfx}ReplayInCollected', 100 * g.replay_cited_in_collected_last14, '{:.0f}')
        put(f'{pfx}CollectedDayToDay', 100 * g.collected_lastday_in_previous13, '{:.0f}')


@safe
def rank_curve(pfx):
    r = read(pfx, 'x05_rank_ci.csv')
    if r is None:
        return
    names = {'1': 'One', '2': 'Two', '3': 'Three', '4-5': 'FourFive', '6-10': 'SixTen',
             '11-20': 'ElevenTwenty', '21+': 'TwentyOnePlus'}
    for row in r.itertuples():
        k = names[str(row.rank_bin)]
        put(f'{pfx}Rank{k}', row.pct_cited)
        put(f'{pfx}Rank{k}Lo', row.ci_low)
        put(f'{pfx}Rank{k}Hi', row.ci_high)
    put(f'{pfx}SourceLists', r.source_lists.iloc[0], '{:.0f}')
    s = read(pfx, 'x05_selection_ci.csv')
    if s is not None:
        for v, k, fmt in (('log_rank', 'RankOR', '{:.2f}'), ('multi_call', 'MultiCallOR', '{:.1f}')):
            row = s[s.variable == v].iloc[0]
            put(f'{pfx}{k}', row.odds_ratio, fmt)
            put(f'{pfx}{k}Lo', row.or_ci_low, fmt)
            put(f'{pfx}{k}Hi', row.or_ci_high, fmt)


@safe
def variance(pfx):
    v = read(pfx, 'x05_variance_ci.csv')
    if v is None:
        return
    t = v[v.outcome == 'total'].set_index('model')
    for m, k in (('page variables', 'PageVar'), ('replay retrieval', 'Retrieval'),
                 ('page variables + replay retrieval + cited', 'AllVar'),
                 ('difference: retrieval - page variables', 'RetrievalMinusPage')):
        put(f'{pfx}{k}', 100 * t.loc[m, 'r2_within'], '{:.1f}' if k == 'PageVar' else '{:.0f}')
        put(f'{pfx}{k}Lo', 100 * t.loc[m, 'ci_low'], '{:.0f}')
        put(f'{pfx}{k}Hi', 100 * t.loc[m, 'ci_high'], '{:.0f}')
    put(f'{pfx}VarPages', t.n_urls.iloc[0], '{:,.0f}')
    put(f'{pfx}VarSites', t.n_websites.iloc[0], '{:,.0f}')
    sh = read(pfx, 'x05_shuffled.csv')
    if sh is not None:
        r = sh[sh.outcome == 'total'].iloc[0]
        put(f'{pfx}Shuffled', 100 * r.r2_within_shuffled_mean, '{:.2f}')
        put(f'{pfx}ShuffledP', r.p_value, '{:.3f}')
    w = read(pfx, 'x05_windows_ci.csv')
    if w is not None:
        put(f'{pfx}WindowFirst', 100 * w.r2_within.iloc[0], '{:.0f}')
        put(f'{pfx}WindowSecondLast', 100 * w.r2_within.iloc[-2], '{:.0f}')
        put(f'{pfx}WindowLast', 100 * w.r2_within.iloc[-1], '{:.0f}')


@safe
def repeat(pfx):
    r = read(pfx, 'x03_same_day_repeat.csv')
    if r is not None:
        r = r.iloc[0]
        put(f'{pfx}SameQueries', 100 * r.jaccard_queries)
        put(f'{pfx}SameConsulted', 100 * r.jaccard_consulted, '{:.0f}')
        put(f'{pfx}SameCited', 100 * r.jaccard_cited, '{:.0f}')
    s = read(pfx, 'x05_same_day.csv')
    if s is not None and 'measure' in s:
        for m, k in (('in_consulted', 'SameDayInConsulted'), ('in_cited', 'SameDayInCited'),
                     ('jaccard_cited', 'SameDayJaccard')):
            row = s[s.measure == m].iloc[0]
            put(f'{pfx}{k}', 100 * row['mean'], '{:.0f}')
        put(f'{pfx}SameDayDays', s.days.iloc[0], '{:.0f}')
    else:                                        # no shared days yet: visible gap in the PDF
        for k in ('SameDayInConsulted', 'SameDayInCited', 'SameDayJaccard', 'SameDayDays'):
            put(f'{pfx}{k}', None)


@safe
def selection(pfx):
    m = read(pfx, 'x06_selection_models.csv')
    if m is not None:
        m = m.set_index('model').mcfadden_r2
        for name, k in (('S0 rank only', 'SelRank'), ('S0b rank + text-missing flags', 'SelRankFlags'),
                        ('S1 + match to the prompt', 'SelMatch'), ('S2 + match to the engine queries', 'SelQueries'),
                        ('S3 + page features', 'SelPage'), ('M only (no rank): match + queries + page', 'SelNoRank')):
            put(f'{pfx}{k}', 100 * m[name], '{:.1f}')
    c = read(pfx, 'x06_selection_coef.csv')
    if c is not None:
        c = c[c.model == 'S3 + page features'].set_index('variable')
        for v, k in (('emb_best_z', 'SelBestPara'), ('q_best_z', 'SelQueryMatch'), ('log_words_z', 'SelLength'),
                     ('log_headings_z', 'SelHeadings'), ('type_list', 'SelList'), ('type_comparison', 'SelComparison'),
                     ('role_brand', 'SelBrand'), ('is_reddit', 'SelReddit')):
            if v in c.index:
                put(f'{pfx}{k}', c.loc[v, 'odds_ratio'], '{:.2f}')
    s = read(pfx, 'x06b_spearman.csv')
    if s is not None:
        s = s.set_index('score').mean_spearman_with_higher_rank
        put(f'{pfx}RhoMeaning', s['q_best'], '{:.2f}')
        put(f'{pfx}RhoKeywords', s['q_bm25_max'], '{:.2f}')


@safe
def rewording(pfx):
    q = read(pfx, 'x07_query_summary.csv')
    if q is None:
        return
    for eng, k in (('ChatGPT', 'Gpt'), ('Gemini', 'Gem')):
        r = q[q.engine == eng].iloc[0]
        put(f'{pfx}{k}QueriesPerAnswer', r.queries_per_answer, '{:.0f}')
        put(f'{pfx}{k}WordsKept', r.pct_question_words_kept, '{:.0f}')
        put(f'{pfx}{k}WordsNew', r.pct_query_words_new, '{:.0f}')
        put(f'{pfx}{k}AddYear', r.pct_add_year, '{:.0f}')
        put(f'{pfx}{k}AddBrand', r.pct_add_brand, '{:.0f}')
        put(f'{pfx}{k}Similarity', r.similarity_to_question, '{:.2f}')
        put(f'{pfx}{k}Identical', r.pct_identical_queries_two_runs, '{:.0f}')


@safe
def deep_serp(pfx):
    d = read(pfx, 'x08_deep_serp.csv')
    if d is None:
        return
    d = d[d.dataset == ('gfiber' if pfx == 'gf' else 'workorb')]
    for eng, prov, kind, k in (('openai', 'google', 'retrieved', 'DeepGptRetGoogle'),
                               ('openai', 'bing', 'retrieved', 'DeepGptRetBing'),
                               ('openai', 'google', 'cited', 'DeepGptCitGoogle'),
                               ('gemini', 'google', 'cited', 'DeepGemCitGoogle')):
        r = d[(d.engine == eng) & (d.provider == prov) & (d.kind == kind)]
        if len(r):
            put(f'{pfx}{k}Absent', r['page_not found'].iloc[0], '{:.0f}')
            put(f'{pfx}{k}SiteFound', 100 - r['site_not found'].iloc[0], '{:.0f}')


@safe
def prior(pfx):
    c = read(pfx, 'x10_collected.csv')
    if c is not None:
        r = c[c.engine == 'ChatGPT'].iloc[0]
        put(f'{pfx}PriorCitations', r.pct_citations_from_prior_site, '{:.0f}')
        put(f'{pfx}PriorBaseline', r.pct_with_other_prompts_prior, '{:.0f}')
    q = read(pfx, 'x10_queries.csv')
    if q is not None:
        r = q[q.engine == 'ChatGPT'].iloc[0]
        put(f'{pfx}PriorAddIfIn', r.pct_added_if_in_prior, '{:.0f}')
        put(f'{pfx}PriorAddIfOut', r.pct_added_if_not_in_prior, '{:.0f}')
    s = read(pfx, 'x10_rank_selection.csv')
    if s is not None:
        s = s.set_index('model')
        put(f'{pfx}PriorRankFactor', s.loc['retrieval: + X6 features', 'rank_factor'], '{:.2f}')
        put(f'{pfx}PriorSelOR', s.loc['selection: rank + X6 features + prior site', 'odds_ratio'], '{:.2f}')


@safe
def brands(pfx):
    d = read(pfx, 'x11_decomposition.csv')
    if d is None:
        return
    for eng, k in (('ChatGPT', 'Gpt'), ('Gemini', 'Gem')):
        r = d[d.engine == eng].iloc[0]
        put(f'{pfx}{k}MentionOwn', r.pct_own_page_cited, '{:.0f}')
        put(f'{pfx}{k}MentionThird', r.pct_third_party_page_naming_it, '{:.0f}')
        put(f'{pfx}{k}MentionNone', r.pct_no_cited_page_naming_it, '{:.0f}')
    m = read(pfx, 'x11_model.csv')
    if m is not None:
        for eng, k in (('ChatGPT', 'Gpt'), ('Gemini', 'Gem')):
            g = m[m.engine == eng].set_index('variable')
            put(f'{pfx}{k}MentionThirdOR', g.loc['third_naming', 'odds_ratio'], '{:.1f}')
            if 'prior' in g.index:
                put(f'{pfx}{k}MentionPriorOR', g.loc['prior', 'odds_ratio'], '{:.1f}')


@safe
def content(pfx):
    """Final results used in Table 5: five-test verdicts, tips, variance split, authority, page links."""
    ir = read(pfx, 'is_it_real.csv')
    if ir is not None:
        for smp, sk in (('all pages', 'All'), ('own + competitor', 'Brand'), ('third party', 'Third')):
            for term, tk in (('title_match', 'Title'), ('best_paragraph_match', 'BestPara'),
                             ('prompt_match_words', 'Overlap')):
                r = ir[(ir['sample'] == smp) & (ir.term == term)].iloc[0]
                put(f'{pfx}{tk}{sk}', r['pct_change'], '{:+.1f}')
                put(f'{pfx}{tk}{sk}Tests', r.tests_passed, '{:.0f}')
    e = read(pfx, 'r08_simpson.csv')
    if e is not None:
        e = e[e['sample'] == 'all pages']
        for tip, k in (('has_faq', 'Faq'), ('has_summary_box', 'Summary'), ('has_author', 'Author'),
                       ('has_schema', 'Schema')):
            for est, ek in (('pooled, alone', 'Pooled'), ('within website', 'Within')):
                r = e[(e.tip == tip) & (e.estimate == est)]
                if len(r):
                    put(f'{pfx}{k}{ek}', r['pct_change'].iloc[0], '{:+.1f}')
    v = read(pfx, 'r13_variance_components.csv')
    if v is not None:
        v = v.set_index('part')
        put(f'{pfx}Between', 100 * v.loc['between websites', 'share_of_total'], '{:.0f}')
        put(f'{pfx}WithinExplained', 100 * v.loc['within websites', 'explained_by_page_variables'], '{:.1f}')
    rel = read(pfx, 'r13_reliability.csv')
    if rel is not None:
        put(f'{pfx}Reliability', 100 * rel.set_index('part').loc['within websites', 'reliability'], '{:.0f}')
    s = read(pfx, 'r14_subsets.csv')
    if s is not None:
        r = s[(s.subset == 'all pages') & (s.outcome == 'all engines') & (s.term == 'cc_harmonic')]
        if len(r):
            put(f'{pfx}SiteAuthority', r.pct_change_per_sd.iloc[0], '{:+.1f}')
    x9 = read(pfx, 'x09_citations.csv')
    if x9 is not None:
        r = x9[(x9['sample'] == 'all pages') & (x9.design == 'within website')].set_index('measure')
        put(f'{pfx}PageLinks', r.loc['pa_links', 'pct'], '{:+.1f}')
        put(f'{pfx}PageLinksAny', r.loc['pa_any', 'pct'], '{:+.1f}')


@safe
def cited_retrieved(pfx):
    """ChatGPT replay: share of cited URLs that were among the pages its searches returned or opened."""
    import json
    from urllib.parse import urlparse

    def n(u):
        q = urlparse(u or '')
        return (q.netloc.lower().removeprefix('www.') + q.path.rstrip('/')).lower()

    tot = hit = 0
    for f in (REPLAY[pfx] / 'openai').glob('*/*.json'):
        r = json.loads(f.read_text(encoding='utf-8'))
        seen = {n(u) for u in (r.get('consulted') or []) + (r.get('opened') or [])}
        for u in r.get('cited') or []:
            tot += 1
            hit += n(u) in seen
    put(f'{pfx}CitedURLs', tot, '{:,}')
    put(f'{pfx}CitedRetrievedPct', 100 * hit / tot if tot else None, '{:.1f}')


def main():
    for pfx in ('gf', 'wo'):
        for fn in (replay_scope, coverage, rank_curve, variance, repeat, selection, rewording,
                   deep_serp, prior, brands, content, cited_retrieved):
            fn(pfx)
    lines = ['% Generated by scripts/paper_numbers.py on ' + datetime.now().strftime('%Y-%m-%d %H:%M') +
             ' - do not edit by hand; rerun the script after the analyses.']
    for k in sorted(macros):
        lines.append(f'\\newcommand{{\\{k}}}{{{macros[k]}}}')
    OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'{len(macros)} macros -> {OUT}')
    if missing:
        print('missing / failed:', *missing, sep='\n  ')


if __name__ == '__main__':
    main()
