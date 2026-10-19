"""R14 - Outside authority: do engines prefer important websites?

Question. Do pages on websites with high outside authority get cited more, how big is this
compared with page features, and do the engines differ?

Authority has one value per website, so it cannot be measured in a model with website
indicators (R2). Two designs instead:
  website level  one row per website: log2(1 + total citations) on an authority score + website
                 kind (role + most common page type of its cited pages, a stand-in until N14
                 labels website types) + log number of distinct pages cited
  page level     one row per URL (R2 rows incl. single-URL websites): y on page features +
                 authority + page type, NO website indicators; (i) mixed model with a random
                 intercept per website (statsmodels MixedLM), (ii) OLS with errors clustered by website
Main authority score: Common Crawl harmonic centrality (found for 94 % of websites); each of the
five scores is also tried in turn. Missing scores: median + "not found" flag.
Steps (guide): correlation with the in-data website score + over/under-performers; website
and page models; authority x prompt-match interaction; best score (fit + time-split error, R9
split); GAM shape / threshold; per engine; brand vs third party; without the giant platforms;
over time (only one snapshot so far - newly cited websites vs established ones instead).

Outputs: results/r14_*.csv, figures/r14_*.png
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pygam import LinearGAM, s as spline
from scipy import stats

import plotstyle
from plotstyle import plt, ENGINE_COLOR, MUTED, INK2
from common import RESULTS, FIGURES, ENGINES, ENGINE_LABEL, read_table, drop_excluded
from r02_main_model import prepare, FEATURES, LABEL
from r09_robustness import period_counts

warnings.filterwarnings('ignore')
SCORES = {'cc_harmonic': 'Common Crawl harmonic centrality', 'log_cc_pagerank': 'Common Crawl PageRank',
          'log_majestic_ref_subnets': 'Majestic linking networks', 'log_tranco_rank': 'Tranco rank (lower = bigger)',
          'log_crux_rank': 'CrUX rank (lower = bigger)'}
MAIN = 'cc_harmonic'
GIANTS = {'reddit.com', 'youtube.com', 'google.com', 'wikipedia.org'}


def fill(d, col):
    d = d.copy()
    d[col + '_na'] = d[col].isna().astype(float)
    d[col] = d[col].fillna(d[col].median())
    return d


def sign_fix(col):
    """Rank scores: smaller = more important; flip so that 'higher = more authority' everywhere."""
    return -1 if col.endswith('_rank') else 1


# ---------------------------------------------------------------- website level
def website_table():
    c = drop_excluded(read_table('citations.csv', usecols=[
        'answer_id', 'date', 'engine', 'url', 'reg_domain', 'role', 'is_redirect', 'repeat_in_answer']))
    c = c[~c.repeat_in_answer & c.reg_domain.notna()]
    first = c.groupby('reg_domain').date.min()
    w = c.groupby('reg_domain').agg(total=('url', 'size'), pages=('url', 'nunique'),
                                    role=('role', lambda s: s.mode().iat[0]))
    for e in ENGINES:
        w[f'total_{e}'] = c[c.engine == e].groupby('reg_domain').size().reindex(w.index, fill_value=0)
    w['first_cited'] = first
    ua = read_table('url_table_all.csv', usecols=['reg_domain', 'page_type', 'y'])
    w['kind'] = ua.groupby('reg_domain').page_type.agg(lambda s: s.mode().iat[0] if len(s.dropna()) else 'other')
    w['in_data_score'] = ua.groupby('reg_domain').y.mean()
    au = read_table('domain_authority.csv').set_index('domain')
    w = w.join(au[[c for c in SCORES]], how='left')
    for s_ in SCORES:                                    # z-score, higher = more authority
        v = sign_fix(s_) * w[s_]
        w[s_ + '_z'] = (v - v.mean()) / v.std()
    w['y_site'] = np.log2(1 + w.total)
    w['kind'] = w.role.where(w.role.isin(['own', 'competitor']), w.kind.fillna('other'))
    return w.rename_axis('website').reset_index()


def website_models(w):
    """Two specifications: the guide's (with number of distinct pages cited) and without it.

    Controlling for pages asks "for the same number of cited pages, do bigger websites get more
    citations?"; without it, "do bigger websites get more citations in total?".
    """
    rows = []
    for spec, extra in (('with pages control (guide)', ' + np.log2(pages)'), ('without pages control', '')):
      for s_ in SCORES:
        d = w.dropna(subset=[s_ + '_z'])
        f = smf.ols(f'y_site ~ {s_}_z + C(kind){extra}', data=d).fit(cov_type='HC1')
        rows.append(dict(spec=spec, score=s_, coef=f.params[s_ + '_z'], se=f.bse[s_ + '_z'], p=f.pvalues[s_ + '_z'],
                         pct_more_citations_per_sd=100 * (2 ** f.params[s_ + '_z'] - 1),
                         r2=f.rsquared, n_websites=len(d),
                         spearman_with_in_data_score=stats.spearmanr(d[s_ + '_z'], d.in_data_score,
                                                                     nan_policy='omit').statistic,
                         spearman_with_total_citations=stats.spearmanr(d[s_ + '_z'], d.total).statistic))
    return pd.DataFrame(rows)


def performers(w):
    d = w.dropna(subset=[MAIN + '_z'])
    d = d[d.total >= 20]                                  # enough citations to judge
    f = smf.ols(f'y_site ~ {MAIN}_z', data=d).fit()             # expected from authority alone
    d = d.assign(expected=f.fittedvalues, residual=f.resid)
    cols = ['website', 'role', 'kind', 'total', 'pages', MAIN, 'expected', 'residual']
    over = d.nlargest(20, 'residual')[cols].assign(group='over-performer')
    under = d.nsmallest(20, 'residual')[cols].assign(group='under-performer')
    return pd.concat([over, under])


# ---------------------------------------------------------------- page level
def page_data():
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    for s_ in SCORES:
        d[s_] = sign_fix(s_) * d[s_]                    # url_table already z-scored
        d = fill(d, s_)
    return d, flags, types


def page_model(d, X, score, cluster=True, mixed=False):
    cols = X + [score, score + '_na'] if d[score + '_na'].any() else X + [score]
    Xc = sm.add_constant(d[cols])
    if mixed:
        m = sm.MixedLM(d.y, Xc, groups=d.domain).fit(reml=True)
        return m.params, m.bse, m.pvalues, None
    m = sm.OLS(d.y, Xc).fit(cov_type='cluster', cov_kwds={'groups': d.domain.astype('category').cat.codes})
    return m.params, m.bse, m.pvalues, m.rsquared


def main():
    # ---- website level
    w = website_table()
    wm = website_models(w)
    wm.to_csv(RESULTS / 'r14_website_models.csv', index=False)
    perf = performers(w)
    perf.to_csv(RESULTS / 'r14_over_under_performers.csv', index=False)
    new = w.assign(new=w.first_cited >= sorted(w.first_cited.unique())[-30])
    newcmp = new.groupby('new')[[s_ + '_z' for s_ in SCORES]].median().T
    newcmp.columns = ['established (first cited before last 30 days)', 'new (first cited in last 30 days)']
    newcmp.to_csv(RESULTS / 'r14_new_vs_established.csv')

    # ---- page level
    d, flags, types = page_data()
    X = FEATURES + flags + types
    rows = []
    for how in ('mixed model', 'clustered OLS'):
        b, se, p, r2 = page_model(d, X, MAIN, mixed=(how == 'mixed model'))
        for term in [MAIN] + FEATURES:
            rows.append(dict(model=how, term=term, coef=b[term], se=se[term], p=p[term],
                             pct_change_per_sd=100 * (2 ** b[term] - 1), n_urls=len(d),
                             n_websites=d.domain.nunique()))
    pm = pd.DataFrame(rows)
    pm.to_csv(RESULTS / 'r14_page_model.csv', index=False)

    # interaction authority x prompt match
    d['auth_x_match'] = d[MAIN] * d.prompt_match_words
    d['auth_x_title'] = d[MAIN] * d.title_match
    b, se, p, _ = page_model(d, X + ['auth_x_match', 'auth_x_title'], MAIN)
    inter = pd.DataFrame([dict(term=t_, coef=b[t_], se=se[t_], p=p[t_]) for t_ in
                          [MAIN, 'prompt_match_words', 'title_match', 'auth_x_match', 'auth_x_title']])
    inter.to_csv(RESULTS / 'r14_interaction.csv', index=False)

    # which score works best: fit + time-split hold-out
    counts, _, _, _, ratio = period_counts(d.url.tolist())
    d['y_train'] = np.log2(1 + ratio * counts.train.to_numpy())
    d['y_hold'] = np.log2(1 + counts.holdout.to_numpy())
    best = []
    for s_ in list(SCORES) + ['none']:
        cols = X + ([s_, s_ + '_na'] if s_ != 'none' and d[s_ + '_na'].any() else [s_] if s_ != 'none' else [])
        full = sm.OLS(d.y, sm.add_constant(d[cols])).fit()
        tr = sm.OLS(d.y_train, sm.add_constant(d[cols])).fit()
        pred = tr.predict(sm.add_constant(d[cols]))
        best.append(dict(score=s_, r2=full.rsquared, holdout_rmse=np.sqrt(((d.y_hold - pred) ** 2).mean()),
                         holdout_r2=1 - ((d.y_hold - pred) ** 2).sum() / ((d.y_hold - d.y_hold.mean()) ** 2).sum(),
                         coef=full.params.get(s_, np.nan)))
    best = pd.DataFrame(best).sort_values('holdout_r2', ascending=False)
    best.to_csv(RESULTS / 'r14_score_comparison.csv', index=False)

    # GAM shape of authority (other variables as linear controls via residualising)
    res_y = sm.OLS(d.y, sm.add_constant(d[X])).fit().resid
    res_a = d[MAIN] - sm.OLS(d[MAIN], sm.add_constant(d[X])).fit().predict(sm.add_constant(d[X])) + d[MAIN].mean()
    raw = d[MAIN].to_numpy()[:, None]
    g = LinearGAM(spline(0, n_splines=12)).gridsearch(raw, res_y.to_numpy(), progress=False)
    grid = np.linspace(*np.percentile(raw, [1, 99]), 100)[:, None]
    gfit = g.predict(grid)

    # per engine, brand vs third party, without giants
    sub = []
    for label, s in [('all pages', d), ('own + competitor', d[d.role.isin(['own', 'competitor'])]),
                     ('third party', d[d.role == 'third_party']),
                     ('without reddit/youtube/google/wikipedia', d[~d.reg_domain.isin(GIANTS)])]:
        b, se, p, _ = page_model(s, X, MAIN)
        for term in (MAIN, 'prompt_match_words', 'title_match'):
            sub.append(dict(subset=label, outcome='all engines', term=term, coef=b[term], se=se[term], p=p[term],
                            n_urls=len(s)))
    for e in ENGINES:
        s = d.assign(y=np.log2(1 + d[f'count_{e}']))
        b, se, p, _ = page_model(s, X, MAIN)
        for term in (MAIN, 'prompt_match_words', 'title_match'):
            sub.append(dict(subset='all pages', outcome=ENGINE_LABEL[e], term=term, coef=b[term],
                            se=se[term], p=p[term], n_urls=len(s)))
    sub = pd.DataFrame(sub)
    sub['pct_change_per_sd'] = 100 * (2 ** sub.coef - 1)
    sub.to_csv(RESULTS / 'r14_subsets.csv', index=False)

    # ---- figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    dd = w.dropna(subset=[MAIN])
    ax.scatter(dd[MAIN], dd.y_site, s=6, alpha=0.35, color='#2a78d6', lw=0)
    lab = perf[perf.group == 'over-performer'].head(5)          # full lists in the CSV
    for _, r in lab.iterrows():
        yy = np.log2(1 + r.total)
        ax.annotate(r.website, (r[MAIN], yy), fontsize=6.5, color=INK2, xytext=(3, 2), textcoords='offset points')
    ax.set_xlabel('Common Crawl harmonic centrality (website)')
    ax.set_ylabel('log2(1 + total citations)')
    rho = wm[wm.spec == 'without pages control'].set_index('score').loc[MAIN, 'spearman_with_total_citations']
    ax.set_title(f'Websites: authority vs citations (Spearman {rho:.2f})', fontsize=9.5)
    ax = axes[1]
    pmm = pm[pm.model == 'mixed model'].set_index('term')
    terms = [MAIN, 'title_match', 'best_paragraph_match', 'prompt_match_words', 'intro_match', 'page_age_days']
    v = pmm.loc[terms]
    y = np.arange(len(terms))[::-1]
    lo = 100 * (2 ** (v.coef - 1.96 * v.se) - 1); hi = 100 * (2 ** (v.coef + 1.96 * v.se) - 1)
    ax.hlines(y, lo, hi, color=['#f7b99d'] + ['#9ec5f4'] * (len(terms) - 1), lw=2)
    ax.scatter(v.pct_change_per_sd, y, color=['#eb6834'] + ['#2a78d6'] * (len(terms) - 1), s=40, zorder=3,
               edgecolor='#fcfcfb')
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, ['outside authority (CC harmonic)'] + [LABEL[t_] for t_ in terms[1:]])
    ax.set_xlabel('% more citations per +1 SD (mixed model, random website intercept)')
    ax.grid(axis='y', visible=False)
    ax.set_title(f'Pages: authority next to page features ({len(d):,} URLs)', fontsize=9.5)
    fig.suptitle('R14: outside authority', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'r14_authority.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].plot(grid[:, 0], gfit - gfit[len(gfit) // 2], color='#2a78d6')
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].set_xlabel('CC harmonic centrality (z)'); axes[0].set_ylabel('effect on log2 citations')
    axes[0].set_title('Shape of the authority effect (page features held fixed)', fontsize=9.5)
    ax = axes[1]
    pe = sub[(sub.term == MAIN) & (sub.outcome != 'all engines')]
    for i, (_, r) in enumerate(pe.iterrows()):
        e = [k for k, v_ in ENGINE_LABEL.items() if v_ == r.outcome][0]
        lo = 100 * (2 ** (r.coef - 1.96 * r.se) - 1); hi = 100 * (2 ** (r.coef + 1.96 * r.se) - 1)
        ax.hlines(i, lo, hi, color=ENGINE_COLOR[e], lw=2, alpha=0.5)
        ax.plot(r.pct_change_per_sd, i, 'o', color=ENGINE_COLOR[e], ms=7, mec='#fcfcfb')
    ax.set_yticks(range(len(pe)), pe.outcome)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel('% more citations per +1 SD of authority')
    ax.grid(axis='y', visible=False)
    ax.set_title('Which engine leans most on big websites?', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGURES / 'r14_shape_engines.png')
    plt.close(fig)

    print('WEBSITE LEVEL'); print(wm.round(3).to_string(index=False))
    print('\nPAGE LEVEL'); print(pm[pm.term.isin(terms)].round(4).to_string(index=False))
    print('\nINTERACTION'); print(inter.round(4).to_string(index=False))
    print('\nSCORE COMPARISON'); print(best.round(4).to_string(index=False))
    print('\nSUBSETS (authority term)'); print(sub[sub.term == MAIN].round(3).to_string(index=False))
    print('\nNEW vs ESTABLISHED websites (median z)'); print(newcmp.round(2).to_string())
    print('\nTOP OVER / UNDER'); print(perf.groupby('group').head(8)[['group', 'website', 'total', MAIN]].to_string(index=False))


if __name__ == '__main__':
    main()
