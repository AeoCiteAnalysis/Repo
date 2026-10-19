"""R13 - How much is the website vs the page?

Question. How much of the difference in citation counts can the website alone explain, and
how much do page features add?

Same rows as R2 (URLs cited >= 2 with page features, websites with >= 2 such URLs). R² of
five OLS models of y:
  (1) nothing             (2) word count only
  (3) outside authority   (Common Crawl harmonic + PageRank, Majestic, Tranco, CrUX; missing
                           values median-filled with a flag)
  (4) website only        (one indicator per website)
  (5) website + all R2 variables (page features, flags, page types)
Plus (6) outside authority + all R2 variables, without website indicators.
Both in-sample R² and 5-fold cross-validated R² (a website's indicator is estimated from its
other URLs; URLs whose website is absent from the training folds get the overall mean).
(3) vs (4): how much of the website effect outside authority explains.
(4) -> (5): the most page changes can add.

In-sample and cross-validated R² disagree strongly for the website (many websites have only
2-3 cited pages: in sample their level is memorised, on unseen pages it is a poor guess), so two
further analyses give the answer R13 asks for:
  variance components  mixed model with a random intercept per website (small websites are shrunk
                       toward the average): share of the variation BETWEEN websites vs WITHIN
                       them, and how much of each part the page variables explain
  CV by website size   cross-validated R² on websites with >= 2, >= 10 and >= 30 cited pages,
                       where a website's level can be estimated reliably
  split-half reliability  citations counted on odd and on even days; their correlation r gives the
                       repeatable (systematic) share of the 90-day count, 2r / (1 + r)
                       (Spearman-Brown). The rest is day-to-day chance that no page feature can
                       explain. Computed overall, within websites and between websites.

Outputs: results/r13_variance_split.csv, results/r13_variance_components.csv,
         results/r13_cv_by_site_size.csv, results/r13_reliability.csv,
         figures/r13_variance_split.png, figures/r13_components.png
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold

import plotstyle
from plotstyle import plt, INK2
from common import RESULTS, FIGURES, SEED, read_table, drop_excluded
from fe import demean
from r02_main_model import prepare, FEATURES

AUTHORITY = ['cc_harmonic', 'log_cc_pagerank', 'log_majestic_ref_subnets', 'log_tranco_rank', 'log_crux_rank']


def r2(y, pred):
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fit_predict(train, test, cols, fe):
    """OLS (optionally with website FE) fitted on train, predictions for test."""
    if fe:
        if cols:
            Z = demean(train, ['y'] + cols, 'domain')
            b = np.linalg.lstsq(Z[cols].to_numpy(), Z.y.to_numpy(), rcond=None)[0]
            xb_tr, xb_te = train[cols].to_numpy() @ b, test[cols].to_numpy() @ b
        else:
            xb_tr, xb_te = np.zeros(len(train)), np.zeros(len(test))
        alpha = pd.Series(train.y.to_numpy() - xb_tr).groupby(train.domain.to_numpy()).mean()
        a_te = test.domain.map(alpha).fillna((train.y.to_numpy() - xb_tr).mean()).to_numpy()
        return a_te + xb_te
    if not cols:
        return np.full(len(test), train.y.mean())
    m = sm.OLS(train.y, sm.add_constant(train[cols])).fit()
    return m.predict(sm.add_constant(test[cols], has_constant='add')).to_numpy()


def variance_components(d, X):
    import warnings
    warnings.filterwarnings('ignore')
    m0 = sm.MixedLM(d.y, np.ones((len(d), 1)), groups=d.domain).fit(reml=True)
    m1 = sm.MixedLM(d.y, sm.add_constant(d[X]), groups=d.domain).fit(reml=True)
    vb0, vw0 = float(m0.cov_re.iloc[0, 0]), float(m0.scale)
    vb1, vw1 = float(m1.cov_re.iloc[0, 0]), float(m1.scale)
    tot = vb0 + vw0
    between, within = vb0 / tot, vw0 / tot
    between_expl, within_expl = 1 - vb1 / vb0, 1 - vw1 / vw0
    return pd.DataFrame([
        dict(part='between websites', share_of_total=between, explained_by_page_variables=between_expl),
        dict(part='within websites', share_of_total=within, explained_by_page_variables=within_expl),
    ]).assign(n_urls=len(d), n_websites=d.domain.nunique())


def cv_by_size(d, X, sizes=(2, 10, 30)):
    rows = []
    for k in sizes:
        s = d[d.groupby('domain').y.transform('size') >= k].reset_index(drop=True)
        r = dict(min_pages_per_website=k, n_urls=len(s), n_websites=s.domain.nunique())
        for name, (cols, fe) in {'website_only': ([], True), 'website_plus_page': (X, True),
                                 'page_only': (X, False)}.items():
            pred = np.empty(len(s))
            for tr, te in KFold(5, shuffle=True, random_state=SEED).split(s):
                pred[te] = fit_predict(s.iloc[tr], s.iloc[te], cols, fe)
            r[f'cv_r2_{name}'] = r2(s.y, pred)
        rows.append(r)
    return pd.DataFrame(rows)


def reliability(d):
    """Split-half (odd vs even days) reliability of y, overall / within / between websites."""
    c = drop_excluded(read_table('citations.csv', usecols=['date', 'engine', 'url', 'is_redirect',
                                                          'repeat_in_answer']))
    c = c[~c.is_redirect & ~c.repeat_in_answer & c.url.isin(d.url)]
    order = {x: i for i, x in enumerate(sorted(c.date.unique()))}
    c['half'] = c.date.map(order) % 2
    n = c.groupby(['url', 'half']).size().unstack(fill_value=0).reindex(d.url, fill_value=0)
    a = pd.Series(np.log2(1 + n[0].to_numpy()))
    b = pd.Series(np.log2(1 + n[1].to_numpy()))
    dom = d.domain.reset_index(drop=True)
    sb = lambda r: 2 * r / (1 + r)
    rows = []
    r = np.corrcoef(a, b)[0, 1]
    rows.append(dict(part='all variation', r_odd_even=r, reliability=sb(r)))
    wa, wb = a - a.groupby(dom).transform('mean'), b - b.groupby(dom).transform('mean')
    r = np.corrcoef(wa, wb)[0, 1]
    rows.append(dict(part='within websites', r_odd_even=r, reliability=sb(r)))
    g = pd.DataFrame({'a': a, 'b': b, 'dom': dom}).groupby('dom').mean()
    r = np.corrcoef(g.a, g.b)[0, 1]
    rows.append(dict(part='between websites (site averages)', r_odd_even=r, reliability=sb(r)))
    out = pd.DataFrame(rows)
    out['chance_share'] = 1 - out.reliability
    return out.assign(n_urls=len(d), n_days=len(order))


def plot_components(vc, cv):
    b, w = vc.set_index('part').loc['between websites'], vc.set_index('part').loc['within websites']
    segs = [(100 * b.share_of_total * b.explained_by_page_variables, '#2a78d6', 'between websites: linked to page variables'),
            (100 * b.share_of_total * (1 - b.explained_by_page_variables), '#9ec5f4', 'between websites: other'),
            (100 * w.share_of_total * w.explained_by_page_variables, '#eb6834', 'within websites: explained by page variables'),
            (100 * w.share_of_total * (1 - w.explained_by_page_variables), '#e1e0d9', 'within websites: unexplained')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={'width_ratios': [1.4, 1]})
    ax, left = axes[0], 0
    for v, col, lab in segs:
        ax.barh([0], [v], left=left, color=col, height=0.5, label=f'{lab} ({v:.1f}%)', edgecolor='#fcfcfb', linewidth=1.5)
        left += v
    ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel('% of all variation in citation counts')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=7.5)
    ax.set_title(f'Where the variation lies ({int(vc.n_urls.iloc[0]):,} URLs, {int(vc.n_websites.iloc[0]):,} websites)', fontsize=9.5)
    ax = axes[1]
    x = np.arange(len(cv))
    for i, (col, name, lab) in enumerate([('#9ec5f4', 'cv_r2_website_only', 'website only'),
                                          ('#2a78d6', 'cv_r2_website_plus_page', 'website + page'),
                                          ('#eb6834', 'cv_r2_page_only', 'page only')]):
        v = 100 * cv[name]
        ax.bar(x + (i - 1) * 0.26, v, width=0.24, color=col, label=lab)
        for xi, vi in zip(x + (i - 1) * 0.26, v):
            ax.text(xi, vi + 0.3, f'{vi:.0f}', ha='center', fontsize=7, color=INK2)
    ax.set_xticks(x, [f'>= {k} pages\n({n} sites)' for k, n in zip(cv.min_pages_per_website, cv.n_websites)], fontsize=8)
    ax.set_ylabel('% explained on unseen pages'); ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left', fontsize=7.5)
    ax.set_title('Prediction of unseen pages, by website size', fontsize=9.5)
    fig.suptitle('R13: website vs page', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'r13_components.png')
    plt.close(fig)


def main():
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    d = d[d.groupby('domain').y.transform('size') >= 2].reset_index(drop=True)
    auth = []
    for a in AUTHORITY:
        d[a + '_na'] = d[a].isna().astype(float)
        d[a] = d[a].fillna(d[a].median())
        auth += [a] + ([a + '_na'] if d[a + '_na'].any() else [])
    allx = FEATURES + flags + types
    models = {'(1) nothing': ([], False), '(2) word count only': (['word_count'], False),
              '(3) outside authority only': (auth, False), '(4) website only': ([], True),
              '(5) website + all page variables': (allx, True),
              '(6) authority + all page variables (no website indicators)': (auth + allx, False)}
    rows = []
    for name, (cols, fe) in models.items():
        ins = r2(d.y, fit_predict(d, d, cols, fe))
        pred = np.empty(len(d))
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(d):
            pred[te] = fit_predict(d.iloc[tr], d.iloc[te], cols, fe)
        rows.append(dict(model=name, r2_in_sample=ins, r2_cross_validated=r2(d.y, pred),
                         n_urls=len(d), n_websites=d.domain.nunique()))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'r13_variance_split.csv', index=False)

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = np.arange(len(out))[::-1]
    ax.barh(y + 0.19, 100 * out.r2_in_sample, height=0.36, color='#9ec5f4', label='in sample')
    ax.barh(y - 0.19, 100 * out.r2_cross_validated.clip(lower=0), height=0.36, color='#2a78d6',
            label='cross-validated')
    for yi, a, b in zip(y, out.r2_in_sample, out.r2_cross_validated):
        ax.text(100 * a + 0.5, yi + 0.19, f'{100 * a:.1f}%', va='center', fontsize=7.5, color=INK2)
        ax.text(max(100 * b, 0) + 0.5, yi - 0.19, f'{100 * b:.1f}%', va='center', fontsize=7.5, color=INK2)
    ax.set_yticks(y, out.model, fontsize=8)
    ax.set_xlabel('% of variation in citation count explained (R²)')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right')
    ax.set_title(f'R13: website vs page ({len(out.n_urls) and out.n_urls.iloc[0]:,} URLs)')
    fig.savefig(FIGURES / 'r13_variance_split.png')
    plt.close(fig)
    print(out.round(4).to_string(index=False))

    vc = variance_components(d, allx)
    vc.to_csv(RESULTS / 'r13_variance_components.csv', index=False)
    cv = cv_by_size(d, allx)
    cv.to_csv(RESULTS / 'r13_cv_by_site_size.csv', index=False)
    plot_components(vc, cv)
    rel = reliability(d)
    rel.to_csv(RESULTS / 'r13_reliability.csv', index=False)
    print(vc.round(4).to_string(index=False))
    print(rel.round(4).to_string(index=False))
    print(cv.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
