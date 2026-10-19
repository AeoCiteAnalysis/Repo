"""R11 - How do the engines differ?

Question. Do the engines differ in how fresh the cited pages are, how much they cite brand
sites, and which platforms they use?

Only prompt-days answered by all three engines (26,069 per engine). Citation = URL once per
answer. Page age at citation = citation day - page date (publish date, update date if none;
ages < 0 dropped as extraction errors).
  freshness   median age per engine and page type; share of citations younger than X months
  brand share own / competitor / third party, plain and weighted by 1/position
  platforms   Reddit, YouTube, LinkedIn, Wikipedia, Facebook, Quora and review / comparison
              sites (fixed list below; N14 will label website types properly)

Outputs: results/r11_freshness.csv, results/r11_brand_share.csv, results/r11_platforms.csv,
         figures/r11_freshness.png, figures/r11_brand_share.png, figures/r11_platforms.png
"""
import numpy as np
import pandas as pd

import plotstyle
from plotstyle import plt, ENGINE_COLOR, INK2, MUTED
from common import CLIENT, TABLES, RESULTS, FIGURES, ENGINES, ENGINE_LABEL, read_table, drop_excluded

REVIEW_SITES = {'highspeedinternet.com', 'broadbandnow.com', 'cnet.com', 'pcmag.com', 'reviews.org',
                'allconnect.com', 'cabletv.com', 'compareinternet.com', 'jdpower.com',
                'trustpilot.com', 'bbb.org', 'consumerreports.org', 'tomsguide.com',
                'broadbandsearch.net', 'internetproviders.ai', 'usnews.com', 'forbes.com',
                'nerdwallet.com', 'consumeraffairs.com', 'pissedconsumer.com', 'ispreports.org',
                'compareinternethub.com', 'inmyarea.com', 'satelliteinternet.com', 'wirecutter.com',
                'nytimes.com', 'zdnet.com', 'techradar.com', 'yelp.com', 'clark.com'}
PLATFORMS = {'Reddit': {'reddit.com'}, 'YouTube': {'youtube.com', 'youtu.be'},
             'LinkedIn': {'linkedin.com'}, 'Wikipedia': {'wikipedia.org'},
             'Facebook': {'facebook.com'}, 'Quora': {'quora.com'},
             'review / comparison sites': REVIEW_SITES}
ROLES = ['own', 'competitor', 'third_party', 'mixed']


def load():
    a = drop_excluded(read_table('answers.csv', usecols=['answer_id', 'date', 'engine', 'prompt_id']))
    k = a.groupby(['prompt_id', 'date']).engine.nunique()
    common = set(k[k == 3].index)
    c = drop_excluded(read_table('citations.csv', usecols=[
        'answer_id', 'date', 'engine', 'prompt_id', 'url', 'domain', 'reg_domain', 'role',
        'position', 'is_redirect', 'repeat_in_answer']))
    c = c[~c.repeat_in_answer & c.domain.notna()]
    c = c[[x in common for x in zip(c.prompt_id, c.date)]].copy()
    pf = pd.read_csv(TABLES / 'page_features.csv', usecols=['url', 'page_date', 'page_type'])
    c = c.merge(pf, on='url', how='left')
    age = (pd.to_datetime(c.date, format='%Y%m%d') - pd.to_datetime(c.page_date, errors='coerce')).dt.days
    c['age_days'] = age.where(age >= 0)
    return c, len(common)


def freshness(c):
    rows = []
    for e in ENGINES:
        g = c[c.engine == e]
        for pt, h in [('all', g)] + list(g.groupby('page_type')):
            a = h.age_days.dropna()
            if len(a) < 30:
                continue
            rows.append(dict(engine=e, page_type=pt, n_citations_with_date=len(a),
                             share_with_date=len(a) / len(h), median_age_days=a.median(),
                             pct_under_3_months=100 * (a < 91).mean(),
                             pct_under_12_months=100 * (a < 365).mean(),
                             pct_over_3_years=100 * (a > 3 * 365).mean()))
    return pd.DataFrame(rows)


def brand_share(c):
    rows = []
    for e in ENGINES:
        g = c[c.engine == e]
        w = 1 / g.position
        for r in ROLES:
            rows.append(dict(engine=e, role=r, pct=100 * (g.role == r).mean(),
                             pct_position_weighted=100 * w[g.role == r].sum() / w.sum(),
                             n_citations=len(g)))
    return pd.DataFrame(rows)


def platforms(c):
    rows = []
    for e in ENGINES:
        g = c[c.engine == e]
        for p, doms in PLATFORMS.items():
            rows.append(dict(engine=e, platform=p, pct=100 * g.reg_domain.isin(doms).mean(),
                             n_citations=len(g)))
    return pd.DataFrame(rows)


def main():
    c, n_common = load()
    print(f'{n_common:,} prompt-days answered by all three engines; {len(c):,} citations')
    fr, bs, pl = freshness(c), brand_share(c), platforms(c)
    fr.to_csv(RESULTS / 'r11_freshness.csv', index=False)
    bs.to_csv(RESULTS / 'r11_brand_share.csv', index=False)
    pl.to_csv(RESULTS / 'r11_platforms.csv', index=False)

    # freshness curves
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    months = np.arange(0, 61)
    for e in ENGINES:
        a = c[c.engine == e].age_days.dropna()
        share = [100 * (a < m * 30.44).mean() for m in months]
        med = a.median()
        ax.plot(months, share, color=ENGINE_COLOR[e],
                label=f'{ENGINE_LABEL[e]}: median {med / 30.44:.0f} months (n = {len(a):,})')
    ax.set_xlabel('page age at citation (months)')
    ax.set_ylabel('% of dated citations younger than this')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    ax.set_title('R11: how fresh are the cited pages?')
    fig.savefig(FIGURES / 'r11_freshness.png')
    plt.close(fig)

    # brand share: grouped bars, plain vs position weighted
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, col, title in zip(axes, ['pct', 'pct_position_weighted'],
                              ['share of citations', 'weighted by 1 / position (first source counts most)']):
        x = np.arange(3)
        for i, e in enumerate(ENGINES):
            v = bs[(bs.engine == e)].set_index('role').reindex(['own', 'competitor', 'third_party'])[col]
            ax.bar(x + (i - 1) * 0.26, v, width=0.24, color=ENGINE_COLOR[e], label=ENGINE_LABEL[e])
            for xi, vi in zip(x + (i - 1) * 0.26, v):
                ax.text(xi, vi + 1, f'{vi:.0f}', ha='center', fontsize=7, color=INK2)
        ax.set_xticks(x, [f'own ({CLIENT})', 'competitor', 'third party'])
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('% of citations')
    axes[0].legend(loc='upper left')
    fig.suptitle('R11: brand websites vs third parties, by engine', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'r11_brand_share.png')
    plt.close(fig)

    # platforms
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(PLATFORMS)
    x = np.arange(len(names))
    for i, e in enumerate(ENGINES):
        v = pl[pl.engine == e].set_index('platform').reindex(names).pct
        ax.bar(x + (i - 1) * 0.26, v, width=0.24, color=ENGINE_COLOR[e], label=ENGINE_LABEL[e])
        for xi, vi in zip(x + (i - 1) * 0.26, v):
            if vi >= 0.5:
                ax.text(xi, vi + 0.5, f'{vi:.0f}', ha='center', fontsize=7, color=INK2)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel('% of citations')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left')
    ax.set_title('R11: platform shares by engine (prompt-days answered by all three)')
    fig.savefig(FIGURES / 'r11_platforms.png')
    plt.close(fig)

    print(fr[fr.page_type.isin(['all', 'pricing/plans', 'article', 'support/help'])]
          .round(1).to_string(index=False))
    print(bs.pivot(index='role', columns='engine', values=['pct', 'pct_position_weighted']).round(1).to_string())
    print(pl.pivot(index='platform', columns='engine', values='pct').round(2).to_string())


if __name__ == '__main__':
    main()
