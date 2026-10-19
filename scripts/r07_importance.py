"""R7 - Which features matter most overall? (SHAP feature importance)

Question. How important is the website compared with page features?

Rows: URLs cited >= 2 times that have page features (all websites, also those with one URL).
Model: LightGBM predicting y = log2(1 + citations), 5-fold cross-validation.
Three versions:
  A  page features + page type + WEBSITE SCORE
  B  page features + page type + OUTSIDE AUTHORITY (Common Crawl, Majestic, Tranco, CrUX)
  C  page features + page type + both
Website score = target encoding: the average y of OTHER URLs on the same website, shrunk
toward the overall average for small websites (weight n / (n + 20)). It is computed inside
cross-validation: test URLs are encoded from the training folds only, and training URLs
from an inner 5-fold split, so a URL never sees its own citation count.
Importance = mean |SHAP value| on the held-out folds.

Outputs: results/r07_importance.csv, results/r07_model_fit.csv, figures/r07_importance.png
"""
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

import plotstyle
from plotstyle import plt, INK2
from common import RESULTS, FIGURES, SEED, read_table
from r02_main_model import FEATURES, LABEL

AUTHORITY = ['cc_harmonic', 'log_cc_pagerank', 'log_majestic_ref_subnets', 'log_tranco_rank',
             'log_crux_rank']
AUTH_LABEL = {'cc_harmonic': 'authority: CC harmonic', 'log_cc_pagerank': 'authority: CC PageRank',
              'log_majestic_ref_subnets': 'authority: Majestic linking networks',
              'log_tranco_rank': 'authority: Tranco rank', 'log_crux_rank': 'authority: CrUX rank',
              'website_score': 'WEBSITE SCORE (in-data)', 'page_type': 'page type'}
SHRINK = 20


def encode(train, test_domains):
    """Shrunk mean y per domain from `train`, applied to `test_domains`."""
    g = train.groupby('domain').y.agg(['sum', 'count'])
    prior = train.y.mean()
    enc = (g['sum'] + SHRINK * prior) / (g['count'] + SHRINK)
    return test_domains.map(enc).fillna(prior).to_numpy()


def oof_encode(train, rng_seed):
    out = np.empty(len(train))
    for tr, te in KFold(5, shuffle=True, random_state=rng_seed).split(train):
        out[te] = encode(train.iloc[tr], train.domain.iloc[te])
    return out


def model():
    return LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                         min_child_samples=20, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=8)


def run(d, cols, use_site):
    X_all = d[cols].copy()
    shap_abs = pd.Series(0.0, index=cols + (['website_score'] if use_site else []))
    pred = np.empty(len(d))
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(d)):
        Xtr, Xte = X_all.iloc[tr].copy(), X_all.iloc[te].copy()
        if use_site:
            Xtr['website_score'] = oof_encode(d.iloc[tr], SEED + k)
            Xte['website_score'] = encode(d.iloc[tr], d.domain.iloc[te])
        m = model().fit(Xtr, d.y.iloc[tr], categorical_feature=['page_type'])
        pred[te] = m.predict(Xte)
        sv = shap.TreeExplainer(m).shap_values(Xte)
        shap_abs += pd.Series(np.abs(sv).mean(0), index=Xte.columns) * len(te)
    shap_abs /= len(d)
    r2 = 1 - ((d.y - pred) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum()
    return shap_abs.sort_values(ascending=False), r2


def main():
    t = read_table('url_table.csv')
    d = t[t.word_count.notna()].copy().reset_index(drop=True)
    d['page_type'] = d.page_type.astype('category')
    versions = {'A: page + website score': (FEATURES + ['page_type'], True),
                'B: page + outside authority': (FEATURES + ['page_type'] + AUTHORITY, False),
                'C: page + website score + authority': (FEATURES + ['page_type'] + AUTHORITY, True)}
    imp, fit = [], []
    for name, (cols, site) in versions.items():
        s, r2 = run(d, cols, site)
        page = s[[c for c in s.index if c in FEATURES]]
        imp.append(pd.DataFrame({'version': name, 'feature': s.index, 'mean_abs_shap': s.values,
                                 'rank': range(1, len(s) + 1)}))
        site = s[[c for c in s.index if c in AUTHORITY + ['website_score']]]
        top_other = site.iloc[0]
        fit.append(dict(version=name, cv_r2=r2, n_urls=len(d), n_websites=d.domain.nunique(),
                        strongest_page_feature=page.index[0], strongest_page_shap=page.iloc[0],
                        strongest_non_page=site.index[0], strongest_non_page_shap=top_other,
                        ratio_non_page_to_page=top_other / page.iloc[0]))
        print(f'{name}: CV R2 = {r2:.3f}; top: ' +
              ', '.join(f'{i} {v:.3f}' for i, v in s.head(6).items()))
    imp = pd.concat(imp, ignore_index=True)
    imp.to_csv(RESULTS / 'r07_importance.csv', index=False)
    fit = pd.DataFrame(fit)
    fit.to_csv(RESULTS / 'r07_model_fit.csv', index=False)
    print(fit[['version', 'cv_r2', 'strongest_non_page', 'strongest_page_feature',
               'ratio_non_page_to_page']].round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True)
    for ax, (name, g) in zip(axes, imp.groupby('version', sort=False)):
        g = g.head(15).iloc[::-1]
        labels = [LABEL.get(f, AUTH_LABEL.get(f, f)) for f in g.feature]
        colors = ['#2a78d6' if f in FEATURES else '#eb6834' for f in g.feature]
        ax.barh(labels, g.mean_abs_shap, color=colors, height=0.7)
        for y, v in enumerate(g.mean_abs_shap):
            ax.text(v + 0.003, y, f'{v:.3f}', va='center', fontsize=7, color=INK2)
        r2 = fit.set_index('version').loc[name, 'cv_r2']
        ax.set_title(f'{name}\ncross-validated R² = {r2:.2f}', fontsize=9.5)
        ax.set_xlabel('mean |SHAP| (log2 citations)')
        ax.grid(axis='y', visible=False)
        ax.tick_params(axis='y', labelsize=7.5)
    from matplotlib.patches import Patch
    axes[-1].legend(handles=[Patch(color='#2a78d6', label='page feature'),
                             Patch(color='#eb6834', label='website / authority / page type')],
                    loc='lower right')
    fig.suptitle(f'R7: what matters most for citation count ({len(d):,} URLs)',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'r07_importance.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
