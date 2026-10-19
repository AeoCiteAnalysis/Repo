"""R5 - Double check with machine learning (double ML).

Question. Does a feature's effect survive when a flexible model takes care of all the
other features?

For each feature X, with the same rows as R2:
  1. Remove the website effect: subtract website averages from y and all variables
     (the same website indicators as R2, removed exactly instead of fed to the trees).
  2. Predict y from all OTHER variables with LightGBM, 5-fold cross-fitting (each row
     predicted by a model that never saw it); keep the leftover part of y.
  3. Same for X; keep the leftover part of X.
  4. Regress leftover y on leftover X with HC1 errors. The slope is the cleaned-up effect.
Pass: 95 % interval excludes zero. Compare with R2 - they should be close.

Outputs: results/r05_double_ml.csv, figures/r05_dml_vs_r2.png
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_predict

import plotstyle
from plotstyle import plt, MUTED
from common import unit_factors, unit_of, pct_in_unit, RESULTS, FIGURES, SEED, read_table
from fe import demean
from r02_main_model import prepare, FEATURES, LABEL, SAMPLES


def model():
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         min_child_samples=20, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=8)


def dml(s, X):
    s = s[s.groupby('domain').y.transform('size') >= 2]
    Z = demean(s, ['y'] + X, 'domain')
    cv = KFold(5, shuffle=True, random_state=SEED)
    rows = []
    for f in FEATURES:
        W = Z[[c for c in X if c != f]]
        ry = Z.y - cross_val_predict(model(), W, Z.y, cv=cv)
        rx = Z[f] - cross_val_predict(model(), W, Z[f], cv=cv)
        fit = sm.OLS(ry, rx).fit(cov_type='HC1')       # both residuals have mean ~0
        b, se = fit.params.iloc[0], fit.bse.iloc[0]
        rows.append(dict(term=f, coef=b, se=se, p=fit.pvalues.iloc[0],
                         ci_low=b - 1.96 * se, ci_high=b + 1.96 * se))
        print(f'    {f}: {b:+.3f} ({se:.3f})')
    out = pd.DataFrame(rows)
    out['n_urls'], out['n_websites'] = len(s), s.domain.nunique()
    return out


def plot(out):
    """Paired dots per feature: R2 estimate (grey) and double-ML estimate with 95 % interval (blue)."""
    order = out[out['sample'] == 'all pages'].sort_values('r2_coef').term.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True, sharex=True)
    for ax, name in zip(axes, SAMPLES):
        g = out[out['sample'] == name].set_index('term').reindex(order)
        uf = g.unit_factor.fillna(1.0) if 'unit_factor' in g else 1.0
        pct = lambda b: 100 * (2 ** (b * uf) - 1)
        y = np.arange(len(order))
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.hlines(y - 0.15, pct(g.coef - 1.96 * g.se), pct(g.coef + 1.96 * g.se), color='#9ec5f4', lw=2)
        ax.plot(pct(g.coef), y - 0.15, 'o', color='#2a78d6', ms=6, mec='#fcfcfb', label='double ML (95 % interval)')
        ax.plot(pct(g.r2_coef), y + 0.15, 'D', color='#898781', ms=4.5, label='R2 estimate')
        ax.set_title(f'{name}\n{g.n_urls.iloc[0]:,} URLs', fontsize=9.5)
        ax.set_xlabel('% more citations per +1 SD\n(yes/no features: has vs not)')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(order)), [LABEL[o] for o in order])
    axes[-1].legend(loc='lower right', fontsize=7.5)
    fig.suptitle('R5: double-ML effects next to the R2 estimates', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'r05_dml_vs_r2.png')
    plt.close(fig)


def main():
    import sys
    if '--plot-only' in sys.argv:
        plot(pd.read_csv(RESULTS / 'r05_double_ml.csv'))
        return
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    X = FEATURES + flags + types
    res = []
    for name, roles in SAMPLES.items():
        print(f'{name} ...')
        s = d if roles is None else d[d.role.isin(roles)]
        res.append(dml(s, X).assign(sample=name))
    out = pd.concat(res, ignore_index=True)
    out['pct_change_per_sd'] = 100 * (2 ** out.coef - 1)
    fac = unit_factors(d)
    out['unit'] = out.term.map(unit_of)
    out['unit_factor'] = out.term.map(fac).fillna(1.0)
    out['pct_change'] = 100 * (2 ** (out.coef * out.unit_factor) - 1)
    out['passes'] = (out.ci_low > 0) | (out.ci_high < 0)
    r2 = pd.read_csv(RESULTS / 'r02_main_model.csv')
    r2 = r2[r2.kind == 'feature'][['sample', 'term', 'coef', 'se']].rename(
        columns={'coef': 'r2_coef', 'se': 'r2_se'})
    out = out.merge(r2, on=['sample', 'term'], how='left')
    out.to_csv(RESULTS / 'r05_double_ml.csv', index=False)

    plot(out)
    show = out.assign(v=out.pct_change_per_sd.round(1).astype(str)
                      + np.where(out.passes, '*', ''))
    print(show.pivot(index='term', columns='sample', values='v')[list(SAMPLES)].to_string())


if __name__ == '__main__':
    main()
