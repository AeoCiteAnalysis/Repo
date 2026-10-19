"""R4 - How often is each feature picked? (repeat-selection test)

Question. If we re-sample the data many times, does a feature keep getting selected?

Same rows and variables as R2. First remove the website effect: subtract each website's
average from y and from every variable. Then draw 200 bootstrap samples of the rows and fit a
Lasso on each; a feature is "picked" when its coefficient is not zero. Missing-value flags and
page-type dummies are included as controls but only the 14 page features are reported.

Two penalty choices, both from 10-fold cross-validation (LassoCV) on each bootstrap sample:
  min   the penalty with the lowest CV error (the guide's LassoCV default). With ~12,000 URLs it
        is tiny and keeps almost every feature, so it hardly discriminates.
  1se   the largest penalty whose CV error is within one standard error of the minimum
        ("one-standard-error rule"). Keeps only features that clearly improve prediction.
        This is the version used in the "is it real?" table.
Pass: picked in >= 60 % of runs.

Outputs: results/r04_stability_selection.csv, figures/r04_selection_rates.png
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, lasso_path

import plotstyle
from plotstyle import plt, MUTED, INK2
from common import RESULTS, FIGURES, SEED, read_table
from fe import demean
from r02_main_model import prepare, FEATURES, LABEL, SAMPLES

N_BOOT = 200
PASS = 0.60


def alpha_1se(cv):
    """Largest alpha whose mean CV error <= min mean error + its standard error."""
    mse = cv.mse_path_                                   # (n_alphas, n_folds)
    mean, se = mse.mean(1), mse.std(1, ddof=1) / np.sqrt(mse.shape[1])
    i = mean.argmin()
    ok = np.where(mean <= mean[i] + se[i])[0]
    return cv.alphas_[ok].max()


def selection_rates(s, X, rng):
    s = s[s.groupby('domain').y.transform('size') >= 2]
    Z = demean(s, ['y'] + X, 'domain')
    Xz = Z[X].to_numpy()
    Xz = Xz / np.where(Xz.std(0) > 0, Xz.std(0), 1)          # Lasso needs a common scale
    y = Z.y.to_numpy()
    picked = {'min': np.zeros(len(X)), '1se': np.zeros(len(X))}
    for b in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        cv = LassoCV(cv=10, alphas=50, random_state=SEED, n_jobs=-1, max_iter=5000).fit(Xz[idx], y[idx])
        picked['min'] += cv.coef_ != 0
        _, coefs, _ = lasso_path(Xz[idx], y[idx], alphas=[alpha_1se(cv)], max_iter=5000)
        picked['1se'] += coefs[:, 0] != 0
        if (b + 1) % 50 == 0:
            print(f'    {b + 1}/{N_BOOT}')
    return ({k: pd.Series(v / N_BOOT, index=X) for k, v in picked.items()},
            len(s), s.domain.nunique())


def main():
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    X = FEATURES + flags + types
    rows = []
    for name, roles in SAMPLES.items():
        print(f'{name} ...')
        s = d if roles is None else d[d.role.isin(roles)]
        rates, n, w = selection_rates(s, X, np.random.default_rng(SEED))
        for f in FEATURES:
            rows.append(dict(sample=name, term=f, selection_rate_min=rates['min'][f],
                             selection_rate=rates['1se'][f], passes=rates['1se'][f] >= PASS,
                             n_urls=n, n_websites=w))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'r04_stability_selection.csv', index=False)

    order = (out[out['sample'] == 'all pages'].sort_values('selection_rate').term.tolist())
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True, sharex=True)
    for ax, name in zip(axes, SAMPLES):
        g = out[out['sample'] == name].set_index('term').reindex(order)
        y = np.arange(len(order))
        ax.barh(y, 100 * g.selection_rate_min, color='#e1e0d9', height=0.8, label='min-error penalty (guide)')
        ax.barh(y, 100 * g.selection_rate, color=['#2a78d6' if p else '#9ec5f4' for p in g.passes],
                height=0.45, label='1-SE penalty (used)')
        for yi, v in zip(y, 100 * g.selection_rate):
            ax.text(v + 1, yi, f'{v:.0f}%', va='center', fontsize=7, color=INK2)
        ax.axvline(100 * PASS, color=MUTED, ls='--', lw=0.8)
        ax.set_title(f'{name}\n{g.n_urls.iloc[0]:,} URLs', fontsize=9.5)
        ax.set_xlabel('% of 200 bootstrap runs picked')
        ax.set_xlim(0, 112)
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(order)), [LABEL[o] for o in order])
    axes[-1].legend(loc='lower right', fontsize=7)
    fig.suptitle('R4: how often Lasso picks each feature (dark blue = passes with the 1-SE penalty, >= 60 %)',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'r04_selection_rates.png')
    plt.close(fig)
    show = out.assign(v=(100 * out.selection_rate_min).round(0).astype(int).astype(str) + ' -> '
                      + (100 * out.selection_rate).round(0).astype(int).astype(str))
    print('selection rate % (min-error -> 1-SE):')
    print(show.pivot(index='term', columns='sample', values='v')[list(SAMPLES)].to_string())


if __name__ == '__main__':
    main()
