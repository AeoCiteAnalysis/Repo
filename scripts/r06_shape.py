"""R6 - Check the shape of each effect.

Question. Is the link straight, does it level off, or does it go up then down?

For every page feature (so the "is it real?" table is complete) and each R2 sample:
website averages are removed from y and all variables (as in R4), then a smooth curve
(pygam LinearGAM, 12 splines, smoothing chosen by grid search) is fitted for the feature,
with all other R2 variables as straight-line controls. The curve is therefore the shape of
the same effect R2 measures (a curve of y on the feature alone mixes in the correlated
features - e.g. title and paragraph match - and was often flat for that reason). Continuous features only; for 0/1 features the shape test does not apply
(a two-point "curve" is always straight) and they pass automatically.

Automatic label from the fitted curve between the feature's 5th and 95th percentile:
  flat            - the centred curve never leaves its own 95 % band around 0
                    (i.e. no point differs clearly from a typical page)
  always up/down  - the curve never moves against its overall direction by more than
                    10 % of its range
  one peak/dip    - exactly one such reversal
  wiggly          - more than one reversal
Pass: always up, always down, or one clear peak/dip. Check the plots by eye.

Outputs: results/r06_shape.csv, figures/r06_shapes.png (top 5 R2 features, all pages)
"""
import numpy as np
import pandas as pd
from functools import reduce
from pygam import LinearGAM, s as spline, l as linear

import plotstyle
from plotstyle import plt, MUTED
from common import RESULTS, FIGURES, read_table
from fe import demean
from r02_main_model import prepare, FEATURES, LABEL, SAMPLES

BINARY = {'has_faq', 'has_summary_box', 'has_author', 'has_schema'}
TOL = 0.10


def shape_label(x, fit, lo, hi):
    half = (hi - lo) / 2
    if np.all(np.abs(fit) <= half):
        return 'flat'
    rng = fit.max() - fit.min()
    if rng == 0:
        return 'flat'
    # count reversals: moves against the running direction larger than TOL * range
    ext, direction, reversals = fit[0], 0, 0
    for v in fit[1:]:
        if direction >= 0 and v > ext:
            ext, direction = v, (1 if direction == 0 else direction)
        elif direction <= 0 and v < ext:
            ext, direction = v, (-1 if direction == 0 else direction)
        elif abs(v - ext) > TOL * rng:
            reversals += 1
            direction, ext = -direction if direction else (1 if v > ext else -1), v
    up = fit[-1] > fit[0]
    if reversals == 0:
        return 'always up' if up else 'always down'
    return 'one peak/dip' if reversals == 1 else 'wiggly'


def fit_shape(Z, f, controls):
    cols = [f] + [c for c in controls if c != f and Z[c].std() > 0]
    x = Z[cols].to_numpy()
    terms = reduce(lambda a, b: a + b, [spline(0, n_splines=12)] +
                   [linear(i, lam=0) for i in range(1, len(cols))])
    lams = np.logspace(-3, 3, 11)
    g = LinearGAM(terms).gridsearch(x, Z.y.to_numpy(), progress=False,
                                    lam=[lams] + [[0]] * (len(cols) - 1))
    lo_x, hi_x = np.percentile(x[:, 0], [5, 95])
    grid = np.zeros((100, len(cols)))
    grid[:, 0] = np.linspace(lo_x, hi_x, 100)
    # curve relative to the median page, with a 95 % band for that DIFFERENCE
    # (pygam's own band also carries the uncertainty of the curve's overall level)
    B = g._modelmat(grid, term=0).toarray()
    x_mid = np.zeros((1, len(cols)))
    x_mid[0, 0] = np.median(x[:, 0])
    D = B - g._modelmat(x_mid, term=0).toarray()
    idx = g.terms.get_coef_indices(0)
    diff = D[:, idx] @ g.coef_[idx]
    cov = g.statistics_['cov'][np.ix_(idx, idx)]
    se = np.sqrt(np.einsum('ij,jk,ik->i', D[:, idx], cov, D[:, idx]))
    return grid[:, 0], diff, diff - 1.96 * se, diff + 1.96 * se


def main():
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    rows, curves = [], {}
    for name, roles in SAMPLES.items():
        s = d if roles is None else d[d.role.isin(roles)]
        s = s[s.groupby('domain').y.transform('size') >= 2]
        X = FEATURES + flags + types
        Z = demean(s, ['y'] + X, 'domain')
        for f in FEATURES:
            if f in BINARY:
                rows.append(dict(sample=name, term=f, shape='binary (n/a)', passes=True))
                continue
            x, fit, lo, hi = fit_shape(Z, f, X)
            lab = shape_label(x, fit, lo, hi)
            rows.append(dict(sample=name, term=f, shape=lab,
                             passes=lab in ('always up', 'always down', 'one peak/dip')))
            curves[(name, f)] = (x, fit, lo, hi)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'r06_shape.csv', index=False)

    r2 = pd.read_csv(RESULTS / 'r02_main_model.csv')
    top = (r2[(r2['sample'] == 'all pages') & (r2.kind == 'feature') & ~r2.term.isin(BINARY)]
           .assign(a=lambda x: (x.coef / x.se).abs()).nlargest(5, 'a').term.tolist())
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    for ax, f in zip(axes, top):
        x, fit, lo, hi = curves[('all pages', f)]
        ax.fill_between(x, lo, hi, color='#9ec5f4', alpha=0.6, lw=0)
        ax.plot(x, fit, color='#2a78d6')
        ax.axhline(0, color=MUTED, lw=0.7)
        lab = out[(out['sample'] == 'all pages') & (out.term == f)]['shape'].iloc[0]
        ax.set_title(f'{LABEL[f]}\n({lab})', fontsize=9)
        ax.set_xlabel('feature (SD from website mean)')
    axes[0].set_ylabel('effect on log2 citations')
    fig.suptitle('R6: shape of the five strongest effects (all pages, within websites, other features held fixed)',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'r06_shapes.png')
    plt.close(fig)
    print(out.pivot(index='term', columns='sample', values='shape')[list(SAMPLES)].to_string())


if __name__ == '__main__':
    main()
