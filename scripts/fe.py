"""Regression with one indicator per website (website fixed effects), done efficiently.

Adding ~2,500 website dummies (the guide's `+ C(domain)`) gives the same slopes as
subtracting each website's average from y and from every feature, then running OLS
(Frisch-Waugh-Lovell theorem). The residuals are identical too, so HC0 standard errors
match; HC1 only needs the degrees-of-freedom factor n / (n - k - G), where G is the
number of websites absorbed. tests/test_fe.py checks this against the dummy formula.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def demean(df, cols, group):
    g = df.groupby(group)[cols].transform('mean')
    return df[cols] - g


def fe_ols(df, y, X, group):
    """OLS of y on X with group fixed effects and HC1 errors. Returns (params DataFrame, info)."""
    d = df[[y] + X + [group]].dropna()
    d = d[d.groupby(group)[y].transform('size') >= 2]           # singletons carry no within info
    Z = demean(d, [y] + X, group)
    keep = [c for c in X if Z[c].abs().sum() > 1e-12]           # drop columns constant within groups
    fit = sm.OLS(Z[y], Z[keep]).fit(cov_type='HC0')
    n, k, G = len(Z), len(keep), d[group].nunique()
    scale = n / (n - k - G)
    se = fit.bse * np.sqrt(scale)
    from scipy import stats
    t = fit.params / se
    p = 2 * stats.norm.sf(np.abs(t))
    out = pd.DataFrame({'coef': fit.params, 'se': se, 't': t, 'p': p,
                        'ci_low': fit.params - 1.96 * se, 'ci_high': fit.params + 1.96 * se})
    info = dict(n_urls=n, n_websites=G, n_params=k, r2_within=fit.rsquared,
                dropped=[c for c in X if c not in keep])
    return out, info
