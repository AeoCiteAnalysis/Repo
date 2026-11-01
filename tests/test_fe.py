import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from fe import fe_ols


def test_fe_matches_dummy_regression():
    rng = np.random.default_rng(42)
    n, G = 400, 40
    d = pd.DataFrame({'domain': rng.integers(0, G, n).astype(str),
                      'x1': rng.normal(size=n), 'x2': rng.integers(0, 2, n).astype(float)})
    site = d.domain.astype(int) * 0.3
    d['y'] = 0.5 * d.x1 - 0.2 * d.x2 + site + rng.normal(size=n)
    ours, info = fe_ols(d, 'y', ['x1', 'x2'], 'domain')
    ref = smf.ols('y ~ x1 + x2 + C(domain)', data=d).fit(cov_type='HC1')
    for x in ('x1', 'x2'):
        assert abs(ours.loc[x, 'coef'] - ref.params[x]) < 1e-10
        assert abs(ours.loc[x, 'se'] - ref.bse[x]) < 1e-10
    assert info['n_websites'] == d.domain.nunique()
