"""R3 - Correct for testing many features at once.

Question. Which R2 effects are still significant after accounting for testing many
features at the same time?

Benjamini-Hochberg correction over the page-feature p-values of R2, separately for each
sample (all pages / own + competitor / third party). Missing-value flags and page-type
terms are not features and are left out of the correction.
Label: q < 0.05 'found', 0.05-0.20 'borderline', > 0.20 'not found'.

Output: results/r03_fdr.csv (the R2 feature table with q and label added)
"""
import pandas as pd
from statsmodels.stats.multitest import multipletests

from common import RESULTS


def label(q):
    return 'found' if q < 0.05 else 'borderline' if q <= 0.20 else 'not found'


def main():
    r = pd.read_csv(RESULTS / 'r02_main_model.csv')
    f = r[r.kind == 'feature'].copy()
    f['q'] = f.groupby('sample').p.transform(lambda p: multipletests(p, method='fdr_bh')[1])
    f['label'] = f.q.map(label)
    f.to_csv(RESULTS / 'r03_fdr.csv', index=False)
    wide = f.pivot(index='term', columns='sample', values='label')
    q = f.pivot(index='term', columns='sample', values='q').round(4)
    eff = f.pivot(index='term', columns='sample', values='pct_change').round(1)
    order = ['all pages', 'own + competitor', 'third party']
    print((eff[order].astype(str) + '%  q=' + q[order].astype(str) + '  ' + wide[order])
          .sort_index().to_string())


if __name__ == '__main__':
    main()
