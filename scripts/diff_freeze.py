"""Compare the study-period results before and after the data freeze (literal numbers in §3-§5).

  python scripts/diff_freeze.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'results' / 'pre_freeze_20261007'
NEW = ROOT / 'results'
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 30)


def show(name, f, cols=None, key=None, sub=''):
    o, n = OLD / sub / f, NEW / sub / f
    if not o.exists() or not n.exists():
        print(f'--- {name}: missing ({o.exists()}, {n.exists()})')
        return
    a, b = pd.read_csv(o), pd.read_csv(n)
    if cols:
        a, b = a[[c for c in cols if c in a]], b[[c for c in cols if c in b]]
    print(f'\n--- {name} ({sub or "gfiber"}/{f})')
    if key:
        m = a.merge(b, on=key, suffixes=('_old', '_new'))
        print(m.round(3).to_string(index=False))
    else:
        print('OLD'); print(a.round(3).head(12).to_string(index=False))
        print('NEW'); print(b.round(3).head(12).to_string(index=False))


for sub in ('', 'workorb'):
    show('is it real (all pages)', 'is_it_real.csv', ['sample', 'term', 'pct_change', 'tests_passed', 'failed'],
         ['sample', 'term'], sub)
show('R1 engine summary', 'r01_engine_summary.csv')
show('R13 variance split', 'r13_variance_split.csv')
show('R13 reliability', 'r13_reliability.csv')
show('R13 CV by site size', 'r13_cv_by_site_size.csv')
show('E1 breadth/depth', 'e01_engines_breadth_depth.csv')
show('E2 choice fit', 'e02_choice_fit.csv')
show('R8 Simpson', 'r08_simpson.csv')
show('X2 within page', 'x02_within_page.csv')
show('X9 citations', 'x09_citations.csv')
