"""The "is it real?" rule - one table with a tick or cross per feature and test.

A feature is a real finding only if it passes all five tests (guide, end of Part B):
  1 R3  significance after correction    q < 0.05
  2 R4  repeat selection                 picked in >= 60 % of 200 runs
  3 R5  double ML                        95 % interval excludes 0
  4 R6  shape                            always up / always down / one peak (0/1 features pass)
  5 R9  robustness                       same sign in >= 4 of 5 checks
Evaluated separately for each R2 sample (all pages, own + competitor, third party).
Features that pass everything go in the main results; the others go in the appendix with
the tests they failed.

Outputs: results/is_it_real.csv, results/is_it_real.md, figures/is_it_real.png
"""
import numpy as np
import pandas as pd

import plotstyle
from plotstyle import plt
from common import RESULTS, FIGURES
from r02_main_model import LABEL, SAMPLES

TESTS = ['1 significance (R3)', '2 repeat selection (R4)', '3 double ML (R5)', '4 shape (R6)',
         '5 robustness (R9)']


def main():
    k = ['sample', 'term']
    r3 = pd.read_csv(RESULTS / 'r03_fdr.csv')[k + ['q', 'pct_change', 'unit']]
    r4 = pd.read_csv(RESULTS / 'r04_stability_selection.csv')[k + ['selection_rate']]
    r5 = pd.read_csv(RESULTS / 'r05_double_ml.csv')[k + ['passes', 'pct_change']].rename(
        columns={'passes': 'dml_pass', 'pct_change': 'dml_pct'})
    r6 = pd.read_csv(RESULTS / 'r06_shape.csv')[k + ['shape', 'passes']].rename(columns={'passes': 'shape_pass'})
    r9 = pd.read_csv(RESULTS / 'r09_stability.csv')[k + ['checks_same_sign', 'passes']].rename(
        columns={'passes': 'robust_pass'})
    m = r3.merge(r4, on=k).merge(r5, on=k).merge(r6, on=k).merge(r9, on=k)
    m[TESTS[0]] = m.q < 0.05
    m[TESTS[1]] = m.selection_rate >= 0.60
    m[TESTS[2]] = m.dml_pass
    m[TESTS[3]] = m.shape_pass
    m[TESTS[4]] = m.robust_pass
    m['tests_passed'] = m[TESTS].sum(axis=1)
    m['real'] = m.tests_passed == 5
    m['failed'] = m[TESTS].apply(lambda r: ', '.join(t.split(' ', 1)[1] for t in TESTS if not r[t]), axis=1)
    m['direction'] = np.where(m['pct_change'] > 0, 'more citations', 'fewer citations')
    m.to_csv(RESULTS / 'is_it_real.csv', index=False)

    tick = lambda b: '✓' if b else '✗'
    lines = ['# "Is it real?" table', '',
             'Effect = R2 estimate: % more citations per +1 SD; for yes/no features (FAQ, summary box, author, structured data) % more citations when the page has it vs not. Pages are compared on the same website.', '']
    for s in SAMPLES:
        g = m[m['sample'] == s].sort_values(['tests_passed', 'pct_change'], ascending=[False, False])
        lines += [f'## {s}', '', '| feature | effect | ' + ' | '.join(TESTS) + ' | real? |',
                  '|---|---|' + '---|' * len(TESTS) + '---|']
        for _, r in g.iterrows():
            lines.append(f'| {LABEL[r.term]} | {r['pct_change']:+.1f}% | ' +
                         ' | '.join(tick(r[t]) for t in TESTS) + f' | **{"yes" if r.real else "no"}** |')
        lines.append('')
    lines += ['## Appendix: features that did not pass', '']
    for s in SAMPLES:
        g = m[(m['sample'] == s) & ~m.real]
        lines.append(f'- **{s}**: ' + '; '.join(f'{LABEL[r.term]} (failed: {r.failed})' for _, r in g.iterrows()))
    (RESULTS / 'is_it_real.md').write_text('\n'.join(lines), encoding='utf-8')

    # figure: grid of ticks, one panel per sample
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    order = (m[m['sample'] == 'all pages'].sort_values(['tests_passed', 'pct_change'])
             .term.tolist())
    for ax, s in zip(axes, SAMPLES):
        g = m[m['sample'] == s].set_index('term').reindex(order)
        for i, term in enumerate(order):
            r = g.loc[term]
            for j, t in enumerate(TESTS):
                ok = r[t]
                ax.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8,
                                           color='#2a78d6' if ok else '#ecebe6', lw=0))
                ax.text(j, i, '✓' if ok else '✗', ha='center', va='center', fontsize=9,
                        color='#fcfcfb' if ok else '#898781')
            ax.text(len(TESTS) - 0.3, i, f'{r['pct_change']:+.0f}%' + ('  REAL' if r.real else ''),
                    va='center', fontsize=8, color='#0b0b0b' if r.real else '#52514e',
                    fontweight='bold' if r.real else 'normal')
        ax.set_xlim(-0.6, len(TESTS) + 1.2)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_xticks(range(len(TESTS)), [t.split(' (')[0] for t in TESTS], rotation=35, ha='right', fontsize=8)
        ax.grid(False)
        ax.set_title(s, fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(order)), [LABEL[o] for o in order])
    fig.suptitle('Is it real? A feature counts only if it passes all five tests',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'is_it_real.png')
    plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
