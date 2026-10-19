"""E1 - Where is the unexplained variation: which engine, breadth or depth?

R13 pooled the three engines and measured total citations. Here the same analysis is repeated
per engine and for two parts of the total:
  total    T = log2(1 + citations)
  breadth  B = log2(1 + number of prompts the page was cited for)
  depth    D = T - B   (roughly log2 of citations per cited prompt)
For each engine x outcome, on the R2 pages cited at least once by that engine (websites with >= 2
such pages):
  within_share        share of the variation within websites (mixed model, random website intercept)
  explained_within    share of the within-website part explained by the R2 page variables
  reliability_within  split-half (odd vs even days) reliability of the within-website part,
                      Spearman-Brown 2r / (1 + r)
  stable_unexplained  within_share x (reliability_within - explained_within): the share of ALL
                      variation that is repeatable, inside websites and not explained - the '72 %'
                      of R13, now per engine and part.

Outputs: results/e01_engines_breadth_depth.csv, figures/e01_engines_breadth_depth.png
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import plotstyle
from plotstyle import plt, ENGINE_COLOR, INK2
from common import RESULTS, FIGURES, ENGINES, ENGINE_LABEL, read_table, drop_excluded
from r02_main_model import prepare, FEATURES

warnings.filterwarnings('ignore')


def outcomes(c):
    """T, B, D from a citation subset, indexed by url."""
    n = c.groupby('url').size()
    b = c.groupby('url').prompt_id.nunique()
    T, B = np.log2(1 + n), np.log2(1 + b)
    return pd.DataFrame({'total': T, 'breadth': B, 'depth': T - B})


def vc(d, y, X):
    X = [x for x in X if d[x].std() > 0 and d[x].sum() >= 5]       # drop empty dummies in this subset
    m0 = sm.MixedLM(d[y], np.ones((len(d), 1)), groups=d.domain).fit(reml=True)
    m1 = sm.MixedLM(d[y], sm.add_constant(d[X]), groups=d.domain).fit(reml=True)
    vb, vw = float(m0.cov_re.iloc[0, 0]), float(m0.scale)
    return vw / (vb + vw), 1 - float(m1.scale) / vw


def rel_within(a, b, dom):
    wa, wb = a - a.groupby(dom).transform('mean'), b - b.groupby(dom).transform('mean')
    r = np.corrcoef(wa, wb)[0, 1]
    return 2 * r / (1 + r)


def main():
    t = read_table('url_table.csv')
    d0, flags, types = prepare(t)
    X = FEATURES + flags + types
    c = drop_excluded(read_table('citations.csv', usecols=['date', 'engine', 'prompt_id', 'url', 'is_redirect',
                                                          'repeat_in_answer']))
    c = c[~c.is_redirect & ~c.repeat_in_answer & c.url.isin(d0.url)]
    order = {x: i for i, x in enumerate(sorted(c.date.unique()))}
    c['half'] = c.date.map(order) % 2
    rows = []
    for e in ENGINES + ['all']:
        ce = c if e == 'all' else c[c.engine == e]
        full = outcomes(ce)
        d = d0[d0.url.isin(full.index)].copy()
        d = d[d.groupby('domain').url.transform('size') >= 2].reset_index(drop=True)
        halves = [outcomes(ce[ce.half == h]).reindex(d.url, fill_value=0).reset_index(drop=True) for h in (0, 1)]
        for y in ('total', 'breadth', 'depth'):
            d[y] = full[y].reindex(d.url).to_numpy()
            ws, ex = vc(d, y, X)
            rel = rel_within(halves[0][y], halves[1][y], d.domain)
            rows.append(dict(engine=ENGINE_LABEL.get(e, 'All engines'), outcome=y, n_urls=len(d),
                             n_websites=d.domain.nunique(), variance=d[y].var(), within_share=ws,
                             explained_within=ex, reliability_within=rel,
                             stable_unexplained=ws * (rel - ex)))
            print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / 'e01_engines_breadth_depth.csv', index=False)
    print(out.round(3).to_string(index=False))
    plot(out)


def plot(out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    cols = {'ChatGPT': ENGINE_COLOR['OPENAI'], 'Gemini': ENGINE_COLOR['GEMINI'],
            'Perplexity': ENGINE_COLOR['PERPLEXITY'], 'All engines': '#52514c'}
    for ax, y in zip(axes, ('total', 'breadth', 'depth')):
        s = out[out.outcome == y].reset_index(drop=True)
        x = np.arange(len(s))
        ax.set_ylim(0, 100)
        ax.bar(x - 0.2, 100 * s.explained_within.clip(lower=0), 0.38, color=[cols[e] for e in s.engine], label='explained by page variables')
        ax.bar(x + 0.2, 100 * s.reliability_within, 0.38, color=[cols[e] for e in s.engine], alpha=0.35,
               label='repeatable (reliability)')
        for xi, a, b in zip(x, s.explained_within, s.reliability_within):
            ax.text(xi - 0.2, 100 * max(a, 0) + 1, f'{100 * max(a, 0):.0f}', ha='center', fontsize=7, color=INK2)
            ax.text(xi + 0.2, 100 * b + 1, f'{100 * b:.0f}', ha='center', fontsize=7, color=INK2)
        ax.set_xticks(x, s.engine, fontsize=8)
        ax.set_title(y, fontsize=9.5)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('% of within-website variation')
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7.5, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('E1: within-website variation explained (solid) vs repeatable (light), by engine and part',
                 x=0.01, ha='left', fontweight='bold', fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES / 'e01_engines_breadth_depth.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
