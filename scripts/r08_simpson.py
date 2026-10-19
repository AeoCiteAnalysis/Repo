"""R8 - Does the website hide or fake effects? (Simpson's paradox test)

Question. Do common "AI optimisation" tips (FAQ block, structured data, summary box,
author box) still look useful once we compare pages within the same website?

For each tip, on the same rows as R2, three estimates:
  pooled, alone      y on the tip only (all pages thrown together)
  pooled, full model y on all R2 variables, no website indicators
  within website     the R2 model (website indicators)
All with HC1 errors. A tip whose effect shrinks to zero or flips sign between "pooled" and
"within website" was really a website effect.
Speed: when data/tables/page_speed.csv exists (scripts/speed.py), LCP, INP and CLS are combined
into one speed score (minus the average of their z-scores, so higher = faster). Only pages with
their OWN Chrome UX data (crux_level = url) are used: for the others speed.py stores the website's
average, which is the same for every page of a website, so a within-website comparison would
only contrast popular pages (which have their own data and are cited more) with the rest.
Few pages qualify (~400); a Lighthouse score for more pages (speed.py psi) gives more power.
Samples: third-party pages (the guide's choice) and all pages.

Outputs: results/r08_simpson.csv, figures/r08_before_after.png
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import plotstyle
from plotstyle import plt, MUTED, INK2
from common import DATASET, unit_factors, unit_of, pct_in_unit, RESULTS, FIGURES, read_table
from fe import fe_ols
from r02_main_model import prepare, FEATURES, LABEL

TIPS = ['has_faq', 'has_schema', 'has_summary_box', 'has_author']
SAMPLES = {'third party': ['third_party'], 'all pages': None}


def ols(s, cols):
    fit = sm.OLS(s.y, sm.add_constant(s[cols])).fit(cov_type='HC1')
    return fit.params, fit.bse


def add_speed(d):
    """Speed score = -(mean z of LCP, INP, CLS); None if speed was not measured."""
    from common import TABLES
    path = TABLES / 'page_speed.csv'
    if not path.exists():
        return d, []
    lvl = pd.read_csv(path, usecols=['url', 'crux_level']).drop_duplicates('url').set_index('url').crux_level
    cols = [c for c in ('lcp', 'inp', 'cls') if c in d and d[c].notna().sum() > 100]
    if not cols:
        return d, []
    d = d.copy()
    page_level = d.url.map(lvl).eq('url')
    d['speed_score'] = (-d[cols].mean(axis=1)).where(page_level)
    d['speed_score'] = (d.speed_score - d.speed_score.mean()) / d.speed_score.std()
    d['speed_score_raw'] = d.speed_score
    LABEL['speed_score'] = 'faster page (speed score)'
    n = d.speed_score.notna().sum()
    print(f'speed score from {cols}: {n:,} URLs with their own (page-level) data')
    if n < 100:
        print('speed not testable: fewer than 100 pages with page-level Chrome UX data')
        return d, []
    return d, ['speed_score']


def main():
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    d, speed = add_speed(d)
    tips = TIPS + speed
    X = FEATURES + flags + types
    fac = unit_factors(d)                       # yes/no tips reported as has vs not
    rows = []
    for name, roles in SAMPLES.items():
        s0 = d if roles is None else d[d.role.isin(roles)]
        for tip in tips:
            s, Xt = (s0, X) if tip not in speed else (s0[s0[tip].notna()], X + [tip])
            s = s[s.groupby('domain').y.transform('size') >= 2]        # same rows as R2
            within, info = fe_ols(s, 'y', Xt, 'domain')
            full_b, full_se = ols(s, Xt)
            alone_b, alone_se = ols(s, [tip])
            for kind, b, se in (('pooled, alone', alone_b[tip], alone_se[tip]),
                                ('pooled, full model', full_b[tip], full_se[tip]),
                                ('within website', within.loc[tip, 'coef'], within.loc[tip, 'se'])):
                rows.append(dict(sample=name, tip=tip, estimate=kind, coef=b, se=se,
                                 ci_low=b - 1.96 * se, ci_high=b + 1.96 * se,
                                 pct_change_per_sd=100 * (2 ** b - 1), unit=unit_of(tip),
                                 pct_change=pct_in_unit(b, tip, fac), unit_factor=fac.get(tip, 1.0),
                                 n_urls=len(s), n_websites=s.domain.nunique(),
                                 raw_share_with_tip=s[tip + '_raw'].mean()))
    out = pd.DataFrame(rows)
    out['significant'] = (out.ci_low > 0) | (out.ci_high < 0)
    out.to_csv(RESULTS / 'r08_simpson.csv', index=False)

    # verdict per tip: 'before' = pooled, tip alone (guide); 'after' = within website
    w = out.pivot_table(index=['sample', 'tip'], columns='estimate', values=['coef', 'significant'],
                        aggfunc='first')
    verdict = []
    for (smp, tip), r in w.iterrows():
        pb, wb = r[('coef', 'pooled, alone')], r[('coef', 'within website')]
        ps, ws = r[('significant', 'pooled, alone')], r[('significant', 'within website')]
        if np.sign(pb) != np.sign(wb) and ps and ws:
            v = 'sign flips'
        elif ps and not ws:
            v = 'disappears within websites'
        elif ws and not ps:
            v = 'appears only within websites'
        elif ws and ps:
            v = 'holds'
        else:
            v = 'no effect either way'
        verdict.append(dict(sample=smp, tip=tip, verdict=v))
    verdict = pd.DataFrame(verdict)
    out = out.merge(verdict, on=['sample', 'tip'])
    out.to_csv(RESULTS / 'r08_simpson.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True, sharey=True)
    style = {'pooled, alone': ('o', '#c3c2b7'), 'pooled, full model': ('s', '#eb6834'),
             'within website': ('o', '#2a78d6')}
    for ax, name in zip(axes, SAMPLES):
        g = out[out['sample'] == name]
        for i, tip in enumerate(tips):
            for j, (kind, (mk, col)) in enumerate(style.items()):
                r = g[(g.tip == tip) & (g.estimate == kind)].iloc[0]
                y = i + (j - 1) * 0.22
                lo, hi = 100 * (2 ** (r.ci_low * r.unit_factor) - 1), 100 * (2 ** (r.ci_high * r.unit_factor) - 1)
                ax.hlines(y, lo, hi, color=col, lw=1.5, alpha=0.7)
                ax.plot(r['pct_change'], y, mk, color=col, ms=6, mec='#fcfcfb', mew=1,
                        label=kind if i == 0 else None)
        ax.axvline(0, color=MUTED, lw=0.8)
        n = g.n_urls.iloc[0]; wn = g.n_websites.iloc[0]
        ax.set_title(f'{name} ({n:,} URLs, {wn:,} websites)', fontsize=9.5)
        ax.set_xlabel('% more citations: page has the tip vs not (speed: per +1 SD)')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(tips)), [LABEL[t] for t in tips])
    axes[0].invert_yaxis()
    axes[-1].legend(loc='lower right', fontsize=7.5)
    if DATASET == 'gfiber':                       # paper version: no embedded title, the LaTeX caption is the title
        fig.tight_layout()
        (FIGURES / 'paper').mkdir(exist_ok=True)
        fig.savefig(FIGURES / 'paper' / 'r08_before_after.png')
    fig.suptitle('R8: "AI optimisation" tips before and after comparing within websites',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'r08_before_after.png')
    plt.close(fig)
    show = out.assign(v=out.pct_change_per_sd.round(1).astype(str) + np.where(out.significant, '*', ''))
    print(show.pivot_table(index=['sample', 'tip', 'verdict'], columns='estimate', values='v',
                           aggfunc='first').to_string())


if __name__ == '__main__':
    main()
