"""R1 - Describe the citations.

Question. How many citations does each engine give, which websites get them, and how
spread out are they?

A citation is a URL counted once per answer (repeats inside an answer dropped).
Website-level numbers keep unresolved Gemini redirects (their site comes from the
citation title); URL-level numbers drop them. Every table is produced twice: on all
answers, and on prompt-days answered by all three engines (the fair engine comparison).

Outputs
  results/r01_engine_summary.csv          one row per engine (+ All), both samples
  results/r01_top_websites.csv            top 20 websites overall and per engine
  results/r01_topic_summary.csv           per engine x topic
  results/r01_coverage_curve.csv          number of websites vs % of citations covered
  figures/r01_top_websites.png, figures/r01_top_websites_by_engine.png, figures/r01_coverage_curve.png
"""
import numpy as np
import pandas as pd

import plotstyle
from plotstyle import plt, ENGINE_COLOR, SINGLE, INK2, MUTED
from common import CLIENT, RESULTS, FIGURES, ENGINES, ENGINE_LABEL, read_table, drop_excluded

PLATFORMS = {'reddit': 'reddit.com', 'youtube': 'youtube.com', 'linkedin': 'linkedin.com'}


def load():
    a = drop_excluded(read_table('answers.csv', usecols=[
        'answer_id', 'date', 'engine', 'topic', 'prompt_id', 'n_citations', 'n_unique_citations']))
    c = drop_excluded(read_table('citations.csv', usecols=[
        'answer_id', 'date', 'engine', 'topic', 'prompt_id', 'url', 'domain', 'reg_domain',
        'role', 'is_redirect', 'repeat_in_answer']))
    c = c[~c.repeat_in_answer & c.domain.notna()]
    # prompt-days answered by all three engines
    k = a.groupby(['prompt_id', 'date']).engine.nunique()
    common = set(k[k == 3].index)
    a['common'] = [x in common for x in zip(a.prompt_id, a.date)]
    c['common'] = [x in common for x in zip(c.prompt_id, c.date)]
    return a, c


def websites_for_share(counts, share=0.8):
    cum = counts.sort_values(ascending=False).cumsum() / counts.sum()
    return int((cum < share).sum() + 1)


def summary(a, c):
    rows = []
    for eng in ENGINES + ['All']:
        aa = a if eng == 'All' else a[a.engine == eng]
        cc = c if eng == 'All' else c[c.engine == eng]
        if aa.empty:
            continue
        site = cc.domain.value_counts()
        urls = cc[~cc.is_redirect]
        r = dict(engine=eng, answers=len(aa), prompts=aa.prompt_id.nunique(),
                 citations=len(cc),
                 listed_per_answer=aa.n_citations.mean(),
                 unique_per_answer=aa.n_unique_citations.mean(),
                 pct_answers_no_citation=100 * (aa.n_citations == 0).mean(),
                 distinct_urls=urls.url.nunique(), distinct_websites=site.size,
                 top10_website_share=100 * site.head(10).sum() / max(site.sum(), 1),
                 websites_for_80pct=websites_for_share(site) if site.size else np.nan)
        for name, dom in PLATFORMS.items():
            r[f'pct_{name}'] = 100 * (cc.reg_domain == dom).mean() if len(cc) else np.nan
        for role in ('own', 'competitor', 'third_party', 'mixed'):
            r[f'pct_{role}'] = 100 * (cc.role == role).mean() if len(cc) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def top_websites(c, n=20):
    out = []
    for eng in ['All'] + ENGINES:
        cc = c if eng == 'All' else c[c.engine == eng]
        vc = cc.domain.value_counts()
        roles = cc.drop_duplicates('domain').set_index('domain').role
        t = vc.head(n).rename('citations').to_frame()
        t['share_pct'] = 100 * t.citations / vc.sum()
        t['rank'] = range(1, len(t) + 1)
        t['role'] = roles.reindex(t.index)
        t['engine'] = eng
        out.append(t.rename_axis('website').reset_index())
    return pd.concat(out, ignore_index=True)


def topic_summary(a, c):
    rows = []
    for (eng, topic), aa in a.groupby(['engine', 'topic']):
        cc = c[(c.engine == eng) & (c.topic == topic)]
        site = cc.domain.value_counts()
        rows.append(dict(
            engine=eng, topic=topic, answers=len(aa), citations=len(cc),
            unique_per_answer=aa.n_unique_citations.mean(),
            pct_answers_no_citation=100 * (aa.n_citations == 0).mean(),
            distinct_websites=site.size,
            top10_website_share=100 * site.head(10).sum() / max(site.sum(), 1),
            websites_for_80pct=websites_for_share(site) if site.size else np.nan,
            pct_reddit=100 * (cc.reg_domain == 'reddit.com').mean() if len(cc) else np.nan,
            pct_youtube=100 * (cc.reg_domain == 'youtube.com').mean() if len(cc) else np.nan,
            pct_own=100 * (cc.role == 'own').mean() if len(cc) else np.nan,
            pct_competitor=100 * (cc.role == 'competitor').mean() if len(cc) else np.nan))
    return pd.DataFrame(rows)


def coverage_curve(c):
    rows = []
    for eng in ENGINES:
        vc = c[c.engine == eng].domain.value_counts()
        if vc.empty:
            continue
        cum = 100 * vc.cumsum() / vc.sum()
        rows.append(pd.DataFrame({'engine': eng, 'n_websites': np.arange(1, len(vc) + 1),
                                  'pct_covered': cum.values}))
    return pd.concat(rows, ignore_index=True)


ROLE_ALPHA = {'own': 1.0, 'competitor': 0.65, 'mixed': 0.4, 'third_party': 0.4}


def plot_top(top):
    t = top[top.engine == 'All'].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = ['#2a78d6' if r == 'own' else '#9ec5f4' if r == 'competitor' else '#c3c2b7'
              for r in t.role]
    ax.barh(t.website, t.share_pct, color=colors, height=0.72)
    for y, (v, w) in enumerate(zip(t.share_pct, t.website)):
        ax.text(v + 0.1, y, f'{v:.1f}%', va='center', fontsize=7.5, color=INK2)
    ax.set_xlabel('share of all citations (%)')
    ax.grid(axis='y', visible=False)
    ax.set_title('Top 20 cited websites, all engines')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#2a78d6', label=f'own ({CLIENT})'),
                       Patch(color='#9ec5f4', label='competitor'),
                       Patch(color='#c3c2b7', label='third party / mixed')],
              loc='lower right')
    fig.savefig(FIGURES / 'r01_top_websites.png')
    plt.close(fig)


def plot_top_by_engine(top):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    for ax, eng in zip(axes, ENGINES):
        t = top[top.engine == eng].head(15).iloc[::-1]
        ax.barh(t.website, t.share_pct, color=ENGINE_COLOR[eng], height=0.72)
        for y, v in enumerate(t.share_pct):
            ax.text(v + 0.1, y, f'{v:.1f}%', va='center', fontsize=7, color=INK2)
        ax.set_title(ENGINE_LABEL[eng])
        ax.grid(axis='y', visible=False)
        ax.tick_params(axis='y', labelsize=7.5)
        ax.set_xlabel('share of engine citations (%)')
    fig.suptitle('Top 15 websites per engine', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES / 'r01_top_websites_by_engine.png')
    plt.close(fig)


def plot_curve(curve, summ):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    from matplotlib.ticker import FuncFormatter
    for eng in ENGINES:
        g = curve[curve.engine == eng]
        if g.empty:
            continue
        n80 = summ.set_index('engine').loc[eng, 'websites_for_80pct']
        ax.plot(g.n_websites, g.pct_covered, color=ENGINE_COLOR[eng],
                label=f'{ENGINE_LABEL[eng]}: {int(n80):,} websites cover 80%')
        ax.plot([n80], [80], 'o', ms=6, color=ENGINE_COLOR[eng], mec='#fcfcfb', mew=1.5)
    ax.axhline(80, color=MUTED, lw=0.8, ls='--')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.legend(loc='lower right')
    ax.set_xlabel('number of websites (most-cited first, log scale)')
    ax.set_ylabel('% of citations covered')
    ax.set_ylim(0, 102)
    ax.set_title('How concentrated are citations?')
    fig.savefig(FIGURES / 'r01_coverage_curve.png')
    plt.close(fig)


def main():
    a, c = load()
    summ_all = summary(a, c).assign(sample='all answers')
    summ_com = summary(a[a.common], c[c.common]).assign(sample='prompt-days answered by all 3')
    summ = pd.concat([summ_all, summ_com], ignore_index=True)
    summ.to_csv(RESULTS / 'r01_engine_summary.csv', index=False)
    top = top_websites(c)
    top.to_csv(RESULTS / 'r01_top_websites.csv', index=False)
    topic_summary(a, c).to_csv(RESULTS / 'r01_topic_summary.csv', index=False)
    curve = coverage_curve(c)
    curve.to_csv(RESULTS / 'r01_coverage_curve.csv', index=False)
    plot_top(top)
    plot_top_by_engine(top)
    plot_curve(curve, summ_all)

    cols = ['engine', 'answers', 'citations', 'listed_per_answer', 'unique_per_answer',
            'pct_answers_no_citation', 'distinct_urls', 'distinct_websites',
            'top10_website_share', 'websites_for_80pct', 'pct_reddit', 'pct_youtube',
            'pct_linkedin', 'pct_own', 'pct_competitor', 'pct_third_party']
    for name, s in (('ALL ANSWERS', summ_all), ('PROMPT-DAYS ANSWERED BY ALL 3', summ_com)):
        print(f'\n{name}')
        print(s[cols].set_index('engine').T.round(2).to_string())
    print('\ntop 10 websites overall:')
    print(top[top.engine == 'All'].head(10)[['website', 'citations', 'share_pct', 'role']]
          .round(2).to_string(index=False))


if __name__ == '__main__':
    main()
