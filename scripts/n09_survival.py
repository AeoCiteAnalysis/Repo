"""N9 - How long does a page stay cited?

Question. Once a page starts being cited, how long does it keep getting cited, and what makes new
pages enter?

Per engine x URL: first and last day cited and number of days with a citation.
Survival: a URL is "still cited" until its last citation day; it has "dropped out" if it is not
cited again in the final 14 days of data (otherwise censored: still active when the data end).
Only URLs first cited after the first 14 days are used, so their true first citation is known
(URLs cited on day 1 may have been cited long before the data start).
  Kaplan-Meier curves per engine and per page type (lifelines)
  Cox model of dropping out on z-scored page features + page type (URLs with page features)
  weekly new URLs: how many, which page types, how old the pages are

Outputs: results/n09_survival_summary.csv, results/n09_cox.csv, results/n09_new_urls_weekly.csv,
         figures/n09_survival.png, figures/n09_new_urls.png
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

import plotstyle
from plotstyle import plt, ENGINE_COLOR, INK2
from common import TABLES, RESULTS, FIGURES, ENGINES, ENGINE_LABEL, read_table, drop_excluded
from r02_main_model import FEATURES, LABEL

BURN_IN, TAIL = 14, 14


def spells():
    c = drop_excluded(read_table('citations.csv', usecols=[
        'date', 'engine', 'url', 'is_redirect', 'repeat_in_answer']))
    c = c[~c.is_redirect & ~c.repeat_in_answer]
    c['day'] = pd.to_datetime(c.date, format='%Y%m%d')
    start, end = c.day.min(), c.day.max()
    s = c.groupby(['engine', 'url']).day.agg(first='min', last='max', days_cited='nunique').reset_index()
    s['duration'] = (s['last'] - s['first']).dt.days + 1
    s['dropped'] = (s['last'] < end - pd.Timedelta(days=TAIL)).astype(int)
    s['new'] = s['first'] > start + pd.Timedelta(days=BURN_IN)
    return s, start, end


def main():
    s, start, end = spells()
    pt = read_table('url_table_all.csv', usecols=['url', 'page_type'] + [f for f in FEATURES]
                    + ['page_date']).set_index('url')
    s = s.join(pt, on='url')
    new = s[s.new].copy()

    rows = []
    kmf = KaplanMeierFitter()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for e in ENGINES:
        g = new[new.engine == e]
        kmf.fit(g.duration, g.dropped, label=ENGINE_LABEL[e])
        med = kmf.median_survival_time_
        rows.append(dict(group='engine', value=ENGINE_LABEL[e], urls=len(g), median_days=med,
                         still_cited_after_7d=float(kmf.predict(7)), after_30d=float(kmf.predict(30)),
                         pct_cited_one_day_only=100 * (g.days_cited == 1).mean()))
        axes[0].step(kmf.survival_function_.index, 100 * kmf.survival_function_.iloc[:, 0],
                     where='post', color=ENGINE_COLOR[e],
                     label=f'{ENGINE_LABEL[e]} (n = {len(g):,}, median {med:.0f} d)')
    axes[0].set_xlabel('days since first citation'); axes[0].set_ylabel('% of pages still being cited')
    axes[0].set_title('By engine', fontsize=9.5); axes[0].legend(loc='upper right'); axes[0].set_ylim(0, 100)
    types = new.page_type.value_counts().head(6).index
    for i, ptp in enumerate(types):
        g = new[new.page_type == ptp]
        kmf.fit(g.duration, g.dropped)
        rows.append(dict(group='page type', value=ptp, urls=len(g), median_days=kmf.median_survival_time_,
                         still_cited_after_7d=float(kmf.predict(7)), after_30d=float(kmf.predict(30)),
                         pct_cited_one_day_only=100 * (g.days_cited == 1).mean()))
        axes[1].step(kmf.survival_function_.index, 100 * kmf.survival_function_.iloc[:, 0], where='post',
                     color=['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#4a3aa7'][i],
                     label=f'{ptp} (median {kmf.median_survival_time_:.0f} d)')
    axes[1].set_xlabel('days since first citation'); axes[1].set_title('By page type (all engines)', fontsize=9.5)
    axes[1].legend(loc='upper right', fontsize=7); axes[1].set_ylim(0, 100)
    fig.suptitle(f'N9: how long pages stay cited (pages first cited after day {BURN_IN})',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(); fig.savefig(FIGURES / 'n09_survival.png'); plt.close(fig)
    pd.DataFrame(rows).to_csv(RESULTS / 'n09_survival_summary.csv', index=False)

    # Cox model on pages with features (one row per engine x URL, engine as strata)
    cx = new[new.word_count.notna()].copy()
    for f in FEATURES:
        cx[f] = cx[f].fillna(cx[f].median())
    dummies = pd.get_dummies(cx.page_type, prefix='type', drop_first=True, dtype=float)
    data = pd.concat([cx[['duration', 'dropped', 'engine'] + FEATURES], dummies], axis=1)
    cph = CoxPHFitter(penalizer=0.01).fit(data, 'duration', 'dropped', strata=['engine'])
    cox = cph.summary[['coef', 'exp(coef)', 'se(coef)', 'p']].rename(columns={'exp(coef)': 'hazard_ratio'})
    cox['n_urls'] = len(cx)
    cox.to_csv(RESULTS / 'n09_cox.csv')

    # weekly new URLs
    all_new = s.sort_values('first').drop_duplicates('url')
    all_new = all_new[all_new['first'] > start + pd.Timedelta(days=BURN_IN)].copy()
    all_new['week'] = all_new['first'].dt.to_period('W').dt.start_time
    age = (all_new['first'] - pd.to_datetime(all_new.page_date, errors='coerce')).dt.days
    all_new['page_age_at_entry'] = age.where(age >= 0)
    wk = all_new.groupby('week').agg(new_urls=('url', 'size'),
                                     median_page_age_days=('page_age_at_entry', 'median'))
    wk = wk.join(all_new.pivot_table(index='week', columns='page_type', aggfunc='size', fill_value=0))
    wk.to_csv(RESULTS / 'n09_new_urls_weekly.csv')
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(wk.index, wk.new_urls, width=5, color='#2a78d6')
    for x, v in zip(wk.index, wk.new_urls):
        ax.text(x, v, f'{v:,}', ha='center', va='bottom', fontsize=7, color=INK2)
    ax.set_ylabel('URLs cited for the first time'); ax.grid(axis='x', visible=False)
    ax.set_title('N9: new URLs entering the citations each week')
    fig.savefig(FIGURES / 'n09_new_urls.png'); plt.close(fig)

    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print('\nCox model (hazard ratio > 1 = drops out sooner), n =', len(cx))
    show = cox.loc[[f for f in FEATURES]].assign(feature=[LABEL[f] for f in FEATURES])
    print(show[['feature', 'hazard_ratio', 'p']].round(3).to_string(index=False))
    print('\nweekly new URLs:'); print(wk[['new_urls', 'median_page_age_days']].to_string())


if __name__ == '__main__':
    main()
