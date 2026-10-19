"""Step A7 - build the main analysis table, one row per URL.

  data/tables/url_table.csv       URLs cited >= 2 times (main models)
  data/tables/url_table_all.csv   every URL (checks)
  data/tables/url_counts.csv      long table: url, engine, topic, week, citations

y = log2(1 + citation_count), counting each URL once per answer. Days/engines
listed in excluded_days.csv and unresolved redirect links are removed first
(the number dropped is printed). Page features (Step A6) and outside authority
(Step A6b) are joined when their tables exist. Every numeric feature is
z-scored within each output table; the original value is kept as <name>_raw.
"""
import argparse

import numpy as np
import pandas as pd

from common import TABLES, read_table, drop_excluded

NOT_FEATURES = {'citation_count', 'y', 'n_prompts', 'n_days', 'n_answers', 'mean_position', 'page_date_after_cite',
                'share_first', 'round'}


def citation_rows():
    c = read_table('citations.csv')
    n0 = len(c)
    c = drop_excluded(c)
    n1 = len(c)
    red = c.is_redirect.sum()
    c = c[~c.is_redirect & ~c.repeat_in_answer].copy()
    print(f'citation rows: {n0:,} total, {n0 - n1:,} in excluded days, '
          f'{red:,} unresolved redirects dropped, {len(c):,} used (once per answer)')
    if 'role' not in c:
        raise SystemExit('run `python scripts/brand_domains.py apply` first (adds role)')
    c['day'] = pd.to_datetime(c.date, format='%Y%m%d')
    c['week'] = c.day.dt.strftime('%G-W%V')
    return c


def per_url(c):
    g = c.groupby('url')
    t = pd.DataFrame({
        'domain': g.domain.first(),
        'reg_domain': g.reg_domain.first(),
        'role': g.role.first(),
        'brand': g.brand.first(),
        'citation_count': g.size(),
        'n_prompts': g.prompt_id.nunique(),
        'n_days': g.date.nunique(),
        'first_date': g.day.min(),
        'last_date': g.day.max(),
        'median_cite_date': g.day.median(),
        'mean_position': g.position.mean(),
        'share_first': g.position.apply(lambda s: (s == 1).mean()),
    })
    by_engine = c.pivot_table(index='url', columns='engine', aggfunc='size', fill_value=0)
    t = t.join(by_engine.add_prefix('count_'))
    t['y'] = np.log2(1 + t.citation_count)
    return t


def join_features(t):
    pf_path = TABLES / 'page_features.csv'
    if pf_path.exists():
        pf = pd.read_csv(pf_path).drop_duplicates('url').set_index('url')
        t = t.join(pf, how='left')
        if 'page_date' in t:
            pd_ = pd.to_datetime(t.page_date, errors='coerce')
            age = (t.median_cite_date - pd_).dt.days                   # relative to median citation day
            # a page cannot be published after it was cited: such dates are extraction errors
            # (pages were downloaded later and often show a newer 'updated' date)
            t['page_date_after_cite'] = (age < 0).astype(int).where(age.notna())
            t['page_age_days'] = age.where(age >= 0)
        print(f'page features joined for {t.word_count.notna().sum():,} / {len(t):,} URLs')
        # pages that could not be downloaded (Reddit, YouTube, Quora, blocked sites) still get a
        # page type from the URL alone, so forum / video pages stay in page-type comparisons
        from page_parse import page_type
        missing = t.page_type.isna()
        t['page_type_source'] = np.where(missing, 'url_only', 'page')
        t.loc[missing, 'page_type'] = [page_type(u) for u in t.index[missing]]
    else:
        print('no page_features.csv yet (Step A6) - table has counts only')
    sp_path = TABLES / 'page_speed.csv'
    if sp_path.exists():
        sp = pd.read_csv(sp_path).drop_duplicates('url', keep='last').set_index('url')
        t = t.join(sp[['lcp', 'inp', 'cls', 'perf_score']], how='left')
    au_path = TABLES / 'domain_authority.csv'
    if au_path.exists():
        au = pd.read_csv(au_path).drop_duplicates('domain').set_index('domain')
        t = t.join(au, on='reg_domain', how='left')
        print(f'authority joined for {t.reg_domain.isin(au.index).sum():,} URLs')
    return t


def zscore(t):
    num = [c for c in t.select_dtypes('number').columns
           if c not in NOT_FEATURES and not c.startswith(('count_', '_'))
           and not c.endswith('_found')]
    t = t.copy()
    for c in num:
        s = t[c].astype(float)
        sd = s.std()
        t[c + '_raw'] = s
        t[c] = (s - s.mean()) / sd if sd and sd > 0 else 0.0
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-count', type=int, default=2)
    a = ap.parse_args()

    c = citation_rows()
    (c.groupby(['url', 'engine', 'topic', 'week']).size().rename('citations')
      .reset_index().to_csv(TABLES / 'url_counts.csv', index=False))

    t = join_features(per_url(c))
    t.index.name = 'url'
    zscore(t).reset_index().to_csv(TABLES / 'url_table_all.csv', index=False)
    main_t = t[t.citation_count >= a.min_count]
    zscore(main_t).reset_index().to_csv(TABLES / 'url_table.csv', index=False)
    print(f'url_table_all: {len(t):,} URLs on {t.domain.nunique():,} websites')
    print(f'url_table (>= {a.min_count} citations): {len(main_t):,} URLs on '
          f'{main_t.domain.nunique():,} websites')
    print(main_t.groupby('role').citation_count.agg(['size', 'sum']).to_string())


if __name__ == '__main__':
    main()
