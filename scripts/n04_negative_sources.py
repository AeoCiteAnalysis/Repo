"""N4 - Which sources sit behind negative mentions?

Question. When an engine says something negative about a brand, which websites does it cite next
to that sentence?

For each brand mention: the links inside its mention text; if there are none, the links in the
paragraph of the answer that contains the mention (same link pattern as Step A2). Mentions with no
nearby link are counted but not attributed. Per website: number of positive / neutral / negative
mentions it is attached to; websites ranked by share of negative mentions (>= 20 mentions);
broken down by brand and engine. Unresolved Gemini redirect links are dropped.

Outputs: results/n04_link_coverage.csv, results/n04_websites_by_sentiment.csv,
         results/n04_by_brand.csv
"""
import numpy as np
import pandas as pd

from common import RESULTS, ENGINE_LABEL, LINK, clean_url, domain_of, is_redirect, read_table, drop_excluded

MIN_MENTIONS = 20


def paragraph_links(text, snippet):
    if not isinstance(text, str) or not isinstance(snippet, str):
        return []
    key = snippet.strip()[:60]
    for para in text.split('\n'):
        if key and key in para:
            return LINK.findall(para)
    return []


def main():
    m = drop_excluded(read_table('mentions.csv', usecols=['answer_id', 'date', 'engine', 'brand', 'sentiment',
                                                         'mention_text']))
    m['links'] = m.mention_text.fillna('').map(LINK.findall)
    m['link_source'] = np.where(m.links.str.len() > 0, 'mention text', '')
    need = m[m.links.str.len() == 0]
    ans = read_table('answers.csv', usecols=['answer_id', 'answer_text'])
    ans = ans[ans.answer_id.isin(need.answer_id)].set_index('answer_id').answer_text
    fill = [paragraph_links(ans.get(a), s) for a, s in zip(need.answer_id, need.mention_text)]
    m.loc[need.index, 'links'] = pd.Series(fill, index=need.index)
    m.loc[need.index, 'link_source'] = np.where(pd.Series(fill, index=need.index).str.len() > 0, 'same paragraph', 'none')
    cov = m.groupby(['engine', 'link_source']).size().unstack(fill_value=0)
    cov = cov.div(cov.sum(axis=1), axis=0).mul(100).round(1)
    cov.rename(index=ENGINE_LABEL).to_csv(RESULTS / 'n04_link_coverage.csv')

    x = m[m.links.str.len() > 0].explode('links')
    x['url'] = x.links.map(clean_url)
    x = x[~x.url.map(is_redirect)]
    x['website'] = x.url.map(domain_of)
    x = x.drop_duplicates(['answer_id', 'brand', 'website'])     # one link per mention and site
    w = x.pivot_table(index='website', columns='sentiment', aggfunc='size', fill_value=0)
    for s in ('POSITIVE', 'NEUTRAL', 'NEGATIVE'):
        if s not in w:
            w[s] = 0
    w['mentions'] = w[['POSITIVE', 'NEUTRAL', 'NEGATIVE']].sum(axis=1)
    w['pct_negative'] = 100 * w.NEGATIVE / w.mentions
    w['pct_positive'] = 100 * w.POSITIVE / w.mentions
    w = w[w.mentions >= MIN_MENTIONS].sort_values('pct_negative', ascending=False)
    w.rename_axis('website').reset_index().to_csv(RESULTS / 'n04_websites_by_sentiment.csv', index=False)

    b = x.pivot_table(index=['brand', 'engine', 'website'], columns='sentiment', aggfunc='size', fill_value=0)
    for s in ('POSITIVE', 'NEUTRAL', 'NEGATIVE'):
        if s not in b:
            b[s] = 0
    b['mentions'] = b[['POSITIVE', 'NEUTRAL', 'NEGATIVE']].sum(axis=1)
    b = b[b.mentions >= MIN_MENTIONS].copy()
    b['pct_negative'] = 100 * b.NEGATIVE / b.mentions
    b = b.reset_index()
    b['engine'] = b.engine.map(ENGINE_LABEL)
    b.sort_values(['brand', 'pct_negative'], ascending=[True, False]).to_csv(RESULTS / 'n04_by_brand.csv', index=False)

    print('where the links come from (% of mentions):'); print(cov.rename(index=ENGINE_LABEL).to_string())
    print(f'\nwebsites with >= {MIN_MENTIONS} attached mentions: {len(w)}; overall negative share '
          f'{100 * x.sentiment.eq("NEGATIVE").mean():.1f}%')
    print('most negative:'); print(w.head(12)[['mentions', 'pct_negative', 'pct_positive']].round(1).to_string())
    print('most positive:'); print(w.sort_values('pct_positive', ascending=False).head(8)[['mentions', 'pct_negative', 'pct_positive']].round(1).to_string())
    gf = b[b.brand == 'Google Fiber'].sort_values('pct_negative', ascending=False).head(8)
    print('\nGoogle Fiber - most negative sources:'); print(gf[['engine', 'website', 'mentions', 'pct_negative']].round(1).to_string(index=False))


if __name__ == '__main__':
    main()
