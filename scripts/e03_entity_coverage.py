"""E3 - Do pages that cover more entities get cited for more prompts (breadth)?

Prompts name brands, speed tiers, prices and places. A page that covers many of them could be
retrieved for many different prompts. Entity-coverage features from the page text (title + first
50 paragraphs, as in Step A6):
  n_brands       tracked brands mentioned (prompt_labels.BRANDS, 9 brands)
  n_speed_tiers  distinct speed tiers ("1 Gig", "500 Mbps", ...)
  n_prices       distinct dollar amounts
  n_places       distinct US state names
  n_prompt_terms distinct content words from the 651 prompts that occur on the page
All log2(1 + n) and z-scored. Added to the R2 page variables; outcome = total, breadth, depth
(as in E1), share of the within-website variation explained (mixed model) and the within-website
effect of each feature per SD (fe_ols, website fixed effects).

Outputs: results/e03_entity_coverage.csv, results/e03_entity_effects.csv
"""
import json
import re
import warnings

import numpy as np
import pandas as pd

from common import TABLES, RESULTS, read_table, drop_excluded
from fe import fe_ols
from r02_main_model import prepare, FEATURES
from prompt_labels import BRANDS
from e01_engines_breadth_depth import outcomes, vc, rel_within

warnings.filterwarnings('ignore')
STATES = ['alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware',
          'florida', 'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky',
          'louisiana', 'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi',
          'missouri', 'montana', 'nebraska', 'nevada', 'new hampshire', 'new jersey', 'new mexico',
          'new york', 'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon', 'pennsylvania',
          'rhode island', 'south carolina', 'south dakota', 'tennessee', 'texas', 'utah', 'vermont',
          'virginia', 'washington', 'west virginia', 'wisconsin', 'wyoming']
SPEED = re.compile(r'\b(\d+(?:\.\d+)?)\s?(gig|gbps|mbps|gb/s|mb/s)\b')
PRICE = re.compile(r'\$\s?(\d{1,4}(?:\.\d{2})?)')
STOP = set('the a an and or of to in for on with is are do does can i my me what which how who why when where '
           'it its be vs than that this from at by as if any best better more most get should there their '
           'you your we our they them about have has will would not no'.split())
ENTITY = ['n_brands', 'n_speed_tiers', 'n_prices', 'n_places', 'n_prompt_terms']


def coverage():
    prompts = pd.read_csv(TABLES / 'prompt_labels.csv', usecols=['prompt']).prompt
    vocab = {w for p in prompts for w in re.findall(r'[a-z][a-z0-9&]+', p.lower()) if w not in STOP and len(w) > 2}
    brand_rx = {b: re.compile(rx) for b, rx in BRANDS.items()}
    state_rx = re.compile(r'\b(' + '|'.join(STATES) + r')\b')
    rows = []
    with open(TABLES / 'page_text.jsonl', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            text = ' '.join([j.get('title') or ''] + (j.get('paragraphs') or [])[:50]).lower()
            words = set(re.findall(r'[a-z][a-z0-9&]+', text))
            rows.append(dict(url=j['url'],
                             n_brands=sum(bool(rx.search(text)) for rx in brand_rx.values()),
                             n_speed_tiers=len({(float(a) * (1000 if u.startswith('g') else 1)) for a, u in SPEED.findall(text)}),
                             n_prices=len(set(PRICE.findall(text))),
                             n_places=len(set(state_rx.findall(text))),
                             n_prompt_terms=len(words & vocab)))
    return pd.DataFrame(rows).drop_duplicates('url', keep='last')


def main():
    cov = coverage()
    t = read_table('url_table.csv')
    d, flags, types = prepare(t)
    X = FEATURES + flags + types
    d = d.merge(cov, on='url', how='left')
    for e in ENTITY:
        d[e] = np.log2(1 + d[e].fillna(0))
        d[e] = (d[e] - d[e].mean()) / d[e].std()
    c = drop_excluded(read_table('citations.csv', usecols=['date', 'engine', 'prompt_id', 'url', 'is_redirect',
                                                          'repeat_in_answer']))
    c = c[~c.is_redirect & ~c.repeat_in_answer & c.url.isin(d.url)]
    order = {x: i for i, x in enumerate(sorted(c.date.unique()))}
    c['half'] = c.date.map(order) % 2
    full = outcomes(c)
    d = d[d.url.isin(full.index)]
    d = d[d.groupby('domain').url.transform('size') >= 2].reset_index(drop=True)
    halves = [outcomes(c[c.half == h]).reindex(d.url, fill_value=0).reset_index(drop=True) for h in (0, 1)]
    rows, eff = [], []
    for y in ('total', 'breadth', 'depth'):
        d[y] = full[y].reindex(d.url).to_numpy()
        rel = rel_within(halves[0][y], halves[1][y], d.domain)
        for name, cols in (('page variables', X), ('page variables + entity coverage', X + ENTITY)):
            ws, ex = vc(d, y, cols)
            rows.append(dict(outcome=y, model=name, n_urls=len(d), within_share=ws, explained_within=ex,
                             reliability_within=rel))
        r, _ = fe_ols(d, y, X + ENTITY, 'domain')
        for e in ENTITY:
            eff.append(dict(outcome=y, feature=e, coef=r.loc[e, 'coef'], se=r.loc[e, 'se'], p=r.loc[e, 'p'],
                            pct_change_per_sd=100 * (2 ** r.loc[e, 'coef'] - 1)))
    out, eff = pd.DataFrame(rows), pd.DataFrame(eff)
    out.to_csv(RESULTS / 'e03_entity_coverage.csv', index=False)
    eff.to_csv(RESULTS / 'e03_entity_effects.csv', index=False)
    print(out.round(3).to_string(index=False))
    print(eff.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
