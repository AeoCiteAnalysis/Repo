"""Suggest website roles for the second dataset with the LLM (Step A4 helper).

The GFiber market has ~10 brands; the Workorb market (AEC proposal software) has dozens of vendor
websites among the cited sources, too many to label by hand first. For every host with >= MIN
citations that brand_domains.csv does not list yet, the LLM (llm.py, cached) is given the host and
up to three cited paths and asked for: role (own / competitor / third_party), the brand name and a
confidence (sure / likely / unsure). Definitions sent to the model:
  own          the client (Workorb)
  competitor   a company selling software in the client's category: proposal or RFP-response
               software, proposal automation, AEC business-development / CRM / ERP software, or
               digital-asset management for AEC firms (the tracked competitors are Loopio, Unanet,
               OpenAsset, Joist AI)
  third_party  everything else: review/comparison sites, media, blogs, forums, consultants,
               universities, and software companies outside the category (file storage, generic
               project management, automation, design software)
Rows are appended to brand_domains.csv with note 'llm suggestion'; review rows marked unsure
(results/brand_suggest_review.csv), then run `brand_domains.py apply`.

  AEO_DATASET=workorb python scripts/brand_suggest.py [--min 10]
"""
import argparse
import json

import pandas as pd

import llm
from common import CLIENT, INDUSTRY, TABLES, RESULTS, read_table
from brand_domains import load_brand_domains, label_hosts

INSTRUCTIONS = f"""You classify websites cited by AI assistants answering questions about {INDUSTRY}.
The client is {CLIENT}. Roles:
own - the client's own website
competitor - a company selling software in the client's category: proposal or RFP-response software,
proposal automation, AEC business-development, CRM or ERP software, or digital asset management for
AEC firms (example competitors: Loopio, Unanet, OpenAsset, Joist AI)
third_party - anything else: review or comparison sites, media, blogs, forums, consultants,
universities, associations, and software companies outside that category (file storage, generic
project management, automation, design software)
Classify the COMPANY THAT OWNS THE WEBSITE, not the cited page: a software vendor's blog, guide or
comparison article is still that vendor's own site. First say in a few words what the owner sells.
Reply with JSON only: {{"sells": "...", "role": "...", "brand": "<company name or empty>", "confidence": "sure|likely|unsure"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min', type=int, default=10)
    a = ap.parse_args()
    c = read_table('citations.csv', usecols=['domain', 'url', 'is_redirect', 'repeat_in_answer'])
    c = c[~c.repeat_in_answer & c.domain.notna()]
    n = c.groupby('domain').size()
    bd = load_brand_domains()
    lab = label_hosts(n.index, bd)
    todo = [d for d, conf in zip(lab.domain, lab.label_confidence) if conf == 'default' and n[d] >= a.min]
    paths = c[~c.is_redirect].groupby('domain').url.apply(lambda s: list(s.value_counts().index[:3]))
    texts = [f'website: {d}\nexample cited pages: ' + '; '.join(paths.get(d, [])) for d in todo]
    print(f'{len(todo)} websites to classify')
    rows = []
    for d, ans in zip(todo, llm.ask_many(INSTRUCTIONS, texts)):
        try:
            j = json.loads(ans[ans.index('{'):ans.rindex('}') + 1])
        except ValueError:
            j = dict(role='third_party', brand='', confidence='unsure')
        role = j.get('role') if j.get('role') in ('own', 'competitor', 'third_party') else 'third_party'
        rows.append(dict(brand=j.get('brand') or '' if role != 'third_party' else '', domain=d, role=role,
                         confidence=j.get('confidence', 'unsure'), note='llm: ' + str(j.get('sells', ''))[:80],
                         citations=int(n[d])))
    new = pd.DataFrame(rows)
    keep = new[new.role != 'third_party'] if len(new) else new
    out = pd.concat([bd, keep.drop(columns='citations')], ignore_index=True)
    out.to_csv(TABLES / 'brand_domains.csv', index=False)
    new.sort_values('citations', ascending=False).to_csv(RESULTS / 'brand_suggest_review.csv', index=False)
    print(new.groupby(['role', 'confidence']).size().to_string())
    print(new[new.role != 'third_party'].sort_values('citations', ascending=False).head(60).to_string(index=False))


if __name__ == '__main__':
    main()
