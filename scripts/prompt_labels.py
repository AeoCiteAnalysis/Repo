"""Label every prompt with an intent (used by R12, N5, N14).

  learning      general questions ("why is fiber more reliable than DSL")
  comparing     brand A vs B, "best ...", "which provider ..."
  ready to buy  prices, fees, deals, sign-up, contracts, switching

Two labelers:
  rules  keyword rules below (checked in order: ready to buy, then comparing, else learning)
  llm    OpenAI model (llm.py), used when OPENAI_API_KEY is set in .env; answers are cached
The `intent` column uses the LLM label when available, else the rule label.
results/intent_check_sample.csv: 100 random prompts with both labels and an empty
`correct_intent` column to fill by hand (the guide's 100 hand labels); report agreement of each
labeler with the hand labels.

Also writes simple prompt features for N5: word count, tracked brand named, comparison,
price question, yes/no question.

Output: data/tables/prompt_labels.csv, results/intent_check_sample.csv
"""
import re

import pandas as pd

from common import DATASET, INDUSTRY, TABLES, RESULTS, SEED, read_table

BUY = re.compile(r'\b(price[sd]?|pricing|cost[s]?|cheap(est|er)?|afford\w*|expensive|how much|deals|best deal|promo\w*|discount\w*|'
                 r'fees?|pay(ing)?|\$|per month|/mo|monthly|sign[- ]?up|order|contracts?|'
                 r'switch(ing)?|cancel\w*|install(ation)? (fee|cost)|bill(ing|s)?|refund\w*|'
                 r'money|budget|value|worth it|save)\b', re.I)
COMPARE = re.compile(r'\b(vs\.?|versus|compare[sd]?|comparison|better|best|worst|which|top|most|'
                     r'fastest|slowest|highest|lowest|rank\w*|than|compared)\b', re.I)
BRANDS_BY_DATASET = {}
BRANDS_BY_DATASET['workorb'] = {'Workorb': r'workorb', 'Loopio': r'loopio', 'Unanet': r'unanet|cosential',
                                'OpenAsset': r'open ?asset', 'Joist AI': r'joist ?\.?ai|kantiv'}
BRANDS_BY_DATASET['gfiber'] = {'Google Fiber': r'g ?fiber|google fiber', 'AT&T Fiber': r'at&t|\batt\b',
          'Xfinity': r'xfinity|comcast', 'Quantum Fiber': r'quantum', 'Spectrum': r'spectrum',
          'T-Mobile': r't-mobile|tmobile', 'Metronet': r'metronet', 'Astound': r'astound',
          'Verizon': r'verizon|fios|frontier'}
BRANDS = BRANDS_BY_DATASET[DATASET]


def intent(p):
    if BUY.search(p):
        return 'ready to buy'
    if COMPARE.search(p):
        return 'comparing'
    return 'learning'


LLM_INSTRUCTIONS = f"""You label questions that people ask an AI assistant about {INDUSTRY}.
Give the user's intent, one of exactly three labels:
learning - general or factual questions, how things work, what a provider does or offers
comparing - comparing brands or options, asking which provider or product is best or better
ready to buy - prices, fees, deals, discounts, contracts, signing up, switching, cancelling, bills
Answer with the label only."""
LABELS = ['learning', 'comparing', 'ready to buy']


def llm_intents(prompts):
    import llm
    if not llm.available():
        print('no OPENAI_API_KEY - using keyword rules only')
        return [None] * len(prompts)
    out = []
    for ans in llm.ask_many(LLM_INSTRUCTIONS, list(prompts)):
        a = ans.strip().lower().strip('." \'')
        out.append(next((l for l in LABELS if a.startswith(l)), None))
    print(f'LLM ({llm.model_name()}) labelled {sum(x is not None for x in out)} / {len(out)} prompts')
    return out


def main():
    a = read_table('answers.csv', usecols=['prompt_id', 'prompt', 'topic']).drop_duplicates('prompt_id')
    a['intent_rule'] = a.prompt.map(intent)
    a['intent_llm'] = llm_intents(a.prompt)
    a['intent'] = a.intent_llm.fillna(a.intent_rule)
    a['intent_source'] = a.intent_llm.notna().map({True: 'llm', False: 'rules'})
    low = a.prompt.str.lower()
    a['n_words'] = a.prompt.str.split().str.len()
    for b, rx in BRANDS.items():
        a['names_' + re.sub(r'\W+', '_', b.lower()).strip('_')] = low.str.contains(rx, regex=True)
    name_cols = [c for c in a if c.startswith('names_')]
    a['n_brands_named'] = a[name_cols].sum(axis=1)
    a['is_comparison'] = low.str.contains(r'\bvs\.?\b|versus|compare|better than|\bor\b', regex=True)
    a['asks_price'] = a.prompt.map(lambda p: bool(BUY.search(p)))
    a['is_yes_no'] = low.str.match(r'^(is|are|do|does|can|should|will|would|has|have)\b')
    a.to_csv(TABLES / 'prompt_labels.csv', index=False)
    check = RESULTS / 'intent_check_sample.csv'
    sample = a.sample(min(100, len(a)), random_state=SEED)[['prompt_id', 'topic', 'prompt', 'intent_rule', 'intent_llm']]
    if check.exists():                          # keep hand labels already entered
        old = pd.read_csv(check, encoding_errors='replace')
        keep = [c for c in ('correct_intent', 'notes') if c in old]
        sample = sample.merge(old[['prompt_id'] + keep], on='prompt_id', how='left')
    for c in ('correct_intent', 'notes'):
        if c not in sample:
            sample[c] = ''
    sample.to_csv(check, index=False)
    hand = sample[sample.correct_intent.fillna('').astype(str).str.strip() != '']
    if len(hand):
        for col in ('intent_rule', 'intent_llm'):
            agree = (hand[col] == hand.correct_intent.str.strip().str.lower()).mean()
            print(f'agreement with {len(hand)} hand labels: {col} {100 * agree:.0f}%')
    print(a.intent.value_counts().to_string())
    print(a.intent_source.value_counts().to_string())
    print(pd.crosstab(a.topic, a.intent).to_string())
    print('prompts naming a tracked brand:', int((a.n_brands_named > 0).sum()))


if __name__ == '__main__':
    main()
