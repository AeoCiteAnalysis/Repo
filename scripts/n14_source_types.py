"""N14 - What kinds of websites do engines trust?

Question. Which types of sources does each engine prefer, and for which topics and intents?

Website types (host level):
  brand site           own / competitor websites (from brand_domains.csv - hand reviewed)
  other ISP            internet providers we do not track (Ziply, Optimum, Allo, ...)
  review / comparison  review, rating and ISP-comparison sites
  forum / community    Reddit, Quora, provider community forums, discussion boards
  video                YouTube and other video platforms
  news / media         newspapers, tech and trade press
  government           government and regulators (FCC, states, cities)
  encyclopedia / reference   Wikipedia, dictionaries, standards / reference sites
  equipment maker      router / network hardware makers and retailers
  other
The 300 most-cited websites are labelled by the LLM (llm.py) from the host name and up to three
titles of its cited pages (the guide: label the top 300 with LLM help, then check); brand sites
keep their hand-reviewed label; the rest get simple rules (.gov -> government, video hosts,
forum page types, Wikipedia) or "other". results/n14_type_check_sample.csv: 50 LLM labels to check
by hand. Shares of citations per engine, topic, intent (LLM label) and week.

Outputs: data/tables/website_types.csv, results/n14_shares_*.csv, figures/n14_*.png
"""
import numpy as np
import pandas as pd

import plotstyle
from plotstyle import plt, INK2
from common import TABLES, RESULTS, FIGURES, ENGINES, ENGINE_LABEL, SEED, read_table, drop_excluded

TYPES = ['brand site', 'other ISP', 'review / comparison', 'forum / community', 'video', 'news / media',
         'government', 'encyclopedia / reference', 'equipment maker', 'other']
COLORS = ['#2a78d6', '#9ec5f4', '#eb6834', '#1baf7a', '#e34948', '#eda100', '#4a3aa7', '#e87ba4',
          '#008300', '#c3c2b7']
TOP_N = 300
INSTRUCTIONS = ('You classify websites that AI assistants cite when answering questions about home '
                'internet providers. Choose exactly one type:\n' + '\n'.join(TYPES[1:]) +
                '\n"other ISP" = any internet / telecom provider\'s own website. "review / comparison" = '
                'sites that review, rate or compare providers or plans. "forum / community" = discussion '
                'boards, Q&A sites, provider community forums. Answer with the type only.')


def rule_type(host, page_type):
    if host.endswith(('.gov', '.mil')) or '.gov.' in host:
        return 'government'
    if host.endswith(('youtube.com', 'youtu.be', 'vimeo.com', 'tiktok.com')):
        return 'video'
    if host.endswith('wikipedia.org'):
        return 'encyclopedia / reference'
    if page_type == 'forum':
        return 'forum / community'
    return 'other'


def label_websites(c):
    import llm
    counts = c.groupby('domain').size().sort_values(ascending=False)
    roles = c.drop_duplicates('domain').set_index('domain').role
    ua = read_table('url_table_all.csv', usecols=['domain', 'page_type', 'title'])
    ptype = ua.groupby('domain').page_type.agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
    titles = ua.dropna(subset=['title']).groupby('domain').title.agg(lambda s: list(s.head(3)))
    w = pd.DataFrame({'citations': counts, 'role': roles.reindex(counts.index)})
    w['rank'] = np.arange(1, len(w) + 1)
    w['type'], w['source'], w['llm_answer'] = None, None, None
    brand = w.role.isin(['own', 'competitor'])
    w.loc[brand, ['type', 'source']] = ['brand site', 'brand_domains.csv']
    top = w[(w['rank'] <= TOP_N) & ~brand].index
    if llm.available():
        texts = [f'Website: {h}\nExample page titles: ' + '; '.join(titles.get(h, [])[:3]) for h in top]
        answers = llm.ask_many(INSTRUCTIONS, texts)
        for h, ans in zip(top, answers):
            a = ans.strip().strip('."\'').lower()
            t = next((x for x in TYPES if a.startswith(x.lower())), None)
            w.loc[h, ['type', 'source', 'llm_answer']] = [t or 'other', 'llm' if t else 'llm (unparsed)', ans]
        print(f'LLM ({llm.model_name()}) labelled {len(top)} websites')
    rest = w.type.isna()
    w.loc[rest, 'type'] = [rule_type(h, ptype.get(h)) for h in w.index[rest]]
    w.loc[rest, 'source'] = 'rules'
    w = w.rename_axis('domain').reset_index()
    w.to_csv(TABLES / 'website_types.csv', index=False)
    w[w.source == 'llm'].sample(50, random_state=SEED)[['domain', 'citations', 'type']].assign(
        correct_type='', notes='').to_csv(RESULTS / 'n14_type_check_sample.csv', index=False)
    return w


def shares(c, by):
    t = c.groupby(by + ['type']).size().unstack(fill_value=0).reindex(columns=TYPES, fill_value=0)
    return 100 * t.div(t.sum(axis=1), axis=0)


def stacked(ax, tab, labels, title):
    left = np.zeros(len(tab))
    for t, col in zip(TYPES, COLORS):
        v = tab[t].to_numpy()
        ax.barh(labels, v, left=left, color=col, height=0.7, label=t, edgecolor='#fcfcfb', linewidth=1)
        for i, (l, x) in enumerate(zip(left, v)):
            if x >= 6:                     # dark text on the light segments, white on the dark ones
                light = col in ('#9ec5f4', '#eda100', '#c3c2b7', '#e87ba4', '#1baf7a')
                ax.text(l + x / 2, i, f'{x:.0f}', ha='center', va='center', fontsize=7,
                        color='#0b0b0b' if light else '#fcfcfb')
        left += v
    ax.set_xlim(0, 100); ax.set_xlabel('% of citations'); ax.set_title(title, fontsize=9.5)
    ax.grid(axis='y', visible=False)


def main():
    c = drop_excluded(read_table('citations.csv', usecols=['date', 'engine', 'topic', 'prompt_id', 'domain',
                                                          'role', 'repeat_in_answer']))
    c = c[~c.repeat_in_answer & c.domain.notna()]
    w = label_websites(c)
    c = c.merge(w[['domain', 'type']], on='domain', how='left')
    c['type'] = c.type.fillna('other')
    lab = pd.read_csv(TABLES / 'prompt_labels.csv', usecols=['prompt_id', 'intent'])
    c = c.merge(lab, on='prompt_id', how='left')
    c['week'] = pd.to_datetime(c.date, format='%Y%m%d').dt.to_period('W').dt.start_time
    se = shares(c, ['engine']); se.index = se.index.map(ENGINE_LABEL)
    st = shares(c, ['engine', 'topic'])
    si = shares(c, ['engine', 'intent'])
    sw = shares(c, ['engine', 'week'])
    for name, tab in (('engine', se), ('engine_topic', st), ('engine_intent', si), ('engine_week', sw)):
        tab.round(2).to_csv(RESULTS / f'n14_shares_{name}.csv')

    fig, ax = plt.subplots(figsize=(10, 2.8))
    stacked(ax, se, list(se.index), 'N14: what kinds of websites each engine cites')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5, fontsize=7.5)
    fig.savefig(FIGURES / 'n14_types_by_engine.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, e in zip(axes, ['OPENAI', 'GEMINI']):
        tab = st.loc[e]
        stacked(ax, tab, list(tab.index), f'{ENGINE_LABEL[e]} by topic')
    axes[1].legend(loc='upper center', bbox_to_anchor=(-0.1, -0.15), ncol=5, fontsize=7.5)
    fig.savefig(FIGURES / 'n14_types_by_topic.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
    for ax, e in zip(axes, ['OPENAI', 'GEMINI']):
        tab = sw.loc[e]
        for t, col in zip(TYPES, COLORS):
            if tab[t].max() >= 4:
                ax.plot(tab.index, tab[t], color=col, label=t)
        ax.set_title(f'{ENGINE_LABEL[e]}: weekly share by website type', fontsize=9.5)
        ax.set_ylabel('% of citations')
    axes[1].legend(loc='upper left', fontsize=7, bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); fig.savefig(FIGURES / 'n14_types_weekly.png'); plt.close(fig)

    print(w.groupby(['source', 'type']).agg(websites=('domain', 'size'), citations=('citations', 'sum')).to_string())
    print('\ncoverage: %.1f%% of citations are on websites with an LLM or brand label' %
          (100 * w[w.source != 'rules'].citations.sum() / w.citations.sum()))
    print(se.round(1).to_string())
    print(si.round(1).to_string())


if __name__ == '__main__':
    main()
