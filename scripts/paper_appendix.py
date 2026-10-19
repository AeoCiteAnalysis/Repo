"""Write the appendix tables of the paper from the results files.

  python scripts/paper_appendix.py      -> paper/appendix_tables.tex

Tables (each a \\newcommand, placed in sections/appendix.tex):
  \\AppTabQuestions   questions per topic and intent, both datasets
  \\AppTabTests       the five tests for the 14 features, all pages, both datasets
  \\AppTabEngines     within-website effects per engine (main dataset)
Rerun after the analyses change (weekly_refresh.py does not call it; the underlying R-analyses are
rerun at the data freeze).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = {'gf': ROOT / 'results', 'wo': ROOT / 'results' / 'workorb'}
TAB = {'gf': ROOT / 'data' / 'tables', 'wo': ROOT / 'data' / 'workorb' / 'tables'}
OUT = ROOT / 'paper' / 'appendix_tables.tex'

FEATURES = [('title_match', 'Title match'), ('best_paragraph_match', 'Best-paragraph match'),
            ('prompt_match_words', 'Word overlap'), ('intro_match', 'First-paragraph match'),
            ('word_count', 'Length (words)'), ('outbound_links', 'Outbound links'), ('n_headings', 'Headings'),
            ('n_tables', 'Tables'), ('n_lists', 'Lists'), ('has_faq', 'FAQ block$^\\dagger$'),
            ('has_summary_box', 'Summary box$^\\dagger$'), ('has_author', 'Author$^\\dagger$'),
            ('has_schema', 'Schema.org markup$^\\dagger$'), ('page_age_days', 'Page age')]
TESTS = ['1 significance (R3)', '2 repeat selection (R4)', '3 double ML (R5)', '4 shape (R6)', '5 robustness (R9)']


def num(x, d=1):
    if pd.isna(x):
        return '--'
    s = f'{x:+.{d}f}'
    return s.replace('-', '\\textminus{}')


def questions():
    rows, cols = [], []
    for k, name in (('gf', 'Consumer'), ('wo', 'Niche')):
        p = pd.read_csv(TAB[k] / 'prompt_labels.csv')
        rows.append((name, p.topic.value_counts(), p.intent.value_counts(), len(p)))
    lines = ['\\begin{tabular}{@{}lr@{\\hspace{1.2em}}lr@{}}', '\\toprule',
             '\\multicolumn{2}{@{}l}{Consumer (651)} & \\multicolumn{2}{l@{}}{Niche (55)} \\\\', '\\midrule']
    ta, tb = list(rows[0][1].items()), list(rows[1][1].items())
    k = max(len(ta), len(tb))
    ta += [('', '')] * (k - len(ta))                      # topics of both markets end on the same row
    tb += [('', '')] * (k - len(tb))
    a = ta + [None] + [(f'\\emph{{{i}}}', n) for i, n in rows[0][2].items()]
    b = tb + [None] + [(f'\\emph{{{i}}}', n) for i, n in rows[1][2].items()]
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else ('', '')
        y = b[i] if i < len(b) else ('', '')
        if x is None:
            lines.append('\\midrule')
            continue
        lines.append(f'{x[0]} & {x[1]} & {y[0]} & {y[1]} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines)


def tests():
    d = {k: pd.read_csv(RES[k] / 'is_it_real.csv') for k in RES}
    d = {k: v[v['sample'] == 'all pages'].set_index('term') for k, v in d.items()}
    lines = ['\\begin{tabular}{@{}lrccccc@{\\hspace{0.8em}}rc@{}}', '\\toprule',
             ' & \\multicolumn{6}{c}{Consumer market} & \\multicolumn{2}{c@{}}{Niche market} \\\\',
             '\\cmidrule(lr){2-7}\\cmidrule(l){8-9}',
             'Feature & Effect & 1 & 2 & 3 & 4 & 5 & Effect & Tests \\\\', '\\midrule']
    for term, label in FEATURES:
        g, w = d['gf'].loc[term], d['wo'].loc[term] if term in d['wo'].index else None
        marks = ' & '.join('$\\bullet$' if bool(g[t]) else '$\\circ$' for t in TESTS)
        lab = f'\\textbf{{{label}}}' if bool(g['real']) else label
        wo_eff = num(w['pct_change']) if w is not None else '--'
        wo_t = f'{int(w["tests_passed"])}' if w is not None else '--'
        if w is not None and bool(w['real']):
            wo_t = f'\\textbf{{{wo_t}}}'
        lines.append(f'{lab} & {num(g["pct_change"])} & {marks} & {wo_eff} & {wo_t} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines)


def engines():
    r = pd.read_csv(RES['gf'] / 'r09_robustness.csv')
    r = r[r.check.str.startswith('engine:') & (r['sample'] == 'all pages')]
    lines = ['\\begin{tabular}{@{}lrrr@{}}', '\\toprule', 'Feature & ChatGPT & Gemini & Perplexity \\\\', '\\midrule']
    for term, label in FEATURES[:4] + [FEATURES[9], FEATURES[12], FEATURES[13]]:
        cells = []
        for e in ('ChatGPT', 'Gemini', 'Perplexity'):
            x = r[(r.term == term) & (r.check == f'engine: {e}')]
            if x.empty:
                cells.append('--')
                continue
            x = x.iloc[0]
            star = '$^{*}$' if x.p < 0.05 else ''
            cells.append(num(x['pct_change']) + star)
        lines.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(lines)


MAIN_ROWS = [('best_paragraph_match', 'Best paragraph matches'), ('title_match', 'Title matches a query'),
             ('prompt_match_words', 'Word overlap with queries'), ('intro_match', 'First paragraph matches'),
             ('page_age_days', 'Page age'), ('word_count', 'Word count'), ('outbound_links', 'Outbound links'),
             ('n_headings', 'Headings'), ('n_tables', 'Tables'), ('n_lists', 'Lists'), ('has_faq', 'Has FAQ block'),
             ('has_summary_box', 'Has summary box'), ('has_author', 'Has author'),
             ('has_schema', 'Has schema.org markup')]


def main_effects():
    """Rows of Table 2 (section 4): effect (tests passed) per sample, bold if real; main dataset."""
    d = pd.read_csv(RES['gf'] / 'is_it_real.csv').set_index(['sample', 'term'])
    m = pd.read_csv(RES['gf'] / 'r02_main_model.csv')
    samples = ['all pages', 'own + competitor', 'third party']
    lines = []
    for term, label in MAIN_ROWS:
        cells = []
        for smp in samples:
            r = d.loc[(smp, term)]
            c = f'{num(r["pct_change"])} ({int(r["tests_passed"])})'
            cells.append(f'\\textbf{{{c}}}' if bool(r['real']) else c)
        lines.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
    lines.append('\\midrule')
    n = [m[m['sample'] == smp].iloc[0] for smp in samples]
    lines.append('Pages / websites & ' + ' & '.join(f'{int(x.n_urls):,} / {int(x.n_websites):,}' for x in n) + ' \\\\')
    return '\n'.join(lines)


def main():
    parts = [f'% Generated by scripts/paper_appendix.py - do not edit by hand.',
             '\\newcommand{\\AppTabQuestions}{%\n' + questions() + '}',
             '\\newcommand{\\AppTabTests}{%\n' + tests() + '}',
             '\\newcommand{\\AppTabEngines}{%\n' + engines() + '}',
             '\\newcommand{\\MainTabEffectsRows}{%\n' + main_effects() + '}']
    OUT.write_text('\n'.join(parts) + '\n', encoding='utf-8')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
