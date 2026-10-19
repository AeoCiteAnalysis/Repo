"""Step A3 - data health report per day and engine.

  results/health_by_day_engine.csv   one row per (date, engine)
  results/health_flags.csv           (date, engine, measure) cells that look unusual
  figures/health_<measure>.png       one line chart per measure over time

  results/health_level_shifts.csv    lasting level changes (collection or model changes)

Flags are suggestions only. To drop a day or engine, add it by hand to
data/tables/excluded_days.csv (date,engine,reason; engine '*' = all engines)
and write the reason in decisions.md.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from common import TABLES, RESULTS, FIGURES, ENGINES, ENGINE_LABEL, read_table

MEASURES = ['n_answers', 'n_prompts', 'pct_no_citations', 'avg_citations',
            'pct_no_mentions', 'n_unresolved_redirects', 'avg_answer_chars']


def health_table(answers, citations):
    g = answers.groupby(['date', 'engine'])
    h = pd.DataFrame({
        'n_answers': g.size(),
        'n_prompts': g.prompt_id.nunique(),
        'pct_no_citations': g.n_citations.apply(lambda s: 100 * (s == 0).mean()),
        'avg_citations': g.n_citations.mean(),
        'pct_citations_from_text': g.citation_source.apply(lambda s: 100 * (s == 'answer_text').mean()),
        'pct_no_mentions': g.n_mentions.apply(lambda s: 100 * (s == 0).mean()),
        'avg_answer_chars': g.answer_chars.mean(),
        'pct_empty_answers': g.answer_chars.apply(lambda s: 100 * (s == 0).mean()),
    })
    red = citations[citations.is_redirect].groupby(['date', 'engine']).size()
    h['n_unresolved_redirects'] = red.reindex(h.index).fillna(0).astype(int)
    return h.reset_index().sort_values(['date', 'engine'])


def flag_unusual(h, z=4.0, window=7):
    """Flag days far from the engine's own centred 7-day median (robust z > 4).

    Comparing with neighbouring days (not the whole-period median) flags single
    broken days but not lasting level changes, which are listed separately by
    level_shifts() and should be handled as periods, not dropped.
    """
    rows = []
    for m in MEASURES:
        for eng, s in h.sort_values('date').groupby('engine')[m]:
            med = s.rolling(window, center=True, min_periods=3).median()
            dev = s - med
            mad = dev.abs().median() * 1.4826
            bad = s[(dev.abs() > z * mad) & (dev.abs() > 0.1 * med.abs())] if mad > 0 \
                else s[dev != 0]
            for i, v in bad.items():
                rows.append(dict(date=h.at[i, 'date'], engine=eng, measure=m,
                                 value=round(v, 2), local_median=round(med[i], 2)))
    return pd.DataFrame(rows, columns=['date', 'engine', 'measure', 'value', 'local_median'])


def level_shifts(h, min_change=0.25, window=7):
    """Days where the 7-day median after differs from the 7-day median before by >25%."""
    rows = []
    for m in MEASURES:
        for eng, s in h.sort_values('date').groupby('engine')[m]:
            s = s.reset_index(drop=True)
            dates = h[h.engine == eng].sort_values('date').date.reset_index(drop=True)
            before = s.rolling(window).median()
            after = s[::-1].rolling(window).median()[::-1]
            for i in range(window, len(s) - window + 1):
                b, a = before[i - 1], after[i]
                if b and abs(a - b) / abs(b) > min_change:
                    rows.append(dict(date=dates[i], engine=eng, measure=m,
                                     median_before=round(b, 2), median_after=round(a, 2)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # keep only the day with the largest jump in each run of consecutive flagged days
    df['jump'] = (df.median_after - df.median_before).abs()
    df['run'] = (df.groupby(['engine', 'measure']).date
                 .transform(lambda d: (pd.to_datetime(d).diff().dt.days != 1).cumsum()))
    top = df.loc[df.groupby(['engine', 'measure', 'run']).jump.idxmax()].copy()
    # the largest jump can land a few days late; move back to the first day already at the new level
    for i, r in top.iterrows():
        s = h[h.engine == r.engine].set_index('date')[r.measure].sort_index()
        pos = s.index.get_loc(r.date)
        closer_to_after = lambda v: abs(v - r.median_after) < abs(v - r.median_before)
        while pos > 0 and closer_to_after(s.iloc[pos - 1]):
            pos -= 1
        top.at[i, 'date'] = s.index[pos]
    return (top.drop(columns=['jump', 'run']).drop_duplicates(['engine', 'measure', 'date'])
            .sort_values(['engine', 'date']))


def plot(h):
    FIGURES.mkdir(exist_ok=True)
    h = h.assign(day=pd.to_datetime(h.date, format='%Y%m%d'))
    for m in MEASURES + ['pct_citations_from_text']:
        fig, ax = plt.subplots(figsize=(9, 3.2))
        for eng in ENGINES:
            s = h[h.engine == eng]
            ax.plot(s.day, s[m], marker='.', lw=1, label=ENGINE_LABEL[eng])
        ax.set_title(m)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(FIGURES / f'health_{m}.png', dpi=120)
        plt.close(fig)


def main():
    answers = read_table('answers.csv', all_days=True, usecols=lambda c: c != 'answer_text')
    citations = read_table('citations.csv', all_days=True, usecols=['date', 'engine', 'is_redirect'])
    h = health_table(answers, citations)
    RESULTS.mkdir(exist_ok=True)
    h.to_csv(RESULTS / 'health_by_day_engine.csv', index=False)
    flags = flag_unusual(h)
    flags.to_csv(RESULTS / 'health_flags.csv', index=False)
    shifts = level_shifts(h)
    shifts.to_csv(RESULTS / 'health_level_shifts.csv', index=False)
    plot(h)
    print(h.groupby('engine')[MEASURES].median().round(1).T.to_string())
    print(f'\n{len(flags)} unusual days flagged -> results/health_flags.csv')
    if not flags.empty:
        print(flags.to_string(index=False, max_rows=40))
    print(f'\n{len(shifts)} lasting level changes -> results/health_level_shifts.csv')
    if not shifts.empty:
        print(shifts.to_string(index=False, max_rows=40))


if __name__ == '__main__':
    main()
