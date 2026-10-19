"""Shared paths and helpers used by every script.

Run scripts from the project root, e.g. `python scripts/load_data.py`.
"""
import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

try:                                        # API keys from ROOT/.env (see .env.example)
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
except ImportError:
    pass
import os

# Dataset switch: AEO_DATASET=gfiber (default, original paths) or workorb (second dataset).
# Every script reads/writes through these paths, so the two datasets never mix.
DATASET = os.environ.get('AEO_DATASET', 'gfiber')
DATASETS = {
    'gfiber': dict(raw=ROOT / 'pfetchdata' / 'pfetch-dumps', sub=None,
                   client='Google Fiber', industry='home internet providers'),
    'workorb': dict(raw=ROOT / 'pfetch-dumps-workorb', sub='workorb',
                    client='Workorb', industry='proposal and marketing software for architecture, engineering '
                                               'and construction (AEC) firms'),
}
if DATASET not in DATASETS:
    raise SystemExit(f'unknown AEO_DATASET={DATASET}; use one of {list(DATASETS)}')
_cfg = DATASETS[DATASET]
CLIENT = _cfg['client']
INDUSTRY = _cfg['industry']
RAW_DIR = _cfg['raw']                               # daily pfetch-<org>-YYYYMMDD.json (never edit)
_sub = (lambda p: p / _cfg['sub']) if _cfg['sub'] else (lambda p: p)
TABLES = _sub(ROOT / 'data') / 'tables' if _cfg['sub'] else ROOT / 'data' / 'tables'
PAGES = _sub(ROOT / 'data') / 'pages' if _cfg['sub'] else ROOT / 'data' / 'pages'
RESULTS = _sub(ROOT / 'results')
FIGURES = _sub(ROOT / 'figures')
for _d in (TABLES, PAGES, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

ENGINES = ['OPENAI', 'GEMINI', 'PERPLEXITY']
ENGINE_LABEL = {'OPENAI': 'ChatGPT', 'GEMINI': 'Gemini', 'PERPLEXITY': 'Perplexity'}
SEED = 42

# Markdown links in the answer text: [label](https://...)
LINK = re.compile(r'\]\((https?://[^)\s]+)\)')
REDIRECT_HOSTS = ('vertexaisearch.cloud.google.com',)


def clean_url(u):
    """Lower-case host, drop www., utm_* parameters, fragment and trailing slash."""
    p = urlparse(u.strip())
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
         if not k.lower().startswith('utm_')]
    host = p.netloc.lower().removeprefix('www.')
    return urlunparse((p.scheme.lower(), host, p.path.rstrip('/'), '', urlencode(q), ''))


def domain_of(u):
    """Host name without www. (e.g. support.google.com)."""
    return urlparse(u).netloc.lower().removeprefix('www.')


_tld = None


def registered_domain(host):
    """Registered domain via the public suffix list (fiber.google.com -> google.com)."""
    global _tld
    if _tld is None:
        import tldextract
        _tld = tldextract.TLDExtract(suffix_list_urls=())   # bundled snapshot, no network
    ext = _tld(host)
    rd = getattr(ext, 'top_domain_under_public_suffix', None) or ext.registered_domain
    return rd or host


def is_redirect(u):
    return domain_of(u) in REDIRECT_HOSTS


def url_id(u):
    """Short stable id used for downloaded page file names."""
    return hashlib.sha1(u.encode('utf-8')).hexdigest()[:16]


def prompt_id(text):
    return 'p_' + hashlib.sha1(text.strip().encode('utf-8')).hexdigest()[:10]


def excluded_day_engines():
    """(date, engine) pairs dropped after the health check; engine '*' drops the whole day."""
    path = TABLES / 'excluded_days.csv'
    if not path.exists():
        return pd.DataFrame(columns=['date', 'engine', 'reason'])
    return pd.read_csv(path, dtype={'date': str})


def drop_excluded(df):
    """Remove rows whose (date, engine) is listed in excluded_days.csv."""
    ex = excluded_day_engines()
    if ex.empty:
        return df
    date = df['date'].astype(str)
    bad = pd.Series(False, index=df.index)
    for r in ex.itertuples():
        bad |= (date == str(r.date)) & ((r.engine == '*') | (df['engine'] == r.engine))
    return df[~bad]


# Study period of the paper's analyses. Weekly pfetch uploads add later days to the daily tables;
# read_table keeps answers/citations/mentions inside the study period unless all_days=True (used by
# the scripts that rewrite these tables, the health check and the same-day replay comparison).
STUDY_START, STUDY_END = '20260624', '20260921'
DAILY_TABLES = ('answers.csv', 'citations.csv', 'mentions.csv')


def read_table(name, all_days=False, **kw):
    if name not in DAILY_TABLES or all_days:
        return pd.read_csv(TABLES / name, dtype={'date': str}, **kw)
    cols = kw.pop('usecols', None)
    if cols is not None and not callable(cols) and 'date' not in cols:
        d = pd.read_csv(TABLES / name, dtype={'date': str}, usecols=list(cols) + ['date'], **kw)
        return d[d.date.between(STUDY_START, STUDY_END)].drop(columns='date').reset_index(drop=True)
    d = pd.read_csv(TABLES / name, dtype={'date': str}, usecols=cols, **kw)
    return d[d.date.between(STUDY_START, STUDY_END)].reset_index(drop=True)


# Reporting unit. Models use z-scored features; yes/no features are reported as "has vs not" (0 -> 1)
# instead of "per SD". Rescaling a regressor rescales its coefficient, SE and CI by the same factor,
# so p-values and all test verdicts are unchanged. factor = z(raw = 1) - z(raw = 0) = 1 / SD.
BINARY_FEATURES = ('has_faq', 'has_summary_box', 'has_author', 'has_schema')


def unit_factors(d):
    """{feature: factor} for the yes/no features present in d (z-scored column + <name>_raw)."""
    out = {}
    for f in BINARY_FEATURES:
        if f in d and f + '_raw' in d:
            z = d.groupby(d[f + '_raw'])[f].first()
            if 0 in z.index and 1 in z.index:
                out[f] = float(z[1] - z[0])
    return out


def unit_of(term):
    return 'has vs not' if term in BINARY_FEATURES else 'per SD'


def pct_in_unit(coef, term, factors):
    """% more citations for a +1 unit change (log2 outcome); unit = has vs not for yes/no features."""
    return 100 * (2 ** (coef * factors.get(term, 1.0)) - 1)
