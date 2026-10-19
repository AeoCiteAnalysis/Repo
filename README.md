# What makes AI engines cite a page? (client 427)

Code for the experiments guide `experiments-guide-geo.docx`: daily answers from ChatGPT,
Gemini and Perplexity to 651 prompts about 8 tracked internet-provider brands (+ Verizon added as competitor), 2026-06-24 → 2026-09-21.

## Set-up

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q tests
```

### API keys (optional)

Copy `.env.example` to `.env` and fill in the keys you have; every script reads it.

- `OPENAI_API_KEY`, `OPENAI_MODEL` — LLM intent labels (`prompt_labels.py`), later N10.
  Test with `python scripts/llm.py`.
- `PSI_API_KEY` — Google key with *Chrome UX Report API* and *PageSpeed Insights API* enabled
  (`python scripts/speed.py crux`, then `build_url_table.py` and `r08_simpson.py`).

Run every script from this folder: `python scripts/<name>.py`. On Windows set
`PYTHONIOENCODING=utf-8` so non-ASCII text prints correctly.

## Folder layout

```
pfetchdata/pfetch-dumps/  one JSON file per day (raw, never edit)
data/tables/              CSV tables built by the scripts (+ brand_domains.csv, excluded_days.csv by hand)
data/tables/authority/    dated snapshots of outside authority scores
data/pages/<round>/       downloaded HTML, one folder per download round
scripts/                  one script per step / experiment; common.py has shared paths & helpers
results/                  one CSV per experiment
figures/                  one PNG per chart
report/                   the written report
decisions.md              every choice that changes a number, with date and reason
```

## Part A — prepare the data

| Step | Command | Output | Status |
|---|---|---|---|
| A2 | `python scripts/load_data.py` | `answers.csv`, `citations.csv`, `mentions.csv` | done |
| A3 | `python scripts/health.py` | `results/health_*.csv`, `figures/health_*.png` | done |
| A4 | `python scripts/brand_domains.py list` → edit `data/tables/brand_domains.csv` → `... apply` | `role`, `brand`, `reg_domain` in `citations.csv` | done: 92 rows, reviewed 2026-09-26 |
| — | `python scripts/resolve_redirects.py` then re-run A2 + A4 | `redirect_map.csv` | ready; run after each new day |
| A5 | `python scripts/fetch_pages.py --round 2026-09-26 --min-citations 1` | `data/pages/<round>/`, `pages_index.csv` | done: round 2026-09-26, all 37,972 URLs (64.9 % ok) |
| A6 | `python scripts/features.py` (+ optional `scripts/speed.py`, needs `PSI_API_KEY`) | `page_features.csv`, `page_text.jsonl`, `page_speed.csv` | done: 24,646 pages (speed not run); **hand checks to do** |
| A6b | `python scripts/authority.py cc --release cc-main-2026-jun-jul-aug` / `majestic` / `tranco` / `crux --month 202608` / `merge` (no keys needed) | `domain_authority.csv` | done 2026-09-26 (refresh monthly) |
| A7 | `python scripts/build_url_table.py` | `url_table.csv` (≥ 2 citations), `url_table_all.csv`, `url_counts.csv` | done: counts, page features, authority |

Full rebuild after new day files arrive:

```bash
python scripts/resolve_redirects.py
python scripts/load_data.py
python scripts/brand_domains.py apply
python scripts/health.py
python scripts/build_url_table.py
```

## Parts B and C — experiments

Done: all of Part B — R1–R14 (`r01_describe.py` … `r14_authority.py`) and the "is it real?" table (`is_it_real.py` → `results/is_it_real.md`). Shared code: `fe.py`, `r02_main_model.prepare`, `prompt_labels.py`, `plotstyle.py`. Stubs for the rest in `scripts/` (`r01_describe.py` … `r14_authority.py`, `is_it_real.py`,
`n01_day_stability.py` … `n14_source_types.py`). Each stub's docstring states the question,
the inputs and which Part A steps it needs.

## Part C — new experiments

Done: N1 day-to-day stability, N2 engine agreement, N3 cited vs mentioned, N4 sources behind
negative mentions, N5 question features, N6 first position, N9 survival, N13 match measures,
N14 website types (LLM-labelled top 300). N7 (brand-sitemap half) in progress.
Not possible yet: N8 (needs download rounds every 2 weeks — next round from 2026-10-10),
N10 (needs two human raters first), N11 (needs edits to client pages), N12 (needs new queries
to the engines), N7 search-results half (needs a paid search API).
Hand checks waiting: `results/intent_check_sample.csv`, `results/n14_type_check_sample.csv`,
`results/page_check_sample.csv`, `results/page_type_check_sample.csv`.

## Current numbers (all 90 days)

- 143,249 answers (ChatGPT 58,590 · Perplexity 58,590 · Gemini 26,069), 651 prompts, 7 topics.
- 390,081 citations counted once per answer after dropping 14,920 unresolved redirects;
  37,972 different URLs on 7,218 websites; 21,815 URLs cited ≥ 2 times (main table).
- Share of citations by role (8 competitor brands incl. Verizon) — ChatGPT: own 24.0 %, competitor 40.3 %, third party 35.5 %;
  Gemini: 7.9 / 8.9 / 83.1 %; Perplexity: 2.9 / 3.9 / 93.2 %.

See `decisions.md` for known data issues (Perplexity citations, Gemini coverage changes, redirects).

## Weekly update (new pfetch files)

1. Copy the new daily files: GFiber into `pfetchdata/pfetch-dumps/`, Workorb into `pfetch-dumps-workorb/`.
2. Run `.venv/Scripts/python.exe scripts/weekly_refresh.py` (both datasets; about 10–30 minutes).

The paper's analyses stay on the study period 24 Jun – 21 Sep 2026 (`STUDY_START`/`STUDY_END` in
`scripts/common.py`); new days only feed the same-day comparison with the daily replay
(`results[/workorb]/x05_same_day.csv`). The daily replay runs at 20:00 (Task Scheduler,
"AeoCiteAnalysis daily replay"); check it with `.venv/Scripts/python.exe scripts/x03_daily.py --status`.
