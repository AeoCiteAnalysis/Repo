"""Daily engine replay (run by Windows Task Scheduler, task "AeoCiteAnalysis daily replay").

Without AEO_DATASET set, it replays every dataset in common.DATASETS (gfiber, then workorb).

Replays all prompts of the dataset once per day through ChatGPT (gpt-5.4-nano + web search) and Gemini
(3.1 Flash Lite + Google Search grounding) with x03_replay.py, so that retrieval (search queries,
ranked consulted pages) is recorded on the same days as the pfetch collection. pfetch asks ChatGPT
around 20:05-21:30 US Eastern, so the task starts at 20:00; ChatGPT runs first (~15 min), then Gemini.

Files: data/replay/<engine>/<YYYYMMDD>/<prompt_id>_r1.json (skipped if present, so a rerun on the
same day only fills gaps). Each engine gets a second pass for prompts that failed.
Log: data/replay/daily_log.csv (one row per engine per day) and results/replay_logs/<YYYYMMDD>.log.

  .venv/Scripts/python.exe scripts/x03_daily.py            run today
  .venv/Scripts/python.exe scripts/x03_daily.py --status   show the last days of the log
"""
import argparse
import contextlib
import os
import subprocess
import csv
import io
import json
import sys
import time
from datetime import date, datetime

import pandas as pd

from common import ROOT, TABLES, RESULTS, DATASETS
import x03_replay

LOGDIR = RESULTS / 'replay_logs'
DAILY = x03_replay.OUT / 'daily_log.csv'
ENGINES = ('openai', 'gemini')


def usage(engine, day):
    tin = tout = searched = n = 0
    for f in (x03_replay.OUT / engine / day).glob('*_r1.json'):
        j = json.loads(f.read_text(encoding='utf-8'))
        u = j.get('usage') or {}
        tin += u.get('input_tokens', u.get('promptTokenCount', 0)) or 0
        tout += u.get('output_tokens', u.get('candidatesTokenCount', 0)) or 0
        searched += bool(j['queries'])
        n += 1
    return n, searched, tin, tout


def run(engine, day, log):
    t0 = time.time()
    for attempt in (1, 2):                                   # second pass fills failed prompts
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                x03_replay.fetch(engine, 0, 1, 8 if engine == 'openai' else 6)
            except Exception as e:                            # e.g. missing key, quota
                print(f'{engine} pass {attempt} stopped: {type(e).__name__}: {e}')
        log.write(buf.getvalue()); log.flush()
        n, *_ = usage(engine, day)
        if n >= len(pd.read_csv(TABLES / 'prompt_labels.csv', usecols=['prompt_id'])):
            break
    n, searched, tin, tout = usage(engine, day)
    row = dict(date=day, engine=engine, prompts_saved=n, searched=searched, input_tokens=tin,
               output_tokens=tout, minutes=round((time.time() - t0) / 60, 1),
               finished=datetime.now().strftime('%Y-%m-%d %H:%M'))
    new = not DAILY.exists()
    with open(DAILY, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    log.write(f'{row}\n'); log.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--engines', nargs='+', default=list(ENGINES))
    a = ap.parse_args()
    if 'AEO_DATASET' not in os.environ:          # scheduled task: ChatGPT for every dataset first, then Gemini
        if a.status:
            for ds in DATASETS:
                print(f'--- {ds}', flush=True)
                subprocess.run([sys.executable, __file__, '--status'], env={**os.environ, 'AEO_DATASET': ds})
            return
        order = sorted(DATASETS, key=lambda d: d != 'workorb')   # Workorb's ChatGPT collection runs 20:00-20:15
        for e in a.engines:
            for ds in order:
                subprocess.run([sys.executable, __file__, '--engines', e], env={**os.environ, 'AEO_DATASET': ds})
        return
    if a.status:
        print(pd.read_csv(DAILY).tail(20).to_string(index=False) if DAILY.exists() else 'no runs yet')
        return
    day = f'{date.today():%Y%m%d}'
    LOGDIR.mkdir(parents=True, exist_ok=True)
    x03_replay.OUT.mkdir(parents=True, exist_ok=True)
    with open(LOGDIR / f'{day}.log', 'a', encoding='utf-8') as log:
        log.write(f'=== {datetime.now():%Y-%m-%d %H:%M} daily replay\n')
        for e in a.engines:
            run(e, day, log)


if __name__ == '__main__':
    main()
