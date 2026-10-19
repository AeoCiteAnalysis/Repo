"""Weekly refresh after new pfetch files are copied in.

Copy the new daily files first:
  GFiber   -> pfetchdata/pfetch-dumps/pfetch-427-YYYYMMDD.json
  Workorb  -> pfetch-dumps-workorb/pfetch-146-YYYYMMDD.json
then run
  .venv/Scripts/python.exe scripts/weekly_refresh.py

For each dataset: load the daily files, resolve new Gemini redirect links (newest first - they expire
within weeks, so do not wait long), reload, label websites, health check, and rerun every analysis that
uses the replay (X3, X5 with the same-day comparison, E6, X6/X6b incl. downloading newly consulted pages,
X7, X10). Then paper_numbers.py regenerates paper/numbers.tex and the paper is recompiled
(paper/build/main.pdf), so every changing number in the paper is updated in one run. The paper's R/N/E analyses stay on the study period (common.STUDY_START..STUDY_END) and are
not affected by the new days.
Log: results[/<dataset>]/refresh_steps_<YYYYMMDD>.log (per dataset)
"""
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [['load_data.py'], ['resolve_redirects.py'], ['load_data.py'], ['brand_domains.py', 'apply'],
         ['health.py'],
         # replay-dependent analyses (they grow with every daily replay run)
         ['x03_analyse.py'], ['x05_bootstrap.py'], ['e06_drift.py'],
         ['x06_selection.py', 'fetch'], ['x06_selection.py', 'extras'], ['x06_selection.py', 'analyse'],
         ['x06b_lexical_vs_semantic.py'], ['x07_query_rewrites.py'],
         ['x10_prior_probe.py'], ['x10_prior_probe.py', '--model', 'gemini']]
DATASETS = {'gfiber': ROOT / 'results', 'workorb': ROOT / 'results' / 'workorb'}


def main():
    for ds, res in DATASETS.items():
        log = res / f'refresh_steps_{date.today():%Y%m%d}.log'
        env = {**os.environ, 'AEO_DATASET': ds, 'PYTHONIOENCODING': 'utf-8'}
        print(f'=== {ds} (log: {log.relative_to(ROOT)})', flush=True)
        with open(log, 'w', encoding='utf-8') as f:
            for step in STEPS:
                print('  ' + ' '.join(step), flush=True)
                r = subprocess.run([sys.executable, str(ROOT / 'scripts' / step[0])] + step[1:], env=env,
                                   stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
                if r.returncode:
                    print(f'  FAILED ({step[0]}), see the log', flush=True)
                    break
        same = res / 'x05_same_day.csv'
        if same.exists() and same.stat().st_size > 5:
            print(open(same, encoding='utf-8').read())
    # every changing number of the paper, then compile
    print('=== paper numbers', flush=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'paper_numbers.py')], cwd=ROOT)
    compile_paper()


def compile_paper():
    paper, build = ROOT / 'paper', ROOT / 'paper' / 'build'
    build.mkdir(exist_ok=True)
    (build / 'references.bib').write_bytes((paper / 'references.bib').read_bytes())
    run = lambda *c: subprocess.run(list(c), cwd=paper, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run('pdflatex', '-interaction=nonstopmode', '-output-directory=build', 'main.tex')
    subprocess.run(['bibtex', 'main'], cwd=build, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(2):
        run('pdflatex', '-interaction=nonstopmode', '-output-directory=build', 'main.tex')
    print('=== paper compiled -> paper/build/main.pdf', flush=True)


if __name__ == '__main__':
    main()
