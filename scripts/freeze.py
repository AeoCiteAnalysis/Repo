"""Data freeze for the paper (run once after the last pfetch upload and weekly_refresh.py).

The weekly refresh updates only the replay-dependent analyses. The study-period analyses (R1-R14, the
five-test table, E1-E3, X2, X9, X11) were computed before the Gemini redirect links were resolved; this
script brings them up to date, per dataset:

  1. download the pages behind newly resolved redirect links (same round: only missing URLs are fetched)
  2. remeasure page features (deterministic for pages already measured)
  3. rebuild the URL table
  4. rerun the study-period analyses
then regenerate paper/numbers.tex and the appendix tables and recompile the paper.

  .venv/Scripts/python.exe scripts/freeze.py            -> log: results/freeze_<YYYYMMDD>.log
A failed step is reported and the run continues; a snapshot of the previous results is kept in
results/pre_freeze_<date>/ (made by hand before the run).
"""
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUND = {'gfiber': '2026-09-26', 'workorb': '2026-09-27'}
ANALYSES = [['r01_describe.py'], ['r02_main_model.py'], ['r03_fdr.py'], ['r04_stability_selection.py'],
            ['r05_double_ml.py'], ['r06_shape.py'], ['r07_importance.py'], ['r08_simpson.py'],
            ['r09_robustness.py'], ['r10_paragraph_depth.py'], ['r11_engines.py'], ['r12_page_type_topic.py'],
            ['r13_variance_split.py'], ['r14_authority.py'], ['is_it_real.py'],
            ['e01_engines_breadth_depth.py'], ['e02_choice.py'], ['e03_entity_coverage.py'],
            ['x02_prompt_match.py'], ['x09_page_authority.py'], ['x11_brand_mentions.py']]


def main():
    log_path = ROOT / 'results' / f'freeze_{date.today():%Y%m%d}.log'
    failed = []
    with open(log_path, 'w', encoding='utf-8') as log:
        for ds in ('gfiber', 'workorb'):
            env = {**os.environ, 'AEO_DATASET': ds, 'PYTHONIOENCODING': 'utf-8'}
            steps = [['fetch_pages.py', '--round', ROUND[ds], '--min-citations', '1'],
                     ['features.py', '--round', ROUND[ds]], ['build_url_table.py']] + ANALYSES
            for step in steps:
                t0 = datetime.now()
                msg = f'=== {ds} {" ".join(step)}  ({t0:%H:%M})'
                print(msg, flush=True)
                log.write(msg + '\n')
                log.flush()
                r = subprocess.run([sys.executable, str(ROOT / 'scripts' / step[0])] + step[1:], env=env,
                                   stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                mins = (datetime.now() - t0).total_seconds() / 60
                if r.returncode:
                    failed.append(f'{ds} {step[0]}')
                    print(f'  FAILED after {mins:.1f} min, see {log_path.name}', flush=True)
                else:
                    print(f'  ok, {mins:.1f} min', flush=True)
        for step in (['paper_numbers.py'], ['paper_appendix.py']):
            subprocess.run([sys.executable, str(ROOT / 'scripts' / step[0])], stdout=log, stderr=subprocess.STDOUT,
                           cwd=ROOT, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    sys.path.insert(0, str(ROOT / 'scripts'))
    import weekly_refresh
    weekly_refresh.compile_paper()
    print('FAILED steps:', ', '.join(failed) if failed else 'none', flush=True)


if __name__ == '__main__':
    main()
