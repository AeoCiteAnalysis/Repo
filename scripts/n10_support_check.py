"""N10 - Does the cited page support the answer?

Question. Human (2 raters) then LLM judgement of 300 sentence-link pairs; scale to 3,000 if LLM agreement > 80 %.
Needs. answers.csv, page_text.jsonl  (run after: A6)
Output. results/n10_support_check.csv and figures/n10_support_check_*.png

Status: not implemented yet - see the experiments guide for the full steps.
"""
import pandas as pd

from common import TABLES, RESULTS, FIGURES, SEED, read_table


def main():
    raise NotImplementedError('n10_support_check: to do')


if __name__ == '__main__':
    main()
