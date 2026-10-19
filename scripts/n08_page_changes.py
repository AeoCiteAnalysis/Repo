"""N8 - When a page changes, do citations change?

Question. Difference-in-differences on pages changed between 2-weekly download rounds vs unchanged pages on the same website.
Needs. pages_index.csv (several rounds), url_counts.csv  (run after: A5 every 2 weeks)
Output. results/n08_page_changes.csv and figures/n08_page_changes_*.png

Status: not implemented yet - see the experiments guide for the full steps.
"""
import pandas as pd

from common import TABLES, RESULTS, FIGURES, SEED, read_table


def main():
    raise NotImplementedError('n08_page_changes: to do')


if __name__ == '__main__':
    main()
