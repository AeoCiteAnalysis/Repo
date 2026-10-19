"""Compare two numbers.tex files: which paper macros changed.

  python scripts/diff_numbers.py results/pre_freeze_20261007/numbers.tex paper/numbers.tex
"""
import re
import sys

PAT = re.compile(r'\\newcommand\{\\(\w+)\}\{(.*)\}')


def read(p):
    return dict(PAT.findall(open(p, encoding='utf-8').read()))


def main(old, new):
    a, b = read(old), read(new)
    changed = [(k, a[k], b[k]) for k in sorted(b) if k in a and a[k] != b[k]]
    print(f'{len(changed)} of {len(b)} macros changed; {len(set(b) - set(a))} new, {len(set(a) - set(b))} removed')
    for k, x, y in changed:
        print(f'  {k:30s} {x:>14s}  ->  {y}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
