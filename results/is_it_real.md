# "Is it real?" table

Effect = R2 estimate: % more citations per +1 SD; for yes/no features (FAQ, summary box, author, structured data) % more citations when the page has it vs not. Pages are compared on the same website.

## all pages

| feature | effect | 1 significance (R3) | 2 repeat selection (R4) | 3 double ML (R5) | 4 shape (R6) | 5 robustness (R9) | real? |
|---|---|---|---|---|---|---|---|
| best paragraph matches | +15.5% | ✓ | ✓ | ✓ | ✓ | ✓ | **yes** |
| title matches a prompt | +14.1% | ✓ | ✓ | ✓ | ✓ | ✓ | **yes** |
| word overlap with prompts | +9.1% | ✓ | ✗ | ✓ | ✓ | ✓ | **no** |
| first paragraph matches | +5.9% | ✓ | ✗ | ✓ | ✓ | ✓ | **no** |
| page age | -5.1% | ✓ | ✗ | ✓ | ✓ | ✓ | **no** |
| has summary box | +6.9% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| headings | -2.7% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| has FAQ | -3.4% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| has author | -4.6% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| has structured data | -5.3% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| outbound links | +1.4% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |
| word count | +0.9% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |
| lists | -1.0% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |
| tables | -1.4% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |

## own + competitor

| feature | effect | 1 significance (R3) | 2 repeat selection (R4) | 3 double ML (R5) | 4 shape (R6) | 5 robustness (R9) | real? |
|---|---|---|---|---|---|---|---|
| first paragraph matches | +7.8% | ✓ | ✗ | ✓ | ✓ | ✓ | **no** |
| title matches a prompt | +7.4% | ✓ | ✗ | ✓ | ✓ | ✓ | **no** |
| has author | -15.8% | ✓ | ✗ | ✓ | ✓ | ✓ | **no** |
| tables | +12.5% | ✓ | ✗ | ✗ | ✓ | ✓ | **no** |
| word overlap with prompts | +11.9% | ✓ | ✗ | ✓ | ✗ | ✓ | **no** |
| page age | -10.6% | ✓ | ✗ | ✗ | ✓ | ✓ | **no** |
| lists | -18.1% | ✓ | ✗ | ✗ | ✓ | ✓ | **no** |
| best paragraph matches | +5.2% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| has summary box | +2.6% | ✗ | ✗ | ✓ | ✓ | ✗ | **no** |
| has FAQ | -1.9% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| has structured data | -12.2% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| outbound links | +6.6% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |
| word count | +2.9% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |
| headings | -1.4% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |

## third party

| feature | effect | 1 significance (R3) | 2 repeat selection (R4) | 3 double ML (R5) | 4 shape (R6) | 5 robustness (R9) | real? |
|---|---|---|---|---|---|---|---|
| best paragraph matches | +25.7% | ✓ | ✓ | ✓ | ✓ | ✓ | **yes** |
| title matches a prompt | +23.6% | ✓ | ✓ | ✓ | ✓ | ✓ | **yes** |
| word overlap with prompts | +12.8% | ✓ | ✓ | ✓ | ✓ | ✓ | **yes** |
| first paragraph matches | +3.1% | ✗ | ✗ | ✓ | ✓ | ✓ | **no** |
| page age | -4.2% | ✗ | ✗ | ✓ | ✓ | ✓ | **no** |
| has FAQ | -9.2% | ✗ | ✗ | ✓ | ✓ | ✓ | **no** |
| has structured data | +11.7% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| has summary box | +5.1% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| has author | +3.7% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| word count | -3.0% | ✗ | ✗ | ✓ | ✗ | ✓ | **no** |
| headings | +1.9% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |
| lists | +0.5% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |
| tables | -4.0% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |
| outbound links | +0.5% | ✗ | ✗ | ✗ | ✗ | ✗ | **no** |

## Appendix: features that did not pass

- **all pages**: word overlap with prompts (failed: repeat selection (R4)); first paragraph matches (failed: repeat selection (R4)); word count (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6)); outbound links (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6)); headings (failed: significance (R3), repeat selection (R4), double ML (R5)); tables (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6)); lists (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6)); has FAQ (failed: significance (R3), repeat selection (R4), double ML (R5)); has summary box (failed: significance (R3), repeat selection (R4), double ML (R5)); has author (failed: significance (R3), repeat selection (R4), double ML (R5)); has structured data (failed: significance (R3), repeat selection (R4), double ML (R5)); page age (failed: repeat selection (R4))
- **own + competitor**: word overlap with prompts (failed: repeat selection (R4), shape (R6)); title matches a prompt (failed: repeat selection (R4)); first paragraph matches (failed: repeat selection (R4)); best paragraph matches (failed: significance (R3), repeat selection (R4), double ML (R5)); word count (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6)); outbound links (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6)); headings (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6)); tables (failed: repeat selection (R4), double ML (R5)); lists (failed: repeat selection (R4), double ML (R5)); has FAQ (failed: significance (R3), repeat selection (R4), double ML (R5)); has summary box (failed: significance (R3), repeat selection (R4), robustness (R9)); has author (failed: repeat selection (R4)); has structured data (failed: significance (R3), repeat selection (R4), double ML (R5)); page age (failed: repeat selection (R4), double ML (R5))
- **third party**: first paragraph matches (failed: significance (R3), repeat selection (R4)); word count (failed: significance (R3), repeat selection (R4), shape (R6)); outbound links (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6), robustness (R9)); headings (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6)); tables (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6)); lists (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6)); has FAQ (failed: significance (R3), repeat selection (R4)); has summary box (failed: significance (R3), repeat selection (R4), double ML (R5)); has author (failed: significance (R3), repeat selection (R4), double ML (R5)); has structured data (failed: significance (R3), repeat selection (R4), double ML (R5)); page age (failed: significance (R3), repeat selection (R4))