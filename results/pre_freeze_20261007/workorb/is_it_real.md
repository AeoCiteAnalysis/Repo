# "Is it real?" table

Effect = R2 estimate: % more citations per +1 SD; for yes/no features (FAQ, summary box, author, structured data) % more citations when the page has it vs not. Pages are compared on the same website.

## all pages

| feature | effect | 1 significance (R3) | 2 repeat selection (R4) | 3 double ML (R5) | 4 shape (R6) | 5 robustness (R9) | real? |
|---|---|---|---|---|---|---|---|
| title matches a prompt | +26.4% | ✓ | ✓ | ✓ | ✓ | ✓ | **yes** |
| best paragraph matches | +12.9% | ✓ | ✓ | ✓ | ✓ | ✓ | **yes** |
| word overlap with prompts | +13.0% | ✓ | ✗ | ✓ | ✓ | ✓ | **no** |
| headings | +7.7% | ✓ | ✗ | ✓ | ✓ | ✓ | **no** |
| has structured data | +16.3% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| has FAQ | -1.9% | ✗ | ✗ | ✓ | ✓ | ✗ | **no** |
| has summary box | +8.0% | ✗ | ✗ | ✗ | ✓ | ✗ | **no** |
| outbound links | +3.2% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |
| word count | +2.2% | ✗ | ✗ | ✓ | ✗ | ✗ | **no** |
| has author | +2.1% | ✗ | ✗ | ✗ | ✓ | ✗ | **no** |
| page age | -2.7% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |
| lists | -4.8% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |
| first paragraph matches | +0.1% | ✗ | ✗ | ✗ | ✗ | ✗ | **no** |
| tables | -2.5% | ✗ | ✗ | ✗ | ✗ | ✗ | **no** |

## own + competitor

| feature | effect | 1 significance (R3) | 2 repeat selection (R4) | 3 double ML (R5) | 4 shape (R6) | 5 robustness (R9) | real? |
|---|---|---|---|---|---|---|---|
| title matches a prompt | +33.6% | ✓ | ✓ | ✓ | ✓ | ✓ | **yes** |
| best paragraph matches | +11.9% | ✗ | ✓ | ✓ | ✗ | ✓ | **no** |
| word overlap with prompts | +14.5% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| headings | +8.9% | ✗ | ✗ | ✓ | ✗ | ✓ | **no** |
| has FAQ | +3.2% | ✗ | ✗ | ✓ | ✓ | ✗ | **no** |
| has author | +4.2% | ✗ | ✗ | ✗ | ✓ | ✗ | **no** |
| has summary box | +3.0% | ✗ | ✗ | ✗ | ✓ | ✗ | **no** |
| outbound links | +1.2% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |
| has structured data | +1.1% | ✗ | ✗ | ✗ | ✓ | ✗ | **no** |
| page age | -4.0% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |
| word count | +1.5% | ✗ | ✗ | ✗ | ✗ | ✗ | **no** |
| first paragraph matches | -2.1% | ✗ | ✗ | ✗ | ✗ | ✗ | **no** |
| lists | -2.8% | ✗ | ✗ | ✗ | ✗ | ✗ | **no** |
| tables | -4.4% | ✗ | ✗ | ✗ | ✗ | ✗ | **no** |

## third party

| feature | effect | 1 significance (R3) | 2 repeat selection (R4) | 3 double ML (R5) | 4 shape (R6) | 5 robustness (R9) | real? |
|---|---|---|---|---|---|---|---|
| title matches a prompt | +17.6% | ✓ | ✓ | ✓ | ✓ | ✓ | **yes** |
| best paragraph matches | +14.6% | ✗ | ✓ | ✓ | ✓ | ✓ | **no** |
| headings | +5.9% | ✓ | ✗ | ✓ | ✓ | ✓ | **no** |
| has structured data | +28.2% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| has summary box | +16.4% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| word overlap with prompts | +10.5% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| outbound links | +4.9% | ✗ | ✗ | ✗ | ✓ | ✓ | **no** |
| has author | -0.0% | ✗ | ✗ | ✗ | ✓ | ✗ | **no** |
| lists | -5.1% | ✗ | ✗ | ✗ | ✗ | ✓ | **no** |
| has FAQ | -5.8% | ✗ | ✗ | ✗ | ✓ | ✗ | **no** |
| first paragraph matches | +1.0% | ✗ | ✗ | ✗ | ✗ | ✗ | **no** |
| page age | +0.4% | ✗ | ✗ | ✗ | ✗ | ✗ | **no** |
| word count | -0.3% | ✗ | ✗ | ✗ | ✗ | ✗ | **no** |
| tables | -1.9% | ✗ | ✗ | ✗ | ✗ | ✗ | **no** |

## Appendix: features that did not pass

- **all pages**: word overlap with prompts (failed: repeat selection (R4)); first paragraph matches (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6), robustness (R9)); word count (failed: significance (R3), repeat selection (R4), shape (R6), robustness (R9)); outbound links (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6)); headings (failed: repeat selection (R4)); tables (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6), robustness (R9)); lists (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6)); has FAQ (failed: significance (R3), repeat selection (R4), robustness (R9)); has summary box (failed: significance (R3), repeat selection (R4), double ML (R5), robustness (R9)); has author (failed: significance (R3), repeat selection (R4), double ML (R5), robustness (R9)); has structured data (failed: significance (R3), repeat selection (R4), double ML (R5)); page age (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6))
- **own + competitor**: word overlap with prompts (failed: significance (R3), repeat selection (R4), double ML (R5)); first paragraph matches (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6), robustness (R9)); best paragraph matches (failed: significance (R3), shape (R6)); word count (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6), robustness (R9)); outbound links (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6)); headings (failed: significance (R3), repeat selection (R4), shape (R6)); tables (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6), robustness (R9)); lists (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6), robustness (R9)); has FAQ (failed: significance (R3), repeat selection (R4), robustness (R9)); has summary box (failed: significance (R3), repeat selection (R4), double ML (R5), robustness (R9)); has author (failed: significance (R3), repeat selection (R4), double ML (R5), robustness (R9)); has structured data (failed: significance (R3), repeat selection (R4), double ML (R5), robustness (R9)); page age (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6))
- **third party**: word overlap with prompts (failed: significance (R3), repeat selection (R4), double ML (R5)); first paragraph matches (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6), robustness (R9)); best paragraph matches (failed: significance (R3)); word count (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6), robustness (R9)); outbound links (failed: significance (R3), repeat selection (R4), double ML (R5)); headings (failed: repeat selection (R4)); tables (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6), robustness (R9)); lists (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6)); has FAQ (failed: significance (R3), repeat selection (R4), double ML (R5), robustness (R9)); has summary box (failed: significance (R3), repeat selection (R4), double ML (R5)); has author (failed: significance (R3), repeat selection (R4), double ML (R5), robustness (R9)); has structured data (failed: significance (R3), repeat selection (R4), double ML (R5)); page age (failed: significance (R3), repeat selection (R4), double ML (R5), shape (R6), robustness (R9))