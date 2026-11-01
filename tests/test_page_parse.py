import json
from pathlib import Path

import pytest

from page_parse import parse_html, page_type

FIXTURE = Path(__file__).parent / 'fixtures' / 'article.html'


@pytest.mark.parametrize('url,title,expected', [
    ('https://youtube.com/watch?v=1', '', 'video'),
    ('https://reddit.com/r/googlefiber/comments/x', '', 'forum'),
    ('https://gfiber.com/faq', 'FAQ', 'support/help'),
    ('https://gfiber.com/support/en/answer/1962', 'Plan prices change', 'support/help'),
    ('https://fiber.google.com/plans', '', 'pricing/plans'),
    ('https://cabletv.com/reviews/google-fiber-vs-att-fiber', '', 'comparison'),
    ('https://cnet.com/home/internet/google-fiber-internet-review', 'Review: plans and pricing',
     'review list'),
    ('https://cnet.com/home/internet/best-fiber-internet-providers', '', 'list'),
    ('https://fiber.google.com/internet', 'Fiber internet', 'product/service'),
    ('https://example.com/blog/why-outages-happen', 'Why outages happen', 'article'),
    ('https://example.com/about', 'About us', 'other'),
])
def test_page_type(url, title, expected):
    assert page_type(url, title) == expected


def test_parse_fixture():
    html = FIXTURE.read_text(encoding='utf-8')
    f = parse_html(html, 'https://example.com/blog/fiber-vs-cable')
    assert f['title'].startswith('Fiber vs cable')
    assert f['word_count'] > 100
    assert f['has_faq'] == 1                 # "Frequently asked questions" heading + FAQPage
    assert f['has_summary_box'] == 1         # "Key takeaways"
    assert f['has_author'] == 1
    assert f['has_schema'] == 1
    assert 'FAQPage' in json.loads(f['schema_types'])
    assert f['outbound_links'] == 2          # fcc.gov and reddit.com; same-site link ignored
    assert f['n_tables'] == 1
    assert f['page_date'] == '2026-05-02'
    assert f['page_type'] == 'comparison'
    assert len(f['paragraphs']) >= 3
