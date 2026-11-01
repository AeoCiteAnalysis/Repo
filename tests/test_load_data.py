import json

import pytest

from common import RAW_DIR
from load_data import parse_day


def _item(aid, engine, citations, text='', mentions=()):
    return {'facetId': aid, 'pfetch': {
        'searchVendorEnum': engine, 'topicName': 'Pricing', 'prompt': 'Is GFiber cheap?',
        'chatConvResponse': {'assistantResponse': text, 'chatModelEnum': 'M',
                             'stopwatch': {'startTime': 1000, 'stopTime': 3500}},
        'analysis': {'citations': citations, 'brandMentions': list(mentions)}}}


def test_parse_day_citations_repeats_redirects_and_text_fallback():
    day = {'date': '20260921', 'items': [
        _item('a1', 'GEMINI', [
            {'url': 'https://www.gfiber.com/plans/?utm_source=x', 'title': 'gfiber.com'},
            {'url': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AB',
             'title': 'facebook.com'},
            {'url': 'https://gfiber.com/plans', 'title': 'gfiber.com'}],
            mentions=[{'brandName': 'Google Fiber', 'rank': None, 'sentiment': 'POSITIVE'}]),
        _item('a2', 'PERPLEXITY', [], text='See [cnet](https://www.cnet.com/x?utm_medium=a).'),
    ]}
    answers, cites, mentions = parse_day(day, redirects={})
    assert len(answers) == 2 and answers[0]['latency_ms'] == 2500
    a1 = [c for c in cites if c['answer_id'] == 'a1']
    assert [c['position'] for c in a1] == [1, 2, 3]
    assert a1[0]['url'] == 'https://gfiber.com/plans'
    assert a1[2]['repeat_in_answer'] and not a1[0]['repeat_in_answer']
    assert a1[1]['is_redirect'] and a1[1]['domain'] == 'facebook.com'   # site from title
    assert answers[0]['n_unique_citations'] == 2
    a2 = [c for c in cites if c['answer_id'] == 'a2']
    assert a2[0]['source'] == 'answer_text' and a2[0]['url'] == 'https://cnet.com/x'
    assert mentions[0]['rank_missing']


def test_redirect_map_applied():
    red = 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AB'
    day = {'date': '20260921', 'items': [_item('a1', 'GEMINI', [{'url': red, 'title': 'x.com'}])]}
    _, cites, _ = parse_day(day, redirects={red: 'https://x.com/page'})
    assert cites[0]['url'] == 'https://x.com/page' and not cites[0]['is_redirect']


SAMPLE = RAW_DIR / 'pfetch-427-20260921.json'


@pytest.mark.skipif(not SAMPLE.exists(), reason='sample day file not present')
def test_sample_day_matches_guide_counts():
    """The guide (Step A2) reports 1,704 answers, 12,179 citation rows, 2,489 mentions."""
    day = json.load(open(SAMPLE, encoding='utf-8'))
    answers, cites, mentions = parse_day(day, redirects={})
    assert (len(answers), len(cites), len(mentions)) == (1704, 12179, 2489)
