import pandas as pd

from common import clean_url, domain_of, registered_domain, is_redirect
from brand_domains import label_hosts


def test_clean_url_drops_utm_www_slash_fragment():
    u = 'https://WWW.Example.com/Path/?utm_source=openai&id=3#top'
    assert clean_url(u) == 'https://example.com/Path?id=3'


def test_clean_url_same_page_same_key():
    assert clean_url('https://gfiber.com/faq/') == clean_url('https://www.gfiber.com/faq?utm_medium=x')


def test_domain_and_registered_domain():
    assert domain_of('https://www.support.google.com/x') == 'support.google.com'
    assert registered_domain('fiber.google.com') == 'google.com'
    assert registered_domain('forums.bbc.co.uk') == 'bbc.co.uk'


def test_is_redirect():
    assert is_redirect('https://vertexaisearch.cloud.google.com/grounding-api-redirect/AB')
    assert not is_redirect('https://gfiber.com')


def test_label_hosts_longest_suffix_wins():
    bd = pd.DataFrame([('Google Fiber', 'fiber.google.com', 'own'),
                       (None, 'support.google.com', 'mixed'),
                       ('AT&T Fiber', 'att.com', 'competitor')],
                      columns=['brand', 'domain', 'role'])
    lab = label_hosts(['fiber.google.com', 'support.google.com', 'about.att.com',
                       'google.com', 'notatt.com'], bd).set_index('domain')
    assert lab.loc['fiber.google.com', 'role'] == 'own'
    assert lab.loc['support.google.com', 'role'] == 'mixed'
    assert lab.loc['about.att.com', 'role'] == 'competitor'
    assert lab.loc['google.com', 'role'] == 'third_party'
    assert lab.loc['notatt.com', 'role'] == 'third_party'      # suffix must start at a dot


def test_label_sites_path_prefix():
    from brand_domains import label_sites, site_key
    bd = pd.DataFrame([('Google Fiber', 'support.google.com/fiber', 'own', 'sure'),
                       (None, 'support.google.com', 'mixed', 'sure')],
                      columns=['brand', 'domain', 'role', 'confidence'])
    keys = [site_key('support.google.com', 'https://support.google.com/fiber/answer/1', False),
            site_key('support.google.com', 'https://support.google.com/fiberglass', False),
            site_key('support.google.com', 'https://support.google.com/gmail', False),
            site_key('support.google.com', 'https://vertexaisearch.cloud.google.com/x', True)]
    lab = label_sites(keys, bd)
    assert list(lab.role) == ['own', 'mixed', 'mixed', 'mixed']
    assert label_sites(['cnet.com/x'], bd).label_confidence[0] == 'default'


def test_play_store_app_rules():
    from brand_domains import label_sites, site_key
    bd = pd.DataFrame([
        ('Google Fiber', 'play.google.com/store/apps/details?id=com.google.android.apps.fiber.myfiber',
         'own', 'sure'),
        ('AT&T Fiber', 'play.google.com/store/apps/details?id=com.att.shm', 'competitor', 'sure'),
        (None, 'play.google.com/store/apps/details', 'third_party', 'sure')],
        columns=['brand', 'domain', 'role', 'confidence'])
    bd['domain'] = bd.domain.str.lower()
    base = 'https://play.google.com/store/apps/details?id='
    keys = [site_key('play.google.com', base + a + '&hl=en', False)
            for a in ('com.google.android.apps.fiber.myfiber', 'com.att.shm', 'org.zwanoo.android.speedtest')]
    assert list(label_sites(keys, bd).role) == ['own', 'competitor', 'third_party']
