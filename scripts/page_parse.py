"""Pure functions that measure one downloaded page (used by features.py).

parse_html(html, url) returns the page's title, main-text paragraphs and the
structure / freshness / page-type features of Step A6. Nothing here touches
the network or the file system, so it is easy to test on fixture pages.
"""
import json
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from common import domain_of, registered_domain

FAQ_RE = re.compile(r'\b(faqs?|frequently asked questions|common questions)\b', re.I)
SUMMARY_RE = re.compile(r'\b(tl;?\s?dr|key takeaways?|in short|at a glance|quick answer|'
                        r'the short answer|bottom line|summary)\b', re.I)
AUTHOR_ATTR_RE = re.compile(r'author|byline', re.I)
MIN_PARA_WORDS = 5

VIDEO_HOSTS = ('youtube.com', 'youtu.be', 'vimeo.com', 'tiktok.com')
FORUM_HOSTS = ('reddit.com', 'quora.com', 'stackexchange.com', 'dslreports.com')

# Page-type rules, tried in this order (first match wins):
#   1. host: video, forum            2. URL path: support/help, pricing/plans
#   3. URL words + title: comparison, review list, list, how-to, pricing words
#   4. shallow brand-style path: product/service   5. editorial path or long text: article
SUPPORT_HOST = re.compile(r'^(support|help|community|forums?)\.')
SUPPORT_PATH = re.compile(r'/(support|help|kb|faqs?|troubleshoot\w*|answer)(/|$)')
PRICING_PATH = re.compile(r'/(pricing|prices?|plans?|deals?|offers?|promotions?)(/|$|-)')
TEXT_RULES = [
    ('comparison', re.compile(r'(^|[\s\-_/])vs\.?($|[\s\-_/])|versus|\bcompare\b|comparison')),
    ('review list', re.compile(r'\breviews?\b|\bratings?\b')),
    ('list', re.compile(r'\bbest\b|\btop[\s\-]\d+|\b\d+[\s\-]best\b')),
    ('how-to', re.compile(r'how[\s\-]to|\bguide\b|\btips\b|step[\s\-]by[\s\-]step')),
    ('pricing/plans', re.compile(r'\bpricing\b|\bprices?\b|\bplans?\b|\bcost\b')),
]
PRODUCT_PATH = re.compile(r'^/(internet|fiber|wifi|wi-fi|products?|services?|equipment|routers?|'
                          r'residential|business|tv|mobile|home-internet)(/|$)')
ARTICLE_PATH = re.compile(r'/(blog|news|articles?|press|stories|insights?|resources?|learn)(/|$)'
                          r'|/20\d\d/')


def page_type(url, title='', word_count=0):
    host = domain_of(url)
    path = urlparse(url).path.lower()
    if host.endswith(VIDEO_HOSTS):
        return 'video'
    if host.endswith(FORUM_HOSTS) or re.match(r'^(community|forums?)\.', host) \
            or re.search(r'/(community|forums?|threads?|discussions?)/', path):
        return 'forum'
    if SUPPORT_HOST.match(host) or SUPPORT_PATH.search(path):
        return 'support/help'
    if PRICING_PATH.search(path):
        return 'pricing/plans'
    words = f'{re.sub(r"[-_/.]", " ", path)} {(title or "").lower()}'
    for label, rx in TEXT_RULES:
        if rx.search(words):
            return label
    if PRODUCT_PATH.match(path) and path.count('/') <= 2:
        return 'product/service'
    if ARTICLE_PATH.search(path) or word_count >= 600:
        return 'article'
    return 'other'


def _schema_types(html, url):
    """@type values from JSON-LD and microdata; also whether any author is declared."""
    try:
        import extruct
        data = extruct.extract(html, base_url=url, syntaxes=['json-ld', 'microdata'],
                               uniform=True, errors='ignore')
    except Exception:
        return [], False
    types, has_author = set(), False

    def walk(o):
        nonlocal has_author
        if isinstance(o, dict):
            t = o.get('@type')
            for x in (t if isinstance(t, list) else [t]):
                if isinstance(x, str):
                    types.add(x.rsplit('/', 1)[-1])
            if o.get('author'):
                has_author = True
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(data.get('json-ld', []))
    walk(data.get('microdata', []))
    return sorted(types), has_author


def _paragraphs(text):
    return [p.strip() for p in (text or '').split('\n')
            if len(p.split()) >= MIN_PARA_WORDS]


def _page_date(html, url, original):
    """original=True: first publish date; False: latest update date (htmldate)."""
    try:
        from htmldate import find_date
        return find_date(html, url=url, original_date=original, extensive_search=False,
                         outputformat='%Y-%m-%d')
    except Exception:
        return None


def parse_html(html, url):
    import trafilatura
    soup = BeautifulSoup(html, 'html.parser')
    text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True,
                               favor_recall=True) or ''
    paras = _paragraphs(text)
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else
             soup.title.get_text(' ', strip=True) if soup.title else '')

    scope = soup.find('main') or soup.find('article') or soup.body or soup
    headings = [h.get_text(' ', strip=True) for h in scope.find_all(re.compile(r'^h[1-6]$'))]

    own = registered_domain(domain_of(url))
    outbound = set()
    for a in soup.find_all('a', href=True):
        href = urljoin(url, a['href'])
        if urlparse(href).scheme in ('http', 'https'):
            host = domain_of(href)
            if host and registered_domain(host) != own:
                outbound.add(host)

    types, schema_author = _schema_types(html, url)
    meta_author = soup.find('meta', attrs={'name': re.compile('^author$', re.I)})
    has_author = bool(
        schema_author
        or (meta_author and meta_author.get('content', '').strip())
        or soup.find(attrs={'rel': 'author'})
        or soup.find(attrs={'itemprop': 'author'})
        or soup.find(class_=AUTHOR_ATTR_RE)
        or (meta and meta.author))

    head_text = ' | '.join(headings)
    top_text = text[:600]
    published = _page_date(html, url, original=True)
    updated = _page_date(html, url, original=False)
    wc = len(text.split())
    ptype = page_type(url, title, wc)
    if ptype == 'forum':
        has_author = False       # forum posts show user handles, not a page author (hand check 2026-09-27)
    return dict(
        title=title,
        paragraphs=paras,
        word_count=wc,
        outbound_links=len(outbound),
        has_faq=int(bool(FAQ_RE.search(head_text)) or 'FAQPage' in types),
        has_summary_box=int(bool(SUMMARY_RE.search(head_text) or SUMMARY_RE.search(top_text))),
        has_author=int(has_author),
        has_schema=int(bool(types)),
        schema_types=json.dumps(types),
        n_headings=len(headings),
        n_tables=len(scope.find_all('table')),
        n_lists=len(scope.find_all(['ul', 'ol'])),
        page_date=published or updated,       # publish date; update date if none found
        page_date_published=published,
        page_date_updated=updated,
        has_date=int((published or updated) is not None),
        page_type=ptype,
    )
