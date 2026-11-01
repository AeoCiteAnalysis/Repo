// Builds report/R1-R14_report_workorb.docx (second dataset, AEC proposal software).
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, HeadingLevel,
  AlignmentType, WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, Footer, PageNumber,
} = require('docx');

const ROOT = 'C:/Users/nickk/IdeaProjects/AeoCiteAnalysis';
const FIG = path.join(ROOT, 'figures', 'workorb');
const FIG_G = path.join(ROOT, 'figures');
const SIZES = JSON.parse(fs.readFileSync(path.join(__dirname, 'figsizes_workorb.json'), 'utf8'));
const CONTENT_W = 9360;           // US Letter, 1" margins (DXA)
const IMG_W = 624;                // 6.5 in at 96 px/in
const FONT = 'Calibri';

// ---------- helpers
function runs(text, base = {}) {
  // **bold** markup
  return text.split('**').map((t, i) => new TextRun({ text: t, bold: i % 2 === 1 || base.bold, italics: base.italics, size: base.size, color: base.color, font: FONT }));
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts), spacing: { after: 120, line: 276 }, alignment: opts.align });
const H1 = (t) => new Paragraph({ keepNext: true, heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })], spacing: { before: 360, after: 160 } });
const H2 = (t) => new Paragraph({ keepNext: true, heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })], spacing: { before: 280, after: 120 } });
const H3 = (t) => new Paragraph({ keepNext: true, heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })], spacing: { before: 200, after: 80 } });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: 'bullets', level }, children: runs(text), spacing: { after: 60, line: 276 } });
const N = (text, instance = 1) => new Paragraph({ numbering: { reference: 'numbers', level: 0, instance }, children: runs(text), spacing: { after: 80, line: 276 } });
const label = (name, text) => new Paragraph({ children: [new TextRun({ text: name + ' ', bold: true, color: '1F4E79', font: FONT }), ...runs(text)], spacing: { after: 100, line: 276 } });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

let figNo = 0;
function figure(file, caption, widthFrac = 1) {
  const [w, h] = SIZES[file];
  const width = Math.round(IMG_W * widthFrac);
  figNo += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
      children: [new ImageRun({ type: 'png', data: fs.readFileSync(path.join(FIG, file)),
        transformation: { width, height: Math.round(width * h / w) },
        altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${figNo}. `, bold: true, size: 18, font: FONT }), new TextRun({ text: caption, italics: true, size: 18, font: FONT })] }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head, shade) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: 'DCE6F2', type: ShadingType.CLEAR } : shade ? { fill: 'F5F8FC', type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(t), { bold: head, size: 18 }) })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }),
      ...rows.map((r, k) => new TableRow({ children: r.map((t, i) => cell(t, i, false, k % 2 === 1)) }))],
  });
}
const spacer = () => new Paragraph({ children: [], spacing: { after: 120 } });

function experiment(id, title, question, did, found, conclude, figs = [], extra = []) {
  return [
    H2(`${id} – ${title}`),
    label('Question.', question),
    label('What we did.', did),
    label('What we found.', ''),
    ...found.map((f) => B(f)),
    ...extra,
    ...figs.flatMap(([f, c, w]) => figure(f, c, w || 1)),
    label('What we can conclude.', conclude),
  ];
}

// ---------- content (second dataset: Workorb, AEC proposal software)
const children = [];
const cmp = (text) => label('Compared with GFiber.', text);
function exp(id, title, question, found, conclude, gf, figs = [], extra = []) {
  return [
    H2(`${id} – ${title}`),
    label('Question.', question),
    label('What we found.', ''),
    ...found.map((f) => B(f)),
    ...extra,
    ...figs.flatMap(([f, c, w]) => figure(f, c, w || 1)),
    label('What we can conclude.', conclude),
    cmp(gf),
  ];
}

children.push(
  new Paragraph({ spacing: { before: 2400, after: 200 }, children: [new TextRun({ text: 'What Makes AI Engines Cite a Page?', bold: true, size: 52, font: FONT, color: '1F4E79' })] }),
  new Paragraph({ spacing: { after: 400 }, children: [new TextRun({ text: 'Second dataset: niche B2B software (AEC proposal software) — experiments R1–R14 and comparison with the first dataset (home internet)', size: 30, font: FONT, color: '404040' })] }),
  P('Data: daily answers from ChatGPT, Gemini and Perplexity to 55 prompts about proposal, RFP and business-development software for architecture, engineering and construction (AEC) firms (client: Workorb), 24 June – 21 September 2026. Same collection system, engine models and settings as the first dataset (GFiber, 651 prompts about US home internet).'),
  P('Project: AeoCiteAnalysis (AEO_DATASET=workorb). Report date: 8 October 2026 (numbers after the data freeze: replay 27 September – 7 October). Speed (part of R8) could not be tested; see R8.'),
  new Paragraph({ spacing: { before: 600 }, children: runs('**Contents.** Summary · 1 The two datasets · 2 Results R1–R14 · 3 The "is it real?" table · 4 What the two datasets show together · 5 Retrieval and selection · 6 Limits') }),
  pageBreak(),
);

children.push(
  H1('Summary'),
  P('We repeated the core experiments R1–R14 on a second, very different market: a crowded niche of B2B software vendors, most of them small and little known, instead of a handful of large consumer internet brands. The same scripts, tests and "is it real?" rule were used. The main results:'),
  N('**Question matching replicates, and is stronger.** A title that matches the questions (+26 % more citations per SD, +14 % in GFiber) and a best-matching paragraph (+13 %, GFiber +16 %) pass all five tests again. The title effect now also passes on brand pages (+34 %), where nothing passed in GFiber.', 3),
  N('**The popular tips fail again.** Pages with FAQ blocks (+23 % pooled), summary boxes (+22 %) and author boxes (+17 %) look better across websites, and the difference vanishes within websites — the same pattern as GFiber (R8). Structured data has no significant effect either way.', 3),
  N('**The website and its authority matter much less.** Only 9 % of the variation lies between websites (GFiber 13 %); outside authority has no page-level effect (−1 %, GFiber +11 %); the page is more informative than the website (R7, R13).', 3),
  N('**Page features explain more — but most variation is still unexplained.** They explain 11 % of the within-website variation (GFiber 4 %); counts are again 90 % repeatable, so about 71 % of all variation is stable but unexplained.', 3),
  N('**The citation landscape is different.** Gemini gives 84 % of the citations (12 unique pages per answer); ChatGPT cites in only 35 % of answers (GFiber 69 %); vendor websites receive about 45 % of citations in every engine; the top 10 websites hold 22 % of citations (GFiber 40 %); YouTube, not Reddit, is the largest platform.', 3),
);
children.push(spacer(), table(
  ['Result (per SD; yes/no features: has vs not)', 'GFiber (home internet)', 'Workorb (AEC software)'],
  [
    ['Title matches a prompt (within websites, per SD)', '+14 % real', '**+26 % real**'],
    ['Best-matching paragraph', '+16 % real', '+13 % real'],
    ['Word overlap with prompts', '+9 % real', '+13 % (4 of 5 tests)'],
    ['Title match on brand pages', '+7 % (not real)', '**+34 % real**'],
    ['FAQ / author / summary box / schema', 'vanish within websites', 'vanish within websites'],
    ['Headings', 'no effect', '+8 % (4 of 5 tests)'],
    ['Outside authority, page level', '+11 % per SD', '**no effect (−1 %)**'],
    ['Variation between websites', '13 %', '9 %'],
    ['Within-website variation explained by page features', '4 %', '11 %'],
    ['Repeatable (split-half reliability)', '90 %', '90 %'],
  ], [4200, 2580, 2580]));
children.push(pageBreak());

children.push(
  H1('1  The two datasets'),
  P('Both datasets come from the same collection system with identical settings: each prompt is sent daily to the engines’ APIs (ChatGPT = GPT-5.4 nano, Gemini 3.1 Flash Lite, Perplexity Sonar) with web search on, and the answer with its links is stored.'),
  table(['', 'GFiber', 'Workorb'], [
    ['Market', 'US home internet (consumer)', 'Proposal / RFP / BD software for AEC firms (B2B niche)'],
    ['Prompts', '651 (7 topics)', '55 (6 topics)'],
    ['Days', '90 (24 Jun – 21 Sep)', '90 (same days)'],
    ['Answers', '143,249 (Gemini partial until 25 July)', '14,850 (all engines, every day)'],
    ['Citations (URL once per answer)', '396,188', '71,849'],
    ['Distinct URLs / websites', '39,043 / 7,234', '9,258 / 3,895'],
    ['Tracked brands', 'Google Fiber + 9 ISPs', 'Workorb + Loopio, Unanet, OpenAsset, Joist AI'],
    ['Competitor websites', '~20 (ISPs, dealers)', '222 (all vendors in the category; LLM-suggested, reviewed; Joist AI = Kantiv)'],
    ['Pages downloaded', '63 %', '77 %'],
    ['Pages in the within-website models (R2)', '12,038 on 1,120 websites', '2,477 on 515 websites'],
    ['Prompt intent', 'learning / comparing / ready to buy', '52 learning, 3 comparing, 0 ready to buy'],
  ], [2600, 3000, 3760]),
  spacer(),
  P('**Consequences for comparison.** With 55 prompts and 2,477 pages in the models, Workorb tests have much less power: effects of about +5 % that are detectable in GFiber are not in Workorb. Most Workorb websites have only one cited page, so the within-website comparisons rest on 515 websites. Website roles were suggested by an LLM (classifying the owner of each website with ≥ 10 citations) and reviewed.'),
  pageBreak(),
);

children.push(H1('2  Results by experiment'));
children.push(...exp('R1', 'Describe the citations',
  'How many sources does each engine cite, and how concentrated are they?',
  [
    '**Gemini dominates:** 84 % of citations, 12.2 unique pages per answer; ChatGPT 2.1; Perplexity 0.2.',
    '**Many answers cite nothing:** ChatGPT 65 %, Gemini 18 %, Perplexity 97 % of answers have no citation.',
    '**A long tail of websites:** the top 10 websites hold 22 % of citations and 448 websites are needed for 80 %. The most-cited websites are vendors (Flowcase, Unanet, Workorb, OpenAsset, Responsive, Loopio) and YouTube (3.5 %); Reddit has only 0.7 %.',
  ],
  'In a niche B2B market the engines draw on many small vendor websites and videos rather than on a few large brands and forums.',
  'GFiber was far more concentrated (top 10 websites 40 %, 167 websites for 80 %), led by Reddit (10 %) and brand websites; ChatGPT cited in 69 % of answers there against 35 % here.',
  [['r01_top_websites_by_engine.png', 'Most-cited websites per engine (own = Workorb, competitor = vendors in the category).']]));

children.push(...exp('R2', 'Main model: which page features go with more citations?',
  'Comparing pages on the same website, which features go with more citations?',
  [
    '**Title matches a prompt: +26 % per SD** (brand pages +34 %, third party +18 %).',
    '**Word overlap with prompts +13 %, best-matching paragraph +13 %, headings +8 %** (all pages).',
    'FAQ, summary box, author, structured data, length, lists, tables and links: no significant effect.',
    'Page types (vs articles): review lists +150 %, lists +48 %, forum pages +149 % (only 13 forum pages).',
  ],
  'The same three question-matching features lead, and the title effect is about twice as large as in GFiber.',
  'Same ranking of features; title +14 %, best paragraph +16 %, word overlap +9 % in GFiber. Headings are new here (GFiber −3 %, not significant).',
  [['r02_effects.png', 'Within-website effects per SD with 95 % intervals, by sample.']]));

children.push(...exp('R3', 'Many tests at once (false discovery rate)',
  'Which effects survive correction for testing 14 features in 3 samples?',
  [
    'All pages: title (q < .001), headings (q < .001), word overlap (q = .009) and best paragraph (q = .013) survive.',
    'Brand pages: only title survives (q < .001). Third party: title and headings; best paragraph and word overlap borderline (q = .07–.10).',
  ],
  'With fewer pages, only the strongest effects survive the correction outside the all-pages sample.',
  'In GFiber five features survived on all pages, including first-paragraph match and page age; neither does here.'));

children.push(...exp('R4', 'Stricter selection (stability selection, 1-SE penalty)',
  'Which features are kept in at least 60 % of 200 resamples under the stricter penalty?',
  [
    'Kept: **title** (98–100 %) and **best paragraph** (94–100 %) in every sample.',
    'Not kept: word overlap (29–46 %), headings (28–50 %) and all tips (0–12 %).',
  ],
  'Title and best paragraph are the robust core; word overlap loses to them here (they overlap in what they measure).',
  'In GFiber word overlap was kept on third-party pages and narrowly missed on all pages (58.5 % of runs); the tips were dropped in both.',
  [['r04_selection_rates.png', 'Share of resamples in which each feature is kept.']]));

children.push(...exp('R5', 'Double machine learning',
  'Do the effects survive a flexible model of all other features?',
  [
    'Title +28 %, best paragraph +16 % (brand pages +24 %), word overlap +13 %, headings +8 %, word count +6 % per SD — all significant on all pages.',
    'Pages with FAQ blocks are cited **less** (−14 % with vs without, significant).',
  ],
  'The linear-model results are not an artefact of the model form.',
  'Same conclusion as GFiber; there the FAQ estimate was also negative (−5 % with vs without) but not significant.',
  [['r05_dml_vs_r2.png', 'R2 (linear) vs double-ML estimates.']]));

children.push(...exp('R6', 'Shape of each effect',
  'Is the relation steady, or does it have a sweet spot?',
  [
    'Title, best paragraph and word overlap: **always up** in every sample. Headings: always up on all and third-party pages.',
    'Length, lists, tables, links, page age: flat — no sweet spot.',
  ],
  'More matching is better, with no ceiling in the observed range.',
  'Same shapes as GFiber; there page age went steadily down, here it is flat.',
  [['r06_shapes.png', 'Partial-effect curves with 95 % bands (other features held fixed).']]));

children.push(...exp('R7', 'Importance: page vs website',
  'In a flexible model, how important is the website compared with page features?',
  [
    'Cross-validated R² 0.09. **Title match is the most important variable (0.24)**, ahead of the website score (0.16), page type and best paragraph.',
    'Website importance is 0.64 × that of the strongest page feature; outside authority only 0.31 ×.',
  ],
  'In this market the page matters more than the website it sits on.',
  '**Reversed from GFiber**, where the website was about twice as important as the strongest page feature.',
  [['r07_importance.png', 'SHAP importance (mean |value|), three model versions.']]));

children.push(...exp('R8', 'The popular tips: pooled vs within websites',
  'Do FAQ blocks, structured data, summary boxes, author boxes and speed go with more citations?',
  [
    '**Pooled across websites, pages with the tip get more citations: FAQ +23 %, summary box +22 %, author +17 %** (significant).',
    'Within websites: FAQ −2 %, summary box +8 %, author +2 % — **none significant**. Structured data: not significant either way (+16 % within websites, wide interval).',
    '**Speed could not be tested:** only 22 pages (6 in the model) have their own Chrome UX data — niche websites have too little traffic for Google to measure single pages.',
  ],
  'The tips are markers of websites that are cited more, not causes of citation. This is the clearest replication across the two datasets.',
  'Identical pattern in GFiber (pooled FAQ +18 %, author +15 %, summary box +21 %; all vanish within websites). In GFiber speed was testable on 466 pages and also vanished.',
  [['r08_before_after.png', 'Tip effects pooled (alone and with other features) vs within websites.']]));

children.push(...exp('R9', 'Robustness checks',
  'Do the effects keep their sign across time periods, engines, samples and model variants?',
  [
    'Title, best paragraph, word overlap and headings keep their sign in all five robustness checks and when any one of the eight most-cited websites is left out.',
    '**By engine the effects come from Gemini:** title +32 %, word overlap +14 %, best paragraph +14 % for Gemini; ChatGPT and Perplexity near zero.',
  ],
  'Robust — but in this dataset the page-matching effects describe Gemini, which gives 84 % of the citations.',
  'In GFiber the best-paragraph effect held in all three engines (ChatGPT +9.5 %, Gemini +7.8 %, Perplexity +4.8 %), while title match went with more citations from Gemini (+16.8 %) and Perplexity (+4.7 %) but not from ChatGPT (−0.2 %).',
  [['r09_robustness.png', 'Effects across robustness checks.']]));

children.push(...exp('R10', 'Where on the page is the best-matching paragraph?',
  'Do engines cite pages whose answer sits near the top?',
  [
    'The best-matching paragraph sits in the top quarter of the page on 41 % of cited pages (bottom quarter 24 %; median relative depth 0.37).',
    'Within websites, depth has no effect on citations (−2.5 % per SD, p = .31).',
  ],
  'Answers tend to be near the top of cited pages, but a deeper answer is not cited less.',
  'Same conclusion as GFiber; the answers there sat a little lower (median depth 0.45, top quarter 32 %).',
  [['r10_depth_hist.png', 'Relative position of the best-matching paragraph on cited pages.']]));

children.push(...exp('R11', 'The engines compared',
  'Do the engines cite different kinds of sources and pages of different age?',
  [
    'Vendor (own + competitor) share: ChatGPT 51 %, Gemini 45 %, Perplexity 45 %; Workorb itself 5.1 % in ChatGPT, 1.6 % in Gemini, 0 % in Perplexity.',
    '**ChatGPT cites the freshest pages:** median age 137 days (Gemini 212, Perplexity 227); 38 % under 3 months (Gemini 25 %).',
    'Platforms: YouTube 4.1 % of Gemini’s citations; Reddit 4.9 % of Perplexity’s; Wikipedia 1.4 % of ChatGPT’s.',
  ],
  'ChatGPT again prefers vendors and fresh pages; here Gemini and Perplexity also give vendors a large share, because in a niche market vendors write most of the content.',
  'In GFiber brand websites were 64 % of ChatGPT’s citations but only 17 % of Gemini’s; ChatGPT was also the freshest (median 201 days).',
  [['r11_brand_share.png', 'Share of citations by website role and engine.'], ['r11_freshness.png', 'Age of cited pages by engine.']]));

children.push(...exp('R12', 'Page types and topics',
  'Do effects differ by page type, topic and intent?',
  [
    'Page types vs articles: review lists +150 %, lists +48 % (significant); forum pages +149 % (13 pages only).',
    'Title match is positive in every topic (+1 % to +17 %) and significant in five of six, strongest for Proposal Automation (+17 %).',
    'Intent cannot be compared: 52 of 55 prompts are "learning".',
  ],
  '"Best tools" lists and review lists are the page types engines favour in a software market.',
  'In GFiber forum pages (+78 %) and lists (+13 %) led; the list effect is much stronger here.',
  [['r12_page_types.png', 'Page types compared with articles (same website, other features held fixed).'], ['r12_heatmap.png', 'Feature effects by topic and intent.']]));

children.push(...exp('R13', 'Website vs page: how much can we explain?',
  'How much of the variation lies between websites, how much can page features explain, and how much is repeatable?',
  [
    '**Only 9 % of the variation is between websites; 91 % is within them.** Page features explain 11 % of the within-website part.',
    'Split-half reliability 0.90 (within websites 0.90): the counts are as repeatable as in GFiber.',
    'On unseen pages the website alone predicts nothing (cross-validated R² < 0 for websites with ≥ 2 pages); page features alone predict 12 %.',
    'So about 71 % of all variation is stable but unexplained (GFiber 75 %).',
  ],
  'The page carries more information than the website in this market, and page features explain three times more than in GFiber — yet most stable variation remains unexplained, as in GFiber.',
  'GFiber: 13 % between websites, 3 % of the within part explained, website alone 4 % on unseen pages.',
  [['r13_components.png', 'Where the variation lies, and prediction of unseen pages by website size.']]));

children.push(...exp('R14', 'Outside authority',
  'Do pages on high-authority websites get more citations?',
  [
    '**No:** −1 % per SD (p = .55) at page level, adding question matching (title +18 %, word overlap +15 %).',
    'By engine: ChatGPT +6 % (p = .03), Gemini −7 % (p = .03).',
    'The over-performers are vendors: Flowcase, Unanet, Loopio, Workorb, OpenAsset, Responsive, AutogenAI — all cited far more than their authority predicts.',
  ],
  'In a niche market authority does not help; being a relevant vendor with matching content does. Only ChatGPT shows a small authority preference, as in GFiber.',
  'GFiber: +11 % per SD overall, +30 % for ChatGPT; authority mattered, though less than the website’s own citation level.',
  [['r14_authority.png', 'Authority vs citations.']]));
children.push(pageBreak());

children.push(
  H1('3  The "is it real?" table'),
  P('A feature is real only if it passes all five tests: significance after correction (R3), stricter selection (R4), double ML (R5), a consistent shape (R6) and robustness (R9).'),
  table(['Feature', 'All pages', 'Brand pages', 'Third party', 'GFiber (all / brand / third)'], [
    ['Title matches a prompt', '**+26 % real**', '**+33 % real**', '**+18 % real**', 'real / no / real'],
    ['Best-matching paragraph', '**+13 % real**', '+12 % (3/5)', '+15 % (4/5)', 'real / no / real'],
    ['Word overlap with prompts', '+13 % (4/5)', '+15 % (2/5)', '+10 % (3/5)', 'no / no / real'],
    ['Headings', '+8 % (4/5)', '+9 % (2/5)', '+6 % (4/5)', 'no'],
    ['FAQ, schema, summary box, author', 'no', 'no', 'no', 'no'],
    ['Length, lists, tables, links, page age, first paragraph', 'no', 'no', 'no', 'no (first paragraph, age 4/5)'],
  ], [2700, 1400, 1400, 1400, 2460]),
  spacer(),
  pageBreak(),
);

children.push(
  H1('4  What the two datasets show together'),
  P('The two markets differ in almost every descriptive respect — size of brands, concentration of sources, the role of Reddit and YouTube, how often ChatGPT cites. Against that background, the findings that hold in both are strong evidence:'),
  N('**Question matching is the page property that goes with citations** — in the title above all, and in a paragraph that answers the question. It passes all tests in both markets and is stronger in the niche market (+26 % vs +14 % for the title).', 4),
  N('**FAQ blocks, author boxes, summary boxes and structured data do not go with more citations** once pages are compared within their website. They look helpful only because websites that use them are cited more anyway. This replicates exactly.', 4),
  N('**Most stable variation is unexplained by page features in both markets** (75 % and 71 %), with the same 90 % repeatability.', 4),
  P('What differs is **how much the website matters**. In the consumer market with large brands the website and its authority carry a large part of the explanation; in the niche market authority has no effect and the page matters more than the website. A practical reading: for small, little-known brands, page content that matches the questions is relatively more important, because there is no authority advantage to lean on.'),
  P('**For the paper:** the two datasets turn a single case into a comparison of a consumer and a B2B-niche market. The replicated findings (question matching; the tips vanishing within websites; the size of the unexplained part; citation following the engine’s own ranking, section 5) can be reported as holding across markets; the website/authority difference is a new, market-dependent finding.'),
);

children.push(
  H1('5  Retrieval and selection: replaying ChatGPT'),
  P('The first dataset’s follow-up showed that citations are decided mostly in the engines’ own search step (report 1, section 6). We replayed the 55 Workorb questions through the same APIs three times on 27 September and then daily until 7 October (13 ChatGPT runs over 11 days), keeping ChatGPT’s search queries and every page it consulted, and repeated the analyses (X3, X5, X6).'),
  B('**ChatGPT searched for only 17 of the 55 questions** (for the others it answered without sources, matching the 65 % of collected answers without citations). When it searched it consulted a median of 29 pages and cited 5.'),
  B('**Citation follows ChatGPT’s own ranking:** 57 % [51, 63] of rank-1 pages are cited, 8 % [7, 10] of pages at ranks 11–20 and 1 % below (231 result lists); every page it cites is one its searches returned. Pages found by several of its searches have much higher odds.'),
  B('**Every answer starts from new queries:** the same question asked twice on the same day shares 2.4 % of its search queries and about 16 % of its consulted pages.'),
  B('**Retrieval explains the citation counts better than page features:** 30 % [15, 43] of the within-website variation in ChatGPT’s 90-day counts against 10 % for page features — but with 629 pages on 130 websites the difference (+20 points [−1, 31]) is still just short of significance. Shuffling the retrieval data within websites removes it (0.7 %, p = .005).'),
  B('**Drift:** the replay explains 7 % of late June’s citations and 30 % of the last days, as in the first dataset.'),
  B('**Query rewording (X7):** ChatGPT again splits each question into about four keyword queries (37 % of the question’s words kept, 51 % new), adds category terms (construction, RFP, bid, platform), and searches a second round naming vendors found in the first results — e.g. "proposal automation tool AI beyond RAG", then "Loopio AI agent beyond retrieval RFP compliance score", "Tribble governed AI proposals citations reviewer control". Unlike in the consumer market it rarely adds a year (0 %) or a brand the question did not name (3 %). Gemini writes natural-language sub-questions ("is OpenAsset good for architecture firms", "does OpenAsset have a tool called Shred.ai") and repeats 14 % of its queries between runs; ChatGPT 4 %.'),
  B('**Selection at equal rank (X6, 231 result lists):** rank explains 29 % of which consulted pages are cited (34 % with flags for pages without text); question matching adds nothing measurable; page features add 7 points — shorter pages (×0.32 per SD of length), brand websites (×1.80), structured data (×1.41), and list (×0.20) and comparison pages (×0.14) losing at equal rank. Reddit was consulted 1,617 times and cited twice. Matching ChatGPT’s own search queries is the content feature that goes with a higher rank — by meaning, not by shared keywords (X6b: correlation with rank 0.39 for meaning vs 0.32 for BM25; with both included, meaning has the larger effect, +0.28 vs +0.14 per SD).'),
  B('**Links to the page (X9):** within websites, pages with more referring websites are cited more (+16 % per SD; +41 % for any link vs none), concentrated on brand pages (+42 %; third-party pages +2 %, not significant) and strongest for Gemini (+17 %; ChatGPT +8 %) — the same direction as in the consumer market (+23 %).'),
  B('**Deeper in Google and Bing? (X8):** for all of ChatGPT’s queries, 99 % of its retrieved pages are absent from Google’s top 50 and 94 % from Bing’s top 30; 92 % of Gemini’s cited pages are absent from Google’s top 50. For 44 % of the retrieved pages the website does appear — the engines select different pages, not deeper ones, as in the consumer market.'),
  B('**Brand mentions (X11):** 46 % (ChatGPT) and 58 % (Gemini) of the mentions of a brand the question does not name are backed by its own cited page, 12 % and 22 % by a third-party page about it, and 42 % and 20 % by no cited page. A cited third-party page about the brand raises the odds of a mention ×40 (ChatGPT) and ×18 (Gemini) — far more than in the consumer market (×4, ×6), because the models do not already know these brands. ChatGPT names a niche brand in only 4 % of possible cases, Gemini in 16 %.'),
  B('**The models’ prior (X10):** asked without search, gpt-5.4-nano (ChatGPT) almost never names the tracked vendors (Unanet 4 %, Loopio 16 %, OpenAsset 2 %) and lists general construction software (Autodesk, Procore, Oracle) instead; Gemini 3.1 Flash Lite knows them well (96 %, 78 %, 60 %). Neither model knows Workorb or Joist AI (0–2 %). Still, 32 % of ChatGPT’s citations come from websites its model names without searching (9 % with another question’s prior), and known websites are preferred at the same rank. For a brand the models do not know, visibility depends entirely on being found by search.'),
  B('**Position in the list (X12):** 20 Workorb questions were part of the position experiment (8 real replay sources each, shown in all 8 orders, two models without search; 1,920 answers). As in the consumer market, ChatGPT’s model (gpt-5.4-nano) cites the same source more often when it is shown early: 83 % in the first slot and 68 % in the last, **+15.9 points** [11.2, 20.3] (consumer market +18.0). Gemini’s model shows no position effect: 77–80 % in every slot (+1.2 points [−3.1, 5.6]). For ChatGPT it is the slot, not a rank label: being placed in the first two slots adds 10 points [7, 13], carrying one of the top two labels 5 [1, 9]. Sources the production engine actually cited are cited more (ChatGPT 85 % vs 63 %; Gemini 84 % vs 73 %); with only 20 questions the engine’s full 1–8 ordering does not show a clear gradient. Both models cite more of the sources here (74–78 % of them) than in the consumer market (50–58 %). Brand recommendations do not depend on the order: across the 8 orders the first tracked vendor named is the same in 97 % of answers.'),
  table(['Cited when shown in slot (Workorb questions)', '1', '2', '3', '4', '5', '6', '7', '8'], [
    ['ChatGPT (gpt-5.4-nano)', '83 %', '84 %', '78 %', '74 %', '69 %', '69 %', '66 %', '68 %'],
    ['Gemini (3.1-flash-lite)', '80 %', '77 %', '78 %', '78 %', '77 %', '78 %', '77 %', '78 %'],
  ], [2960, 800, 800, 800, 800, 800, 800, 800, 800]),
  spacer(),
  B('**Brand and publisher names (X14):** 20 of the 60 packets of the identity experiment are niche-market questions (five of the ten vendors per packet: Workorb, Loopio, Unanet, OpenAsset, Joist AI, Deltek, Responsive, AutoGenAI, Inventive AI, Arphie), each described by a fixed fact sheet and one third-party excerpt, with brand and publisher names hidden or shown independently (2×2). Hidden, the unknown vendors cannot be recognised by their facts (Workorb, Arphie, AutoGenAI, Responsive: 0 %), while Deltek (Gemini 100 %), OpenAsset and Loopio are. Showing the names reorders the ranking: Loopio gains 16 points in top-three rate with both models and Unanet 11 (ChatGPT), Deltek 8 and Workorb 6–8 (not significant), while Inventive AI loses 16–19 and Arphie 13 (ChatGPT). With the names assigned at random to the same fact sheets and the model told they carry no information, ChatGPT still puts the sheet in its top three in 74 % of answers under the name Loopio and 44 % under Arphie (p < .001); Gemini shows no name effect for these vendors. Publisher names matter little. For a vendor the models do not know, the name is neutral at best; the vendors that gain are those the models already know.'),
  ...figure('x14_identity.png', 'Left: change in a vendor’s top-three rate when its name is shown (same facts) against its name effect with random names; niche vendors highlighted, consumer brands faded. Right: how much showing brand or publisher names changes top-three rates, relative to repeating the same prompt (both markets).'),
  table(['Change in top-3 rate when the name is shown (points, same facts)', 'ChatGPT', 'Gemini'], [
    ['Loopio', '+16 [1, 35]', '+16 [3, 33]'],
    ['Unanet', '+11 [1, 25]', '−1 (n.s.)'],
    ['Deltek', '+8 (n.s.)', '0 (n.s.)'],
    ['Workorb', '+6 (n.s.)', '+8 (n.s.)'],
    ['OpenAsset / Joist AI / Responsive', '−1 / −1 / −1 (n.s.)', '−1 / −1 / −6 (n.s.)'],
    ['AutoGenAI', '−6 (n.s.)', '+1 (n.s.)'],
    ['Arphie', '−13 [−25, −3]', '+1 (n.s.)'],
    ['Inventive AI', '−19 [−31, −5]', '−16 [−31, −3]'],
  ], [4960, 2200, 2200]),
  spacer(),
  B('**Brand order in real answers (X13):** for the niche vendors, few answers name two tracked vendors together (4,593 pairs of vendors in 1,338 answers; only 6 pairs with enough answers for ChatGPT, 44 for Gemini, 38 for Perplexity), so these results are thinner than in the consumer market. The pattern is the same: the same vendor is named first in 70–80 % of the answers naming a pair, there is one majority cycle (Gemini) and it does not survive resampling, and one score per vendor and question predicts held-out days best — 66 % (Gemini), 79 % (ChatGPT) and 78 % (Perplexity), against 57 %, 71 % and 68 % for one score per vendor for all questions. Perplexity is again the least stable over time (20 % of pairs change significantly between the two halves of the 90 days; Gemini 6 %).'),
  ...figure('x13_rank_models.png', 'Held-out pairs of vendors whose order (first mention) is predicted correctly, by the level at which the vendor score is fitted (niche market).'),
  B('**Controlled ranking stability (X15):** 8 of the 20 questions of the controlled experiment are niche-market questions, with six vendors each described by their fixed fact sheets. As in the consumer market, pairwise choices are consistent (majority share 90 % ChatGPT, 92 % Gemini) and transitive (no robust cycle), one score per vendor fitted on the pairs predicts the full six-vendor rankings as well as another full ranking does (ChatGPT 77 % vs 75 %; Gemini 80 % vs 80 %), and — pooled over both markets — removing or adding a product hardly changes the order of the others. The order in which vendors are listed matters more here for ChatGPT: the vendor listed first reaches an average rank of 2.6, the one listed last 4.0 (consumer market 3.3 and 3.7). For a niche vendor, a place near the top of the engine’s list of candidates is therefore worth more.'),
  P('**How to read “one score per vendor predicts the rankings” (X15).** For each question the model chose between every pair of the six vendors, in both orders and three times; from these answers alone we fit one strength score per vendor (a Bradley–Terry model), which implies a full ranking. Separately, the model ranked all six at once, twelve times, with the vendors listed in different orders. A ranking of six contains 15 pairs; the scores place 77 % (ChatGPT) and 80 % (Gemini) of them the right way round in the niche questions. The reference is the model itself: one full ranking agrees with another shown in a different order on 75 % and 80 % of the pairs — about the best any prediction can do. Because the single score reaches that level, one number per vendor and question loses essentially nothing; what remains is the model’s own inconsistency, mostly from the listing order. It holds per question only: a vendor’s score on one question says little about the next (X13).'),
  ...figure('x15_stability.png', 'Controlled stability, both markets: agreement of two rankings under repetition, rewording and a different listing order (left), and how well one score per brand predicts the full rankings (right).'),
  B('**Ranking stability with web search (X16):** the controlled stability experiment was repeated with the model searching the web for its own evidence, and with no evidence at all (memory only). For the niche vendors the contrast is sharp: from memory alone, the models rank them inconsistently (ChatGPT: two identical prompts agree with τ 0.36, and one score per vendor predicts only 64 % of pairs; Gemini 67 %), because they barely know them. With search, the rankings become consistent (ChatGPT τ 0.59 and 80 %; Gemini 80 %). For the large consumer brands it is the other way round: memory alone is already consistent and search adds some noise. For a vendor like Workorb, a stable place in AI rankings therefore depends on search finding evidence about it, not on what the model remembers.'),
  ...figure('x16_search.png', 'Ranking stability with fixed facts, web search and memory only (both markets): agreement of two identical prompts (left) and pairs predicted by one score per brand (right).'),
  ...figure('x06_selection.png', 'Left: share of the choice among consulted pages explained as feature groups are added. Right: % cited by within-list match to the question, at equal rank.'),
  table(['Replay result (ChatGPT)', 'GFiber', 'Workorb'], [
    ['Questions with a search', '448 of 651', '17 of 55'],
    ['Pages consulted / cited per question', '26 / 3', '29 / 5'],
    ['Rank 1 cited', '44 %', '57 %'],
    ['Ranks 11–20 cited', '4 %', '8 %'],
    ['Same search queries, same question asked twice', '2.7 %', '2.4 %'],
    ['Query adds a year / a brand not in the question (X7)', '23 % / 28 %', '1 % / 3 %'],
    ['ChatGPT citations from websites its model names without search (X10)', '47 %', '32 %'],
    ['ChatGPT retrieved pages absent from Google top 50 (X8)', '97 %', '99 %'],
    ['A cited third-party page about the brand: odds of a mention, ChatGPT / Gemini (X11)', '×4 / ×6', '×40 / ×18'],
    ['Same source first vs last slot, ChatGPT / Gemini model (X12)', '+18.0 / −0.8 points', '+15.9 / +1.2 points'],
    ['Spread in top-3 rate between brand names on the same facts, ChatGPT model (X14)', '41 points', '30 points'],
    ['Held-out brand order predicted by one score per brand and question, ChatGPT / Gemini / Perplexity (X13)', '74 / 77 / 79 %', '79 / 66 / 78 %'],
    ['Controlled: product listed first vs last, average rank, ChatGPT model (X15)', '3.3 vs 3.7', '2.6 vs 4.0'],
    ['Rankings from memory only vs with web search, ChatGPT: identical prompts agree (τ) (X16)', '0.61 vs 0.54', '0.36 vs 0.59'],
    ['Tracked brands known to gpt-5.4-nano / Gemini (X10)', '61–72 % / 63–78 % (large ISPs)', '0–16 % / 0–96 %'],
    ['Within-website variation: retrieval vs page features', '32 % vs 3 % (difference significant)', '30 % vs 10 % (just short of significance)'],
    ['Choice explained by rank / + match / + page features', '29 % / 30 % / 32.5 %', '34 % / 34 % / 40 %'],
    ['List and comparison pages at equal rank', '×0.54 / ×0.34', '×0.20 / ×0.14'],
  ], [4360, 2500, 2500]),
  spacer(),
  P('**Conclusion.** The retrieval mechanism replicates in the niche market: ChatGPT rewrites each question, cites from the top of its own ranking, and content matters mainly by matching the queries it actually sends. At the citation step, concise pages from the original source win over roundups in both markets, and ChatGPT’s model favours whatever is shown first in both (X12).'),
  pageBreak(),
);

children.push(
  H1('6  Limits'),
  B('**Small prompt set.** 55 prompts, and 2,477 pages on 515 websites in the models; effects around ±5 % cannot be detected. Results for brand pages rest on 1,065 pages.'),
  B('**Gemini-driven.** Gemini gives 84 % of citations, and the page-feature effects come from Gemini (R9).'),
  B('**Website roles** were suggested by an LLM (owner of each website with ≥ 10 citations) and reviewed; competitor = any vendor in the proposal / RFP / AEC business-development / AEC asset-management category (222 websites). Correction (28 Sep): Joist AI has rebranded as Kantiv, so kantiv.com is the competitor Joist AI (it was first labelled third party), and the unrelated contractor app Joist is third party; all Workorb results were rerun — only the brand-page and third-party samples moved slightly (e.g. title on brand pages +33 % → +34 %), no verdict on all pages changed.'),
  B('**Speed not tested** (22 pages with page-level Chrome UX data). **Intent** not comparable (52 of 55 prompts are "learning").'),
  B('**Gemini redirect links:** 202 of 861 resolved (older links expire); unresolved citations count for websites but not for pages.'),
  B('Scripts are the same as for the first dataset (AEO_DATASET=workorb); results in results/workorb/, figures in figures/workorb/, choices in decisions.md.'),
);

// ---------- document
const doc = new Document({
  creator: 'AeoCiteAnalysis', title: 'What Makes AI Engines Cite a Page? Second dataset (Workorb)',
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: '1F4E79' }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: '2E75B6' }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 22, bold: true, font: FONT }, paragraph: { outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: 'numbers', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: 'What makes AI engines cite a page? · second dataset (AEC proposal software) · page ', size: 16, color: '808080', font: FONT }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: '808080', font: FONT })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, 'report', 'R1-R14_report_workorb.docx');
  fs.writeFileSync(out, buf);
  console.log('wrote', out, (buf.length / 1e6).toFixed(1), 'MB,', figNo, 'figures');
});
