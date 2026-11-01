// Builds report/R1-R14_report.docx for the AeoCiteAnalysis project.
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, HeadingLevel,
  AlignmentType, WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, Footer, PageNumber,
} = require('docx');

const ROOT = 'C:/Users/nickk/IdeaProjects/AeoCiteAnalysis';
const FIG = path.join(ROOT, 'figures');
const SIZES = JSON.parse(fs.readFileSync(path.join(__dirname, 'figsizes.json'), 'utf8'));
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

// ---------- content
const children = [];

// Title page
children.push(
  new Paragraph({ spacing: { before: 2400, after: 200 }, alignment: AlignmentType.LEFT,
    children: [new TextRun({ text: 'What Makes AI Engines Cite a Page?', bold: true, size: 52, font: FONT, color: '1F4E79' })] }),
  new Paragraph({ spacing: { after: 400 }, children: [new TextRun({ text: 'Results of the core experiments R1–R14, the new experiments of Part C, the follow-up experiments, and what they mean for practitioners', size: 30, font: FONT, color: '404040' })] }),
  P('Data: daily answers from ChatGPT, Gemini and Perplexity to 651 prompts about home-internet providers (client 427, assumed Google Fiber), 24 June – 21 September 2026.'),
  P('Project: AeoCiteAnalysis. Report date: 8 October 2026 (numbers after the data freeze: replay 27 September – 7 October).'),
  new Paragraph({ spacing: { before: 600 }, children: runs('**Contents.** Summary · 1 Data and method · 2 Results R1–R14 · 3 The "is it real?" table · 4 New experiments (Part C) · 5 Follow-up: what explains the rest? · 6 Opening the engines’ search step · 7 What a practitioner should do · 8 Limits · Appendix') }),
  pageBreak(),
);

// Summary
children.push(
  H1('Summary'),
  P('We asked which properties of a web page go with it being cited more often by AI answer engines, comparing pages **on the same website** so that the website itself (its size, reputation and traffic) does not masquerade as a page effect. A feature counts as a real finding only if it passes five independent tests. The main results:'),
  N('**Matching the questions people actually ask is the one page property that passes every test.** A title that matches a prompt (+14 % more citations per standard deviation) and a paragraph that answers a prompt closely (+16 %) are real on all pages and on third-party pages (up to +26 %); broad word overlap with the prompts (+9 %) is real on third-party pages and on all pages narrowly misses one of the five tests.'),
  N('**The popular "AI optimisation" tips do not survive the within-website comparison.** FAQ blocks, structured data, summary boxes, author boxes and page speed all look helpful when websites are thrown together (+8 % to +29 %), and all lose their effect when pages of the same website are compared (R8). Length, headings, lists, tables and links show no effect either.'),
  N('**Which website a page is on matters more than any page feature** — meaning the website’s own typical citation level in these answers, not its authority (see section 1.3). In a flexible model it is about twice as important as the strongest page feature (R7). Outside authority is only a small part of it: +6 % to +11 % per standard deviation at page level, most for ChatGPT (+30 %) (R14).'),
  N('**The three engines behave very differently.** ChatGPT cites few sources (2 per answer), mostly brand websites (64 %) and fresher pages; Gemini cites about 10 different pages per answer and leans on Reddit (13 %) and review sites; Perplexity cites something in only 4 % of answers, mostly review and comparison sites (R1, R11).'),
  N('**Which pages get cited is stable, but mostly not explained by page features.** Citation counts are about 90 % repeatable (odd vs even days). About 13 % of the variation lies between websites and 87 % between pages of the same website; our page features explain about 3 % of the within-website part, leaving about 75 % of all variation stable but unexplained (R13). The page effects we found are real but modest. Links pointing to the page itself are the one other page property that survives the within-website comparison (+23 % per SD of referring websites), mainly through Gemini (X9).'),
  N('**The unexplained part is the engines’ search step (section 6).** Replaying the questions through the engines’ APIs shows ChatGPT reads about 26 pages per question and cites about 3, almost always from the top of its own search results (44 % of rank-1 pages are cited, 4 % at ranks 11–20), and every page it cites is one its search returned. Thirteen replay runs over 11 days explain 32 % [24–42] of the within-website variation in ChatGPT’s 90-day counts, against 3 % for all page features. The engines reword their searches on every answer, and the set of cited pages drifts over weeks (after 10 weeks only 14–29 % of the pattern remains), so a page’s citations depend on staying near the top of the engines’ changing search results. The model’s own knowledge adds to this: ChatGPT writes its searches around brands it already knows and prefers websites it knows, and nearly half of its citations come from websites it would name without searching (X10).'),
);
children.push(spacer(), table(
  ['Page feature (effect per +1 SD, within websites; yes/no features: has vs not)', 'All pages', 'Brand pages', 'Third party'],
  [
    ['Title matches a prompt', '**+14 % real**', '+7 %', '**+24 % real**'],
    ['Best-matching paragraph', '**+16 % real**', '+5 %', '**+26 % real**'],
    ['Word overlap with prompts', '+9 % (fails R4 narrowly)', '+12 %', '**+13 % real**'],
    ['First paragraph matches', '+6 %', '+8 %', '+3 %'],
    ['Page age (older)', '−5 %', '−11 %', '−4 %'],
    ['FAQ, structured data, summary box, author, length, headings, lists, tables, links', 'no', 'no', 'no'],
  ], [4560, 1600, 1600, 1600]));
children.push(P('"Real" = passes all five tests of the guide’s "is it real?" rule (section 3). Brand pages = own and competitor websites (77 websites); no feature passes all five tests there, the match features pass four.', { italics: true, size: 18 }));
children.push(pageBreak());

// 1 Data and method
children.push(
  H1('1  Data and method'),
  H2('1.1 Data'),
  B('**Answers.** 90 daily files (24 June – 21 September 2026), 651 prompts in 7 topics, 3 engines: 143,249 answers (ChatGPT 58,590, Perplexity 58,590, Gemini 26,069).'),
  B('**Citations.** 841,399 cited links; counted **once per answer** (Gemini lists the same page several times) this is 396,188 citations, of which 393,233 point to a resolvable page (the rest are expired Gemini redirect links): 39,043 different URLs on 7,234 websites.'),
  B('**Brands.** Google Fiber = own; competitors = AT&T, Xfinity, Quantum Fiber, Spectrum, T-Mobile, Metronet, Astound and Verizon (including Frontier, now owned by Verizon). Authorised dealer sites count as the brand. All labels were reviewed by hand (see decisions.md).'),
  B('**Pages.** Every cited URL was downloaded once (26 September 2026; pages behind Gemini redirect links resolved later were downloaded on 7 October): 63.2 % succeeded; most failures are Reddit and Quora (robots.txt forbids crawling) and YouTube (videos, not downloaded). Page features were measured for 24,663 pages; brand pages are 89–97 % covered.'),
  B('**Outside authority.** Common Crawl web graph (harmonic centrality, PageRank; 94 % of websites), Majestic Million, Tranco and Chrome UX Report ranks.'),
  B('**Speed.** Real-user Chrome UX Report data; only 466 pages have their own (page-level) data, the rest only a website average.'),
  H2('1.2 Method'),
  B('**Outcome.** y = log2(1 + number of citations) of a URL over the 90 days; main models use URLs cited at least twice (22,330). An effect of +10 % means "10 % more citations for a page one standard deviation (SD) higher on the feature". One SD is roughly the step from a typical page to one in the top sixth. For the yes/no features (FAQ, summary box, author box, structured data) effects are instead given for a page that has the feature compared with one that does not.'),
  B('**Comparing within websites.** Almost every model includes one indicator per website, so a page is compared only with other pages of the same website. Features that describe the website (authority) are studied separately in R14.'),
  B('**Three samples.** All pages; own + competitor pages (the pages brands control); third-party pages.'),
  B('**The "is it real?" rule.** A feature is real only if it passes: (1) significance after correcting for testing 14 features (R3); (2) picked in at least 60 % of 200 re-sampled Lasso fits (R4, stricter one-standard-error penalty); (3) survives double machine learning (R5); (4) has a clean shape — always up, always down or one peak (R6); (5) keeps its sign in at least 4 of 5 robustness checks (R9).'),
  B('**Everything is observational.** Words like "goes with" or "predicts" are correct; "causes" is not. Only a controlled test (N11) can show cause and effect.'),
  H2('1.3 What "website" means, and how it differs from authority'),
  P('Both are **one number per website**, but they come from different places:'),
  B('**Authority** comes from outside our data — for example, Common Crawl gives forbes.com a high score because many important websites link to it. It is the same number whatever questions we study.'),
  B('**"Website"** comes from our own citation data: **how often that website’s pages typically get cited in these AI answers.** It is not a property we measure about the site; it is the site’s own citation level, learned from the citations themselves.'),
  P('In the regressions (R2 and most other experiments) this is one indicator per website, so the model estimates a separate level for each website — 1,117 numbers in R2. Holding each level fixed means a page is compared only with other pages of its own website ("comparing within websites"). In R7 the same idea is summarised in one column, the "website score": the average citation level of the **other** pages of the same website (the page’s own count is never used).'),
  P('The two can rank websites very differently:'),
  table(['Website', 'Typical citations per page (median)', 'Authority (percentile among cited websites)'], [
    ['broadbandnow.com', '20.5', '82nd'],
    ['highspeedinternet.com', '13', '78th'],
    ['cnet.com', '11', '97th'],
    ['gfiber.com', '7', '68th'],
    ['forbes.com', '5', '98th'],
    ['reddit.com', '5', '98th'],
    ['att.com', '5', '94th'],
  ], [3160, 3100, 3100]),
  P('URLs cited at least twice; authority = Common Crawl harmonic centrality.', { italics: true, size: 18 }),
  P('By authority forbes.com ranks above broadbandnow.com, yet a broadbandnow.com page is typically cited about four times as often. The website number captures this; authority does not. That is why the website explains far more of the variation than authority (R13: about 13 % of all variation lies between websites; authority explains 2.3 %) and why the two are almost uncorrelated at the level of citations per page (R14).'),
  P('The website number tells us **that** being on a given website goes with many citations, not **why**. It bundles everything about the site: its topical focus (broadbandnow.com and highspeedinternet.com are entirely about internet providers), whether it is the brand being asked about (N5), how the engines access it, and authority as well. Its main job in this study is to make page comparisons fair: a page feature only counts if it helps a page relative to other pages of the same website.'),
  pageBreak(),
);

// 2 Results
children.push(H1('2  Results by experiment'));

children.push(...experiment('R1', 'Describe the citations',
  'How many citations does each engine give, which websites get them, and how spread out are they?',
  'Counted citations per answer, websites and their shares, per engine, and how many websites are needed to cover 80 % of citations.',
  [
    '**ChatGPT** cites 2.0 different URLs per answer (31 % of answers cite nothing); 49 websites cover 80 % of its citations; 24 % go to Google Fiber’s own sites and 40 % to competitors. Its top source is gfiber.com (11.9 %).',
    '**Gemini** cites 10.1 different URLs per answer (it lists 24.9 links, many repeated); 172 websites cover 80 %; 83 % go to third parties and Reddit is its top source (13.1 %).',
    '**Perplexity** cites nothing in 96 % of answers (0.3 URLs per answer on average); when it does, 93 % goes to third parties.',
    'Across all engines the top websites are reddit.com (9.6 %), gfiber.com (5.8 %), fiber.google.com (4.0 %), att.com (3.9 %) and highspeedinternet.com (3.7 %).',
  ],
  'Each engine has its own source profile: ChatGPT concentrates on a few brand websites, Gemini spreads widely and relies on Reddit and review sites, and Perplexity (in this collection) rarely cites at all. Engine comparisons therefore always use the prompt-days all three engines answered, and Perplexity is treated as a small separate group.',
  [['r01_top_websites_by_engine.png', 'Top 15 websites per engine (share of that engine’s citations).'],
   ['r01_coverage_curve.png', 'How many websites are needed to cover a given share of citations.', 0.8]],
  [table(['', 'ChatGPT', 'Gemini', 'Perplexity'], [
    ['Different URLs per answer', '2.0', '10.1', '0.3'],
    ['Answers with no citation', '31 %', '16 %', '96 %'],
    ['Websites for 80 % of citations', '49', '172', '142'],
    ['Own / competitor / third party', '24 / 40 / 36 %', '8 / 9 / 83 %', '3 / 4 / 93 %'],
    ['Reddit share', '2.3 %', '13.1 %', '5.5 %'],
  ], [3360, 2000, 2000, 2000]), spacer()]));

children.push(...experiment('R2', 'Main model: what predicts citation count?',
  'Which page features go with more citations when we compare pages on the same website?',
  'Regression of y on 14 page features, page type and one indicator per website, robust (HC1) errors, on 12,038 URLs from 1,120 websites (brand pages: 4,794 URLs, 77 websites; third party: 7,185 URLs, 1,044 websites).',
  [
    'The four "matches the question" features are the strongest: best-matching paragraph +15.5 %, title +14.1 %, word overlap +9.1 %, first paragraph +5.9 % per SD (all pages). On third-party pages title and paragraph matching are even stronger (+24 %, +26 %).',
    'Older pages are cited less: −5.1 % per SD of page age (about 450 days).',
    'None of FAQ, structured data, summary box, author box, word count, headings, lists, tables or outbound links is significant on all pages.',
    'Forum pages get 78 % more citations than articles on the same website, list pages 13 % more.',
    'The page features explain only about 5 % of the variation within websites (within R² 0.049).',
  ],
  'Within a website, the pages that get cited are the ones whose wording and meaning match the questions users ask. Structural page features show no link. The effects are real but explain a small part of the picture.',
  [['r02_effects.png', 'R2 effects with 95 % intervals, comparing pages on the same website. Filled = p < 0.05.']]));

children.push(...experiment('R3', 'Correct for testing many features at once',
  'Which R2 effects are still significant after accounting for testing 14 features at the same time?',
  'Benjamini–Hochberg correction of the 14 feature p-values in each sample; q < 0.05 = found, 0.05–0.20 = borderline.',
  [
    '**All pages:** found — best paragraph, title, word overlap (all q < 0.001), first paragraph (q = 0.003), page age (q = 0.004). Everything else q > 0.24.',
    '**Brand pages:** found — title, word overlap, first paragraph, page age, and three structural features: lists (−18 %), tables (+12.5 %), author box (−16 %).',
    '**Third party:** found — best paragraph, title, word overlap; borderline — FAQ (−9 %), first paragraph, page age.',
  ],
  'Correcting for multiple testing removes nothing important from the match features; the structural brand-page results (lists, tables, author) pass here but fail later tests (R5), so they are treated as unreliable.',
  []));

children.push(...experiment('R4', 'How often is each feature picked? (repeat selection)',
  'If we re-sample the data many times, does a feature keep getting selected?',
  'After removing website averages, 200 bootstrap samples; on each, a Lasso with cross-validated penalty. We report the guide’s penalty (lowest error) and the stricter one-standard-error penalty, which we use for the rule because the default kept almost everything with 12,000 URLs.',
  [
    'With the default penalty 13 of 14 features were picked in ≥ 60 % of runs — the test did not discriminate.',
    'With the stricter penalty only two features pass on all pages: best paragraph (100 %) and title (100 %); word overlap is picked in 58.5 % of runs, just under the 60 % threshold. On third-party pages all three pass (100 %, 100 %, 84 %). First paragraph reaches 46 %, page age 0 %.',
    'On brand pages nothing passes (title 47 %, best paragraph 55 %).',
    'FAQ, structured data, summary box, author, length, headings, lists, tables and links are picked in 0–4 % of runs.',
  ],
  'When the model must earn every feature, only the prompt-matching features are needed to predict citations; the structural features add nothing.',
  [['r04_selection_rates.png', 'Share of 200 bootstrap runs in which each feature is picked (grey: default penalty; blue: stricter penalty used in the rule).']]));

children.push(...experiment('R5', 'Double machine learning',
  'Does the effect of a feature survive when a flexible model takes care of all the other features?',
  'For each feature, gradient-boosted trees (5-fold cross-fitting) remove what the other features and the website predict about both y and the feature; the leftover parts are regressed on each other.',
  [
    'The match features survive with similar or larger effects: word overlap +24 %, title +18 %, best paragraph +17 %, first paragraph +7 % (all pages); page age −4 %.',
    'The brand-page effects of tables and lists disappear (+2.5 % and −6.6 %, not significant); the author-box effect remains (−13 %).',
    'Structured data becomes positive on third-party pages (+23 % with vs without, significant only in this test) but fails other tests.',
  ],
  'The prompt-matching effects are not an artefact of assuming straight-line relations between the features; the structural brand-page findings of R2/R3 are not robust.',
  [['r05_dml_vs_r2.png', 'Double-ML effects (blue, 95 % interval) next to the R2 estimates (grey).']]));

children.push(...experiment('R6', 'Check the shape of each effect',
  'Is the link straight, does it level off, or does it go up then down?',
  'Smooth curves (GAM) for each feature, holding the other features fixed, within websites; curves are shown relative to a typical page with a 95 % band.',
  [
    'Title, best paragraph, word overlap and first-paragraph match rise steadily over most of their range; the "one peak/dip" label comes from a small dip at the very low end, inside a wide band.',
    'Page age declines steadily: older pages are cited less at every age.',
    'Length, links, tables and lists are flat.',
  ],
  'The match effects are monotone — more matching is better across the whole range, with no sign of "too much". No threshold or sweet spot was found for length or structure.',
  [['r06_shapes.png', 'Shape of the five strongest effects (all pages, within websites, other features held fixed).']]));

children.push(...experiment('R7', 'Which features matter most overall? (SHAP importance)',
  'How important is the website compared with page features?',
  'LightGBM predicting y on 14,523 URLs, 5-fold cross-validation; three versions: page features + an in-data website score (computed without seeing the URL’s own count), page features + outside authority, and both. Importance = mean absolute SHAP value.',
  [
    'The website score (the typical citation level of the website’s other pages; section 1.3) is the most important input, **about 2× the strongest page feature** (0.29 vs 0.15).',
    'Without it, the page-match features lead (best paragraph 0.19, title 0.16), ahead of every outside authority score (Majestic linking networks 0.09).',
    'All versions predict only 16–17 % of the variation on held-out URLs.',
  ],
  'Which website a page lives on matters more than anything about the page itself. This is the website’s own citation level, not its authority: replacing it with the outside authority scores loses most of its importance. Among the things a page owner controls, matching the questions is what matters most.',
  [['r07_importance.png', 'Top 15 inputs by mean |SHAP| in the three model versions.']]));

children.push(...experiment('R8', 'Does the website hide or fake effects? (Simpson’s paradox)',
  'Do common "AI optimisation" tips still look useful once we compare within websites?',
  'For FAQ block, structured data, summary box, author box and a page-speed score (LCP, INP, CLS; pages with their own speed data only), three estimates: tip alone with all websites pooled, pooled with all features, and within websites (R2).',
  [
    '**Pooled, every tip looks helpful:** a page with an FAQ block has +18 % more citations than one without (third party +29 %), structured data +8 % (+18 %), a summary box +21 % (+17 %), an author box +15 % (+21 %), and speed +12 % per SD (+19 %) — all significant.',
    '**Within websites, all of it disappears:** −5 % to +7 % on all pages, none significant; on third-party pages the FAQ effect even turns negative (−9 %).',
    'Speed: a first run that mixed page-level and website-average speed showed +10.9 %; this came from comparing popular pages (which have their own speed data and are cited more) with the rest, and vanished (−1 %) on page-level data.',
  ],
  'The tips are used more on websites that are cited more for other reasons. Pooled comparisons — the kind behind most published "AI SEO" advice — mistake the website effect for a page effect.',
  [['r08_before_after.png', 'Each tip pooled (grey, orange) and within websites (blue), with 95 % intervals.']]));

children.push(...experiment('R9', 'Robustness checks',
  'Do the main results hold on different slices of the data and over time?',
  'Re-fitted R2 on URLs cited ≥ 1 and ≥ 5 times, with the top 1 % of counts capped, on the first 75 % and last 25 % of days, without each of the 8 largest websites, and per engine.',
  [
    'The match features and page age keep their sign in 5 of 5 checks and in all 8 leave-one-website-out fits.',
    'A model fitted on the first 75 % of days explains 10 % of the hold-out variation (website level alone: 7 %); rank correlation with hold-out counts 0.25.',
    'Per engine the title effect is strongest for Gemini (+17 %), best paragraph for ChatGPT (+9.5 %), word overlap for Perplexity (+7 %).',
  ],
  'The direction of every main result is stable across subsets, time and websites; prediction of future citations is weak, consistent with the heavy day-to-day churn of citations.',
  [['r09_robustness.png', 'R2 effects (all pages) across all robustness checks.']]));

children.push(...experiment('R10', 'Where on the page is the matching text?',
  'Do engines prefer pages whose best-matching paragraph is near the top?',
  'Position of the best-matching paragraph (0 = top, 1 = bottom) for 22,731 cited pages with at least 3 paragraphs.',
  [
    'Median position 0.45 — slightly above the middle (significant but small); 32 % in the top quarter vs 28 % in the bottom quarter.',
    'Similar for all engines (ChatGPT 0.47, Gemini 0.44, Perplexity 0.47); support pages 0.35, forum pages 0.80.',
    'Within websites, pages with the match nearer the top are **not** cited more (−2 % per SD, p = 0.07).',
  ],
  'Engines find the relevant passage wherever it is on the page; moving the answer to the top is not supported by the data.',
  [['r10_depth_hist.png', 'Position of the best-matching paragraph on cited pages.']]));

children.push(...experiment('R11', 'How do the engines differ?',
  'Do the engines differ in how fresh the cited pages are, how much they cite brand sites, and which platforms they use?',
  'Only the 26,069 prompt-days answered by all three engines.',
  [
    '**Freshness:** median page age at citation is 201 days for ChatGPT, 223 for Perplexity and 311 for Gemini; 70 % of ChatGPT’s dated citations are under a year old vs 55 % for Gemini.',
    '**Brand sites:** ChatGPT 25 % own + 39 % competitor (29 % own when the first source counts most), Gemini 8 % + 9 %, Perplexity 2 % + 3 %.',
    '**Platforms:** review and comparison sites take 42 % of Perplexity’s, 23 % of Gemini’s and 9 % of ChatGPT’s citations; Reddit 13 % of Gemini’s; YouTube about 3 % for Gemini and Perplexity.',
  ],
  'Being cited by ChatGPT runs mainly through a brand’s own website; being cited by Gemini and Perplexity runs mainly through third parties — review sites, comparison sites and Reddit.',
  [['r11_brand_share.png', 'Share of citations to own, competitor and third-party websites, plain and weighted by position.'],
   ['r11_platforms.png', 'Platform shares by engine.', 0.85],
   ['r11_freshness.png', 'Share of dated citations younger than a given age.', 0.75]]));

children.push(...experiment('R12', 'Page type and topic effects',
  'Do some kinds of pages get cited more, and do yes/no page features work differently by user intent or topic?',
  'Page-type effects from R2; prompt intent labelled by an LLM (learning / comparing / ready to buy); the R2 model re-fitted on citations from each intent and topic.',
  [
    'Forum pages +78 % and list pages +13 % vs articles on the same website; other page types do not differ significantly.',
    'FAQ blocks help for "ready to buy" questions (+12 %) and the Pricing topic (+13 %) but go with fewer citations for "learning" questions (−10 %) — which is why FAQ averages out to nothing.',
    'Title and word matching matter most for comparison questions (+18 % to +19 %) and the Speed topic (+17 % word overlap).',
  ],
  'Structural features are context-dependent at best: an FAQ block is associated with more citations only for price and purchase questions. Matching the question is useful across intents and strongest for comparison questions.',
  [['r12_page_types.png', 'Page types vs articles on the same website.', 0.8],
   ['r12_heatmap.png', 'Feature effects by user intent and topic (% per SD; yes/no features: has vs not; * p < 0.05).']]));

children.push(...experiment('R13', 'How much is the website vs the page?',
  'How much of the difference in citation counts can the website alone explain, and how much do page features add?',
  'R² of nested models on the R2 rows (12,031 URLs, 1,117 websites), in sample and for unseen pages (cross-validated). Because these two disagree strongly for the website, we added a mixed model that splits the variation into a between-website and a within-website part (small websites are shrunk toward the average instead of being memorised or ignored), repeated the unseen-page test on websites with enough pages to estimate their level, and checked how repeatable the citation counts are: citations counted on odd days vs even days (split-half reliability).',
  [
    'Word count alone explains 0.2 % and outside authority alone 2.3 %.',
    'Website alone: 18.7 % in sample but only 4.0 % on unseen pages. The gap comes from small websites: many have only 2–3 cited pages, so in sample their level is memorised and on unseen pages it is a poor guess. Neither number is the right size.',
    '**Mixed model (the right size):** 13 % of the variation lies between websites and 87 % within them. The page variables explain 3.4 % of the within-website variation — the real page effect — and 42 % of the between-website differences (websites cited more are partly websites whose pages match the questions better).',
    '**Websites with enough pages:** once a website has at least 10 cited pages, the website alone predicts about 10 % of unseen pages’ citations (11 % from 30 pages), and the page variables add about 3 points.',
    '**The counts are highly repeatable:** a URL’s citations on odd days and on even days correlate at 0.82, so about 90 % of the variation in the 90-day counts is systematic and only about 10 % is day-to-day chance (89 % within websites, 87 % between them). The unexplained part is therefore not noise: the same pages keep being cited more than others, for reasons our features do not capture.',
  ],
  'R13 is conclusive. (1) About 13 % of the differences in citation counts lie between websites and 87 % between pages of the same website. (2) Our page features explain about 3 % of the within-website differences: real (R2–R9) but small. (3) Outside authority explains little of the website part (2.3 % in total; section 1.3). (4) The unexplained part is not chance: the counts are about 90 % repeatable, so about 75 % of all variation is **stable but unexplained**. Which pages get cited is decided largely by something we did not measure — most likely how the engines’ search step ranks each page for each question (see section 5).',
  [['r13_components.png', 'Left: where the variation lies (mixed model). Right: prediction of unseen pages by website size.'],
   ['r13_variance_split.png', 'Share of variation explained by each nested model, in sample and on unseen pages.', 0.85]],
  [
    H3('Where the variation lies (mixed model)'),
    table(['Part of the variation', 'Share of all variation', 'Explained by page variables'], [
      ['Between websites (which website a page is on)', '**13 %**', '42 % of this part (5.5 points)'],
      ['Within websites (pages of the same website)', '**87 %**', '3.4 % of this part (2.9 points)'],
      ['Not explained by anything measured', '**about 84 %**', 'of which about 75 points stable, about 9 points day-to-day chance'],
    ], [3960, 2400, 3000]),
    spacer(),
    H3('How repeatable are the counts? (odd vs even days)'),
    table(['Part', 'Correlation, odd vs even days', 'Repeatable share of the 90-day count', 'Day-to-day chance'], [
      ['All variation', '0.82', '**90 %**', '10 %'],
      ['Within websites', '0.81', '89 %', '11 %'],
      ['Between websites (site averages)', '0.77', '87 %', '13 %'],
    ], [3160, 2000, 2400, 1800]),
    spacer(),
    H3('Prediction of unseen pages, by website size'),
    table(['Websites included', 'URLs / websites', 'Website only', 'Website + page', 'Page only'], [
      ['≥ 2 cited pages (all)', '12,038 / 1,120', '4.0 %', '7.6 %', '7.4 %'],
      ['≥ 10 cited pages', '8,965 / 179', '9.8 %', '13.1 %', '8.5 %'],
      ['≥ 30 cited pages', '7,061 / 58', '11.4 %', '13.8 %', '9.6 %'],
    ], [2560, 1900, 1600, 1700, 1600]),
    spacer(),
  ]));

children.push(...experiment('R14', 'Outside authority',
  'Do pages on websites with high outside authority get cited more, how big is this compared with page features, and do the engines differ?',
  'Website-level models (total citations on authority) and page-level models without website indicators (mixed model with a random website effect; clustered OLS), with interactions, per engine and on subsets.',
  [
    '**Websites:** one SD more authority goes with 21–30 % more total citations (all five scores). For the same number of cited pages the sign reverses (−5 % to −8 %): large websites spread their citations over many pages.',
    '**Pages:** +7 % (mixed model) to +12 % (clustered OLS) per SD of authority — about the size of word overlap (+7 %) and half the title or best-paragraph match (+13 %).',
    '**Engines:** ChatGPT +30 % per SD, Perplexity +3 %, Gemini −9 % (not significant).',
    'The effect holds on brand pages (+13 %), third-party pages (+9 %) and without Reddit, YouTube, Google and Wikipedia (+10 %). Pages on websites below about one SD under the average authority are cited clearly less.',
    'Title matching matters slightly more on less-authoritative websites (interaction −0.05, p = 0.05). The Chrome traffic rank and Majestic predict future citations slightly better than the Common Crawl scores.',
    'Over-performers (cited far more than their authority predicts): reddit.com, gfiber.com, att.com, highspeedinternet.com, verizon.com, cnet.com.',
  ],
  'Authority helps, especially with ChatGPT, but less than matching the question, and it cannot be changed quickly. Small websites can compensate partly through closer matching.',
  [['r14_authority.png', 'Left: websites, authority vs citations. Right: authority next to page features (mixed model).'],
   ['r14_shape_engines.png', 'Left: shape of the authority effect. Right: authority effect per engine.']]));

children.push(pageBreak());

// 3 Is it real
children.push(
  H1('3  The "is it real?" table'),
  P('A feature is a real finding only if it passes all five tests. Features that fail any test are kept in the appendix of results/is_it_real.md with the tests they failed.'),
  ...figure('is_it_real.png', 'Test results per feature and sample (blue = pass). Effect = R2 estimate.'),
  table(['Sample', 'Real findings (pass 5 of 5)', 'Pass 4 of 5'], [
    ['All pages', 'best paragraph +16 %, title +14 %', 'word overlap +9 % (fails R4: picked in 58.5 % of runs), first paragraph +6 % (fails R4), page age −5 % (fails R4)'],
    ['Third party', 'best paragraph +26 %, title +24 %, word overlap +13 %', 'structured data +11 % (fails R3), page age −4 % (fails R3)'],
    ['Own + competitor', 'none', 'title +7 %, first paragraph +8 %, author −8 % (fail R4); word overlap +12 % (fails R6)'],
  ], [1800, 3780, 3780]),
  spacer(),
  pageBreak(),
);

// 4 Part C
children.push(
  H1('4  New experiments (Part C)'),
  P('The guide’s Part C looks at what the core experiments do not cover: stability over time, agreement between engines, brand mentions, the questions themselves, and pages that were never cited. Ten of the fourteen could be run on the data we have; the other four need time, people, page edits or new collection (section 4.11).'),
);

children.push(...experiment('N1', 'How stable are citations from day to day?',
  'If we ask the same engine the same question on two different days, how many of the cited pages are the same?',
  'For each engine and prompt, the overlap (shared / all) of the cited URLs and websites on days 1 to 60 apart.',
  [
    'Next day, only **15 %** of ChatGPT’s cited URLs are the same for the same question, 26 % for Gemini and 13 % for Perplexity; at the website level about one third for ChatGPT and Gemini.',
    'Overlap declines slowly after that for ChatGPT and Gemini (URLs: 11 % and 20 % after 30 days) but almost vanishes for Perplexity (1 % after 30 days).',
  ],
  'On any single day the sources for a question change a lot, so one day of data says little. Over weeks the same pages keep coming back (R13: 90 % repeatable totals). Measurements and tests should cover several weeks.',
  [['n01_overlap_curve.png', 'Overlap of cited URLs (left) and websites (right) for the same question, by days apart.']]));

children.push(...experiment('N2', 'Do the engines agree with each other?',
  'For the same question on the same day, do ChatGPT, Gemini and Perplexity cite the same pages and websites?',
  'Pairwise overlap of cited websites and URLs per prompt-day; websites cited by one, two or all three engines; page features of shared vs engine-specific pages.',
  [
    'On the same question and day the engines hardly agree: ChatGPT and Gemini share **6 %** of cited websites (at least one website in common in 36 % of cases) and 2 % of URLs; Perplexity shares almost nothing with either.',
    'Over the whole period, **193 websites cited by all three engines take 66 %** of all citations; 4,616 websites cited only by Gemini take 13.5 %.',
    'The 194 pages cited by all three engines match the questions better (median title similarity 0.68 vs 0.58), are longer (1,342 vs 730 words) and younger (194 vs 374 days) than pages cited by one engine only.',
  ],
  'Each engine picks its own sources for a given answer, but they draw on a shared core of about 200 websites. Pages that every engine cites are the ones that match the questions best.',
  []));

children.push(...experiment('N3', 'Does being cited go with being mentioned (and liked)?',
  'When a brand’s own website is cited, is the brand more likely to be mentioned, mentioned first, and mentioned positively?',
  'One row per answer and tracked brand (the 8 brands with mention data; Verizon has none). Logistic regressions with engine, topic, brand and "the prompt names the brand" as controls; errors clustered by prompt.',
  [
    'When a brand’s own site is cited the brand is mentioned in **97 %** of answers, against 14 % otherwise (odds ×172 after controls; naming the brand in the prompt: ×75).',
    'A cited brand is **2.3× more likely to be ranked first** among the brands mentioned.',
    'Tone differs by engine: ChatGPT is more positive about brands whose site it cites (odds ×3.8); Gemini (×0.61) and Perplexity (×0.44) are less positive.',
    'A review site in the sources goes with more brand mentions (×1.85); a competitor’s site in the sources with fewer first places for the brand (×0.30).',
  ],
  'Being cited and being recommended go together, especially for ranking first. This is association, not effect: an answer about a brand naturally both names it and cites it.',
  []));

children.push(...experiment('N4', 'Which sources sit behind negative mentions?',
  'When an engine says something negative about a brand, which websites does it cite next to that sentence?',
  'Links inside each brand mention, else links in the same paragraph of the answer; share of negative / positive mentions per website (websites with at least 20 attached mentions).',
  [
    'Most mentions have no link nearby: only 33 % of Gemini’s, 29 % of ChatGPT’s and 2 % of Perplexity’s mentions can be tied to a source. N4 therefore describes a minority of mentions.',
    'Most negative sources: Spectrum’s community forum (73 % negative), BBB (45 %), Xfinity’s forums (42 %), spectrum.com (29 %). Overall 10 % of attributed mentions are negative.',
    'Most positive: PCMag (82 % positive), trade press (Telecompetitor 84 %), USA Today (78 %).',
    'For Google Fiber the most negative attached sources are Light Reading (29 %) and ConsumerAffairs (10 %); its own help forum (support.google.com) 7 %.',
  ],
  'Negative statements come mostly from customer forums and complaint sites, positive ones from review and trade press. Monitoring and answering questions on community forums is where negative source material originates.',
  []));

children.push(...experiment('N5', 'Which questions lead to which sources?',
  'Do features of the question change how many and what kind of pages are cited?',
  'Per answer: number of URLs and shares of own, competitor and Reddit/YouTube citations, regressed on prompt features (length, brand named, comparison, price, yes/no, LLM intent) and engine; errors clustered by prompt.',
  [
    'Naming Google Fiber in the question adds **+36 percentage points** to the share of its own sites among the sources; naming a competitor adds **+26 points** to competitor sites and removes 12 points from Google Fiber.',
    'Comparison questions lower the share of Google Fiber’s sites by 6 points and of competitors by 4.',
    'Longer questions get fewer sources (−0.5 per 10 words) and more competitor sites (+9 points).',
    'When a prompt names AT&T, AT&T’s site is cited in 70 % (ChatGPT) and 66 % (Gemini) of answers, Google Fiber’s in 40 % and 30 %.',
  ],
  'The question decides the sources more than anything on the page: a named brand’s own website is pulled in. Content that clearly answers "brand X vs brand Y" questions on the brand’s own site is directly relevant.',
  []));

children.push(...experiment('N6', 'What gets cited first?',
  'Which pages appear at the top of the source list, and does position matter for brand mentions?',
  'Share of a URL’s citations in first position (URLs cited at least 5 times) regressed on page features with website indicators; agreement between the brand of the first source and the first-mentioned brand.',
  [
    'When the first source is a brand site, that brand is also mentioned first in **83 %** of answers (45–54 % expected by chance).',
    'Only the best-matching paragraph helps a page reach first place (+2.1 percentage points per SD, p < 0.001); an author box goes with fewer first places (−2.8 points, with vs without).',
    'Brand sites are cited first far more often than third parties (median share first: own 18 %, competitor 14 %, third party 0 %).',
  ],
  'The first source and the first recommendation go together. A page that answers the question directly is more likely to be that first source.',
  []));

children.push(...experiment('N7', 'Pages that were never cited (brand websites)',
  'Are the features that predict HOW OFTEN a page is cited the same as those that decide WHETHER it is cited at all?',
  'Never-cited pages sampled from the tracked brands’ sitemaps (up to 600 per website; 5,993 downloaded and measured), compared with the cited pages of the same websites in a logistic regression with website indicators. Only the 10 websites with at least 20 pages in each group could be used (4,513 cited vs 5,262 never-cited pages). Its search-results half is covered by X1 (section 5).',
  [
    'Matching the questions strongly predicts being cited **at all**: best paragraph odds ×11.9 per SD, title ×3.8, word overlap ×2.8.',
    'FAQ blocks (×0.60), structured data (×0.66) and author boxes (×0.64) go with being cited **less** often at all.',
    'Older pages are more often among the cited ones (×2.2), probably because sitemaps contain many new location, legal and account pages.',
  ],
  'The same features that predict how often a page is cited also decide whether it is cited at all — and much more strongly. Pages that do not match any question are simply not picked. The "AI optimisation" tips again show no benefit.',
  [],
  [table(['Feature (per +1 SD; yes/no: has vs not)', 'Odds of being cited at all (N7)', 'More citations if cited (R2, brand pages)'], [
    ['Best paragraph matches', '**×11.9**', '+5 %'],
    ['Title matches a prompt', '**×3.8**', '+7 %'],
    ['Word overlap with prompts', '**×2.8**', '+12 %'],
    ['Has FAQ', '×0.35', '−2 %'],
    ['Has structured data', '×0.43', '−12 %'],
    ['Has author box', '×0.30', '−16 %'],
  ], [3960, 2700, 2700]), spacer()]));

children.push(...experiment('N9', 'How long does a page stay cited?',
  'Once a page starts being cited, how long does it keep getting cited, and what makes new pages enter?',
  'For URLs first cited after day 14: days until their last citation (a page that is still cited in the final 14 days counts as still active); Kaplan–Meier curves and a Cox model on page features.',
  [
    'Half of newly cited pages stop being cited within **5 days** on ChatGPT and Perplexity and within **19 days** on Gemini; about half are cited on one day only.',
    'List pages last longest (median 18 days), "other" pages shortest (5 days).',
    'Better title and best-paragraph matching keep a page cited longer (risk of dropping out ×0.94 and ×0.87 per SD); structured data goes with dropping out slightly sooner (×1.09, with vs without).',
    'About 1,700–2,900 new URLs enter each week, and they are getting younger (median page age at entry falls from about 400 to about 250 days).',
  ],
  'There is a constant stream of new sources and most last only days or weeks. Matching the questions helps a page not only to be picked but to stay picked.',
  [['n09_survival.png', 'Share of newly cited pages still being cited, by engine (left) and page type (right).'],
   ['n09_new_urls.png', 'New URLs entering the citations each week (the last week has one day).', 0.75]]));

children.push(...experiment('N13', 'Which way of measuring "matches the question" works best?',
  'Is simple word overlap the best measure, or do search-engine (BM25) or meaning-based (embedding) scores predict citations better?',
  'Seven match measures, each swapped in turn into the R2 model; gain in within-website R² and prediction of the hold-out period. Topic level: matching a topic’s own prompts vs all prompts.',
  [
    'Meaning-based measures beat word counting: title similarity from a second embedding model (bge-small) +3.5 points of within-website R², best paragraph +2.8; word overlap +1.4; BM25 +0.8. All four Step-A6 measures together: +3.8.',
    'Matching the prompts of **the page’s own topic** predicts that topic’s citations far better than matching all prompts: in every topic the topic-specific score carries the effect and the all-prompts score adds nothing.',
  ],
  'Engines match on meaning, not only on words, and on specific questions rather than on a general topic. Writing for the specific questions of a topic is what counts (see also X2 in section 5).',
  [['n13_measures.png', 'Gain in within-website R² from each match measure.', 0.85]]));

children.push(...experiment('N14', 'What kinds of websites do engines trust?',
  'Which types of sources does each engine prefer, and for which topics and intents?',
  'The 300 most-cited websites labelled by type (brand sites from the reviewed brand list, the others by the LLM from the host name and page titles); shares of citations per engine, topic, intent and week.',
  [
    '**ChatGPT:** 64 % brand sites, 13 % review / comparison sites, 3 % forums; for "ready to buy" questions 86 % brand sites.',
    '**Gemini:** the most mixed — 23 % review / comparison sites, 17 % forums, 17 % brand sites, 14 % other internet providers.',
    '**Perplexity:** 41 % review / comparison sites, 9 % other providers, 8 % forums, 7 % brand sites.',
    'For comparison questions review sites gain further (Gemini 31 %, Perplexity 50 %).',
  ],
  'To be cited by ChatGPT a brand mainly needs its own website to answer the questions; to be cited by Gemini and Perplexity it needs to be well covered on review and comparison sites and in forums.',
  [['n14_types_by_engine.png', 'Types of cited websites by engine.'],
   ['n14_types_by_topic.png', 'Types of cited websites by topic, ChatGPT and Gemini.']]));

children.push(
  H2('4.11 Experiments not run yet'),
  B('**N8 — page changes over time:** needs page downloads every two weeks for several months (next round from 10 October 2026).'),
  B('**N10 — does the cited page support the answer?:** needs two human raters on 300 sentence–link pairs before the LLM may take over.'),
  B('**N11 — controlled test:** needs permission to edit client pages; the only way to show cause and effect.'),
  B('**N12 — reworded questions:** needs new queries sent to the three engines (50 prompts × 5 wordings × 7 days).'),
  pageBreak(),
);

// 5 Follow-up
children.push(
  H1('5  Follow-up: what explains the rest?'),
  P('R13 showed that citation counts are about 90 % repeatable, yet about 75 % of the variation is stable but unexplained by the page features, the website and authority. We ran four follow-up experiments on the two most likely explanations: matching specific questions (X2) and search-engine ranking (X1, X1b, X1c).'),
);

children.push(...experiment('X2', 'Matching specific questions instead of all questions',
  'The Step-A6 features compare a page with all 651 questions at once. Is a page cited for the specific questions it matches, and does matching many questions explain how often it is cited?',
  'For all 12,038 R2 pages and all 651 prompts (7.8 million page × question pairs), the similarity of the page to each question (embeddings, title, BM25), compared with whether the page was cited for that question. (A) Within each page: are the questions it is cited for the ones it matches best? (B) Across pages: does the number of questions a page matches strongly explain its total citations?',
  [
    '**(A) Very strongly yes.** Within a page, a question it is cited for matches it better than a question it is not cited for in 88 % of comparisons (AUC 0.88; ChatGPT 0.87, Gemini 0.88, Perplexity 0.89), and this holds for 96 % of pages. Per SD of match to a question, the odds that the page is cited for it rise ×1.45 (meaning), ×1.48 (title) and ×1.73 (BM25). Because each page is compared with itself, neither the website nor overall page quality can explain this.',
    '**(B) Only a little.** The number of questions a page matches strongly adds +17 % citations per SD, but raises the explained within-website variation only from 4.9 % to 5.9 % (mixed model 3.6 % → 3.9 %).',
  ],
  'Matching decides **which** questions a page is cited for, almost completely. It does not decide **how often** the page is picked for those questions — that is where the unexplained part lies.',
  [['x02_prompt_match.png', 'Left: within-page AUC by engine. Right: match scores of cited and not-cited page–question pairs.']]));

children.push(...experiment('X1', 'Search-engine ranking',
  'All three engines search the web before answering. Do the pages they cite rank in Google for the same question, and does ranking explain how often a page is cited?',
  'Google top-10 results (SerpAPI) for all 651 questions, typed as asked (September 27). A pilot on 50 questions (X1b) repeated this with a national location ("United States") and with an LLM-written, engine-style search query, to rule out local results and query wording as the reason for a mismatch.',
  [
    '**Google’s top 10 contains almost none of the engines’ sources:** 0.1 % (ChatGPT), 0.6 % (Gemini) and 0.6 % (Perplexity) of cited URLs, and 4–9 % of cited websites, are in the top 10 for the same question.',
    '**Neither location nor query wording changes this (X1b):** cited websites in the top 10 were 5.2 % (literal), 5.5 % (national location) and 5.2 % (LLM-rewritten query) for the same 50 questions.',
    'Where Google and the engines coincide, rank matters: a #1 result is cited for its question 14–16 % of the time, falling with rank; the 408 pages that rank top 10 for any question get +144 % citations, and search presence raises the explained within-website variation from 3.6 % to 6.6 %.',
    '**Google’s AI Overview looks much closer to the engines (X1c):** it appears for 634 of 651 questions; in the 7 responses that included its source list, 27–35 % of the engines’ cited websites were among its sources — five times the overlap with organic results. The source lists of the other AI Overviews could not be retrieved through SerpAPI.',
  ],
  'The engines’ retrieval is not "Google’s first page" — not even for Gemini, which is grounded on Google Search. They most likely send several queries per question, go deeper than the top 10, and use their own indexes or partner search providers. Search ranking can therefore not be tested with ordinary search results; AI answer engines appear to share sources with each other far more than with classic search rankings.',
  [['x01_search_rank_google.png', 'Left: share of cited sources in Google’s top 10 for the same question. Right: chance that a top-10 result is cited, by rank.']],
  [table(['Google search for the same 50 questions', 'Cited websites in top 10', 'Cited URLs in top 10', 'Rank-1 result cited'], [
    ['Question as typed (X1)', '5.2 %', '0.5 %', '16 %'],
    ['Question as typed, location "United States"', '5.5 %', '0.6 %', '16 %'],
    ['LLM-written search query, location "United States"', '5.2 %', '0.6 %', '16 %'],
  ], [3960, 1800, 1800, 1800]), spacer()]));

children.push(
  H2('What we now know about the unexplained part'),
  table(['Candidate explanation', 'Tested in', 'Result'], [
    ['Day-to-day chance', 'R13 (odd vs even days)', 'Only about 10 % of the variation'],
    ['Website and outside authority', 'R7, R13, R14', '13 % between websites; authority explains 2.3 %'],
    ['Page features (match, structure, age)', 'R2–R9, R13', 'About 4 % of the within-website variation'],
    ['Matching specific questions', 'X2, N7', 'Decides WHICH questions a page is cited for (AUC 0.88) and WHETHER it is cited at all; adds little to HOW OFTEN'],
    ['Google ranking for the question', 'X1, X1b', 'Only ~5 % of cited websites rank top 10; not the engines’ retrieval'],
    ['Shared sources between AI answer engines', 'X1c (7 questions)', 'Promising (27–35 % overlap) but not testable at scale with SerpAPI'],
  ], [2900, 2000, 4460]),
  spacer(),
  P('Still untested: page-level links and popularity (paid page-level link scores and traffic data), the content facts the engines actually use (N10, with human raters first), and each engine’s own retrieval, which is not observable from outside. The controlled test (N11) remains the only way to show that changing a page changes its citations.'),
  pageBreak(),
);

const X4 = {"found": ["**ChatGPT’s retrieval is not Bing’s or Google’s top 10, even for ChatGPT’s own queries:** only 0.5 % (Bing) and 0.7 % (Google) of the pages ChatGPT retrieved, and 0.7 % / 2.4 % of the pages it cited, are in the top 10 for the query it issued.", "**Gemini’s sources are mostly not Google’s top 10 for Gemini’s own queries either:** 5.6 % of its grounding pages are among them; where they coincide, the order agrees moderately (rank correlation 0.5).", "At the level of websites the overlap is larger — 13–28 % of ChatGPT’s and 32 % of Gemini’s sources come from websites in the top 10 — so the engines draw on the same websites as the search engines, but on different pages."], "conclude": "The engines retrieve from their own indexes, or from far deeper in a search engine’s results than the first page, and rank pages their own way. Public search rankings — for the question as typed (X1) or for the engines’ actual queries (X4) — are therefore not a usable stand-in for the engines’ retrieval; the retrieval has to be observed directly, as in X3.", "short": "not the top 10 of Bing or Google, even for the engines’ own queries — and the pages are not further down either: 93–97 % are absent from Google’s top 50 (X8)"};
// 6 Phase 1 and 2
children.push(
  H1('6  Opening the engines’ search step (Phase 1 and 2)'),
  P('Section 5 left the main question open: why are some pages cited far more often than others on the same website, when page features explain only about 4 % of that? We split the problem in three stages — **retrieval** (does the engine’s search step find the page for a question?), **selection** (among the pages it found, does it cite this one?) and **repetition** (does that happen again day after day?) — and ran two sets of experiments: analyses of the existing data (Phase 1: E1–E6) and a replay of the collection through the engines’ own APIs, this time keeping what the collection threw away (Phase 2: X3, X4). The controlled page test (N11, Phase 3) cannot be run because client pages cannot be changed.'),
  P('Two parts of a page’s 90-day count are used throughout: **breadth** — how many questions the page is cited for (median 2) — and **depth** — how often it is picked per question it is cited for (median 2.8 times). They carry about equal shares of the variation (54 % and 47 %) and are unrelated to each other.'),
);

children.push(...experiment('E1', 'Per engine, breadth and depth',
  'Does pooling three engines hide page effects, and is the unexplained part in breadth or in depth?',
  'R13’s mixed-model split repeated for each engine and for total, breadth and depth, on the R2 pages each engine cited; page variables = all R2 variables; repeatable share = split-half reliability within websites.',
  [
    'Page variables explain 1 % (ChatGPT), 11 % (Gemini) and 6 % (Perplexity) of the within-website variation in total citations — pooling does not hide a large effect.',
    '**Depth is almost untouched by page features** (1–4 % in every engine); breadth somewhat more (4–12 %).',
    'The stable unexplained share is 57–70 % for breadth and 70–83 % for the total and for depth, in every engine.',
  ],
  'The gap is not an artefact of mixing engines. Page features speak a little to **which and how many** questions a page is cited for, and almost not at all to **how often** it is picked for them.',
  [['e01_engines_breadth_depth.png', 'Within-website variation explained by page variables (solid) and the repeatable ceiling (light), by engine and part.']]));

children.push(...experiment('E2', 'Choice among competing pages',
  'For a given question, which of the pages that compete for it does the engine pick on a given day?',
  'For 300 random questions, each engine’s candidate pages = R2 pages it cited for the question in the first 60 days; outcome = cited or not in each answer of the last 30 days (out of time; about 5,000 answers per engine, 36–39 candidates each). Conditional logit with variables relative to the other candidates: match to the question, the page’s general pull (citations for other questions), the page’s history with this question, cited yesterday.',
  [
    '**Relative match hardly decides the daily pick:** it explains 0.6 % (ChatGPT) and 2.8 % (Gemini) of the choice (McFadden pseudo-R²).',
    'The page’s general pull adds about 3 points; **the page’s own history with this question** lifts the fit to 19 % (ChatGPT) and 26 % (Gemini), with odds ×1.8 and ×2.6 per SD. Cited yesterday adds 2 points (×1.3–×1.6).',
  ],
  'Depth is a persistent **page–question bond** that neither the page’s match nor its general quality captures. That pattern fits a ranking position inside the engine’s own search for that question — which X3 then observes directly.'));

children.push(...experiment('E3', 'Coverage of brands, plans, prices and places',
  'Are pages that mention more of the things questions ask about (brands, speed tiers, prices, states) cited for more questions?',
  'Counts of the 9 tracked brands, distinct speed tiers, prices, US states and question words in each page’s text, added to the R2 model for total, breadth and depth.',
  [
    'Pages that name more tracked brands are cited for more questions (+10 % breadth per SD, p < .001).',
    'Speed tiers, prices and places add nothing useful; explained breadth rises only from 9.8 % to 12.5 %.',
  ],
  'Covering several brands helps a page reach more comparison questions, but entity coverage is not the missing driver.'));

children.push(...experiment('E4', 'AI-crawler access (robots.txt and JavaScript)',
  'Are pages that block an engine’s crawler, or show no text without JavaScript, cited less by that engine?',
  'robots.txt of all 7,218 cited websites checked for OAI-SearchBot, ChatGPT-User, GPTBot, Googlebot, Google-Extended and PerplexityBot; "no text without JavaScript" = fewer than 100 words in the downloaded HTML.',
  [
    '7 % of the URLs ChatGPT cites are officially blocked for its own search crawler; the 27 websites that block it get the same share of their citations from ChatGPT (20 %) as those that allow it.',
    'Reddit blocks every crawler in its robots.txt and is still Gemini’s top source (34,409 citations) — Google licenses Reddit’s content.',
    'Pages with no text without JavaScript are cited somewhat less within their website (Gemini −11 %, p = .008; ChatGPT −7 %, p = .08).',
  ],
  'robots.txt is not what decides citation, in either direction. Server-rendered text is a small plus.'));

children.push(...experiment('X3', 'Replaying the engines with their search results visible',
  'The collection called the engines’ APIs with web search on but stored only the answer. If we replay the same questions and keep the search queries and every page the engine consulted, how much of the unexplained part is decided at retrieval?',
  'The 651 questions sent through the same APIs, first on 27 September (ChatGPT = gpt-5.4-nano with web search, 3 full runs; Gemini 3.1 Flash Lite with Google Search grounding, 1 run) and then daily until 7 October: 13 ChatGPT runs over 11 days (results below use all runs unless stated). For ChatGPT the API returns its search queries and the list of pages each search call returned, of which it cites only a few. A search call almost always sends four queries (965 of 986 calls; the rest three) and returns ONE list of about 27 pages (at most 39) without saying which query found which page; the list is not the queries’ results one after another, nor taken in turns (a page’s best-matching query is unrelated to its position, correlation 0.00), so the pages are pooled and re-ranked into one order. Rank in this report means the position in that pooled list — the order in which the search tool hands pages to the model; Gemini returns its queries and only the sources it uses. Cost about $10 per ChatGPT run.',
  [
    '**The replay behaves like the collection.** Questions for which the replay did not search had no citations in 83 % of collected ChatGPT answers (7 % when it searched); 59 % of the replay’s citations had been cited for the same question in the collection’s last 14 days, against 66 % for the collection’s own last day.',
    '**ChatGPT reads far more than it cites:** a median of 26 pages consulted per question, 3 cited. Only 10 % of consulted pages are cited.',
    '**Selection follows the engine’s own ranking.** A consulted page is cited 44 % [43–45] of the time at rank 1, 29 % at rank 3, 12 % [12–13] at ranks 6–10 and almost never below rank 20 (odds ×0.42 [0.41–0.43] per doubling of rank; 4,946 result lists). Pages found by more than one of ChatGPT’s searches for the same question have ×10.8 [9.3–12.5] the odds. Every page ChatGPT cites is one its searches returned (22,315 of 22,315 cited URLs).',
    '**Retrieval predicts the 90-day counts.** Pages ranked first in the replay averaged 13.9 collected citations for that question, pages below rank 20 averaged 0.1. The replay runs explain **32 % [95 % CI 24–42] of the within-website variation** in ChatGPT’s 90-day counts — against 3 % [2–6] for all page features (fixed-effects measure; 0.7 % in the mixed model), a difference of 29 points [20–38]; with page features and replay citations added, 39 %. Shuffling the retrieval data among pages of the same website removes the effect (0.05 %, p = .005), so it comes from which specific pages the engine finds. Intervals from 1,000 bootstrap resamples of websites (X5). For breadth, what ChatGPT retrieved predicts better than what it cited (20.5 % vs 16.8 %): breadth is decided at retrieval.',
    '**Every answer starts from new queries.** Asked the same question twice on the same day, ChatGPT almost never sends the same search query (2.7 % overlap), and only 18 % of consulted and 15 % of cited pages overlap. Typical queries are short rewrites with brand, attribute and sometimes the year ("Google Fiber self install kit steps power fiber cable ethernet setup").',
  ],
  'Most of the variation is created in the engine’s search step: which pages its (constantly reworded) searches return, and how high. A page’s count is essentially how often it lands near the top across the many queries the engine invents for a question. Page features act only through that step, which is why they explain so little on their own.',
  [['x03_retrieval.png', 'Left: share of the pages ChatGPT consulted that it cited, by rank in its own search results (runs 2–3). Right: within-website variation in ChatGPT’s 90-day counts explained by page variables alone and with the first three replay runs; the line marks the repeatable ceiling.']],
  [table(['ChatGPT, one question', 'Same day, two runs', ''], [
    ['Search queries identical', '2.7 %', 'every answer rewrites the question differently'],
    ['Pages consulted in both runs', '18 %', 'retrieval is mostly different each time'],
    ['Pages cited in both runs', '15 %', 'the cited set changes as much as the consulted set'],
    ['Searched in one run but not the other', '7 %', 'whether to search is itself not fixed'],
  ], [3200, 1800, 4360]), spacer()]));

children.push(...experiment('E6', 'The cited set drifts over time',
  'R13 found 90-day counts 90 % repeatable (odd vs even days). But both halves span the whole period. Are the same pages cited in July and in September?',
  'Six non-overlapping 14-day windows. For every URL an engine cited (websites with ≥ 2 such URLs), the within-website R² between the counts of two windows, by the distance between them; lag 0 = odd vs even days inside one window. For ChatGPT also: how much of each window the 27 September replay explains.',
  [
    '**What is cited changes steadily.** Within one 14-day window 63–67 % of the within-website variation is repeatable. Two weeks later 40 % (ChatGPT), 47 % (Gemini) and 22 % (Perplexity) is still shared; ten weeks later 14 %, 29 % and 0 %.',
    'The replay (27 September – 7 October) explains 14 % of late June’s counts, 21 % of mid-August’s, 33 % of 2–15 September’s and 43 % of the last six days.',
    'At a distance of about two weeks the replay explains 33 % while the collection itself only reaches 40 %: **the engine’s retrieval accounts for about 80 % of what is predictable at that distance.**',
  ],
  'A large part of the "stable but unexplained" 75 % is not a fixed property of pages at all: pages enter and leave the engines’ search results over weeks, so a 90-day total mixes several different cited sets. A snapshot of page features taken in September cannot explain which pages were cited in July.',
  [['e06_drift.png', 'Left: share of within-website variation still shared between two 14-day windows, by the days between them. Right: variation in each window explained by the 27 September ChatGPT replay.']]));

children.push(...experiment('X4', 'Do the engines retrieve what Bing or Google rank for their own queries?',
  'X1 compared the engines’ sources with Google for the question as typed. Now we know the engines’ real queries: does ChatGPT’s retrieval follow Bing or Google for them, and does Gemini’s follow Google?',
  'For 120 random questions, the first two queries of ChatGPT’s first search → Bing and Google (top 10, US); Gemini’s first two queries → Google. Compared with the pages each engine retrieved and cited for that answer.',
  X4.found,
  X4.conclude));

children.push(...experiment('X8', 'Deeper in Google and Bing, or not there at all?',
  'X4 compared the engines’ pages with the top 10 only. Do the engines take pages from further down the same search results, or pages the public search engines do not return at all?',
  'For 50 random ChatGPT answers and 40 Gemini answers (plus 10 and 8 from the niche market), every search query the engine sent was searched on Google down to the top 50 and on Bing down to the top 30 (about 2,900 searches; Google and Bing return 10 results per request). Each page the engine retrieved or cited was located at its best position over all of the answer’s queries.',
  [
    '**The pages are not further down — they are not there.** 97 % of the pages ChatGPT retrieved are in neither Google’s top 50 (niche market 99 %) nor, for 99 %, Bing’s top 30; going from the top 10 to the top 50 adds only 0.5 points.',
    '**The same holds for cited pages and for Gemini:** 95 % of ChatGPT’s and 93 % of Gemini’s cited pages are absent from Google’s top 50 for the engine’s own queries — although Gemini is grounded on Google Search.',
    '**The websites overlap, the pages do not:** for about half of the retrieved pages (51 %; niche 44 %), the website appears in Google’s top 50 for the same queries.',
  ],
  'The engines do not take a deeper slice of public search results; they use their own index or ranking, which selects different pages — often from the same websites Google and Bing would show. Public search rankings, even deep ones, are therefore not a proxy for AI retrieval.',
  [],
  [table(['Page not found in the public results (all of the answer’s queries)', 'Consumer market', 'Niche market'], [
    ['ChatGPT retrieved pages — Google top 50 / Bing top 30', '97.0 % / 98.9 %', '98.8 % / 94.0 %'],
    ['ChatGPT cited pages — Google top 50 / Bing top 30', '94.7 % / 97.7 %', '98.5 % / 90.9 %'],
    ['Gemini cited pages — Google top 50', '93.0 %', '92.4 %'],
    ['Website of ChatGPT’s retrieved page found in Google top 50', '51 %', '44 %'],
  ], [4960, 2200, 2200]), spacer()]));

children.push(...experiment('X6', 'Selection at equal rank: does content matter once a page is retrieved?',
  'X3 showed that ChatGPT cites mostly from the top of its own search results. Page content could act in two places: getting a page retrieved and ranked high, and — among the retrieved pages — being chosen over pages at the same rank. Which is it?',
  'Every page ChatGPT consulted in the replay searches (4,946 result lists, 139,420 page–question rows, 29,816 pages) was downloaded; PDFs were converted to text and Wikipedia articles taken from its official API (1,456 pages recovered); Reddit forbids crawlers and stays without text. For each row: rank and whether several searches found it; match to the question (best paragraph, title, BM25) and to ChatGPT’s own search queries; length, headings, the four tips (has vs not), list or comparison page, brand website, authority. Features are compared with the other pages of the same result list. (1) Selection: conditional logit of cited vs not cited within each list, adding feature groups step by step. (2) Retrieval: which features go with a higher rank inside a list. Rows without text stay in the models with a flag; a text-only version gives the same results.',
  [
    '**Rank decides most of the choice.** Rank alone explains 27 % of which consulted pages are cited (McFadden R²; 29 % with the text-missing flags). Adding all match features raises this only to 30 %, and all page features to 32.5 %. Without rank, all features together explain 17.5 %.',
    '**At the same rank, a better-matching page is somewhat more likely to be cited:** ×1.22 per SD of match to ChatGPT’s own queries (p < .001), ×1.12 for BM25, ×1.05 for best-paragraph and title match. Among ranks 1–3, 43 % of the best-matching third of pages are cited against 34 % of the worst-matching third; at ranks 4–10, 18 % against 14 %.',
    '**Other page properties matter at selection:** shorter pages (×0.59 per SD of length) and pages with more headings (×1.28) are preferred; brand and official websites ×1.71; **list pages ×0.54 and comparison pages ×0.34** — roundups are retrieved but lose the citation to the pages they summarise. Reddit is cited far less than its rank predicts (×0.30): consulted 30,690 times, cited 306 times (1.0 %); YouTube, Wikipedia and Tom’s Guide behave the same way.',
    '**The popular tips do not help at selection either; if anything they go with slightly lower odds:** author box ×0.77, FAQ ×0.93, structured data ×0.92 (all significant with the larger sample), summary box ×0.92 (not significant); outside authority ×0.97.',
    '**Retrieval: what ranks a page higher in ChatGPT’s search** is, above all, matching ChatGPT’s own reworded queries (+0.17 on −log₂ rank per SD, the strongest content feature — about 12 % closer to the top), then being a brand or official website (+0.63), being shorter and having more headings; matching the user’s question as typed adds little (+0.06). An author box goes with a lower rank (−0.40); FAQ blocks and summary boxes go with a slightly higher one (+0.11, +0.07) — which does not carry through to more citations within websites (R8).',
    '**Meaning, not keywords (X6b).** ChatGPT’s queries look like keyword searches, so we compared two ways a page can match them: by meaning (embeddings) and by shared words (BM25, per query and summed over the queries). Meaning match predicts a higher position (+0.22 per SD, p < .001); keyword match adds little once meaning is known (+0.05 per SD, significant with the larger sample) and on its own raises the explained share of rank only from 2.3 % to 2.4 % (meaning: 3.6 %). At equal rank, meaning match also goes with being cited (×1.24, p < .001), keyword match does not (×1.00–1.05, not significant). The niche market gives the same result. Text match of any kind explains only a small part of the order (about 5 %), so most of what decides a page’s position lies in the search index itself — which websites and pages it favours — not in the page text.',
    '**The niche market (Workorb, 231 lists) shows the same structure:** rank explains 29 % (34 % with the flags); question matching adds nothing measurable; page features add 7 points — shorter pages, brand websites (×1.80) and, again, list (×0.20) and comparison pages (×0.14) losing at equal rank. Matching ChatGPT’s own queries is again the content feature that goes with a higher rank.',
  ],
  'Content acts **mainly through retrieval**: pages that match the queries the engine actually sends are ranked higher, and rank then decides most of the citation. At the citation step itself content adds a modest second effect — at the same rank, concise, well-structured pages that directly answer the question, and pages from the brand or original source, beat roundups, forums and long pages. The popular tips play no role at either stage.',
  [['x06_selection.png', 'Left: share of the choice among consulted pages explained as feature groups are added. Right: % cited by within-list match to the question (thirds), at equal rank.']],
  [table(['Model of which consulted pages are cited (4,841 lists)', 'Explained (McFadden R²)'], [
    ['Rank and "found by several searches"', '26.8 %'],
    ['+ flags for pages without text (PDF errors, Reddit, 403)', '29.3 %'],
    ['+ match to the question', '29.7 %'],
    ['+ match to ChatGPT’s own search queries', '29.9 %'],
    ['+ page features (length, headings, tips, page type, brand, authority)', '32.5 %'],
    ['All features without rank', '17.5 %'],
  ], [6360, 3000]), spacer(),
  table(['Match to ChatGPT’s own queries (X6b)', 'Correlation with a higher rank (GFiber / Workorb)', 'Adds to rank model (GFiber)'], [
    ['By meaning (embeddings)', '0.20 / 0.39', '+0.22 per SD, p < .001'],
    ['By shared words, best query (BM25)', '0.08 / 0.32', '+0.05 per SD, p < .001'],
    ['By shared words, all queries (BM25 sum)', '0.08 / 0.33', 'negative once meaning is included'],
    ['For reference: user’s question by meaning / by words', '0.17 / 0.28 and 0.09 / 0.21', '—'],
  ], [3960, 2900, 2500]), spacer()]));

children.push(...experiment('X7', 'How the engines reword a question into search queries',
  'X6 showed that matching the engine’s own search queries — not the question as typed — is what ranks a page higher. What do those queries look like?',
  'All search queries saved in the replays (ChatGPT: 5,394 searched answers, about 24,000 queries; Gemini: 4,723 answers). Per query: length, share of the question’s words kept and of new words, meaning similarity to the question (embeddings), and whether it adds a year, a brand the question did not name, price or plan words, a site name or a search operator; for repeated runs of the same question, the share of identical queries.',
  [
    '**ChatGPT turns a question into about four short keyword queries, one per entity or aspect.** Only 43 % of the question’s words are kept and 60 % of the query words are new (Gemini 53 % and 42 %).',
    '**It makes queries more specific than the question:** it adds the current year to 23 % of queries (prices, plans, reviews), a brand the question did not name to 28 %, price or plan words to 10 %, and uses site names or operators (site:, quotes) in about 5–7 %.',
    '**It searches in two rounds:** first generic queries, then queries naming the specific vendors or products found in the first results.',
    '**Every answer starts from new queries:** only 3 % of queries repeat between two runs of the same question, and under half of their words overlap.',
    '**Gemini rephrases instead:** natural-language sub-questions close to the user’s wording (similarity 0.78 vs 0.66 for ChatGPT), often yes/no questions, rarely adding years or brands; 20 % of its queries repeat between runs (14 % in the niche dataset).',
  ],
  'The engines do not search for the question the user asked. ChatGPT searches for narrow, keyword-style facts about named brands, plans and products — often with the current year — and learns new names from its first results. This is why matching the engine’s queries predicts rank better than matching the question, why pages about one named product with concrete facts are retrieved, and why the set of retrieved pages keeps changing (E6).',
  [],
  [table(['Question', 'ChatGPT search queries (one run)', 'Gemini search queries'], [
    ['Is Spectrum’s internet cheaper than Google Fiber after the first year?',
     'Spectrum internet pricing after 12 months promo rate increases 2026 · Google Fiber pricing after 12 months promo rate increases · Spectrum Select 300 Mbps price for new customers term 12 months · Google Fiber 2026 pricing 2 year rates promotional pricing',
     'Google Fiber internet pricing after 12 months · Spectrum internet pricing after 12 months · is Spectrum cheaper than Google Fiber after first year · does Google Fiber raise prices after first year · does Spectrum raise prices after first year'],
    ['Which fiber companies have dedicated, faster customer service lines for business accounts?',
     'fiber internet business customer service dedicated phone number business accounts · AT&T Business Internet customer service phone number for business accounts · Comcast Business phone support number dedicated lines · Verizon Fios business customer service phone number',
     'top fiber internet providers for business with dedicated support · fiber internet providers with dedicated business customer service lines · do business fiber internet providers offer faster customer support for business accounts'],
  ], [2400, 3780, 3180]),
  spacer(),
  table(['Rewording (GFiber)', 'ChatGPT', 'Gemini'], [
    ['Queries per answer (median)', '4', '3'],
    ['Question words kept / query words new', '43 % / 60 %', '53 % / 42 %'],
    ['Meaning similarity to the question', '0.66', '0.78'],
    ['Adds a year / a new brand / price or plan words', '23 % / 28 % / 10 %', '2 % / 5 % / 3 %'],
    ['Site name or search operator', '7 % / 5 %', '1 % / 0 %'],
    ['Identical queries in two runs of the same question', '3 %', '20 %'],
  ], [4360, 2500, 2500]), spacer()]));

children.push(...experiment('X9', 'Links to the page itself (page-level authority)',
  'R14 measured authority per website. Do links pointing to the individual page go with more citations, when pages of the same website are compared?',
  'For every page in the models and every page ChatGPT consulted (about 35,000 pages), the number of other websites linking to the page (referring domains) and its link-based rank were obtained from the DataForSEO Backlinks API (a commercial link index approximating the data search engines use). Each page was looked up under its possible address forms (with/without www. and trailing slash), keeping the best match. Measures added to the R2 model (website indicators) and to the X6 ranking and selection models.',
  [
    '**Links to the page go with more citations within websites:** +23 % per SD of referring websites (rank +16 %), and +39 % for a page with any referring website against one with none — as large as within the pooled comparison (+20 %), so unlike the "AI optimisation" tips it does not disappear when the website is held fixed. Brand pages +31 %, third-party pages +17 %.',
    '**It adds to question matching rather than replacing it** (title and best paragraph keep +15 % and +17 %), and raises the explained within-website variation from 4.4 % to 6.3 %.',
    '**It is mainly a Gemini effect:** +34 % per SD for Gemini, +7 % for Perplexity, +4 % for ChatGPT — Gemini is grounded on Google Search, whose ranking is built on links.',
    '**Inside ChatGPT’s own ranking links matter little:** pages with any referring website sit somewhat higher in its result list, but the fit improves only from 32.3 % to 32.9 %, and at equal rank links do not affect which page is cited.',
    '**The niche market agrees** (+16 % per SD, +41 % for any link; Gemini +17 %, ChatGPT +8 %), with the effect concentrated on brand pages (+42 %).',
  ],
  'Page-level links are, next to question matching, the one page property that survives the within-website comparison — mostly through Gemini. The links were measured in September 2026, after the citations, so some pages may have gained links because they were visible; the direction is not established. Most cited pages have no external links at all (60 % in the consumer market).',
  [],
  [table(['Page-level links (within websites)', 'Consumer market', 'Niche market'], [
    ['Per SD more referring websites', '+23 %', '+16 %'],
    ['Any referring website vs none', '+39 %', '+41 %'],
    ['By engine: Gemini / ChatGPT / Perplexity (per SD)', '+34 % / +4 % / +7 %', '+17 % / +8 % / +4 %'],
    ['For comparison: title matches the question (per SD)', '+15 %', '+27 %'],
  ], [4960, 2200, 2200]), spacer()]));

children.push(...experiment('X10', 'The model’s prior: what the model already knows',
  'Do the brands and websites a model knows from pretraining shape what it searches for, how its results are ranked and what it cites — beyond the search results themselves?',
  'Each of the 706 questions was sent to the model behind each engine’s search — gpt-5.4-nano (ChatGPT) and Gemini 3.1 Flash Lite — without web search, asking which brands or products it would mention and which websites it would cite (up to 10 each). This is the model’s prior. Tests: (1) is a tracked brand the question does not name added to the engine’s search queries more often when it is in the prior; (2) for ChatGPT, are pages from websites in the prior ranked higher and cited more at equal rank (X6 data, all X6 features controlled); (3) what share of each engine’s 90-day citations comes from websites in the prior for that question, compared with the prior of a different question (a baseline for general popularity).',
  [
    '**ChatGPT writes its searches around the brands it knows.** A brand the user did not name is added to its queries in 67 % of cases when it is in the model’s prior, against 11 % when it is not; 34 % of the prior’s brands appear in the queries. Gemini also draws on its own prior (18 % vs 0.7 %) but rarely adds brands at all.',
    '**Known websites rank higher and win at equal rank (ChatGPT).** A quarter of consulted pages come from websites in the prior (48 % of the cited ones, 22 % of the rest). With all X6 features controlled — content match, brand role, authority — they sit 1.4× closer to the top and are 1.6× as likely to be cited at the same rank (p < .001).',
    '**Nearly half of ChatGPT’s citations come from websites its model would name without searching:** 47 % for the consumer market (25 % with another question’s prior) and 32 % for the niche market (9 %). Gemini’s citations are only weakly tied to its prior (11 % vs 6 %) — it cites many sources per answer, taken from Google Search grounding.',
    '**The two models know different markets.** Both know the large internet providers (Google Fiber, AT&T, Verizon: 61–78 % of questions). For AEC proposal software, gpt-5.4-nano almost never names the tracked vendors (Unanet 4 %, Loopio 16 %, OpenAsset 2 %) and lists general construction software instead, while Gemini knows them well (96 %, 78 %, 60 %). Neither model knows Workorb or Joist AI (0–2 %).',
  ],
  'ChatGPT’s citations combine its search results with its own prior knowledge: the model writes its queries around brands it knows, and pages from websites it knows rank higher and are preferred at the same rank. Well-known consumer brands benefit. Brands a model does not know must be found by search — in the niche market ChatGPT learns vendor names from its first round of results (X7). Being known to the models is therefore part of AI visibility, and it differs by model.',
  [],
  [table(['Model’s prior (share of questions naming the brand)', 'gpt-5.4-nano (ChatGPT)', 'Gemini 3.1 Flash Lite'], [
    ['Google Fiber / AT&T / Verizon', '72 % / 71 % / 61 %', '78 % / 66 % / 63 %'],
    ['Unanet / Loopio / OpenAsset (niche market)', '4 % / 16 % / 2 %', '96 % / 78 % / 60 %'],
    ['Workorb / Joist AI (niche market)', '2 % / 0 %', '0 % / 0 %'],
  ], [4360, 2500, 2500]), spacer(),
  table(['Share of 90-day citations from websites in the prior for that question', 'ChatGPT (own prior)', 'Gemini (own prior)'], [
    ['Consumer market (another question’s prior)', '47 % (25 %)', '11 % (6 %)'],
    ['Niche market (another question’s prior)', '32 % (9 %)', '7 % (4 %)'],
  ], [4360, 2500, 2500]), spacer()]));

children.push(...experiment('X11', 'Brand level: what makes an answer name a brand?',
  'The analysis so far is about which pages are cited. A brand can also be named without its own page being cited — through a third-party page about it, or from the model’s own knowledge. What supports a brand mention?',
  'Unit: one answer × one tracked brand that the question does not name (1.2 million cases; the pattern-based mention agrees with the collection’s own mention records in 96 %). For each case: whether the brand’s own website is cited, whether a cited third-party page names the brand (from its downloaded text), and whether the brand is in the engine’s model prior (X10). Share of mentions by support, and a logistic model per engine with brand indicators.',
  [
    '**A cited own page makes the brand named almost always** (97–100 %), but most mentions are not backed by the brand’s own page: 39 % (ChatGPT) and 43 % (Gemini) are.',
    '**Third-party pages carry many mentions:** 15 % of ChatGPT’s and 39 % of Gemini’s mentions are supported only by another website’s page about the brand; such a page raises the odds of a mention ×4 (ChatGPT) to ×6 (Gemini) — in the niche market ×40 and ×18, where the models do not already know the brands.',
    '**ChatGPT names many brands from memory:** 46 % of its mentions are backed by no cited page naming the brand (35 % even when the text of every cited page is known), and 81 % of those brands are in its model’s prior. Perplexity cites in only 4 % of answers, so 95 % of its mentions have no supporting citation.',
    '**Being known to the model multiplies the chance of being named** at equal citation support: ×5.8 (ChatGPT) and ×9.7 (Gemini). A brand with only a third-party page cited is named in 51 % of ChatGPT answers when the model knows it and in 12 % when it does not.',
  ],
  'Brand visibility rests on two supports: being cited — through the brand’s own pages or, often, third-party pages about it — and being known to the model. Third-party coverage matters most for brands the models do not know; the model’s prior lifts known brands even without any supporting citation.',
  [],
  [table(['Share of a brand’s mentions supported by…', 'ChatGPT', 'Gemini', 'Perplexity'], [
    ['Its own page cited', '39 %', '43 %', '1 %'],
    ['A third-party page naming it (no own page)', '15 %', '39 %', '4 %'],
    ['No cited page naming it', '46 %', '18 %', '95 %'],
    ['…of which the brand is in the model’s prior', '81 %', '84 %', '—'],
  ], [4360, 1660, 1660, 1680]), spacer()]));

children.push(...experiment('X12', 'Position: does the order of the sources change what is cited?',
  'ChatGPT cites the page it ranks first far more often than pages further down (X3). Is that because the top page is better, or simply because it comes first? A position effect would inflate the rank curve without saying anything about the page.',
  'For 100 questions (80 consumer, 20 niche), 8 real sources from the engines’ replay lists, each shown as its title and the three paragraphs closest to the question (at most 90 words each) under a random three-letter ID. Each source list was shown in all 8 cyclic orders, so every source appears in every slot equally often; the model was told to answer from the sources only and to cite them by ID. Two further versions add a visible rank label, either matching the slot or assigned independently of it, to separate placement from the label. Two repeats per order, two models without web search (gpt-5.4-nano for ChatGPT, gemini-3.1-flash-lite for Gemini): 9,600 answers. Effects are measured for the same source in different slots (source fixed effects, confidence intervals by question).',
  [
    '**ChatGPT’s model favours sources shown early; Gemini’s does not.** The same source is cited in 60 % of answers when shown first and in 43 % when shown last: **+17.6 points** [15.1–20.3], falling steadily down the list. For Gemini the rate stays at 57–59 % in every slot (first − last −0.4 points [−2.3 to 1.6]). Both markets show the same pattern (ChatGPT +18.0 consumer, +15.9 niche; Gemini −0.8 and +1.2).',
    '**For ChatGPT it is the slot, not a label.** With rank labels assigned independently of the order, being placed in the first two slots adds 11 points, while carrying one of the top two labels adds only 3. Labels that match the order barely add to the plain position effect (19 vs 18 points).',
    '**The engine’s own ranking still matters when the slot is randomised.** Among the 8 sources, the one the production engine ranked highest is cited more often than the one it ranked lowest even though both appear in every slot equally often: 55 % vs 43 % (ChatGPT) and 67 % vs 50–56 % (Gemini). Sources the production engine actually cited are cited in 70–74 % of answers, the others in 41–54 %.',
    '**Brand recommendations barely depend on the order.** Across the 8 orders of the same sources, the brands an answer names overlap 80–87 %, the first brand named is the same in 91–93 % of answers, and a brand’s place in the answer is only weakly related to the slot of its best source (Spearman 0.15–0.17).',
  ],
  'Part of ChatGPT’s steep rank curve is a presentation effect: pages near the top of its list are cited more partly because they come first — about 18 points across 8 slots in this setting. The rest is real: the engine’s ranking also picks the sources the model finds more worth citing. For Gemini, order alone has no effect, so its preference for top-ranked pages reflects the pages. For a publisher, the conclusion is the same as in X3 and X6, only stronger for ChatGPT: rank near the top of the engine’s own search, because the top position is worth something on its own. Limits: small models without search, 8 sources rather than the 20–40 of production, and short excerpts; the sizes describe this setting, not the production engines.',
  [['x12_position.png', 'Left: citation rate of the same sources when shown in each slot. Right: citation rate by the source’s rank in the engine’s replay list (among the 8), with the slot randomised. 9,600 answers, 100 questions.']],
  [table(['Cited when shown in slot', '1', '2', '3', '4', '5', '6', '7', '8'], [
    ['ChatGPT (gpt-5.4-nano)', '60 %', '58 %', '53 %', '51 %', '46 %', '45 %', '42 %', '43 %'],
    ['Gemini (3.1-flash-lite)', '58 %', '59 %', '58 %', '58 %', '58 %', '59 %', '57 %', '59 %'],
  ], [2960, 800, 800, 800, 800, 800, 800, 800, 800]), spacer()]));

children.push(...experiment('X14', 'Brand and publisher names: facts, brand identity or the website behind the evidence?',
  'When a model prefers one product, how much of that comes from the facts it is given, from the brand’s name, and from the name of the website supporting it? Does an unknown brand benefit from being backed by a well-known publisher?',
  '60 packets built from real material: one real question (40 consumer comparison questions, 20 niche vendor-choice questions), five competing products and one excerpt per product from a third-party website. Each product is described by a fixed fact sheet written from its own website in the same seven fields and 150–195 words, with the company referred to only by a placeholder; each excerpt is a real paragraph about that product, chosen as the one closest to the question, with publishers alternating between well-known sites (e.g. CNET, PCMag, USA Today) and little-known ones. Brand identity and publisher identity were then hidden or shown independently (2×2): a hidden brand appears as "Vendor K" everywhere, including inside the excerpts; a hidden publisher appears as "not shown", with its own name inside the excerpt replaced by "this site" and other publications named by "another site". Facts, wording, order and length are identical across the four versions. The model ranked the five products for the question and named the excerpts supporting its top three. Each packet was shown in two orders (X12) and twice, to two models without search (gpt-5.4-nano, gemini-3.1-flash-lite): 1,920 answers. In a hypothetical version, the prompt stated that the profiles were hypothetical and complete and that the names had been assigned at random and carry no information; the real fact sheets were shown with the brand names rotated over them (1,200 answers), and in a further check five identical profiles carried five different names (1,200 answers). A separate test asked both models to name the company behind each hidden fact sheet.',
  [
    '**Hiding the name does not make a well-known brand anonymous.** From the hidden fact sheet alone, Gemini’s model names the company in 52–56 % of cases for the internet providers and 33 % for the niche vendors, ChatGPT’s in 26–33 % and 13–20 % (chance about 10 %); Astound, T-Mobile, Verizon, Xfinity and Deltek are recognised by their facts. The unknown vendors (Workorb, Arphie, AutoGenAI, Responsive) are never recognised.',
    '**Showing brand names reorders the ranking far more than asking again.** Every answer puts exactly three of the five products in its top three, so showing the names cannot raise all brands at once; it moves some up and others down. The average change in a product’s top-three rate when its name is shown is 26 points (ChatGPT) and 24 points (Gemini), against 12 and 7 points when the same prompt is simply repeated: 2.1× [1.8–2.6] and 3.3× [2.5–4.5] as much. Showing publisher names changes the ranking much less (1.3× and 1.6×).',
    '**Who gains and who loses on the same facts:** Spectrum −33 points (ChatGPT) and −19 (Gemini); Inventive AI −19 and −16; Arphie −13 (ChatGPT). Loopio +16 with both models, Google Fiber +12, Unanet +11, Metronet +10 to +13, Quantum Fiber +9 and Verizon +7 (ChatGPT). Workorb +6 to +8, not significant. Spectrum’s profile ranks well while it is anonymous; with the name shown, the models apparently fall back on knowing that it is a cable company.',
    '**Names still matter when the model is told they mean nothing.** In the hypothetical version, the same fact sheet reaches ChatGPT’s top three in 80 % of answers under the name Google Fiber and in 39 % under Xfinity (a 41-point spread between names, p < .001; niche vendors: Loopio 74 %, Arphie 44 %). Gemini’s spread is 18 points for the providers (p < .001) and not significant for the niche vendors.',
    '**The two results line up.** A brand’s gain when its name is shown in the main experiment follows its name effect in the hypothetical version (Spearman 0.81 for ChatGPT, 0.49 for Gemini), but not whether its hidden profile could be recognised (0.40 and 0.08, not significant). When all five profiles are identical, however, both models simply take the first three listed (95–100 %): the name does not break a pure tie, it changes how the facts are weighed.',
    '**Publisher names matter little.** ChatGPT relies on an excerpt 7.6 points more [3.0–12.2] when a well-known publisher is named than when a little-known one is, but this does not carry through to the product’s place in the top three (+2, not significant); Gemini shows no publisher effect. An unknown brand does not gain more from a well-known publisher than a known brand does (+5 points, p = .18).',
  ],
  'The models do not rank products on the supplied facts alone. A brand’s name brings in what the model already believes about it — here most clearly the difference between fiber and cable providers — and that belief moves products up or down the ranking even on identical facts and even when the model is told that names carry no information. For niche vendors the direction differs from vendor to vendor. The website behind the evidence counts for much less. This fits X10 and X11: the model’s prior knowledge of a brand shapes the answer beyond what it retrieves. Limits: small models without search, constructed packets rather than production answers, and only 10 packets per niche vendor, so their intervals are wide.',
  [['x14_identity.png', 'Left: change in a brand’s top-three rate when its name is shown (same facts), against its name effect in the hypothetical version; one point per brand and model. Right: how much showing brand or publisher names changes top-three rates, relative to repeating the same prompt (95 % CI by packet).']],
  [table(['Change in top-3 rate when the name is shown (points, same facts)', 'ChatGPT', 'Gemini'], [
    ['Spectrum', '−33 [−46, −20]', '−19 [−33, −7]'],
    ['Google Fiber', '+12 [5, 19]', '+4 (n.s.)'],
    ['Metronet', '+10 [2, 20]', '+13 [3, 23]'],
    ['Verizon', '+7 [2, 13]', '+5 (n.s.)'],
    ['Loopio', '+16 [1, 35]', '+16 [3, 33]'],
    ['Unanet', '+11 [1, 25]', '−1 (n.s.)'],
    ['Workorb', '+6 (n.s.)', '+8 (n.s.)'],
    ['Inventive AI', '−19 [−31, −5]', '−16 [−31, −3]'],
    ['Arphie', '−13 [−25, −3]', '+1 (n.s.)'],
  ], [4960, 2200, 2200]), spacer()]));

children.push(...experiment('X13', 'Is there a stable brand order in real answers?',
  'When an answer names several brands, is their order consistent from day to day, or does it depend on which other brands appear and on the date? Can one score per brand describe it, and at what level: overall, per topic or per question?',
  'For every collected answer, the tracked brands named in it were put in the order of their first mention, leaving out brands the question itself names. Every pair of brands named in the same answer gives one comparison (which is named first). Per question and pair with at least 10 answers in common we measured: how often the same brand comes first; majority cycles (A before B, B before C, C before A), kept only if they survive resampling of the answers; whether the presence of a third brand changes the pair’s order; and whether the order changes between the first and second half of the 90 days. Finally, Bradley–Terry models (one score per brand) fitted at three levels — engine-wide, per topic and per question — and a pair-by-pair majority without any single score were fitted on a random half of the days and tested on the other half, and on the last 30 days after fitting on the first 60.',
  [
    '**Pair order is mostly consistent:** for a given question, the same brand is named first in 78–81 % of the answers that name both; 31–40 % of pairs are near-unanimous (at least 90 %) and 13–16 % are contested (under 60 %).',
    '**Almost no cycles:** 0.5 % (Gemini), 4 % (ChatGPT) and 6 % (Perplexity) of questions show a majority cycle, and only one of them (ChatGPT) survives resampling.',
    '**Other brands barely change a pair’s order:** whether a third brand is named changes the order of a pair significantly in 0.2 % (Gemini), 1 % (ChatGPT) and 6 % (Perplexity) of cases, with actual reversals in 0.2–3 %.',
    '**Stable over the 90 days for Gemini and ChatGPT, not for Perplexity:** between the two halves, pair orders change significantly in 0 % (Gemini), 0.3 % (ChatGPT) and 16 % (Perplexity) of pairs.',
    '**One score per brand works only per question.** On held-out days, one score per brand and question predicts 74 % (ChatGPT), 77 % (Gemini) and 79 % (Perplexity) of pair orders — as well as a pair-by-pair majority (73–79 %) — while one score per brand for all questions predicts only 64–67 %, and one per topic hardly more (65–68 %). Training on the first 60 days and testing on the last 30 gives the same pattern.',
  ],
  'Within a question, the order in which an engine names brands is stable, transitive and largely independent of the other brands present, and for ChatGPT and Gemini it holds over three months. Across questions it is not: a brand’s standing on one question says little about the next, even within a topic. A brand’s prominence should therefore be modelled per question or per narrowly defined type of question, not as one number per brand. Limits: the order of first mention measures prominence rather than an explicit recommendation, and in real answers the sources change from day to day, so this cannot separate the model’s preference from its evidence — the controlled experiment X15 does.',
  [['x13_rank_models.png', 'Held-out pairs of brands whose order (first mention) is predicted correctly, by the level at which the brand score is fitted; fitted on a random half of the days, tested on the other half.']],
  []));

children.push(...experiment('X15', 'Controlled ranking stability: one score per brand, or does it depend on the alternatives and the wording?',
  'X13 found a stable order per question in real answers, where the evidence changes every day. With the evidence held fixed, can a model’s choices be represented by one stable score per brand, or do they depend on which other products are present, on the order they are listed in, and on the exact wording of the question?',
  'For 20 questions (12 consumer, 8 niche), six brands each were described by their fixed fact sheets from X14 under their real names. For every question the model (gpt-5.4-nano and gemini-3.1-flash-lite, no search) answered: all 15 pairs, each shown in both orders, three times; the full ranking of all six in six different orders, twice; the ranking with each brand removed, once leaving the gap and once filling it with a neutral “general buying considerations” note of the same length, so that fewer products does not also mean less text; the ranking with a seventh brand added (three different ones); and the ranking under three equivalent rewordings of the question, in six orders: 6,000 answers. A Bradley–Terry model (one score per brand) fitted on the pairwise answers alone was tested on the full, reduced, extended and reworded rankings and on held-out pairs; a pair-by-pair majority without a single score and the agreement between two full rankings are the references. Gemini declined to choose in 9 % of the pairwise questions; these are left out.',
  [
    '**Pairwise choices are consistent and transitive.** Over the six answers per pair, the majority choice gets 90 % (ChatGPT) and 94 % (Gemini) of them, and 72 % and 87 % of pairs are unanimous. The product shown first wins 53 % [51–55] and 50 % of pairs. Majority cycles occur in 2 of 400 brand triples (ChatGPT) and none (Gemini); none survives resampling.',
    '**One score per brand predicts the rankings as well as anything we tried.** Fitted on the pairs only, it orders 82 % (ChatGPT) and 77 % (Gemini) of the pairs in the full rankings correctly — as well as a pair-by-pair majority (84 % and 81 %) and as well as one full ranking predicts another shown in a different order (81 % and 79 %). It predicts held-out pairs (90 %, 86 %), the rankings with a product removed (82 %, 78 %) or added (78 %, 76 %), and those for reworded questions (81 %, 77 %) about equally well.',
    '**Other products hardly matter.** A pair ordered the same way in all twelve full rankings is reversed in 2.2 % (ChatGPT) and 0.6 % (Gemini) of rankings when another product is removed, 2.8 % and 0.9 % when one is added, against 1.5 % and 0.9 % when the same prompt is simply repeated. Replacing the removed product by text of the same length changes nothing (2.2 % and 0.7 %). A strong pairwise preference (at least 5 of 6 answers) is reversed in the full ranking in 7 % and 10 % of cases.',
    '**Rewording matters little; the listing order matters most.** Rankings of the same six products agree (Kendall τ) 0.77 (ChatGPT) and 0.94 (Gemini) between identical repeats and 0.76 and 0.84 between the original and a reworded question, but only 0.62 and 0.53 when the products are listed in a different order. For ChatGPT, rewording changes the ranking no more than repetition (difference −0.01 [−0.05, 0.02]); Gemini’s answers are almost deterministic, so rewording adds a small but real change (−0.10 [−0.14, −0.06]). The product listed first reaches an average rank of 3.0 (ChatGPT) and 2.9 (Gemini), the one listed last 3.9 and 4.2 — unlike Gemini’s citations in X12, which did not depend on the order.',
    '**Both markets show the same pattern.** For ChatGPT, the order effect is stronger among the niche vendors (listed first: average rank 2.6, listed last 4.0), so rankings shown in different orders agree less there (75 % of pairs, against 85 % for the internet providers).',
  ],
  'With the evidence fixed, a model’s choices for a question are well described by one score per brand: they are transitive, depend very little on which other products are present, and survive rewording. The main instability is the order in which the products are presented, followed for ChatGPT by the randomness of repeated answers. Together with X13 this answers what a prediction system needs: a score per brand and question (or narrowly defined question type), averaged over presentation order — in production, that order is set by the engine’s retrieval — rather than one number per brand or a model of the entire candidate set. A “top-three probability” is therefore a property of a brand on a given question, with its competitors and the engine’s retrieval, not of the brand alone. Limits: small models without search and fixed fact sheets; 20 questions.',
  [['x15_stability.png', 'Left: agreement of two rankings of the same six products when the prompt is repeated, the question reworded, or the products listed in a different order. Right: share of pairs in the full rankings predicted by one score per brand fitted on pairwise answers, by the pair-by-pair majority, and by another full ranking in a different order.']],
  [label('How to read “one score per brand predicts the rankings”.', 'For each question, the model was asked about every pair of the six brands (“which is better for this question, A or B?”), each pair in both orders and three times. From these answers alone we fit one strength score per brand (a Bradley–Terry model); the higher score is expected to win every pair, so the scores also imply a full ranking. An illustrative set of scores for the question “lowest latency for gaming”:'),
    table(['Brand (illustrative)', 'Score'], [
      ['Google Fiber', '+1.8'], ['AT&T Fiber', '+1.1'], ['Verizon', '+0.6'],
      ['Metronet', '+0.2'], ['Spectrum', '−1.4'], ['Xfinity', '−2.3'],
    ], [4960, 2200]),
    spacer(),
    P('**The test.** Separately, the model ranked all six brands at once, twelve times, with the products listed in different orders. A ranking of six contains 15 pairs (who is placed above whom); we count how many of them the scores get right. On average they match **82 %** of the pairs for ChatGPT and **77 %** for Gemini, although they were fitted without ever seeing a full ranking.'),
    P('**The reference.** The model does not give the same full ranking every time, especially when the listing order changes. Comparing the model with itself — one full ranking against another for the same question shown in a different order — the two agree on **81 %** (ChatGPT) and **79 %** (Gemini) of the pairs. That is about the best any prediction can do: if the model disagrees with itself on one pair in five just because the list was reordered, no predictor can get much above 80 %.'),
    P('**The conclusion.** The single score reaches that level (82 % vs 81 %, 77 % vs 79 %), so describing the model’s preferences on a question with one number per brand loses essentially nothing; what remains is the model’s own inconsistency, mostly from the listing order. Had the preferences depended strongly on which other brands were present, or contained cycles (A > B > C > A), one number per brand could not reproduce the full rankings and would fall well below the model’s self-agreement. The caveat is that this holds per question: a brand’s score on one question says little about its score on another (X13).'),
    table(['Controlled stability (X15)', 'ChatGPT', 'Gemini'], [
    ['Majority share of the six answers per pair', '90 %', '94 %'],
    ['Majority cycles among 400 brand triples (robust)', '2 (0)', '0 (0)'],
    ['Full-ranking pairs predicted by one score per brand', '82 %', '77 %'],
    ['… by the pair-by-pair majority / by another full ranking', '84 % / 81 %', '81 % / 79 %'],
    ['Held-out pairs predicted by one score per brand', '90 %', '86 %'],
    ['Unanimous pairs reversed: product removed / added / repeat', '2.2 % / 2.8 % / 1.5 %', '0.6 % / 0.9 % / 0.9 %'],
    ['Ranking agreement (τ): repeat / reworded / other order', '0.77 / 0.76 / 0.62', '0.94 / 0.84 / 0.53'],
    ['Average rank of the product listed first / last', '3.0 / 3.9', '2.9 / 4.2'],
  ], [4960, 2200, 2200]), spacer()]));

children.push(...experiment('X16', 'Ranking stability with web search: does the conclusion of X15 hold when the model finds its own evidence?',
  'X15 gave the model fixed fact sheets. In real use the engine searches the web and every answer finds its own evidence. Is a brand’s standing still one stable score per question when the model searches — and how much of the instability comes from the evidence rather than the model?',
  'The 20 questions and six-brand sets of X15 were repeated in three arms that differ only in the evidence the model uses: the X15 fact sheets in the message (“fixed facts”); web search, with the same models and search tools as the replay and every query and source logged (“search”); and neither, so the model answers from what it knows (“memory”). All three arms use the same wording — “… briefly explain, and finish with one line of JSON” — because with a JSON-only prompt Gemini did not search at all. Per set and arm: all 15 pairs in both orders twice, the full ranking in six orders twice, each brand removed, three brands added, and three rewordings in two orders. 10,440 answers; the X15 measures are computed unchanged on each arm.',
  [
    '**The answer format is not the issue.** The fixed-facts arm with the new wording reproduces X15 (ChatGPT: agreement between identical prompts τ 0.75 vs 0.77; one score per brand predicts 81 % vs 82 % of the pairs). So the differences between the arms below come from the evidence.',
    '**The structure of X15 holds with search.** One score per brand predicts 78 % (ChatGPT) and 85 % (Gemini) of the pairs in the full rankings and 85 % of held-out pairs; cycles stay rare (none robust for ChatGPT, one for Gemini); removing or adding a brand, or rewording the question, changes the ranking hardly more than repeating the prompt.',
    '**Consistency depends on the evidence.** Two identical prompts agree with τ 0.75 for ChatGPT on fixed facts, 0.56 with search and 0.51 from memory alone (Gemini: 0.89, 0.76, 0.82). With search, two identical prompts share only 2 % (ChatGPT) and 11 % (Gemini) of their queries and about 24 % of the pages they consult, and the more pages they share, the more their rankings agree (ChatGPT p = .001, Gemini p = .008). From memory alone, ChatGPT is least consistent and has the most cycles (5 % of brand triples).',
    '**The listing order matters much less with search.** For Gemini, rankings of the same brands in a different order agree with τ 0.72 with search, against 0.53 on fixed facts: when the products are only named, the model relies on what it finds or knows, not on the order of the profiles in front of it.',
    '**Known and unknown brands behave in opposite ways.** For the internet providers, which the models know, answers from memory alone are already consistent (Gemini: one score predicts 92 % of pairs) and search adds some noise. For the niche vendors, which the models barely know, answers from memory are inconsistent (ChatGPT τ 0.36, 64 % of pairs predicted; Gemini 67 %) and search restores consistency (ChatGPT 0.59 and 80 %; Gemini 80 %).',
  ],
  'Searching does not change the conclusion of X15: within a question, a brand’s standing is one stable score, with no cycles beyond noise and little dependence on the other brands or the wording. What search changes is how noisy the answers are, and that depends on the evidence each answer happens to find. For a brand the models know, the model’s memory alone yields a consistent ranking; for a brand they do not know, the ranking is consistent only when search finds evidence about it. Limits: small models and 20 questions; the search arm logs what was retrieved but not how each page was read.',
  [['x16_search.png', 'Left: agreement of two identical prompts (Kendall τ) with fixed facts (X15 and the X16 wording), web search and memory only. Right: share of pairs in the full rankings predicted by one score per brand fitted on pairwise answers.']],
  [table(['Ranking stability (20 questions)', 'Fixed facts', 'Web search', 'Memory only'], [
    ['ChatGPT: identical prompts agree (τ)', '0.75', '0.56', '0.51'],
    ['ChatGPT: full rankings predicted by one score per brand', '81 %', '78 %', '70 %'],
    ['ChatGPT: brand triples with a cycle (robust)', '0.3 % (0)', '1.5 % (0)', '5 % (1)'],
    ['Gemini: identical prompts agree (τ)', '0.89', '0.76', '0.82'],
    ['Gemini: full rankings predicted by one score per brand', '78 %', '85 %', '82 %'],
    ['Gemini: products listed in a different order (τ)', '0.64', '0.72', '0.71'],
    ['Niche vendors, ChatGPT: identical prompts (τ) / predicted', '0.69 / 79 %', '0.59 / 80 %', '0.36 / 64 %'],
  ], [4360, 1660, 1660, 1680]), spacer()]));

children.push(
  H2('6.1 What the 75 % consists of'),
  table(['Part of the variation', 'Evidence', 'Size'], [
    ['Day-to-day chance', 'R13 (odd vs even days)', 'about 10 %'],
    ['Drift: pages entering and leaving the engines’ search results over weeks', 'E6', 'within a 14-day window 65 % is repeatable; after 70 days 14–29 % remains (ChatGPT, Gemini)'],
    ['Retrieval: rank in the engine’s own search for its reworded queries', 'X3, E2', 'explains 32 % of ChatGPT’s 90-day within-website variation; about 80 % of what is predictable two weeks apart'],
    ['Selection among retrieved pages', 'X3, X6', 'follows retrieval rank (44 % cited at rank 1, 4 % at 11–20; rank explains 27 % of the choice); at equal rank content adds a little (+3 points): better match to the engine’s queries, shorter and structured pages, brand sources; roundups and forums lose; the popular tips do not help'],
    ['Position in the list the model reads', 'X12', 'ChatGPT’s model cites the same source 60 % of the time when shown first and 43 % when shown last (+18 points); Gemini’s shows no position effect. The engine’s ranking still predicts citation when the slot is randomised'],
    ['Stability of the brand order', 'X13, X15, X16', 'within a question the order is stable, transitive and hardly depends on the other brands (one score per brand and question predicts 74–79 % of held-out pairs in real answers, 77–82 % with fixed evidence, 78–85 % with web search); across questions it is not (64–67 % with one score per brand); with fixed evidence the listing order moves rankings most, with search the evidence each answer finds'],
    ['Brand names in the answer step', 'X14', 'on identical facts, showing brand names reorders the model’s ranking 2–3× as much as asking again; brands gain or lose according to what the model believes about the name (Spearman 0.81 with the hypothetical-name effect, ChatGPT); publisher names matter little'],
    ['The model’s prior knowledge of brands and websites', 'X10', 'ChatGPT writes queries around brands it knows (67 % vs 11 %); known websites 1.6× as likely to be cited at equal rank; 47 % of ChatGPT’s citations come from websites its model names without searching'],
    ['Page features, entity coverage, crawler access', 'E1, E3, E4', '1–12 % of the within-website variation'],
  ], [3600, 1500, 4260]),
  spacer(),
  P('**Conclusion.** The unexplained part is not mysterious and not mostly chance: it is the engines’ search step. Which pages are cited, and how often, follows how highly each page ranks in the searches the engine runs for a question — searches that are reworded every time and whose results change from week to week. Page properties act only through that ranking, and the ranking itself (see X4) is ' + X4.short + '. Content acts mainly through that ranking — pages that match the engine’s own reworded queries are ranked higher — and only modestly at the citation step itself (X6). The model’s own prior knowledge adds to both: it writes its queries around brands it knows, and it prefers websites it knows (X10). What remains unexplained is how the engines’ search indexes rank pages; that cannot be observed from outside, but it can now be **monitored**: replaying the questions through the APIs shows each week which pages are retrieved, at what rank, and which rank-and-lose.'),
  P('**Not run.** E5 (how often the Wayback Machine archived each page, a proxy for crawl popularity) stopped after 71 pages because web.archive.org blocked further requests. Perplexity was not replayed (no API key). The controlled page test (N11) needs page changes that are not possible for this client.'),
  pageBreak(),
);

// 4 Implications
children.push(
  H1('7  What a practitioner should do'),
  P('This section separates what our tests support **conclusively** (passes all five tests) from what they suggest, and from what they do not support. All results are associations in observational data; the only way to prove that a change causes more citations is a controlled test on your own pages (guide experiment N11).'),
  H2('7.1 Supported conclusively — do this'),
  N('**Write titles in the words of the questions people ask.** Collect the questions your audience actually puts to AI assistants (our study used 651 of them) and title each page after the question it answers. Title–question similarity is the most consistent predictor we found (+14 % per SD overall, +24 % on third-party pages, passes all tests).', 2),
  N('**Give every important question a paragraph that answers it directly, in its own terms.** The best-matching paragraph is the strongest page-level predictor (+16 %, +26 % on third-party pages). It does not need to be at the top of the page (R10): engines find it wherever it is.', 2),
  N('**Cover the vocabulary of the questions across the page.** Word overlap with the question set adds +9 % to +13 % (conclusive on third-party pages; on all pages it narrowly misses the stability test); meaning-based matching (embeddings) predicts even better than exact words (N13), so paraphrases count — but the words users use are the safest choice.', 2),
  P('The follow-up experiments confirm this at the level of single questions: a page is cited for the questions it matches best (X2) and pages that match no question are rarely cited at all (N7). So write each page for a specific set of questions rather than for a topic in general.'),
  P('These three work together and are strongest for comparison questions ("X vs Y", "which provider is best for…"), where title and word matching reach +18 % to +19 % (R12).'),
  H2('7.2 Suggested by the evidence, but not conclusive'),
  B('**Keep pages current.** Older pages are cited less (−5 % per SD, significant, robust and with a steady downward shape), but the stricter selection test does not keep it; ChatGPT in particular cites younger pages (R11).'),
  B('**Answer the question in the first paragraph.** +6 % overall and +8 % on brand pages, passes four of five tests.'),
  B('**Rank in the engines’ own searches, not just in Google for the question.** ChatGPT cites mostly the top 5 of its own search results and rewrites every question into several short queries (brand + attribute, sometimes the year). Pages that are found by more than one of these queries are cited far more (odds ×9.8). Write pages that answer the reworded, specific versions of a question — brand, plan, attribute — not only the question as a user types it (X3).'),
  B('**Monitor retrieval, not only citations.** Replaying your question set through the engines’ APIs once a week (about $10 per run for ChatGPT) shows which of your pages are retrieved, at which rank, and which are retrieved but not cited — the pages closest to being cited. Citations drift over weeks (E6), so a one-off audit goes stale.'),
  B('**Write for the engine’s queries, not only the user’s question.** ChatGPT splits a question into short keyword searches about named brands, plans and products, often with the current year (X7). Pages that are clearly about one named product and state its concrete facts — prices, speeds, terms, dates — match those searches; keep them current and name the year where it matters. What counts is covering the substance of those sub-questions: pages rank by how well their meaning matches the queries, not by repeating the exact keywords (X6b).'),
  B('**Be known to the models.** ChatGPT builds its searches around brands its model already knows and prefers websites it knows at the same rank (X10); a brand the model does not know has to be found by search. Ask each engine’s model, without search, which brands and websites it would name for your questions — a cheap check of your standing, and one that differs by model (Gemini knows niche vendors that ChatGPT’s model does not).'),
  B('**Brand visibility is more than your own pages.** Most mentions of a brand in AI answers are not backed by the brand’s own page: they come from cited third-party pages about it, or from the model’s own knowledge (X11). Coverage on review, comparison and news sites helps most for brands the models do not yet know.'),
  B('**For ChatGPT, the top of the list is worth something on its own.** In a controlled test the same source was cited 18 points more often when shown first than when shown last; Gemini showed no such effect (X12). Being ranked first by ChatGPT’s search is therefore worth more than being ranked fifth even for an equally good page.'),
  B('**Track your standing per question, not per brand.** A brand’s place in AI answers is stable within a question and over weeks (ChatGPT, Gemini), but differs from question to question even within a topic (X13); with fixed evidence, one score per brand and question explains the rankings, and the order in which the engine lists the alternatives moves them most (X15). Measure visibility on the specific questions that matter, and treat a “top-three share” as a property of a question, not of the brand. For a brand the models do not know, the ranking is consistent only when search finds evidence about it (X16).'),
  B('**Know what the models believe about your brand name.** On identical facts, showing the brand name moves products up or down a model’s ranking, following what the model associates with the name (e.g. a cable company for a fiber question), even when told names carry no information (X14). Facts on the page do not fully override that belief; correcting it needs the brand to be described accurately where the models learn — on third-party pages and in the text that answers are built from.'),
  B('**Be the source, not the roundup.** Among pages ChatGPT retrieves for the same question and rank, concise pages with clear headings and pages from the brand or original source are cited more; list and comparison pages, Reddit and YouTube are consulted often but rarely cited (X6). The tips (FAQ, schema, summary and author boxes) make no difference at this step either.'),
  B('**Do not assume classic search rank transfers.** Only about 5 % of the websites the engines cite rank in Google’s top 10 for the same question (X1); AI engines seem to share sources with each other more than with Google’s ranking.'),
  B('**Be present where each engine looks.** ChatGPT cites brand websites (64 % of its citations) and high-authority domains; Gemini and Perplexity rely on review and comparison sites and Reddit (R11). For Gemini and Perplexity, third-party coverage matters more than your own pages.'),
  B('**Topical focus of the whole website (hypothesis).** Websites entirely about the topic (broadbandnow.com, highspeedinternet.com) are cited far more than their authority predicts (section 1.3, R14 over-performers). This fits a "known for the topic" explanation, but we did not test it directly.'),
  B('**Authority helps but is slow to change.** Website authority: +6 % to +11 % per SD at page level, +30 % for ChatGPT (R14), and none in the niche market. Links to the page itself matter more consistently: +23 % per SD of referring websites within websites, +39 % for pages with any link, mainly for Gemini (X9) — but links were measured after the citations, so the direction is not established. Closer question-matching partly compensates on smaller websites.'),
  B('**Some page types do better on the same website:** forum/community pages (+78 %) and list pages (+13 %) compared with articles (R12).'),
  H2('7.3 Not supported — do not expect citations from these alone'),
  B('**FAQ blocks, structured data (schema.org), summary / "key takeaways" boxes, author boxes and page speed.** They look helpful only because websites that use them are cited more anyway; within a website they show no effect (R8). An FAQ block is associated with more citations only for price and purchase questions (+12 %) and with fewer for general questions (−10 %) (R12).'),
  B('**Length, number of headings, lists, tables and outbound links.** No effect in any test; no sweet spot for length (R6).'),
  B('These may still be worth doing for human readers, accessibility or classic search — our data only says they do not change how often AI engines cite a page.'),
  H2('7.4 Realistic expectations'),
  B('About 13 % of the differences in citations come from which website a page is on, and page features explain only about 3 % of the differences between pages of the same website. The rest is not chance — over 90 days the same pages keep being cited (90 % repeatable) — and it is mostly decided in the engines’ own search step: how highly a page ranks for the engine’s reworded queries, which changes from week to week (section 6). Good pages raise the odds; they do not guarantee citations.'),
  B('On any single day the sources for a question change a lot (N1), and half of newly cited pages stop being cited within one to three weeks (N9); but totals over weeks are stable. Measure over weeks, not single days.'),
  B('Naming matters: questions that name a brand draw heavily on that brand’s own site (N5), so content that clearly answers "brand X vs brand Y" questions on your own site is directly relevant.'),
  B('Before rolling changes out widely, test them: rewrite titles and answer paragraphs on a random half of similar pages and compare citations with the other half for six weeks (N11).'),
  pageBreak(),
);

// 5 Limits
children.push(
  H1('8  Limits'),
  B('**Observational data.** All findings are associations. Pages that match questions well may also differ in ways we did not measure (quality, internal linking, popularity).'),
  B('**One client, one market, one period.** 651 prompts about US internet providers over 90 days; results may differ for other industries.'),
  B('**The replay (section 6)** ran on 27 September, after the collection ended; the engines drift over weeks, so it fits the last weeks of the collection best. ChatGPT’s API model (gpt-5.4-nano) and Gemini 3.1 Flash Lite were used as in the collection; the consumer apps may search differently. Gemini’s API returns only the sources it uses, so its retrieval set is not observed.'),
  B('**Perplexity** cites something in only 4 % of answers in this collection (possibly a collection issue); its results rest on few citations.'),
  B('**Gemini coverage changed** on 25 July (about 120 → 350 answers per day); engine comparisons use prompt-days all engines answered.'),
  B('**Pages were downloaded once, after they were cited;** some pages may have changed. Reddit, Quora and YouTube pages could not be measured, so forum and video pages are under-represented in the feature models.'),
  B('**Brand pages** come from only 77 websites, so tests there have less power; nothing passes all five tests on brand pages.'),
  B('**Speed** has page-level data for only 466 pages; a Lighthouse measurement of more pages would give a stronger test.'),
  B('**Hand checks still open:** page text/type/date checks (partly done, no errors so far), 100 intent labels, 50 website-type labels.'),
  H1('Appendix'),
  B('Scripts: scripts/r01_describe.py … r14_authority.py and is_it_real.py; results/r01_* … r14_*, results/is_it_real.md; figures/.'),
  B('Section 6: scripts/e01_engines_breadth_depth.py, e02_choice.py, e03_entity_coverage.py, e04_ai_crawlers.py, e06_drift.py, x03_replay.py (replay; data/replay/), x03_analyse.py, x04_engine_query_serp.py.'),
  B('Every choice that changes a number (dropped rows, labels, test variants, bug fixes) is logged with date and reason in decisions.md.'),
  B('Main deviations from the guide: stricter R4 penalty; R6 curves with other features held fixed; LLM intent labels; Verizon added as a competitor; page age from the publish date; speed restricted to page-level data.'),
);

// ---------- document
const doc = new Document({
  creator: 'AeoCiteAnalysis', title: 'What Makes AI Engines Cite a Page? Results report',
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
      children: [new TextRun({ text: 'What makes AI engines cite a page? · results report · page ', size: 16, color: '808080', font: FONT }),
        new TextRun({ children: [PageNumber.CURRENT], size: 16, color: '808080', font: FONT })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(ROOT, 'report', 'R1-R14_report.docx');
  fs.writeFileSync(out, buf);
  console.log('wrote', out, (buf.length / 1e6).toFixed(1), 'MB,', figNo, 'figures');
});
