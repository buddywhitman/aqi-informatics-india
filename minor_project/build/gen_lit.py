"""Generate src/02_lit.md and merge verified refs into src/90_refs.md.
Metadata comes from Crossref/OpenAlex lookups (ok.json, extras.json); text lives in lit_text.md with {key} / {key:n} cite macros.
Run from minor_project/build:  python gen_lit.py
"""
import json, re, sys, unicodedata
SC = '.'
sys.path.insert(0, SC)
from lit_cands import C
YR = {c[0]: c[2] for c in C}
meta = json.load(open('lit_refs_verified.json', encoding='utf8'))
for k in 'kiviet zhang2021sens liu2024proxy shpitser2023 lewis2021dyn hyndman2008 tchetgen2020 miao2018b rosenbaum1983 rothe2017 abadie2010 greenstone2014 masten2018 kuroki2014 bartolucci2014 hamilton2016 cinelli2019 kruskal1977 zhang2019mnar'.split(): meta.pop(k, None)
meta.update(json.load(open('lit_refs_extra.json', encoding='utf8')))
MANUAL = {
 'imbens2015': ('Imbens', 'Guido W Imbens and Donald B Rubin. *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press, 2015. doi:10.1017/CBO9781139025751.', 2, 2015),
 'louizos2017': ('Louizos', 'Christos Louizos, Uri Shalit, Joris M Mooij, David Sontag, Richard Zemel, and Max Welling. Causal effect inference with deep latent-variable models. In *Advances in Neural Information Processing Systems (NeurIPS)*, volume 30, 2017.', 6, 2017),
 'shi2019dragonnet': ('Shi', 'Claudia Shi, David M Blei, and Victor Veitch. Adapting neural networks for the estimation of treatment effects. In *Advances in Neural Information Processing Systems (NeurIPS)*, volume 32, 2019.', 3, 2019),
 'kallus2019': ('Kallus', 'Nathan Kallus, Xiaojie Mao, and Angela Zhou. Interval estimation of individual-level causal effects under unobserved confounding. In *International Conference on Artificial Intelligence and Statistics (AISTATS)*, volume 89 of *Proceedings of Machine Learning Research*, 2019.', 3, 2019),
 'anandkumar2014': ('Anandkumar', 'Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M Kakade, and Matus Telgarsky. Tensor decompositions for learning latent variable models. *Journal of Machine Learning Research*, 15 (80): 2773-2832, 2014.', 5, 2014),
 'khan2010': ('Khan', 'Shakeeb Khan and Elie Tamer. Irregular identification, support conditions, and inverse weight estimation. *Econometrica*, 78 (6): 2021-2042, 2010. doi:10.3982/ecta7372.', 2, 2010),
 'manski1990': ('Manski', 'Charles F Manski. Nonparametric bounds on treatment effects. *American Economic Review*, 80 (2): 319-323, 1990.', 1, 1990),
 'bhattacharya2007': ('Bhattacharya', 'Jay Bhattacharya and William B Vogt. Do instrumental variables belong in propensity scores? NBER Technical Working Paper 343, 2007. doi:10.3386/t0343.', 2, 2007),
 'cinelli2019': ('Cinelli', 'Carlos Cinelli, Daniel Kumor, Bryant Chen, Judea Pearl, and Elias Bareinboim. Sensitivity analysis of linear structural causal models. In *International Conference on Machine Learning (ICML)*, volume 97 of *Proceedings of Machine Learning Research*, 2019.', 5, 2019),
 'chernozhukov2022riesz': ('Chernozhukov', 'Victor Chernozhukov, Whitney K Newey, Victor Quintas-Martinez, and Vasilis Syrgkanis. RieszNet and ForestRiesz: automatic debiased machine learning with neural nets and random forests. In *International Conference on Machine Learning (ICML)*, 2022. arXiv:2110.03031.', 4, 2022),
 'ranganath2018': ('Ranganath', 'Rajesh Ranganath and Adler Perotte. Multiple causal inference with latent confounding. *arXiv preprint arXiv:1805.08273*, 2018.', 2, 2018),
 'kallus2022': ('Kallus', 'Nathan Kallus, Xiaojie Mao, and Masatoshi Uehara. Causal inference under unmeasured confounding with negative controls: a minimax learning approach. *arXiv preprint arXiv:2103.14029*, 2021.', 3, 2021),
 'deaner2018': ('Deaner', 'Ben Deaner. Proxy controls and panel data. *arXiv preprint arXiv:1810.00283*, 2018.', 1, 2018),
 'veitch2019': ('Veitch', 'Victor Veitch, Yixin Wang, and David M Blei. Using embeddings to correct for unobserved confounding in networks. In *Advances in Neural Information Processing Systems (NeurIPS)*, volume 32, 2019.', 3, 2019),
}
LOWER_OK = {'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'for', 'to', 'with', 'by', 'via', 'under', 'from', 'when', 'after', 'among', 'at', 'is', 'are', 'as'}
PROPER = set('California Earth India Delhi Beijing China Chinese Swiss European U.S. United States Catholic California Bayesian Markov Neyman Hamilton Granger Lancet MODIS COVID-19 PM2.5 PM10 NO2 EM GBD Monte Carlo Gaussian Bernoulli Riesz'.split())
def sentence_case(t):
    t = re.sub(r'<[^>]+>', '', t).replace('\n', ' ')
    t = re.sub(r'\s+', ' ', t).strip().rstrip('.')
    w = t.split(' ')
    caps = sum(1 for x in w if x[:1].isupper())
    if caps < 0.5 * len(w): return t
    out = []
    for i, x in enumerate(w):
        core = re.sub(r"'s$", '', re.sub(r"[^A-Za-z0-9.'-]", '', x))
        keep = (i == 0) or (out and out[-1].endswith(':')) or core in PROPER or (core.isupper() and len(core) > 1) or re.search(r'[a-z][A-Z]', core) or re.search(r'\d', core)
        out.append(x if keep else x[:1].lower() + x[1:] if x[:1].isupper() else x)
    r = ' '.join(out)
    r = re.sub(r'(?<=[a-z])-([A-Z])([a-z]+)', lambda m: '-' + (m.group(0)[1:] if m.group(1)+m.group(2) in PROPER else m.group(1).lower() + m.group(2)), r)
    return r
def fix(s): return (s or '').replace('&amp;', '&').replace('\u2010', '-').replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'")
def fmt_authors(a):
    n = [(g + ' ' + f).strip() for g, f in a]
    n = [re.sub(r'\s+', ' ', fix(x)) for x in n]
    n = [' '.join(w.capitalize() if (w.isupper() and len(w) > 2 and '.' not in w) else w for w in x.split(' ')) for x in n]
    if len(n) > 6: return n[0] + ' et al'
    if len(n) == 1: return n[0]
    if len(n) == 2: return n[0] + ' and ' + n[1]
    return ', '.join(n[:-1]) + ', and ' + n[-1]
PAGEFIX = {'politis1994': '1303-1313', 'granger1969': '424-438', 'hahn1998': '315-331', 'andrews1991': '817-858', 'pearl1995': '669-688', 'pope2002': '1132-1141', 'runge2019': 'eaau4996', 'robinson1988': '931-954'}
def bib(k):
    if k in MANUAL: return MANUAL[k][1]
    v = meta[k]
    yr = YR.get(k) if (k in YR and abs(YR[k] - v['y']) <= 3 and v.get('src') != 'crossref-doi') else v['y']
    doi = v.get('doi') or ''
    t = sentence_case(fix(v['title'])).replace('--', '—')
    t = re.sub(r'PM\s+(\d)', r'PM', t)
    t = {'robinson1988': 'Root-N-consistent semiparametric regression'}.get(k, t)
    cont = fix(re.sub(r'<[^>]+>', '', v.get('cont') or '')).strip()
    s = fmt_authors(v['authors']).replace('Kunsch', 'Künsch') + '. ' + t + '. '
    if doi.lower().startswith('10.48550/arxiv.'):
        s += '*arXiv preprint arXiv:' + doi.split('arxiv.')[1] + '*, ' + str(yr) + '.'
        return s
    if cont:
        vol, iss, pg = v.get('vol'), v.get('iss'), PAGEFIX.get(k) or (v.get('page') or '').replace('--', '-')
        s += '*' + cont + '*'
        if vol: s += ', ' + str(vol) + (' (' + str(iss) + ')' if iss else '') + (': ' + pg if pg else '')
        s += ', ' + str(yr) + '.'
    else:
        s += str(yr) + '.'
    if doi: s += ' doi:' + doi + '.'
    return s
def cite(k, narrative=False):
    if k in MANUAL: sur, _, n, y = MANUAL[k]; names = None
    else:
        v = meta[k]; y = YR.get(k) if (k in YR and abs(YR[k] - v['y']) <= 3 and v.get('src') != 'crossref-doi') else v['y']
        names = [f.capitalize() if f.isupper() else f for g, f in v['authors']]; n = len(names); sur = names[0] if names else '?'
    sur = sur.title() if sur.isupper() else sur
    if n == 2 and names: lab = sur + ' and ' + names[1]
    elif n == 2: lab = {'khan2010': 'Khan and Tamer', 'bhattacharya2007': 'Bhattacharya and Vogt', 'ranganath2018': 'Ranganath and Perotte', 'imbens2015': 'Imbens and Rubin'}.get(k, sur + ' et al.')
    elif n == 1: lab = sur
    else: lab = sur + ' et al.'
    return f'{lab} ({y})' if narrative else f'{lab}, {y}'
used = []
def sub(m):
    k, n = m.group(1), bool(m.group(2))
    if k not in meta and k not in MANUAL: raise SystemExit('MISSING ' + k)
    if k not in used: used.append(k)
    return cite(k, n)
text = open('lit_text.md', encoding='utf8').read()
text = re.sub(r'\{([a-z][a-z0-9]+)(:n)?\}', sub, text)
open('src/02_lit.md', 'w', encoding='utf8', newline='\n').write(text)
# merge refs
old = open('src/90_refs.md', encoding='utf8').read()
head, body = old.split('<div class="refs">', 1)
body = body.split('</div>')[0]
entries = [e.strip() for e in body.split('\n\n') if e.strip()]
base = [e for e in entries if not e.startswith('@NEW')]
def sortkey(e): return unicodedata.normalize('NFKD', e).encode('ascii', 'ignore').decode().lower()
old_keys = {sortkey(e)[:40] for e in base}
new = [bib(k) for k in used]
allr = base + [e for e in new if sortkey(e)[:40] not in old_keys]
allr.sort(key=sortkey)
open('src/90_refs.md', 'w', encoding='utf8', newline='\n').write(head + '<div class="refs">\n\n' + '\n\n'.join(allr) + '\n\n</div>\n')
print('cited new keys:', len(used), '| total refs:', len(allr))
