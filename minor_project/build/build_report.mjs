import fs from 'fs';
import katex from 'katex';
import { Marked } from 'marked';
import markedKatex from 'marked-katex-extension';
import { markedHighlight } from 'marked-highlight';
import hljs from 'highlight.js';

const GEN = '../../paper/generated/';
const marked = new Marked(
  markedHighlight({ langPrefix: 'hljs language-', highlight: (c, l) => hljs.highlight(c, { language: hljs.getLanguage(l) ? l : 'plaintext' }).value }),
  markedKatex({ throwOnError: false, nonStandard: true }),
);
marked.use({ breaks: false });

const files = fs.readdirSync('src').filter(f => /^\d\d.*\.md$/.test(f)).sort();
const reg = { fig: {}, tab: {} };
const lists = { fig: [], tab: [] };
let chap = 0, figN = 0, tabN = 0, label = '';
const toc = [];

const texMath = s => s.replace(/\$([^$]+)\$/g, (_, m) => katex.renderToString(m, { throwOnError: false }));
function texToHtml(src, cls) {
  const body = src.replace(/\\begin\{tabular\}\{[^}]*\}/, '').replace(/\\end\{tabular\}/, '').replace(/^%.*$/gm, '');
  const rows = body.split(/\\\\/).map(r => r.trim()).filter(Boolean);
  let html = '<table class="t ' + (cls || '') + '">'; let first = true; let inBody = false; let head = '', rest = '';
  let hdrRows = [];
  let pendingSep = false;
  for (let r of rows) {
    let sep = false;
    if (/\\midrule/.test(r)) { sep = true; }
    r = r.replace(/\\(toprule|bottomrule|midrule|hline)/g, '').replace(/\\cmidrule(\([lr]+\))?\{[^}]*\}/g, '').trim();
    if (!r) continue;
    let cells = r.split(/(?<!\\)&/).map(c => c.trim());
    let tds = ''; let isMulti = false; let span = 0;
    for (let c of cells) {
      let m = c.match(/^\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$/s);
      let n = 1, t = c;
      if (m) { n = +m[1]; t = m[2]; isMulti = true; }
      span += n;
      t = t.replace(/\\textbf\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}/g, '<b>$1</b>').replace(/\\emph\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}/g, '<i>$1</i>').replace(/\\%/g, '%').replace(/\\,/g, ' ').replace(/--/g, '–').replace(/~/g, ' ').replace(/\\_/g, '_').replace(/\\&/g, '&amp;');
      t = texMath(t);
      tds += `<td${n > 1 ? ` colspan="${n}"` : ''}>${t}</td>`;
    }
    const subrow = cells.length === 1 && /\\multicolumn\{\d+\}\{l\}/.test(cells[0]);
    const isHeader = first && !subrow;
    if (isHeader) { hdrRows.push(`<tr>${tds.replace(/<td/g, '<th').replace(/<\/td>/g, '</th>')}</tr>`); }
    else { first = false; rest += `<tr class="${sep ? 'sep ' : ''}${subrow ? 'sub' : ''}">${tds}</tr>`; }
    if (isHeader && (isMulti || hdrRows.length < 2) && !/\\midrule/.test(rows[rows.indexOf(r)] || '') ) { /* keep scanning header rows until midrule */ }
    if (isHeader && sep) first = false;
  }
  return html + `<thead>${hdrRows.join('')}</thead><tbody>${rest}</tbody></table>`;
}

function texHeaderAware(src, cls) {
  // header rows = everything before first \midrule
  const parts = src.split(/\\midrule/);
  const head = parts[0], tail = parts.slice(1).join('\\midrule');
  const hh = texToHtml(head + '\\\\', cls).replace(/<tbody>.*<\/tbody>/s, '<tbody></tbody>');
  // all rows in head are header rows
  const hrows = (hh.match(/<tr>.*?<\/tr>/gs) || []);
  const hdr = hrows.map(r => r.replace(/<td/g, '<th').replace(/<\/td>/g, '</th>')).join('');
  const hdrFixed = hh.match(/<thead>(.*)<\/thead>/s)[1] ? '' : '';
  // body: convert tail with a dummy header row so first-row logic treats everything as body
  const bodyHtml = texToBodyRows(tail);
  return `<table class="t ${cls || ''}"><thead>${(hh.match(/<thead>(.*?)<\/thead>/s) || [, ''])[1]}</thead><tbody>${bodyHtml}</tbody></table>`;
}
function texToBodyRows(src) {
  const body = src.replace(/\\end\{tabular\}/, '').replace(/\\begin\{tabular\}\{[^}]*\}/, '').replace(/^%.*$/gm, '');
  const chunks = body.split(/\\\\/);
  let out = '', sep = false;
  for (let r of chunks) {
    if (/\\midrule/.test(r)) sep = true;
    r = r.replace(/\\(toprule|bottomrule|midrule|hline)/g, '').replace(/\\cmidrule(\([lr]+\))?\{[^}]*\}/g, '').trim();
    if (!r) continue;
    const cells = r.split(/(?<!\\)&/).map(c => c.trim());
    let tds = '';
    for (let c of cells) {
      const m = c.match(/^\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$/s);
      let n = 1, t = c; if (m) { n = +m[1]; t = m[2]; }
      t = t.replace(/\\textbf\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}/g, '<b>$1</b>').replace(/\\emph\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}/g, '<i>$1</i>').replace(/\\%/g, '%').replace(/\\,/g, ' ').replace(/--/g, '–').replace(/~/g, ' ').replace(/\\_/g, '_');
      tds += `<td${n > 1 ? ` colspan="${n}"` : ''}>${texMath(t)}</td>`;
    }
    const sub = cells.length === 1 && /\\multicolumn\{\d+\}\{l\}/.test(cells[0]);
    out += `<tr class="${sep ? 'sep ' : ''}${sub ? 'sub' : ''}">${tds}</tr>`; sep = false;
  }
  return out;
}

// pass 1: numbering
let raw = files.map(f => fs.readFileSync('src/' + f, 'utf8').split(String.fromCharCode(13)).join('')).join('\n\n');
raw.split('\n').forEach(l => {
  let m;
  if (/^# /.test(l)) { if (!/^# !/.test(l)) { chap++; label = String(chap); } else { const a = l.match(/Annexure ([A-Z])/); label = a ? a[1] : 'R'; } figN = 0; tabN = 0; }
  if (m = l.match(/^@(fig|svg)\s+(\S+)\s*\|/)) { figN++; reg.fig[m[2]] = `${label}.${figN}`; }
  if (m = l.match(/^@tex\s+(\S+)\s*\|/)) { tabN++; reg.tab[m[1]] = `${label}.${tabN}`; }
  if (m = l.match(/^@tab\s+(\S+)\s*\|/)) { tabN++; reg.tab[m[1]] = `${label}.${tabN}`; }
});
const ref = s => s.replace(/\{\{(fig|tab):(\S+?)\}\}/g, (_, k, id) => reg[k][id] ?? (console.error('MISSING', k, id), '??'));

// pass 2: render
function renderBlock(md) {
  const lines = md.split('\n'); let out = []; let buf = [];
  const flush = () => { if (buf.length) { out.push(marked.parse(buf.join('\n'))); buf = []; } };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i]; let m;
    if (m = l.match(/^@(fig|svg)\s+(\S+)\s*\|\s*([^|]+?)\s*\|\s*(.*?)(?:\s*\|\s*(w))?\s*$/)) {
      flush(); const id = m[2]; const n = reg.fig[id];
      const inner = m[1] === 'svg' ? fs.readFileSync('svg/' + m[3] + '.svg', 'utf8') : `<img src="../assets/${m[3]}">`;
      lists.fig.push({ id, n, cap: m[4] });
      out.push(`<figure id="fig-${id}" class="${m[5] || ''}">${inner}<figcaption><b>Figure ${n}.</b> ${marked.parseInline(ref(m[4]))}</figcaption></figure>`);
    } else if (m = l.match(/^@tex\s+(\S+)\s*\|\s*([^|]+?)\s*\|\s*(.*?)(?:\s*\|\s*(sm(?: long)?))?\s*$/)) {
      flush(); const id = m[1]; const n = reg.tab[id];
      const src = fs.readFileSync(GEN + m[2], 'utf8');
      lists.tab.push({ id, n, cap: m[3] });
      out.push(`<div class="tabwrap ${(m[4]||'').includes('long')?'long':''}" id="tab-${id}"><div class="cap"><b>Table ${n}.</b> ${marked.parseInline(ref(m[3]))}</div>${texHeaderAware(src, 'sm')}</div>`);
    } else if (m = l.match(/^@tab\s+(\S+)\s*\|\s*(.*)$/)) {
      flush(); const id = m[1]; const n = reg.tab[id]; lists.tab.push({ id, n, cap: m[2] });
      let tbl = []; i++; while (i < lines.length && lines[i].startsWith('|')) tbl.push(lines[i++]); i--;
      out.push(`<div class="tabwrap" id="tab-${id}"><div class="cap"><b>Table ${n}.</b> ${marked.parseInline(ref(m[2]))}</div>${marked.parse(tbl.join('\n')).replace('<table>', '<table class="md">')}</div>`);
    } else if (m = l.match(/^@(thm|callout)\s*(.*)$/)) {
      flush(); const kind = m[1], title = m[2]; let inner = []; i++;
      while (i < lines.length && !/^@end\s*$/.test(lines[i])) inner.push(lines[i++]);
      out.push(`<div class="${kind}"><div class="h">${title}</div>${renderBlock(inner.join('\n'))}</div>`);
    } else buf.push(l);
  }
  flush(); return out.join('\n');
}

// chapters
let body = ''; let cn = 0; const parts = raw.split(/^(?=# )/m).filter(s => s.trim());
for (const p of parts) {
  const m = p.match(/^# (!?)(.*)\n/); if (!m) continue;
  const unnum = !!m[1]; const title = m[2]; const id = 'c' + (unnum ? title.replace(/\W/g, '') : ++cn);
  toc.push({ lvl: 'c', id, text: (unnum ? '' : cn + '.  ') + title });
  let content = ref(p.slice(m[0].length));
  content = content.replace(/^## (.*)$/gm, (_, t) => { const sid = 's' + toc.length; toc.push({ lvl: 's', id: sid, text: t }); return `<h2 id="${sid}">${t}</h2>`; });
  body += `<section class="chapter ${cn===1&&!unnum?'first':''} ${unnum ? 'unnum' : ''} ${/^Annexure/.test(title) ? 'annex' : ''} ${/^Annexure B/.test(title) ? 'keep' : ''}"><h1 class="chapter-title" id="${id}">${unnum ? '' : `<span class="num">Chapter ${cn}</span>`}<span class="tt">${title}</span></h1>${renderBlock(content)}</section>`;
}

const tocHtml = toc.map(t => `<div class="${t.lvl}"><a href="#${t.id}"><span>${t.text}</span><i></i></a></div>`).join('');
const lof = lists.fig.map(f => `<a href="#fig-${f.id}"><span>Figure ${f.n}&nbsp;&nbsp;${marked.parseInline(ref(f.cap)).replace(/<[^>]+>/g, '')}</span><i></i></a>`).join('');
const lot = lists.tab.map(f => `<a href="#tab-${f.id}"><span>Table ${f.n}&nbsp;&nbsp;${marked.parseInline(ref(f.cap)).replace(/<[^>]+>/g, '')}</span><i></i></a>`).join('');

const front = texMath(fs.readFileSync('front.html', 'utf8')).replace('{{TOC}}', tocHtml).replace('{{LOF}}', lof).replace('{{LOT}}', lot);
const html = `<!doctype html><html><head><meta charset="utf-8"><title>Minor Project Report</title>
<link rel="stylesheet" href="katex.min.css"><link rel="stylesheet" href="report.css">
<script>window.PagedConfig={auto:false}</script><script src="node_modules/pagedjs/dist/paged.polyfill.js"></script></head>
<body>${front}<div class="main">${body}</div></body></html>`;
fs.writeFileSync('report.html', html);
console.log('chapters', cn, 'figs', lists.fig.length, 'tabs', lists.tab.length, 'bytes', html.length);
