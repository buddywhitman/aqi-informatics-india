import fs from 'fs';
import katex from 'katex';
import hljs from 'highlight.js';

let src = fs.readFileSync('deck.src.html', 'utf8').split(String.fromCharCode(13)).join('');
src = src.replace(/\{\{SVG:(\w+)\}\}/g, (_, n) => fs.readFileSync(`svg/${n}.svg`, 'utf8'));
src = src.replace(/<pre class="code" data-lang="(\w+)"([^>]*)>([\s\S]*?)<\/pre>/g, (_, l, rest, code) => {
  const dec = code.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  const h = hljs.highlight(dec, { language: hljs.getLanguage(l) ? l : 'plaintext' }).value;
  return `<pre class="code"${rest}><code>${h}</code></pre>`;
});
src = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, m) => katex.renderToString(m, { displayMode: true, throwOnError: false }));
src = src.replace(/\$([^$\n]+?)\$/g, (_, m) => katex.renderToString(m, { throwOnError: false }));

// footer + numbering
let i = 0; const total = (src.match(/<section class="slide/g) || []).length;
const notes = [];
src = src.replace(/<section class="slide([^"]*)"([^>]*)>([\s\S]*?)<\/section>/g, (_, cls, attr, inner) => {
  i++;
  const nm = inner.match(/<aside class="notes">([\s\S]*?)<\/aside>/); notes.push(nm ? nm[1].trim() : '');
  inner = inner.replace(/<aside class="notes">[\s\S]*?<\/aside>/, '');
  const foot = /data-nofoot/.test(attr) ? '' : `<div class="foot"><span>MIT Manipal · Data Science Minor · Minor Project</span><span>${String(i).padStart(2, '0')} / ${String(total).padStart(2, '0')}</span></div>`;
  return `<section class="slide${cls}" id="s${i}">${inner}${foot}</section>`;
});
fs.writeFileSync('notes.json', JSON.stringify(notes, null, 1));
const html = `<!doctype html><html><head><meta charset="utf-8"><title>Minor Project Presentation</title>
<link rel="stylesheet" href="katex.min.css"><link rel="stylesheet" href="deck.css"></head><body>${src}
<script>
if(!/print=1/.test(location.search)){document.body.classList.add('deckview');const S=[...document.querySelectorAll('.slide')];let k=0;
const fit=()=>{const s=Math.min(innerWidth/1920,innerHeight/1080);S.forEach(e=>{e.style.transformOrigin='0 0';e.style.transform='scale('+s+')';e.style.margin=((innerHeight-1080*s)/2)+'px 0 0 '+((innerWidth-1920*s)/2)+'px'})};
const go=n=>{k=Math.max(0,Math.min(S.length-1,n));S.forEach((e,j)=>e.classList.toggle('on',j==k));location.hash=k+1};
addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key))go(k+1);if(['ArrowLeft','PageUp'].includes(e.key))go(k-1)});
addEventListener('click',e=>go(e.clientX>innerWidth/2?k+1:k-1));addEventListener('resize',fit);fit();go((+location.hash.slice(1)||1)-1)}
</script></body></html>`;
fs.writeFileSync('deck.html', html);
console.log('slides', total);
