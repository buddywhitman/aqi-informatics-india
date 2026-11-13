import re
s = open('paper/references.tex', encoding='utf8').read()
items = s.split('\\bibitem')[1:]
acc = {"{\\'e}": "é", "{\\`e}": "è", '{\\"o}': "ö", "{\\'o}": "ó", "{\\'a}": "á", "{\\'i}": "í", "{\\~a}": "ã", "{\\c{c}}": "ç"}
out = []
for it in items:
    kk = re.match(r'^\[[^\]]*\]\{([^}]*)\}', it.strip()).group(1)
    it = re.sub(r'^\[[^\]]*\]\{[^}]*\}', '', it.strip())
    it = it.replace('\\newblock', '').replace('\\penalty0', '').replace('{\\natexlab{b}}', 'b').replace('{\\natexlab{a}}', 'a')
    for k, v in acc.items():
        it = it.replace(k, v)
    it = re.sub(r'\\emph\{([^}]*)\}', r'*\1*', it)
    it = re.sub(r'\\url\{([^}]*)\}', r'\1', it)
    it = it.replace('~', ' ').replace('\\&', '&').replace('--', '–').replace('{', '').replace('}', '')
    out.append((kk, re.sub(r'\s+', ' ', it).strip()))
out.append(('lopezdeprado2018advances', 'Marcos López de Prado. Advances in Financial Machine Learning. Wiley, 2018.'))
out.sort(key=lambda x: x[0].lower())
out=[o[1].replace('\doi','doi').replace('doi10.','doi:10.') for o in out]
open('minor_project/build/src/90_refs.md', 'w', encoding='utf8').write('# !References\n\n<div class="refs">\n\n' + '\n\n'.join(out) + '\n\n</div>\n')
print(len(out)); print(out[3]); print(out[-2])
