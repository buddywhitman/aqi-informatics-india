"""Assemble paper/appendix.tex from paper/appendix_src.tex and the submitted draft's proofs (see header of appendix_src.tex)."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'archive', 'manuscript_lineages', 'feature_submitted_draft_main.tex')
lines = open(SRC).read().split('\n')
SUBS = [
    (r'\eqref{eq:regime_scm}-\eqref{eq:outcome}', r'\eqref{eq:scm1}--\eqref{eq:scm2}'),
    (r'\eqref{eq:regime_scm}', r'\eqref{eq:scm1}'),
    (r'\label{lem:orth}', r'\label{lem:orth_weighted}'),
    (r'\ref{lem:orth}', r'\ref{lem:orth_weighted}'),
]
DROP = ['Across our multi-season ground-sensor streams, the empirical ratio']
def seg(a, b):
    out = []
    for l in lines[a - 1:b]:
        for s, r in SUBS:
            l = l.replace(s, r)
        for d in DROP:
            if d in l:
                l = l[:l.index(d)] + r'$\blacksquare$'
        out.append(l)
    return '\n'.join(out)
txt = open(os.path.join(HERE, 'appendix_src.tex')).read()
txt = re.sub(r'%%SEG (\d+)-(\d+)%%', lambda m: seg(int(m.group(1)), int(m.group(2))), txt)
open(os.path.join(HERE, 'appendix.tex'), 'w').write('% AUTO-ASSEMBLED by paper/build_appendix.py from appendix_src.tex -- edit the source\n' + txt)
print('appendix.tex written')
