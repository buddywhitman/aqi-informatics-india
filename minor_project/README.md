# Minor project deliverables

- `out/Minor_Project_Report.pdf` - thesis-style report (MIT Manipal letterhead, 11 chapters, annexures).
- `out/Minor_Project_Presentation.pdf` / `.pptx` - 17-slide minimal deck (pptx slides are images; speaker notes included).
- `build/` - sources. Report: `src/*.md`, `front.html`, `build_report.mjs`, `render.py`. Deck: `deck.src.html`, `build_deck.mjs`, `export_deck.py`.
- Numbers/tables come from `paper/generated/` (same files as the AISTATS manuscript).

Rebuild (from `build/`, after `npm i --legacy-peer-deps`): `node build_report.mjs && python render.py report.html ../out/Minor_Project_Report.pdf`; `node build_deck.mjs && python export_deck.py`.

