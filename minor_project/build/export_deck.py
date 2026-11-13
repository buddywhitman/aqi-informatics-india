import sys, json, pathlib, threading, functools, http.server, socketserver
from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.util import Emu

here = pathlib.Path(__file__).resolve().parent
root = here.parent
shots = here / "slides_png"; shots.mkdir(exist_ok=True)
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
H = functools.partial(Q, directory=str(root))
srv = socketserver.TCPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(f"http://127.0.0.1:{port}/build/deck.html?print=1"); pg.wait_for_timeout(2500)
    pg.evaluate("document.fonts.ready")
    n = pg.evaluate("document.querySelectorAll('.slide').length")
    for i in range(n):
        pg.locator(f"#s{i+1}").screenshot(path=str(shots / f"s{i+1:02d}.png"))
    pg.pdf(path=str(root / "out" / "Minor_Project_Presentation.pdf"), width="1920px", height="1080px", print_background=True)
    b.close()
notes = json.load(open(here / "notes.json", encoding="utf8"))
prs = Presentation(); prs.slide_width = Emu(12192000); prs.slide_height = Emu(6858000)
for i in range(n):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.shapes.add_picture(str(shots / f"s{i+1:02d}.png"), 0, 0, prs.slide_width, prs.slide_height)
    s.notes_slide.notes_text_frame.text = notes[i]
prs.save(str(root / "out" / "Minor_Project_Presentation.pptx")); print("slides", n)
