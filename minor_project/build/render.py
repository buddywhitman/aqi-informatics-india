import sys, threading, functools, http.server, socketserver, pathlib
from playwright.sync_api import sync_playwright
root=pathlib.Path(sys.argv[1]).resolve().parent.parent  # minor_project/
page_path="/build/"+pathlib.Path(sys.argv[1]).name; out=sys.argv[2]
H=functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(root))
pass
srv=socketserver.TCPServer(("127.0.0.1",0),H); port=srv.server_address[1]
threading.Thread(target=srv.serve_forever,daemon=True).start()
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page()
    pg.on("console", lambda m: print("console:", m.text[:200]) if m.type=="error" else None)
    pg.goto(f"http://127.0.0.1:{port}{page_path}"); pg.wait_for_function("window.PagedPolyfill!==undefined", timeout=30000)
    pg.evaluate("window.PagedPolyfill.preview()")
    pg.wait_for_function("document.querySelectorAll('.pagedjs_page').length>3", timeout=600000)
    last=-1
    for _ in range(300):
        pg.wait_for_timeout(1000)
        n=pg.evaluate("document.querySelectorAll('.pagedjs_page').length")
        if n==last and pg.evaluate("!!document.querySelector('.pagedjs_pages')") and _>5: break
        last=n

    pg.evaluate("""()=>{
      const pages=[...document.querySelectorAll('.pagedjs_page')];
      const idx=el=>pages.indexOf(el.closest('.pagedjs_page'));
      const c1=document.getElementById('c1'); const first=idx(c1);
      const roman=n=>['','i','ii','iii','iv','v','vi','vii','viii','ix','x','xi','xii','xiii','xiv'][n];
      const label=i=>i<2?'':(i<first?roman(i-1):String(i-first+1));
      pages.forEach((p,i)=>{const l=label(i); if(l){const d=document.createElement('div');d.className='pn';d.textContent=l;(p.querySelector('.pagedjs_pagebox')||p).appendChild(d)}});
      document.querySelectorAll('.toc a, .fl a').forEach(a=>{const t=document.getElementById(a.getAttribute('href').slice(1)); if(!t)return; const s=document.createElement('span'); s.className='pg'; s.textContent=label(idx(t)); a.appendChild(s)});
    }""")
    print("pages",n)
    pg.pdf(path=out, prefer_css_page_size=True, print_background=True)
    b.close()
