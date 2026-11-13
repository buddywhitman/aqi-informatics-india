import sys, pymupdf
d=pymupdf.open(sys.argv[1]); pages=[int(x) for x in sys.argv[3].split(',')]
w,h=560,792; cols=int(sys.argv[4]) if len(sys.argv)>4 else 4
rows=(len(pages)+cols-1)//cols
sheet=pymupdf.Pixmap(pymupdf.csRGB,pymupdf.IRect(0,0,w*cols,h*rows),False); sheet.clear_with(255)
for i,p in enumerate(pages):
    pm=d[p-1].get_pixmap(matrix=pymupdf.Matrix(w/595,h/842)); pm=pymupdf.Pixmap(pymupdf.csRGB,pm) if pm.n>3 else pm
    pm.set_origin((i%cols)*w,(i//cols)*h); sheet.copy(pm,pm.irect)
sheet.save(sys.argv[2]); print(len(d))
