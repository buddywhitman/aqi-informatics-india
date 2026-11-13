import sys
from PIL import Image
ids=[int(x) for x in sys.argv[2].split(',')]; cols=2
ims=[Image.open(f"slides_png/s{i:02d}.png").resize((960,540)) for i in ids]
rows=(len(ims)+1)//2; sh=Image.new("RGB",(960*cols,540*rows),"white")
for k,im in enumerate(ims): sh.paste(im,((k%cols)*960,(k//cols)*540))
sh.save(sys.argv[1])
