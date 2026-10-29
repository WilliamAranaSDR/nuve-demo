from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np, sys
im=Image.open(sys.argv[1]).convert('RGB'); a=np.array(im).astype(int)
bg=tuple(int(v) for v in a[300:420,900:1000].reshape(-1,3).mean(0))
d=ImageDraw.Draw(im)
# erase price block (x>=380 clears the valve fittings)
d.rectangle((380,200,840,438),fill=bg)
# 'COMPLETE KIT' line: erase only near-dark text pixels left of the stat bezel
for y in range(438,456):
    for x in range(440,740):
        r,g,b=a[y,x]
        if x<640 or (r+g+b)>200 or y<446: im.putpixel((x,y),bg)
# coming soon pill
F=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',58)
t='COMING SOON'; w=d.textlength(t,font=F)
cx=628; y0=262
d.rounded_rectangle((cx-w/2-34,y0,cx+w/2+34,y0+92),radius=14,fill=(30,200,240))
d.text((cx,y0+48),t,font=F,fill='white',anchor='mm')
F2=ImageFont.truetype('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf',30)
d.text((cx,y0+128),'Remote main water shut-off',font=F2,fill=(19,47,87),anchor='mm')
# blank the YOUR LOGO HERE box on the Sofia face
face=tuple(int(v) for v in a[805:850,650:668].reshape(-1,3).mean(0))
d.rounded_rectangle((671,793,790,865),radius=4,fill=face)
im=im.crop((0,0,im.width,1278))
im.save(sys.argv[2],quality=93)
print(bg,face,im.size)
