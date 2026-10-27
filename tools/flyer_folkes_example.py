import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
SRC='/mnt/user-data/uploads/1913969c-image.png'; LOGO='/tmp/folkes-logo.png'
RED=(196,18,48); RED_LT=(255,92,104)
img=cv2.imread(SRC).astype(np.float32)
B,G,R=[img[:,:,k] for k in range(3)]

def fill(img, mask, sigma, feather=1.2, ignore=None):
    m=(mask>0).astype(np.float32); keep=1-m
    if ignore is not None: keep=keep*(1-ignore.astype(np.float32))
    est=img.copy(); filled=np.zeros(m.shape,bool)
    for sg in (sigma, sigma*2, sigma*4, sigma*8):          # widen until every hole pixel has support
        num=cv2.GaussianBlur(img*keep[:,:,None],(0,0),sg); den=cv2.GaussianBlur(keep,(0,0),sg)
        ok=(den>1e-3)&~filled
        est[ok]=num[ok]/den[ok][:,None]; filled|=ok
    soft=np.clip(cv2.GaussianBlur(m,(0,0),feather)*(1.0 if feather<2 else 1.6),0,1)[:,:,None]
    return img*(1-soft)+est*soft

def box(x0,y0,x1,y1):
    m=np.zeros(img.shape[:2],np.uint8); m[y0:y1,x0:x1]=1; return m

# 1) top logo + 2) blue line 2: erase on the soft wall background
nonbg=((B-R>40)|(img.max(2)<150)).astype(np.uint8)
m1=nonbg*box(392,48,714,178); m1=cv2.dilate(m1,np.ones((9,9),np.uint8))
img=fill(img,m1,18)
blue=((B-R>50)&(B>120)).astype(np.uint8)
m2=box(285,283,848,366)
img=fill(img,m2,14,feather=9,ignore=(img.max(2)<110))

# 3) phone screen: navy -> charcoal, bright blues -> Folkes red (screen area only)
x0,y0,x1,y1=282,401,598,1030
sub=np.clip(img[y0:y1,x0:x1],0,255).astype(np.uint8)
hsv=cv2.cvtColor(sub,cv2.COLOR_BGR2HSV).astype(np.float32)
h,s,v=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
isblue=(h>88)&(h<135)&(s>60)
dark=isblue&(v<150); bright=isblue&(v>=150)
hsv[:,:,1]=np.where(dark,s*0.12,s)
hsv[:,:,2]=np.where(dark,v*0.55,v)
hsv[:,:,0]=np.where(bright,175,h)
sub2=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR).astype(np.float32)
# only inside the phone screen (rounded rect), not the hand
scr=np.zeros(sub.shape[:2],np.uint8); cv2.rectangle(scr,(8,8),(x1-x0-8,y1-y0-8),1,-1)
img[y0:y1,x0:x1]=sub2

def screen_bg(x0,y0,x1,y1):
    ring=np.concatenate([img[y0-4:y0,x0:x1].reshape(-1,3),img[y1:y1+4,x0:x1].reshape(-1,3)])
    return np.median(ring,0)
for bx in [(343,486,550,570),(366,808,560,841),(362,879,514,909)]:
    a,b,c,d=bx; img=fill(img,box(a,b,c,d),10,feather=3)

# 7) blue sub line -> light red
xs0,ys0,xs1,ys1=230,1172,892,1212
sub=np.clip(img[ys0:ys1,xs0:xs1],0,255).astype(np.uint8)
hsv=cv2.cvtColor(sub,cv2.COLOR_BGR2HSV).astype(np.float32)
isb=(hsv[:,:,0]>88)&(hsv[:,:,0]<135)&(hsv[:,:,1]>60)
hsv[:,:,0]=np.where(isb,177,hsv[:,:,0]); hsv[:,:,1]=np.where(isb,hsv[:,:,1]*0.85,hsv[:,:,1])
img[ys0:ys1,xs0:xs1]=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)

# 9) url: erase white text on the dark photo
wh=(img.min(2)>150).astype(np.uint8)*box(430,1290,692,1324)
img=fill(img,cv2.dilate(wh,np.ones((5,5),np.uint8)),8)

out=Image.fromarray(cv2.cvtColor(np.clip(img,0,255).astype(np.uint8),cv2.COLOR_BGR2RGB)).convert('RGBA')
d=ImageDraw.Draw(out)
F=lambda n,s: ImageFont.truetype('/tmp/fonts/'+n,s)

def fitlogo(maxw,maxh):
    L=Image.open(LOGO).convert('RGBA'); L=L.crop(L.getbbox())
    k=min(maxw/L.width,maxh/L.height); return L.resize((round(L.width*k),round(L.height*k)),Image.LANCZOS)
# top logo with a soft shadow
L=fitlogo(300,150); cx,cy=552,113
sh=Image.new('RGBA',out.size,(0,0,0,0)); a=L.split()[3].point(lambda p:int(p*.35))
blk=Image.new('RGBA',L.size,(0,0,0,255)); blk.putalpha(a); sh.paste(blk,(cx-L.width//2+3,cy-L.height//2+5),blk)
out.alpha_composite(sh.filter(ImageFilter.GaussianBlur(6))); out.alpha_composite(L,(cx-L.width//2,cy-L.height//2))
# line 2
def ctext(text,font,cx,top,fill,track=0):
    if track:
        ws=[d.textlength(ch,font=font) for ch in text]; w=sum(ws)+track*(len(text)-1); x=cx-w/2
        for ch,cw in zip(text,ws): d.text((x,top),ch,font=font,fill=fill); x+=cw+track
    else:
        w=d.textlength(text,font=font); d.text((cx-w/2,top),text,font=font,fill=fill)
f2=F('Anton-Regular.ttf',10); 
# size Anton so cap height = 56px (A cap ~0.73 em in Anton? measure)
def cap_size(name,target):
    for s in range(10,200):
        f=F(name,s); bb=f.getbbox('H')
        if bb[3]-bb[1]>=target: return f
f2=cap_size('Anton-Regular.ttf',56); bb=f2.getbbox('H')
ctext('FOLKES GETS IT TOO.',f2,561,295-bb[1],RED+(255,))
# phone logo
L2=fitlogo(185,72); out.alpha_composite(L2,(446-L2.width//2,528-L2.height//2))
# notified + contact
fn=F('Poppins-SemiBold.ttf',16)
d.text((371,814),'Folkes has been notified',font=fn,fill=(255,255,255,255))
fc=F('Poppins-SemiBold.ttf',21); bbc=fc.getbbox('C')
ctext('Contact Folkes',fc,436,885-bbc[1],(255,255,255,255))
# CTA pill
d.rounded_rectangle((268,1223,852,1279),radius=28,fill=RED+(255,))
fp=F('Poppins-SemiBold.ttf',20); bbp=fp.getbbox('A')
ctext('Ask Folkes about your smart thermostat upgrade.',fp,560,1243-bbp[1],(255,255,255,255))
# url
fu=F('Poppins-Medium.ttf',21); bbu=fu.getbbox('f')
ctext('folkeshomeservices.com',fu,561,1297-bbu[1],(255,255,255,255),track=2.2)
out.convert('RGB').save('/tmp/flyer/folkes_flyer.jpg',quality=93)
print('ok')
