#!/usr/bin/env python3
"""Build a personalized Nuve prospect page.

Usage:
  python3 tools/make_page.py --company "Summit Heating & Air" --logo logo.png
        [--slug summit-heating] [--city "Phoenix"] [--first "Mike"]
        [--keep-dark] [--scale 1.0] [--nudge-x 0] [--nudge-y 0]

Writes <repo>/<slug>/index.html, stat.png (hero), og.jpg (iMessage preview 1200x630)
and _check.png (zoomed logo crop for review; not committed).
With a logo it also brands the clips (tools/brand_videos.py): app.mp4/jpg (logo tiles),
alerts.mp4/jpg (logo on the stat face) and dashboard.mp4/jpg (brand color + logo), plus
_check_videos.png (one frame of each, for review; not committed). --color "#hex" overrides
the dashboard color; --no-videos skips this and uses the stock clips.
"""
import argparse, html, os, re, urllib.parse
from collections import deque
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import brand_videos

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, 'tools')
BASE_URL = 'https://williamaranasdr.github.io/nuve-demo'
PHONE = '+17144744974'

# stat_base.png geometry (532x543): screen circle + "YOUR LOGO HERE" box
BOX = (196, 413, 337, 492)        # erase area (box + its glow)
BOX_C = (266, 452)                # logo center
LOGO_MAX = (150, 62)              # max logo w, h on the screen
SCREEN = (19, 19, 20)
UP = 2                            # work at 2x for crisp output


def prep_logo(path, keep_dark=False):
    """Remove a solid background (edge flood fill) and lift black parts to white."""
    a = np.array(Image.open(path).convert('RGBA')).astype(int)
    a[a[:, :, 3] < 20, 3] = 0
    if (a[:, :, 3] == 0).mean() < 0.02:
        h, w = a.shape[:2]
        rgb = a[:, :, :3]
        bg = np.median(np.array([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]]), axis=0)
        close = np.abs(rgb - bg).sum(2) < 60
        seen = np.zeros((h, w), bool); q = deque()
        for x in range(w):
            for y in (0, h - 1):
                if close[y, x] and not seen[y, x]: seen[y, x] = True; q.append((y, x))
        for y in range(h):
            for x in (0, w - 1):
                if close[y, x] and not seen[y, x]: seen[y, x] = True; q.append((y, x))
        while q:
            y, x = q.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and close[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True; q.append((ny, nx))
        a[seen, 3] = 0
    if not keep_dark:
        lum = a[:, :, 0] * .2126 + a[:, :, 1] * .7152 + a[:, :, 2] * .0722
        dark = (lum < 62) & (a[:, :, 3] > 0)  # would vanish on the black screen
        a[dark, :3] = 255
    out = Image.fromarray(a.astype(np.uint8))
    bb = out.getbbox()
    if not bb:
        raise SystemExit('Logo is empty after background removal; try --keep-dark.')
    return out.crop(bb)


def logo_color(logo):
    """Most saturated common color in the logo, for small accents. Falls back to Nuve cyan."""
    a = np.array(logo.convert('RGBA')).reshape(-1, 4).astype(float)
    a = a[a[:, 3] > 200][:, :3]
    if len(a) == 0: return '#45e3f7'
    mx, mn = a.max(1), a.min(1)
    sat = (mx - mn) / np.maximum(mx, 1)
    lum = a[:, 0] * .2126 + a[:, 1] * .7152 + a[:, 2] * .0722
    good = a[(sat > 0.45) & (lum > 85)]
    if len(good) < 50: return '#45e3f7'
    q = (good // 32).astype(int)
    keys, counts = np.unique(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], return_counts=True)
    k = keys[counts.argmax()]
    sel = good[(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]) == k].mean(0)
    return '#%02x%02x%02x' % tuple(int(v) for v in sel)


def brand_color(logo):
    """Main brand color for the dashboard theme: the most common saturated color in the logo
    (dark navies count). Black/white/gray logos get a charcoal theme."""
    a = np.array(logo.convert('RGBA')).reshape(-1, 4).astype(float)
    a = a[a[:, 3] > 200][:, :3]
    if len(a):
        mx, mn = a.max(1), a.min(1)
        sat = (mx - mn) / np.maximum(mx, 1)
        good = a[(sat > 0.35) & (mx > 45)]
        if len(good) >= max(50, 0.04 * len(a)):
            q = (good // 32).astype(int); key = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
            keys, counts = np.unique(key, return_counts=True)
            return tuple(int(v) for v in good[key == keys[counts.argmax()]].mean(0))
    return (42, 47, 54)


def make_stat(logo, scale, nx, ny):
    base = Image.open(os.path.join(TOOLS, 'stat_base.png')).convert('RGBA')
    base = base.resize((base.width * UP, base.height * UP), Image.LANCZOS)
    d = ImageDraw.Draw(base)
    x0, y0, x1, y1 = [v * UP for v in BOX]
    d.rounded_rectangle((x0, y0, x1, y1), radius=10 * UP, fill=SCREEN + (255,))
    mw, mh = LOGO_MAX[0] * UP * scale, LOGO_MAX[1] * UP * scale
    k = min(mw / logo.width, mh / logo.height)
    w, h = max(1, round(logo.width * k)), max(1, round(logo.height * k))
    L = logo.resize((w, h), Image.LANCZOS)
    cx, cy = (BOX_C[0] + nx) * UP, (BOX_C[1] + ny) * UP
    pos = (int(cx - w / 2), int(cy - h / 2))
    r, g, b, a = L.split()
    glow = Image.merge('RGBA', (r, g, b, a.point(lambda p: int(p * 0.45)))).filter(ImageFilter.GaussianBlur(5 * UP))
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    layer.alpha_composite(glow, pos)
    lit = Image.merge('RGBA', (*[c.point(lambda p: int(p * 0.95)) for c in (r, g, b)], a))
    layer.alpha_composite(lit, pos)
    base.alpha_composite(layer)
    return base


def make_og(stat, accent):
    W, H = 1200, 630
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    # deep graphite with a soft cyan bloom behind the stat
    d = np.hypot((xx - W / 2) / W, (yy - H / 2) / H)
    bgc = np.array([8, 10, 14], float)
    bloom = np.array([40, 190, 215], float)
    t = np.clip(1 - d / 0.55, 0, 1) ** 2.2 * 0.38
    img = bgc + (bloom - bgc) * t[..., None]
    # faint diagonal sheen
    s = np.clip(1 - np.abs((xx - yy * 1.2 - 150) / 260), 0, 1) * 6
    img = np.clip(img + s[..., None], 0, 255)
    bg = Image.fromarray(img.astype(np.uint8), 'RGB').convert('RGBA')
    sh = 525
    st = stat.resize((round(stat.width * sh / stat.height), sh), Image.LANCZOS)
    # contact shadow
    shadow = Image.new('RGBA', bg.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse((W / 2 - 200, H / 2 + sh / 2 - 40, W / 2 + 200, H / 2 + sh / 2 + 6), fill=(0, 0, 0, 150))
    bg.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(18)))
    bg.alpha_composite(st, ((W - st.width) // 2, (H - st.height) // 2))
    return bg.convert('RGB')


def slugify(s):
    s = re.sub(r'&', ' and ', s.lower())
    s = re.sub(r'\b(llc|inc|co|corp|ltd)\b\.?', '', s)
    return re.sub(r'[^a-z0-9]+', '-', s).strip('-')[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--company', required=True)
    ap.add_argument('--logo')
    ap.add_argument('--slug')
    ap.add_argument('--city', default='')
    ap.add_argument('--first', default='')
    ap.add_argument('--keep-dark', action='store_true')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--nudge-x', type=float, default=0)
    ap.add_argument('--nudge-y', type=float, default=0)
    ap.add_argument('--color', help='brand color for the dashboard theme, e.g. "#c8102e" (default: from logo)')
    ap.add_argument('--no-videos', action='store_true', help='skip branding the app, stat and dashboard clips')
    args = ap.parse_args()

    slug = args.slug or slugify(args.company)
    out = os.path.join(ROOT, slug); os.makedirs(out, exist_ok=True)

    if args.logo:
        logo = prep_logo(args.logo, args.keep_dark)
        accent = logo_color(logo)
        stat = make_stat(logo, args.scale, args.nudge_x, args.nudge_y)
    else:
        logo, accent = None, '#45e3f7'
        stat = Image.open(os.path.join(TOOLS, 'stat_base.png')).convert('RGBA')
        stat = stat.resize((stat.width * UP, stat.height * UP), Image.LANCZOS)
    stat.resize((stat.width // 1, stat.height // 1)).save(os.path.join(out, 'stat.png'), optimize=True)
    make_og(stat, accent).save(os.path.join(out, 'og.jpg'), quality=88)
    x0, y0, x1, y1 = [v * UP for v in BOX]
    stat.crop((x0 - 120, y0 - 120, x1 + 120, y1 + 80)).save(os.path.join(out, '_check.png'))

    vids = {k: f'../assets/{k}' for k in ('app.mp4', 'app.jpg', 'alerts.mp4', 'alerts.jpg', 'dashboard.mp4', 'dashboard.jpg')}
    if logo is not None and not args.no_videos:
        if args.color:
            h = args.color.lstrip('#'); brand = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        else:
            brand = brand_color(prep_logo(args.logo, keep_dark=True))
        brand_videos.brand_all(prep_logo(args.logo, keep_dark=True), logo, brand, out, bg=brand_videos.logo_bg(args.logo))
        vids = {k: k for k in vids}
        print('videos branded, theme color #%02x%02x%02x' % brand)

    co = html.escape(args.company)
    first = args.first.strip()
    greet = f'{html.escape(first)}, here' if first else 'Here'
    sms = urllib.parse.quote(f"Hi Will, it's {first + ' ' if first else ''}from {args.company}. I checked out the Nuve page, let's talk.")
    city_line = f' across {html.escape(args.city)}' if args.city else ''
    tpl = open(os.path.join(TOOLS, 'template.html'), encoding='utf-8').read()
    rep = {
        '{{COMPANY}}': co,
        '{{GREET}}': greet,
        '{{CITY_LINE}}': city_line,
        '{{SLUG}}': slug,
        '{{URL}}': f'{BASE_URL}/{slug}/',
        '{{OG}}': f'{BASE_URL}/{slug}/og.jpg',
        '{{SMS}}': f'sms:{PHONE}?&body={sms}',
        '{{ACCENT}}': accent,
        '{{APP_MP4}}': vids['app.mp4'], '{{APP_JPG}}': vids['app.jpg'],
        '{{ALERTS_MP4}}': vids['alerts.mp4'], '{{ALERTS_JPG}}': vids['alerts.jpg'],
        '{{DASH_MP4}}': vids['dashboard.mp4'], '{{DASH_JPG}}': vids['dashboard.jpg'],
    }
    for k, v in rep.items(): tpl = tpl.replace(k, v)
    open(os.path.join(out, 'index.html'), 'w', encoding='utf-8').write(tpl)
    print(f'built {slug}/  accent {accent}  logo {logo.size if logo else None}')
    print(f'{BASE_URL}/{slug}/')
    print(f'review: {slug}/_check.png, {slug}/og.jpg' + (f', {slug}/_check_videos.png (app tiles, stat face, dashboard)' if os.path.exists(os.path.join(out, '_check_videos.png')) else ''))


if __name__ == '__main__':
    main()
