#!/usr/bin/env python3
"""Brand the three feature clips with a contractor's logo.

  app.mp4        "YOUR LOGO HERE" tiles  -> their logo on app-icon tiles
  alerts.mp4     "nuve" on the stat face -> their logo (lit, like the hero stat)
  dashboard.mp4  cyan theme + "nuve"     -> their brand color + their logo

Called from make_page.py. Everything is measured from the base clips in assets/,
so the only inputs are the logo and a brand color. Pure compute, no network.
"""
import os, subprocess, colorsys
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, 'assets')
CRF = '25'


# ---------- io ----------
def read(name):
    cap = cv2.VideoCapture(os.path.join(ASSETS, name))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    return fr, fps


def write(frames, fps, path, poster_idx, poster_path):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                          '-s', f'{w}x{h}', '-r', f'{fps:.5f}', '-i', '-', '-an', '-c:v', 'libx264',
                          '-preset', 'medium', '-crf', CRF, '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path],
                         stdin=subprocess.PIPE)
    for f in frames: p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()
    if p.returncode: raise SystemExit(f'ffmpeg failed on {path}')
    cv2.imwrite(poster_path, frames[poster_idx], [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------- drawing helpers ----------
def fit(logo, mw, mh, ss=4):
    """Resize RGBA logo to fit mw x mh (float px), supersampled for crisp tiny output."""
    k = min(mw / logo.width, mh / logo.height)
    w, h = max(1, round(logo.width * k)), max(1, round(logo.height * k))
    big = logo.resize((w * ss, h * ss), Image.LANCZOS)
    return big.resize((w, h), Image.LANCZOS)


def rgba_np(im):
    a = np.array(im.convert('RGBA')).astype(np.float32)
    return a[:, :, 2::-1], a[:, :, 3:] / 255.0   # BGR, alpha


def paste(frame, im, x, y, opacity=1.0):
    """Alpha-composite PIL RGBA onto BGR frame (in place) at integer x, y with clipping."""
    bgr, al = rgba_np(im); al = al * opacity
    H, W = frame.shape[:2]; h, w = al.shape[:2]
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    sb = bgr[y0 - y:y1 - y, x0 - x:x1 - x]; sa = al[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(roi * (1 - sa) + sb * sa, 0, 255).astype(np.uint8)


def silhouette(logo, rgb):
    a = logo.split()[3]
    s = Image.new('RGBA', logo.size, tuple(rgb) + (255,)); s.putalpha(a)
    return s


def lum(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def cyanness(f):
    f = f.astype(np.int16)
    return np.clip(np.minimum(f[:, :, 0], f[:, :, 1]) - f[:, :, 2], 0, 255).astype(np.float32)


# ---------- app clip: tiles ----------
APP_TILES = [(454, 166, 198, 198), (54, 364, 198, 198), (454, 470, 198, 198)]
APP_REF = 70
APP_PLATE = 32                          # phone settled, no tiles yet: clean background for the fade-out
APP_GAIN_BOX = (150, 185, 140, 140)     # app icon: static, never covered, used to follow the global fade


def logo_bg(path):
    """The logo file's own solid background color (RGB), or None if it's transparent."""
    a = np.array(Image.open(path).convert('RGBA'))
    if (a[:, :, 3] < 20).mean() >= 0.02: return None
    c = np.median(np.array([a[0, 0, :3], a[0, -1, :3], a[-1, 0, :3], a[-1, -1, :3]]), axis=0)
    return tuple(int(v) for v in c)


def app_tile(logo_light, brand, size=198):
    """App-icon tile: dark brand-tinted glass with a soft brand glow behind the logo and a thin
    brand-colored rim, logo in its light (dark-screen) version, trimmed to its visible pixels and
    centered so it reads big and balanced on every tile."""
    ss = 4; S = size * ss
    h, l, s_ = colorsys.rgb_to_hls(*[v / 255 for v in brand])
    dark = tuple(int(v * 255) for v in colorsys.hls_to_rgb(h, 0.09, min(s_, 0.55)))
    rim = tuple(int(v * 255) for v in colorsys.hls_to_rgb(h, 0.62, min(1.0, max(s_, 0.5))))
    glow = tuple(int(v * 255) for v in colorsys.hls_to_rgb(h, 0.45, min(1.0, max(s_, 0.6))))
    base = np.zeros((S, S, 3), np.float32); base[:] = dark
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    r = np.sqrt(((xx - S / 2) / (S * 0.48)) ** 2 + ((yy - S / 2) / (S * 0.34)) ** 2)
    g = np.clip(1 - r, 0, 1) ** 1.6 * 0.35
    base = base * (1 - g[..., None]) + np.array(glow, np.float32) * g[..., None]
    fill = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert('RGBA')
    mask = Image.new('L', (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, S - 1, S - 1), radius=int(S * 0.2), fill=255)
    t = Image.new('RGBA', (S, S), (0, 0, 0, 0)); t.paste(fill, (0, 0), mask)
    bw = int(S * 0.013)
    ImageDraw.Draw(t).rounded_rectangle((bw // 2, bw // 2, S - 1 - bw // 2, S - 1 - bw // 2),
                                        radius=int(S * 0.2), outline=rim + (255,), width=bw)
    bbox = logo_light.split()[3].point(lambda v: 255 if v > 24 else 0).getbbox()
    L = logo_light.crop(bbox) if bbox else logo_light
    ar = L.width / L.height
    mw, mh = (S * 0.82, S * 0.46) if ar > 2.2 else (S * 0.70, S * 0.62)   # wide wordmarks vs square marks
    L = fit(L, mw, mh, ss=1)
    t.alpha_composite(L, ((S - L.width) // 2, (S - L.height) // 2))
    return t.resize((size, size), Image.LANCZOS)


def brand_app(logo_light, brand, out_dir):
    fr, fps = read('app.mp4')
    tile = app_tile(logo_light, brand)
    tile_bgr, tile_a = rgba_np(tile)
    ref = fr[APP_REF]; refc = cyanness(ref)
    refs = []
    for (x, y, w, h) in APP_TILES:
        rc = ref[y:y + h, x:x + w].astype(np.float32)
        cm = refc[y:y + h, x:x + w] > 90                       # cyan (non-text) pixels of the tile
        new_over_ref = rc * (1 - tile_a[:h, :w]) + tile_bgr[:h, :w] * tile_a[:h, :w]
        refs.append((rc, cm, refc[y:y + h, x:x + w][cm].mean(), new_over_ref))
    centers = [(x + w / 2, y + h / 2) for x, y, w, h in APP_TILES]
    plate = cv2.GaussianBlur(fr[APP_PLATE], (0, 0), 3).astype(np.float32)   # frosted-glass look
    gx, gy, gw, gh = APP_GAIN_BOX
    plate_ref = float(fr[APP_PLATE][gy:gy + gh, gx:gx + gw].mean())
    out = []
    landed = [False] * len(APP_TILES)
    for f in fr:
        g = f.copy(); c = cyanness(f)
        n, lab, st, cen = cv2.connectedComponentsWithStats((c > 60).astype(np.uint8))
        comps = [st[i] for i in range(1, n) if st[i][4] > 8000]
        used = {}
        for s in comps:
            cx, cy = s[0] + s[2] / 2, s[1] + s[3] / 2
            k = int(np.argmin([(cx - a) ** 2 + (cy - b) ** 2 for a, b in centers]))
            if k not in used or s[4] > used[k][4]: used[k] = s
        for k, (x, y, w, h) in enumerate(APP_TILES):
            rc, cm, rmean, nor = refs[k]
            s = used.get(k)
            off = s is not None and (abs(s[0] + s[2] / 2 - centers[k][0]) > 25 or abs(s[1] + s[3] / 2 - centers[k][1]) > 25)
            if s is not None and not off and abs(s[2] - w) <= 8 and abs(s[3] - h) <= 8: landed[k] = True
            if s is not None and not landed[k]:
                # tile is moving (flying in / out): erase it back to the clean phone plate,
                # then paint the new tile over its (possibly blurred / edge-clipped) box
                bx, by, bw, bh = [int(v) for v in s[:4]]
                comp = (c[by:by + bh, bx:bx + bw] > 12).astype(np.uint8)
                comp = cv2.dilate(comp, np.ones((13, 13), np.uint8))
                comp = cv2.GaussianBlur(comp.astype(np.float32), (0, 0), 1.5)[..., None]
                gain = float(f[gy:gy + gh, gx:gx + gw].mean()) / max(1.0, plate_ref)
                clean = fr[APP_PLATE][by:by + bh, bx:bx + bw].astype(np.float32) * gain
                roi = g[by:by + bh, bx:bx + bw].astype(np.float32)
                g[by:by + bh, bx:bx + bw] = np.clip(roi * (1 - comp) + clean * comp, 0, 255).astype(np.uint8)
                a = float(np.clip(c[by:by + bh, bx:bx + bw].max() / refc.max(), 0, 1))
                if bx == 0: bx, bw = bx + bw - max(bw, bh), max(bw, bh)
                if bx + bw >= f.shape[1] - 1: bw = max(bw, bh)
                paste(g, tile.resize((max(1, bw), max(1, bh)), Image.LANCZOS), bx, by, opacity=a)
                continue
            # landed or fading: tiles never move after landing, so work in the fixed box and
            # estimate opacity from how cyan it still is
            if not landed[k]: continue
            rcm = refc[y:y + h, x:x + w]
            bx, by, bw, bh = x, y, w, h
            ref_r = cv2.resize(rc, (bw, bh), interpolation=cv2.INTER_AREA)
            rcm_r = cv2.resize(rcm, (bw, bh), interpolation=cv2.INTER_AREA)
            nor_r = cv2.resize(nor, (bw, bh), interpolation=cv2.INTER_AREA)
            obs_c = c[by:by + bh, bx:bx + bw]
            a = float(np.clip((obs_c * rcm_r).sum() / max(1, (rcm_r * rcm_r).sum()), 0, 1))
            if a < 0.08: continue
            roi = g[by:by + bh, bx:bx + bw].astype(np.float32)
            if a >= 0.97:
                g[by:by + bh, bx:bx + bw] = np.clip(roi + a * (nor_r - ref_r), 0, 255).astype(np.uint8)
            else:
                # fading out: the original turns to frosted glass with its own text ghosting through,
                # so rebuild the tile area from the clean phone plate instead of unmixing
                gain = float(f[gy:gy + gh, gx:gx + gw].mean()) / max(1.0, plate_ref)
                bgp = plate[by:by + bh, bx:bx + bw] * gain
                ta = cv2.resize(tile_a, (bw, bh), interpolation=cv2.INTER_AREA)[..., None]
                tb = cv2.resize(tile_bgr, (bw, bh), interpolation=cv2.INTER_AREA)
                inner = bgp * (1 - a) + tb * a
                g[by:by + bh, bx:bx + bw] = np.clip(roi * (1 - ta) + inner * ta, 0, 255).astype(np.uint8)
        out.append(g)
    write(out, fps, os.path.join(out_dir, 'app.mp4'), 60, os.path.join(out_dir, 'app.jpg'))
    return out


# ---------- alerts clip: stat face ----------
AL_REF = 65
AL_WORD = (390, 301, 46, 12)          # "nuve" on the stat face at AL_REF
AL_MAX = (80, 24)                      # logo box (stays clear of the menu/calendar icons)


def brand_alerts(logo_light, out_dir):
    fr, fps = read('alerts.mp4')
    x, y, w, h = AL_WORD; P = 4
    cref = cyanness(fr[AL_REF])
    tpl = cref[y - P:y + h + P, x - P:x + w + P]
    rpeak = cref[y:y + h, x:x + w].max()
    # find the wordmark's x in each frame (the stat slides in horizontally, same scale)
    xs, ks, good = [], [], []
    for f in fr:
        c = cyanness(f)
        band = c[y - P - 6:y + h + P + 6, 250:]
        r = cv2.matchTemplate(band, tpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        fx = ml[0] + 250 + P
        xs.append(fx); good.append(mv > 0.55 and abs(ml[1] - 6) <= 2)
    gi = [i for i, ok in enumerate(good) if ok]
    if not gi: raise SystemExit('alerts: could not track the wordmark')
    X = np.array(xs, float)
    # fill untracked frames: hold the nearest tracked x, extrapolate before the first using early velocity
    first = gi[0]; v = (X[gi[min(3, len(gi) - 1)]] - X[first]) / max(1, gi[min(3, len(gi) - 1)] - first)
    for i in range(len(fr)):
        if good[i]: continue
        if i < first: X[i] = X[first] + v * (i - first)
        else: X[i] = X[max(j for j in gi if j < i)]
    L = fit(logo_light, *AL_MAX)
    out = []
    for i, f in enumerate(fr):
        g = f.copy(); fx = int(round(X[i]))
        c = cyanness(f)
        k = float(np.clip(c[y:y + h, fx:fx + w].max() / rpeak, 0, 1)) if 0 <= fx < f.shape[1] - w else 0
        if k > 0.03:
            # erase "nuve": fill with the dark screen color sampled around it
            ex0, ey0, ex1, ey1 = fx - 3, y - 3, fx + w + 3, y + h + 3
            ring = np.concatenate([g[ey0 - 4:ey0, ex0:ex1].reshape(-1, 3), g[ey1:ey1 + 4, ex0:ex1].reshape(-1, 3)])
            g[ey0:ey1, ex0:ex1] = np.median(ring, 0).astype(np.uint8)
            cx, cy = fx + w / 2, y + h / 2
            paste(g, L, int(round(cx - L.width / 2)), int(round(cy - L.height / 2)), opacity=k)
        out.append(g)
    write(out, fps, os.path.join(out_dir, 'alerts.mp4'), 60, os.path.join(out_dir, 'alerts.jpg'))
    return out


# ---------- dashboard clip: theme + header logo ----------
DB_CYAN = (84, 227, 253)              # header cyan (RGB)
DB_HUE = 93                            # OpenCV hue of the cyan
DB_WORD = (9, 7, 38, 10)               # "nuve" relative to the header's left/top (228, 181) when settled
DB_TOP = 181
DB_MAX = (54, 14)


def to_lab(rgb):
    px = np.uint8([[list(rgb)[::-1]]])
    return cv2.cvtColor(px.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)[0, 0]


def brand_dashboard(logo_color, brand, out_dir):
    fr, fps = read('dashboard.mp4')
    ref = to_lab(DB_CYAN); tgt = to_lab(brand).copy()
    tgt[0] = min(tgt[0], 62)              # keep the header dark enough for white nav text
    delta = (tgt - ref).astype(np.float32)
    ref_chroma = float(np.hypot(ref[1], ref[2]))
    txt = (255, 255, 255)
    tint = (np.array(dashboard_tint(brand), np.float32)[::-1] / 255).reshape(1, 1, 3)   # BGR
    # header left edge per frame (the laptop slides left as the modal comes in)
    hx = []
    for f in fr:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        m = (np.abs(hsv[:, :, 0].astype(int) - DB_HUE) < 14) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 100)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        cand = [st[i] for i in range(1, n) if st[i][4] > 5000 and abs(st[i][1] - DB_TOP) <= 3]
        hx.append(min(s[0] for s in cand) if cand else None)
    gi = [i for i, v in enumerate(hx) if v is not None]
    if not gi: raise SystemExit('dashboard: could not find the header')
    first = gi[0]; v = (hx[gi[1]] - hx[first]) if len(gi) > 1 else 0
    # the logo always sits on a small white badge in its real colors: any logo (navy, black,
    # multicolor, blocky) stays readable on any header color, and nothing turns into a white shape
    ss = 4; pad = 2.5
    Lc = fit(logo_color, (DB_MAX[0] + 6 - 2 * pad) * ss, (DB_MAX[1] + 2 - 2 * pad) * ss, ss=1)
    chip = Image.new('RGBA', (int(Lc.width + 2 * pad * ss), int(Lc.height + 2 * pad * ss)), (0, 0, 0, 0))
    ImageDraw.Draw(chip).rounded_rectangle((0, 0, chip.width - 1, chip.height - 1), radius=3 * ss, fill=(255, 255, 255, 255))
    chip.alpha_composite(Lc, (int(pad * ss), int(pad * ss)))
    L = chip.resize((max(1, chip.width // ss), max(1, chip.height // ss)), Image.LANCZOS)
    out = []
    for i, f in enumerate(fr):
        # recolor every cyan pixel, weighted by how cyan it is (keeps gradients, fades and edges)
        lab = cv2.cvtColor(f.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        dh = np.abs(hsv[:, :, 0].astype(np.float32) - DB_HUE)
        hw = np.clip((14 - dh) / 8, 0, 1)
        cw = np.clip(np.hypot(lab[:, :, 1], lab[:, :, 2]) / ref_chroma, 0, 1)
        wgt = (hw * cw)[..., None]
        if wgt.max() > 0.01:
            lab = lab + wgt * delta
            g = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR) * 255, 0, 255).astype(np.uint8)
        else:
            g = f.copy()
        # white backdrop -> pale brand shade (multiply, so the soft shadows keep their depth)
        bm = bg_mask(f)
        gf = g.astype(np.float32)
        g = np.clip(gf * (1 - bm) + gf * tint * bm, 0, 255).astype(np.uint8)
        x = hx[i] if hx[i] is not None else (hx[first] + v * (i - first) if i < first else hx[max(j for j in gi if j < i)])
        if i >= first - 4:
            dx, dy, w, h = DB_WORD
            wx, wy = int(x + dx), DB_TOP + dy
            ex0, ey0, ex1, ey1 = wx - 2, wy - 2, wx + w + 3, wy + h + 2
            if 2 <= ex0 and ex1 < g.shape[1] - 2:
                left = g[ey0:ey1, ex0 - 2:ex0].astype(np.float32).mean(1)
                right = g[ey0:ey1, ex1:ex1 + 2].astype(np.float32).mean(1)
                t = np.linspace(0, 1, ex1 - ex0, dtype=np.float32)[None, :, None]
                g[ey0:ey1, ex0:ex1] = (left[:, None] * (1 - t) + right[:, None] * t).astype(np.uint8)
                op = float(wgt[wy + h // 2, ex0 - 2, 0]) if wy + h // 2 < g.shape[0] else 1
                paste(g, L, wx, int(round(wy + h / 2 - L.height / 2)), opacity=min(1.0, op * 1.1))
        out.append(g)
    write(out, fps, os.path.join(out_dir, 'dashboard.mp4'), 89, os.path.join(out_dir, 'dashboard.jpg'))
    return out


def dashboard_tint(brand):
    """Pale shade of the brand for the dashboard backdrop: clearly not white, still light behind the white modal."""
    return tuple(int(round(0.30 * c + 0.70 * 255)) for c in brand)


def bg_mask(f):
    """The clip's white backdrop and its soft shadows: flood fill from the borders that follows gentle
    gradients (shadows) but stops at hard edges (the white modal card, the laptop)."""
    H, W = f.shape[:2]
    m = np.zeros((H + 2, W + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    g = f
    # hard edges (modal card outline, laptop) act as walls so the fill can't seep in at soft corners
    gr = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(gr, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gr, cv2.CV_32F, 0, 1, ksize=3)) / 8
    wall = cv2.dilate((mag > 3.5).astype(np.uint8), np.ones((3, 3), np.uint8))
    m[1:-1, 1:-1] = wall
    seeds = [(x, 0) for x in range(0, W, 40)] + [(x, H - 1) for x in range(0, W, 40)] + \
            [(0, y) for y in range(0, H, 40)] + [(W - 1, y) for y in range(0, H, 40)]
    for x, y in seeds:
        if m[y + 1, x + 1] or g[y, x].min() < 240: continue
        cv2.floodFill(g, m, (x, y), 0, (2, 2, 2), (2, 2, 2), flags)
    a = (m[1:-1, 1:-1] == 255).astype(np.uint8)
    # take back the wall pixels on the shadow side, but never step onto the white card
    grow = cv2.dilate(a, np.ones((5, 5), np.uint8)) & wall & (gr < 254).astype(np.uint8)
    a = (a | grow).astype(np.float32)
    return cv2.GaussianBlur(a, (0, 0), 0.7)[..., None]


# ---------- tap image: stat on the wall ----------
TAP_WORD = (164, 147, 196, 160)        # "nuve" on the stat face (x0, y0, x1, y1) in assets/tap.jpg
TAP_C, TAP_MAX = (179.5, 153.5), (40, 15)


def brand_tap(logo_light, out_dir):
    im = cv2.imread(os.path.join(ASSETS, 'tap.jpg'))
    x0, y0, x1, y1 = TAP_WORD
    ring = np.concatenate([im[y0 - 3:y0, x0:x1].reshape(-1, 3), im[y1:y1 + 3, x0:x1].reshape(-1, 3)])
    im[y0:y1, x0:x1] = np.median(ring, 0).astype(np.uint8)
    L = fit(logo_light, *TAP_MAX)
    paste(im, L, int(round(TAP_C[0] - L.width / 2)), int(round(TAP_C[1] - L.height / 2)))
    cv2.imwrite(os.path.join(out_dir, 'tap.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return im


# ---------- phone clip: app home screen ----------
PH_REF = (259, 460, 41)                 # phone left, right, top at the settled frame
PH_C, PH_MAX = (360, 263), (92, 32)     # logo box: under the "Holding" pill, between the arc ends, above Fan/Mode


def brand_phone(logo_light, out_dir):
    fr, fps = read('phone.mp4')
    L0 = fit(logo_light, PH_MAX[0] * 2, PH_MAX[1] * 2)
    boxes, ks = [], []
    for f in fr:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        k = float(g.max()) / 255
        ks.append(k)
        if k < 0.5: boxes.append(None); continue
        ys, xs = np.where(g > 60 * k)
        boxes.append((xs.min(), xs.max(), ys.min()) if len(xs) else None)
    gi = [i for i, b in enumerate(boxes) if b is not None]
    out = []
    for i, f in enumerate(fr):
        g = f.copy()
        b = boxes[i] if boxes[i] is not None else boxes[min(gi, key=lambda j: abs(j - i))]
        s = (b[1] - b[0]) / (PH_REF[1] - PH_REF[0])
        cx = (b[0] + b[1]) / 2 + (PH_C[0] - (PH_REF[0] + PH_REF[1]) / 2) * s
        cy = b[2] + (PH_C[1] - PH_REF[2]) * s
        L = fit(L0, PH_MAX[0] * s, PH_MAX[1] * s)
        if ks[i] > 0.02:
            paste(g, L, int(round(cx - L.width / 2)), int(round(cy - L.height / 2)), opacity=min(1.0, ks[i]))
        out.append(g)
    write(out, fps, os.path.join(out_dir, 'phone.mp4'), 61, os.path.join(out_dir, 'phone.jpg'))
    return out


# ---------- remote diagnostics clip: small stat next to the condenser ----------
RD_C, RD_MAX = (397, 249.5), (42, 15)   # between the menu and calendar icons (1280x720 base)


def brand_remote(logo_light, out_dir):
    fr, fps = read('remotediag.mp4')
    L = fit(logo_light, *RD_MAX)
    x, y = int(round(RD_C[0] - L.width / 2)), int(round(RD_C[1] - L.height / 2))
    out = []
    for f in fr:
        g = f.copy(); paste(g, L, x, y); out.append(g)
    write(out, fps, os.path.join(out_dir, 'remotediag.mp4'), 36, os.path.join(out_dir, 'remotediag.jpg'))
    return out


def brand_all(logo_color, logo_light, brand, out_dir, bg=None):
    a = brand_app(logo_light, brand, out_dir)
    b = brand_alerts(logo_light, out_dir)
    c = brand_dashboard(logo_color, brand, out_dir)
    t = brand_tap(logo_light, out_dir)
    p = brand_phone(logo_light, out_dir)
    r = brand_remote(logo_light, out_dir)
    # review sheet: one settled frame from each clip, plus zooms on the small logos (gitignored)
    th = 300
    row = lambda ims: np.hstack([cv2.resize(t, (int(t.shape[1] * th / t.shape[0]), th)) for t in ims])
    zoom = [t[100:210, 115:245], p[61][180:330, 280:440], r[36][190:310, 320:475]]
    cv2.imwrite(os.path.join(out_dir, '_check_videos.png'), row([a[60], b[60], c[89]]))
    cv2.imwrite(os.path.join(out_dir, '_check_more.png'), row([t, p[61], r[36]] + zoom))
