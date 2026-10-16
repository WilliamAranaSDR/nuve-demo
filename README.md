# nuve-demo

Personalized Nuve pages for prospects, one folder per company, hosted on GitHub Pages.

Build a page:

    python3 tools/make_page.py --company "Summit Heating & Air" --logo logo.png --city Phoenix --first Mike

This writes `<slug>/index.html`, `stat.png` (hero) and `og.jpg` (iMessage preview).
Link: https://williamaranasdr.github.io/nuve-demo/<slug>/

Options: `--slug`, `--keep-dark` (keep dark logo colors), `--scale 0.9`, `--nudge-x/--nudge-y` (px).
Shared media is in `assets/`; copy and layout are in `tools/template.html`.
