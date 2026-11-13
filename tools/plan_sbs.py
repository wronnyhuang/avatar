#!/usr/bin/env python3
"""Planner: side-by-side frame strips of the same open-set item under two configs (for looking, not scoring).

  tools/plan_sbs.py --a <round>/<configA> --b <round>/<configB> --ids id1,id2,... --out <dir> [--set <jsonl>]

Per item one JPEG: the request on top, then config A's character close-up strip (zstrip_00, else strip_00) and config
B's under it, each labelled with its config, judge score and first-motion time. Open sets only.
"""
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def font(sz):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def panel(d: Path, judge: str, width: int):
    rj = json.loads((d / 'render.json').read_text()) if (d / 'render.json').exists() else {}
    z = (rj.get('zoom') or {}).get('files') or []
    f = d / (z[0] if z else (rj.get('strips') or ['strip_00.jpg'])[0])
    im = Image.open(f).convert('RGB') if f.exists() else Image.new('RGB', (width, 200), (40, 40, 40))
    im = im.resize((width, int(im.height * width / im.width)))
    j = json.loads((d / f'judge.{judge}.json').read_text()) if (d / f'judge.{judge}.json').exists() else {}
    sim = json.loads((d / 'sim.json').read_text()) if (d / 'sim.json').exists() else {}
    return im, j.get('score'), sim.get('first_motion_ms'), j.get('reason', '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True); ap.add_argument('--b', required=True); ap.add_argument('--ids', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--set'); ap.add_argument('--judge', default='opus-zoom')
    ap.add_argument('--width', type=int, default=1200)
    a = ap.parse_args()
    A, B, out = Path(a.a), Path(a.b), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    if a.set:
        rows = {r['id']: r for r in map(json.loads, Path(a.set).read_text().splitlines()) if r.get('id')}
    F, f2 = font(26), font(20)
    for i in a.ids.split(','):
        pa = panel(A / i, a.judge, a.width)
        pb = panel(B / i, a.judge, a.width)
        H = 60 + 40 + pa[0].height + 40 + pb[0].height
        img = Image.new('RGB', (a.width, H), (255, 255, 255))
        dr = ImageDraw.Draw(img)
        dr.text((10, 15), f'{i}: "{(rows.get(i) or {}).get("utterance", "")}"', fill=(0, 0, 0), font=F)
        y = 60
        for name, (im, sc, fm, _) in ((A.name, pa), (B.name, pb)):
            dr.text((10, y + 8), f'{name}   judge {sc}   first motion {fm} ms', fill=(20, 20, 160), font=f2)
            img.paste(im, (0, y + 40))
            y += 40 + im.height
        img.save(out / f'{i}.jpg', quality=85)
        print(out / f'{i}.jpg', pa[1], pb[1])


if __name__ == '__main__':
    main()
