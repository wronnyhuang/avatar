#!/usr/bin/env python3
"""Planner scorer, stage 3: <itemdir>/rig.jsonl (tools/plan_sim.cjs, already trimmed) -> clip.mp4 + strip_NN.jpg, and with
--face also face.mp4 + face_strip_NN.jpg (a head close-up: the same rig with the camera forced to the medium close-up,
rendered at 2x and cropped to the head, so eyes / lids / brows / mouth are readable).

  python3 tools/plan_render.py <itemdir> [--face] [--threads 4] [--force]

Renderer = THIS worktree's renderer/tools/render_timeline.py on CPU. The body clip keeps the camera motion.js chose (it zooms
out and follows the root for whole-body motion). Strips: tiles at >= 3 fps (4 fps up to 12 s), each stamped with its time
in the clip; 24 tiles a page. render.json lists every file with its sha256 (the judge caches on them).
Env: PYTHON = the interpreter for the renderer child (default: this one); TA_OUT = the renderer's cache dir
(default <repo>/runs/cache/render)."""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = os.environ.get('PYTHON') or sys.executable
CACHE = os.environ.get('TA_OUT') or str(ROOT / 'runs' / 'cache' / 'render')
FPS = 24
BODY = (640, 360)
FACE = (960, 540)
FACE_CROP = (290, 10, 380, 380)        # x, y, w, h in the FACE frame at cam_zoom 0 (the muse's head + ears), see CALIBRATION.md
PAD = 0.5                              # static lead rows so the mixer's smoothing has settled before the clip starts
FACE_CROP_SEATED = (210, 0, 540, 540)   # the seated characters (a bust that fills the frame): the middle square
TILE_BODY = (320, 180)
TILE_FACE = (240, 240)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def render(rig_lines, head, w, h, out_mp4, threads, log, character='muse'):
    tmp = out_mp4.parent / ('_tmp_' + out_mp4.stem)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    n = int(round(PAD * 30))
    rows = rig_lines
    body = [json.dumps(head)] + [json.dumps({'type': 'rig', 't': round(i / 30, 4), 'a': rows[0]['a']}) for i in range(n)]
    body += [json.dumps({'type': 'rig', 't': round(r['t'] + PAD, 4), 'a': r['a']}) for r in rows]
    (tmp / 'rig.jsonl').write_text('\n'.join(body) + '\n')
    dur = rows[-1]['t'] + 1 / 30
    env = dict(os.environ, AVATAR_RT_DEVICE='cpu', TA_OUT=CACHE, LP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
    cmd = [PY, '-B', str(ROOT / 'renderer/tools/render_timeline.py'), '--character', character, '--rig', str(tmp / 'rig.jsonl'), '--w', str(w), '--h', str(h),
           '--fps', str(FPS), '--backend', 'cpu', '--threads', str(threads), '--duration', f'{PAD + dur + 0.05:.3f}', '--crf', '16', '--out', str(tmp / 'hq.mp4')]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(ROOT / 'renderer'))
    log.append(r.stdout[-1500:] + '\n' + r.stderr[-2500:])
    if r.returncode or not (tmp / 'hq.mp4').exists():
        raise RuntimeError(f'render failed: {r.stderr[-500:]}')
    return tmp / 'hq.mp4', dur, tmp


def strips(mp4, dur, tile, prefix, outdir, title):
    """tiles at 4 fps (<= 12 s) else 3 fps, at most 96; 24 a page"""
    from PIL import Image, ImageDraw
    fps = 4.0 if dur <= 12 else 3.0
    n = max(2, min(96, int(dur * fps)))
    step = dur / n
    fd = outdir / ('_tiles_' + prefix)
    for old in outdir.glob(prefix + '_[0-9][0-9].jpg'):
        old.unlink()
    shutil.rmtree(fd, ignore_errors=True)
    fd.mkdir()
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp4), '-vf', f'fps={1 / step:.5f}:start_time={step / 2:.4f}:round=near,scale={tile[0]}:{tile[1]}',
                    '-q:v', '3', str(fd / 't%03d.jpg')], check=True)
    tiles = sorted(fd.glob('t*.jpg'))[:n]
    cols = 6 if tile[0] >= 300 else 8
    per = cols * 4
    lh = 15
    files, times = [], []
    for pg in range(0, len(tiles), per):
        chunk = tiles[pg:pg + per]
        rn = (len(chunk) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * tile[0], rn * (tile[1] + lh)), (245, 245, 245))
        d = ImageDraw.Draw(sheet)
        for i, f in enumerate(chunk):
            t = (pg + i + 0.5) * step
            times.append(round(t, 2))
            x, y = (i % cols) * tile[0], (i // cols) * (tile[1] + lh)
            sheet.paste(Image.open(f).convert('RGB'), (x, y + lh))
            d.text((x + 4, y + 1), f't={t:.2f}s', fill=(20, 20, 160))
        p = outdir / f'{prefix}_{pg // per:02d}.jpg'
        sheet.save(p, quality=82)
        files.append(p.name)
    shutil.rmtree(fd)
    return files, times, fps


ZOOM_TILE = (384, 448)                 # a tile of the character close-up strips (4 x 2 a page)
ZOOM_MAX_TILES = 16
REST_BOX = (0.36, 0.03, 0.64, 0.97)    # where the muse stands at rest in the body frame (x0, y0, x1, y1 as fractions)


def zoom_strips(out, dur, character='muse'):
    """Character close-up strips from the clip that is already rendered (no new render): zstrip_NN.jpg.
    A strip tile of the whole room gives a vision model only a dozen patches of the character, so a small in-place motion
    (a wave's swings, a shrug, a lean, a nod) is invisible to it. When the motion stays in place (the pixels that change
    over the clip stay inside a box around the character and the camera holds still) the box is cut out of every sampled
    frame at native resolution and enlarged: up to 16 tiles, 6 fps for a short clip. A whole-body move (walking, jumping,
    lying down, the camera following) keeps the wide strips: -> {'mode': 'wide'}."""
    import numpy as np
    from PIL import Image, ImageDraw
    for old in out.glob('zstrip_[0-9][0-9].jpg'):
        old.unlink()
    W, H = BODY
    w, h = W // 2, H // 2
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(out / 'clip.mp4'), '-vf', f'fps=8,scale={w}:{h},format=gray', '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
    n = len(raw) // (w * h)
    if n < 2:
        return {'mode': 'wide', 'why': 'clip too short'}
    fr = np.frombuffer(raw[:n * w * h], dtype=np.uint8).reshape(n, h, w).astype(np.int16)
    med = np.median(fr, axis=0)
    ch = (np.abs(fr - med) > 22).any(axis=0)
    frac = float(ch.mean())
    # drifting dust motes / light shimmer change stray pixels all over the room: only columns and rows with a real run of
    # changed pixels count (an arm is thicker than that)
    colsum, rowsum = ch.sum(axis=0), ch.sum(axis=1)
    keep = np.outer(rowsum >= max(3, 0.03 * w), colsum >= max(3, 0.04 * h))
    ch = ch & keep
    frac = float(ch.mean())
    # the room has its own life away from the character (the plant sways, the sunbeam shimmers): the box grows only with
    # changes inside the character's reach; a real move across the room moves the camera too (frac is large then)
    reach = np.zeros_like(ch)
    reach[:, int(0.2 * w):int(0.8 * w)] = True
    far = float((ch & ~reach).mean())
    ys, xs = np.nonzero(ch & reach)
    if character == 'muse':
        x0, y0, x1, y1 = REST_BOX[0] * w, REST_BOX[1] * h, REST_BOX[2] * w, REST_BOX[3] * h
    else:
        x0, y0, x1, y1 = 0.2 * w, 0, 0.8 * w, h
    if len(xs):
        # robust extent of the changed pixels (2nd..98th percentile: stray pixels do not widen the box)
        x0, x1 = min(x0, float(np.percentile(xs, 1))), max(x1, float(np.percentile(xs, 99)))
        y0, y1 = min(y0, float(np.percentile(ys, 1))), max(y1, float(np.percentile(ys, 99)))
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    if frac > 0.30 or far > 0.06 or bw > 0.62:
        return {'mode': 'wide', 'why': f'whole-body / camera motion (changed {frac:.2f} of the frame, box width {bw:.2f})'}
    # the box at the tile's aspect, padded, clamped to the frame (native clip pixels)
    cx, cy = (x0 + x1) / w, (y0 + y1) / h
    bh = min(1.0, bh * 1.06)
    bwpx = max((x1 - x0) * 2 * 1.10, bh * H * ZOOM_TILE[0] / ZOOM_TILE[1])
    bhpx = max(bh * H, bwpx * ZOOM_TILE[1] / ZOOM_TILE[0])
    if bhpx > H:
        bhpx = H; bwpx = max(bwpx, H * ZOOM_TILE[0] / ZOOM_TILE[1])
    bwpx = min(bwpx, W)
    X = int(round(min(max(cx * W / 2 - bwpx / 2, 0), W - bwpx))); Y = int(round(min(max(cy * H / 2 - bhpx / 2, 0), H - bhpx)))
    bwpx, bhpx = int(bwpx) // 2 * 2, int(bhpx) // 2 * 2
    fps = 6.0 if dur * 6 <= ZOOM_MAX_TILES else max(1.5, ZOOM_MAX_TILES / dur)
    k = max(2, min(ZOOM_MAX_TILES, int(dur * fps)))
    step = dur / k
    fd = out / '_tiles_zoom'
    shutil.rmtree(fd, ignore_errors=True)
    fd.mkdir()
    # pad the crop to the tile aspect instead of stretching it
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(out / 'clip.mp4'), '-vf',
                    f'fps={1 / step:.5f}:start_time={step / 2:.4f}:round=near,crop={bwpx}:{bhpx}:{X}:{Y},scale={ZOOM_TILE[0]}:{ZOOM_TILE[1]}:force_original_aspect_ratio=decrease:flags=lanczos,'
                    f'pad={ZOOM_TILE[0]}:{ZOOM_TILE[1]}:(ow-iw)/2:(oh-ih)/2:color=0xF5F5F5', '-q:v', '3', str(fd / 't%03d.jpg')], check=True)
    tiles = sorted(fd.glob('t*.jpg'))[:k]
    cols, per, lh = 4, 8, 16
    files = []
    for pg in range(0, len(tiles), per):
        chunk = tiles[pg:pg + per]
        rn = (len(chunk) + cols - 1) // cols
        sheet = Image.new('RGB', (cols * ZOOM_TILE[0], rn * (ZOOM_TILE[1] + lh)), (245, 245, 245))
        d = ImageDraw.Draw(sheet)
        for i, f in enumerate(chunk):
            t = (pg + i + 0.5) * step
            x, y = (i % cols) * ZOOM_TILE[0], (i // cols) * (ZOOM_TILE[1] + lh)
            sheet.paste(Image.open(f).convert('RGB'), (x, y + lh))
            d.text((x + 4, y + 2), f't={t:.2f}s', fill=(20, 20, 160))
        p = out / f'zstrip_{pg // per:02d}.jpg'
        sheet.save(p, quality=85)
        files.append(p.name)
    shutil.rmtree(fd)
    return {'mode': 'zoom', 'files': files, 'shas': [sha(out / f) for f in files], 'box': [X, Y, bwpx, bhpx], 'fps': round(1 / step, 2), 'tiles': len(tiles), 'changed': round(frac, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('itemdir'); ap.add_argument('--face', action='store_true'); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--force', action='store_true'); ap.add_argument('--title', default=''); ap.add_argument('--zoom-only', action='store_true')
    ap.add_argument('--root', help='source tree whose renderer/ draws the clip (default: this worktree)'); ap.add_argument('--character', default='muse')
    a = ap.parse_args()
    global ROOT
    if a.root:
        ROOT = Path(a.root).resolve()
    out = Path(a.itemdir).resolve()
    rj = out / 'render.json'
    if a.zoom_only:
        # add the character close-up strips to an item that is already rendered (the clip is not rendered again)
        old = json.loads(rj.read_text())
        if not old.get('ok'):
            sys.exit('not rendered')
        dur = old.get('duration_s') or float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(out / 'clip.mp4')], capture_output=True, text=True).stdout.strip())
        try:
            old['zoom'] = zoom_strips(out, dur, old.get('character') or a.character)
        except Exception as e:  # noqa: BLE001
            old['zoom'] = {'mode': 'wide', 'why': 'zoom failed: ' + str(e)[:200]}
        rj.write_text(json.dumps(old, indent=1))
        print('zoom', out.name, old['zoom'].get('mode'), old['zoom'].get('why') or old['zoom'].get('tiles'))
        return
    if rj.exists() and not a.force:
        old = json.loads(rj.read_text())
        if old.get('ok') and (not a.face or old.get('face')) and old.get('rig_sha') == sha(out / 'rig.jsonl'):
            print('cached', out); return
    lines = (out / 'rig.jsonl').read_text().splitlines()
    head = json.loads(lines[0]); rows = [json.loads(x) for x in lines[1:] if x.strip()]
    if not rows:
        sys.exit('empty rig')
    names = head['rig_params']
    t1 = time.time(); log = []; res = {'ok': False, 'rig_sha': sha(out / 'rig.jsonl'), 'fps': FPS, 'character': a.character, 'root': str(ROOT)}
    try:
        hq, dur, tmp = render(rows, head, BODY[0], BODY[1], out / 'clip.mp4', a.threads, log, a.character)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{PAD}', '-i', str(hq), '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '24',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'clip.mp4')], check=True)
        shutil.rmtree(tmp)
        files, times, sfps = strips(out / 'clip.mp4', dur, TILE_BODY, 'strip', out, a.title)
        res.update({'clip': 'clip.mp4', 'clip_sha': sha(out / 'clip.mp4'), 'duration_s': round(dur, 3), 'size': list(BODY), 'strips': files,
                    'strip_shas': [sha(out / f) for f in files], 'strip_fps': sfps, 'strip_times': times, 'render_s': round(time.time() - t1, 1)})
        if a.face:
            t2 = time.time()
            ix = {n: i for i, n in enumerate(names)}
            frows = []
            for r in rows:
                v = list(r['a'])
                if 'cam_zoom' in ix:
                    v[ix['cam_zoom']] = 0.0
                frows.append({'t': r['t'], 'a': v})
            hq, dur, tmp = render(frows, head, FACE[0], FACE[1], out / 'face.mp4', a.threads, log, a.character)
            x, y, w, h = FACE_CROP if a.character == 'muse' else FACE_CROP_SEATED
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', f'{PAD}', '-i', str(hq), '-an', '-vf', f'crop={w}:{h}:{x}:{y},scale=480:480', '-c:v', 'libx264',
                            '-preset', 'medium', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'face.mp4')], check=True)
            shutil.rmtree(tmp)
            ff, ft, _ = strips(out / 'face.mp4', dur, TILE_FACE, 'face_strip', out, a.title)
            res.update({'face': 'face.mp4', 'face_sha': sha(out / 'face.mp4'), 'face_strips': ff, 'face_strip_shas': [sha(out / f) for f in ff],
                        'face_render_s': round(time.time() - t2, 1)})
        try:
            res['zoom'] = zoom_strips(out, dur, a.character)
        except Exception as e:  # noqa: BLE001
            res['zoom'] = {'mode': 'wide', 'why': 'zoom failed: ' + str(e)[:200]}
        res['ok'] = True
    except Exception as e:  # noqa: BLE001
        res['error'] = str(e)[:600]
    (out / 'render.log').write_text('\n'.join(log))
    rj.write_text(json.dumps(res, indent=1))
    print(('rendered ' if res['ok'] else 'FAILED ') + str(out), f'{time.time() - t1:.0f}s', res.get('error', ''))
    sys.exit(0 if res['ok'] else 1)


if __name__ == '__main__':
    main()
