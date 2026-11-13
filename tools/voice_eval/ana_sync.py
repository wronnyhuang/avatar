"""offset = picture flash onset - heard beep onset (ms), per beep. python ana_sync.py rec.mkv"""
import subprocess, sys, json, os, numpy as np
f = sys.argv[1]
a = np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-i', f, '-map', '0:a:0', '-ac', '1', '-ar', '48000', '-f', 's16le', '-'],
                                 capture_output=True).stdout, '<i2').astype(np.float32) / 32768
W, H = 100, 125
v = np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-i', f, '-map', '0:v:0', '-vf', f'scale={W}:{H},format=gray', '-f', 'rawvideo', '-'],
                                 capture_output=True).stdout, np.uint8).reshape(-1, H, W).astype(np.float32)
pts = [float(x.strip(',')) for x in subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'frame=pts_time', '-of', 'csv=p=0', f],
                                        capture_output=True, text=True).stdout.split()]
astart = float(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=start_time', '-of', 'csv=p=0', f], capture_output=True, text=True).stdout.strip() or 0)
pts = np.array(pts[:len(v)])
# the flashing cell: the 10x10 block with the largest temporal std
blk = v.reshape(len(v), H // 5, 5, W // 10, 10).mean((2, 4))
sd = blk.std(0); sd[:4, :] = 0; sd[13:, :] = 0   # stage rows only (y 127..542 of 1000 -> blocks 3..13)
i, j = np.unravel_index(np.argmax(sd), sd.shape)
lum = blk[:, i, j]; thr = (lum.min() + lum.max()) / 2
von = [pts[k] for k in range(1, len(lum)) if lum[k] >= thr > lum[k - 1]]
# audio onsets: 5 ms envelope above 0.05 after >= 300 ms below
hop = 240; env = np.sqrt(np.convolve(a * a, np.ones(hop) / hop, 'same'))[::hop]; ta = np.arange(len(env)) * hop / 48000 + astart
aon = []; quiet = 999; TH = float(os.environ.get('ATH', '0.05')); SKEW = float(os.environ.get('SKEW', '0'))
for k, e in enumerate(env):
    if e > TH:
        if quiet >= 60: aon.append(ta[k])
        quiet = 0
    else: quiet += 1
offs = []
for t in aon:
    c = [x - SKEW for x in von if -0.6 < x - SKEW - t < 0.6]
    if c: offs.append(round((min(c, key=lambda x: abs(x - t)) - t) * 1000))
print('aon', [round(x,3) for x in aon]); print('von', [round(x,3) for x in von])
print(json.dumps({'beeps': len(aon), 'flashes': len(von), 'matched': len(offs), 'offset_ms': offs,
                  'median': float(np.median(offs)) if offs else None, 'cell': [int(i), int(j)]}))
