import sys, time, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.environ.get('CANON1_DIR', 'models/canon1'))   # the folder with canon1.py (the reference canon feature extractor)
from anim.motion2.audiofeat import StreamFeat
from canon1 import causal_audio as A
rng = np.random.default_rng(0)
n = 24000*6
t = np.arange(n)/24000
x = (0.3*np.sin(2*np.pi*220*t)*(np.sin(2*np.pi*0.7*t)>0) + 0.01*rng.standard_normal(n))
pcm = (x*32767).astype(np.int16)
af = StreamFeat(); rows=[]; t0=time.perf_counter()
for i in range(0, n, 480):
    r = af.push(i, pcm[i:i+480]); rows.append(r)
dt=time.perf_counter()-t0
S = np.concatenate(rows)
x16 = A.resample_to_16k(pcm.astype(np.float64)/32768, 24000)
O = A.audio_features(x16, n_frames=S.shape[0])
print(S.shape, O.shape, 'ms/frame', 1000*dt/S.shape[0])
d = np.abs(S-O); print('mel maxdiff', d[:, :40].max(), 'median', np.median(d[:, :40]), 'loud', d[:,40].max(), 'vad mism', (S[:,41]!=O[:,41]).sum())
k = np.argsort(-d[:,40])[:5]; print(k, S[k,40], O[k,40])
print('excluding first 2 frames: mel', d[2:, :40].max(), 'loud', d[2:,40].max())
