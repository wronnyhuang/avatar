import json, math, itertools, os, sys
D=os.path.join(os.environ.get('AVATAR_DATA', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')), 'perception', '')
clips={'sam_story':1,'sam_question':1,'sam_excited':1,'emotion_happy':1,'vcd_face_motion':1,'watch_object_silent':0}
data={}
for c in clips:
    F=[json.loads(l) for l in open(D+c+'.jsonl') if l.startswith('{"t"')]
    data[c]=[(f['t'], f['face']) if f.get('face',{}).get('present') else (f['t'],None) for f in F]
def sim(frames, cx, cy, sx, sy, gw, fast):
    bx=by=0; seen=None; lastT=None; hi=lo=None; att=False; n=a=0
    for t,fc in frames:
        if fc is None: seen=None; continue
        if seen is None: seen=t
        lx = fc['yaw'] + gw*fc['gazeX']; ly = -fc['pitch'] + gw*fc['gazeY']
        ex=-fc['ax']; ey=-fc['ay']-0.07
        dx0=lx-ex; dy0=ly-ey
        dt=0 if lastT is None else min(0.2,t-lastT); lastT=t
        if math.hypot(dx0-bx,dy0-by)<0.32 and dt>0:
            tau=fast if t-seen<3 else 10; k=1-math.exp(-dt/tau)
            bx=max(-cx,min(cx,bx+(dx0-bx)*k)); by=max(-cy,min(cy,by+(dy0-by)*k))
        A=math.exp(-((dx0-bx)/sx)**2-((dy0-by)/sy)**2)
        if A>0.55: lo=None; hi=hi if hi is not None else t
        else: hi=None
        if A<0.3: lo=lo if lo is not None else t
        elif A>0.3: lo=None
        if not att and hi is not None and t-hi>=0.25: att=True
        elif att and lo is not None and t-lo>=0.4: att=False
        n+=1; a+=att
    return a/max(1,n)
best=[]
for cx,cy,sx,sy,gw,fast in itertools.product([0.08,0.12,0.16,0.2],[0.15,0.2,0.25],[0.18,0.2,0.24],[0.24,0.28],[0.5,0.75,1.0],[1.5,3]):
    r={c:sim(data[c],cx,cy,sx,sy,gw,fast) for c in clips}
    pos=min(r[c] for c in clips if clips[c]); neg=r['watch_object_silent']
    best.append((pos-neg, pos, neg, (cx,cy,sx,sy,gw,fast), r))
best.sort(key=lambda x:-x[0])
for b in best[:8]: print(round(b[0],2), round(b[1],2), round(b[2],2), b[3], {k:round(v,2) for k,v in b[4].items()})
print('current', {c:round(sim(data[c],0.3,0.3,0.26,0.28,1.0,1.5),2) for c in clips})
