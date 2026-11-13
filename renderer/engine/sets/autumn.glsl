// ---------------------------------------------------------------------------
//  engine/sets/autumn.glsl — forest.glsl plus a near-focus autumn dressing around the subject.
//  Bird's clearing, rescaled and recoloured; every look lives in `set_params` of the animal
//  JSON, which arrives as `#define SP_<NAME> ...` (see engine/render.py) and overrides the defaults.
//
//  SET API (every file in engine/sets must provide these):
//    void  setInit();                         per-fragment setup
//    vec2  setMap(vec3 p, bool shadow);       distance, material
//    vec3  setSky(vec3 rd);                   sky radiance
//    void  setSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s);
//    vec3  setAmbient(vec3 p, vec3 n);        sky + bounce + fill irradiance
//    float setCookie(vec3 p);                 0..1 dappled light at a world point — TIME VARYING, CHEAP
//    float setVol(vec3 p);                    0..1 light reaching the air (god-ray shafts) — CHEAP
//    float setFog(vec3 p);                    fog density — CHEAP
// ---------------------------------------------------------------------------
#ifndef SP_SCALE
#define SP_SCALE 5.0                                // world units per bird-scene unit
#endif
#ifndef SP_GROUND_Y
#define SP_GROUND_Y (-2.7)
#endif
#ifndef SP_CANOPY_H
#define SP_CANOPY_H 2.6                             // canopy light plane (set units)
#endif
#ifndef SP_CANOPY_A
#define SP_CANOPY_A vec3(0.44,0.13,0.03)            // shadow side of the leaves
#endif
#ifndef SP_CANOPY_B
#define SP_CANOPY_B vec3(0.95,0.52,0.08)            // lit side of the leaves
#endif
#ifndef SP_GROUND_A
#define SP_GROUND_A vec3(0.14,0.065,0.025)            // ground base
#endif
#ifndef SP_GROUND_B
#define SP_GROUND_B vec3(0.40,0.145,0.03)            // ground highlight (leaf litter)
#endif
#ifndef SP_GROUND_C
#define SP_GROUND_C vec3(0.09,0.12,0.03)            // moss / grass patches
#endif
#ifndef SP_LEAF_A
#define SP_LEAF_A vec3(0.50,0.13,0.025)              // fallen-leaf colour 1
#endif
#ifndef SP_LEAF_B
#define SP_LEAF_B vec3(0.60,0.27,0.04)              // fallen-leaf colour 2
#endif
#ifndef SP_GRASS_A
#define SP_GRASS_A vec3(0.20,0.11,0.03)
#endif
#ifndef SP_GRASS_B
#define SP_GRASS_B vec3(0.46,0.24,0.06)
#endif
#ifndef SP_BARK_A
#define SP_BARK_A vec3(0.08,0.05,0.035)
#endif
#ifndef SP_BARK_B
#define SP_BARK_B vec3(0.22,0.145,0.09)
#endif
#ifndef SP_MOSS
#define SP_MOSS vec3(0.12,0.13,0.035)
#endif
#ifndef SP_SKY_LOW
#define SP_SKY_LOW vec3(1.00,0.76,0.44)*1.05         // horizon glow
#endif
#ifndef SP_SKY_HIGH
#define SP_SKY_HIGH vec3(0.95,0.82,0.62)*0.9
#endif
#ifndef SP_FOG_LO
#define SP_FOG_LO vec3(0.46,0.25,0.10)
#endif
#ifndef SP_FOG_HI
#define SP_FOG_HI vec3(1.00,0.68,0.32)
#endif
#ifndef SP_FOG_GAIN
#define SP_FOG_GAIN 1.0
#endif
#ifndef SP_AMB_SKY
#define SP_AMB_SKY vec3(0.52,0.54,0.58)              // open-sky ambient
#endif
#ifndef SP_BOUNCE
#define SP_BOUNCE vec3(0.46,0.30,0.12)              // light bounced off the ground
#endif
#ifndef SP_TREES
#define SP_TREES 1.0                                // 0 = no explicit big trunks
#endif
#ifndef SP_PINE
#define SP_PINE 0.0                                 // 1 = slim straight trunks
#endif
#ifndef SP_GRASS_AMT
#define SP_GRASS_AMT 0.35
#endif
#ifndef SP_CLEAR_C
#define SP_CLEAR_C vec2(0.6,-2.0)                    // clearing centre (set units)
#endif
#ifndef SP_CLEAR_R
#define SP_CLEAR_R 3.4
#endif
#ifndef SP_BUSHES
#define SP_BUSHES 0.55                              // 0 = no understory
#endif
#ifndef SP_BUSH_C
#define SP_BUSH_C vec3(0.46,0.07,0.03)              // third leaf colour (crimson)
#endif
#ifndef SP_DAPPLE_FREQ
#define SP_DAPPLE_FREQ 1.0                          // scale of the dapples on the character
#endif

const float SC = SP_SCALE;
const float G0 = SP_GROUND_Y;
const vec2  CLR = SP_CLEAR_C;
const float CLR_R = SP_CLEAR_R;
const float CANH = SP_CANOPY_H;

#define M_GROUND 1.0
#define M_BARK   2.0
#define M_CANOPY 9.0
#define M_LEAF   12.0
#define M_GRASS  17.0
#define M_STEM   11.0
#define M_BUSH   13.0

#define NT 11
const vec3 TRK[NT] = vec3[NT](
  vec3(-2.05,-7.20,0.30),   // frame left of the head
  vec3( 2.40,-8.40,0.28),   // frame right
  vec3(-4.90,-11.0,0.55),
  vec3( 5.30,-12.0,0.50),
  vec3( 0.55,-14.0,0.30),
  vec3( 3.60,-18.0,0.40),
  vec3(-3.00,-17.0,0.42),
  vec3(-8.40,-8.10,0.60),
  vec3( 8.60,-9.00,0.62),
  vec3(-10.5,-15.5,0.70),
  vec3(11.00,-16.0,0.70)
);

vec2  HOLE;
vec2  gWind;
float gGrassH;

void setInit(){
  HOLE = uSunDir.xz*((CANH - 0.0)/uSunDir.y);
  float T = uTime;
  gWind = 0.10*vec2(sin(T*0.80), cos(T*0.65)) + 0.035*vec2(sin(T*2.1+1.0), sin(T*1.7));
}

float groundHs(vec2 xz){       // set-space height above G0
  float h = 0.09*sin(xz.x*0.45+0.3)*sin(xz.y*0.35+1.1);
  h += 0.12*(vnoise2(xz*0.55)-0.5);
  return h;
}
float canopyPresence(vec2 c){
  float dc = length((c - CLR)*vec2(0.80,1.0));
  float pr = smoothstep(CLR_R, CLR_R+3.5, dc);
  pr *= smoothstep(0.30, 1.05, length(c - HOLE));
  pr *= smoothstep(-40.0, -36.0, c.y);
  return pr;
}
float sdTree(vec3 p, vec3 tr, float seed, float best){
  vec3 q = p - vec3(tr.x, 0.0, tr.y);
  float r0 = tr.z*mix(1.0, 0.55, SP_PINE);
  float l0 = length(q.xz);
  float bound = r0*(1.22 + 1.25*exp(-max(q.y,0.0)/(0.55*r0)));
  if((l0 - bound)*0.7 > best) return 1e9;
  float y = q.y;
  q.x += 0.14*r0*sin(y*0.8 + seed)*(1.0 - SP_PINE*0.7);
  q.z += 0.10*r0*sin(y*0.6 + seed*1.7)*(1.0 - SP_PINE*0.7);
  float a = atan(q.z, q.x);
  float l = length(q.xz);
  float yy = max(y, 0.0);
  float rad = r0*(1.0 + 0.6*exp(-yy/(0.55*r0)));
  rad *= 1.0 + 0.40*exp(-yy/(0.30*r0))*pow(0.5+0.5*cos(a*5.0+seed*3.0), 2.0);
  rad += r0*0.03*sin(a*13.0 + 2.0*sin(y*2.5+seed));
  float d = (l - rad)*0.7;
  return max(d, y - 7.0);
}
float sdForestTrunks(vec3 p){
  if(p.y > 5.1) return p.y - 5.0;
  vec2 g = p.xz/3.0 - 0.5; vec2 c0 = floor(g);
  float d = 1.4;
  for(int j=0;j<2;j++) for(int i=0;i<2;i++){
    vec2 ci = c0 + vec2(float(i), float(j));
    vec3 h = hash33(vec3(ci, 3.0));
    vec2 cpos = (ci + 0.5 + (h.xy-0.5)*0.40)*3.0;
    float pres = canopyPresence(cpos)*smoothstep(5.0, 6.0, length(cpos - vec2(0.0,-1.0)));
    if(h.z > pres*0.85) continue;
    float r = (0.30 + 0.22*h.y)*mix(1.0,0.55,SP_PINE);
    vec3 q = p - vec3(cpos.x, 0.0, cpos.y);
    q.x += 0.15*sin(q.y*0.5 + h.x*6.0);
    float rad = r*(1.0 + 0.6*exp(-max(q.y,0.0)/(0.6*r)));
    float dt = (length(q.xz) - rad)*0.8;
    d = min(d, max(dt, q.y - 5.0));
  }
  return d;
}
vec2 sdCanopy(vec3 p){
  if(p.y < 0.55) return vec2(0.70 - p.y, M_CANOPY);
  if(p.y > 6.6)  return vec2(p.y - 6.5, M_CANOPY);
  vec2 g = p.xz/2.2;
  vec2 ci = floor(g);
  float d = 1e9;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
    vec2 cc = ci + vec2(float(i), float(j));
    vec3 h = hash33(vec3(cc, 7.0));
    vec2 cpos = (cc + 0.2 + 0.6*h.xy)*2.2;
    float pres = canopyPresence(cpos);
    if(h.z > pres*1.25) continue;
    float r = 0.90 + 0.70*h.x;
    vec3 c = vec3(cpos.x, 2.05 + 1.3*h.y, cpos.y);
    float e = sdEllipsoid(p - c, vec3(r, r*0.78, r));
    d = smin(d, e, 0.5);
  }
  if(d < 0.4) d += 0.55*(vnoise3(p*1.5) - 0.5) + 0.22*(vnoise3(p*4.0) - 0.5);    // leafy, lumpy, no clean blobs
  return vec2(d*0.8, M_CANOPY);
}
// grass / bracken tufts: a sparse band around the clearing
vec2 sdGrass(vec3 p, float gh, float best){
  float hy = p.y - gh;
  if(hy > 0.20 || hy - 0.2 > best) return vec2(1e9, M_GRASS);
  const float CS = 0.062;
  vec2 g = p.xz/CS - 0.5; vec2 c0 = floor(g);
  float d = 1e9; float hf = 0.0;
  for(int j=0;j<2;j++) for(int i=0;i<2;i++){
    vec2 ci = c0 + vec2(float(i), float(j));
    vec3 h = hash33(vec3(ci, 11.0));
    vec2 base = (ci + 0.5 + (h.xy - 0.5)*0.7)*CS;
    float pres = SP_GRASS_AMT*(0.30 + 0.70*smoothstep(0.40, 0.68, vnoise2(base*1.6)))*smoothstep(-0.10, -0.70, base.y);      // a blade this close would read as a stick
    if(h.z > pres) continue;
    float H = mix(0.040, 0.085, h.x);
    for(int k=0;k<4;k++){
      float a = h.y*6.283 + float(k)*1.7 + 0.6*float(k*k);
      vec2 lean = vec2(cos(a), sin(a));
      float bh = H*(0.65 + 0.35*fract(h.z*7.0 + float(k)*0.37));
      vec3 b0 = vec3(base.x + lean.x*0.006*float(k), gh - 0.01, base.y + lean.y*0.006*float(k));
      vec3 b1 = b0 + vec3(lean.x*bh*0.18, bh*0.55, lean.y*bh*0.18);
      vec3 b2 = b0 + vec3(lean.x*bh*0.55, bh, lean.y*bh*0.55);
      float db = min(sdTaper(p, b0, b1, 0.0034, 0.0024), sdTaper(p, b1, b2, 0.0024, 0.0004));
      if(db < d){ d = db; hf = clamp((p.y - gh)/bh, 0.0, 1.0); }
    }
  }
  if(d < best) gGrassH = hf;
  return vec2(d*0.9, M_GRASS);
}

// autumn understory: clumpy shrubs between the trunks (bake only, so noise is cheap here)
float sdBushes(vec3 p){
  if(p.y > 1.0) return p.y - 0.8;
  vec2 g = p.xz/0.8;
  vec2 ci = floor(g);
  float d = 1e9;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
    vec2 cc = ci + vec2(float(i), float(j));
    vec3 h = hash33(vec3(cc, 23.0));
    vec2 cpos = (cc + 0.2 + 0.6*h.xy)*0.8;
    float back = smoothstep(-4.5, -7.0, cpos.y);          // only behind the subject
    if(h.z > back*SP_BUSHES) continue;
    float r = 0.16 + 0.24*h.x;
    vec3 c = vec3(cpos.x, r*0.55, cpos.y);
    d = smin(d, sdEllipsoid(p - c, vec3(r*1.2, r*0.9, r*1.1)), 0.14);
  }
  if(d < 0.2) d += 0.11*(vnoise3(p*6.0) - 0.5) + 0.065*(vnoise3(p*15.0) - 0.5) + 0.038*abs(vnoise3(p*36.0) - 0.5) + 0.016*(vnoise3(p*80.0) - 0.5);
  return d*0.7;
}

// ---------------------------------------------------------------------------
//  Near-focus dressing, in WORLD units, all of it behind the subject and at the frame edge:
//  a mossy fallen log across the bottom, toadstools on it, bracken rising behind it, a leafy branch in
//  each top corner.  `if` does not save time on llvmpipe, so every part sits in a one-shot loop guarded by
//  a bound: SIMD groups far from that part skip it, and the far clearing bakes at the old price.
// ---------------------------------------------------------------------------
#define M_LOG   20.0
#define M_MOSS  21.0
#define M_CAP   22.0
#define M_STIPE 23.0
#define M_FERN  24.0
#define M_TWIG  25.0
#define M_BLEAF 26.0

const vec3  LOG_A = vec3(-3.70,-1.13,-1.30), LOG_B = vec3(3.70,-1.56,-0.85);
const float LOG_R = 0.46;
#define NMU 6
const vec4 MU_P[NMU] = vec4[NMU](       // base xyz, stem height
  vec4( 1.63,-0.92,-0.86,0.34), vec4( 1.86,-0.93,-0.82,0.22), vec4( 1.44,-0.91,-0.85,0.13),
  vec4(-1.72,-0.80,-1.06,0.19), vec4(-1.54,-0.80,-1.04,0.12), vec4(-1.90,-0.79,-1.05,0.10));
const float MU_R[NMU] = float[NMU](0.19, 0.125, 0.075, 0.10, 0.07, 0.055);
const vec3 FERN_L = vec3(-2.10,-1.18,-1.86), FERN_R = vec3(2.22,-1.28,-1.62);
#define NFR 7
#define NTW 3
const vec3 TWR[NTW+1] = vec3[NTW+1](vec3(2.75,1.62,-0.92), vec3(2.05,1.27,-0.86), vec3(1.55,1.10,-0.80), vec3(1.16,0.80,-0.74));
const vec3 TWL[NTW+1] = vec3[NTW+1](vec3(-2.85,1.66,-0.98), vec3(-2.20,1.32,-0.92), vec3(-1.72,1.17,-0.86), vec3(-1.40,1.02,-0.80));
#define NLF 11

vec3  gNearQ;          // local coordinate for the painter (mushroom: cap space / R; leaf: x/W, t, 0)
float gNearId;         // hash of the piece
vec3  gTmpQ; float gTmpId;
vec2 opUn(vec2 a, vec2 b){ if(b.x < a.x){ gNearQ = gTmpQ; gNearId = gTmpId; return b; } return a; }

vec2 logAndMoss(vec3 p){
  vec3 ba = LOG_B - LOG_A, pa = p - LOG_A;
  float h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
  vec3 c = pa - ba*h;
  float a = atan(c.z, c.y), r = length(c), u = h*7.4;
  float rad = LOG_R*(1.0 + 0.05*sin(u*1.3 + 1.0));
  rad += 0.012*sin(a*19.0 + 1.8*sin(u*1.4)) + 0.040*(vnoise2(vec2(a*5.0, u*2.2)) - 0.5) + 0.012*(vnoise2(vec2(a*24.0, u*3.0)) - 0.5);
  float d = (r - rad)*0.75;
  // moss sits on top, in thick soft cushions
  float up = c.y/max(r, 1e-3);
  float mz = smoothstep(0.50, 0.78, up + 0.95*(fbm3(p*vec3(2.1,3.6,2.1)) - 0.5));
  float dm = d - mz*(0.018 + 0.095*vnoise3(p*vec3(3.1,5.0,3.1)) + 0.030*vnoise3(p*11.0) + 0.010*vnoise3(p*38.0));
  return vec2(min(d, dm)*0.9, dm < d - 0.002 ? M_MOSS : M_LOG);
}
vec2 toadstools(vec3 p){
  vec2 res = vec2(1e9, M_STIPE);
  for(int i=0;i<NMU;i++){
    float H = MU_P[i].w, Rc = MU_R[i], fi = float(i);
    vec3 q = p - MU_P[i].xyz;
    q.xz += vec2(sin(fi*2.4), cos(fi*1.7))*0.28*q.y;         // each one leans its own way
    float st = sdTaper(q, vec3(0.0,-0.04,0.0), vec3(0.0,H,0.0), Rc*0.36, Rc*0.21);
    vec3 cq = q - vec3(0.0, H, 0.0);
    float tilt = 0.20*sin(fi*3.1);
    cq.xy = rot(tilt)*cq.xy;
    float cap = sdEllipsoid(cq, vec3(Rc, Rc*0.66, Rc));
    cap = smax(cap, -cq.y - Rc*0.10 + 0.34*length(cq.xz)*0.30, Rc*0.25);    // a rim that curls under
    if(st < res.x){ res = vec2(st, M_STIPE); gTmpQ = q/Rc; gTmpId = fi; }
    if(cap < res.x){ res = vec2(cap, M_CAP); gTmpQ = cq/Rc; gTmpId = fi; }
  }
  res.x *= 0.85;
  return res;
}
// a bracken frond: an arc of radius Ra in the plane (cos az, 0, -sin az) x (0,1,0), length L, half width W
float sdFrond(vec3 p, vec3 o, float az, float L, float Ra, float W){
  vec3 q = p - o;
  q.xz = rot(az)*q.xz;
  vec2 v = vec2(q.x - Ra, q.y);
  float th = atan(q.y, Ra - q.x);
  float s = clamp(th, 0.0, L/Ra)*Ra;
  float u = length(v) - Ra;
  // the blade twists about the rachis to face the camera: an arc seen side on is otherwise a thin line
  vec2 uz = rot(1.12)*vec2(u, q.z);
  u = uz.x; float z = uz.y;
  float tt = s/L;
  u -= 0.55*z*z;                                    // the blade droops off the rachis
  float rach = length(vec2(u, z)) - mix(0.0065, 0.0015, tt);
  float ds = th*Ra - s;
  rach = length(vec2(max(rach, 0.0), ds));
  // pinnae: slanted strips, stepped in width, each one a tapered leaflet
  const float SP = 0.047;
  float x = abs(z);
  float n = (s - x*0.55)/SP;
  float id = floor(n), f = fract(n) - 0.5;
  float tc = clamp((id + 0.5)*SP/L, 0.0, 1.0);
  float w = W*smoothstep(0.0, 0.16, tc)*pow(1.0 - tc, 0.75);
  float rel = clamp(x/max(w, 1e-3), 0.0, 1.0);
  float pw = 0.40*SP*sqrt(max(1.0 - rel*rel, 0.0))*(1.0 + 0.16*sin(x*150.0));
  float d2 = max(abs(f)*SP - pw, x - w);
  float d = max(d2*0.75, abs(u) - 0.0030);
  d = max(d, max(-s, s - L));
  return min(d*0.6, rach);
}
vec2 bracken(vec3 p, vec3 c, float sd, float dirx){
  float d = 1e9;
  for(int i=0;i<NFR;i++){
    vec3 h = hash31(float(i)*3.17 + sd);
    float az = ((i & 3) == 3 ? 3.14159 - dirx : dirx) + (h.x - 0.5)*1.1;     // most arch into the frame
    vec3 o = c + vec3(h.y - 0.5, 0.0, h.z - 0.5)*0.40 + vec3(0.0, 0.0, -0.05*float(i));
    float df = sdFrond(p, o, az, mix(1.05, 1.50, h.x), mix(0.62, 1.15, h.y), mix(0.15, 0.22, h.z));
    if(df < d){ d = df; gTmpId = h.y; gTmpQ = p - o; }
  }
  return vec2(d, M_FERN);
}
// a hanging twig and its leaves
vec2 leafySprig(vec3 p, vec3 a, vec3 b, vec3 c, vec3 e, float sd){
  float tw = sdTaper(p, a, b, 0.024, 0.016);
  tw = min(tw, sdTaper(p, b, c, 0.016, 0.010));
  tw = min(tw, sdTaper(p, c, e, 0.010, 0.0045));
  vec2 res = vec2(tw, M_TWIG);
  for(int i=0;i<NLF;i++){
    float fi = float(i);
    vec3 h = hash31(fi*1.93 + sd), h2 = hash31(fi*4.37 + sd + 3.0);
    float t = (fi + 0.5*h.x)/float(NLF);
    t = 0.22 + 0.80*t;
    vec3 o = t < 0.5 ? mix(b, c, t*2.0) : mix(c, e, min(t*2.0 - 1.0, 1.0));
    // a short stalk, then the blade hangs and fans out on alternate sides
    float sgn = (i & 1) == 0 ? 1.0 : -1.0;
    vec3 dir = normalize(vec3(sgn*(0.25 + 0.75*h2.x), -0.65 - 0.9*h2.y, 0.35*(h.y - 0.3)));
    vec3 nr = normalize(vec3(0.5*(h2.z - 0.5), 0.45*(h.z - 0.3) + 0.25, 1.0));
    vec3 xa = normalize(cross(dir, nr)), za = cross(xa, dir);
    vec3 v = p - o;
    vec3 lq = vec3(dot(v, xa), dot(v, dir), dot(v, za));
    float S = mix(0.23, 0.34, h.z), W = S*mix(0.27, 0.34, h.x);
    float stalk = sdTaper(lq, vec3(0.0), vec3(0.0,0.05,0.0), 0.0045, 0.0035);
    if(stalk < res.x) res = vec2(stalk, M_TWIG);
    lq.y -= 0.045;
    lq.z += 0.55*lq.x*lq.x/W - 0.5*sin(lq.y/S*3.14159)*0.04;     // cupped, and curled along its length
    float tl = clamp(lq.y/S, 0.0, 1.0);
    float wd = W*pow(sin(3.14159*pow(tl, 0.72)), 0.85)*(1.0 + 0.05*sin(tl*44.0));
    float d = max(abs(lq.x) - wd, max(-lq.y, lq.y - S));
    d = max(d*0.72, abs(lq.z) - 0.0032)*0.8;
    if(d < res.x){ res = vec2(d, M_BLEAF); gTmpQ = vec3(lq.x/max(wd, 1e-3), tl, 0.0); gTmpId = h2.x; }
  }
  return res;
}

// every guard: a lower bound outside the part, so the march walks up to it without paying for it
vec2 nearMap(vec3 p, bool shadow){
  const float G = 0.16;
  float bL = sdCapsule(p, LOG_A, LOG_B, LOG_R + 0.10);
  vec2 res = vec2(max(bL, G), M_LOG);
  for(int i=0;i<1 && bL < G;i++) res = opUn(res, logAndMoss(p));

  float bM = min(length(p - vec3(1.63,-0.72,-0.85)) - 0.52, length(p - vec3(-1.72,-0.70,-1.05)) - 0.38);
  res = opUn(res, vec2(max(bM, G), M_LOG));
  for(int i=0;i<1 && bM < G;i++) res = opUn(res, toadstools(p));

  float bFl = length(p - FERN_L - vec3(0.0,0.85,0.0)) - 1.60, bFr = length(p - FERN_R - vec3(0.0,0.85,0.0)) - 1.60;
  res = opUn(res, vec2(max(min(bFl, bFr), G), M_LOG));
  for(int i=0;i<1 && bFl < G;i++) res = opUn(res, bracken(p, FERN_L, 1.3, 0.0));
  for(int i=0;i<1 && bFr < G;i++) res = opUn(res, bracken(p, FERN_R, 4.1, 3.14159));

  float bTr = sdCapsule(p, TWR[1], TWR[3], 0.62), bTl = sdCapsule(p, TWL[1], TWL[3], 0.62);
  res = opUn(res, vec2(max(min(bTr, bTl), G), M_LOG));
  for(int i=0;i<1 && bTr < G;i++) res = opUn(res, leafySprig(p, TWR[0], TWR[1], TWR[2], TWR[3], 2.2));
  for(int i=0;i<1 && bTl < G;i++) res = opUn(res, leafySprig(p, TWL[0], TWL[1], TWL[2], TWL[3], 7.7));
  return res;
}

// the log lies on a bank of the forest floor: the ground rises to meet it and falls away slowly behind, under the ferns.  World units
float logBank(vec2 xz){
  vec3 c = mix(LOG_A, LOG_B, clamp((xz.x - LOG_A.x)/(LOG_B.x - LOG_A.x), -0.6, 1.6));
  float dz = xz.y - c.z;
  return (c.y - LOG_R - G0 + 0.16)*exp(-dz*dz/(dz > 0.0 ? 1.5 : 9.0));
}

vec2 setMap(vec3 pw, bool shadow){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z)/SC;
  float gh = groundHs(p.xz) + logBank(pw.xz)/SC;
  float dg = (p.y - gh)*0.62;
  vec2 res = vec2(dg, M_GROUND);
  float dtr = 1e9;
  if(SP_TREES > 0.5){
    for(int i=0;i<NT;i++) dtr = min(dtr, sdTree(p, TRK[i], float(i)*1.37 + 0.5, res.x + 0.1));
  }
  if(dtr < res.x + 0.1){
    float bl = smin(res.x, dtr, 0.12);
    res = vec2(bl, dtr < res.x ? M_BARK : res.y);
  }
  if(!shadow || p.y - gh < 0.2) res = opU(res, sdGrass(p, gh, res.x));
  if(SP_BUSHES > 0.0) res = opU(res, vec2(sdBushes(p), M_BUSH));
  if(!shadow){
    res = opU(res, vec2(sdForestTrunks(p), M_BARK));
    res = opU(res, sdCanopy(p));
  }
  res.x *= SC;
  return opU(res, nearMap(pw, shadow));
}

vec3 setSky(vec3 rd){
  float sd = max(dot(rd, uSunDir), 0.0);
  vec3 col = mix(SP_SKY_LOW, SP_SKY_HIGH, smoothstep(0.02, 0.60, rd.y));
  col += vec3(1.0,0.80,0.5)*(1.2*pow(sd,5.0) + 8.0*pow(sd,160.0));
  return col;
}

void setSurface(vec3 pw, vec3 rd, float t, float m, vec3 n, inout Surf s){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z)/SC;
  vec2 xz = p.xz;
  if(m == M_GROUND){
    float n1 = fbm2(xz*1.3);
    float n2 = vnoise2(xz*9.0);
    vec3 a = mix(SP_GROUND_A, SP_GROUND_B, smoothstep(0.3,0.75,n1));
    a = mix(a, SP_GROUND_C, 0.75*smoothstep(0.52,0.72,fbm2(xz*0.6+9.0)));
    a *= 0.80 + 0.35*n2;
    // leaf litter, three scales
    for(int i=0;i<3;i++){
      float fi = float(i);
      float sc = 5.0 + fi*4.0;
      vec2 u = xz*sc + fi*13.1;
      vec2 lc = floor(u); vec2 lh = hash22(lc + fi);
      vec2 lf = fract(u) - 0.25 - 0.5*lh;
      float leaf = smoothstep(0.17, 0.12, length(rot(lh.x*6.0)*lf*vec2(1.0,1.9)));
      a = mix(a, mix(SP_LEAF_A, SP_LEAF_B, lh.y), leaf*step(0.42, lh.x));
    }
    // the floor close to the camera: leaf-sized litter in world units, over dark soil, which a wide frame looks at
    float nearF = smoothstep(26.0, 11.0, t);
    vec2 tilt = vec2(0.0);
    a = mix(a, a*(0.45 + 0.75*vnoise2(pw.xz*6.0)), nearF);
    for(int i=0;i<4;i++){
      float fi = float(i);
      float sc = 3.2 + fi*2.3;
      vec2 u = pw.xz*sc + fi*7.7;
      vec2 lc = floor(u); vec2 lh = hash22(lc + fi + 20.0), lk = hash22(lc + fi + 41.0);
      vec2 lf = rot(lh.x*6.283)*(fract(u) - 0.25 - 0.5*lh);
      float d = length(lf*vec2(1.0, 2.2)) + 0.09*abs(lf.y)*sin(lf.y*40.0);
      float leaf = smoothstep(0.30, 0.25, d)*step(0.22, lk.x)*nearF;
      vec3 lcol = mix(SP_LEAF_A, SP_LEAF_B, lk.y)*mix(0.55, 1.15, lh.y);
      lcol = mix(lcol, vec3(0.44,0.27,0.06), smoothstep(0.82, 0.98, lk.x));
      lcol *= 1.0 - 0.28*smoothstep(0.018, 0.0, abs(lf.x));                   // the midrib
      a = mix(a, lcol, leaf);
      tilt = mix(tilt, (lk - 0.5)*0.9, leaf);
    }
    s.alb = a;
    vec2 gn = vec2(vnoise2(xz*55.0), vnoise2(xz*55.0+31.0)) - 0.5;
    s.n = normalize(n + vec3(gn.x + tilt.x, 0.0, gn.y + tilt.y)*0.55*smoothstep(40.0, 5.0, t));
    s.occ = 0.62 + 0.38*vnoise2(xz*70.0);
    s.wrap = 0.2; s.sheen = 0.25;
  } else if(m == M_BARK){
    float bn = fbm3(vec3(p.x*6.0, p.y*1.6, p.z*6.0));
    vec3 a = mix(SP_BARK_A, SP_BARK_B, bn);
    a = mix(a, a*0.6, smoothstep(0.55,0.75, vnoise3(vec3(p.x*14.0,p.y*3.0,p.z*14.0))));
    float mb = smoothstep(0.9, 0.1, p.y + 0.4*fbm3(p*3.0));
    s.alb = mix(a, SP_MOSS, mb*0.6);
    s.wrap = 0.15; s.spec = 0.03; s.shin = 8.0;
  } else if(m == M_GRASS){
    vec2 cid = floor(xz/0.085);
    float v = hash12(cid);
    s.alb = mix(SP_GRASS_A*0.6, mix(SP_GRASS_A, SP_GRASS_B, v), smoothstep(0.0, 0.9, gGrassH));
    s.wrap = 0.6; s.trans = 0.8; s.transCol = s.alb; s.spec = 0.15; s.shin = 30.0; s.occ = 0.70 + 0.30*gGrassH;
  } else if(m == M_BUSH){
    float n1 = vnoise3(p*7.0);
    float n2 = vnoise3(p*19.0);
    float n3 = vnoise3(p*2.1 + 4.0);
    s.n = normalize(n + 0.7*(vec3(vnoise3(p*9.0), vnoise3(p*9.0+5.0), vnoise3(p*9.0+9.0)) - 0.5));
    float n4 = vnoise3(p*42.0), n5 = vnoise3(p*88.0);             // leaf-sized light and dark
    s.alb = mix(mix(SP_CANOPY_A, SP_CANOPY_B*0.72, n1), SP_BUSH_C, smoothstep(0.36,0.58,n3))*(0.40 + 0.55*n2 + 0.45*n4);
    s.alb = mix(s.alb, mix(SP_BUSH_C, SP_CANOPY_B, n3)*1.15, smoothstep(0.62,0.74,n5)*0.8);
    s.occ = 0.28 + 0.42*n2 + 0.38*smoothstep(0.30,0.72,n4);
    s.wrap = 0.5; s.trans = 0.7; s.transCol = s.alb; s.spec = 0.08; s.shin = 16.0;
  } else if(m == M_CANOPY){
    float n1 = vnoise3(p*5.0);
    float n2 = vnoise3(p*13.0);
    s.n = normalize(n + 0.55*(vec3(vnoise3(p*4.0), vnoise3(p*4.0+5.0), vnoise3(p*4.0+9.0)) - 0.5));
    float n3 = vnoise3(p*0.9 + 2.0);
    s.alb = mix(mix(SP_CANOPY_A, SP_CANOPY_B, n1), SP_BUSH_C, 0.7*smoothstep(0.50,0.68,n3))*(0.55 + 0.55*n2);
    s.occ = 0.45 + 0.55*n2;
    s.wrap = 0.5; s.trans = 0.5; s.transCol = s.alb; s.spec = 0.1; s.shin = 20.0;
  } else {
    // near dressing: re-run the map at the hit so the painter has its local coordinates
    nearMap(pw, false);
    vec3 q = gNearQ;
    if(m == M_LOG){
      vec3 ba = LOG_B - LOG_A, pa = pw - LOG_A;
      float h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
      vec3 c = pa - ba*h;
      float a = atan(c.z, c.y), u = h*7.4;
      float gr = vnoise2(vec2(a*16.0, u*3.0)), gr2 = vnoise2(vec2(a*42.0, u*5.0));
      vec3 bk = mix(vec3(0.075,0.045,0.028), vec3(0.27,0.17,0.10), smoothstep(0.25,0.8,gr*0.7 + gr2*0.3));
      s.alb = bk*(0.8 + 0.4*vnoise3(pw*9.0));
      s.occ = 0.45 + 0.55*smoothstep(0.2,0.6,gr);
      s.wrap = 0.25; s.spec = 0.04; s.shin = 12.0; s.sheen = 0.2;
    } else if(m == M_MOSS){
      float f1 = fbm3(pw*26.0), f2 = fbm3(pw*74.0), f3 = fbm3(pw*2.7);
      vec3 mo = mix(vec3(0.095,0.115,0.030), vec3(0.25,0.285,0.068), smoothstep(0.32,0.68,f1*0.55 + f2*0.45));
      s.alb = mix(mo, mo*vec3(1.30,0.95,0.60), smoothstep(0.44,0.64,f3));
      vec3 lba = LOG_B - LOG_A, lpa = pw - LOG_A;
      float lh_ = clamp(dot(lpa, lba)/dot(lba, lba), 0.0, 1.0);
      vec3 lc_ = lpa - lba*lh_;
      vec2 lm = vec2(lh_*7.4, atan(lc_.z, lc_.y)*0.46);
      // fallen leaves caught in the moss
      for(int i=0;i<2;i++){
        float sc = 3.4 + float(i)*2.2;
        vec2 u = lm*sc + float(i)*7.3;
        vec2 lc = floor(u), lh = hash22(lc + 3.0 + float(i));
        vec2 lf = rot(lh.x*6.28)*(fract(u) - 0.25 - 0.5*lh);
        float leaf = smoothstep(0.17, 0.14, length(lf*vec2(1.0, 2.1)))*step(0.45, lh.y);
        vec3 lcol = mix(vec3(0.60,0.14,0.03), vec3(0.74,0.36,0.05), fract(lh.x*7.0));
        s.alb = mix(s.alb, lcol*(0.8 + 0.3*f2), leaf);
      }
      s.n = normalize(n + 0.40*(vec3(vnoise3(pw*90.0), vnoise3(pw*90.0 + 5.0), vnoise3(pw*90.0 + 9.0)) - 0.5));
      s.occ = 0.55 + 0.45*smoothstep(0.25,0.65,f2);
      s.wrap = 0.55; s.sheen = 0.8; s.spec = 0.02; s.shin = 8.0;
    } else if(m == M_CAP){
      float cr = length(q.xz);
      vec3 base = gNearId < 2.5 ? vec3(0.62,0.052,0.022) : vec3(0.46,0.24,0.09);
      vec3 rim  = gNearId < 2.5 ? vec3(0.86,0.30,0.05)   : vec3(0.72,0.50,0.26);
      vec3 ca = mix(base, rim, smoothstep(0.45, 1.0, cr));
      // warts
      vec3 g = (q + float(gNearId)*3.3)*3.6;
      vec3 gi = floor(g), gf = fract(g) - 0.5;
      vec3 gh = hash33(gi + 5.0);
      float wart = smoothstep(0.30, 0.22, length(gf - (gh - 0.5)*0.34))*step(0.28, gh.x)*smoothstep(-0.05, 0.12, q.y);
      s.alb = mix(ca, vec3(0.92,0.87,0.74), wart*(gNearId < 2.5 ? 1.0 : 0.0));
      // gills: pale underneath
      float under = smoothstep(0.02, -0.14, q.y);
      float gill = 0.72 + 0.28*sin(atan(q.z, q.x)*46.0);
      s.alb = mix(s.alb, vec3(0.80,0.70,0.52)*gill, under);
      s.occ = mix(1.0, 0.55, under);
      s.spec = 0.35; s.shin = 40.0; s.wrap = 0.35; s.sheen = 0.35; s.trans = 0.25; s.transCol = base;
    } else if(m == M_STIPE){
      float rings = 0.90 + 0.10*sin(q.y*34.0 + 3.0*vnoise2(q.xz*9.0));
      s.alb = vec3(0.84,0.76,0.60)*rings*mix(0.72, 1.0, smoothstep(0.0, 0.5, q.y));
      s.wrap = 0.4; s.sheen = 0.3; s.spec = 0.05;
    } else if(m == M_FERN){
      float tp = smoothstep(0.1, 1.4, length(q));
      vec3 a1 = mix(vec3(0.26,0.075,0.018), vec3(0.42,0.17,0.03), gNearId);
      vec3 a2 = mix(vec3(0.50,0.19,0.03), vec3(0.58,0.32,0.06), gNearId);
      s.alb = mix(a1, a2, tp)*(0.78 + 0.40*vnoise3(pw*30.0));
      s.wrap = 0.55; s.trans = 0.60; s.transCol = s.alb; s.spec = 0.10; s.shin = 24.0; s.sheen = 0.3;
    } else if(m == M_TWIG){
      s.alb = mix(vec3(0.07,0.042,0.028), vec3(0.20,0.12,0.075), vnoise3(pw*vec3(60.0,20.0,60.0)));
      s.wrap = 0.2; s.spec = 0.05; s.shin = 14.0;
    } else if(m == M_BLEAF){
      vec3 c1 = gNearId < 0.40 ? vec3(0.50,0.055,0.018) : (gNearId < 0.75 ? vec3(0.62,0.19,0.024) : vec3(0.66,0.38,0.045));
      vec3 c2 = gNearId < 0.40 ? vec3(0.66,0.20,0.024)  : (gNearId < 0.75 ? vec3(0.70,0.36,0.045) : vec3(0.54,0.31,0.04));
      float mott = fbm3(pw*11.0);
      vec3 ca = mix(c1, c2, smoothstep(0.30, 0.68, mott));
      float vein = smoothstep(0.045, 0.012, abs(q.x));                                     // midrib
      float side = abs(fract(q.y*9.0 - abs(q.x)*2.2) - 0.5);
      vein = max(vein, 0.65*smoothstep(0.07, 0.02, side)*smoothstep(1.0, 0.25, abs(q.x)));  // side veins
      s.alb = mix(ca, ca*vec3(1.25,1.05,0.70) + 0.05, vein*0.6);
      s.alb *= mix(0.72, 1.0, smoothstep(1.0, 0.72, abs(q.x)));                             // tired edges
      s.alb = mix(s.alb, s.alb*vec3(0.55,0.42,0.32), smoothstep(0.58, 0.70, vnoise3(pw*17.0))*0.7);   // a few spots
      s.wrap = 0.5; s.trans = 0.7; s.transCol = s.alb; s.spec = 0.14; s.shin = 30.0; s.sheen = 0.25;
    }
  }
}

vec3 setAmbient(vec3 pw, vec3 n){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z)/SC;
  float openSky = 1.0 - 0.75*canopyPresence(p.xz);
  vec3 sky = mix(vec3(0.20,0.20,0.14), SP_AMB_SKY, openSky)*(0.6 + 0.4*n.y);
  vec3 bounce = SP_BOUNCE*0.55*clamp(0.6 - 0.5*n.y, 0.0, 1.0);
  vec3 clearDir = normalize(vec3(0.15, 0.25, -1.0));
  vec3 fill = vec3(1.0,0.86,0.58)*0.55*pow(clamp(dot(n, clearDir)*0.5+0.5, 0.0, 1.0), 3.0);
  return sky + bounce + fill;
}

// art-directed canopy gaps: a pool on the subject + background shafts
float gapMask(vec2 q, vec2 w){
  float e = 0.30*(vnoise2(q*3.0 + w*2.0) - 0.5) + 0.12*(vnoise2(q*9.0 - w*3.0) - 0.5);
  float g = smoothstep(0.75, 0.38, length(q - HOLE) + e);
  g = max(g, smoothstep(0.55, 0.25, length(q - HOLE - vec2(-1.1,-1.9)) + e)*0.95);
  g = max(g, smoothstep(0.50, 0.22, length(q - HOLE - vec2( 1.4,-1.1)) + e)*0.85);
  g = max(g, smoothstep(0.60, 0.28, length(q - HOLE - vec2(-0.2,-3.3)) + e)*0.95);
  g = max(g, smoothstep(0.45, 0.20, length(q - HOLE - vec2( 2.1,-3.0)) + e)*0.8);
  return g;
}
float setCookie(vec3 pw){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z)/SC;
  vec2 q = p.xz + uSunDir.xz*(max(CANH - p.y, 0.0)/uSunDir.y);
  vec2 w = gWind;
  float open = 1.0 - canopyPresence(q);
  float big  = fbm2_3(q*0.40 + 3.1 + w*0.3);
  float leaf = fbm2_3(q*2.0*SP_DAPPLE_FREQ + w) + 0.06*(vnoise2(q*9.0 + w*4.0) - 0.5);
  float gaps  = smoothstep(0.64, 0.86, big)*0.8;
  float fleck = smoothstep(0.52, 0.78, leaf);
  float l = max(max(gaps, fleck*0.85), gapMask(q, w));
  l = max(l, open*smoothstep(0.35, 0.55, leaf + open*0.5));
  // leaf-sized flecks: what you see crossing the character
  float fine = vnoise2(q*5.3*SP_DAPPLE_FREQ + w*2.0)*0.45 + vnoise2(q*11.0*SP_DAPPLE_FREQ - w*3.0)*0.55;
  l *= 0.52 + 0.48*smoothstep(0.26, 0.72, fine);
  return clamp(l, 0.0, 1.0);
}
float setVol(vec3 pw){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z)/SC;
  vec2 q = p.xz + uSunDir.xz*(max(CANH - p.y, 0.0)/uSunDir.y);
  float open = 1.0 - canopyPresence(q);
  float beam = gapMask(q, gWind)*smoothstep(CANH + 0.6, CANH - 0.2, p.y);
  return max(open*0.06, beam*beam);
}
float setFog(vec3 pw){
  vec3 p = vec3(pw.x, pw.y - G0, pw.z)/SC;
  float forest = canopyPresence(p.xz);
  return (0.0070*exp(-max(p.y, 0.0)*0.30) + 0.010*forest*smoothstep(CANH+0.3, 0.8, p.y))/SC*SP_FOG_GAIN;
}
vec3 setFogAmb(vec3 rd){
  return mix(SP_FOG_LO, SP_FOG_HI, 0.5+0.5*dot(rd, uSunDir))*0.30;
}
