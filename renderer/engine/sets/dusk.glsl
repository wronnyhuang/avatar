// ---------------------------------------------------------------------------
//  engine/sets/dusk.glsl — a hilltop at dusk.  A dark soft meadow rolls over a crest behind the subject, grass tufts
//  and small night flowers on it, tall near-focus grass in the bottom corners.  Past the crest the valley is a matte:
//  a far ridge, a treeline, a village with its lights, mist, a huge low warm moon, first stars, slow cloud.
//  Implements the SET API documented in engine/sets/autumn.glsl.
//
//  Light.  The moon the camera sees (MOON, SP_MOON_DIR) and the key on the character (uSunDir) are different on
//  purpose, as in lagoon.glsl: the key stays in front so the face is lit.  So everything baked is shaded to RADIANCE
//  in setSurface, split in two layers: A (holds still) and S (the moonlight and the lit cloud, which PASS_LIVE
//  multiplies by the slow drifting pattern: that is the cloud going by).  The carrier is the lagoon's:
//  s.alb = S/(sun term), setAmbient() = gRad = A/s.alb.
//
//  The sky is a real surface (a dome of radius DOME) for the same reason: a miss has no S layer.
//
//  Cost.  llvmpipe skips loops only, so every expensive part sits in a one-shot loop behind a bound, and once a pixel
//  is shaded (gRadShade) nothing in setMap wakes up again: the engine's shadow and AO taps cost nothing here.
// ---------------------------------------------------------------------------
#ifndef SP_GROUND_Y
#define SP_GROUND_Y (-1.10)
#endif
#ifndef SP_BOUNCE
#define SP_BOUNCE vec3(0.070,0.085,0.135)            // what little comes up off the dark meadow
#endif
#ifndef SP_AMB_SKY
#define SP_AMB_SKY vec3(0.200,0.190,0.380)           // the blue-violet sky
#endif
#ifndef SP_AMB_WARM
#define SP_AMB_WARM vec3(0.50,0.27,0.12)             // warm fill from the moon side, low
#endif
#ifndef SP_MOON_DIR
#define SP_MOON_DIR vec3(0.2606,0.0279,-0.9650)      // 1710,592 in a 1920x1080 frame of animals/star.json
#endif
#ifndef SP_MOON_R
#define SP_MOON_R 0.0594                             // radians: 165 px
#endif
#ifndef SP_SKY_HZ
#define SP_SKY_HZ vec3(0.700,0.170,0.105)            // the last warmth at the horizon
#endif
#ifndef SP_SKY_ROSE
#define SP_SKY_ROSE vec3(0.200,0.070,0.120)          // dusty rose
#endif
#ifndef SP_SKY_VIOLET
#define SP_SKY_VIOLET vec3(0.047,0.030,0.128)        // violet
#endif
#ifndef SP_SKY_DEEP
#define SP_SKY_DEEP vec3(0.0095,0.0150,0.062)          // deep blue overhead
#endif
#ifndef SP_STAR_AT
#define SP_STAR_AT vec3(0.0)                         // the subject: a warm pool of its glow on the grass
#endif
#ifndef SP_STAR_GLOW
#define SP_STAR_GLOW vec3(0.80,0.42,0.105)
#endif

const float G0 = SP_GROUND_Y;
const vec3  MOON = normalize(SP_MOON_DIR);
const float DOME = 215.0;

#define M_CANOPY 99.0     // unused: nothing in this set ignores direct shadowing
#define M_SKY    0.0
#define M_GROUND 1.0
#define M_BLADE  2.0
#define M_STEM   3.0
#define M_PETAL  4.0
#define M_CORE   5.0

vec2  gWind;
float gRadShade = 0.0;     // PASS_BG: 1 once a pixel is shaded to radiance
vec3  gRad = vec3(1.0);

void setInit(){
  float T = uTime;
  // PASS_LIVE reads this, and adds its own T*0.03: together a steady 7 px/s at 1080p, one way, never coming back
  gWind = vec2(-T*0.0030, -T*0.0006) + 0.004*vec2(sin(T*0.13), cos(T*0.11));
}

// ------------------------------------------------------------------ sky, light (per frame on the character)
float sqr_(float x){ return x*x; }
vec3 skyBase(vec3 rd){
  float y = rd.y;
  vec3 c = mix(SP_SKY_HZ, SP_SKY_ROSE, smoothstep(-0.03, 0.050, y));
  c = mix(c, SP_SKY_VIOLET, smoothstep(0.030, 0.130, y));
  c = mix(c, SP_SKY_DEEP, smoothstep(0.100, 0.245, y));
  float sd = clamp(dot(rd, MOON), 0.0, 1.0);
  float th2 = 2.0*(1.0 - sd);                         // angle squared
  c += vec3(0.75,0.25,0.09)*(0.26*exp(-th2/0.0300) + 0.34*exp(-th2/0.0090) + 0.05*exp(-th2/0.22));
  return c;
}
#ifdef PASS_BG
vec3 setSky(vec3 rd){ return skyBase(rd); }
#else
vec3 setSky(vec3 rd){ return skyBase(rd); }
#endif

vec3 setAmbient(vec3 p, vec3 n){
#ifdef PASS_BG
  if(gRadShade > 0.5) return gRad;
#endif
  vec3 sky = SP_AMB_SKY*(0.60 + 0.40*n.y);
  vec3 bounce = SP_BOUNCE*clamp(0.58 - 0.72*n.y, 0.0, 1.0);
  float w = clamp(dot(n, normalize(vec3(0.62,0.05,0.30)))*0.5 + 0.5, 0.0, 1.0);
  return sky + bounce + SP_AMB_WARM*w*w*w;
}
float setCookie(vec3 p){ return 0.97 + 0.03*sin(uTime*0.13 + p.x*0.3); }
// the sprite pass reads this for its glow, and in the vertex stage the mote's own id is here: fireflies pulse, each
// on its own slow beat, and stay dark across the moment their drift wraps round
#if defined(MOTES) && defined(VS)
float setVol(vec3 p){
  float T = uTime, id = float(gl_VertexID);
  vec3 h = hash31(id*1.731 + 0.37), h2 = hash31(id*0.913 + 7.1);
  float wr = mod(T*(h.x - 0.4) + h2.x*30.0, 30.0);
  float beat = 0.5 + 0.5*sin(T*(0.9 + 0.8*h.z) + h2.z*40.0);
  beat = beat*beat;
  // the sprite divides its glow by the blur disc: give most of it back, so a near firefly is a big soft ember
  float z = dot(p - uCamPos, uCamW);
  float coc = min(uDofK*abs(1.0/uFocus - 1.0/max(z, 0.1)), 0.03*uRes.y);
  float base = (1.2 + 2.0*h2.y)*uRes.y/1080.0, k = max(coc/base, 1.0);
  // none crosses the face: over the pillow it goes out
  vec2 sc = p.xy*(length(uCamPos)/max(z, 0.1));
  float clear_ = smoothstep(0.80, 1.08, length(sc - vec2(0.0,0.05)));
  return (0.10 + 0.90*beat)*pow(k, 1.70)*clear_*smoothstep(0.0, 1.6, wr)*smoothstep(30.0, 28.4, wr) - 0.18;
}
#else
float setVol(vec3 p){ return 0.0; }
#endif
float setFog(vec3 p){ return 0.00012; }
vec3 setFogAmb(vec3 rd){ return mix(vec3(0.30,0.24,0.52), vec3(1.0,0.60,0.34), pow(clamp(dot(rd, MOON), 0.0, 1.0), 16.0)); }

// =========================================================================== bake only
#ifdef PASS_BG
// -------- the hill.  Its crest runs across the frame behind the subject; it falls away toward the camera, at the
// sides and behind, so past the crest a ray never comes down again and reaches the matte
float groundH(vec2 xz){
  vec2 c = (xz - vec2(0.15,-2.30))*vec2(0.235,1.0);
  float h = -0.850 - 0.092*dot(c, c);
  h += 0.070*(vnoise2(xz*vec2(0.62,0.90) + 3.3) - 0.5) + 0.026*(vnoise2(xz*vec2(1.9,2.6) + 8.1) - 0.5);
  return h;
}

// -------- grass
vec3  gNearQ = vec3(0.0); float gNearId = 0.0;     // what the painter needs: local coordinate, hash of the piece
vec3  gTmpQ = vec3(0.0);  float gTmpId = 0.0;
vec2 opUn(vec2 a, vec2 b){ if(b.x < a.x){ gNearQ = gTmpQ; gNearId = gTmpId; return b; } return a; }

float arcS(vec3 q, float Ra, out float u){
  float th = atan(q.y, Ra - q.x);
  u = length(vec2(q.x - Ra, q.y)) - Ra;
  return th*Ra;
}
// a leaf blade: a thin ribbon along an arc, turning slowly so a blade seen edge on still shows a face
float sdBlade(vec3 p, vec3 o, float az, float L, float Ra, float W, float tw, out vec2 lc){
  vec3 q = p - o;
  q.xz = rot(az)*q.xz;
  float u; float sr = arcS(q, Ra, u);
  float s = clamp(sr, 0.0, L), t = s/L;
  vec2 c = rot(tw + 1.5*t)*vec2(u, q.z);
  float w = W*pow(1.0 - t, 0.62)*(0.72 + 0.28*smoothstep(0.0, 0.30, t));
  float cx = c.x - 0.30*abs(c.y);
  float dsec = length(vec2(max(abs(c.y) - w, 0.0), cx)) - mix(0.0036, 0.0012, t);
  lc = vec2(c.y/max(w, 1e-4), t);
  return max(dsec, sr - L)*0.62;
}
// a flower head: thin five-lobed cup facing n, radius R
float sdFlower(vec3 p, vec3 c, vec3 n, float R, float sd, out float rn, out float core){
  vec3 v = p - c;
  vec3 x = normalize(cross(n, vec3(0.31,0.17,0.93))), y = cross(n, x);
  vec3 l = vec3(dot(v, x), dot(v, y), dot(v, n));
  float a = atan(l.y, l.x) + sd*6.0, r = length(l.xy);
  float lobe = 0.70 + 0.30*abs(cos(2.5*a));
  l.z -= 0.38*r*r/R;                                  // cupped
  float d = max(r - R*lobe, abs(l.z) - 0.0040)*0.72;
  rn = r/(R*lobe);
  float cd = length(l - vec3(0.0,0.0,0.006)) - 0.17*R;
  core = cd < d ? 1.0 : 0.0;
  return min(d, cd);
}

// a hero clump at the frame edge: c root, sp spread, H height, dirx the way it leans, NB blades, NF flowers
vec2 heroClump(vec3 p, vec3 c, vec2 sp, float H, float dirx, float sd, int NB, int NF){
  float d = 1e9, mt = M_BLADE;
  for(int i=0;i<NB;i++){
    float fi = float(i);
    vec3 h = hash31(fi*1.37 + sd), h2 = hash31(fi*2.91 + sd + 4.0);
    vec3 o = c + vec3((h.x - 0.5)*sp.x, 0.0, (h.y - 0.5)*sp.y);
    o.y = groundH(o.xz) - 0.02;
    float az = dirx + (h2.x - 0.5)*1.3;
    float L = H*mix(0.50, 1.10, h.z);
    vec2 lc;
    float db = sdBlade(p, o, az, L, L*mix(0.95, 2.8, h2.y), mix(0.013, 0.021, h2.z), 1.05 + h.x*1.2, lc);
    if(db < d){ d = db; mt = M_BLADE; gTmpQ = vec3(lc, 0.0); gTmpId = h.y; }
  }
  for(int i=0;i<NF;i++){
    float fi = float(i);
    vec3 h = hash31(fi*5.13 + sd + 9.0), h2 = hash31(fi*3.77 + sd + 2.0);
    vec3 o = c + vec3((h.x - 0.5)*sp.x*1.15, 0.0, (h.y - 0.5)*sp.y);
    o.y = groundH(o.xz) - 0.02;
    float H1 = H*mix(0.34, 0.86, h.z);
    float Ra = H1*mix(1.8, 3.6, h2.x);
    float az = dirx + (h2.y - 0.5)*1.6;
    vec3 q = p - o; q.xz = rot(az)*q.xz;
    float u; float sr = arcS(q, Ra, u);
    float ds = max(length(vec2(u, q.z)) - 0.0050, sr - H1)*0.8;
    float th = H1/Ra;
    vec3 tip = o + vec3(Ra*(1.0 - cos(th))*cos(az), Ra*sin(th), Ra*(1.0 - cos(th))*sin(az));
    vec3 fn = normalize(vec3(0.20*(h2.z - 0.5) - 0.12*sign(tip.x), 0.46, 0.78));
    float R = mix(0.050, 0.074, h2.z), rn, core;
    float df = sdFlower(p, tip, fn, R, h.x, rn, core);
    if(ds < d){ d = ds; mt = M_STEM; gTmpQ = vec3(0.0); gTmpId = h.y; }
    if(df < d){ d = df; mt = core > 0.5 ? M_CORE : M_PETAL; gTmpQ = vec3(rn, 0.0, 0.0); gTmpId = h2.y; }
  }
  gTmpQ.z = 0.0;
  return vec2(d, mt);
}
// meadow tufts on a jittered grid: a constant 4 cells per lookup, 5 blades or a flower each
vec2 tufts(vec3 p, float gh){
  const float CS = 0.135;
  vec2 g = p.xz/CS - 0.5; vec2 c0 = floor(g);
  float d = 1e9, mt = M_BLADE;
  for(int j=0;j<2;j++) for(int i=0;i<2;i++){
    vec2 ci = c0 + vec2(float(i), float(j));
    vec3 h = hash33(vec3(ci, 11.0));
    vec2 base = (ci + 0.5 + (h.xy - 0.5)*0.78)*CS;
    float patch_ = smoothstep(0.34, 0.60, vnoise2(base*1.3 + 5.0));
    float pres = 0.30 + 0.62*patch_;
    if(h.z > pres) continue;
    float H = mix(0.11, 0.26, fract(h.x*5.3))*(0.8 + 0.5*patch_);
    bool flower = fract(h.y*7.7) > 0.93;
    float ghB = groundH(base) - 0.012;
    vec3 o = vec3(base.x, ghB, base.y);
    for(int k=0;k<5;k++){
      float fk = float(k);
      vec3 hk = hash31(dot(ci, vec2(1.37,4.11)) + fk*3.1);
      float az = hk.x*6.283;
      float L = H*mix(0.6, 1.1, hk.y);
      vec3 ok = o + vec3((hk.y - 0.5)*0.05, 0.0, (hk.z - 0.5)*0.05);
      vec2 lc;
      float db = sdBlade(p, ok, az, L, L*mix(0.9, 2.2, hk.z), 0.0062, 1.1 + hk.x, lc);
      if(db < d){ d = db; mt = M_BLADE; gTmpQ = vec3(lc, 0.0); gTmpId = hk.y; }
    }
    for(int k=0;k<1 && flower;k++){
      vec3 tip = o + vec3(0.02, H*1.05, 0.02);
      float ds = sdCapsule(p, o, tip, 0.0034);
      float rn, core;
      vec3 fn = normalize(vec3((h.x - 0.5)*0.7, 0.70, 0.62));
      float df = sdFlower(p, tip, fn, mix(0.022, 0.034, h.y), h.x, rn, core);
      if(ds < d){ d = ds; mt = M_STEM; gTmpQ = vec3(0.0); gTmpId = h.y; }
      if(df < d){ d = df; mt = core > 0.5 ? M_CORE : M_PETAL; gTmpQ = vec3(rn, 0.0, 0.0); gTmpId = h.z; }
    }
  }
  return vec2(d, mt);
}

const vec3 CL_L = vec3(-2.06,-1.0,0.25), CL_R = vec3(2.10,-1.0,0.30), CL_L2 = vec3(-1.36,-1.0,0.95), CL_R2 = vec3(1.50,-1.0,1.05);

vec2 setMap(vec3 p, bool shadow){
  bool off = gRadShade > 0.5;
  const float G = 0.20;
  // the dome.  Shaded pixels return "no shadow here": the S layer keeps all its light
  vec2 res = vec2(off ? 1e3 : DOME - length(p), M_SKY);
  float gh = 1e3;
  for(int i=0;i<1 && !off;i++){
    gh = groundH(p.xz);
    res = opU(res, vec2((p.y - gh)*0.86, M_GROUND));
  }
  if(!shadow){
    float hy = p.y - gh;
    float bT = max(max(hy - 0.40, p.z - 2.8), -p.z - 5.2);
    res = opU(res, vec2(off ? 1e3 : max(bT, G), M_GROUND));
    for(int i=0;i<1 && !off && bT < G;i++) res = opUn(res, tufts(p, gh));
    float bL = min(sdCapsule(p, vec3(-2.10,-1.7,0.25), vec3(-2.45,0.0,0.25), 1.00), sdCapsule(p, vec3(-1.36,-1.8,0.95), vec3(-1.5,-0.8,0.95), 0.60));
    float bR = min(sdCapsule(p, vec3( 2.10,-1.7,0.30), vec3( 2.50,0.0,0.30), 1.00), sdCapsule(p, vec3( 1.50,-1.8,1.05), vec3( 1.65,-0.8,1.05), 0.60));
    res = opU(res, vec2(off ? 1e3 : max(min(bL, bR), G), M_GROUND));
    for(int i=0;i<1 && !off && bL < G;i++){
      res = opUn(res, heroClump(p, CL_L, vec2(1.10,1.0), 1.42, 3.14159, 1.3, 28, 7));
      res = opUn(res, heroClump(p, CL_L2, vec2(0.62,0.5), 0.56, 3.14159, 3.1, 12, 4));
    }
    for(int i=0;i<1 && !off && bR < G;i++){
      res = opUn(res, heroClump(p, CL_R, vec2(1.00,1.0), 1.30, 0.0, 5.9, 26, 5));
      res = opUn(res, heroClump(p, CL_R2, vec2(0.60,0.5), 0.52, 0.0, 7.7, 12, 4));
    }
  }
  return res;
}

// -------- the matte past the crest, in slopes (x, y) = rd.xy/-rd.z.  The camera's axis is at y = 0.048
float treeTop(float x, float W, float H, float seed){
  float cx = floor(x/W), top = 0.0;
  for(int i=-2;i<=2;i++){
    float c = cx + float(i);
    vec3 h = hash31(c*1.13 + seed);
    float xc = (c + 0.5 + 0.8*(h.x - 0.5))*W;
    float hw = W*(0.66 + 0.62*h.y), ht = H*(0.52 + 0.62*h.z);
    float tall = step(0.92, fract(h.x*7.3));
    hw *= 1.0 - 0.34*tall; ht *= 1.0 + 0.34*tall;
    float u = (x - xc)/hw;
    top = max(top, ht*pow(max(1.0 - u*u, 0.0), mix(0.50, 0.85, tall)));
  }
  return top;
}
// houses on the field's top edge: m = silhouette, l = window light
void village(float x, float y, float y0, out float m, out float l){
  const float W = 0.0098;
  float cx = floor(x/W);
  m = 0.0; l = 0.0;
  for(int i=-1;i<=1;i++){
    float c = cx + float(i);
    vec3 h = hash31(c*2.31 + 4.0), h2 = hash31(c*3.77 + 1.0);
    float xc = (c + 0.5 + 0.5*(h.x - 0.5))*W;
    float pres = smoothstep(-0.345, -0.300, xc)*smoothstep(-0.075, -0.125, xc);
    if(h.z > pres*0.92) continue;
    float hw = 0.0030 + 0.0026*h.y, hh = 0.0045 + 0.0040*h2.x, hr = 0.0034 + 0.0020*h2.y;
    float base = y0 - 0.0090*h2.z - 0.0008;
    float ax = abs(x - xc);
    float wall = step(ax, hw)*step(base, y)*step(y, base + hh);
    float roof = step(y, base + hh + hr*(1.0 - ax/(hw + 0.0012)))*step(base + hh - 0.0002, y);
    m = max(m, max(wall, roof));
    vec2 w1 = vec2(x - xc - hw*0.45*(h.x - 0.5)*2.0, y - base - hh*0.52);
    l += step(0.48, h2.y)*(0.25 + 0.75*fract(h2.x*5.1))*exp(-dot(w1, w1)/(0.00078*0.00078));
  }
  // the church, off centre of the village: tower and spire
  float cxx = -0.192;
  float ax = abs(x - cxx), b = y0 - 0.001;
  float tower = step(ax, 0.0028)*step(b, y)*step(y, b + 0.0150);
  float spire = step(y, b + 0.0150 + 0.0150*(1.0 - ax/0.0030))*step(b + 0.0148, y);
  float nave = step(abs(x - cxx - 0.0085), 0.0072)*step(b, y)*step(y, b + 0.0072 + 0.0036*(1.0 - abs(x - cxx - 0.0085)/0.0072));
  m = max(m, max(max(tower, spire), nave));
  vec2 w2 = vec2(x - cxx - 0.0085, y - b - 0.0045);
  l += 0.9*exp(-dot(w2, w2)/(0.0011*0.0011));
}

void farWorld(vec3 rd, out vec3 A, out vec3 S){
  vec2 f = rd.xy/max(-rd.z, 0.02);
  float x = f.x, y = f.y;
  A = skyBase(rd);
  S = vec3(0.0);
  float sd = clamp(dot(rd, MOON), 0.0, 1.0);
  float th = sqrt(max(2.0*(1.0 - sd), 0.0));

  // first stars: few, only where the sky has gone dark, and clear of the moon
  vec2 sc = vec2(x, y)/0.0165;
  vec2 si = floor(sc);
  vec3 sh = hash33(vec3(si, 3.0));
  vec2 sp = (si + 0.5 + 0.70*(sh.xy - 0.5))*0.0165;
  float sa = smoothstep(0.085, 0.175, sp.y)*step(sh.z, 0.12)*smoothstep(0.17, 0.30, th);
  float srad = 0.00060 + 0.00060*fract(sh.x*9.1)*fract(sh.x*9.1);
  float dots = exp(-dot(f - sp, f - sp)/(srad*srad));
  A += sa*dots*mix(vec3(1.0,0.86,0.70), vec3(0.72,0.84,1.0), fract(sh.y*5.0))*(0.7 + 5.0*pow(fract(sh.y*13.0), 3.0));

  // the moon: warm, huge, barely off the land
  float disc = smoothstep(SP_MOON_R + 0.0007, SP_MOON_R - 0.0007, th);
  vec3 mp = (rd - MOON*dot(rd, MOON))/SP_MOON_R;
  float mar = fbm2(mp.xy*vec2(1.55,1.75) + vec2(4.2,2.3));
  float mar2 = vnoise2(mp.xy*5.3 + 7.0);
  vec3 mc = mix(vec3(2.30,0.78,0.22), vec3(1.20,0.34,0.12), smoothstep(0.42, 0.62, mar)*0.86 + 0.14*mar2);
  mc *= 1.0 - 0.20*sqr_(th/SP_MOON_R)*sqr_(th/SP_MOON_R);
  mc = mix(mc, mc*vec3(0.92,0.52,0.42), smoothstep(0.35, -1.0, mp.y));   // redder near the land
  A = mix(A, mc, disc);

  // cloud: long soft banks, lit from under by the moon.  Their light is in S, so it drifts
  vec2 cu = vec2(x*2.3, y*15.0);
  float wp = fbm2_3(cu*vec2(1.1,1.5) + 11.0);
  float n1 = fbm2(cu + vec2(wp*1.6, wp*0.5) + vec2(3.1,0.7));
  float n2 = fbm2(cu*vec2(2.1,2.3) + vec2(wp*1.2, 0.0) + 7.7);
  float band1 = exp(-sqr_((y - 0.070 - 0.030*sin(x*3.4 + 0.8))/0.042));
  float band2 = exp(-sqr_((y - 0.018 - 0.010*sin(x*5.0 + 2.0))/0.018));
  float band3 = exp(-sqr_((y - 0.160 + 0.040*sin(x*2.6))/0.030));
  float cl = smoothstep(0.44, 0.70, n1*0.72 + n2*0.28);
  float c1 = cl*max(band1, 0.85*band2), c3 = 0.55*smoothstep(0.50, 0.72, n1*0.5 + n2*0.5)*band3;
  float nearM = exp(-th*th/0.060);
  vec3 lit = mix(vec3(0.046,0.028,0.092), vec3(0.72,0.23,0.13), nearM);
  A = mix(A, A*0.80, 0.8*max(c1, c3)*(1.0 - 0.9*nearM*0.0));
  S += lit*(c1 + 0.55*c3);

  // the cloud that moves.  PASS_LIVE multiplies S by a soft noise it slides sideways every frame, so an even veil of
  // light in S comes out as cloud banks drifting across the sky at PASS_LIVE's rate.  A gives back what the noise's
  // floor (bg_sun[0]) would add, so the sky between the banks is the sky above
  float vl = smoothstep(0.004, 0.050, y)*(1.0 - 0.72*smoothstep(0.10, 0.23, y))*(1.0 - disc);
  vec3 veil = mix(vec3(0.044,0.026,0.088), vec3(0.42,0.135,0.075), exp(-th*th/0.050))*vl;
  S += veil;
  A = max(A - veil*0.62, vec3(0.0));

  // the far ridge, hazy, a shade under the sky; then the treeline; then the village field
  float mh = exp(-sqr_(x - MOON.x/-MOON.z)/0.040);          // the land is lit from the moon's side
  float r1 = 0.0070 + 0.0120*(fbm2_3(vec2(x*5.5, 3.0)) - 0.5) + 0.0040*sin(x*11.0 + 1.0);
  float a1 = smoothstep(0.00050, -0.00050, y - r1);
  vec3 c1a = mix(vec3(0.032,0.026,0.085), vec3(0.120,0.050,0.075), mh);
  A = mix(A, c1a, a1); S *= 1.0 - a1;

  float base2 = -0.0120 + 0.0050*(fbm2_3(vec2(x*7.0, 8.0)) - 0.5) + 0.0040*sin(x*3.1 + 2.2);
  float tt = treeTop(x, 0.0190, 0.0200, 2.0) + 0.45*treeTop(x, 0.0068, 0.0085, 5.0);
  tt *= 1.0 - 0.72*smoothstep(0.24, 0.10, abs(x + 0.165));             // the trees thin out behind the village
  float r2 = base2 + tt;
  float a2 = smoothstep(0.00045, -0.00045, y - r2);
  vec3 c2a = mix(vec3(0.0075,0.0085,0.028), vec3(0.020,0.0120,0.026), mh);
  A = mix(A, c2a, a2); S *= 1.0 - a2;

  // mist lying in front of the trees, warmer toward the moon
  float mist = exp(-sqr_((y - base2 + 0.0040)/0.0100))*(0.55 + 0.45*vnoise2(vec2(x*15.0, y*120.0)));
  A += mix(vec3(0.016,0.016,0.052), vec3(0.11,0.045,0.052), mh)*mist*0.80;

  float y3 = -0.0270 + 0.0034*sin(x*7.0 + 1.3) + 0.0030*(fbm2_3(vec2(x*16.0, 1.0)) - 0.5);
  float a3 = smoothstep(0.00045, -0.00045, y - y3);
  float fld = fbm2_3(vec2(x*16.0, y*95.0) + 2.0);
  vec3 c3a = mix(vec3(0.0060,0.0085,0.024), vec3(0.0140,0.0190,0.040), fld)*(1.0 + 1.1*mh);
  c3a = mix(c3a, c3a*0.55, smoothstep(0.0, 0.040, y3 - y));
  float m, l;
  village(x, y, y3, m, l);
  A = mix(A, vec3(0.0040,0.0045,0.016), m*(1.0 - a3));
  A = mix(A, c3a, a3); S *= 1.0 - a3;
  // a pond in the low ground on the moon's side, holding the sky and a broken track of the moon
  float mxs = MOON.x/-MOON.z;
  vec2 pc = vec2((x - mxs + 0.055)/0.215, (y + 0.0420)/0.0105);
  float pe = 0.14*(vnoise2(vec2(x*26.0, 3.0)) - 0.5) + 0.06*sin(x*40.0);
  float pond = smoothstep(1.0 + pe, 0.90 + pe, length(pc))*a3;
  vec3 pw = mix(vec3(0.050,0.030,0.082), vec3(0.150,0.062,0.082), exp(-sqr_(x - mxs)/0.018));
  float rip = smoothstep(0.42, 0.70, vnoise2(vec2(x*46.0, y*720.0))*0.6 + vnoise2(vec2(x*120.0, y*1500.0))*0.4);
  float track = exp(-sqr_(x - mxs)/0.0016);
  A = mix(A, pw*(0.82 + 0.36*rip), pond);
  S += vec3(1.25,0.42,0.13)*track*rip*pond;
  // scattered windows down the valley, and a farm or two on the right
  vec2 fi = vec2(x, y)/vec2(0.052, 0.018);
  vec2 ff = floor(fi);
  vec3 lh = hash33(vec3(ff, 8.0));
  vec2 lp = (ff + 0.5 + 0.8*(lh.xy - 0.5))*vec2(0.052, 0.018);
  float lf = step(lh.z, 0.36)*a3*smoothstep(0.0, 0.006, y3 - lp.y);
  l += lf*(0.45 + 0.55*fract(lh.x*7.0))*exp(-dot(f - lp, f - lp)/(0.00085*0.00085));
  A += vec3(2.30,0.95,0.26)*l*3.2;
  // their light in the air over the roofs
  A += vec3(0.070,0.032,0.020)*exp(-sqr_((x + 0.200)/0.095))*exp(-sqr_((y - y3 - 0.004)/0.012));
}

// radiance of one surface in the night.  A holds still, S is what the moon gives and goes when cloud crosses it
void night(vec3 p, vec3 n, vec3 rd, vec3 alb, float occ, float trans, out vec3 A, out vec3 S){
  float up = 0.5 + 0.5*n.y;
  vec3 ml = normalize(MOON + vec3(0.0,0.30,0.0));
  float dm = clamp(dot(n, ml)*0.70 + 0.30, 0.0, 1.0);
  // back-lit: the blade or the grass turning toward the moon lights up along its edge
  float nv = clamp(dot(n, -rd), 0.0, 1.0);
  float toM = pow(clamp(dot(rd, MOON)*0.5 + 0.5, 0.0, 1.0), 6.0);
  vec3 sky = SP_AMB_SKY*(0.20 + 0.72*up)*occ;
  vec3 v = p - SP_STAR_AT;
  float pool = exp(-dot(v, v)/4.2);
  vec3 glow = SP_STAR_GLOW*pool*(0.30 + 0.70*clamp(dot(n, -v/max(length(v), 1e-3)), 0.0, 1.0))*occ;
  A = alb*(sky + glow);
  vec3 moon = vec3(0.62,0.46,0.34)*dm*dm*occ;
  vec3 back = vec3(1.0,0.66,0.40)*toM*(pow(1.0 - nv, 2.0)*0.75 + trans*0.75);
  // the moon is on the right of the frame: the tips on that side catch more of it
  float side = smoothstep(-2.2, 2.4, p.x);
  S = alb*(moon + back*(0.4 + 0.6*occ)*(0.55 + 1.5*side*side));
}

// hand A and S to the engine: see the header
void radiance(vec3 A, vec3 S, inout Surf s){
  s.n = vec3(0.0,1.0,0.0); s.wrap = 1.0;
  float dif = (dot(s.n, uSunDir) + 1.0)*0.5;
  s.alb = S/(uSunCol*dif) + 1e-4;
  gRad = A/s.alb;
  s.spec = 0.0; s.sheen = 0.0; s.trans = 0.0; s.occ = 1.0;
  gRadShade = 1.0;
}

void setSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s){
  if(!(dot(n, n) > 0.5)) n = vec3(0.0,0.0,1.0);    // out of steps in the grass: no gradient, and one NaN kills the FFT
  vec3 A, S;
  if(m == M_SKY){
    farWorld(rd, A, S);
  } else if(m == M_GROUND){
    vec2 xz = p.xz;
    float f1 = fbm2_3(xz*vec2(0.9,1.5) + 2.0);
    float f2 = vnoise2(xz*vec2(7.0,13.0));
    float f3 = vnoise2(xz*vec2(31.0,57.0) + 4.0);
    vec3 a = mix(vec3(0.020,0.040,0.060), vec3(0.046,0.082,0.078), smoothstep(0.30, 0.74, f1));
    a *= 0.52 + 0.62*f2;
    a *= 0.72 + 0.56*f3;
    vec3 gn = vec3(vnoise2(xz*44.0), 0.0, vnoise2(xz*44.0 + 31.0)) - vec3(0.5,0.0,0.5);
    vec3 N = normalize(n + gn*0.80);
    night(p, N, rd, a, 0.42 + 0.58*f3, 0.0, A, S);
  } else {
    setMap(p, false);                               // again, for the painter's local coordinates
    if(m == M_BLADE){
      float t1 = gNearQ.y;
      vec3 c = mix(vec3(0.016,0.040,0.052), mix(vec3(0.060,0.120,0.100), vec3(0.100,0.135,0.085), gNearId), smoothstep(0.0, 0.8, t1));
      c *= 0.86 + 0.28*vnoise2(vec2(gNearQ.x*3.0, t1*30.0));
      c = mix(c, c*0.72, smoothstep(0.08, 0.0, abs(gNearQ.x)));        // midrib
      night(p, n, rd, c, 0.30 + 0.70*smoothstep(0.0, 0.7, t1), 0.55, A, S);
    } else if(m == M_STEM){
      night(p, n, rd, vec3(0.055,0.105,0.075), 0.65, 0.3, A, S);
    } else if(m == M_PETAL){
      float rn = gNearQ.x;
      vec3 c = mix(mix(vec3(0.20,0.24,0.95), vec3(0.42,0.26,0.92), gNearId), vec3(0.74,0.68,0.98), smoothstep(0.30, 1.0, rn));
      night(p, n, rd, c, 0.70, 0.60, A, S);
      A += c*vec3(0.070,0.070,0.130);                                    // night flowers hold a little light
    } else {
      night(p, n, rd, vec3(0.95,0.66,0.18), 0.9, 0.3, A, S);
      A += vec3(0.20,0.13,0.035);
    }
  }
  // one NaN turns the whole FFT lens into NaN; a scissor edge through the bake's strips has made a few
  if(!(dot(A, A) < 1e8)) A = vec3(0.0);
  if(!(dot(S, S) < 1e8)) S = vec3(0.0);
  radiance(A, S, s);
}
#else
vec2 setMap(vec3 p, bool shadow){ return vec2(1.0, 0.0); }
void setSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s){}
#endif
