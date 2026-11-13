// ---------------------------------------------------------------------------
//  engine/sets/garden.glsl — a morning garden just after rain, at dew-drop scale.  A huge glossy leaf with a bright
//  midrib and veins runs from the near left to its tip far on the right, beaded with dew; behind it a giant stem
//  and a flower stand in the mist, the sun soft in the upper left.  Implements the SET API of engine/sets/autumn.glsl.
//
//  Shading.  As lagoon.glsl: every surface is shaded to radiance in setSurface, and scene_main multiplies by s.alb,
//  so s.alb = what the SUN layer should hold, setAmbient() = gRad = wanted radiance / s.alb.  Only the leaf keeps a
//  real sun term, which is what the drifting light in PASS_LIVE moves over it.
//
//  gardenWorld(rd) is the whole place as one cheap function of direction: sky, flower, stem, the leaf with its
//  midrib and veins.  It is what the drop refracts and everything glossy mirrors, in every pass.
//
//  Cost.  llvmpipe does not skip `if`, only loops: the beads and the far garden sit in one-shot loops behind
//  bounds (see G in setMap) and everything heavy is inside #ifdef PASS_BG.
// ---------------------------------------------------------------------------
#ifndef SP_GROUND_Y
#define SP_GROUND_Y (-0.575)                         // the leaf under the drop
#endif
#ifndef SP_SUN_VIS
#define SP_SUN_VIS vec3(-0.150,0.085,-0.985)         // the sun the camera sees, not the key light
#endif
#ifndef SP_BOUNCE
#define SP_BOUNCE vec3(0.20,0.36,0.11)               // green light off the leaf
#endif
#ifndef SP_AMB_SKY
#define SP_AMB_SKY vec3(0.42,0.47,0.40)              // pale wet morning
#endif
#ifndef SP_SKY_LOW
#define SP_SKY_LOW vec3(0.085,0.170,0.105)              // the mist low down, over the garden
#endif
#ifndef SP_SKY_MID
#define SP_SKY_MID vec3(0.30,0.40,0.26)              // pale green-gold
#endif
#ifndef SP_SKY_HIGH
#define SP_SKY_HIGH vec3(0.16,0.30,0.30)             // cool, clearing overhead
#endif
#ifndef SP_SUN_GLOW
#define SP_SUN_GLOW vec3(1.75,1.28,0.66)
#endif
#ifndef SP_FOG_NEAR
#define SP_FOG_NEAR vec3(0.30,0.40,0.30)             // mist away from the sun
#endif
#ifndef SP_FOG_SUN
#define SP_FOG_SUN vec3(1.90,1.45,0.82)              // mist where the sun is
#endif
#ifndef SP_FOG_GAIN
#define SP_FOG_GAIN 1.0
#endif
#ifndef SP_LEAF_DEEP
#define SP_LEAF_DEEP vec3(0.005,0.040,0.012)         // the leaf, in its troughs
#endif
#ifndef SP_LEAF_MID
#define SP_LEAF_MID vec3(0.030,0.118,0.014)
#endif
#ifndef SP_LEAF_LIT
#define SP_LEAF_LIT vec3(0.26,0.47,0.060)            // backlit, far along the leaf
#endif
#ifndef SP_RIB
#define SP_RIB vec3(0.70,0.80,0.30)                  // midrib and veins
#endif
#ifndef SP_PETAL_A
#define SP_PETAL_A vec3(0.86,0.36,0.43)              // the flower, lit
#endif
#ifndef SP_PETAL_B
#define SP_PETAL_B vec3(0.44,0.075,0.17)              // toward its heart
#endif

const float WL = SP_GROUND_Y;
const vec3  SUNV = normalize(SP_SUN_VIS);
// the leaf: a = along the midrib, toward the far tip; b = across, + to screen right
const vec2  LEAF_O = vec2(-1.55, 1.20);
const vec2  LEAF_D = vec2(0.437,-0.900);
const vec2  LEAF_P = vec2(0.900, 0.437);
const float LEAF_A0 = 1.757, LEAF_B0 = 0.871;        // the drop, in those coordinates
const float LEAF_TIP = 11.6;
const float VEIN_DS = 1.28;

#define M_CANOPY 99.0     // unused: nothing in this set ignores direct shadowing
#define M_LEAF  1.0
#define M_BEAD  2.0
#define M_STEM  3.0
#define M_PETAL 4.0
#define M_HEART 5.0
#define M_BACK  6.0
#define M_NEAR  8.0

vec2  gWind;
float gHzSoft = 0.015;      // half width of the horizon in gardenWorld: the drop's lens sets it wider
float gVeinK = 1.0;         // and the strength of the veins
float gRadShade = 0.0;
vec3  gRad = vec3(1.0);
#ifdef PASS_BG
#define NBEAD 15
float gBY[NBEAD];
#endif

// ------------------------------------------------------------------ the leaf's form (all passes)
float leafRaw(float a, float b){
  float ab = abs(b);
  float h = 0.30*(1.0 - exp(-ab/1.7));               // the blades rise off the midrib
  float up = smoothstep(-1.5, 7.2, a);
  h += 0.92*up*up;                                   // a long cup, rising away from the camera ...
  float fall = smoothstep(6.2, LEAF_TIP, a);
  h -= 2.6*fall*fall;                                // ... then the tip falls away
  h -= 0.010*ab*ab;
  return h;
}

// ------------------------------------------------------------------ the world in one function
vec3 skyBase(vec3 rd){
  float y = rd.y;
  vec3 c = mix(SP_SKY_LOW, SP_SKY_MID, smoothstep(-0.10, 0.10, y));
  c = mix(c, SP_SKY_HIGH, smoothstep(0.06, 0.62, y));
  float sd = max(dot(rd, SUNV), 0.0);
  float s2 = sd*sd, s4 = s2*s2, s8 = s4*s4, s16 = s8*s8, s32 = s16*s16, s64 = s32*s32, s128 = s64*s64, s256 = s128*s128;
  c += SP_SUN_GLOW*(0.10*s16 + 0.40*s64 + 0.90*s256 + 2.6*s256*s256);
  return c;
}
vec3 gardenWorld(vec3 rd){
  vec3 c = skyBase(rd);
  // ---- the stem and the flower, behind and to the right
  float zz = max(-rd.z, 0.02);
  vec2 ss = rd.xy/zz;
  float sx = ss.x - 0.41 - 0.10*max(0.19 - ss.y, 0.0);
  float stem = smoothstep(0.040, 0.022, abs(sx))*smoothstep(0.19, 0.14, ss.y)*smoothstep(0.0, 0.1, -rd.z);
  c = mix(c, vec3(0.42,0.56,0.22) + vec3(0.30,0.28,0.10)*smoothstep(0.03, -0.03, sx), 0.85*stem);
  vec2 fv = ss - vec2(0.420,0.215);                  // further out than the real one: the lens brings it clear of the mouth
  float fr = length(fv), fa = atan(fv.y, fv.x);
  float pet = smoothstep(0.02, -0.015, fr - 0.105*(0.72 + 0.28*abs(cos(fa*3.5 + 0.4))))*smoothstep(0.0, 0.1, -rd.z);
  vec3 fc = mix(SP_PETAL_B, SP_PETAL_A, smoothstep(0.02, 0.11, fr));
  fc = mix(vec3(1.15,0.80,0.20), fc, smoothstep(0.012, 0.030, fr));
  c = mix(c, fc, pet);
  // ---- the leaf under everything: a plane, with its midrib and veins
  float dn = max(-rd.y, 0.0);
  float hgt = 0.80;
  vec2 pl = rd.xz/max(dn, 0.012)*hgt;
  vec2 dp = pl - LEAF_O;
  float a = dot(dp, LEAF_D), b = dot(dp, LEAF_P);
  float ab = abs(b);
  float vn = fract((a - 0.82*ab)/VEIN_DS + (b > 0.0 ? 0.0 : 0.5)) - 0.5;
  float ve = smoothstep(0.42, 0.495, abs(vn));
  float rib = exp(-b*b/0.022);
  vec3 lf = mix(SP_LEAF_MID*2.3, SP_LEAF_LIT*1.5, smoothstep(2.0, 11.0, a));
  lf = mix(lf, SP_RIB*1.1, gVeinK*clamp(0.62*ve + rib, 0.0, 1.0));
  float haze = smoothstep(0.0, 0.16, dn);
  lf = mix(mix(SP_SKY_LOW, SP_SKY_MID, 0.6)*1.1, lf, haze);
  return mix(c, lf, smoothstep(gHzSoft, -gHzSoft, rd.y));
}

#ifdef PASS_BG
void beadInit();
#endif
void setInit(){
  float T = uTime;
  gWind = 0.05*vec2(sin(T*0.31), cos(T*0.27)) + 0.015*vec2(sin(T*0.83 + 1.0), sin(T*0.71));
#ifdef PASS_BG
  beadInit();
#endif
}

// ------------------------------------------------------------------ light (per frame on the character: plain arithmetic)
#ifndef PASS_BG
vec3 setSky(vec3 rd){ return gardenWorld(rd); }
#endif
vec3 setAmbient(vec3 p, vec3 n){
#ifdef PASS_BG
  if(gRadShade > 0.5) return gRad;
#endif
  vec3 sky = SP_AMB_SKY*(0.66 + 0.34*n.y);
  vec3 bounce = SP_BOUNCE*clamp(0.58 - 0.72*n.y, 0.0, 1.0);
  float w = clamp(dot(n, SUNV)*0.5 + 0.5, 0.0, 1.0);
  return sky + bounce + vec3(1.0,0.80,0.52)*0.30*w*w*w;
}
// slow cloud in the mist: the light breathes, nothing dapples a wet open leaf
float setCookie(vec3 p){ return 0.90 + 0.10*sin(uTime*0.23 + p.x*0.4 + 1.1*sin(uTime*0.09)); }
// long soft shafts from the upper left, leaning across the frame, drifting very slowly
float setVol(vec3 p){
  float u = p.x*0.26 + p.y*0.34 + uTime*0.012;
  return 0.006 + 0.016*smoothstep(0.1, 0.9, 0.5 + 0.5*sin(u*3.1 + 0.8*sin(u*1.3 + 1.0)));
}
float setFog(vec3 p){
  float h = max(p.y - WL, 0.0);
  float d = 0.0016*exp(-h*0.055);
#ifdef PASS_BG
  // soft banks of mist behind the leaf, deep in the garden
  float far_ = smoothstep(8.0, 34.0, length(p.xz));
  float n1 = vnoise2(p.xz*vec2(0.05,0.09) + vec2(3.1,1.7) + p.y*0.06);
  d += 0.0034*far_*smoothstep(0.30, 0.76, n1);
#endif
  return d*SP_FOG_GAIN;
}
vec3 setFogAmb(vec3 rd){
  float sd = clamp(dot(rd, SUNV), 0.0, 1.0);
  float k = sd*sd; k *= k; k *= k; k *= k; k *= k; k *= k;       // ^64
  return mix(SP_FOG_NEAR, SP_FOG_SUN, 0.04 + 0.96*k);
}

// =========================================================================== bake only
#ifdef PASS_BG
// x, z, radius of the dew; y comes from the leaf
const vec3 BEAD[NBEAD] = vec3[NBEAD](
  vec3( 1.20, 0.38, 0.135), vec3( 1.62,-0.58, 0.215), vec3( 2.34,-0.22, 0.085), vec3(-1.46, 0.10, 0.160),
  vec3(-1.86,-0.72, 0.095), vec3(-0.92, 0.98, 0.075), vec3( 0.56, 1.02, 0.100), vec3( 0.98,-1.72, 0.125),
  vec3(-0.52,-2.05, 0.190), vec3( 2.60,-2.10, 0.170), vec3(-2.70,-2.30, 0.130), vec3( 1.05, 1.42, 0.055),
  vec3(-2.62, 0.42, 0.070), vec3( 3.60,-3.40, 0.120), vec3(-0.90,-4.10, 0.150)
);

float leafY(vec2 xz){
  vec2 d = xz - LEAF_O;
  float a = dot(d, LEAF_D), b = dot(d, LEAF_P);
  float h = leafRaw(a, b) - leafRaw(LEAF_A0, LEAF_B0);
  float rr = dot(xz, xz);
  return WL + h*smoothstep(0.10, 2.00, rr);          // level under the drop
}
float leafW(float a){
  float t = clamp((a - 1.5)/(LEAF_TIP - 1.5), 0.0, 1.0);
  float w = 6.5*pow(1.0 - t, 0.72)*sqrt(1.0 + t);
  return w + 0.10*sin(a*1.3 + 1.0)*(1.0 - t) + 0.05*sin(a*3.7);
}
void beadInit(){
  for(int i=0;i<NBEAD;i++) gBY[i] = leafY(BEAD[i].xy);
}

// the giant flower, turned three quarters away toward the upper right so it is an ellipse, not a badge
const vec3 FC = vec3(5.05,3.75,-22.0);
const float FR = 3.30;
const vec3 FZ = vec3(-0.640,-0.200,0.742);
const vec3 FX = vec3(0.757,0.0,0.653);
const vec3 FY = vec3(-0.131,0.980,0.151);
float gPetalU, gPetalS;
vec2 flowerMap(vec3 p, vec2 res){
  vec3 d = p - FC;
  vec3 q = vec3(dot(d, FX), dot(d, FY), dot(d, FZ));
  float bd = length(q) - FR*1.20;
  if(bd >= 1.0) res.x = min(res.x, bd);
  for(int k=0;k<1 && bd<1.0;k++){
    float rr = length(q.xy);
    vec3 c = q; c.z += 0.40*rr*rr/FR - 0.12*FR;                 // the cup
    float az = atan(c.y, c.x);
    for(int l=0;l<2;l++){
      float ph = 0.3927*float(l);
      float w = 0.7854, an = az + ph;
      float id = floor((an + 0.5*w)/w);
      float aa = an - id*w;
      float hp = hash11(id*1.7 + float(l)*5.0 + 2.0);
      aa += (hp - 0.5)*0.16;
      float len = FR*(0.50 - 0.04*float(l))*(0.88 + 0.22*hp);
      vec2 sl = rr*vec2(cos(aa), sin(aa));
      float wv = 0.05*sin(rr*2.3 + hp*6.0)*FR;                  // a petal is never flat
      float pp = sdEllipsoid(vec3(sl.x - len*1.16, sl.y, c.z + 0.10 - 0.22*float(l) + wv), vec3(len, FR*(0.25 - 0.03*float(l)), 0.13))*0.62;
      if(pp < res.x){ res = vec2(pp, M_PETAL); gPetalU = rr/FR; gPetalS = hp; }
    }
    float heart = sdEllipsoid(vec3(q.xy, q.z - 0.85), vec3(FR*0.17, FR*0.17, 0.40))*0.8;
    if(heart < res.x){ res = vec2(heart, M_HEART); gPetalU = rr/FR; }
  }
  return res;
}
vec3 gBackQ = vec3(0.0);
float stemX(float y, float x0, float lean){ return x0 + lean*(y - 3.0) + 0.018*(y - 3.0)*(y - 3.0)*sign(lean); }
vec2 backMap(vec3 p, vec2 res){
  // the giant stem up to the flower, a thinner one far left
  float s1 = (length(vec2(p.x - stemX(p.y, 5.50, 0.10), p.z + 22.4)) - 0.66)*0.9;
  s1 = max(s1, p.y - FC.y);
  if(s1 < res.x) res = vec2(s1, M_STEM);
  float s2 = (length(vec2(p.x - stemX(p.y, -9.0, -0.06), p.z + 27.0)) - 0.42)*0.9;
  if(s2 < res.x) res = vec2(s2, M_STEM);
  // two huge leaves behind, with a real outline
  vec3 a = p - vec3(-8.4,0.2,-17.0); a.xy = rot(-0.50)*a.xy; a.yz = rot(0.46)*a.yz;
  float t1 = clamp(a.x/7.6, -1.0, 1.0);
  float w1 = 2.5*sqrt(1.0 - t1*t1)*(1.0 - 0.34*t1);
  float b1 = smax(abs(a.y + 0.10*a.z*a.z) - 0.05, abs(a.z) - w1, 0.10)*0.42;
  b1 = max(b1, abs(a.x) - 7.6);
  if(b1 < res.x){ res = vec2(b1, M_BACK); gBackQ = vec3(a.x/7.6, a.z/max(w1, 0.1), a.y); }
  vec3 c = p - vec3(10.5,-1.0,-14.0); c.xy = rot(0.34)*c.xy; c.yz = rot(0.52)*c.yz;
  float t2 = clamp(-c.x/7.0, -1.0, 1.0);
  float w2 = 2.2*sqrt(1.0 - t2*t2)*(1.0 - 0.34*t2);
  float b2 = smax(abs(c.y + 0.10*c.z*c.z) - 0.05, abs(c.z) - w2, 0.10)*0.42;
  b2 = max(b2, abs(c.x) - 7.0);
  if(b2 < res.x){ res = vec2(b2, M_BACK); gBackQ = vec3(-c.x/7.0, c.z/max(w2, 0.1), c.y); }
  return res;
}
// near the lens: a leaf tip reaching in from the lower right corner, a bright bead on it
vec2 nearMap(vec3 p, vec2 res){
  vec2 q = p.xy - vec2(0.470,0.095);
  vec2 ax = vec2(0.80,-0.60);
  float u = dot(q, ax), v = dot(q, vec2(-ax.y, ax.x));
  v += 0.10*u*u;                                      // a little curve
  float us = clamp(u/1.2, 0.0, 1.0);
  float w = 0.25*sqrt(us)*(1.0 - 0.45*us);
  float dz = p.z - (3.78 + 0.12*u);
  float d = smax(abs(dz) - 0.006, abs(v) - w, 0.012);
  d = smax(d, -u, 0.010)*0.5;
  if(d < res.x){ res = vec2(d, M_NEAR); gBackQ = vec3(us, v/max(w, 0.01), 0.0); }
  float e = length(p - vec3(0.615,0.044,3.80)) - 0.026;
  if(e < res.x) res = vec2(e, M_BEAD);
  return res;
}

float gNoSun = 0.0;
vec2 setMap(vec3 p, bool shadow){
  // painted: the engine's AO and sun shadow see nothing but the drop, and a far surface skips the shadow march
  if(gRadShade > 0.5) return vec2((shadow && gNoSun > 0.5) ? -1.0 : 1e3, M_LEAF);
  vec2 res = vec2(1e4, M_LEAF);
  vec2 d = p.xz - LEAF_O;
  float a = dot(d, LEAF_D), b = dot(d, LEAF_P);
  float y = leafY(p.xz);
  float dy = p.y - y;
  float sheet = abs(dy + 0.05) - 0.05;
  float side = (abs(b) - leafW(a))*0.8;
  float back = a - LEAF_TIP;
  float leaf = smax(max(sheet, back), side, 0.05)*0.72;
  res.x = leaf;
  const float G = 0.60;
  if(leaf >= G) res.x = min(res.x, leaf - 0.50);
  for(int k=0;k<1 && leaf<G;k++){
    for(int i=0;i<NBEAD;i++){
      vec3 B = BEAD[i];
      vec3 c = vec3(B.x, gBY[i] + B.z*0.56, B.y);
      float e = sdEllipsoid(p - c, vec3(B.z, B.z*0.78, B.z));
      if(e < res.x) res = vec2(e, M_BEAD);
    }
  }
  if(shadow) return res;
  res = flowerMap(p, res);
  res = backMap(p, res);
  res = nearMap(p, res);
  return res;
}

// ------------------------------------------------------------------ sky, its mist and its bokeh
// sun on wet leaves deep in the garden: round, soft, rim a little bright, as a real lens draws them
vec3 bokeh(vec2 ss){
  vec3 acc = vec3(0.0);
  for(int i=0;i<52;i++){
    vec3 h = hash31(float(i)*3.17 + 0.61), h2 = hash31(float(i)*1.93 + 5.3);
    vec2 o = vec2(mix(-0.37, 0.37, h.x), mix(-0.07, 0.17, h.y));
    float r = mix(0.0055, 0.0150, h.z*h.z);
    float d = length(ss - o);
    float disc = smoothstep(r, r - 0.0020, d)*(0.70 + 0.30*smoothstep(0.35*r, r, d));
    vec3 tint = mix(vec3(1.0,0.90,0.62), mix(vec3(0.74,0.96,0.62), vec3(1.0,0.66,0.72), step(0.72, h2.x)), step(0.42, h2.x));
    acc += tint*disc*mix(0.11, 0.78, h2.y*h2.y);
  }
  return acc;
}
vec3 setSky(vec3 rd){
  vec3 c = skyBase(rd);
  vec2 ss = rd.xy/max(-rd.z, 0.02);
  // the garden deep in the mist: round soft masses of foliage, two ranges, paler toward the sun
  vec2 sv = SUNV.xy/max(-SUNV.z, 0.02);
  float sunS = smoothstep(0.30, 0.02, length(ss - sv));
  float f1 = -1.0, f2 = -1.0;
  for(int i=0;i<9;i++){
    vec3 h = hash31(float(i)*2.37 + 1.1);
    f1 = max(f1, mix(0.070, 0.135, h.z) - length((ss - vec2(mix(-0.46, 0.46, (float(i) + h.x)/9.0), mix(-0.03, 0.05, h.y)))*vec2(0.72,1.0)));
    vec3 g = hash31(float(i)*4.11 + 7.3);
    f2 = max(f2, mix(0.045, 0.100, g.z) - length((ss - vec2(mix(-0.46, 0.46, (float(i) + g.x)/9.0), mix(-0.09, -0.02, g.y)))*vec2(0.72,1.0)));
  }
  float wob = 0.012*(fbm2_3(ss*16.0) - 0.5);
  c = mix(c, mix(vec3(0.150,0.260,0.175), vec3(0.50,0.52,0.34), sunS), 0.78*smoothstep(-0.010, 0.012, f1 + wob));
  c = mix(c, mix(vec3(0.050,0.125,0.075), vec3(0.34,0.38,0.22), sunS), 0.90*smoothstep(-0.008, 0.010, f2 + wob));
  return c + bokeh(ss)*smoothstep(-0.02, 0.05, rd.y + 0.10);
}

// ------------------------------------------------------------------ the leaf's relief and paint
float vorEdge(vec2 x){
  vec2 n = floor(x), f = fract(x);
  float f1 = 8.0, f2 = 8.0;
  for(int j=-1;j<=1;j++) for(int i=-1;i<=1;i++){
    vec2 g = vec2(float(i), float(j));
    vec2 r = g + hash22(n + g) - f;
    float d = dot(r, r);
    if(d < f1){ f2 = f1; f1 = d; } else if(d < f2) f2 = d;
  }
  return sqrt(f2) - sqrt(f1);
}
vec3  gVein;       // midrib, secondary veins, the fine net
float gPil;
float leafRelief(float a, float b){
  float ab = abs(b);
  float side = b > 0.0 ? 0.0 : 0.5;
  float s = (a - 0.82*ab + 0.020*ab*ab)/VEIN_DS + side;
  float f = abs(fract(s) - 0.5);                     // 0 mid-panel .. 0.5 on a vein
  float dv = (0.5 - f)*VEIN_DS*0.77;
  float mt = clamp(a/LEAF_TIP, 0.0, 1.0);
  float mw = mix(0.092, 0.018, mt);
  float mid = 1.0 - smoothstep(mw*0.55, mw, ab);
  float vw = mix(1.0, 0.45, clamp(ab/5.0, 0.0, 1.0));
  float sec = (1.0 - smoothstep(0.012*vw, 0.034*vw, dv))*smoothstep(mw*0.5, mw*1.4, ab);
  vec2 wq = vec2(a, b) + 0.10*vec2(vnoise2(vec2(a, b)*3.1), vnoise2(vec2(a, b)*3.1 + 5.0));   // never a honeycomb
  float e1 = vorEdge(wq*2.3);
  float e2 = vorEdge(wq*9.0 + 3.3);
  float net = 1.0 - smoothstep(0.0, 0.042, e1);
  float net2 = 1.0 - smoothstep(0.0, 0.11, e2);
  float pil = sqrt(clamp(dv/0.36, 0.0, 1.0));        // the panels between the veins bulge like a quilt
  float cell = smoothstep(0.0, 0.34, e1);
  float groove = smoothstep(mw*3.2, mw*0.7, ab);
  gVein = vec3(mid, sec, max(0.70*net, 0.42*net2));
  gPil = pil;
  return 0.034*pil*(1.0 - groove) + 0.0035*cell - 0.026*groove + 0.040*mid + 0.008*sec;
}

// the same ball lens as the drop's, for clear water
vec3 beadRad(vec3 p, vec3 rd, vec3 n){
  vec3 r1 = refract(rd, n, 1.0/1.33);
  vec3 nb = reflect(n, r1);
  vec3 r2 = refract(r1, -nb, 1.33);
  if(dot(r2, r2) < 0.5) r2 = rd;
  vec3 seen = gardenWorld(r2);
  float nv = clamp(dot(n, -rd), 0.0, 1.0), fr = 1.0 - nv;
  seen *= mix(0.72, 1.30, smoothstep(0.30, -0.85, n.y));
  seen *= 1.0 - 0.80*smoothstep(0.60, 0.92, fr);
  float cd = dot(n, normalize(vec3(0.40,-0.82,0.40)));
  seen += vec3(0.80,0.88,0.56)*smoothstep(0.55, 0.93, cd);
  vec3 R = reflect(rd, n);
  float F = 0.03 + 0.97*pow(fr, 5.0);
  vec3 col = mix(seen, gardenWorld(R), F);
  float w = dot(R, normalize(vec3(-0.56,0.80,0.22)));
  col += vec3(3.0,2.9,2.6)*smoothstep(0.962, 0.972, w);
  return col;
}

void paintLeaf(vec3 p, vec3 rd, float t, vec3 n, inout Surf s){
  vec2 d = p.xz - LEAF_O;
  float a = dot(d, LEAF_D), b = dot(d, LEAF_P);
  const float E = 0.011;
  float h0 = leafRelief(a, b);
  vec3 V = gVein; float pil = gPil;
  float ha = (leafRelief(a + E, b) - h0)/E, hb = (leafRelief(a, b + E) - h0)/E;
  vec3 nb = normalize(n - vec3(ha*LEAF_D.x + hb*LEAF_P.x, 0.0, ha*LEAF_D.y + hb*LEAF_P.y));
  float rr = length(p.xz);

  float big = fbm2_3(vec2(a, b)*0.42 + 1.3);
  float far_ = smoothstep(2.5, 10.0, a);
  float near_ = smoothstep(0.8, 2.6, length(p.xz - vec2(0.4,-0.5)));
  vec3 alb = mix(SP_LEAF_DEEP, SP_LEAF_MID, clamp(0.05 + 1.15*pil*(0.40 + 0.95*big), 0.0, 1.0));
  alb *= mix(1.25, 0.72, near_*smoothstep(-1.0, 3.0, p.z));         // the light pools around the drop; the near leaf falls into shade
  alb = mix(alb, SP_LEAF_LIT, 0.78*far_);
  alb *= 1.0 - 0.26*V.z*(1.0 - 0.7*far_);
  alb = mix(alb, SP_RIB*vec3(0.62,0.72,0.55), 0.72*V.y);
  alb = mix(alb, SP_RIB, V.x);

  float nv = clamp(dot(nb, -rd), 0.0, 1.0);
  float F = 0.040 + 0.960*pow(1.0 - nv, 5.0);
  vec3 R = reflect(rd, nb);
  R.y = abs(R.y) + 0.002; R = normalize(R);
  vec3 env = gardenWorld(R);
  // the drop stands in the mirror
  {
    vec3 oc = p - vec3(0.0,0.0,0.0);
    float bq = dot(oc, R), c = dot(oc, oc) - 0.62*0.62;
    float dsc = bq*bq - c;
    float hit = smoothstep(0.0, 0.10, dsc)*step(bq, 0.0);
    vec3 amber = mix(vec3(0.95,0.46,0.07), vec3(1.5,0.98,0.34), smoothstep(0.5, -0.1, R.y));
    env = mix(env, amber, hit);
  }
  vec3 amb = SP_AMB_SKY*(0.66 + 0.34*nb.y);
  vec3 col = alb*amb + env*F*0.62;

  // sun on the wet skin: tiny hard glints, thick along the veins' shoulders
  vec3 H = normalize(SUNV - rd);
  float gl = pow(max(dot(nb, H), 0.0), 900.0);
  col += vec3(2.6,2.4,1.9)*gl;

  // under the drop: contact, a shadow the colour of amber, the sun gathered on the leaf
  float contact = exp(-max(rr - 0.30, 0.0)/0.075);
  col *= 1.0 - 0.78*contact;
  alb *= 1.0 - 0.78*contact;
  vec2 v = p.xz - vec2(0.58,-0.34);
  float pool = exp(-dot(v*vec2(1.0,1.25), v*vec2(1.0,1.25))/0.34);
  float an = atan(v.y, v.x);
  float ray = vnoise2(vec2(an*3.2 + 2.0, length(v)*2.2));
  float rays = smoothstep(0.30, 0.90, ray);
  col += vec3(0.62,0.33,0.055)*pool*(0.50 + 0.40*rays);
  vec2 v2 = p.xz - vec2(0.72,-0.12);
  col += vec3(1.2,0.78,0.22)*exp(-dot(v2*vec2(0.62,2.0), v2*vec2(0.62,2.0))/0.012);

  // the dew's contact, and its small caustic
  for(int i=0;i<NBEAD;i++){
    vec3 B = BEAD[i];
    vec2 u = (p.xz - B.xy)/B.z;
    float ru = length(u);
    col *= 1.0 - 0.60*exp(-max(ru - 0.72, 0.0)*5.0);
    alb *= 1.0 - 0.60*exp(-max(ru - 0.72, 0.0)*5.0);
    col += vec3(1.5,1.5,0.8)*exp(-dot(u - vec2(0.75,0.75), u - vec2(0.75,0.75))*5.5);
  }

  // the fine dew: painted, far too small to need geometry
  vec2 cs = p.xz/0.26;
  vec2 ci = floor(cs), cf = fract(cs) - 0.5;
  vec3 hh = hash33(vec3(ci, 3.0));
  float rb = mix(0.06, 0.26, hh.z*hh.z)*step(0.56, hh.x)*smoothstep(0.55, 1.1, rr);
  vec2 cu = (cf - (hh.xy - 0.5)*0.34)/max(rb, 1e-3);
  float cdisc = smoothstep(1.0, 0.86, length(cu))*step(1e-3, rb);
  if(cdisc > 0.0){
    float zb = sqrt(max(1.0 - dot(cu, cu), 0.0));
    vec3 nd = normalize(vec3(cu.x, zb, cu.y));
    vec3 c2 = beadRad(p, rd, nd);
    col = mix(col, c2, cdisc);
  }

  // far along the leaf the wet skin glitters: the lens makes these into small discs
  vec2 gs = vec2(a, b)*2.3;
  vec3 gh = hash33(vec3(floor(gs), 7.0));
  float gd = length(fract(gs) - 0.25 - 0.5*gh.xy);
  col += vec3(1.6,1.45,1.0)*(3.0*gh.z*gh.z + 0.8)*smoothstep(0.070, 0.030, gd)*step(0.55, gh.z)*smoothstep(4.0, 8.0, a);

  // the margin: backlit, a thin bright edge
  float mg = smoothstep(0.32, 0.0, leafW(a) - abs(b));
  col += SP_LEAF_LIT*vec3(1.9,1.6,0.9)*mg;

  s.alb = max(alb, vec3(1e-4));
  s.n = nb; s.wrap = 0.30; s.spec = 0.0; s.sheen = 0.0; s.trans = 0.0;
  gRad = col/s.alb;
  gRadShade = 1.0;
}

vec3 stemRad(vec3 p, vec3 n){
  float l = dot(n, normalize(vec3(-0.84,0.20,0.50)));
  vec3 c = mix(vec3(0.07,0.16,0.07), vec3(0.40,0.54,0.20), smoothstep(-0.55, 0.85, l));
  c *= 0.94 + 0.10*sin(atan(n.x, n.z)*14.0 + 0.5*sin(p.y*0.6));
  c += vec3(0.30,0.36,0.16)*pow(1.0 - clamp(n.z, 0.0, 1.0), 3.0);
  return c;
}
vec3 petalRad(vec3 p, vec3 n, vec3 rd){
  float u = gPetalU;
  vec3 c = mix(SP_PETAL_B, SP_PETAL_A, smoothstep(0.16, 0.92, u));
  c *= 0.84 + 0.30*gPetalS;
  vec3 d = p - FC;
  float an = atan(dot(d, FY), dot(d, FX));
  c *= 0.92 + 0.10*sin(an*38.0 + 1.5*sin(an*5.0));
  float l = dot(n, normalize(vec3(-0.45,0.45,0.75)));
  c *= 0.60 + 0.46*smoothstep(-0.5, 0.9, l);
  // the sun comes through it from behind: the thin edges glow
  c += vec3(0.46,0.24,0.16)*smoothstep(0.55, 1.05, u);
  return c;
}
vec3 backLeafRad(vec3 p, vec3 n){
  float u = gBackQ.x, v = gBackQ.y;
  float l = smoothstep(-0.2, 0.9, dot(n, normalize(vec3(-0.30,0.70,0.55))));
  vec3 c = mix(vec3(0.030,0.105,0.052), vec3(0.15,0.31,0.095), l);
  float vn = abs(fract(u*5.0 - 1.6*abs(v)) - 0.5);
  c = mix(c, vec3(0.30,0.42,0.17), 0.36*smoothstep(0.42, 0.50, vn));
  c = mix(c, vec3(0.50,0.62,0.26), smoothstep(0.050, 0.020, abs(v)));
  c += vec3(0.24,0.28,0.09)*smoothstep(0.88, 1.0, abs(v))*(0.4 + 0.6*l);
  return mix(c, vec3(0.19,0.30,0.20), 0.30);
}

void setSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s){
  if(!(dot(n, n) > 0.5)) n = vec3(0.0,0.0,1.0);
  s.n = n;
  if(m == M_LEAF){
    paintLeaf(p, rd, t, n, s);
  } else {
    vec3 col;
    if(m == M_BEAD){
      col = beadRad(p, rd, n);
    } else if(m == M_STEM){
      col = stemRad(p, n);
    } else if(m == M_PETAL){
      col = petalRad(p, n, rd);
    } else if(m == M_HEART){
      float sp = vnoise2(p.xy*7.0);
      col = mix(vec3(0.60,0.27,0.03), vec3(1.30,0.84,0.16), smoothstep(0.34, 0.72, sp))*(0.72 + 0.34*clamp(n.z, 0.0, 1.0));
    } else if(m == M_BACK){
      col = backLeafRad(p, n);
    } else {
      float v = gBackQ.y, u = gBackQ.x;
      col = mix(vec3(0.010,0.045,0.020), vec3(0.040,0.120,0.032), smoothstep(0.0, 1.0, 1.0 - u));
      col = mix(col, vec3(0.24,0.34,0.11), smoothstep(0.10, 0.03, abs(v)));
      col += vec3(0.46,0.54,0.14)*smoothstep(0.80, 1.0, abs(v));
    }
    if(m == M_BACK || m == M_STEM || m == M_PETAL || m == M_HEART) col += 0.60*bokeh(rd.xy/max(-rd.z, 0.02));
    s.alb = vec3(1e-4);
    s.spec = 0.0; s.sheen = 0.0; s.trans = 0.0; s.wrap = 0.0;
    gRad = col/1e-4;
    gNoSun = 1.0;
    gRadShade = 1.0;
  }
}
#else
vec2 setMap(vec3 p, bool shadow){ return vec2(1.0, 0.0); }
void setSurface(vec3 p, vec3 rd, float t, float m, vec3 n, inout Surf s){}
#endif
