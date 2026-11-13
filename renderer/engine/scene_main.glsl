// ---------------------------------------------------------------------------
//  engine/scene_main.glsl — every pass that traces rays or reads the set.
//  engine/render.py string-concatenates:  header + param defines + JSON defines + this file,
//  and resolves the three #pragma include lines (common, chosen set, chosen animal).
//
//  Passes (selected by #define):
//    PASS_BG    once per animal: set only, high spp -> ambient layer (rgb, a=depth) + sun layer (rgb)
//    PASS_PRE   per frame, 1/8 res: coarse march -> start distance per screen tile
//    PASS_CHAR  per frame, per sample: only the tiles that hit -> character premultiplied + depth
//    PASS_LIVE  per frame: cached layers + moving dapples = the living background
//    PASS_VOL   per frame, low res: god rays
// ---------------------------------------------------------------------------
uniform vec2  uRes;
uniform vec2  uJitter;
uniform float uTime;
uniform int   uFrame;
uniform float uP[NPARAMS];
uniform vec3  uCamPos, uCamU, uCamV, uCamW;
uniform float uFl;
uniform vec3  uSunDir, uSunCol;
uniform vec3  uRimDir, uRimCol;
uniform float uDappleFloor;
uniform float uCharKey;      // exposure of the character relative to the set

#pragma include common
//props-blk{  (stripped from the background cache key: with no props file every block is the identity; places.glsl
// and the PLACE defines are keyed themselves).  The set is wrapped so the places props rest on are part of it.
#define setMap setMapS
#define setSurface setSurfaceS
//props-blk}
#pragma include set
//props-blk{
#undef setMap
#undef setSurface
#pragma include places
#ifdef PASS_BG
// (a set that has shaded this pixel already answers "nothing here" (>= 1e3) to the AO taps: the places respect that)
vec2 setMap(vec3 p, bool shadow){ vec2 r = setMapS(p, shadow); return (!shadow && r.x >= 999.0) ? r : opU(r, placesMap(p)); }
void setSurface(vec3 pw, vec3 rd, float t, float m, vec3 n, inout Surf s){
  if(m == M_PLACE){ placesSurface(pw, n, s); return; }
  setSurfaceS(pw, rd, t, m, n, s);
}
#else
#define setMap setMapS
#define setSurface setSurfaceS
#endif
// props: the animal's four entry points are renamed (…A) so the props join the character SDF here —
// engine/props_hold.glsl (what each paw holds, the counting columns: read by the animal's hand code) before the
// animal, engine/props.glsl (models, attach, painter) after it.  With PROPS_N 0 the wrappers are pass-throughs.
#pragma include props_hold
#ifndef CHAR_ROOT      // props x body: a character without the root wrappers: the props wrap the animal itself
#define charInit charInitA
#define charEval charEvalA
#define charShadow charShadowA
#define charSurface charSurfaceA
#endif
//props-blk}
//body-root{  (stripped from the background cache key: identity unless uRootOn, so the cached bakes stay valid)
// ---- body: whole-character root transform (docs/INTENT-RULES.md section 1).  The animal file is compiled
// in CHARACTER-LOCAL space (its entry points renamed *L below); the world-space entry points after the include map
// the world point into it.  world = uRootA*local + uRootO, uRootA = R*S (R = Ry(yaw)Rx(pitch)Rz(roll) about the
// centre of mass, S = the volume-preserving squash about the base point); engine/rt_engine.py computes them per
// frame.  uRootOn = 0 is the exact identity path (all root params 0 = today's picture bit for bit).
#define ROOT_XFORM 1
uniform float uRootOn;
uniform mat3  uRootA, uRootAi, uRootRm;
uniform vec3  uRootO;
uniform float uRootLip;            // min squash scale: local distance * uRootLip is a bound in world space
uniform float uRootTNear, uRootTFar;
uniform vec3  uCamH0;              // the camera in HEAD space at rest (gaze_free's head-forward reference)
mat3 gRootR = mat3(1.0);           // char-local -> world rotation (for the props agent; set once per fragment)
vec3 gCamPosL, gSunDirL, gRimDirL;
vec3 rootFwd(vec3 pl){ return uRootOn > 0.5 ? uRootA*pl + uRootO : pl; }
vec3 rootInv(vec3 pw){ return uRootOn > 0.5 ? uRootAi*(pw - uRootO) : pw; }
vec3 rootDirInv(vec3 dw){ return uRootOn > 0.5 ? normalize(uRootAi*dw) : dw; }
void rootInit(){
  gRootR = uRootOn > 0.5 ? uRootRm : mat3(1.0);
  gCamPosL = rootInv(uCamPos);
  gSunDirL = uRootOn > 0.5 ? transpose(uRootRm)*uSunDir : uSunDir;
  gRimDirL = uRootOn > 0.5 ? transpose(uRootRm)*uRimDir : uRimDir;
}
vec3 rootGaze(vec3 camH);          // gaze_free: below
#define CHAR_T_NEAR_RT (uRootOn > 0.5 ? uRootTNear : CHAR_T_NEAR)
#define CHAR_T_FAR_RT (uRootOn > 0.5 ? uRootTFar : CHAR_T_FAR)
// inside the animal: local entry points, the camera and the lights in local space, set lookups back in world.
// Only a character that opts in (char_defines CHAR_ROOT: the muse) is wrapped: the other characters compile exactly
// today's code (the wrappers, though the identity at uRootOn = 0, change the compiled float math by an LSB here and
// there), and the engine keeps their uRootOn at 0.
#ifdef CHAR_ROOT
#define charInit charInitL
#define charEval charEvalL
#define charShadow charShadowL
#define charSurface charSurfaceL
#define charWaterRefl charWaterReflL
#define charWaterDist charWaterDistL
#define uCamPos gCamPosL
#define uSunDir gSunDirL
#define uRimDir gRimDirL
#define setSky(d) setSky(gRootR*(d))
#define setCookie(p) setCookie(rootFwd(p))
#define setAmbient(p, n) setAmbient(rootFwd(p), gRootR*(n))
#endif
//body-root}
#pragma include animal
//body-root{
#ifdef CHAR_ROOT
#undef charInit
#undef charEval
#undef charShadow
#undef charSurface
#undef charWaterRefl
#undef charWaterDist
#undef uCamPos
#undef uSunDir
#undef uRimDir
#undef setSky
#undef setCookie
#undef setAmbient
//body-root}
//props-blk{
// props: the root transform's world-space entry points (above) are the props' inner character (…A); the
// props join the character SDF in world space after them (engine/props.glsl).  With PROPS_N 0 these are pass-throughs.
#define charInit charInitA
#define charEval charEvalA
#define charShadow charShadowA
#define charSurface charSurfaceA
//props-blk}
//body-root{
void charInit(){ rootInit(); charInitL(); }
void charEval(vec3 p){
  if(uRootOn < 0.5){ charEvalL(p); return; }
  charEvalL(uRootAi*(p - uRootO));
  gcBase *= uRootLip; gcBare *= uRootLip; gcLen *= uRootLip;
}
float charShadow(vec3 p){ return uRootOn < 0.5 ? charShadowL(p) : charShadowL(uRootAi*(p - uRootO))*uRootLip; }
void charSurface(vec3 pw, vec3 rd, float t, inout Surf s){
  if(uRootOn < 0.5){ charSurfaceL(pw, rd, t, s); return; }
  s.n = normalize(transpose(uRootA)*s.n);              // normals: n_local ~ A^T n_world
  charSurfaceL(uRootAi*(pw - uRootO), normalize(uRootAi*rd), t, s);
  s.n = normalize(transpose(uRootAi)*s.n);
  s.T = normalize(uRootA*s.T);
}
#ifdef SP_CHAR_WATER
float charWaterRefl(vec3 X, vec3 m){ return charWaterReflL(rootInv(X), uRootOn > 0.5 ? normalize(uRootAi*m) : m); }
vec2 charWaterDist(vec3 X){ vec2 r = charWaterDistL(rootInv(X)); if(uRootOn > 0.5) r.x *= uRootLip; return r; }
#endif
#endif   // CHAR_ROOT
// gaze_free (0..1): the eyes' target, in head space.  0 = the camera (today: eye contact); 1 = straight out of the
// head (where the camera would be with the head at rest), so a character turned away does not look back at the
// camera through its own head.  While the root moves the camera direction is also kept inside the eyes' reach.
vec3 rootGaze(vec3 camH){
  float g = clamp(uP[P_GAZE_FREE], 0.0, 1.0);
  if(uRootOn < 0.5 && g <= 0.0) return camH;
  float L = length(camH);
  vec3 a = camH/max(L, 1e-4), b = normalize(uCamH0);
  float c = dot(a, b);
  const float CMAX = 0.40;                             // ~66 degrees off head-forward at most
  if(c < CMAX){
    vec3 o = a - b*c;
    o = dot(o, o) > 1e-8 ? normalize(o) : normalize(cross(b, vec3(1.0, 0.0, 0.0)));
    a = b*CMAX + o*sqrt(1.0 - CMAX*CMAX); c = CMAX;
  }
  float th = acos(clamp(c, -1.0, 1.0));
  if(g > 0.0 && th > 1e-4) a = (sin((1.0 - g)*th)*a + sin(g*th)*b)/sin(th);
  return a*L;
}
//body-root}
//props-blk{
#undef charInit
#undef charEval
#undef charShadow
#undef charSurface
#pragma include props
bool gPropWin = false;       // the character pass: this pixel shows a prop (its normal / painter / AO taps)
void charInit(){ propsPre(); charInitA(); propsInit(); }      // props
void charEval(vec3 p){                                        // props: a prop is a bare (unfurred) solid
#if PROPS_N > 0
  // (no bound stand-in for the character here: a box distance under the prepass's tile margin marks tiles round
  // the whole box — measured: it cost more than it saved)
#ifdef PASS_CHAR
  if(gPropWin && PROPS_CHAR_BOUND(p) > 0.45){ gcBase = 1e3; gcBare = 1e3; gcLen = 0.0; gcRest = p; }
  else
#endif
  charEvalA(p);
#ifndef PASS_CHAR
  // (the character pass marches the props on their own and adds them to its AO taps itself: no prop SDF inlined
  // at every charEval call site there — the shader stays small, its occupancy the character's own)
  if(gPrOn) gcBare = min(gcBare, propsDist(p));
#endif
#else
  charEvalA(p);
#endif
}
float charShadow(vec3 p){                                     // props: props cast on the character, not in the baked set
#if PROPS_N > 0 && !defined(PASS_BG) && !defined(PASS_LIVE)
  return min(charShadowA(p), propsShadow(p));
#else
  return charShadowA(p);
#endif
}
void charSurface(vec3 p, vec3 rd, float t, inout Surf s){     // props: the nearer of prop and character paints
#if PROPS_N > 0 && defined(PASS_CHAR)
  if(gPropWin){ propsPaint(gPrHitId, gPrHitQ, gPrHitPart, p, s); return; }    // perf: the march's hit
#endif
#if PROPS_N > 0 && !defined(PASS_CHAR)
  float dp = gPrOn ? propsDist(p) : 1e3;
  if(dp < 0.004){
    if(PROPS_CHAR_BOUND(p) > 0.06){ propsSurface(p, rd, s); return; }      // nowhere near the character
    charEvalA(p);
    if(dp <= min(gcBase, gcBare) + 0.0015){ propsSurface(p, rd, s); return; }
  }
#endif
  charSurfaceA(p, rd, t, s);
}
//props-blk}

vec3 camRay(vec2 pix){
  vec2 uv = (2.0*pix - uRes)/uRes.y;
  return normalize(uv.x*uCamU + uv.y*uCamV + uFl*uCamW);
}
float HG(float mu, float g){ float g2=g*g; return (1.0-g2)/pow(1.0+g2-2.0*g*mu, 1.5); }

// the three distances the engine marches, derived from the animal's charEval()
float charCoarse(vec3 p){ charEval(p); return min(gcBare, gcBase - gcLen); }
float charFine(vec3 p){ charEval(p); return min(gcBare, gcBase - gcLen*charTuft()); }
float charAO(vec3 p){ charEval(p); return min(gcBare, gcBase); }

// ===========================================================================
#ifdef PASS_BG
layout(location=0) out vec4 fragA;
layout(location=1) out vec4 fragS;
const float MAXD = 240.0;

float bgShadow(vec3 ro, vec3 rd){
  float res = 1.0, t = 0.03;
  for(int i=0;i<56;i++){
    vec3 p = ro + rd*t;
//body-root{  BG_NO_CHAR: the set without the character's shadow (the live floor shadow draws it under root motion)
#ifdef BG_NO_CHAR
    float h = setMap(p, true).x;
#else
//body-root}
    float h = min(setMap(p, true).x, charShadow(p));
//body-root{
#endif
//body-root}
    res = min(res, 10.0*h/t);
    t += clamp(h, 0.02, 1.6);
    if(res < 0.002 || t > 60.0) break;
  }
  res = clamp(res, 0.0, 1.0);
  return res*res*(3.0 - 2.0*res);
}
float bgAO(vec3 p, vec3 n, float sc){
  float occ = 0.0, w = 1.0;
  for(int i=0;i<4;i++){
    float h = sc*(0.05 + 0.3*float(i));
    float d = setMap(p + n*h, false).x;
    occ += (h - d)*w/sc;
    w *= 0.75;
  }
  return clamp(1.0 - 0.9*occ, 0.0, 1.0);
}
void main(){
  setInit(); charInit();
  vec2 pix = gl_FragCoord.xy - 0.5 + uJitter;
  vec3 ro = uCamPos, rd = camRay(pix);
  float t = 0.05; float m = -1.0;
  for(int i=0;i<220;i++){
    vec3 p = ro + rd*t;
    vec2 h = setMap(p, false);
    m = h.y;
    if(abs(h.x) < 0.0006*t) break;
    t += h.x;
    if(t > MAXD) break;
  }
  if(t > MAXD) m = -1.0;
  vec3 amb, sun = vec3(0.0);
  if(m < 0.0){
    amb = setSky(rd); t = MAXD;
  } else {
    vec3 p = ro + rd*t;
    vec2 e = vec2(1,-1)*0.0006*max(t,1.0);
    vec3 n = normalize(e.xyy*setMap(p+e.xyy,false).x + e.yyx*setMap(p+e.yyx,false).x +
                       e.yxy*setMap(p+e.yxy,false).x + e.xxx*setMap(p+e.xxx,false).x);
    Surf s = surfDefault(n);
    setSurface(p, rd, t, m, n, s);
    n = s.n;
    float canopy = (m == M_CANOPY) ? 1.0 : 0.0;     // PASS_BG only: a set may omit M_CANOPY by defining it unused
    vec3 L = uSunDir;
    float ndl = dot(n, L);
    float dif = clamp((ndl + s.wrap)/(1.0 + s.wrap), 0.0, 1.0);
    float sh = (canopy > 0.5) ? smoothstep(12.0, 30.0, p.y)*0.55 + 0.10 : bgShadow(p + n*0.02, L);
    float ao = bgAO(p, n, canopy > 0.5 ? 30.0 : 5.0)*s.occ;
    amb = s.alb*setAmbient(p, n)*ao;
    sun = s.alb*uSunCol*dif*sh;
    float back = pow(clamp(dot(rd, L), 0.0, 1.0), 3.0);
    sun += s.transCol*uSunCol*s.trans*back*sh*(0.4 + 0.6*clamp(-ndl,0.,1.));
    float fres = pow(1.0 - clamp(dot(n, -rd), 0.0, 1.0), 4.0);
    sun += s.sheen*fres*uSunCol*0.25*sh*s.alb*2.0*ao;
    amb += s.sheen*fres*vec3(0.9,0.85,0.7)*0.5*s.alb*ao;
    vec3 hv = normalize(L - rd);
    float sp = pow(clamp(dot(n, hv), 0.0, 1.0), s.shin)*(s.shin + 8.0)/(8.0*PI);
    float F = 0.04 + 0.96*pow(1.0 - clamp(dot(hv, -rd), 0.0, 1.0), 5.0);
    sun += uSunCol*sp*s.spec*F*4.0*dif*sh;
    if(canopy > 0.5){ amb += sun; sun = vec3(0.0); }   // the canopy does not take the cookie
  }
  // aerial haze (the sun-lit shafts themselves are live, PASS_VOL)
  float jitter = hash12(gl_FragCoord.xy*1.37 + uJitter*17.0);
  float tEnd = min(t, 180.0);
  const int NV = 24;
  float dtv = tEnd/float(NV);
  vec3 ins = vec3(0.0); float tr = 1.0;
  vec3 fa = setFogAmb(rd);
  for(int i=0;i<NV;i++){
    vec3 pv = ro + rd*(float(i) + jitter)*dtv;
    float dens = setFog(pv);
    ins += tr*dens*dtv*fa;
    tr *= exp(-dens*dtv);
  }
  if(m < 0.0) tr = mix(tr, 1.0, 0.35);
  fragA = vec4(amb*tr + ins, t);
  fragS = vec4(sun*tr, 1.0);
}
#endif

// ===========================================================================
#ifdef PASS_PRE
out vec4 fragColor;
uniform float uTileEps;     // world-space margin that keeps fur and silhouettes inside a tile
uniform float uTile;
void main(){
  charInit();
  vec3 ro = uCamPos, rd = camRay(gl_FragCoord.xy*uTile);
//props-blk{
  gPrMask = propsRayMask(ro, rd, 1e3, 0.02 + 4.0*uTileEps);
//props-blk}
  float t = CHAR_T_NEAR_RT, hit = 0.0, best = t;
  for(int i=0;i<40;i++){
    float h = charCoarse(ro + rd*t);
    if(h < uTileEps){ hit = 1.0; break; }
    t += h;
    if(t > CHAR_T_FAR_RT) break;
  }
  fragColor = vec4(hit > 0.5 ? t : 1e6, hit, 0.0, 1.0);
}
#endif

// ===========================================================================
#ifdef PASS_CHAR
layout(location=0) out vec4 fragC;
layout(location=1) out vec4 fragD;
flat in float vTStart;

float softShadowC(vec3 ro, vec3 rd){
  float res = 1.0, t = 0.03;
//props-blk{
  int mask0 = gPrMask; gPrMask = propsRayMask(ro, rd, 3.3, 0.02);
  if(gPrMask == 0 && !propsCharBoxHit(ro, rd, 3.3)){ gPrMask = mask0; return 1.0; }   // nothing on the way to the sun
  if(gPropWin){            // a prop pixel: a shorter, coarser march (props are hard-edged; the TAA smooths it)
    for(int i=0;i<10;i++){
      float h = charShadow(ro + rd*t);
      res = min(res, 5.0*h/t);
      t += clamp(h, 0.03, 0.35);
      if(res < 0.01 || t > 3.2) break;
    }
    gPrMask = mask0;
    res = clamp(res, 0.0, 1.0);
    return res*res*(3.0 - 2.0*res);
  }
//props-blk}
  for(int i=0;i<CHAR_SHADOW_STEPS;i++){
    float h = charShadow(ro + rd*t);
    res = min(res, 5.0*h/t);
    t += clamp(h, 0.02, 0.17);
    if(res < 0.01 || t > 3.2) break;
  }
//props-blk{
  gPrMask = mask0;
//props-blk}
  res = clamp(res, 0.0, 1.0);
  return res*res*(3.0 - 2.0*res);
}
float aoC(vec3 p, vec3 n){
  float occ = 0.0, w = 1.0;
//props-blk{
  int mask0 = gPrMask; gPrMask = 0xFFFF;
//props-blk}
  for(int i=0;i<CHAR_AO_STEPS;i++){
    float h = 0.03 + 0.11*float(i);
    float d = charAO(p + n*h);
//props-blk{
    d = min(d, propsDist(p + n*h));
//props-blk}
    occ += max(h - d, 0.0)*w;
    w *= 0.62;
  }
//props-blk{
  gPrMask = mask0;
//props-blk}
  return clamp(1.0 - 3.0*occ, 0.0, 1.0);
}
// nF = fur-level normal (strand ridges), nS = smooth skin normal
void normalsC(vec3 p, float t, out vec3 nF, out vec3 nS){
  nF = vec3(0.0); nS = vec3(0.0);
  float e = 0.0045*t/5.0;
  for(int i=0;i<4;i++){
    vec3 k = 0.5773*(2.0*vec3(float(((i+3)>>1)&1), float((i>>1)&1), float(i&1)) - 1.0);
    charEval(p + k*e);
    nS += k*min(gcBase, gcBare);
    nF += k*min(gcBase - gcLen*charTuft(), gcBare);
  }
  nF = normalize(nF); nS = normalize(nS);
}

#if PROPS_N > 0
// perf: the sun shadow march and the AO taps of a character-pass pixel in ONE loop, so the props' models are
// inlined once for both (every prop-SDF call site is a full copy of every model; the shader's size and registers cost
// every pixel of the pass).  The same numbers as softShadowC(ps, L) and aoC(p, n).
void shadowAoC(vec3 ps, vec3 L, vec3 p, vec3 n, out float sh, out float ao){
  int mask0 = gPrMask;
  int mS = propsRayMask(ps, L, 3.3, 0.02);
  float hMax = 0.03 + 0.11*float(int(CHAR_AO_STEPS) - 1);
  int mA = propsNearMask(p, 2.0*hMax + 0.05);      // the AO taps' slots (exactly the all-slot result)
  bool doS = mS != 0 || propsCharBoxHit(ps, L, 3.3);
  int nS = !doS ? 0 : (gPropWin ? 10 : int(CHAR_SHADOW_STEPS));   // a prop pixel: shorter, coarser (TAA smooths it)
  float sLo = gPropWin ? 0.03 : 0.02, sHi = gPropWin ? 0.35 : 0.17;
  float res = 1.0, t = 0.03, occ = 0.0, w = 1.0;
  int iS = 0, j = nS > 0 ? -1 : 0;           // j -1: the shadow march; 0.. the AO taps
  for(int it=0; it<max(10, int(CHAR_SHADOW_STEPS)) + int(CHAR_AO_STEPS); it++){
    if(j >= int(CHAR_AO_STEPS)) break;
    bool S = j < 0;
    float hA = 0.03 + 0.11*float(max(j, 0));
    vec3 x = S ? ps + L*t : p + n*hA;
    gPrMask = S ? mS : mA;
    float dP = propsDistB(x, S ? 0.10 : 0.03);
    if(S){
      float h = min(charShadowA(x), dP);
      res = min(res, 5.0*h/t);
      t += clamp(h, sLo, sHi);
      iS++;
      if(res < 0.01 || t > 3.2 || iS >= nS) j = 0;
    } else {
      float d = min(charAO(x), dP);
      occ += max(hA - d, 0.0)*w;
      w *= 0.62;
      j++;
    }
  }
  gPrMask = mask0;
  res = clamp(res, 0.0, 1.0);
  sh = !doS ? 1.0 : res*res*(3.0 - 2.0*res);
  ao = clamp(1.0 - 3.0*occ, 0.0, 1.0);
}
#endif
//look-v2{  (stripped from the background cache key: PASS_CHAR only, nothing in a bake reads it)
// look (pass 2): soft WARM occlusion from the character's cheap shadow proxy (under the chin, between arm and
// body, the lap) that leaves the broad lit surfaces alone, a clear back rim light, a fuzz rim on pile, a soft sky
// gloss on hard glossy things.  The exposure of the character is kept.  LOOK_V2 undefined = the picture before the look upgrade.
#ifdef LOOK_V2
uniform float uLookWide;     // 1 = the wide occlusion taps (the adaptive controller drops them on its lowest tier)
uniform vec4  uLookK;        // x key gain, y rim gain, z wide-occlusion strength, w velvet (fuzz rim on pile)
uniform vec4  uLookF;        // x fill gain, y gloss on hard surfaces, z rim tightness (exponent), w unused
uniform vec3  uLookRimCol;   // the rim light's colour
float aoWideC(vec3 p, vec3 n){
  if(uLookWide < 0.5) return 1.0;
  // toward the open sky and the key: what sits above this point shades it (chin on chest, arm on belly)
  vec3 d = normalize(n + vec3(0.0, 0.45, 0.0));
  // relative to the proxy's own value here: where the proxy and the true surface disagree (tufts, ears, cheeks,
  // a ruff) an absolute test would paint blotches; and only a proxy that is clearly nearer along the tap counts
  float o = 0.0, d0 = min(charShadowA(p), 0.06);
  float h = 0.10; o += 0.42*smoothstep(0.25, 1.0, (h - (charShadowA(p + d*h) - d0))/h);
  h = 0.24;       o += 0.34*smoothstep(0.25, 1.0, (h - (charShadowA(p + d*h) - d0))/h);
  h = 0.48;       o += 0.24*smoothstep(0.25, 1.0, (h - (charShadowA(p + d*h) - d0))/h);
  return clamp(1.0 - uLookK.z*o, 0.0, 1.0);
}
#endif
//look-v2}
vec3 lightChar(vec3 ro, vec3 rd, vec3 p, Surf s, vec3 ng){
  vec3 L = uSunDir;
  vec3 n = s.n;
  float ndl = dot(n, L);
  float ndlG = dot(ng, L);
  float dif = clamp((ndl + s.wrap)/(1.0 + s.wrap), 0.0, 1.0);
  float ck = setCookie(p);
  ck = mix(uDappleFloor, 1.0, ck);
#if PROPS_N > 0
  float sh, ao;
  shadowAoC(p + ng*0.03, L, p, ng, sh, ao);
  ao *= s.occ;
#else
  float sh = softShadowC(p + ng*0.03, L);
  float ao = aoC(p, ng)*s.occ;
#endif
  vec3 sun = uSunCol*dif*sh*ck;
  vec3 amb = setAmbient(p, n)*mix(vec3(0.50,0.26,0.16)*ao, vec3(1.0), ao);
//look-v2{
#ifdef LOOK_V2
  float aoW = aoWideC(p, ng);
  {
    // occlusion is a WARM shade (light bounced off the coat itself), never a grey veil; where the key really lands
    // it mostly gives way (a lit cheek stays bright), in the shade it deepens: chin, armpits, lap
    float lit = dif*sh;
    float aoA = mix(aoW, 1.0, 0.50*lit);
    vec3 warm = mix(vec3(0.60,0.36,0.25), vec3(0.80,0.64,0.54), clamp(s.alb, 0.0, 1.0));     // (a white coat's shade is pale, not brown)
    amb *= uLookF.x*mix(warm, vec3(1.0), aoA);
    sun *= uLookK.x*mix(0.50, 1.0, aoW);
    ao *= mix(0.80, 1.0, aoW);
  }
#endif
//look-v2}
  vec3 col = s.alb*(sun + amb);
  float nv = clamp(dot(n, -rd), 0.0, 1.0);
  // warm subsurface band at the terminator
  float band = smoothstep(-0.55, -0.05, ndlG)*smoothstep(0.55, 0.05, ndlG);
  col += s.fur*s.alb*sqrt(s.alb)*uSunCol*vec3(1.0,0.55,0.32)*0.16*band*ck*mix(0.5,1.0,sh);
  // translucency: thin parts glow with whichever light is behind them
  float behindS = clamp(dot(-ng, L)*0.5 + 0.5, 0.0, 1.0);
  float behindR = clamp(dot(-ng, uRimDir)*0.5 + 0.5, 0.0, 1.0);
  col += s.alb*s.transCol*s.trans*(uSunCol*pow(behindS, 2.0)*ck*0.55 + uRimCol*pow(behindR, 2.0)*0.8);
  // rim from the bright clearing behind
  float fres = pow(1.0 - clamp(dot(ng, -rd), 0.0, 1.0), 3.0);
  // a rim belongs to OUTER silhouettes: if something sits behind this edge, along the view ray, no rim
  float behind = min(charAO(p + rd*0.30), charAO(p + rd*0.65));
  fres *= smoothstep(0.0, 0.07, behind);
  float rimL = clamp(dot(n, uRimDir)*0.6 + 0.5, 0.0, 1.0);
  col += s.sheen*fres*rimL*uRimCol*(0.35 + 0.65*s.alb)*(0.4 + 0.6*ao);
//look-v2{
#ifdef LOOK_V2
  {
    // rim light: the lamp behind the character draws its outer edge (gated to OUTER silhouettes by `behind`), wide
    // enough to read at a glance; mirrors (eyes, a wet nose) take reflections instead
    float rimB = smoothstep(-0.35, 0.45, dot(ng, uRimDir));
    float edge = pow(1.0 - clamp(dot(ng, -rd), 0.0, 1.0), uLookF.z)*smoothstep(0.0, 0.07, behind);
    float mat = max(s.sheen, 0.30)*smoothstep(700.0, 200.0, s.shin)*(gPropWin ? 0.20 : 1.0);     // (a prop's flat top is all 'edge')
    // on pile the light comes back through the hairs, so it carries the coat's hue (a brown coat glows copper, an
    // orange one amber, a white one stays white) — never a frosty white outline on a dark coat
    vec3 hue = s.alb/max(max(s.alb.r, max(s.alb.g, s.alb.b)), 0.04);
    vec3 rimT = mix(vec3(1.0), hue, 0.65*s.fur);
    vec3 rimC = uLookK.y*edge*rimB*uLookRimCol*rimT*(0.40 + 0.60*s.alb)*(0.45 + 0.55*ao)*mat;
    col += rimC/(1.0 + 0.55*rimC);                 // soft shoulder: a glow on the edge, never a hard white outline
    // fuzz: short pile scatters light forward at grazing angles (asperity), lit by the key and the fill
    col += uLookK.w*s.fur*s.alb*pow(1.0 - nv, 1.8)*(0.60*amb + 0.40*sun)*ao;
    // gloss: hard smooth things (glaze, plastic, lacquer) carry a soft reflection of the room at grazing angles
    float gl = uLookF.y*(1.0 - s.fur)*smoothstep(0.25, 0.90, s.spec)*smoothstep(700.0, 200.0, s.shin);
    if(gl > 0.0) col += gl*pow(1.0 - nv, 3.0)*setAmbient(p, reflect(rd, n))*ao;
  }
#endif
//look-v2}
  col += s.sheen*fres*(sun*0.6)*s.alb;
  vec3 hv = normalize(L - rd);
  if(s.fur > 0.0){
    // Kajiya-Kay, two lobes: a pale tight one and an albedo-tinted broad one
    vec3 T1 = normalize(s.T + n*0.18), T2 = normalize(s.T - n*0.22);
    float d1 = dot(T1, hv), d2 = dot(T2, hv);
    float k1 = pow(max(1.0 - d1*d1, 0.0), 42.0), k2 = pow(max(1.0 - d2*d2, 0.0), 12.0);
    float gate = smoothstep(-0.2, 0.3, ndl);
    col += min(s.fur*uSunCol*gate*sh*ck*ao*(0.055*k1*vec3(1.0,0.95,0.86) + 0.11*k2*s.alb), vec3(0.16));   // no fireflies
  }
  float sp = pow(clamp(dot(n, hv), 0.0, 1.0), s.shin)*(s.shin + 8.0)/(8.0*PI);
  float F = 0.04 + 0.96*pow(1.0 - clamp(dot(hv, -rd), 0.0, 1.0), 5.0);
  col += (1.0 - s.fur)*uSunCol*sp*s.spec*F*4.0*sh*ck*smoothstep(-0.1,0.2,ndl);
  if(s.mirror > 0.0){
    vec3 r = reflect(rd, n);
    vec3 env = setSky(r)*smoothstep(-0.25, 0.25, r.y) + SP_BOUNCE*0.4*smoothstep(0.25,-0.25,r.y);
    float Fr = 0.04 + 0.96*pow(1.0 - nv, 5.0);
    col += s.mirror*env*Fr*ao;
  }
  return col*uCharKey + s.emit;
}

//props-blk{
void normalsCP(vec3 p, float t, out vec3 nF, out vec3 nS){
#if PROPS_N > 0
  if(gPropWin){ nS = gPrHitN; nF = nS; return; }      // perf: taken round the hit by propsMarch
#else
  if(gPropWin){ nS = propsNormal(p); nF = nS; return; }
#endif
  normalsC(p, t, nF, nS);
}
#define normalsC normalsCP
//props-blk}
void main(){
  setInit(); charInit();
  vec2 pix = gl_FragCoord.xy - 0.5 + uJitter;
  vec3 ro = uCamPos, rd = camRay(pix);
  float t = vTStart;
  // stage 1: the smooth envelope of the fur tips (no noise, big steps)
//props-blk{
  // the props are plain solids: marched on their own along this ray, then the fur march sees the character alone;
  // the nearer wins (a fur fringe in front of a prop blends over it)
  gPrMask = propsRayMask(ro, rd, 1e3, 0.03);
  float tP = propsMarch(ro, rd, vTStart);
  gPrOn = false;
  float tB0, tB1;      // a prop in front of the character's box (or a ray that never enters it): no fur march at all
  if(!propsCharBox(ro, rd, tB0, tB1) || tP < tB0) t = 1e3;      // (finite: past the far end, no overflow)
//props-blk}
  for(int i=0;i<CHAR_COARSE_STEPS;i++){
    float h = charCoarse(ro + rd*t);
    if(h < 0.0015) break;
    t += h;
    if(t > CHAR_T_FAR_RT) break;
  }
  // stage 2: through the fur as a volume, strand by strand (shell march).  Rays that leave the fur
  // carry on to whatever is behind: silhouettes come out soft, and interior edges stay opaque.
  float a = 0.0, tHit = -1.0, hHit = 1.0;
  bool escaped = false;
  for(int i=0;i<CHAR_FINE_STEPS;i++){
    charEval(ro + rd*t);
    float solid = min(gcBase, gcBare);
    if(solid < 0.0009){
      if(tHit < 0.0){ tHit = t; hHit = 0.0; }
      a = 1.0;
      break;
    }
    float outer = gcBase - gcLen;
    if(outer > 0.0){
      t += max(min(outer, gcBare), 0.0045);
    } else {
      float hN = gcBase/max(gcLen, 1e-4);
      // where fur tapers to nothing (lips, nose, eyelids) it is a closed nap: without this only the noise
      // lattice peaks clear the test and the taper shows as a grid of dots
      float cov = smoothstep(hN - 0.10, hN + 0.10, charTuft() + 1.0 - smoothstep(0.0, 0.022, gcLen));
      if(cov > 0.2 && tHit < 0.0){ tHit = t; hHit = hN; }
      a += (1.0 - a)*cov*0.85;
      if(a > 0.97){ a = 1.0; break; }
      t += min(max(gcLen*0.20, 0.0028), max(gcBare, 0.0009));
    }
    if(t > CHAR_T_FAR_RT){ escaped = true; break; }
  }
  if(!escaped && a < 1.0){
    // out of steps: is there anything behind?  If so never leak the background; if not, keep the soft edge.
    float tt = t; bool esc = false;
    for(int i=0;i<CHAR_ESCAPE_STEPS;i++){
      float h = charCoarse(ro + rd*tt);
      if(h < 0.001) break;
      tt += max(h, 0.012);
      if(tt > CHAR_T_FAR_RT){ esc = true; break; }
    }
    if(!esc){ a = 1.0; if(tHit < 0.0) tHit = tt; }
  }
//props-blk{
  // the prop wins in front of the character or under a thin fur fringe; a denser fringe is opaque over it (one
  // lighting call site: the shader stays the character's size)
  bool propWin = tP < 1e8 && (tHit < 0.0 || a < 0.5 || tHit > tP);
  if(tP < 1e8) a = 1.0;
  if(propWin){ tHit = tP; hHit = 1.0; }
  gPropWin = propWin;
//props-blk}
  if(tHit < 0.0 || a < 0.004){ fragC = vec4(0.0); fragD = vec4(0.0); return; }
  vec3 p = ro + rd*tHit;
  vec3 nF, n;
  normalsC(p, tHit, nF, n);
  Surf s = surfDefault(normalize(mix(n, nF, CHAR_FUR_RELIEF)));
  charSurface(p, rd, tHit, s);
  s.occ *= mix(0.80, 1.0, smoothstep(0.0, 0.75, hHit + (1.0 - s.fur)));   // deep in the coat it is darker
  vec3 col = lightChar(ro, rd, p, s, n);
  fragC = vec4(col*a, a);
  fragD = vec4(tHit*a, 0.0, 0.0, a);
}
#endif

// ===========================================================================
#ifdef PASS_LIVE
out vec4 fragColor;
uniform sampler2D uBgA, uBgS;
uniform float uBgSunFloor, uBgSunGain;
//body-root{
// The character's shadow on the set, live while the root moves (the bake's own is at the rest pose; the layers bound
// here are then the ones baked without it, engine/rt_passes.glsl PASS_BGVIEW).  Only rays that pass the character's
// bounding sphere march its shadow proxy.
uniform float uRootLive;
uniform vec4  uRootBound;
float rootShadowLive(vec3 p){
  vec3 oc = p - uRootBound.xyz;
  float b = dot(oc, uSunDir), c = dot(oc, oc) - uRootBound.w*uRootBound.w;
  float disc = b*b - c;
  if(disc <= 0.0) return 1.0;
  float sq = sqrt(disc), t1 = -b + sq;
  if(t1 <= 0.03) return 1.0;
  charInit();
  float res = 1.0, t = max(-b - sq, 0.03);
  for(int i=0;i<40;i++){
    float h = charShadow(p + uSunDir*t);
    res = min(res, 10.0*h/t);
    t += clamp(h, 0.02, 0.5);
    if(res < 0.002 || t > t1) break;
  }
  res = clamp(res, 0.0, 1.0);
  return res*res*(3.0 - 2.0*res);
}
// ... and its contact occlusion on the floor (ambient), the bake's formula (sets/studio.glsl) live
float rootContactLive(vec3 p){
#ifdef SP_GROUND_Y
  if(abs(p.y - SP_GROUND_Y) > 0.06 || length(p - uRootBound.xyz) > uRootBound.w + 0.45) return 1.0;
  charInit();
  float c0 = charShadow(p + vec3(0.0, 0.05, 0.0)), c1 = charShadow(p + vec3(0.0, 0.16, 0.0)), c2 = charShadow(p + vec3(0.0, 0.38, 0.0));
  float co = max(0.05 - c0, 0.0)*2.2 + max(0.16 - c1, 0.0)*1.3 + max(0.38 - c2, 0.0)*0.55;
  return clamp(1.0 - 2.2*co, 0.22, 1.0);
#else
  return 1.0;
#endif
}
//body-root}
//look-v2{
#if defined(LOOK_V2) && defined(SP_GROUND_Y)
// look: a soft pool of shade on the floor under the character (wide, from its shadow proxy; it follows the
// feet, a jump lets go of it).  The bake's own tight contact stays; this is the part that reads at a glance.
uniform float uLookFloor;
float lookFloor(vec3 p){
  if(uLookFloor <= 0.0 || abs(p.y - SP_GROUND_Y) > 0.06) return 0.0;
  vec3 oc = p - uRootBound.xyz;
  float rr = uRootBound.w + 0.35;
  if(dot(oc.xz, oc.xz) > rr*rr) return 0.0;
  charInit();
  float c1 = charShadow(p + vec3(0.0, 0.06, 0.0)), c2 = charShadow(p + vec3(0.0, 0.22, 0.0)), c3 = charShadow(p + vec3(0.0, 0.50, 0.0));
  float o = 0.55*smoothstep(0.30, 0.0, c1) + 0.35*smoothstep(0.55, 0.05, c2) + 0.30*smoothstep(0.90, 0.10, c3);
  return clamp(uLookFloor*o, 0.0, 0.85);
}
#endif
//look-v2}
//scene-surf{  (stripped from the background cache key: the live pass only; the board / TV geometry itself is keyed
// through places.glsl and the PLACE defines).  The app server's two live surfaces (docs/SCENE.md section 7):
// the chalk on the board's slate and the TV's picture, IN the set: lit by the light the bake put on the slate (the
// baked slate is divided back out), behind the character (it composites over this pass), under its live shadow.
#if defined(PLACE_BOARD) || defined(PLACE_TV)
#define SURF_ON 1
uniform sampler2D uSurfB, uSurfT;     // premultiplied RGBA8, row 0 = the top of the canvas
uniform vec2  uSurfOn;                // x board, y tv: 1 = a texture is bound
uniform float uTvGain;
// the aerial haze the bake added in front of a surface point (same integral as PASS_BG: setFog incl. the window
// shafts of the studio)
vec3 surfIns(vec3 ro, vec3 rd, float t){
  float tEnd = min(t, 180.0), dtv = tEnd/24.0;
  vec3 ins = vec3(0.0), fa = setFogAmb(rd); float tr = 1.0;
  for(int i=0;i<24;i++){
    vec3 pv = ro + rd*(float(i) + 0.5)*dtv;
    float dens = setFog(pv);
#ifdef SET_STUDIO
    vec3 r = toRoom(pv);
    dens += 0.020*beamMask(r)*smoothstep(-0.6, -1.8, r.z)*step(XL, r.x)*step(r.x, XR)*step(ZB, r.z)*step(r.y, HC)*SP_FOG_GAIN;
#endif
    ins += tr*dens*dtv*fa; tr *= exp(-dens*dtv);
  }
  return ins;
}
// ray -> (local x, local y, t) on a vertical plane through C with right axis U, normal N
vec3 surfHit(vec3 ro, vec3 rd, vec3 C, vec3 U, vec3 N){
  float dn = dot(rd, N);
  float t = dn < -1e-4 ? dot(C - ro, N)/dn : -1.0;
  vec3 P = ro + rd*t - C;
  return vec3(dot(P, U), P.y, t);
}
#endif
//scene-surf}
void main(){
  setInit();
  vec2 uv = gl_FragCoord.xy/uRes;
  // tiny slow foliage breathing, stronger high in the frame
  float T = uTime;
  vec2 wob = vec2(vnoise2(uv*5.0 + vec2(T*0.20, 0.0)) - 0.5, vnoise2(uv*5.0 + vec2(7.0, T*0.17)) - 0.5);
//scene-surf{
#ifdef SURF_ON
  vec3 sRo = uCamPos, sRd = camRay(gl_FragCoord.xy);
#ifdef PLACE_BOARD
  vec3 hb = surfHit(sRo, sRd, BOARD_C, BOARD_U, BOARD_N);
  vec2 buv = vec2(hb.x/BOARD_SZ.x + 0.5, 0.5 - hb.y/BOARD_SZ.y);
  vec2 bdx = dFdx(buv), bdy = dFdy(buv);
  bool onB = uSurfOn.x > 0.5 && hb.z > 0.0 && all(lessThan(abs(buv - 0.5), vec2(0.5)));
#else
  bool onB = false;
#endif
#ifdef PLACE_TV
  vec3 ht = surfHit(sRo, sRd, TV_C, TV_U, TV_N);
  vec2 tuv = vec2(ht.x/TV_SZ.x + 0.5, 0.5 - ht.y/TV_SZ.y);
  vec2 tdx = dFdx(tuv), tdy = dFdy(tuv);
  bool onT = uSurfOn.y > 0.5 && ht.z > 0.0 && all(lessThan(abs(tuv - 0.5), vec2(0.5)));
#else
  bool onT = false;
#endif
  if(onB || onT) wob = vec2(0.0);       // the surfaces sit still under their (unwobbled) live layer
#endif
//scene-surf}
  uv += wob*(1.6/uRes.y)*(0.3 + 1.4*smoothstep(0.35, 1.0, uv.y));
  vec4 A = texture(uBgA, uv);
  vec3 S = texture(uBgS, uv).rgb;
//body-root{
  if(uRootLive > 0.0 && A.a < 200.0){
    vec3 pw = uCamPos + camRay(gl_FragCoord.xy)*A.a;
    S *= mix(1.0, rootShadowLive(pw + vec3(0.0, 0.02, 0.0)), uRootLive);
    A.rgb *= mix(1.0, rootContactLive(pw), uRootLive);
  }
//body-root}
//look-v2{
#if defined(LOOK_V2) && defined(SP_GROUND_Y)
  if(A.a < 200.0){
    float lf = lookFloor(uCamPos + camRay(gl_FragCoord.xy)*A.a);
    A.rgb *= mix(vec3(1.0), vec3(0.42,0.30,0.27), lf);
    S *= 1.0 - 0.80*lf;
  }
#endif
//look-v2}
//props-blk{
#if defined(PROPS_LIVE) && PROPS_N > 0
  // props: their shadow and contact shade on the set, live (they move; the bake never sees them)
  if(A.a < 200.0){
    propsFromPose();
    vec2 so = propsOnSet(uCamPos + camRay(gl_FragCoord.xy)*A.a + vec3(0.0, 0.004, 0.0));
    S *= so.x*so.y; A.rgb *= so.y;
  }
#endif
//props-blk}
  // Far background is out of focus and its depth is blurred, so no world-space cookie here (a blurred
  // depth sweeps a world pattern into ripples).  Soft light patches drift in screen space instead,
  // squashed near the horizon so they lie down on the ground.
  float asp = uRes.x/uRes.y;
  vec2 g = vec2(uv.x*asp*2.2, uv.y*4.0/(0.8 + 1.0*smoothstep(0.62, 0.0, uv.y)) + 0.5*uv.x);
  float n1 = fbm2_3(g + gWind*5.0 + vec2(T*0.03, 0.0));
  float n2 = vnoise2(g*2.6 - gWind*7.0);
  float ck = uBgSunFloor + uBgSunGain*smoothstep(0.38, 0.70, n1*0.8 + n2*0.2);
//scene-surf{
#ifdef SURF_ON
  vec3 col = A.rgb + S*ck;
#ifdef PLACE_BOARD
  if(onB && A.a > hb.z - 0.06){          // the slate is the nearest thing in the set here (not the tray lip, a post ...)
    vec4 pm = textureGrad(uSurfB, buv, bdx, bdy);
    if(pm.a > 0.002){
      vec2 px = buv*vec2(1024.0, 512.0);
      float fw = length(fwidth(px));
      // chalk grain: a fine tooth (fades once a pixel covers several texels) and a coarse powdery unevenness
      float fine = vnoise2(px*0.85 + 3.0), coarse = vnoise2(px*0.21 + 17.0);
      float a = pm.a*(0.80 + 0.20*coarse)*(1.0 - 0.42*(1.0 - fine)*smoothstep(3.0, 1.0, fw));
      vec3 chalk = pow(clamp(pm.rgb/pm.a, 0.0, 1.0), vec3(2.2))*0.82;          // its albedo (sRGB colour -> linear)
      vec3 ins = surfIns(sRo, sRd, hb.z);
      vec3 slate = chalkSlateAlb(vec2(hb.x, hb.y));
      vec3 lit = (col - ins)/max(slate, vec3(0.01))*chalk + ins;          // the slate's light on the chalk
      col = mix(col, lit, clamp(a, 0.0, 1.0));
    }
  }
#endif
#ifdef PLACE_TV
  if(onT && A.a > ht.z - 0.08){
    vec4 pm = textureGrad(uSurfT, tuv, tdx, tdy);
    if(pm.a > 0.002){
      vec3 emit = pow(clamp(pm.rgb/pm.a, 0.0, 1.0), vec3(2.2))*uTvGain;
      col = mix(col, emit + 0.05*col, pm.a);          // the picture, a faint reflection of the room on the glass
    }
  }
#endif
  fragColor = vec4(col, A.a);
  return;
#endif
//scene-surf}
  fragColor = vec4(A.rgb + S*ck, A.a);
}
#endif

// ===========================================================================
#ifdef PASS_VOL
out vec4 fragColor;
uniform sampler2D uDepth;
uniform float uVolGain;
void main(){
  setInit();
  vec2 uv = gl_FragCoord.xy/uRes;
  vec3 ro = uCamPos, rd = camRay(uv*uRes);
  float dA = texture(uDepth, uv).a;
  // Fixed world-space samples, faded at the surface.  Scaling the samples by depth would make thin
  // shafts alias differently on each side of every depth edge: hard outlines around the trunks.
  const int NV = 24;
  const float DT = 90.0/float(NV);
  float mu = dot(rd, uSunDir);
  float ph = 0.45*HG(mu, 0.62) + 0.55;
  float acc = 0.0;
  for(int i=0;i<NV;i++){
    float tv = (float(i) + 0.5)*DT;
    vec3 pv = ro + rd*tv;
    float vis = clamp((dA - tv)/DT + 0.5, 0.0, 1.0);
    acc += vis*setVol(pv)*setFog(pv)*(1.0 + 3.5*smoothstep(0.0, 6.0, pv.y - SP_GROUND_Y - 2.0));
  }
  float dtv = DT;
  float dust = 0.85 + 0.3*vnoise2(uv*3.0 + vec2(uTime*0.05, -uTime*0.03));
  fragColor = vec4(uSunCol*ph*acc*dtv*uVolGain*dust, 1.0);
}
#endif
